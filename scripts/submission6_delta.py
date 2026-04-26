"""
Submission 6: Pairwise delta learning + concentration-aware HTS pre-training.

New in this submission vs Sub 5:
  - DeltaChempropModel: trained on pairwise (Δ pEC50) differences; at inference
    anchors to k=10 nearest training neighbors. Directly optimizes for relative
    activity within scaffold families — the analog-set test structure.
  - Concentration-aware HTS pre-training: all 4 dose-response points kept;
    log10[concentration_M] passed as x_d to the Chemprop FFN. 4× more training
    signal vs Hill-fitting, no R²≥0.5 acceptance filter.
  - GNN encoder transfer: only message_passing.* weights are transferred from
    HTS pretraining → fine-tuning, so FFN architecture mismatches are safe.
  - chemprop_scratch zeroed out by ElasticNet (coef=0.0); signal came from
    chemprop_hts (0.38), RF (0.29), delta (0.20), kNN (0.06), LGBM (0.01).

Model roster:
  delta          — pairwise Δ pEC50 Chemprop, kNN-anchored inference (k=10)
  chemprop_scratch — direct pEC50 (random init, 2 seeds CV / 3 seeds final)
  chemprop_hts     — direct pEC50 (conc-aware HTS pretrained, 2/3 seeds)
  knn              — Tanimoto k=5
  lgbm             — ECFP4 + ECFP6 + RDKit
  rf               — ECFP4 + ECFP6 + RDKit

Requires Python 3.11 venv (.venv311):
    .venv311/bin/python scripts/submission6_delta.py
"""

import gc
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

# Patch out datasets 4.8+ usage-tracking request that can hang on restrictive networks
try:
    import datasets.load as _dl
    _dl.increase_load_count = lambda *a, **kw: None
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("submission6")

CV_SEEDS = [42, 7]
FINAL_SEEDS = [42, 7, 13]
RESULTS_PATH = "logs/submission6_cv_results.json"

# Set to "mps" on Apple Silicon, "cuda" if GPU available, else "cpu"
DEVICE = "mps"

# Delta model training budget.
# CV uses fewer pairs/epochs since it's diagnostic; final training uses more.
DELTA_CV_PAIRS = 20_000
DELTA_CV_EPOCHS = 40
DELTA_FINAL_PAIRS = 40_000
DELTA_FINAL_EPOCHS = 80


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def build_sample_weights(train) -> np.ndarray:
    w = (
        train["sample_weight"].values.copy()
        if "sample_weight" in train.columns
        else np.ones(len(train))
    )
    if "is_nonspecific" in train.columns:
        w[train["is_nonspecific"].values] *= 0.3
    if "cliff_sample_weight" in train.columns:
        w *= train["cliff_sample_weight"].values
    w = w / w.mean()
    return w.astype(np.float32)


def train_chemprop(sm_tr, y_tr, sw_tr, seed, epochs, lr, init_sd=None, x_d=None):
    from src.models.chemprop_model import ChempropModel
    m = ChempropModel(
        epochs=epochs, hidden_size=300, depth=3, ffn_num_layers=3,
        dropout=0.1, batch_size=64, lr=lr, device=DEVICE,
        snapshot_epochs=5, extra_features=False, seed=seed,
    )
    m.fit(sm_tr, y_tr, sample_weight=sw_tr, init_state_dict=init_sd, x_d=x_d)
    return m


def avg_chemprop(sm_tr, y_tr, sw_tr, sm_va, seeds, epochs, lr, init_sd=None, x_d=None):
    return np.mean(
        [train_chemprop(sm_tr, y_tr, sw_tr, s, epochs, lr, init_sd, x_d).predict(sm_va)
         for s in seeds], axis=0
    )


def avg_chemprop_full(sm_all, y_all, sw_all, sm_test, seeds, epochs, lr, init_sd=None, x_d=None):
    return np.mean(
        [train_chemprop(sm_all, y_all, sw_all, s, epochs, lr, init_sd, x_d).predict(sm_test)
         for s in seeds], axis=0
    )


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    # ---------------------------------------------------------- #
    # 1. Load data
    # ---------------------------------------------------------- #
    from src.data.load_data import load_all_tiers, COL_SMILES, COL_PECSO
    from src.data.cliff_analysis import identify_activity_cliffs, annotate_cliff_compounds

    logger.info("Loading dataset...")
    ds = load_all_tiers(cache_dir="data/hf_cache")
    train = ds.train
    test = ds.test
    hts = ds.hts

    logger.info("Identifying activity cliffs...")
    cliff_pairs = identify_activity_cliffs(
        smiles=train[COL_SMILES].tolist(),
        activities=train[COL_PECSO].to_numpy(),
        sim_threshold=0.7,
        activity_threshold=1.0,
    )
    train = annotate_cliff_compounds(train, cliff_pairs)

    train_smiles = train[COL_SMILES].tolist()
    train_y = train[COL_PECSO].to_numpy()
    test_smiles = test[COL_SMILES].tolist()
    train_mean = float(train_y.mean())
    sample_weights = build_sample_weights(train)
    logger.info(f"Train: {len(train)} | Test: {len(test)} | HTS: {len(hts)}")

    # ---------------------------------------------------------- #
    # 2. Concentration-aware HTS pre-training
    #    All 4 concentration points kept; log10[conc] as x_d feature.
    #    Gives 4× more training signal vs Hill-fitting.
    # ---------------------------------------------------------- #
    from src.models.hts_pretrain import prepare_hts_concentration_data
    from src.models.chemprop_model import ChempropModel

    logger.info("\n=== Concentration-aware HTS pre-training ===")
    pt_smiles, pt_y, pt_xd = prepare_hts_concentration_data(hts_df=hts, primary_train=train)
    logger.info(f"Pre-training on {len(pt_smiles)} rows ({len(set(pt_smiles))} unique compounds)")

    pretrain_model = ChempropModel(
        epochs=60, hidden_size=300, depth=3, ffn_num_layers=3,
        dropout=0.1, batch_size=64, lr=1e-3, device=DEVICE,
        snapshot_epochs=5, extra_features=False, seed=42,
    )
    pretrain_model.fit(pt_smiles, pt_y, x_d=pt_xd)
    hts_state_dict = pretrain_model.get_state_dict()
    del pretrain_model, pt_smiles, pt_y, pt_xd
    gc.collect()
    logger.info("HTS pre-training complete.")

    # ---------------------------------------------------------- #
    # 3. Feature matrices
    # ---------------------------------------------------------- #
    from src.features.feature_engineering import ecfp4, FeaturePipeline

    logger.info("\n=== Building feature matrices ===")
    fps_train = ecfp4(train_smiles)
    fps_test = ecfp4(test_smiles)

    pipeline = FeaturePipeline(include_mordred=False, include_ecfp6=True, include_fcfp4=False)
    X_train = pipeline.fit_transform(train_smiles)
    X_test = pipeline.transform(test_smiles)
    logger.info(f"Feature matrix: train={X_train.shape}, test={X_test.shape}")

    # ---------------------------------------------------------- #
    # 4. Butina 5-fold CV — OOF predictions for all 6 models
    # ---------------------------------------------------------- #
    from src.evaluation.validate import ButinaKFold, rae
    from src.models.local_models import TanimotoKNN
    from src.models.gbm_models import LGBMWrapper, RFWrapper
    from src.models.delta_model import DeltaChempropModel

    logger.info("\n=== Butina 5-fold CV ===")
    splitter = ButinaKFold(n_splits=5, tanimoto_threshold=0.4)
    folds = list(splitter.split(train_smiles))

    model_names = ["delta", "chemprop_scratch", "chemprop_hts", "knn", "lgbm", "rf"]
    oof = {m: np.zeros(len(train)) for m in model_names}
    fold_raes = {m: [] for m in model_names}

    for fold, (tr_idx, va_idx) in enumerate(folds):
        logger.info(f"\n--- Fold {fold} (train={len(tr_idx)}, val={len(va_idx)}) ---")
        sm_tr = [train_smiles[i] for i in tr_idx]
        sm_va = [train_smiles[i] for i in va_idx]
        y_tr, y_va = train_y[tr_idx], train_y[va_idx]
        sw_tr = sample_weights[tr_idx]
        fps_tr, fps_va = fps_train[tr_idx], fps_train[va_idx]
        X_tr, X_va = X_train[tr_idx], X_train[va_idx]
        fold_mean = float(y_tr.mean())

        # Delta model (warm-started GNN encoder from HTS pretraining)
        logger.info(f"Fold {fold}: Delta model...")
        delta_cv = DeltaChempropModel(
            epochs=DELTA_CV_EPOCHS, hidden_size=300, depth=3, ffn_num_layers=3,
            dropout=0.1, batch_size=64, lr=1e-3, device=DEVICE,
            snapshot_epochs=5, seed=42, n_pairs_per_epoch=DELTA_CV_PAIRS,
            cliff_oversample=3, k_neighbors=10,
        )
        delta_cv.fit(sm_tr, y_tr, fps_train=fps_tr, init_state_dict=hts_state_dict)
        oof["delta"][va_idx] = delta_cv.predict(sm_va, sm_tr, y_tr, fps_tr, fps_va)
        fold_raes["delta"].append(rae(y_va, oof["delta"][va_idx], y_train_mean=fold_mean))
        logger.info(f"Fold {fold}: delta RAE={fold_raes['delta'][-1]:.4f}")
        del delta_cv; gc.collect()

        # Chemprop scratch
        logger.info(f"Fold {fold}: Chemprop scratch...")
        oof["chemprop_scratch"][va_idx] = avg_chemprop(
            sm_tr, y_tr, sw_tr, sm_va, CV_SEEDS, epochs=60, lr=1e-3
        )
        fold_raes["chemprop_scratch"].append(
            rae(y_va, oof["chemprop_scratch"][va_idx], y_train_mean=fold_mean)
        )
        logger.info(f"Fold {fold}: chemprop_scratch RAE={fold_raes['chemprop_scratch'][-1]:.4f}")

        # Chemprop HTS-pretrained (fine-tune without x_d; only GNN encoder transferred)
        logger.info(f"Fold {fold}: Chemprop HTS-pretrained...")
        oof["chemprop_hts"][va_idx] = avg_chemprop(
            sm_tr, y_tr, sw_tr, sm_va, CV_SEEDS, epochs=60, lr=5e-4,
            init_sd=hts_state_dict,
        )
        fold_raes["chemprop_hts"].append(
            rae(y_va, oof["chemprop_hts"][va_idx], y_train_mean=fold_mean)
        )
        logger.info(f"Fold {fold}: chemprop_hts RAE={fold_raes['chemprop_hts'][-1]:.4f}")

        # kNN
        knn = TanimotoKNN(k=5)
        knn.fit(fps_tr, y_tr)
        oof["knn"][va_idx] = knn.predict(fps_va)
        fold_raes["knn"].append(rae(y_va, oof["knn"][va_idx], y_train_mean=fold_mean))

        # LGBM (n_jobs=1 avoids OMP thread exhaustion after multiple Chemprop runs)
        n_es = max(50, int(0.1 * len(tr_idx)))
        lgbm = LGBMWrapper({"n_jobs": 1})
        lgbm.fit(X_tr[:-n_es], y_tr[:-n_es],
                 X_val=X_tr[-n_es:], y_val=y_tr[-n_es:],
                 sample_weight=sw_tr[:-n_es])
        oof["lgbm"][va_idx] = lgbm.predict(X_va)
        fold_raes["lgbm"].append(rae(y_va, oof["lgbm"][va_idx], y_train_mean=fold_mean))

        # RF
        rf = RFWrapper()
        rf.fit(X_tr, y_tr, sample_weight=sw_tr)
        oof["rf"][va_idx] = rf.predict(X_va)
        fold_raes["rf"].append(rae(y_va, oof["rf"][va_idx], y_train_mean=fold_mean))

        logger.info(
            f"Fold {fold}: kNN={fold_raes['knn'][-1]:.4f}, "
            f"LGBM={fold_raes['lgbm'][-1]:.4f}, RF={fold_raes['rf'][-1]:.4f}"
        )

    # Checkpoint OOF to disk — crash recovery without re-running all folds
    oof_checkpoint = {m: oof[m].tolist() for m in model_names}
    oof_checkpoint["train_y"] = train_y.tolist()
    oof_checkpoint["fold_raes"] = {m: fold_raes[m] for m in model_names}
    Path("logs").mkdir(exist_ok=True)
    with open("logs/submission6_oof_checkpoint.json", "w") as f:
        json.dump(oof_checkpoint, f)
    logger.info("OOF checkpoint saved.")

    # ---------------------------------------------------------- #
    # 5. CV summary
    # ---------------------------------------------------------- #
    logger.info("\n=== CV Summary ===")
    cv_summary = {}
    for m in model_names:
        mean_r = float(np.mean(fold_raes[m]))
        std_r = float(np.std(fold_raes[m]))
        cv_summary[m] = {"mean": mean_r, "std": std_r}
        logger.info(f"  {m:<20}: {mean_r:.4f} ± {std_r:.4f}")

    # ---------------------------------------------------------- #
    # 6. ElasticNet stacking on OOF predictions
    # ---------------------------------------------------------- #
    from src.ensemble.stack_and_submit import ElasticNetStacker

    logger.info("\n=== ElasticNet stacking ===")
    oof_matrix = np.column_stack([oof[m] for m in model_names])
    stacker = ElasticNetStacker(l1_ratio=0.7, cv=5)
    stacker.fit(oof_matrix, train_y, model_names=model_names)

    oof_stacked = stacker.predict(oof_matrix)
    stacked_rae = float(rae(train_y, oof_stacked, y_train_mean=train_mean))
    logger.info(f"ElasticNet OOF RAE: {stacked_rae:.4f}")

    inv_rae_w = np.array([1.0 / cv_summary[m]["mean"] for m in model_names])
    inv_rae_w /= inv_rae_w.sum()
    inv_rae_val = float(rae(train_y, (oof_matrix * inv_rae_w).sum(axis=1), y_train_mean=train_mean))
    logger.info(f"Inv-RAE ensemble OOF RAE (comparison): {inv_rae_val:.4f}")

    # ---------------------------------------------------------- #
    # 7. Full training on all data
    # ---------------------------------------------------------- #
    logger.info("\n=== Full training (all data) ===")
    test_preds = {}

    logger.info("Full training: Delta model (3 seeds)...")
    delta_preds_all = []
    for seed in FINAL_SEEDS:
        dm = DeltaChempropModel(
            epochs=DELTA_FINAL_EPOCHS, hidden_size=300, depth=3, ffn_num_layers=3,
            dropout=0.1, batch_size=64, lr=1e-3, device=DEVICE,
            snapshot_epochs=5, seed=seed, n_pairs_per_epoch=DELTA_FINAL_PAIRS,
            cliff_oversample=3, k_neighbors=10,
        )
        dm.fit(train_smiles, train_y, fps_train=fps_train, init_state_dict=hts_state_dict)
        delta_preds_all.append(
            dm.predict(test_smiles, train_smiles, train_y, fps_train, fps_test)
        )
        del dm; gc.collect()
    test_preds["delta"] = np.mean(delta_preds_all, axis=0)

    logger.info("Full training: Chemprop scratch (3 seeds)...")
    test_preds["chemprop_scratch"] = avg_chemprop_full(
        train_smiles, train_y, sample_weights, test_smiles,
        FINAL_SEEDS, epochs=80, lr=1e-3,
    )

    logger.info("Full training: Chemprop HTS (3 seeds)...")
    test_preds["chemprop_hts"] = avg_chemprop_full(
        train_smiles, train_y, sample_weights, test_smiles,
        FINAL_SEEDS, epochs=80, lr=5e-4, init_sd=hts_state_dict,
    )
    del hts_state_dict; gc.collect()

    logger.info("Full training: kNN, LGBM, RF...")
    knn_full = TanimotoKNN(k=5)
    knn_full.fit(fps_train, train_y)
    test_preds["knn"] = knn_full.predict(fps_test)

    n_es = max(50, int(0.1 * len(train_smiles)))
    lgbm_full = LGBMWrapper({"n_jobs": 1})
    lgbm_full.fit(X_train[:-n_es], train_y[:-n_es],
                  X_val=X_train[-n_es:], y_val=train_y[-n_es:],
                  sample_weight=sample_weights[:-n_es])
    test_preds["lgbm"] = lgbm_full.predict(X_test)

    rf_full = RFWrapper()
    rf_full.fit(X_train, train_y, sample_weight=sample_weights)
    test_preds["rf"] = rf_full.predict(X_test)

    for m in model_names:
        p = test_preds[m]
        logger.info(f"  {m:<20}: mean={p.mean():.3f}, std={p.std():.3f}")

    # ---------------------------------------------------------- #
    # 8. Apply stacker and save submission
    # ---------------------------------------------------------- #
    test_matrix = np.column_stack([test_preds[m] for m in model_names])
    final_preds = stacker.predict(test_matrix)
    logger.info(f"Ensemble: mean={final_preds.mean():.3f}, std={final_preds.std():.3f}")

    from src.ensemble.stack_and_submit import make_submission, validate_submission

    submission_path = make_submission(
        predictions=final_preds,
        test_df=test,
        description="Sub6: delta learning + conc-aware HTS pretrain + ElasticNet",
        val_rae=stacked_rae,
    )
    validate_submission(str(submission_path), test)
    logger.info(f"Submission saved: {submission_path}")

    # ---------------------------------------------------------- #
    # 9. Save CV results
    # ---------------------------------------------------------- #
    results = {
        "fold_raes": {m: [float(r) for r in fold_raes[m]] for m in model_names},
        "cv_summary": cv_summary,
        "ensemble_rae_elasticnet": stacked_rae,
        "ensemble_rae_inv_rae": inv_rae_val,
        "elasticnet_coefs": stacker.coefs,
        "submission_file": str(submission_path),
        "prediction_stats": {
            m: {"mean": float(test_preds[m].mean()), "std": float(test_preds[m].std())}
            for m in model_names
        },
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"CV results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
