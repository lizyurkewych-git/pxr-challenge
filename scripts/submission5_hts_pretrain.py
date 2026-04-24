"""
Submission 5: HTS pre-training + ElasticNet stacking + Butina CV.

New in this submission vs Sub 4:
  - HTS pre-training: Chemprop D-MPNN pre-trained on ~5.5k PXR HTS pseudo-pEC50
    (Hill-fitted from 4-point dose-response) then fine-tuned on primary DRC.
  - 2 Chemprop variants: scratch (Sub 4 baseline) + HTS-pretrained (new)
  - 2 seeds per variant in CV, 3 seeds for final training → seed ensembling
  - ElasticNetCV stacker on OOF predictions (replaces inv-RAE weighting)
  - Butina cluster CV (Tanimoto threshold=0.4) → more honest than Murcko scaffold
  - Dropped: foundation model embeddings (proven not to help in Sub 4)

Model roster:
  chemprop_scratch   (2 seeds CV / 3 seeds final)
  chemprop_hts       (2 seeds CV / 3 seeds final, HTS pretrained)
  knn                (Tanimoto k=5)
  lgbm               (ECFP4 + ECFP6 + RDKit, MAE objective)
  rf                 (ECFP4 + ECFP6 + RDKit)

Requires Python 3.11 venv (.venv311):
    .venv311/bin/python scripts/submission5_hts_pretrain.py
"""

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
    import datasets.load as _datasets_load
    _datasets_load.increase_load_count = lambda *a, **kw: None
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("submission5")

CV_SEEDS = [42, 7]
FINAL_SEEDS = [42, 7, 13]
RESULTS_PATH = "logs/submission5_cv_results.json"

# Set to "mps" on Apple Silicon, "cuda" if GPU available, else "cpu"
DEVICE = "cpu"


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


def train_chemprop_seed(sm_tr, y_tr, sw_tr, seed: int, epochs: int, lr: float = 1e-3,
                        init_state_dict=None):
    """Train one Chemprop model (one seed, optionally from pre-trained weights)."""
    from src.models.chemprop_model import ChempropModel
    m = ChempropModel(
        epochs=epochs, hidden_size=300, depth=3, ffn_num_layers=3,
        dropout=0.1, batch_size=64, lr=lr, device=DEVICE,
        snapshot_epochs=5, extra_features=False, seed=seed,
    )
    m.fit(sm_tr, y_tr, sample_weight=sw_tr, init_state_dict=init_state_dict)
    return m


def avg_seed_preds(sm_tr, y_tr, sw_tr, sm_va, seeds, epochs, lr, init_sd=None):
    """Train multiple seeds and return averaged OOF predictions."""
    return np.mean(
        [train_chemprop_seed(sm_tr, y_tr, sw_tr, seed=s, epochs=epochs, lr=lr,
                             init_state_dict=init_sd).predict(sm_va)
         for s in seeds], axis=0
    )


def avg_seed_preds_full(sm_all, y_all, sw_all, sm_test, seeds, epochs, lr, init_sd=None):
    """Train multiple seeds on full data and return averaged test predictions."""
    return np.mean(
        [train_chemprop_seed(sm_all, y_all, sw_all, seed=s, epochs=epochs, lr=lr,
                             init_state_dict=init_sd).predict(sm_test)
         for s in seeds], axis=0
    )


def main():
    # ------------------------------------------------------------------ #
    # 1. Load data
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    # 2. HTS pre-training
    #    Hill-fit 4-concentration dose-response → pseudo-pEC50 → pre-train Chemprop
    # ------------------------------------------------------------------ #
    from src.models.hts_pretrain import prepare_hts_pretrain_data
    from src.models.chemprop_model import ChempropModel

    logger.info("\n=== HTS pre-training ===")
    pretrain_df = prepare_hts_pretrain_data(hts_df=hts, primary_train=train)
    pretrain_smiles = pretrain_df[COL_SMILES].tolist()
    pretrain_y = pretrain_df["pseudo_pec50"].to_numpy()
    logger.info(f"Pre-training on {len(pretrain_df)} compounds")

    pretrain_model = ChempropModel(
        epochs=60, hidden_size=300, depth=3, ffn_num_layers=3,
        dropout=0.1, batch_size=64, lr=1e-3, device=DEVICE,
        snapshot_epochs=5, extra_features=False, seed=42,
    )
    pretrain_model.fit(pretrain_smiles, pretrain_y)
    hts_state_dict = pretrain_model.get_state_dict()
    del pretrain_model
    import gc; gc.collect()
    logger.info("HTS pre-training complete.")

    # ------------------------------------------------------------------ #
    # 3. Build feature matrices (ECFP4 + ECFP6 + RDKit)
    # ------------------------------------------------------------------ #
    from src.features.feature_engineering import ecfp4, FeaturePipeline

    logger.info("\n=== Building feature matrices ===")
    fps_train = ecfp4(train_smiles)
    fps_test = ecfp4(test_smiles)

    pipeline = FeaturePipeline(include_mordred=False, include_ecfp6=True, include_fcfp4=False)
    X_train = pipeline.fit_transform(train_smiles)
    X_test = pipeline.transform(test_smiles)
    logger.info(f"Feature matrix: train={X_train.shape}, test={X_test.shape}")

    # ------------------------------------------------------------------ #
    # 4. Butina cluster CV — OOF predictions for all models
    # ------------------------------------------------------------------ #
    from src.evaluation.validate import ButinaKFold, rae
    from src.models.local_models import TanimotoKNN
    from src.models.gbm_models import LGBMWrapper, RFWrapper

    logger.info("\n=== Butina 5-fold CV ===")
    splitter = ButinaKFold(n_splits=5, tanimoto_threshold=0.4)
    folds = list(splitter.split(train_smiles))

    model_names = ["chemprop_scratch", "chemprop_hts", "knn", "lgbm", "rf"]
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

        # Chemprop scratch (2 seeds averaged)
        oof["chemprop_scratch"][va_idx] = avg_seed_preds(
            sm_tr, y_tr, sw_tr, sm_va, seeds=CV_SEEDS, epochs=60, lr=1e-3
        )
        fold_raes["chemprop_scratch"].append(
            rae(y_va, oof["chemprop_scratch"][va_idx], y_train_mean=fold_mean)
        )
        logger.info(f"Fold {fold}: chemprop_scratch RAE={fold_raes['chemprop_scratch'][-1]:.4f}")

        # Chemprop HTS-pretrained (lower lr for fine-tuning)
        oof["chemprop_hts"][va_idx] = avg_seed_preds(
            sm_tr, y_tr, sw_tr, sm_va, seeds=CV_SEEDS, epochs=60, lr=5e-4,
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

    # Checkpoint OOF to disk — allows crash recovery without re-running all folds
    oof_checkpoint = {m: oof[m].tolist() for m in model_names}
    oof_checkpoint["train_y"] = train_y.tolist()
    oof_checkpoint["fold_raes"] = {m: fold_raes[m] for m in model_names}
    Path("logs").mkdir(exist_ok=True)
    with open("logs/submission5_oof_checkpoint.json", "w") as f:
        json.dump(oof_checkpoint, f)
    logger.info("OOF checkpoint saved.")

    # ------------------------------------------------------------------ #
    # 5. CV summary
    # ------------------------------------------------------------------ #
    logger.info("\n=== CV Summary ===")
    cv_summary = {}
    for m in model_names:
        mean_r = float(np.mean(fold_raes[m]))
        std_r = float(np.std(fold_raes[m]))
        cv_summary[m] = {"mean": mean_r, "std": std_r}
        logger.info(f"  {m:<20}: {mean_r:.4f} ± {std_r:.4f}")

    # ------------------------------------------------------------------ #
    # 6. ElasticNet stacking on OOF predictions
    # ------------------------------------------------------------------ #
    from src.ensemble.stack_and_submit import ElasticNetStacker

    logger.info("\n=== ElasticNet stacking ===")
    oof_matrix = np.column_stack([oof[m] for m in model_names])

    stacker = ElasticNetStacker(l1_ratio=0.7, cv=5)
    stacker.fit(oof_matrix, train_y, model_names=model_names)

    oof_stacked = stacker.predict(oof_matrix)
    stacked_rae = float(rae(train_y, oof_stacked, y_train_mean=train_mean))
    logger.info(f"ElasticNet OOF RAE: {stacked_rae:.4f}")

    mean_raes = [cv_summary[m]["mean"] for m in model_names]
    inv_rae_weights = np.array([1.0 / r for r in mean_raes])
    inv_rae_weights /= inv_rae_weights.sum()
    inv_rae_oof = (oof_matrix * inv_rae_weights).sum(axis=1)
    inv_rae_val = float(rae(train_y, inv_rae_oof, y_train_mean=train_mean))
    logger.info(f"Inv-RAE ensemble OOF RAE (comparison): {inv_rae_val:.4f}")

    # ------------------------------------------------------------------ #
    # 7. Full training on all data (3 seeds per Chemprop variant)
    # ------------------------------------------------------------------ #
    logger.info("\n=== Full training (all data) ===")
    test_preds = {}

    test_preds["chemprop_scratch"] = avg_seed_preds_full(
        train_smiles, train_y, sample_weights, test_smiles,
        seeds=FINAL_SEEDS, epochs=80, lr=1e-3,
    )

    test_preds["chemprop_hts"] = avg_seed_preds_full(
        train_smiles, train_y, sample_weights, test_smiles,
        seeds=FINAL_SEEDS, epochs=80, lr=5e-4, init_sd=hts_state_dict,
    )
    del hts_state_dict
    gc.collect()

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

    # ------------------------------------------------------------------ #
    # 8. Apply ElasticNet stacker to test predictions
    # ------------------------------------------------------------------ #
    test_matrix = np.column_stack([test_preds[m] for m in model_names])
    final_preds = stacker.predict(test_matrix)
    logger.info(f"Ensemble: mean={final_preds.mean():.3f}, std={final_preds.std():.3f}")

    # ------------------------------------------------------------------ #
    # 9. Save submission
    # ------------------------------------------------------------------ #
    from src.ensemble.stack_and_submit import make_submission, validate_submission

    submission_path = make_submission(
        predictions=final_preds,
        test_df=test,
        description="Sub5: HTS-pretrained Chemprop + ElasticNet stacking + Butina CV",
        val_rae=stacked_rae,
    )
    validate_submission(str(submission_path), test)
    logger.info(f"Submission saved: {submission_path}")

    # ------------------------------------------------------------------ #
    # 10. Save CV results to JSON
    # ------------------------------------------------------------------ #
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
    Path(RESULTS_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"CV results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
