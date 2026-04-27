"""
Submission 7: TabPFN + CheMeleon in-context learning added to Sub 6 stack.

New in this submission vs Sub 6:
  - TabPFNCheMeleonModel: TabPFN v2 regressor using frozen CheMeleon 2048-dim
    embeddings compressed to PCA(200). No gradient updates at fit time — TabPFN
    performs Bayesian in-context learning by conditioning on the full training
    set at inference. Motivated by Ben Hicham et al. (2025), which shows
    TabPFN + CheMeleon achieves up to 100% win rate on the MoleculeACE activity
    cliff benchmark — the same analog-series structure as our test set.
  - Sub 4 used CheMeleon + LGBM and failed. The key difference: TabPFN sees
    (X_train, y_train) at prediction time; LGBM does not.
  - All Sub 6 models carried over unchanged (delta, chemprop_hts,
    chemprop_scratch, knn, lgbm, rf).

ElasticNet findings:
  tabpfn=0.42, chemprop_hts=0.33, rf=0.13, delta=0.15, knn=0.04,
  lgbm=0.00, chemprop_scratch=-0.12 (bias corrector, not zeroed)

OOF RAE: 0.5297 (improved from 0.5481 in Sub 6)

Requires Python 3.11 + tabpfn==2.2.1 (install with --no-deps to avoid
huggingface-hub downgrade conflict with transformers):
    pip install "tabpfn==2.2.1" --no-deps
    .venv311/bin/python scripts/submission7_tabpfn.py
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
logger = logging.getLogger("submission7")

CV_SEEDS = [42, 7]
FINAL_SEEDS = [42, 7, 13]
RESULTS_PATH = "logs/submission7_cv_results.json"

# Set to "mps" on Apple Silicon, "cuda" if GPU available, else "cpu"
DEVICE = "mps"

# Delta model training budget
DELTA_CV_PAIRS = 20_000
DELTA_CV_EPOCHS = 40
DELTA_FINAL_PAIRS = 20_000
DELTA_FINAL_EPOCHS = 50


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
    del pretrain_model; gc.collect()
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
    # 4. Butina 5-fold CV
    # ---------------------------------------------------------- #
    from src.evaluation.validate import ButinaKFold, rae
    from src.models.local_models import TanimotoKNN
    from src.models.gbm_models import LGBMWrapper, RFWrapper
    from src.models.delta_model import DeltaChempropModel
    from src.models.tabpfn_model import TabPFNCheMeleonModel

    logger.info("\n=== Butina 5-fold CV ===")
    splitter = ButinaKFold(n_splits=5, tanimoto_threshold=0.4)
    folds = list(splitter.split(train_smiles))

    model_names = ["delta", "chemprop_hts", "chemprop_scratch", "tabpfn", "knn", "lgbm", "rf"]
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

        # Delta
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

        # Chemprop HTS
        logger.info(f"Fold {fold}: Chemprop HTS-pretrained...")
        oof["chemprop_hts"][va_idx] = avg_chemprop(
            sm_tr, y_tr, sw_tr, sm_va, CV_SEEDS, epochs=60, lr=5e-4,
            init_sd=hts_state_dict,
        )
        fold_raes["chemprop_hts"].append(
            rae(y_va, oof["chemprop_hts"][va_idx], y_train_mean=fold_mean)
        )
        logger.info(f"Fold {fold}: chemprop_hts RAE={fold_raes['chemprop_hts'][-1]:.4f}")

        # Chemprop scratch
        logger.info(f"Fold {fold}: Chemprop scratch...")
        oof["chemprop_scratch"][va_idx] = avg_chemprop(
            sm_tr, y_tr, sw_tr, sm_va, CV_SEEDS, epochs=60, lr=1e-3
        )
        fold_raes["chemprop_scratch"].append(
            rae(y_va, oof["chemprop_scratch"][va_idx], y_train_mean=fold_mean)
        )
        logger.info(f"Fold {fold}: chemprop_scratch RAE={fold_raes['chemprop_scratch'][-1]:.4f}")

        # TabPFN + CheMeleon (in-context learning — fast, no gradient updates)
        logger.info(f"Fold {fold}: TabPFN + CheMeleon...")
        tabpfn_cv = TabPFNCheMeleonModel(
            n_components=200, n_estimators=16, device="cpu", random_state=42
        )
        tabpfn_cv.fit(sm_tr, y_tr)
        oof["tabpfn"][va_idx] = tabpfn_cv.predict(sm_va)
        fold_raes["tabpfn"].append(rae(y_va, oof["tabpfn"][va_idx], y_train_mean=fold_mean))
        logger.info(f"Fold {fold}: tabpfn RAE={fold_raes['tabpfn'][-1]:.4f}")
        del tabpfn_cv; gc.collect()

        # kNN
        knn = TanimotoKNN(k=5)
        knn.fit(fps_tr, y_tr)
        oof["knn"][va_idx] = knn.predict(fps_va)
        fold_raes["knn"].append(rae(y_va, oof["knn"][va_idx], y_train_mean=fold_mean))

        # LGBM
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
    # 6. ElasticNet stacking
    # ---------------------------------------------------------- #
    from src.ensemble.stack_and_submit import ElasticNetStacker

    logger.info("\n=== ElasticNet stacking ===")
    oof_matrix = np.column_stack([oof[m] for m in model_names])
    stacker = ElasticNetStacker(l1_ratio=0.7, cv=5)
    stacker.fit(oof_matrix, train_y, model_names=model_names)

    oof_stacked = stacker.predict(oof_matrix)
    stacked_rae = float(rae(train_y, oof_stacked, y_train_mean=train_mean))
    logger.info(f"ElasticNet OOF RAE: {stacked_rae:.4f}")

    Path("logs").mkdir(exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump({
            "cv_summary": cv_summary,
            "elasticnet_oof_rae": stacked_rae,
            "elasticnet_coefs": stacker.coefs,
        }, f, indent=2)

    # ---------------------------------------------------------- #
    # 7. Full training
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

    logger.info("Full training: Chemprop HTS (3 seeds)...")
    test_preds["chemprop_hts"] = avg_chemprop_full(
        train_smiles, train_y, sample_weights, test_smiles,
        FINAL_SEEDS, epochs=80, lr=5e-4, init_sd=hts_state_dict,
    )

    logger.info("Full training: Chemprop scratch (3 seeds)...")
    test_preds["chemprop_scratch"] = avg_chemprop_full(
        train_smiles, train_y, sample_weights, test_smiles,
        FINAL_SEEDS, epochs=80, lr=1e-3,
    )
    del hts_state_dict; gc.collect()

    logger.info("Full training: TabPFN + CheMeleon...")
    tabpfn_full = TabPFNCheMeleonModel(
        n_components=200, n_estimators=16, device="cpu", random_state=42
    )
    tabpfn_full.fit(train_smiles, train_y)
    test_preds["tabpfn"] = tabpfn_full.predict(test_smiles)
    del tabpfn_full; gc.collect()

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

    # ---------------------------------------------------------- #
    # 8. Apply stacker and save submission
    # ---------------------------------------------------------- #
    test_matrix = np.column_stack([test_preds[m] for m in model_names])
    final_preds = stacker.predict(test_matrix)
    logger.info(f"Ensemble: mean={final_preds.mean():.3f}, std={final_preds.std():.3f}")

    from src.ensemble.stack_and_submit import make_submission, validate_submission

    submission_path = make_submission(
        test_df=test,
        predictions=final_preds,
        output_dir="submissions",
    )
    logger.info(f"Submission saved to {submission_path}")

    ok = validate_submission(submission_path, test)
    if ok:
        logger.info("Submission validation PASSED.")
    else:
        logger.error("Submission validation FAILED — check output above.")


if __name__ == "__main__":
    main()
