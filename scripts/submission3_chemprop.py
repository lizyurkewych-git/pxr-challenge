"""
Submission 3: Chemprop D-MPNN + kNN + GBM ensemble.

New in this submission:
  - Chemprop v2 D-MPNN (message-passing neural network on molecular graphs)
  - Snapshot ensembling: average last 5 epoch checkpoints
  - MPS (Apple Silicon M4) acceleration
  - Ensemble: Chemprop + kNN + LightGBM + XGBoost + RF (inverse-RAE weights)

IMPORTANT — ordering in this script:
  Chemprop is trained and run for inference BEFORE FeaturePipeline.fit_transform().
  Mordred (called inside FeaturePipeline) uses multiprocessing that leaks semaphores
  on macOS Apple M4, which causes a segfault if PyTorch is still active. Running
  Chemprop first avoids this entirely.

Requires Python 3.11 venv (.venv311):
    .venv311/bin/python scripts/submission3_chemprop.py
"""

import logging
import os
import sys
from pathlib import Path

import numpy as np

# Must be set before importing lightgbm/xgboost/sklearn — prevents the Intel OpenMP
# runtime from aborting when it detects PyTorch's bundled libiomp5 alongside brew libomp.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# Single-threaded OpenMP avoids thread-pool crashes on macOS Apple Silicon.
os.environ.setdefault("OMP_NUM_THREADS", "1")
# datasets 4.8+ makes a usage-tracking HEAD request on every load_dataset() call.
# Setting offline mode skips the network call and reads straight from local cache.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("submission3")


def build_sample_weights(train) -> np.ndarray:
    """Combine inverse-variance, cliff, and non-specific compound weights."""
    w = train["sample_weight"].values.copy() if "sample_weight" in train.columns \
        else np.ones(len(train))
    if "is_nonspecific" in train.columns:
        w[train["is_nonspecific"].values] *= 0.3
    if "cliff_sample_weight" in train.columns:
        w *= train["cliff_sample_weight"].values
    w = w / w.mean()
    return w.astype(np.float32)


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

    logger.info("Identifying activity cliffs...")
    cliff_pairs = identify_activity_cliffs(
        smiles=train[COL_SMILES].tolist(),
        activities=train[COL_PECSO].values,
        sim_threshold=0.7,
        activity_threshold=1.0,
    )
    train = annotate_cliff_compounds(train, cliff_pairs)

    train_smiles = train[COL_SMILES].tolist()
    train_y = train[COL_PECSO].values
    test_smiles = test[COL_SMILES].tolist()
    train_mean = float(train_y.mean())
    sample_weights = build_sample_weights(train)

    logger.info(f"Train: {len(train)} | Test: {len(test)}")

    # ------------------------------------------------------------------ #
    # 2. ECFP4 fingerprints (no multiprocessing — safe before Chemprop)
    # ------------------------------------------------------------------ #
    from src.features.feature_engineering import ecfp4
    from src.evaluation.validate import ScaffoldKFold, rae

    logger.info("Computing ECFP4 fingerprints...")
    fps_train = ecfp4(train_smiles)
    fps_test = ecfp4(test_smiles)

    splitter = ScaffoldKFold(n_splits=5)
    folds = list(splitter.split(train_smiles))

    # ------------------------------------------------------------------ #
    # 3. Chemprop CV — MUST run before any Mordred/FeaturePipeline calls
    # ------------------------------------------------------------------ #
    from src.models.chemprop_model import ChempropModel

    logger.info("\n=== Chemprop Scaffold 5-fold CV ===")
    chemprop_oof = np.zeros(len(train))
    chemprop_fold_raes = []

    for fold, (tr_idx, va_idx) in enumerate(folds):
        logger.info(f"--- Fold {fold} ---")
        sm_tr = [train_smiles[i] for i in tr_idx]
        sm_va = [train_smiles[i] for i in va_idx]
        y_tr, y_va = train_y[tr_idx], train_y[va_idx]
        sw_tr = sample_weights[tr_idx]
        fold_mean = float(y_tr.mean())

        m = ChempropModel(
            epochs=80,
            hidden_size=300,
            depth=3,
            ffn_num_layers=3,
            dropout=0.1,
            batch_size=64,
            lr=1e-3,
            device="cpu",
            snapshot_epochs=5,
            extra_features=False,
            seed=42 + fold,
        )
        m.fit(sm_tr, y_tr, sample_weight=sw_tr)
        chemprop_oof[va_idx] = m.predict(sm_va)
        r = rae(y_va, chemprop_oof[va_idx], y_train_mean=fold_mean)
        chemprop_fold_raes.append(r)
        logger.info(f"Fold {fold}: Chemprop RAE={r:.4f}")

    chemprop_mean_rae = float(np.mean(chemprop_fold_raes))
    logger.info(f"Chemprop CV RAE: {chemprop_mean_rae:.4f} ± {np.std(chemprop_fold_raes):.4f}")

    # ------------------------------------------------------------------ #
    # 4. Full Chemprop training + test predictions (still before Mordred)
    # ------------------------------------------------------------------ #
    logger.info("\n=== Full Chemprop training ===")
    chemprop_full = ChempropModel(
        epochs=100,
        hidden_size=300,
        depth=3,
        ffn_num_layers=3,
        dropout=0.1,
        batch_size=64,
        lr=1e-3,
        device="cpu",
        snapshot_epochs=5,
        extra_features=False,
        seed=42,
    )
    chemprop_full.fit(train_smiles, train_y, sample_weight=sample_weights)
    chemprop_test_preds = chemprop_full.predict(test_smiles)
    logger.info(f"Chemprop test: mean={chemprop_test_preds.mean():.3f}, std={chemprop_test_preds.std():.3f}")

    # Release Chemprop model from memory before Mordred multiprocessing
    import gc
    del chemprop_full
    gc.collect()
    logger.info("Chemprop model released — safe to start Mordred.")

    # ------------------------------------------------------------------ #
    # 5. GBM feature matrix (Mordred multiprocessing happens here)
    # ------------------------------------------------------------------ #
    from src.features.feature_engineering import FeaturePipeline

    logger.info("\n=== Building GBM feature matrix ===")
    # include_mordred=False: mordred-community's multiprocessing.Manager() leaks a
    # semaphore on macOS Python 3.11, crashing the process. ECFP4+ECFP6+RDKit is
    # sufficient for the GBM models; Chemprop covers the graph-level signal.
    pipeline = FeaturePipeline(
        include_mordred=False,
        include_ecfp6=True,
        include_fcfp4=False,
    )
    X_train = pipeline.fit_transform(train_smiles)
    X_test = pipeline.transform(test_smiles)
    logger.info(f"GBM feature matrix: train={X_train.shape}, test={X_test.shape}")

    # ------------------------------------------------------------------ #
    # 6. GBM + kNN CV
    # ------------------------------------------------------------------ #
    from src.models.local_models import TanimotoKNN
    from src.models.gbm_models import LGBMWrapper, XGBWrapper, RFWrapper

    gbm_model_names = ["knn", "lgbm", "xgb", "rf"]
    fold_raes = {m: [] for m in gbm_model_names}
    fold_raes["chemprop"] = chemprop_fold_raes
    fold_preds = {m: np.zeros(len(train)) for m in gbm_model_names}
    fold_preds["chemprop"] = chemprop_oof

    logger.info("\n=== GBM + kNN Scaffold 5-fold CV ===")
    for fold, (tr_idx, va_idx) in enumerate(folds):
        logger.info(f"--- Fold {fold} ---")
        fps_tr, fps_va = fps_train[tr_idx], fps_train[va_idx]
        X_tr, X_va = X_train[tr_idx], X_train[va_idx]
        y_tr, y_va = train_y[tr_idx], train_y[va_idx]
        sw_tr = sample_weights[tr_idx]
        fold_mean = float(y_tr.mean())

        n_es = max(50, int(0.1 * len(tr_idx)))
        X_es, y_es = X_tr[-n_es:], y_tr[-n_es:]
        X_tr_fit, y_tr_fit = X_tr[:-n_es], y_tr[:-n_es]
        sw_fit = sw_tr[:-n_es]

        # kNN
        knn = TanimotoKNN(k=5)
        knn.fit(fps_tr, y_tr)
        fold_preds["knn"][va_idx] = knn.predict(fps_va)

        # LightGBM (n_jobs=1: avoids OpenMP/libomp conflict with PyTorch on macOS)
        lgbm = LGBMWrapper({"n_jobs": 1})
        lgbm.fit(X_tr_fit, y_tr_fit, X_val=X_es, y_val=y_es, sample_weight=sw_fit)
        fold_preds["lgbm"][va_idx] = lgbm.predict(X_va)

        # XGBoost (n_jobs=1: same reason)
        xgb = XGBWrapper({"n_jobs": 1})
        xgb.fit(X_tr_fit, y_tr_fit, X_val=X_es, y_val=y_es, sample_weight=sw_fit)
        fold_preds["xgb"][va_idx] = xgb.predict(X_va)

        # Random Forest (n_jobs=1: same reason)
        rf = RFWrapper({"n_jobs": 1})
        rf.fit(X_tr, y_tr, sample_weight=sw_tr)
        fold_preds["rf"][va_idx] = rf.predict(X_va)

        for m in gbm_model_names:
            r = rae(y_va, fold_preds[m][va_idx], y_train_mean=fold_mean)
            fold_raes[m].append(r)

        logger.info(
            f"Fold {fold}: kNN={fold_raes['knn'][-1]:.4f}, "
            f"LGBM={fold_raes['lgbm'][-1]:.4f}, "
            f"XGB={fold_raes['xgb'][-1]:.4f}, "
            f"RF={fold_raes['rf'][-1]:.4f}"
        )

    all_model_names = ["chemprop", "knn", "lgbm", "xgb", "rf"]
    logger.info("\nCV Summary:")
    mean_raes = {}
    for m in all_model_names:
        mean_raes[m] = float(np.mean(fold_raes[m]))
        logger.info(f"  {m:10s}: {mean_raes[m]:.4f} ± {np.std(fold_raes[m]):.4f}")

    # Inverse-RAE ensemble weights
    weights = {m: 1.0 / mean_raes[m] for m in all_model_names}
    total_w = sum(weights.values())
    ens_preds_cv = sum(weights[m] * fold_preds[m] for m in all_model_names) / total_w
    ens_rae = rae(train_y, ens_preds_cv, y_train_mean=train_mean)
    logger.info(f"\n  ensemble (inv-RAE): {ens_rae:.4f}")
    for m in all_model_names:
        logger.info(f"    {m}: weight = {weights[m]/total_w:.3f}")

    # ------------------------------------------------------------------ #
    # 7. Train full GBM models on complete training set
    # ------------------------------------------------------------------ #
    logger.info("\n=== Training full GBM models ===")

    n_es_full = max(100, int(0.1 * len(X_train)))
    X_es_full, y_es_full = X_train[-n_es_full:], train_y[-n_es_full:]
    X_fit, y_fit = X_train[:-n_es_full], train_y[:-n_es_full]
    sw_fit_full = sample_weights[:-n_es_full]

    knn_full = TanimotoKNN(k=5)
    knn_full.fit(fps_train, train_y)

    lgbm_full = LGBMWrapper({"n_jobs": 1})
    lgbm_full.fit(X_fit, y_fit, X_val=X_es_full, y_val=y_es_full, sample_weight=sw_fit_full)

    xgb_full = XGBWrapper({"n_jobs": 1})
    xgb_full.fit(X_fit, y_fit, X_val=X_es_full, y_val=y_es_full, sample_weight=sw_fit_full)

    rf_full = RFWrapper({"n_jobs": 1})
    rf_full.fit(X_train, train_y, sample_weight=sample_weights)

    # ------------------------------------------------------------------ #
    # 8. Predict on test set and ensemble
    # ------------------------------------------------------------------ #
    logger.info("\n=== Generating test predictions ===")

    test_preds = {
        "chemprop": chemprop_test_preds,
        "knn":      knn_full.predict(fps_test),
        "lgbm":     lgbm_full.predict(X_test),
        "xgb":      xgb_full.predict(X_test),
        "rf":       rf_full.predict(X_test),
    }

    for m in all_model_names:
        logger.info(f"  {m}: mean={test_preds[m].mean():.3f}, std={test_preds[m].std():.3f}")

    ensemble_test = sum(weights[m] * test_preds[m] for m in all_model_names) / total_w
    ensemble_test = np.clip(ensemble_test, 1.5, 8.0)

    logger.info(f"Ensemble: mean={ensemble_test.mean():.3f}, std={ensemble_test.std():.3f}")

    # ------------------------------------------------------------------ #
    # 9. Submit
    # ------------------------------------------------------------------ #
    from src.ensemble.stack_and_submit import make_submission, validate_submission

    submission_path = make_submission(
        predictions=ensemble_test,
        test_df=test,
        description=(
            f"Sub3_Chemprop+kNN+LGBM+XGB+RF_invRAE | "
            f"CV_RAE={ens_rae:.4f} | "
            f"Chemprop={mean_raes['chemprop']:.4f} "
            f"kNN={mean_raes['knn']:.4f} "
            f"LGBM={mean_raes['lgbm']:.4f} "
            f"XGB={mean_raes['xgb']:.4f} "
            f"RF={mean_raes['rf']:.4f}"
        ),
        val_rae=float(ens_rae),
    )

    valid = validate_submission(str(submission_path), test)
    if valid:
        logger.info(f"\nSubmission saved: {submission_path}")
        logger.info("Validation PASSED — ready to upload to HuggingFace.")
    else:
        logger.error("Validation FAILED — check errors above.")

    return submission_path


if __name__ == "__main__":
    path = main()
    print(f"\nDone. Submission at: {path}")
