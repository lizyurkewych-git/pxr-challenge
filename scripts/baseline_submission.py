"""
Week 1 Baseline: kNN + GBM ensemble → first submission.

Runs end-to-end:
  1. Load all data tiers
  2. Compute ECFP4 + RDKit features
  3. Scaffold-stratified 5-fold CV (sanity check: must beat RAE=1.0)
  4. Train kNN (k=5) and LightGBM on full training set
  5. Ensemble predictions (equal weight kNN + LGBM)
  6. Generate submission CSV

Usage:
    conda activate pxr
    cd /path/to/pxr-challenge
    python scripts/baseline_submission.py
"""

import logging
import sys
from pathlib import Path

import numpy as np

# Ensure project src is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("baseline")


def main():
    # ------------------------------------------------------------------ #
    # 1. Load data
    # ------------------------------------------------------------------ #
    from src.data.load_data import load_all_tiers, COL_SMILES, COL_PECSO

    logger.info("Loading dataset...")
    ds = load_all_tiers(cache_dir="data/hf_cache")
    train = ds.train
    test = ds.test

    train_smiles = train[COL_SMILES].tolist()
    train_y = train[COL_PECSO].values
    test_smiles = test[COL_SMILES].tolist()
    train_mean = float(train_y.mean())

    logger.info(f"Train: {len(train)}, Test: {len(test)}")
    logger.info(f"pEC50 mean (train): {train_mean:.3f}, std: {train_y.std():.3f}")

    # ------------------------------------------------------------------ #
    # 2. Compute features
    # ------------------------------------------------------------------ #
    from src.features.feature_engineering import ecfp4, FeaturePipeline

    logger.info("Computing ECFP4 fingerprints...")
    fps_train = ecfp4(train_smiles)
    fps_test = ecfp4(test_smiles)

    logger.info("Building combined feature matrix (ECFP4 + RDKit + Mordred)...")
    pipeline = FeaturePipeline(include_mordred=True, mordred_pca_components=200)
    X_train = pipeline.fit_transform(train_smiles)
    X_test = pipeline.transform(test_smiles)
    logger.info(f"Feature matrix: train={X_train.shape}, test={X_test.shape}")

    # ------------------------------------------------------------------ #
    # 3. Scaffold-stratified cross-validation
    # ------------------------------------------------------------------ #
    from src.evaluation.validate import ScaffoldKFold, rae, full_metrics
    from src.models.local_models import TanimotoKNN
    from src.models.gbm_models import LGBMWrapper

    logger.info("\n=== Scaffold 5-fold CV ===")
    splitter = ScaffoldKFold(n_splits=5)

    knn_raes, lgbm_raes, ensemble_raes = [], [], []

    for fold, (tr_idx, va_idx) in enumerate(splitter.split(train_smiles)):
        fps_tr, fps_va = fps_train[tr_idx], fps_train[va_idx]
        X_tr, X_va = X_train[tr_idx], X_train[va_idx]
        y_tr, y_va = train_y[tr_idx], train_y[va_idx]
        fold_mean = float(y_tr.mean())

        # kNN
        knn = TanimotoKNN(k=5)
        knn.fit(fps_tr, y_tr)
        knn_pred = knn.predict(fps_va)

        # LightGBM (no mordred for speed; use fingerprints only for CV)
        lgbm = LGBMWrapper({"n_estimators": 1000, "learning_rate": 0.05, "verbose": -1})
        lgbm.fit(X_tr, y_tr, X_val=X_va, y_val=y_va)
        lgbm_pred = lgbm.predict(X_va)

        # Ensemble (equal weight)
        ens_pred = 0.5 * knn_pred + 0.5 * lgbm_pred

        r_knn = rae(y_va, knn_pred, y_train_mean=fold_mean)
        r_lgbm = rae(y_va, lgbm_pred, y_train_mean=fold_mean)
        r_ens = rae(y_va, ens_pred, y_train_mean=fold_mean)

        knn_raes.append(r_knn)
        lgbm_raes.append(r_lgbm)
        ensemble_raes.append(r_ens)

        logger.info(
            f"Fold {fold}: kNN={r_knn:.4f}, LGBM={r_lgbm:.4f}, Ensemble={r_ens:.4f} "
            f"(n_val={len(va_idx)})"
        )

    logger.info("\nCross-Validation Summary:")
    logger.info(f"  kNN   RAE: {np.mean(knn_raes):.4f} ± {np.std(knn_raes):.4f}")
    logger.info(f"  LGBM  RAE: {np.mean(lgbm_raes):.4f} ± {np.std(lgbm_raes):.4f}")
    logger.info(f"  Ensemble RAE: {np.mean(ensemble_raes):.4f} ± {np.std(ensemble_raes):.4f}")

    if np.mean(ensemble_raes) >= 1.0:
        logger.warning("Ensemble RAE >= 1.0 in CV — not beating the mean baseline. Check data loading!")
    else:
        logger.info("Sanity check passed: ensemble beats mean-prediction baseline in CV.")

    # ------------------------------------------------------------------ #
    # 4. Train on full training set
    # ------------------------------------------------------------------ #
    logger.info("\n=== Training on full training set ===")

    # kNN (no training needed — just stores data)
    knn_full = TanimotoKNN(k=5)
    knn_full.fit(fps_train, train_y)

    # LightGBM on full data (split last 10% for early stopping)
    n_val = max(100, int(0.1 * len(X_train)))
    lgbm_full = LGBMWrapper()
    sw = train["sample_weight"].values[:-n_val] if "sample_weight" in train.columns else None
    lgbm_full.fit(
        X_train[:-n_val], train_y[:-n_val],
        X_val=X_train[-n_val:], y_val=train_y[-n_val:],
        sample_weight=sw,
    )

    # ------------------------------------------------------------------ #
    # 5. Predict on test set
    # ------------------------------------------------------------------ #
    logger.info("\n=== Generating test predictions ===")

    knn_test = knn_full.predict(fps_test)
    lgbm_test = lgbm_full.predict(X_test)

    # Determine ensemble weights from CV RAE
    w_knn = 1.0 / np.mean(knn_raes)
    w_lgbm = 1.0 / np.mean(lgbm_raes)
    w_total = w_knn + w_lgbm

    ensemble_test = (w_knn * knn_test + w_lgbm * lgbm_test) / w_total
    ensemble_test = np.clip(ensemble_test, 1.5, 8.0)

    logger.info(f"Test predictions: mean={ensemble_test.mean():.3f}, std={ensemble_test.std():.3f}")
    logger.info(f"Ensemble weights: kNN={w_knn/w_total:.3f}, LGBM={w_lgbm/w_total:.3f}")

    # ------------------------------------------------------------------ #
    # 6. Submit
    # ------------------------------------------------------------------ #
    from src.ensemble.stack_and_submit import make_submission, validate_submission

    submission_path = make_submission(
        predictions=ensemble_test,
        test_df=test,
        description=f"Week1_kNN+LGBM_ensemble | CV_RAE={np.mean(ensemble_raes):.4f}",
        val_rae=float(np.mean(ensemble_raes)),
    )
    logger.info(f"\nSubmission saved to: {submission_path}")

    valid = validate_submission(str(submission_path), test)
    if valid:
        logger.info("Submission validation PASSED — ready to upload to HuggingFace.")
    else:
        logger.error("Submission validation FAILED — check the errors above.")

    return submission_path


if __name__ == "__main__":
    path = main()
    print(f"\nDone. Submission at: {path}")
