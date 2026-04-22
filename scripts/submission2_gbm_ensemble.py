"""
Week 2: kNN + LightGBM + XGBoost + Random Forest — inverse-RAE weighted ensemble.

Improvements over submission 1:
  - XGBoost and Random Forest added as ensemble members
  - Inverse-RAE weighted ensemble (CV-calibrated weights instead of equal weights)
  - ECFP6 fingerprints added to feature matrix
  - Activity cliff sample reweighting in GBM training
  - Non-specific compound downweighting (counter-assay flagging)
  - Combined inverse-variance + cliff + nonspecific sample weights

Usage:
    cd /path/to/pxr-challenge-public
    python scripts/submission2_gbm_ensemble.py
"""

import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("submission2")


def build_sample_weights(train) -> np.ndarray:
    """Combine inverse-variance, cliff, and non-specific compound weights."""
    w = train["sample_weight"].values.copy() if "sample_weight" in train.columns \
        else np.ones(len(train))

    # Downweight non-specific compounds (likely counter-assay artefacts)
    if "is_nonspecific" in train.columns:
        w[train["is_nonspecific"].values] *= 0.3

    # Upweight activity cliff compounds (hardest cases)
    if "cliff_sample_weight" in train.columns:
        cliff_w = train["cliff_sample_weight"].values
        w *= cliff_w

    # Normalize to mean=1 so loss scale is stable
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

    # Annotate activity cliffs for sample reweighting
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

    logger.info(f"Train: {len(train)} | Test: {len(test)}")
    logger.info(f"Cliff compounds: {train['is_cliff_member'].sum()} "
                f"({100*train['is_cliff_member'].mean():.1f}%)")
    logger.info(f"Non-specific compounds: {train['is_nonspecific'].sum()} "
                f"({100*train['is_nonspecific'].mean():.1f}%)")

    sample_weights = build_sample_weights(train)

    # ------------------------------------------------------------------ #
    # 2. Compute features
    # ------------------------------------------------------------------ #
    from src.features.feature_engineering import ecfp4, FeaturePipeline

    logger.info("Computing ECFP4 fingerprints...")
    fps_train = ecfp4(train_smiles)
    fps_test = ecfp4(test_smiles)

    logger.info("Building combined feature matrix (ECFP4 + ECFP6 + RDKit + Mordred)...")
    pipeline = FeaturePipeline(
        include_mordred=True,
        mordred_pca_components=200,
        include_ecfp6=True,   # adds ECFP6 (radius=3) fingerprints
        include_fcfp4=False,
    )
    X_train = pipeline.fit_transform(train_smiles)
    X_test = pipeline.transform(test_smiles)
    logger.info(f"Feature matrix: train={X_train.shape}, test={X_test.shape}")

    # ------------------------------------------------------------------ #
    # 3. Scaffold-stratified CV — all 4 models
    # ------------------------------------------------------------------ #
    from src.evaluation.validate import ScaffoldKFold, rae, full_metrics
    from src.models.local_models import TanimotoKNN
    from src.models.gbm_models import LGBMWrapper, XGBWrapper, RFWrapper

    logger.info("\n=== Scaffold 5-fold CV ===")
    splitter = ScaffoldKFold(n_splits=5)

    model_names = ["knn", "lgbm", "xgb", "rf"]
    fold_raes = {m: [] for m in model_names}
    fold_preds = {m: np.zeros(len(train)) for m in model_names}

    for fold, (tr_idx, va_idx) in enumerate(splitter.split(train_smiles)):
        fps_tr, fps_va = fps_train[tr_idx], fps_train[va_idx]
        X_tr, X_va = X_train[tr_idx], X_train[va_idx]
        y_tr, y_va = train_y[tr_idx], train_y[va_idx]
        sw_tr = sample_weights[tr_idx]
        fold_mean = float(y_tr.mean())

        # Split last 10% of fold train for early stopping
        n_es = max(50, int(0.1 * len(tr_idx)))
        X_es, y_es = X_tr[-n_es:], y_tr[-n_es:]
        X_tr_fit, y_tr_fit = X_tr[:-n_es], y_tr[:-n_es]
        sw_fit = sw_tr[:-n_es]

        # kNN
        knn = TanimotoKNN(k=5)
        knn.fit(fps_tr, y_tr)
        fold_preds["knn"][va_idx] = knn.predict(fps_va)

        # LightGBM
        lgbm = LGBMWrapper()
        lgbm.fit(X_tr_fit, y_tr_fit, X_val=X_es, y_val=y_es, sample_weight=sw_fit)
        fold_preds["lgbm"][va_idx] = lgbm.predict(X_va)

        # XGBoost
        xgb = XGBWrapper()
        xgb.fit(X_tr_fit, y_tr_fit, X_val=X_es, y_val=y_es, sample_weight=sw_fit)
        fold_preds["xgb"][va_idx] = xgb.predict(X_va)

        # Random Forest (no early stopping)
        rf = RFWrapper()
        rf.fit(X_tr, y_tr, sample_weight=sw_tr)
        fold_preds["rf"][va_idx] = rf.predict(X_va)

        for m in model_names:
            r = rae(y_va, fold_preds[m][va_idx], y_train_mean=fold_mean)
            fold_raes[m].append(r)

        logger.info(
            f"Fold {fold}: kNN={fold_raes['knn'][-1]:.4f}, "
            f"LGBM={fold_raes['lgbm'][-1]:.4f}, "
            f"XGB={fold_raes['xgb'][-1]:.4f}, "
            f"RF={fold_raes['rf'][-1]:.4f}"
        )

    logger.info("\nCV Summary:")
    mean_raes = {}
    for m in model_names:
        mean_raes[m] = np.mean(fold_raes[m])
        logger.info(f"  {m:6s}: {mean_raes[m]:.4f} ± {np.std(fold_raes[m]):.4f}")

    # Inverse-RAE weighted ensemble
    weights = {m: 1.0 / mean_raes[m] for m in model_names}
    total_w = sum(weights.values())
    ens_preds_cv = sum(weights[m] * fold_preds[m] for m in model_names) / total_w
    ens_rae = rae(train_y, ens_preds_cv, y_train_mean=train_mean)
    logger.info(f"\n  ensemble (inv-RAE weighted): {ens_rae:.4f}")
    for m in model_names:
        logger.info(f"    {m}: weight = {weights[m]/total_w:.3f}")

    if ens_rae >= 1.0:
        logger.warning("Ensemble RAE >= 1.0 in CV — not beating the mean baseline!")
    else:
        logger.info("Sanity check passed: ensemble beats mean-prediction baseline.")

    # ------------------------------------------------------------------ #
    # 4. Train on full training set
    # ------------------------------------------------------------------ #
    logger.info("\n=== Training on full training set ===")

    n_es_full = max(100, int(0.1 * len(X_train)))
    X_es_full, y_es_full = X_train[-n_es_full:], train_y[-n_es_full:]
    X_fit, y_fit = X_train[:-n_es_full], train_y[:-n_es_full]
    sw_fit_full = sample_weights[:-n_es_full]

    knn_full = TanimotoKNN(k=5)
    knn_full.fit(fps_train, train_y)

    lgbm_full = LGBMWrapper()
    lgbm_full.fit(X_fit, y_fit, X_val=X_es_full, y_val=y_es_full, sample_weight=sw_fit_full)

    xgb_full = XGBWrapper()
    xgb_full.fit(X_fit, y_fit, X_val=X_es_full, y_val=y_es_full, sample_weight=sw_fit_full)

    rf_full = RFWrapper()
    rf_full.fit(X_train, train_y, sample_weight=sample_weights)

    # ------------------------------------------------------------------ #
    # 5. Predict on test set
    # ------------------------------------------------------------------ #
    logger.info("\n=== Generating test predictions ===")

    test_preds = {
        "knn":  knn_full.predict(fps_test),
        "lgbm": lgbm_full.predict(X_test),
        "xgb":  xgb_full.predict(X_test),
        "rf":   rf_full.predict(X_test),
    }

    ensemble_test = sum(weights[m] * test_preds[m] for m in model_names) / total_w
    ensemble_test = np.clip(ensemble_test, 1.5, 8.0)

    logger.info(f"Test predictions: mean={ensemble_test.mean():.3f}, "
                f"std={ensemble_test.std():.3f}")

    # ------------------------------------------------------------------ #
    # 6. Submit
    # ------------------------------------------------------------------ #
    from src.ensemble.stack_and_submit import make_submission, validate_submission

    submission_path = make_submission(
        predictions=ensemble_test,
        test_df=test,
        description=(
            f"Week2_kNN+LGBM+XGB+RF_invRAE_weights | "
            f"CV_RAE={ens_rae:.4f} | "
            f"kNN={mean_raes['knn']:.4f} LGBM={mean_raes['lgbm']:.4f} "
            f"XGB={mean_raes['xgb']:.4f} RF={mean_raes['rf']:.4f}"
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
