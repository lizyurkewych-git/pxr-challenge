"""
Submission 4: Chemprop + foundation model embeddings (CheMeleon + ChemBERTa) + GBM ensemble.

New in this submission:
  - CheMeleon (2048-dim): pretrained Chemprop D-MPNN fingerprints from Zenodo
  - ChemBERTa (384-dim): SMILES-based BERT pretrained on 77M molecules
  - Two GBM model flavors:
      * "trad":  ECFP4 + ECFP6 + RDKit (same as Sub 3)
      * "found": CheMeleon + ChemBERTa + RDKit (new, foundation model features)
  - Ensemble: Chemprop + kNN + LGBM_trad + LGBM_found + XGB_found + RF_found (inv-RAE)
  - Embeddings cached to data/embed_cache/ — second run is fast

Design rationale:
  Keeping traditional and foundation-model GBM tracks separate lets us isolate
  the contribution of pretrained embeddings. If LGBM_found >> LGBM_trad, we know
  CheMeleon/ChemBERTa are genuinely adding signal beyond ECFP; if not, we fall
  back to the Sub 3 stack for Sub 5.

Requires Python 3.11 venv (.venv311):
    .venv311/bin/python scripts/submission4_foundation_models.py
"""

import logging
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
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
logger = logging.getLogger("submission4")

EMBED_CACHE = "data/embed_cache"


def build_sample_weights(train) -> np.ndarray:
    w = train["sample_weight"].values.copy() if "sample_weight" in train.columns \
        else np.ones(len(train))
    if "is_nonspecific" in train.columns:
        w[train["is_nonspecific"].values] *= 0.3
    if "cliff_sample_weight" in train.columns:
        w *= train["cliff_sample_weight"].values
    w = w / w.mean()
    return w.astype(np.float32)


def pca_compress(X_train: np.ndarray, X_test: np.ndarray, n_components: int = 200):
    """Fit PCA on train, transform both. Clips n_components to actual rank."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    n = min(n_components, X_tr.shape[1], X_tr.shape[0])
    pca = PCA(n_components=n, random_state=42)
    return pca.fit_transform(X_tr).astype(np.float32), pca.transform(X_te).astype(np.float32)


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

    logger.info(f"Train: {len(train)} | Test: {len(test)}")

    # ------------------------------------------------------------------ #
    # 2. Foundation model embeddings (CheMeleon + ChemBERTa)
    #    Run before Chemprop to avoid any MPS/OpenMP interaction issues.
    # ------------------------------------------------------------------ #
    from src.models.foundation_embeddings import CheMeleonEmbedder, ChemBERTaEmbedder

    logger.info("\n=== Computing CheMeleon embeddings ===")
    chemeleon = CheMeleonEmbedder(device="cpu", batch_size=256, cache_dir=EMBED_CACHE)
    cm_train = chemeleon.transform(train_smiles)
    cm_test = chemeleon.transform(test_smiles)
    logger.info(f"CheMeleon: train={cm_train.shape}, test={cm_test.shape}")

    # Release CheMeleon model before loading ChemBERTa
    del chemeleon

    logger.info("\n=== Computing ChemBERTa embeddings ===")
    chemberta = ChemBERTaEmbedder(device="cpu", batch_size=64, cache_dir=EMBED_CACHE)
    cb_train = chemberta.transform(train_smiles)
    cb_test = chemberta.transform(test_smiles)
    logger.info(f"ChemBERTa: train={cb_train.shape}, test={cb_test.shape}")
    del chemberta

    # PCA-compress each foundation model block independently (following competitor approach)
    logger.info("\n=== PCA-compressing foundation embeddings ===")
    cm_train_pca, cm_test_pca = pca_compress(cm_train, cm_test, n_components=200)
    cb_train_pca, cb_test_pca = pca_compress(cb_train, cb_test, n_components=200)
    logger.info(f"CheMeleon PCA: {cm_train_pca.shape} | ChemBERTa PCA: {cb_train_pca.shape}")

    # ------------------------------------------------------------------ #
    # 3. ECFP4 fingerprints for kNN + traditional GBM track
    # ------------------------------------------------------------------ #
    from src.features.feature_engineering import ecfp4, FeaturePipeline
    from src.evaluation.validate import ScaffoldKFold, rae

    logger.info("\n=== Computing ECFP4 fingerprints ===")
    fps_train = ecfp4(train_smiles)
    fps_test = ecfp4(test_smiles)

    splitter = ScaffoldKFold(n_splits=5)
    folds = list(splitter.split(train_smiles))

    # ------------------------------------------------------------------ #
    # 4. Traditional GBM feature matrix (ECFP4 + ECFP6 + RDKit)
    # ------------------------------------------------------------------ #
    logger.info("\n=== Building traditional GBM feature matrix ===")
    pipeline_trad = FeaturePipeline(include_mordred=False, include_ecfp6=True, include_fcfp4=False)
    X_trad_train = pipeline_trad.fit_transform(train_smiles)
    X_trad_test = pipeline_trad.transform(test_smiles)
    logger.info(f"Traditional features: train={X_trad_train.shape}, test={X_trad_test.shape}")

    # ------------------------------------------------------------------ #
    # 5. Foundation GBM feature matrix (CheMeleon_PCA + ChemBERTa_PCA + RDKit)
    # ------------------------------------------------------------------ #
    from src.features.feature_engineering import rdkit_descriptors

    logger.info("\n=== Building foundation GBM feature matrix ===")
    rdkit_tr = rdkit_descriptors(train_smiles)
    rdkit_te = rdkit_descriptors(test_smiles)
    rdkit_medians = rdkit_tr.median()
    rdkit_tr_arr = rdkit_tr.fillna(rdkit_medians).values.astype(np.float32)
    rdkit_te_arr = rdkit_te.fillna(rdkit_medians).values.astype(np.float32)

    X_found_train = np.hstack([cm_train_pca, cb_train_pca, rdkit_tr_arr])
    X_found_test = np.hstack([cm_test_pca, cb_test_pca, rdkit_te_arr])
    logger.info(f"Foundation features: train={X_found_train.shape}, test={X_found_test.shape}")

    # ------------------------------------------------------------------ #
    # 6. Chemprop CV (before any OpenMP-dependent code)
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

        m = ChempropModel(epochs=80, hidden_size=300, depth=3, ffn_num_layers=3,
                          dropout=0.1, batch_size=64, lr=1e-3, device="cpu",
                          snapshot_epochs=5, extra_features=False, seed=42 + fold)
        m.fit(sm_tr, y_tr, sample_weight=sw_tr)
        chemprop_oof[va_idx] = m.predict(sm_va)
        r = rae(y_va, chemprop_oof[va_idx], y_train_mean=fold_mean)
        chemprop_fold_raes.append(r)
        logger.info(f"Fold {fold}: Chemprop RAE={r:.4f}")

    logger.info(f"Chemprop CV RAE: {np.mean(chemprop_fold_raes):.4f} ± {np.std(chemprop_fold_raes):.4f}")

    # Full Chemprop training
    logger.info("\n=== Full Chemprop training ===")
    chemprop_full = ChempropModel(epochs=100, hidden_size=300, depth=3, ffn_num_layers=3,
                                  dropout=0.1, batch_size=64, lr=1e-3, device="cpu",
                                  snapshot_epochs=5, extra_features=False, seed=42)
    chemprop_full.fit(train_smiles, train_y, sample_weight=sample_weights)
    chemprop_test_preds = chemprop_full.predict(test_smiles)
    import gc; del chemprop_full; gc.collect()

    # ------------------------------------------------------------------ #
    # 7. GBM + kNN CV (traditional + foundation tracks)
    # ------------------------------------------------------------------ #
    from src.models.local_models import TanimotoKNN
    from src.models.gbm_models import LGBMWrapper, XGBWrapper, RFWrapper

    model_names = ["knn", "lgbm_trad", "lgbm_found", "xgb_found", "rf_found"]
    fold_raes = {m: [] for m in model_names}
    fold_raes["chemprop"] = chemprop_fold_raes
    fold_preds = {m: np.zeros(len(train)) for m in model_names}
    fold_preds["chemprop"] = chemprop_oof

    logger.info("\n=== GBM + kNN Scaffold 5-fold CV ===")
    for fold, (tr_idx, va_idx) in enumerate(folds):
        logger.info(f"--- Fold {fold} ---")
        fps_tr, fps_va = fps_train[tr_idx], fps_train[va_idx]
        Xtr_t, Xva_t = X_trad_train[tr_idx], X_trad_train[va_idx]
        Xtr_f, Xva_f = X_found_train[tr_idx], X_found_train[va_idx]
        y_tr, y_va = train_y[tr_idx], train_y[va_idx]
        sw_tr = sample_weights[tr_idx]
        fold_mean = float(y_tr.mean())

        n_es = max(50, int(0.1 * len(tr_idx)))
        sw_fit = sw_tr[:-n_es]

        # kNN (Tanimoto)
        knn = TanimotoKNN(k=5)
        knn.fit(fps_tr, y_tr)
        fold_preds["knn"][va_idx] = knn.predict(fps_va)

        # LGBM traditional
        lgbm_t = LGBMWrapper({"n_jobs": 1})
        lgbm_t.fit(Xtr_t[:-n_es], y_tr[:-n_es], X_val=Xtr_t[-n_es:], y_val=y_tr[-n_es:], sample_weight=sw_fit)
        fold_preds["lgbm_trad"][va_idx] = lgbm_t.predict(Xva_t)

        # LGBM foundation
        lgbm_f = LGBMWrapper({"n_jobs": 1})
        lgbm_f.fit(Xtr_f[:-n_es], y_tr[:-n_es], X_val=Xtr_f[-n_es:], y_val=y_tr[-n_es:], sample_weight=sw_fit)
        fold_preds["lgbm_found"][va_idx] = lgbm_f.predict(Xva_f)

        # XGB foundation
        xgb_f = XGBWrapper({"n_jobs": 1})
        xgb_f.fit(Xtr_f[:-n_es], y_tr[:-n_es], X_val=Xtr_f[-n_es:], y_val=y_tr[-n_es:], sample_weight=sw_fit)
        fold_preds["xgb_found"][va_idx] = xgb_f.predict(Xva_f)

        # RF foundation
        rf_f = RFWrapper({"n_jobs": 1})
        rf_f.fit(Xtr_f, y_tr, sample_weight=sw_tr)
        fold_preds["rf_found"][va_idx] = rf_f.predict(Xva_f)

        for m in model_names:
            r = rae(y_va, fold_preds[m][va_idx], y_train_mean=fold_mean)
            fold_raes[m].append(r)

        logger.info(
            f"Fold {fold}: kNN={fold_raes['knn'][-1]:.4f}, "
            f"LGBM_trad={fold_raes['lgbm_trad'][-1]:.4f}, "
            f"LGBM_found={fold_raes['lgbm_found'][-1]:.4f}, "
            f"XGB_found={fold_raes['xgb_found'][-1]:.4f}, "
            f"RF_found={fold_raes['rf_found'][-1]:.4f}"
        )

    all_model_names = ["chemprop"] + model_names
    logger.info("\nCV Summary:")
    mean_raes = {}
    for m in all_model_names:
        mean_raes[m] = float(np.mean(fold_raes[m]))
        logger.info(f"  {m:12s}: {mean_raes[m]:.4f} ± {np.std(fold_raes[m]):.4f}")

    # Inverse-RAE ensemble weights
    weights = {m: 1.0 / mean_raes[m] for m in all_model_names}
    total_w = sum(weights.values())
    ens_preds_cv = sum(weights[m] * fold_preds[m] for m in all_model_names) / total_w
    ens_rae = rae(train_y, ens_preds_cv, y_train_mean=train_mean)
    logger.info(f"\n  ensemble (inv-RAE): {ens_rae:.4f}")
    for m in all_model_names:
        logger.info(f"    {m}: weight = {weights[m]/total_w:.3f}")

    # ------------------------------------------------------------------ #
    # 8. Train full models on complete training set
    # ------------------------------------------------------------------ #
    logger.info("\n=== Training full models ===")

    n_es_full = max(100, int(0.1 * len(X_trad_train)))
    sw_fit_full = sample_weights[:-n_es_full]

    knn_full = TanimotoKNN(k=5)
    knn_full.fit(fps_train, train_y)

    lgbm_trad_full = LGBMWrapper({"n_jobs": 1})
    lgbm_trad_full.fit(X_trad_train[:-n_es_full], train_y[:-n_es_full],
                       X_val=X_trad_train[-n_es_full:], y_val=train_y[-n_es_full:],
                       sample_weight=sw_fit_full)

    lgbm_found_full = LGBMWrapper({"n_jobs": 1})
    lgbm_found_full.fit(X_found_train[:-n_es_full], train_y[:-n_es_full],
                        X_val=X_found_train[-n_es_full:], y_val=train_y[-n_es_full:],
                        sample_weight=sw_fit_full)

    xgb_found_full = XGBWrapper({"n_jobs": 1})
    xgb_found_full.fit(X_found_train[:-n_es_full], train_y[:-n_es_full],
                       X_val=X_found_train[-n_es_full:], y_val=train_y[-n_es_full:],
                       sample_weight=sw_fit_full)

    rf_found_full = RFWrapper({"n_jobs": 1})
    rf_found_full.fit(X_found_train, train_y, sample_weight=sample_weights)

    # ------------------------------------------------------------------ #
    # 9. Test predictions + ensemble
    # ------------------------------------------------------------------ #
    logger.info("\n=== Generating test predictions ===")

    test_preds = {
        "chemprop":   chemprop_test_preds,
        "knn":        knn_full.predict(fps_test),
        "lgbm_trad":  lgbm_trad_full.predict(X_trad_test),
        "lgbm_found": lgbm_found_full.predict(X_found_test),
        "xgb_found":  xgb_found_full.predict(X_found_test),
        "rf_found":   rf_found_full.predict(X_found_test),
    }

    for m in all_model_names:
        logger.info(f"  {m}: mean={test_preds[m].mean():.3f}, std={test_preds[m].std():.3f}")

    ensemble_test = sum(weights[m] * test_preds[m] for m in all_model_names) / total_w
    ensemble_test = np.clip(ensemble_test, 1.5, 8.0)
    logger.info(f"Ensemble: mean={ensemble_test.mean():.3f}, std={ensemble_test.std():.3f}")

    # ------------------------------------------------------------------ #
    # 10. Submit
    # ------------------------------------------------------------------ #
    from src.ensemble.stack_and_submit import make_submission, validate_submission

    submission_path = make_submission(
        predictions=ensemble_test,
        test_df=test,
        description=(
            f"Sub4_Chemprop+kNN+LGBM_trad+LGBM_found+XGB_found+RF_found_invRAE | "
            f"CV_RAE={ens_rae:.4f} | "
            + " ".join(f"{m}={mean_raes[m]:.4f}" for m in all_model_names)
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
