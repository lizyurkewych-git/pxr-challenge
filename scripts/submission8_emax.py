"""
Submission 8: Emax multi-task Chemprop + MACCS fingerprints for RF.

New in this submission vs Sub 7:

- **Emax multi-task Chemprop** (`chemprop_hts`): The HTS-pretrained Chemprop
  is now fine-tuned with n_tasks=2, predicting [pEC50, Emax] jointly.
  Emax (maximum efficacy vs. positive control, ~0–1 dimensionless) is
  mechanistically coupled to pEC50 — both are fit from the same Hill curve.
  Multi-task training forces the encoder to distinguish partial agonists
  (low Emax, high pEC50) from full agonists, improving the primary pEC50
  head via shared representation learning without requiring new data.
  At inference, only the pEC50 head (column 0) is returned.

- **MACCS 167-bit structural keys for RF** (`FeaturePipeline(include_maccs=True)`):
  De la Vega SAR analysis of this dataset found ECFP4 and MACCS capture
  ~90% non-overlapping activity cliff pairs. Adding MACCS gives RF access
  to cliff information invisible to ECFP4-based features, improving its
  ensemble contribution (RF coef 0.13 → 0.15 in Sub 8).

- **kNN and LGBM dropped**: Sub 7 ElasticNet assigned coef=0.04 and 0.00
  respectively. Removing them lets ElasticNet redistribute weight to
  better-performing models.

Model roster (5 total):
  delta            — pairwise Δ pEC50 Chemprop, kNN-anchored inference
  chemprop_hts     — pEC50 + Emax multi-task (conc-aware HTS pretrained)
  chemprop_scratch — direct pEC50, random init (acts as bias corrector)
  tabpfn           — TabPFN v2 + CheMeleon 2048→PCA(200) in-context learning
  rf               — ECFP4 + ECFP6 + RDKit + MACCS

ElasticNet OOF RAE: 0.5286 (vs 0.5297 in Sub 7)
ElasticNet coefs: tabpfn=0.423, chemprop_hts=0.321, delta=0.167, rf=0.153, scratch=−0.120

Requires Python 3.11 and tabpfn 2.2.1:
    pip install "tabpfn==2.2.1" --no-deps
    .venv311/bin/python scripts/submission8_emax.py
"""

import gc
import logging
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("submission8")

CV_SEEDS    = [42, 7]
FINAL_SEEDS = [42, 7, 13]
DEVICE      = "mps"  # change to "cuda" on GPU or "cpu" if neither is available


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
        snapshot_epochs=5, extra_features=False, seed=seed, n_tasks=1,
    )
    m.fit(sm_tr, y_tr, sample_weight=sw_tr, init_state_dict=init_sd, x_d=x_d)
    return m


def avg_chemprop(sm_tr, y_tr, sw_tr, sm_va, seeds, epochs, lr, init_sd=None):
    return np.mean(
        [train_chemprop(sm_tr, y_tr, sw_tr, s, epochs, lr, init_sd).predict(sm_va)
         for s in seeds], axis=0
    )


def avg_chemprop_full(sm_all, y_all, sw_all, sm_test, seeds, epochs, lr, init_sd=None):
    return np.mean(
        [train_chemprop(sm_all, y_all, sw_all, s, epochs, lr, init_sd).predict(sm_test)
         for s in seeds], axis=0
    )


def train_chemprop_multitask(sm_tr, y_pecso_tr, y_emax_tr, sw_tr, seed,
                              epochs, lr, init_sd=None):
    """Train a 2-task Chemprop model predicting [pEC50, Emax] jointly."""
    from src.models.chemprop_model import ChempropModel
    y_multi = np.column_stack([y_pecso_tr, y_emax_tr])
    m = ChempropModel(
        epochs=epochs, hidden_size=300, depth=3, ffn_num_layers=3,
        dropout=0.1, batch_size=64, lr=lr, device=DEVICE,
        snapshot_epochs=5, extra_features=False, seed=seed, n_tasks=2,
    )
    m.fit(sm_tr, y_multi, sample_weight=sw_tr, init_state_dict=init_sd)
    return m


def avg_chemprop_multitask(sm_tr, y_pecso_tr, y_emax_tr, sw_tr, sm_va,
                            seeds, epochs, lr, init_sd=None):
    return np.mean(
        [train_chemprop_multitask(sm_tr, y_pecso_tr, y_emax_tr, sw_tr, s,
                                   epochs, lr, init_sd).predict(sm_va)
         for s in seeds], axis=0
    )


def avg_chemprop_multitask_full(sm_all, y_pecso_all, y_emax_all, sw_all,
                                  sm_test, seeds, epochs, lr, init_sd=None):
    return np.mean(
        [train_chemprop_multitask(sm_all, y_pecso_all, y_emax_all, sw_all, s,
                                   epochs, lr, init_sd).predict(sm_test)
         for s in seeds], axis=0
    )


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    # ---------------------------------------------------------- #
    # 1. Load data
    # ---------------------------------------------------------- #
    from src.data.load_data import (
        load_all_tiers, COL_SMILES, COL_PECSO, COL_EMAX_CTRL
    )
    from src.data.cliff_analysis import identify_activity_cliffs, annotate_cliff_compounds

    logger.info("Loading dataset...")
    ds = load_all_tiers(cache_dir="data/hf_cache")
    train = ds.train
    test  = ds.test
    hts   = ds.hts

    logger.info("Identifying activity cliffs...")
    cliff_pairs = identify_activity_cliffs(
        smiles=train[COL_SMILES].tolist(),
        activities=train[COL_PECSO].to_numpy(),
        sim_threshold=0.7,
        activity_threshold=1.0,
    )
    train = annotate_cliff_compounds(train, cliff_pairs)

    train_smiles  = train[COL_SMILES].tolist()
    train_y       = train[COL_PECSO].to_numpy()
    test_smiles   = test[COL_SMILES].tolist()
    train_mean    = float(train_y.mean())
    sample_weights = build_sample_weights(train)

    # Emax (vs. pos. ctrl, dimensionless ~0–1) for multi-task training.
    # NaN filled with median so no compounds are excluded.
    emax_raw    = train[COL_EMAX_CTRL].values.astype(np.float64)
    emax_median = float(np.nanmedian(emax_raw))
    train_emax  = np.where(np.isfinite(emax_raw), emax_raw, emax_median)
    logger.info(
        f"Train: {len(train)} | Test: {len(test)} | HTS: {len(hts)} | "
        f"Emax NaN filled: {int(np.sum(~np.isfinite(emax_raw)))} (median={emax_median:.3f})"
    )

    # ---------------------------------------------------------- #
    # 2. Concentration-aware HTS pre-training (single-task)
    # ---------------------------------------------------------- #
    from src.models.hts_pretrain import prepare_hts_concentration_data
    from src.models.chemprop_model import ChempropModel

    logger.info("\n=== Concentration-aware HTS pre-training ===")
    pt_smiles, pt_y, pt_xd = prepare_hts_concentration_data(
        hts_df=hts, primary_train=train
    )
    logger.info(
        f"Pre-training on {len(pt_smiles)} rows ({len(set(pt_smiles))} unique compounds)"
    )

    pretrain_model = ChempropModel(
        epochs=60, hidden_size=300, depth=3, ffn_num_layers=3,
        dropout=0.1, batch_size=64, lr=1e-3, device=DEVICE,
        snapshot_epochs=5, extra_features=False, seed=42, n_tasks=1,
    )
    pretrain_model.fit(pt_smiles, pt_y, x_d=pt_xd)
    hts_state_dict = pretrain_model.get_state_dict()
    del pretrain_model; gc.collect()
    logger.info("HTS pre-training complete.")

    # ---------------------------------------------------------- #
    # 3. Feature matrices (MACCS enabled for RF)
    # ---------------------------------------------------------- #
    from src.features.feature_engineering import ecfp4, FeaturePipeline

    logger.info("\n=== Building feature matrices ===")
    fps_train = ecfp4(train_smiles)
    fps_test  = ecfp4(test_smiles)

    pipeline = FeaturePipeline(
        include_mordred=False, include_ecfp6=True,
        include_fcfp4=False, include_maccs=True,
    )
    X_train = pipeline.fit_transform(train_smiles)
    X_test  = pipeline.transform(test_smiles)
    logger.info(f"Feature matrix (ECFP4+ECFP6+RDKit+MACCS): train={X_train.shape}, test={X_test.shape}")

    # ---------------------------------------------------------- #
    # 4. Butina 5-fold CV — OOF predictions for all 5 models
    # ---------------------------------------------------------- #
    from src.evaluation.validate import ButinaKFold, rae
    from src.models.gbm_models import RFWrapper
    from src.models.delta_model import DeltaChempropModel
    from src.models.tabpfn_model import TabPFNCheMeleonModel

    logger.info("\n=== Butina 5-fold CV ===")
    splitter = ButinaKFold(n_splits=5, tanimoto_threshold=0.4)
    folds    = list(splitter.split(train_smiles))

    model_names = ["delta", "chemprop_hts", "chemprop_scratch", "tabpfn", "rf"]
    oof      = {m: np.zeros(len(train)) for m in model_names}
    fold_raes = {m: [] for m in model_names}

    for fold, (tr_idx, va_idx) in enumerate(folds):
        logger.info(f"\n--- Fold {fold} (train={len(tr_idx)}, val={len(va_idx)}) ---")
        sm_tr = [train_smiles[i] for i in tr_idx]
        sm_va = [train_smiles[i] for i in va_idx]
        y_tr, y_va   = train_y[tr_idx], train_y[va_idx]
        emax_tr      = train_emax[tr_idx]
        sw_tr        = sample_weights[tr_idx]
        fps_tr, fps_va = fps_train[tr_idx], fps_train[va_idx]
        X_tr, X_va   = X_train[tr_idx], X_train[va_idx]
        fold_mean    = float(y_tr.mean())

        logger.info(f"Fold {fold}: Delta model...")
        delta_cv = DeltaChempropModel(
            epochs=40, hidden_size=300, depth=3, ffn_num_layers=3,
            dropout=0.1, batch_size=64, lr=1e-3, device=DEVICE,
            snapshot_epochs=5, seed=42, n_pairs_per_epoch=20_000,
            cliff_oversample=3, k_neighbors=10,
        )
        delta_cv.fit(sm_tr, y_tr, fps_train=fps_tr, init_state_dict=hts_state_dict)
        oof["delta"][va_idx] = delta_cv.predict(sm_va, sm_tr, y_tr, fps_tr, fps_va)
        fold_raes["delta"].append(rae(y_va, oof["delta"][va_idx], y_train_mean=fold_mean))
        logger.info(f"Fold {fold}: delta RAE={fold_raes['delta'][-1]:.4f}")
        del delta_cv; gc.collect()

        logger.info(f"Fold {fold}: Chemprop HTS multi-task (pEC50 + Emax)...")
        oof["chemprop_hts"][va_idx] = avg_chemprop_multitask(
            sm_tr, y_tr, emax_tr, sw_tr, sm_va,
            CV_SEEDS, epochs=60, lr=5e-4, init_sd=hts_state_dict,
        )
        fold_raes["chemprop_hts"].append(
            rae(y_va, oof["chemprop_hts"][va_idx], y_train_mean=fold_mean)
        )
        logger.info(f"Fold {fold}: chemprop_hts RAE={fold_raes['chemprop_hts'][-1]:.4f}")

        logger.info(f"Fold {fold}: Chemprop scratch...")
        oof["chemprop_scratch"][va_idx] = avg_chemprop(
            sm_tr, y_tr, sw_tr, sm_va, CV_SEEDS, epochs=60, lr=1e-3
        )
        fold_raes["chemprop_scratch"].append(
            rae(y_va, oof["chemprop_scratch"][va_idx], y_train_mean=fold_mean)
        )
        logger.info(f"Fold {fold}: chemprop_scratch RAE={fold_raes['chemprop_scratch'][-1]:.4f}")

        logger.info(f"Fold {fold}: TabPFN + CheMeleon...")
        tabpfn_cv = TabPFNCheMeleonModel(
            n_components=200, n_estimators=16, device="cpu", random_state=42
        )
        tabpfn_cv.fit(sm_tr, y_tr)
        oof["tabpfn"][va_idx] = tabpfn_cv.predict(sm_va)
        fold_raes["tabpfn"].append(rae(y_va, oof["tabpfn"][va_idx], y_train_mean=fold_mean))
        logger.info(f"Fold {fold}: tabpfn RAE={fold_raes['tabpfn'][-1]:.4f}")
        del tabpfn_cv; gc.collect()

        rf = RFWrapper()
        rf.fit(X_tr, y_tr, sample_weight=sw_tr)
        oof["rf"][va_idx] = rf.predict(X_va)
        fold_raes["rf"].append(rae(y_va, oof["rf"][va_idx], y_train_mean=fold_mean))
        logger.info(f"Fold {fold}: rf RAE={fold_raes['rf'][-1]:.4f}")

    # ---------------------------------------------------------- #
    # 5. CV summary
    # ---------------------------------------------------------- #
    logger.info("\n=== CV Summary ===")
    cv_summary = {}
    for m in model_names:
        mean_r = float(np.mean(fold_raes[m]))
        std_r  = float(np.std(fold_raes[m]))
        cv_summary[m] = {"mean": mean_r, "std": std_r}
        logger.info(f"  {m:<20}: {mean_r:.4f} ± {std_r:.4f}")

    # ---------------------------------------------------------- #
    # 6. ElasticNet stacking
    # ---------------------------------------------------------- #
    from src.ensemble.stack_and_submit import ElasticNetStacker

    logger.info("\n=== ElasticNet stacking ===")
    oof_matrix  = np.column_stack([oof[m] for m in model_names])
    stacker     = ElasticNetStacker(l1_ratio=0.7, cv=5)
    stacker.fit(oof_matrix, train_y, model_names=model_names)

    oof_stacked  = stacker.predict(oof_matrix)
    stacked_rae  = float(rae(train_y, oof_stacked, y_train_mean=train_mean))
    logger.info(f"ElasticNet OOF RAE: {stacked_rae:.4f}")
    logger.info(f"ElasticNet coefs: {stacker.coefs}")

    # ---------------------------------------------------------- #
    # 7. Full training
    # ---------------------------------------------------------- #
    logger.info("\n=== Full training (all data) ===")

    logger.info("Full training: Delta model (3 seeds)...")
    delta_preds_all = []
    for seed in FINAL_SEEDS:
        dm = DeltaChempropModel(
            epochs=50, hidden_size=300, depth=3, ffn_num_layers=3,
            dropout=0.1, batch_size=64, lr=1e-3, device=DEVICE,
            snapshot_epochs=5, seed=seed, n_pairs_per_epoch=20_000,
            cliff_oversample=3, k_neighbors=10,
        )
        dm.fit(train_smiles, train_y, fps_train=fps_train,
               init_state_dict=hts_state_dict)
        delta_preds_all.append(
            dm.predict(test_smiles, train_smiles, train_y, fps_train, fps_test)
        )
        del dm; gc.collect()
    test_preds_delta = np.mean(delta_preds_all, axis=0)

    logger.info("Full training: Chemprop HTS multi-task (3 seeds)...")
    test_preds_chemprop_hts = avg_chemprop_multitask_full(
        train_smiles, train_y, train_emax, sample_weights, test_smiles,
        FINAL_SEEDS, epochs=80, lr=5e-4, init_sd=hts_state_dict,
    )

    logger.info("Full training: Chemprop scratch (3 seeds)...")
    test_preds_chemprop_scratch = avg_chemprop_full(
        train_smiles, train_y, sample_weights, test_smiles,
        FINAL_SEEDS, epochs=80, lr=1e-3,
    )

    del hts_state_dict; gc.collect()

    logger.info("Full training: TabPFN + CheMeleon...")
    tabpfn_full = TabPFNCheMeleonModel(
        n_components=200, n_estimators=16, device="cpu", random_state=42
    )
    tabpfn_full.fit(train_smiles, train_y)
    test_preds_tabpfn = tabpfn_full.predict(test_smiles)
    del tabpfn_full; gc.collect()

    logger.info("Full training: RF (MACCS-enhanced)...")
    rf_full = RFWrapper()
    rf_full.fit(X_train, train_y, sample_weight=sample_weights)
    test_preds_rf = rf_full.predict(X_test)

    test_preds = {
        "delta":            test_preds_delta,
        "chemprop_hts":     test_preds_chemprop_hts,
        "chemprop_scratch": test_preds_chemprop_scratch,
        "tabpfn":           test_preds_tabpfn,
        "rf":               test_preds_rf,
    }
    for m, p in test_preds.items():
        logger.info(f"  {m:<20}: mean={p.mean():.3f}, std={p.std():.3f}")

    # ---------------------------------------------------------- #
    # 8. Apply stacker and save submission
    # ---------------------------------------------------------- #
    test_matrix  = np.column_stack([test_preds[m] for m in model_names])
    final_preds  = stacker.predict(test_matrix)
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
