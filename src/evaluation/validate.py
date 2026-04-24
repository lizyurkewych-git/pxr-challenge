"""
Evaluation utilities for the PXR activity prediction challenge.

Implements:
  - Relative Absolute Error (RAE) — the challenge primary metric
  - Bootstrap confidence intervals for RAE
  - Scaffold-stratified 5-fold cross-validation splitter
  - Per-subgroup reporting (cliff vs. non-cliff compounds)

The RAE is defined as:
    RAE = MAE(model) / MAE(mean_baseline)
  where mean_baseline always predicts the training set mean.
  RAE < 1.0 beats the trivial baseline; lower is better.
"""

from __future__ import annotations

import logging
from typing import Iterator, Optional, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RAE metric
# ---------------------------------------------------------------------------

def rae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train_mean: Optional[float] = None,
    sample_weights: Optional[np.ndarray] = None,
) -> float:
    """Relative Absolute Error.

    Parameters
    ----------
    y_true        : ground-truth pEC50 values
    y_pred        : model predictions
    y_train_mean  : if None, use mean(y_true) as the baseline (appropriate for CV)
                    in final evaluation, pass the training set mean
    sample_weights: optional per-compound weights (inverse variance)

    Returns
    -------
    RAE ∈ [0, ∞); values < 1.0 beat the mean baseline
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if y_train_mean is None:
        baseline = np.full_like(y_true, fill_value=y_true.mean())
    else:
        baseline = np.full_like(y_true, fill_value=y_train_mean)

    if sample_weights is not None:
        w = np.asarray(sample_weights, dtype=np.float64)
        mae_model = np.average(np.abs(y_true - y_pred), weights=w)
        mae_baseline = np.average(np.abs(y_true - baseline), weights=w)
    else:
        mae_model = np.mean(np.abs(y_true - y_pred))
        mae_baseline = np.mean(np.abs(y_true - baseline))

    if mae_baseline < 1e-10:
        logger.warning("Baseline MAE is near zero — RAE is undefined. Returning NaN.")
        return float("nan")
    return float(mae_model / mae_baseline)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from scipy.stats import pearsonr
    r, _ = pearsonr(y_true, y_pred)
    return float(r)


def spearman_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from scipy.stats import spearmanr
    r, _ = spearmanr(y_true, y_pred)
    return float(r)


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_rae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train_mean: Optional[float] = None,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """Compute bootstrap CI for RAE.

    Returns
    -------
    dict with keys: rae, ci_lower, ci_upper, std
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    bootstrap_raes = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        r = rae(y_true[idx], y_pred[idx], y_train_mean=y_train_mean)
        bootstrap_raes.append(r)

    bootstrap_raes = np.array(bootstrap_raes)
    alpha = 1 - ci
    ci_lower = float(np.percentile(bootstrap_raes, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_raes, 100 * (1 - alpha / 2)))
    point_rae = rae(y_true, y_pred, y_train_mean=y_train_mean)

    return {
        "rae": point_rae,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "std": float(np.std(bootstrap_raes)),
    }


# ---------------------------------------------------------------------------
# Full metrics report
# ---------------------------------------------------------------------------

def full_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train_mean: Optional[float] = None,
    label: str = "",
    n_bootstrap: int = 1000,
) -> dict:
    """Compute and log all evaluation metrics."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics = {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "pearson_r": pearson_r(y_true, y_pred),
        "spearman_r": spearman_r(y_true, y_pred),
    }
    bstrap = bootstrap_rae(y_true, y_pred, y_train_mean=y_train_mean, n_bootstrap=n_bootstrap)
    metrics.update(bstrap)

    prefix = f"[{label}] " if label else ""
    logger.info(
        f"{prefix}RAE = {metrics['rae']:.4f} "
        f"({metrics['ci_lower']:.4f}–{metrics['ci_upper']:.4f} 95% CI) | "
        f"MAE = {metrics['mae']:.4f} | "
        f"RMSE = {metrics['rmse']:.4f} | "
        f"Pearson r = {metrics['pearson_r']:.4f} | "
        f"Spearman r = {metrics['spearman_r']:.4f}"
    )
    return metrics


# ---------------------------------------------------------------------------
# Murcko scaffold computation
# ---------------------------------------------------------------------------

def get_murcko_scaffold(smi: str, generic: bool = False) -> str:
    """Return Murcko scaffold SMILES. Falls back to full SMILES if parsing fails."""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return smi
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if generic:
            scaffold = MurckoScaffold.MakeScaffoldGeneric(scaffold)
        return Chem.MolToSmiles(scaffold)
    except Exception:
        return smi


def assign_scaffolds(smiles: Sequence[str]) -> np.ndarray:
    """Return array of Murcko scaffold SMILES strings (one per compound)."""
    return np.array([get_murcko_scaffold(smi) for smi in smiles])


# ---------------------------------------------------------------------------
# Scaffold-stratified K-Fold splitter
# ---------------------------------------------------------------------------

class ScaffoldKFold:
    """K-Fold cross-validation that holds out entire scaffold families per fold.

    This mimics the train-to-analog-test generalization challenge:
    the model must extrapolate within scaffold families, not just interpolate.

    Usage:
        splitter = ScaffoldKFold(n_splits=5)
        for fold, (train_idx, val_idx) in enumerate(splitter.split(smiles)):
            ...
    """

    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, smiles: Sequence[str]) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield (train_indices, val_indices) for each fold."""
        scaffolds = assign_scaffolds(smiles)
        unique_scaffolds = np.unique(scaffolds)

        rng = np.random.default_rng(self.random_state)
        if self.shuffle:
            unique_scaffolds = rng.permutation(unique_scaffolds)

        # Assign scaffolds round-robin to folds
        scaffold_to_fold = {s: i % self.n_splits for i, s in enumerate(unique_scaffolds)}
        fold_ids = np.array([scaffold_to_fold[s] for s in scaffolds])

        for fold in range(self.n_splits):
            val_idx = np.where(fold_ids == fold)[0]
            train_idx = np.where(fold_ids != fold)[0]
            yield train_idx, val_idx

    def get_n_splits(self) -> int:
        return self.n_splits


# ---------------------------------------------------------------------------
# Cross-validation runner
# ---------------------------------------------------------------------------

def cross_validate_model(
    model,
    smiles: Sequence[str],
    y: np.ndarray,
    feature_pipeline,
    n_splits: int = 5,
    cliff_mask: Optional[np.ndarray] = None,
) -> dict:
    """Run scaffold-stratified CV and return aggregated metrics.

    Parameters
    ----------
    model           : object with fit(X, y) and predict(X) methods
    smiles          : list of training SMILES
    y               : target pEC50 values
    feature_pipeline: FeaturePipeline instance (fit per fold)
    cliff_mask      : bool array marking activity cliff compounds

    Returns
    -------
    dict with per-fold and aggregate metrics
    """
    splitter = ScaffoldKFold(n_splits=n_splits)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(splitter.split(smiles)):
        smiles_arr = np.array(smiles)
        y_arr = np.array(y)

        train_smiles = smiles_arr[train_idx].tolist()
        val_smiles = smiles_arr[val_idx].tolist()
        y_train = y_arr[train_idx]
        y_val = y_arr[val_idx]

        # Fit feature pipeline per fold to avoid data leakage
        X_train = feature_pipeline.fit_transform(train_smiles)
        X_val = feature_pipeline.transform(val_smiles)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        y_train_mean = float(y_train.mean())
        fold_rae = rae(y_val, y_pred, y_train_mean=y_train_mean)
        fold_mae = mae(y_val, y_pred)

        result = {
            "fold": fold,
            "n_val": len(val_idx),
            "rae": fold_rae,
            "mae": fold_mae,
            "train_mean": y_train_mean,
        }

        # Cliff vs. non-cliff breakdown
        if cliff_mask is not None:
            cliff_val = cliff_mask[val_idx]
            if cliff_val.sum() > 0:
                result["rae_cliff"] = rae(
                    y_val[cliff_val], y_pred[cliff_val], y_train_mean=y_train_mean
                )
            if (~cliff_val).sum() > 0:
                result["rae_noncliff"] = rae(
                    y_val[~cliff_val], y_pred[~cliff_val], y_train_mean=y_train_mean
                )

        fold_results.append(result)
        logger.info(
            f"Fold {fold}: RAE={fold_rae:.4f}, MAE={fold_mae:.4f}, n_val={len(val_idx)}"
        )

    df = pd.DataFrame(fold_results)
    mean_rae = df["rae"].mean()
    std_rae = df["rae"].std()
    logger.info(f"CV complete: mean RAE = {mean_rae:.4f} ± {std_rae:.4f}")

    return {
        "mean_rae": mean_rae,
        "std_rae": std_rae,
        "fold_results": fold_results,
        "df": df,
    }


# ---------------------------------------------------------------------------
# Local kNN correction (post-hoc residual adjustment)
# ---------------------------------------------------------------------------

def apply_knn_correction(
    y_pred_global: np.ndarray,
    test_fps: np.ndarray,
    train_fps: np.ndarray,
    train_y: np.ndarray,
    train_y_pred_oof: np.ndarray,
    k: int = 5,
    threshold_sim: float = 0.7,
) -> np.ndarray:
    """Apply local kNN residual correction to global model predictions.

    For each test compound:
    1. Find k nearest training neighbors (Tanimoto similarity)
    2. Compute the average out-of-fold residual of those neighbors
    3. Correct: final_pred = global_pred + correction

    Only corrects if nearest neighbor similarity > threshold_sim.

    Parameters
    ----------
    train_y_pred_oof : out-of-fold predictions for training set (from CV)
    """
    from src.features.feature_engineering import tanimoto_matrix

    sim = tanimoto_matrix(test_fps, train_fps)  # (n_test, n_train)
    k = min(k, train_fps.shape[0])
    top_k_idx = np.argsort(-sim, axis=1)[:, :k]
    top_k_sim = sim[np.arange(len(test_fps))[:, None], top_k_idx]
    max_sim = top_k_sim[:, 0]  # max similarity per test compound

    train_residuals = train_y - train_y_pred_oof
    top_k_residuals = train_residuals[top_k_idx]
    top_k_weights = top_k_sim / (top_k_sim.sum(axis=1, keepdims=True) + 1e-10)
    corrections = (top_k_weights * top_k_residuals).sum(axis=1)

    # Only apply correction where confidence is high
    apply_mask = max_sim > threshold_sim
    y_corrected = y_pred_global.copy()
    y_corrected[apply_mask] += corrections[apply_mask]

    logger.info(
        f"kNN correction applied to {apply_mask.sum()} / {len(test_fps)} compounds "
        f"(Tanimoto > {threshold_sim})."
    )
    return y_corrected


# ---------------------------------------------------------------------------
# Butina cluster K-Fold splitter
# ---------------------------------------------------------------------------

class ButinaKFold:
    """K-Fold CV using Butina clustering (Tanimoto-based) for structurally honest splits.

    Clusters compounds by Tanimoto similarity so that structurally related
    compounds are held out together. More realistic than Murcko scaffold folds
    for analog-set test compounds.

    Usage:
        splitter = ButinaKFold(n_splits=5, tanimoto_threshold=0.4)
        for fold, (train_idx, val_idx) in enumerate(splitter.split(smiles)):
            ...
    """

    def __init__(
        self,
        n_splits: int = 5,
        tanimoto_threshold: float = 0.4,
        random_state: int = 42,
    ):
        self.n_splits = n_splits
        self.tanimoto_threshold = tanimoto_threshold
        self.random_state = random_state

    def split(self, smiles: Sequence[str]) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        from rdkit.Chem import DataStructs
        from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
        from rdkit.ML.Cluster import Butina

        n = len(smiles)
        gen = GetMorganGenerator(radius=2, fpSize=2048)
        fps = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles]

        dists = []
        for i in range(1, n):
            sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
            dists.extend(1.0 - s for s in sims)

        cutoff = 1.0 - self.tanimoto_threshold
        clusters = Butina.ClusterData(dists, n, cutoff, isDistData=True)

        cluster_ids = np.zeros(n, dtype=int)
        for cluster_id, cluster in enumerate(clusters):
            for idx in cluster:
                cluster_ids[idx] = cluster_id

        unique_clusters = np.unique(cluster_ids)
        rng = np.random.default_rng(self.random_state)
        unique_clusters = rng.permutation(unique_clusters)
        cluster_to_fold = {int(c): i % self.n_splits for i, c in enumerate(unique_clusters)}
        fold_ids = np.array([cluster_to_fold[int(c)] for c in cluster_ids])

        logger.info(
            f"ButinaKFold: {len(unique_clusters)} clusters → {self.n_splits} folds "
            f"(threshold={self.tanimoto_threshold})"
        )
        for fold in range(self.n_splits):
            val_idx = np.where(fold_ids == fold)[0]
            train_idx = np.where(fold_ids != fold)[0]
            yield train_idx, val_idx

    def get_n_splits(self) -> int:
        return self.n_splits


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    rng = np.random.default_rng(0)
    y_true = rng.uniform(3, 8, 100)
    y_pred = y_true + rng.normal(0, 0.5, 100)
    metrics = full_metrics(y_true, y_pred, label="test")
    print(metrics)
