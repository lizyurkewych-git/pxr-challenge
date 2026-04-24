"""
Ensemble construction and submission pipeline.

Implements:
  - Inverse-RAE weighted average ensemble (Phase 1)
  - Caruana greedy ensemble selection (Phase 2, after Set 1 unblinding)
  - GP uncertainty-gated adaptive weighting
  - Submission CSV generation with validation

Usage:
    from src.ensemble.stack_and_submit import WeightedEnsemble, CaruanaEnsemble, make_submission
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SUBMISSIONS_DIR = Path("submissions")
SUBMISSIONS_LOG = SUBMISSIONS_DIR / "submissions_log.csv"


# ---------------------------------------------------------------------------
# Weighted average ensemble (Phase 1)
# ---------------------------------------------------------------------------

class WeightedEnsemble:
    """Ensemble by inverse-validation-RAE weighting.

    For each model i: w_i = 1 / RAE_i_val
    Final prediction: Σ(w_i * pred_i) / Σ(w_i)
    """

    def __init__(self, clip_range: tuple[float, float] = (1.5, 8.0)):
        self.clip_range = clip_range
        self._weights: Optional[np.ndarray] = None
        self._model_names: list[str] = []

    def fit(self, val_raes: Sequence[float], model_names: Optional[Sequence[str]] = None) -> "WeightedEnsemble":
        """Set model weights from validation RAE scores.

        Parameters
        ----------
        val_raes     : list of RAE scores on validation set (one per model)
        model_names  : optional list of model names for logging
        """
        val_raes = np.array(val_raes, dtype=np.float64)
        if np.any(val_raes <= 0):
            raise ValueError("All validation RAEs must be positive.")

        self._weights = 1.0 / val_raes
        self._weights /= self._weights.sum()  # normalize to sum=1

        if model_names is not None:
            self._model_names = list(model_names)
        else:
            self._model_names = [f"model_{i}" for i in range(len(val_raes))]

        for name, w, r in zip(self._model_names, self._weights, val_raes):
            logger.info(f"  {name}: val_RAE={r:.4f}, weight={w:.4f}")

        return self

    def predict(self, predictions: Sequence[np.ndarray]) -> np.ndarray:
        """Compute weighted average of model predictions.

        Parameters
        ----------
        predictions : list of 1D arrays, one per model

        Returns
        -------
        ensemble prediction, clipped to self.clip_range
        """
        if self._weights is None:
            raise RuntimeError("Call fit() before predict().")
        if len(predictions) != len(self._weights):
            raise ValueError(
                f"Expected {len(self._weights)} prediction arrays, got {len(predictions)}."
            )
        preds = np.stack(predictions, axis=0)  # (n_models, n_compounds)
        ensemble = (self._weights[:, None] * preds).sum(axis=0)
        ensemble = np.clip(ensemble, *self.clip_range)
        return ensemble


# ---------------------------------------------------------------------------
# Caruana greedy ensemble selection (Phase 2)
# ---------------------------------------------------------------------------

class CaruanaEnsemble:
    """Greedy forward model selection (Caruana et al., 2004).

    Iteratively selects the model (with replacement) that most reduces
    RAE on a held-out validation set. Returns a sparse weight vector.

    Reference: R. Caruana et al., "Ensemble Selection from Libraries of Models",
               ICML 2004.
    """

    def __init__(
        self,
        n_iterations: int = 100,
        clip_range: tuple[float, float] = (1.5, 8.0),
    ):
        self.n_iterations = n_iterations
        self.clip_range = clip_range
        self._counts: Optional[np.ndarray] = None
        self._model_names: list[str] = []

    def fit(
        self,
        val_predictions: Sequence[np.ndarray],
        y_val: np.ndarray,
        model_names: Optional[Sequence[str]] = None,
        y_train_mean: Optional[float] = None,
    ) -> "CaruanaEnsemble":
        """Greedy selection on validation set.

        Parameters
        ----------
        val_predictions : list of 1D arrays, one per model (predictions on validation set)
        y_val           : ground-truth validation pEC50
        y_train_mean    : training mean for RAE computation
        """
        from src.evaluation.validate import rae

        n_models = len(val_predictions)
        preds = np.stack(val_predictions, axis=0)  # (n_models, n_val)

        self._counts = np.zeros(n_models, dtype=np.float64)
        current_ensemble = np.zeros(len(y_val), dtype=np.float64)
        best_rae = float("inf")

        for iteration in range(self.n_iterations):
            best_model_idx = None
            for i in range(n_models):
                # Add model i to current ensemble (average)
                candidate = (current_ensemble * iteration + preds[i]) / (iteration + 1)
                r = rae(y_val, candidate, y_train_mean=y_train_mean)
                if r < best_rae:
                    best_rae = r
                    best_model_idx = i

            if best_model_idx is None:
                break

            self._counts[best_model_idx] += 1
            current_ensemble = (
                current_ensemble * iteration + preds[best_model_idx]
            ) / (iteration + 1)

            if iteration % 10 == 0:
                logger.info(f"Caruana iter {iteration}: best_rae={best_rae:.4f}")

        if model_names is not None:
            self._model_names = list(model_names)
        else:
            self._model_names = [f"model_{i}" for i in range(n_models)]

        logger.info(f"Caruana selection complete. Final RAE on val: {best_rae:.4f}")
        for name, count in zip(self._model_names, self._counts):
            if count > 0:
                logger.info(f"  {name}: selected {int(count)} times ({count/self.n_iterations*100:.1f}%)")

        return self

    def predict(self, predictions: Sequence[np.ndarray]) -> np.ndarray:
        if self._counts is None:
            raise RuntimeError("Call fit() before predict().")
        weights = self._counts / self._counts.sum()
        preds = np.stack(predictions, axis=0)
        ensemble = (weights[:, None] * preds).sum(axis=0)
        return np.clip(ensemble, *self.clip_range)

    @property
    def weights(self) -> np.ndarray:
        if self._counts is None:
            return np.array([])
        return self._counts / self._counts.sum()


# ---------------------------------------------------------------------------
# ElasticNet meta-learner stacker
# ---------------------------------------------------------------------------

class ElasticNetStacker:
    """ElasticNet meta-learner trained on out-of-fold predictions.

    Learns optimal linear combination of base model OOF predictions using
    L1+L2 regularization. L1 component automatically zeroes out weak models.
    Alpha is selected by inner CV on the OOF set.

    Usage:
        stacker = ElasticNetStacker()
        stacker.fit(oof_matrix, train_y, model_names)  # oof_matrix: (n_train, n_models)
        preds = stacker.predict(test_matrix)            # test_matrix: (n_test, n_models)
    """

    def __init__(
        self,
        l1_ratio: float = 0.7,
        cv: int = 5,
        clip_range: tuple[float, float] = (1.5, 8.0),
    ):
        self.l1_ratio = l1_ratio
        self.cv = cv
        self.clip_range = clip_range
        self._model = None
        self._scaler = None
        self._model_names: list[str] = []

    def fit(
        self,
        oof_predictions: np.ndarray,
        y_train: np.ndarray,
        model_names: Optional[Sequence[str]] = None,
    ) -> "ElasticNetStacker":
        """Fit ElasticNetCV on OOF predictions.

        Parameters
        ----------
        oof_predictions : shape (n_train, n_models)
        y_train         : shape (n_train,)
        """
        from sklearn.linear_model import ElasticNetCV
        from sklearn.preprocessing import StandardScaler

        self._scaler = StandardScaler()
        X = self._scaler.fit_transform(oof_predictions)

        self._model = ElasticNetCV(
            l1_ratio=self.l1_ratio,
            cv=self.cv,
            max_iter=10000,
            random_state=42,
        )
        self._model.fit(X, y_train)

        self._model_names = (
            list(model_names) if model_names is not None
            else [f"model_{i}" for i in range(oof_predictions.shape[1])]
        )

        logger.info(f"ElasticNetStacker: alpha={self._model.alpha_:.4f}, l1_ratio={self.l1_ratio}")
        for name, coef in zip(self._model_names, self._model.coef_):
            logger.info(f"  {name}: coef={coef:.4f}")

        return self

    def predict(self, test_predictions: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() before predict().")
        X = self._scaler.transform(test_predictions)
        return np.clip(self._model.predict(X), *self.clip_range)

    @property
    def coefs(self) -> dict[str, float]:
        if self._model is None:
            return {}
        return dict(zip(self._model_names, self._model.coef_))


# ---------------------------------------------------------------------------
# GP uncertainty-gated adaptive ensemble
# ---------------------------------------------------------------------------

def uncertainty_gated_predict(
    predictions: dict[str, np.ndarray],
    gp_std: np.ndarray,
    low_confidence_threshold: float = 0.3,
    high_confidence_threshold: float = 0.5,
    clip_range: tuple[float, float] = (1.5, 8.0),
) -> np.ndarray:
    """Adaptive ensemble weighting based on GP posterior uncertainty.

    For each test compound:
    - GP std < low_confidence_threshold → upweight local models (kNN, DeepDelta)
    - GP std > high_confidence_threshold → upweight global GNN models (Chemprop)
    - In between → equal weighting

    Parameters
    ----------
    predictions : dict mapping model_name → prediction array
    gp_std      : GP posterior standard deviation per compound
    """
    n = len(gp_std)
    required = {"knn", "chemprop"}
    available = set(predictions.keys())

    # Build compound-wise weighted average
    ensemble = np.zeros(n, dtype=np.float64)

    for i in range(n):
        std = gp_std[i]
        weights = {}

        if std < low_confidence_threshold:
            # High confidence region: trust local models
            weights = {k: 2.0 if k in ("knn", "deepdelta") else 0.5 for k in available}
        elif std > high_confidence_threshold:
            # Low confidence region: trust global GNN models
            weights = {k: 2.0 if "chemprop" in k else 0.5 for k in available}
        else:
            weights = {k: 1.0 for k in available}

        total = sum(weights.values())
        ensemble[i] = sum(
            weights[k] / total * predictions[k][i] for k in available
        )

    return np.clip(ensemble, *clip_range)


# ---------------------------------------------------------------------------
# Submission pipeline
# ---------------------------------------------------------------------------

def make_submission(
    predictions: np.ndarray,
    test_df: pd.DataFrame,
    clip_range: tuple[float, float] = (1.5, 8.0),
    output_dir: Optional[str] = None,
    description: str = "",
    val_rae: Optional[float] = None,
) -> Path:
    """Generate a submission CSV and log the submission.

    Submission format required by the challenge:
        SMILES | Molecule Name | pEC50  (exactly 513 rows)

    Parameters
    ----------
    predictions : 1D array of predicted pEC50 for all 513 test compounds
    test_df     : test DataFrame (must contain 'Molecule Name' and 'SMILES' columns)
    description : human-readable description (saved to submissions log)
    val_rae     : internal validation RAE (for log)

    Returns
    -------
    Path to the saved submission CSV
    """
    assert len(predictions) == len(test_df), (
        f"predictions length {len(predictions)} != test_df length {len(test_df)}"
    )
    assert "Molecule Name" in test_df.columns, "test_df must contain 'Molecule Name' column"
    assert "SMILES" in test_df.columns, "test_df must contain 'SMILES' column"

    predictions = np.clip(predictions, *clip_range)
    predictions = np.round(predictions, 4)

    submission = pd.DataFrame({
        "SMILES": test_df["SMILES"].values,
        "Molecule Name": test_df["Molecule Name"].values,
        "pEC50": predictions,
    })

    # Create output directory
    out_dir = Path(output_dir) if output_dir else SUBMISSIONS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"submission_{timestamp}.csv"
    filepath = out_dir / filename
    submission.to_csv(filepath, index=False)

    logger.info(f"Submission saved to: {filepath}")
    logger.info(f"Prediction stats: mean={predictions.mean():.3f}, "
                f"std={predictions.std():.3f}, "
                f"min={predictions.min():.3f}, max={predictions.max():.3f}")

    # Log submission
    _log_submission(
        filepath=str(filepath),
        description=description,
        val_rae=val_rae,
        pred_mean=float(predictions.mean()),
        pred_std=float(predictions.std()),
    )

    return filepath


def _log_submission(
    filepath: str,
    description: str = "",
    val_rae: Optional[float] = None,
    pred_mean: Optional[float] = None,
    pred_std: Optional[float] = None,
) -> None:
    """Append a row to the submissions log CSV."""
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    log_row = {
        "timestamp": datetime.now().isoformat(),
        "filepath": filepath,
        "description": description,
        "val_rae": val_rae if val_rae is not None else "",
        "leaderboard_rae": "",  # to be filled manually after submission
        "pred_mean": pred_mean if pred_mean is not None else "",
        "pred_std": pred_std if pred_std is not None else "",
        "notes": "",
    }

    log_df = pd.DataFrame([log_row])

    if SUBMISSIONS_LOG.exists():
        existing = pd.read_csv(SUBMISSIONS_LOG)
        log_df = pd.concat([existing, log_df], ignore_index=True)

    log_df.to_csv(SUBMISSIONS_LOG, index=False)
    logger.info(f"Submission logged to {SUBMISSIONS_LOG}")


def validate_submission(filepath: str, test_df: pd.DataFrame) -> bool:
    """Validate a submission CSV using the official challenge validator.

    Requires columns: SMILES, Molecule Name, pEC50 — exactly 513 rows.
    Uses the validator from PXR-Challenge-Tutorial/validation/activity_validation.py.
    """
    import sys
    tutorial_root = Path(__file__).parent.parent.parent.parent / "PXR-Challenge-Tutorial"
    if tutorial_root.exists() and str(tutorial_root) not in sys.path:
        sys.path.insert(0, str(tutorial_root))

    try:
        from validation.activity_validation import validate_activity_submission
        expected_ids = set(test_df["Molecule Name"].values)
        is_valid, errors = validate_activity_submission(Path(filepath), expected_ids=expected_ids)
        if is_valid:
            logger.info(f"Official validation passed: {filepath}")
        else:
            for err in errors:
                logger.error(f"Validation error: {err}")
        return is_valid
    except ImportError:
        logger.warning("Official validator not found — falling back to basic checks.")

    # Fallback basic validation
    sub = pd.read_csv(filepath)
    required = {"SMILES", "Molecule Name", "pEC50"}
    missing_cols = required - set(sub.columns)
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return False
    expected_ids = set(test_df["Molecule Name"].values)
    submitted_ids = set(sub["Molecule Name"].values)
    missing_ids = expected_ids - submitted_ids
    if missing_ids:
        logger.error(f"Missing {len(missing_ids)} compound IDs.")
        return False
    if sub["pEC50"].isna().any() or np.any(np.isinf(sub["pEC50"].values)):
        logger.error("Submission contains NaN or infinite pEC50 values.")
        return False
    logger.info(f"Basic validation passed: {len(sub)} compounds.")
    return True


# ---------------------------------------------------------------------------
# Simple run function for quick baseline submission
# ---------------------------------------------------------------------------

def run_baseline_submission(
    train_smiles: list[str],
    train_y: np.ndarray,
    test_df: pd.DataFrame,
    k: int = 5,
    description: str = "Tanimoto kNN baseline",
) -> Path:
    """End-to-end: fit kNN baseline and generate submission CSV.

    Parameters
    ----------
    test_df : must have 'SMILES' and 'OCNT_ID' columns
    """
    from src.features.feature_engineering import ecfp4
    from src.models.local_models import TanimotoKNN

    logger.info("Computing fingerprints for kNN baseline...")
    fps_train = ecfp4(train_smiles)
    fps_test = ecfp4(test_df["SMILES"].tolist())

    knn = TanimotoKNN(k=k)
    knn.fit(fps_train, train_y)
    preds = knn.predict(fps_test)

    logger.info(f"kNN predictions: mean={preds.mean():.3f}, std={preds.std():.3f}")
    return make_submission(preds, test_df, description=description)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Quick smoke test
    rng = np.random.default_rng(0)
    preds = rng.uniform(4, 7, 10)
    test_df = pd.DataFrame({"OCNT_ID": [f"ID_{i}" for i in range(10)], "SMILES": ["C"] * 10})
    path = make_submission(preds, test_df, description="smoke_test", val_rae=0.75)
    print(f"Saved to {path}")
    print(validate_submission(str(path), test_df))
