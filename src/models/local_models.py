"""
Local similarity-based models:
  - TanimotoKNN : k-nearest-neighbor regression using Tanimoto similarity on ECFP4
  - TanimotoGP  : Gaussian Process regression with Tanimoto kernel (provides uncertainty)

Both are sklearn-compatible (fit / predict interface) and can be dropped into
the ensemble pipeline.

Usage:
    from src.models.local_models import TanimotoKNN, TanimotoGP
    from src.features.feature_engineering import ecfp4

    fps_train = ecfp4(train_smiles)
    fps_test  = ecfp4(test_smiles)

    knn = TanimotoKNN(k=5)
    knn.fit(fps_train, y_train)
    preds_knn = knn.predict(fps_test)

    gp = TanimotoGP()
    gp.fit(fps_train, y_train)
    preds_gp, std_gp = gp.predict(fps_test, return_std=True)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tanimoto similarity helpers
# ---------------------------------------------------------------------------

def _tanimoto_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Vectorised Tanimoto similarity. A: (n, d), B: (m, d) → (n, m) float32."""
    A = A.astype(np.float64)
    B = B.astype(np.float64)
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        intersect = A @ B.T
    sum_A = A.sum(axis=1, keepdims=True)
    sum_B = B.sum(axis=1, keepdims=True).T
    union = sum_A + sum_B - intersect
    sim = np.zeros_like(intersect)
    mask = union > 0
    sim[mask] = intersect[mask] / union[mask]
    return sim.astype(np.float32)


# ---------------------------------------------------------------------------
# Tanimoto kNN
# ---------------------------------------------------------------------------

class TanimotoKNN:
    """Similarity-weighted k-nearest-neighbor regression on binary fingerprints.

    Prediction for query x:
        ŷ = Σ_i sim(x, x_i) * y_i  /  Σ_i sim(x, x_i)

    where the sum is over the k most similar training compounds.
    """

    def __init__(self, k: int = 5, min_sim: float = 0.0):
        """
        Parameters
        ----------
        k       : number of neighbors
        min_sim : minimum Tanimoto similarity to include a neighbor (0 = include all)
        """
        self.k = k
        self.min_sim = min_sim
        self._train_fps: Optional[np.ndarray] = None
        self._train_y: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TanimotoKNN":
        self._train_fps = X.astype(np.uint8)
        self._train_y = np.array(y, dtype=np.float64)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._train_fps is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        sim = _tanimoto_matrix(X, self._train_fps)  # (n_query, n_train)
        k = min(self.k, sim.shape[1])

        # Get top-k indices
        top_k_idx = np.argsort(-sim, axis=1)[:, :k]
        top_k_sim = sim[np.arange(len(X))[:, None], top_k_idx]

        # Apply minimum similarity threshold
        mask = top_k_sim >= self.min_sim
        top_k_sim = top_k_sim * mask  # zero out below-threshold neighbors

        top_k_y = self._train_y[top_k_idx]
        weight_sum = top_k_sim.sum(axis=1)

        # Weighted mean; fall back to unweighted mean if all weights are 0
        preds = np.where(
            weight_sum > 0,
            (top_k_sim * top_k_y).sum(axis=1) / weight_sum,
            top_k_y.mean(axis=1),
        )
        return preds.astype(np.float64)

    def predict_with_neighbors(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return predictions along with neighbor indices and similarities.

        Returns
        -------
        preds    : shape (n,) predicted pEC50
        nn_idx   : shape (n, k) int32 indices into training set
        nn_sim   : shape (n, k) float32 Tanimoto similarities
        """
        sim = _tanimoto_matrix(X, self._train_fps)
        k = min(self.k, sim.shape[1])
        top_k_idx = np.argsort(-sim, axis=1)[:, :k]
        top_k_sim = sim[np.arange(len(X))[:, None], top_k_idx]
        mask = top_k_sim >= self.min_sim
        weighted_sim = top_k_sim * mask
        top_k_y = self._train_y[top_k_idx]
        weight_sum = weighted_sim.sum(axis=1)
        preds = np.where(
            weight_sum > 0,
            (weighted_sim * top_k_y).sum(axis=1) / weight_sum,
            top_k_y.mean(axis=1),
        )
        return preds.astype(np.float64), top_k_idx.astype(np.int32), top_k_sim.astype(np.float32)


# ---------------------------------------------------------------------------
# Tanimoto Kernel for sklearn GP
# ---------------------------------------------------------------------------

class TanimotoKernel:
    """Tanimoto kernel for use with sklearn's GaussianProcessRegressor.

    K(x_i, x_j) = |x_i ∩ x_j| / |x_i ∪ x_j|

    Implements the sklearn kernel interface sufficiently for GPR.
    """

    def __init__(self, sigma_f: float = 1.0):
        self.sigma_f = sigma_f

    def __call__(self, X: np.ndarray, Y: Optional[np.ndarray] = None, eval_gradient: bool = False):
        if Y is None:
            Y = X
        K = self.sigma_f ** 2 * _tanimoto_matrix(X, Y)
        if eval_gradient:
            # Gradient w.r.t. log(sigma_f): dK/d(log sigma_f) = 2 * K
            return K, 2 * K[:, :, np.newaxis]
        return K

    def diag(self, X: np.ndarray) -> np.ndarray:
        # Tanimoto(x, x) = 1 for all-nonzero vectors
        ones = np.ones(len(X), dtype=np.float32)
        return self.sigma_f ** 2 * ones

    def is_stationary(self) -> bool:
        return False

    def get_params(self, deep: bool = True) -> dict:
        return {"sigma_f": self.sigma_f}

    def set_params(self, **params) -> "TanimotoKernel":
        for key, val in params.items():
            setattr(self, key, val)
        return self

    def clone_with_theta(self, theta: np.ndarray) -> "TanimotoKernel":
        new = TanimotoKernel(sigma_f=np.exp(theta[0]))
        return new

    @property
    def theta(self) -> np.ndarray:
        return np.log([self.sigma_f])

    @theta.setter
    def theta(self, value: np.ndarray) -> None:
        self.sigma_f = np.exp(value[0])

    @property
    def bounds(self) -> np.ndarray:
        return np.array([[-4.0, 4.0]])

    @property
    def n_dims(self) -> int:
        return 1

    def __repr__(self) -> str:
        return f"TanimotoKernel(sigma_f={self.sigma_f:.3f})"


# ---------------------------------------------------------------------------
# Tanimoto GP
# ---------------------------------------------------------------------------

class TanimotoGP:
    """Gaussian Process Regressor with Tanimoto kernel.

    Wraps sklearn.gaussian_process.GaussianProcessRegressor.
    For large datasets (>2000 compounds) training will be slow (O(n³)) —
    subsample or use approximate methods.
    """

    def __init__(
        self,
        sigma_f: float = 1.0,
        alpha: float = 0.1,
        n_restarts_optimizer: int = 3,
        max_train_size: int = 2000,
    ):
        """
        Parameters
        ----------
        alpha             : noise variance (regularization)
        max_train_size    : if training set > this, subsample for GP (speed)
        """
        self.sigma_f = sigma_f
        self.alpha = alpha
        self.n_restarts_optimizer = n_restarts_optimizer
        self.max_train_size = max_train_size
        self._gpr = None
        self._train_fps: Optional[np.ndarray] = None
        self._train_y: Optional[np.ndarray] = None
        self._train_mean: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TanimotoGP":
        from sklearn.gaussian_process import GaussianProcessRegressor

        X = X.astype(np.float32)
        y = np.array(y, dtype=np.float64)

        # Subsample if necessary (GP training is O(n³))
        if len(X) > self.max_train_size:
            idx = np.random.choice(len(X), self.max_train_size, replace=False)
            X, y = X[idx], y[idx]
            logger.warning(
                f"TanimotoGP: subsampled training set to {self.max_train_size} "
                f"compounds for tractable GP fitting."
            )

        self._train_mean = float(y.mean())
        y_centered = y - self._train_mean

        kernel = TanimotoKernel(sigma_f=self.sigma_f)
        self._gpr = GaussianProcessRegressor(
            kernel=kernel,
            alpha=self.alpha,
            n_restarts_optimizer=self.n_restarts_optimizer,
            normalize_y=False,
        )
        self._gpr.fit(X, y_centered)
        self._train_fps = X
        self._train_y = y
        logger.info(f"TanimotoGP fitted on {len(X)} compounds. Kernel: {self._gpr.kernel_}")
        return self

    def predict(
        self, X: np.ndarray, return_std: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Predict pEC50 with optional posterior standard deviation.

        High std → low confidence → rely more on global GNN models in ensemble.
        """
        if self._gpr is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X = X.astype(np.float32)
        if return_std:
            mean, std = self._gpr.predict(X, return_std=True)
            return mean + self._train_mean, std
        else:
            mean = self._gpr.predict(X)
            return mean + self._train_mean

    def uncertainty(self, X: np.ndarray) -> np.ndarray:
        """Shorthand to get posterior standard deviation only."""
        _, std = self.predict(X, return_std=True)
        return std


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.features.feature_engineering import ecfp4

    smiles = [
        "Cc1ccc(cc1)S(=O)(=O)N",
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "c1ccc(cc1)C(=O)O",
        "CN1CCC[C@H]1c2cccnc2",
        "COc1ccc(cc1)C(=O)O",
    ]
    y = np.array([6.5, 5.2, 4.8, 7.1, 5.9])
    fps = ecfp4(smiles)

    knn = TanimotoKNN(k=3)
    knn.fit(fps, y)
    print("kNN predictions:", knn.predict(fps))
