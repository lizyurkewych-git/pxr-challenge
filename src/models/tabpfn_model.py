"""
TabPFN regressor using CheMeleon embeddings as molecular representation.

Combines two ideas from "Tabular Foundation Models for Molecular Property
Prediction" (Ben Hicham et al. 2025):
  - TabPFN v2: a transformer pretrained on synthetic tabular data that performs
    in-context learning — no gradient updates at fit time, just a forward pass
    that conditions on the full (X_train, y_train) context.
  - CheMeleon embeddings: frozen 2048-dim D-MPNN fingerprints that capture rich
    structural information, shown to outperform ECFP4/RDKit descriptors on
    activity cliff benchmarks when paired with TFMs.

PCA compression (2048→n_components, default 200) is applied to keep the TabPFN
inference tractable. A StandardScaler is applied before PCA.

Usage:
    from src.models.tabpfn_model import TabPFNCheMeleonModel

    model = TabPFNCheMeleonModel(n_components=200, n_estimators=16)
    model.fit(train_smiles, train_y)
    preds = model.predict(test_smiles)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class TabPFNCheMeleonModel:
    """TabPFN regressor backed by CheMeleon embeddings + PCA.

    Parameters
    ----------
    n_components   : PCA components to keep (applied to 2048-dim CheMeleon output)
    n_estimators   : TabPFN ensemble size (more = slightly better, slower)
    device         : 'cpu' or 'cuda'
    cache_dir      : where CheMeleon embeddings are cached (default 'data/embed_cache')
    random_state   : reproducibility seed
    """

    def __init__(
        self,
        n_components: int = 200,
        n_estimators: int = 16,
        device: str = "cpu",
        cache_dir: str = "data/embed_cache",
        random_state: int = 42,
    ):
        self.n_components = n_components
        self.n_estimators = n_estimators
        self.device = device
        self.cache_dir = cache_dir
        self.random_state = random_state

        self._pca = None
        self._scaler = None
        self._tabpfn = None
        self._X_train_pca: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def _embed(self, smiles: list[str]) -> np.ndarray:
        """Return raw CheMeleon embeddings (n, 2048)."""
        from src.models.foundation_embeddings import CheMeleonEmbedder
        embedder = CheMeleonEmbedder(device="cpu", cache_dir=self.cache_dir)
        return embedder.transform(smiles)

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, smiles: list[str], y: np.ndarray) -> "TabPFNCheMeleonModel":
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        from tabpfn import TabPFNRegressor

        logger.info(f"TabPFNCheMeleon: embedding {len(smiles)} training compounds...")
        X_raw = self._embed(smiles)  # (n, 2048)

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_raw)

        n_comp = min(self.n_components, X_scaled.shape[0] - 1, X_scaled.shape[1])
        self._pca = PCA(n_components=n_comp, random_state=self.random_state)
        self._X_train_pca = self._pca.fit_transform(X_scaled)  # (n, n_comp)
        self._y_train = np.asarray(y, dtype=np.float32)

        self._tabpfn = TabPFNRegressor(
            n_estimators=self.n_estimators,
            device=self.device,
            random_state=self.random_state,
            ignore_pretraining_limits=True,
        )
        self._tabpfn.fit(self._X_train_pca, self._y_train)
        logger.info(
            f"TabPFNCheMeleon: fit on {len(smiles)} compounds, "
            f"PCA({X_raw.shape[1]}→{n_comp}), n_estimators={self.n_estimators}"
        )
        return self

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(self, smiles: list[str]) -> np.ndarray:
        logger.info(f"TabPFNCheMeleon: predicting {len(smiles)} compounds...")
        X_raw = self._embed(smiles)
        X_scaled = self._scaler.transform(X_raw)
        X_pca = self._pca.transform(X_scaled)
        preds = self._tabpfn.predict(X_pca).astype(np.float32)
        logger.info(
            f"TabPFNCheMeleon: mean={preds.mean():.3f}, std={preds.std():.3f}"
        )
        return preds
