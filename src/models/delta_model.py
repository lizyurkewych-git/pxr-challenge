"""
Pairwise delta learning for PXR activity prediction.

Trains a Chemprop D-MPNN to predict the *difference* in pEC50 between two
molecules. At inference, anchors predictions to k nearest training neighbors:

    pred(t) = mean over k neighbors: y_n + delta(t, n)

This directly optimizes for relative activity within scaffold families —
which is exactly what the analog-set test structure requires.

The model trains on sampled ordered pairs (i, j), target = pEC50_i - pEC50_j.
Cliff pairs (Tanimoto > 0.7, |Δ pEC50| > 1.0) are oversampled by default.
Antisymmetry averaging at inference: delta(t,n) = 0.5*(forward - reverse).

Usage:
    from src.models.delta_model import DeltaChempropModel

    model = DeltaChempropModel(epochs=60, device='cpu')
    model.fit(train_smiles, train_y, fps_train=fps_train)
    preds = model.predict(test_smiles, train_smiles, train_y, fps_train, fps_test)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Internal PyTorch module (shared encoder + delta FFN)
# ------------------------------------------------------------------

class _DeltaMPNN(nn.Module):
    """Shared Chemprop encoder + delta prediction FFN. Internal use only."""

    def __init__(self, mp: nn.Module, agg: nn.Module, ffn: nn.Module):
        super().__init__()
        self.mp = mp
        self.agg = agg
        self.ffn = ffn

    def encode(self, bmg) -> torch.Tensor:
        """Return molecule-level embedding for a BatchMolGraph."""
        H = self.mp(bmg)               # (n_atoms_total, hidden_size)
        return self.agg(H, bmg.batch)  # (n_mols, hidden_size)

    def forward(self, bmg_i, bmg_j) -> torch.Tensor:
        h_i = self.encode(bmg_i)
        h_j = self.encode(bmg_j)
        return self.ffn(torch.cat([h_i, h_j], dim=1))  # (B, 1)


# ------------------------------------------------------------------
# Public model class
# ------------------------------------------------------------------

class DeltaChempropModel:
    """Chemprop D-MPNN trained on pairwise activity differences.

    Parameters
    ----------
    epochs            : training epochs
    hidden_size       : message passing hidden dimension
    depth             : message passing steps
    ffn_num_layers    : depth of the delta FFN (input = 2 * hidden_size)
    dropout           : FFN dropout probability
    batch_size        : pairs per minibatch
    lr                : initial learning rate (cosine-annealed)
    device            : 'cpu', 'mps', or 'cuda'
    snapshot_epochs   : average last N epoch checkpoints (0 = off)
    seed              : random seed
    n_pairs_per_epoch : pairs sampled per epoch
    cliff_oversample  : repeat multiplier for cliff pairs in sampling pool
    k_neighbors       : training neighbors used at inference
    """

    def __init__(
        self,
        epochs: int = 60,
        hidden_size: int = 300,
        depth: int = 3,
        ffn_num_layers: int = 3,
        dropout: float = 0.1,
        batch_size: int = 64,
        lr: float = 1e-3,
        device: str = "cpu",
        snapshot_epochs: int = 5,
        seed: int = 42,
        n_pairs_per_epoch: int = 50_000,
        cliff_oversample: int = 3,
        k_neighbors: int = 10,
    ):
        self.epochs = epochs
        self.hidden_size = hidden_size
        self.depth = depth
        self.ffn_num_layers = ffn_num_layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.snapshot_epochs = snapshot_epochs
        self.seed = seed
        self.n_pairs_per_epoch = n_pairs_per_epoch
        self.cliff_oversample = cliff_oversample
        self.k_neighbors = k_neighbors

        self._model: Optional[_DeltaMPNN] = None

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------

    def _build_model(self) -> _DeltaMPNN:
        from chemprop.nn import BondMessagePassing, MeanAggregation

        mp = BondMessagePassing(d_h=self.hidden_size, depth=self.depth)
        agg = MeanAggregation()

        layers: list[nn.Module] = []
        in_dim = 2 * self.hidden_size
        for _ in range(self.ffn_num_layers - 1):
            layers += [nn.Linear(in_dim, self.hidden_size), nn.ReLU(), nn.Dropout(self.dropout)]
            in_dim = self.hidden_size
        layers.append(nn.Linear(in_dim, 1))
        ffn = nn.Sequential(*layers)

        return _DeltaMPNN(mp, agg, ffn)

    # ------------------------------------------------------------------
    # Pair sampling
    # ------------------------------------------------------------------

    @staticmethod
    def _find_cliff_pairs(
        fps: np.ndarray,
        y: np.ndarray,
        sim_threshold: float = 0.7,
        delta_threshold: float = 1.0,
    ) -> np.ndarray:
        """Return (n_cliff, 2) array of upper-triangle cliff pair indices."""
        from src.features.feature_engineering import tanimoto_matrix
        n = len(y)
        sim = tanimoto_matrix(fps, fps)
        rows, cols = np.where(
            (sim >= sim_threshold)
            & (np.abs(y[:, None] - y[None, :]) >= delta_threshold)
            & (np.arange(n)[:, None] < np.arange(n)[None, :])
        )
        pairs = np.column_stack([rows, cols]) if len(rows) > 0 else np.empty((0, 2), dtype=np.int64)
        logger.info(f"DeltaModel: {len(pairs)} cliff pairs identified for oversampling.")
        return pairs

    def _sample_pairs(
        self,
        n: int,
        rng: np.random.Generator,
        cliff_pairs: Optional[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample n_pairs_per_epoch ordered (i, j) pairs, oversampling cliff pairs."""
        budget = self.n_pairs_per_epoch

        if cliff_pairs is not None and len(cliff_pairs) > 0:
            # Include cliff pairs (both orderings) multiple times
            ci = np.tile(np.concatenate([cliff_pairs[:, 0], cliff_pairs[:, 1]]), self.cliff_oversample)
            cj = np.tile(np.concatenate([cliff_pairs[:, 1], cliff_pairs[:, 0]]), self.cliff_oversample)
            n_cliff = min(len(ci), budget // 4)
            sel = rng.choice(len(ci), size=n_cliff, replace=False)
            fixed_i, fixed_j = ci[sel], cj[sel]
            budget -= n_cliff
        else:
            fixed_i = fixed_j = np.empty(0, dtype=np.int64)

        rand_i = rng.integers(0, n, size=budget)
        rand_j = rng.integers(0, n, size=budget)
        same = rand_i == rand_j
        rand_j[same] = (rand_j[same] + 1) % n

        all_i = np.concatenate([fixed_i, rand_i]).astype(np.int64)
        all_j = np.concatenate([fixed_j, rand_j]).astype(np.int64)
        return all_i, all_j

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        smiles: list[str],
        y: np.ndarray,
        fps_train: Optional[np.ndarray] = None,
        init_state_dict: Optional[dict] = None,
    ) -> "DeltaChempropModel":
        import copy
        from rdkit import Chem
        from chemprop.data import BatchMolGraph
        from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer

        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        y_arr = np.array(y, dtype=np.float32)
        n = len(smiles)

        featurizer = SimpleMoleculeMolGraphFeaturizer()
        mols = [Chem.MolFromSmiles(s) for s in smiles]
        mol_graphs = [featurizer(mol) for mol in mols]

        device = torch.device(self.device)
        model = self._build_model().to(device)

        if init_state_dict is not None:
            # Transfer only message_passing.* (GNN encoder); delta FFN always starts fresh.
            encoder_state = {
                k: v.to(device)
                for k, v in init_state_dict.items()
                if k.startswith("message_passing.")
            }
            missing, _ = model.load_state_dict(encoder_state, strict=False)
            logger.info(
                f"DeltaModel: transferred {len(encoder_state)} encoder keys "
                f"({len(missing)} at random init — delta FFN re-initialized)."
            )

        cliff_pairs: Optional[np.ndarray] = None
        if fps_train is not None:
            try:
                cliff_pairs = self._find_cliff_pairs(fps_train, y_arr)
            except Exception as e:
                logger.warning(f"Cliff pair detection failed ({e}), proceeding without oversampling.")

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=self.lr * 0.01
        )
        criterion = nn.L1Loss()
        snapshot_states: list[dict] = []

        for epoch in range(self.epochs):
            model.train()
            idx_i, idx_j = self._sample_pairs(n, rng, cliff_pairs)
            targets = y_arr[idx_i] - y_arr[idx_j]
            perm = rng.permutation(len(idx_i))
            idx_i, idx_j, targets = idx_i[perm], idx_j[perm], targets[perm]

            epoch_loss, n_batches = 0.0, 0
            for start in range(0, len(idx_i), self.batch_size):
                bi = idx_i[start : start + self.batch_size]
                bj = idx_j[start : start + self.batch_size]

                bmg_i = BatchMolGraph([mol_graphs[k] for k in bi])
                bmg_j = BatchMolGraph([mol_graphs[k] for k in bj])
                bmg_i.to(device)
                bmg_j.to(device)

                tgt = torch.tensor(
                    targets[start : start + self.batch_size],
                    dtype=torch.float32, device=device,
                ).unsqueeze(1)

                optimizer.zero_grad()
                pred = model(bmg_i, bmg_j)
                loss = criterion(pred, tgt)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()

            if (epoch + 1) % 10 == 0 or epoch == self.epochs - 1:
                logger.info(
                    f"DeltaModel epoch {epoch+1}/{self.epochs} — "
                    f"loss={epoch_loss/n_batches:.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )

            if self.snapshot_epochs > 0 and epoch >= self.epochs - self.snapshot_epochs:
                snapshot_states.append(copy.deepcopy(model.state_dict()))

        if snapshot_states:
            avg_state = {
                k: torch.stack([sd[k].float() for sd in snapshot_states]).mean(0)
                for k in snapshot_states[0]
            }
            model.load_state_dict(avg_state)
            logger.info(f"DeltaModel: averaged {len(snapshot_states)} snapshots.")

        self._model = model
        logger.info("DeltaChempropModel training complete.")
        return self

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def _predict_deltas_batched(
        self, smiles_i: list[str], smiles_j: list[str]
    ) -> np.ndarray:
        """Batch-predict delta pEC50 for a flat list of (mol_i, mol_j) pairs."""
        from rdkit import Chem
        from chemprop.data import BatchMolGraph
        from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer

        featurizer = SimpleMoleculeMolGraphFeaturizer()
        device = next(self._model.parameters()).device
        self._model.eval()

        mgs_i = [featurizer(Chem.MolFromSmiles(s)) for s in smiles_i]
        mgs_j = [featurizer(Chem.MolFromSmiles(s)) for s in smiles_j]

        all_deltas: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(smiles_i), self.batch_size):
                bmg_i = BatchMolGraph(mgs_i[start : start + self.batch_size])
                bmg_j = BatchMolGraph(mgs_j[start : start + self.batch_size])
                bmg_i.to(device)
                bmg_j.to(device)
                out = self._model(bmg_i, bmg_j)
                all_deltas.append(out.cpu().numpy())

        return np.concatenate(all_deltas).flatten()

    def predict(
        self,
        test_smiles: list[str],
        train_smiles: list[str],
        train_y: np.ndarray,
        fps_train: np.ndarray,
        fps_test: np.ndarray,
        k: Optional[int] = None,
    ) -> np.ndarray:
        """Predict pEC50 via kNN-anchored delta inference.

        For each test compound t:
        1. Find k nearest training neighbors by Tanimoto similarity
        2. Predict delta(t, n) for each neighbor (with antisymmetry averaging)
        3. Return mean(y_n + delta(t, n))
        """
        from src.features.feature_engineering import tanimoto_matrix

        k_use = min(k or self.k_neighbors, len(train_smiles))
        train_y = np.asarray(train_y, dtype=np.float32)
        n_test = len(test_smiles)

        sim = tanimoto_matrix(fps_test, fps_train)
        top_k_idx = np.argsort(-sim, axis=1)[:, :k_use]  # (n_test, k)

        # Build flat pair lists for a single batched forward pass
        pair_i, pair_j, pair_i_rev, pair_j_rev = [], [], [], []
        for i in range(n_test):
            for nb in top_k_idx[i]:
                pair_i.append(test_smiles[i])
                pair_j.append(train_smiles[nb])
                pair_i_rev.append(train_smiles[nb])
                pair_j_rev.append(test_smiles[i])

        deltas_fwd = self._predict_deltas_batched(pair_i, pair_j).reshape(n_test, k_use)
        deltas_rev = self._predict_deltas_batched(pair_i_rev, pair_j_rev).reshape(n_test, k_use)
        # Antisymmetry: delta(t,n) ≈ -delta(n,t); average both estimates
        deltas = 0.5 * (deltas_fwd - deltas_rev)

        preds = np.array([
            np.mean(train_y[top_k_idx[i]] + deltas[i])
            for i in range(n_test)
        ], dtype=np.float32)

        logger.info(
            f"DeltaModel predict: {n_test} test compounds, k={k_use} neighbors. "
            f"pred mean={preds.mean():.3f}, std={preds.std():.3f}"
        )
        return preds
