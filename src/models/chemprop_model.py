"""
Chemprop v2 D-MPNN wrapper for the PXR challenge.

Supports:
  - Single-task regression (pEC50)
  - Extra molecule-level features (ECFP4 + RDKit descriptors appended to FFN input)
  - Sample weights (inverse-variance + cliff upweighting)
  - Snapshot ensembling (average checkpoints from last N epochs)
  - MPS (Apple Silicon) accelerator

Usage:
    from src.models.chemprop_model import ChempropModel

    model = ChempropModel(epochs=100, device='mps')
    model.fit(train_smiles, train_y, sample_weight=weights)
    preds = model.predict(test_smiles)
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class ChempropModel:
    """Chemprop v2 D-MPNN regressor with optional extra features and snapshot ensembling.

    Parameters
    ----------
    epochs          : total training epochs
    hidden_size     : message passing hidden dimension
    depth           : number of message passing steps
    ffn_num_layers  : FFN depth after aggregation
    dropout         : dropout probability in FFN
    batch_size      : molecules per minibatch
    lr              : initial learning rate (cosine-annealed)
    device          : 'mps', 'cuda', or 'cpu'
    snapshot_epochs : average predictions from last N epoch checkpoints (set 0 to disable)
    extra_features  : if True, append ECFP4 + RDKit descriptors to FFN input
    seed            : random seed
    """

    def __init__(
        self,
        epochs: int = 100,
        hidden_size: int = 300,
        depth: int = 3,
        ffn_num_layers: int = 3,
        dropout: float = 0.1,
        batch_size: int = 64,
        lr: float = 1e-3,
        device: str = "mps",
        snapshot_epochs: int = 5,
        extra_features: bool = True,
        seed: int = 42,
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
        self.extra_features = extra_features
        self.seed = seed

        self._snapshot_preds: list[np.ndarray] = []
        self._train_smiles: list[str] = []
        self._train_y: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Feature helpers
    # ------------------------------------------------------------------

    def _compute_extra_features(self, smiles: list[str]) -> Optional[np.ndarray]:
        """Compute ECFP4 + RDKit descriptor matrix for extra molecule features."""
        if not self.extra_features:
            return None
        try:
            from src.features.feature_engineering import ecfp4, rdkit_descriptors
            fps = ecfp4(smiles).astype(np.float32)       # (n, 2048)
            desc = rdkit_descriptors(smiles).astype(np.float32)  # (n, ~200)
            X = np.hstack([fps, desc])
            # Replace NaN/inf with 0
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            return X
        except Exception as e:
            logger.warning(f"Extra features failed ({e}), training without them.")
            return None

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------

    def _build_mpnn(self, extra_dim: int = 0):
        import torch.nn as nn
        from chemprop.models import MPNN
        from chemprop.nn import BondMessagePassing, MeanAggregation, RegressionFFN

        mp = BondMessagePassing(d_h=self.hidden_size, depth=self.depth)
        agg = MeanAggregation()

        # FFN input = hidden_size (from aggregation) + extra_dim
        ffn_input_dim = self.hidden_size + extra_dim
        predictor = RegressionFFN(
            input_dim=ffn_input_dim,
            n_layers=self.ffn_num_layers,
            dropout=self.dropout,
        )

        # X_d_transform must be a callable module when extra features are used;
        # nn.Identity passes x_d through unchanged (features are already scaled)
        x_d_transform = nn.Identity() if extra_dim > 0 else None

        mpnn = MPNN(
            message_passing=mp,
            agg=agg,
            predictor=predictor,
            X_d_transform=x_d_transform,
        )
        return mpnn

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def get_state_dict(self) -> dict:
        """Return a CPU copy of the MPNN state dict for transfer learning."""
        if not hasattr(self, "_mpnn"):
            raise RuntimeError("Model has not been fitted yet.")
        return {k: v.cpu().clone() for k, v in self._mpnn.state_dict().items()}

    def fit(
        self,
        smiles: list[str],
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        smiles_val: Optional[list[str]] = None,
        y_val: Optional[np.ndarray] = None,
        init_state_dict: Optional[dict] = None,
        x_d: Optional[np.ndarray] = None,
    ) -> "ChempropModel":
        """Fit the model.

        Parameters
        ----------
        x_d : optional molecule-level descriptor matrix, shape (n, d).
              Appended to the FFN input alongside the GNN embedding.
              Use for concentration-aware HTS pre-training (d=1, log10[conc]).
        """
        import copy
        import torch
        import torch.nn as nn
        from rdkit import Chem
        from chemprop.data import BatchMolGraph
        from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer

        torch.manual_seed(self.seed)
        y = np.array(y, dtype=np.float64)
        n = len(smiles)

        weights_arr = (
            sample_weight.astype(np.float32)
            if sample_weight is not None
            else np.ones(n, dtype=np.float32)
        )
        targets_arr = y.astype(np.float32)

        x_d_arr: Optional[np.ndarray] = None
        extra_dim = 0
        if x_d is not None:
            x_d_arr = np.asarray(x_d, dtype=np.float32)
            if x_d_arr.ndim == 1:
                x_d_arr = x_d_arr[:, None]
            extra_dim = x_d_arr.shape[1]

        # Pre-featurize all molecules once — avoids DataLoader + semaphore issues
        featurizer = SimpleMoleculeMolGraphFeaturizer()
        mols = [Chem.MolFromSmiles(s) for s in smiles]
        mol_graphs = [featurizer(mol) for mol in mols]

        device = torch.device(self.device)
        mpnn = self._build_mpnn(extra_dim=extra_dim).to(device)

        if init_state_dict is not None:
            # Transfer only the GNN encoder (message_passing.*) weights.
            # The predictor/FFN is deliberately excluded: its input_dim depends on
            # extra_dim and will mismatch if pretraining used x_d but fine-tuning does not.
            encoder_state = {
                k: v.to(device)
                for k, v in init_state_dict.items()
                if k.startswith("message_passing.")
            }
            missing, unexpected = mpnn.load_state_dict(encoder_state, strict=False)
            logger.info(
                f"Transferred {len(encoder_state)} encoder keys from pre-trained weights "
                f"({len(missing)} keys left at random init — FFN re-initialized)."
            )

        optimizer = torch.optim.Adam(mpnn.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=self.lr * 0.01
        )
        criterion = nn.L1Loss(reduction="none")

        snapshot_states: list = []
        indices = np.arange(n)

        for epoch in range(self.epochs):
            mpnn.train()
            np.random.shuffle(indices)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n, self.batch_size):
                idx = indices[start : start + self.batch_size]
                mgs = [mol_graphs[i] for i in idx]
                bmg = BatchMolGraph(mgs)
                bmg.to(device)

                tgt = torch.tensor(targets_arr[idx], dtype=torch.float32, device=device).unsqueeze(1)
                wt  = torch.tensor(weights_arr[idx], dtype=torch.float32, device=device).unsqueeze(1)

                x_d_batch = None
                if x_d_arr is not None:
                    x_d_batch = torch.tensor(x_d_arr[idx], dtype=torch.float32, device=device)

                optimizer.zero_grad()
                preds = mpnn(bmg, V_d=None, X_d=x_d_batch)
                loss = (criterion(preds, tgt) * wt).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(mpnn.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()

            if (epoch + 1) % 10 == 0 or epoch == self.epochs - 1:
                logger.info(
                    f"Epoch {epoch+1}/{self.epochs} — "
                    f"loss={epoch_loss/n_batches:.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )

            if self.snapshot_epochs > 0 and epoch >= self.epochs - self.snapshot_epochs:
                snapshot_states.append(copy.deepcopy(mpnn.state_dict()))

        if snapshot_states:
            avg_state = {
                k: torch.stack([sd[k].float() for sd in snapshot_states]).mean(0)
                for k in snapshot_states[0]
            }
            mpnn.load_state_dict(avg_state)
            logger.info(f"Averaged {len(snapshot_states)} snapshots.")

        self._mpnn = mpnn
        self._extra_dim = extra_dim
        logger.info("Training complete.")
        return self

    # ------------------------------------------------------------------
    # predict — fully Lightning-free direct PyTorch inference
    # ------------------------------------------------------------------

    def predict(self, smiles: list[str]) -> np.ndarray:
        logger.info(f"predict() called: {len(smiles)} molecules")
        import torch
        from rdkit import Chem
        from chemprop.data import BatchMolGraph
        from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer

        featurizer = SimpleMoleculeMolGraphFeaturizer()
        logger.info("featurizer created")
        device = next(self._mpnn.parameters()).device
        logger.info(f"device: {device}")
        self._mpnn.eval()

        all_preds = []
        with torch.no_grad():
            for i in range(0, len(smiles), self.batch_size):
                batch_smiles = smiles[i : i + self.batch_size]
                mols = [Chem.MolFromSmiles(s) for s in batch_smiles]
                mgs = [featurizer(mol) for mol in mols]
                bmg = BatchMolGraph(mgs)
                bmg.to(device)  # in-place, returns None
                out = self._mpnn(bmg, V_d=None, X_d=None)
                all_preds.append(out.cpu().numpy())

        return np.concatenate(all_preds).flatten()

    # ------------------------------------------------------------------
    # CV helper: predict on a held-out fold without re-training
    # ------------------------------------------------------------------

    @staticmethod
    def cross_val_predict(
        smiles: list[str],
        y: np.ndarray,
        sample_weight: Optional[np.ndarray],
        fold_indices: list[tuple[np.ndarray, np.ndarray]],
        **model_kwargs,
    ) -> np.ndarray:
        """Run scaffold CV and return out-of-fold predictions."""
        oof = np.zeros(len(y))
        for fold, (tr_idx, va_idx) in enumerate(fold_indices):
            logger.info(f"Fold {fold}: train={len(tr_idx)}, val={len(va_idx)}")
            sm_tr = [smiles[i] for i in tr_idx]
            sm_va = [smiles[i] for i in va_idx]
            y_tr = y[tr_idx]
            sw_tr = sample_weight[tr_idx] if sample_weight is not None else None

            m = ChempropModel(**model_kwargs)
            m.fit(sm_tr, y_tr, sample_weight=sw_tr)
            oof[va_idx] = m.predict(sm_va)
        return oof
