"""
Uni-Mol fine-tuned regression model for molecular property prediction.

Fine-tunes the Uni-Mol transformer (pretrained on 209M ZINC/PubChem conformations)
on the PXR pEC50 regression task.

Conformers are pre-generated with RDKit ETKDGv3 rather than delegating to
unimol_tools' ConformerGen, which crashes when any molecule fails embedding.
This wrapper handles failures gracefully and passes atoms+coordinates directly
to DataHub, bypassing ConformerGen entirely.

Install on GPU server:
    pip install unimol_tools
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _rdkit_conformers(
    smiles_list: List[str],
) -> List[Optional[Tuple[List[str], List]]]:
    """Generate ETKDGv3 conformers with RDKit.

    Returns list of (atom_symbols, coordinates) per molecule, or None on failure.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    results = []
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                results.append(None)
                continue
            mol = Chem.AddHs(mol)
            ps = AllChem.ETKDGv3()
            ps.randomSeed = 42
            ok = AllChem.EmbedMolecule(mol, ps)
            if ok == -1:
                ok = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            if ok == -1:
                results.append(None)
                continue
            try:
                AllChem.MMFFOptimizeMolecule(mol)
            except Exception:
                pass
            conf = mol.GetConformer()
            atoms = [a.GetSymbol() for a in mol.GetAtoms()]
            coords = conf.GetPositions().tolist()
            results.append((atoms, coords))
        except Exception:
            results.append(None)
    return results


class UniMolModel:
    """Uni-Mol fine-tuned regression model.

    Requires unimol_tools and a CUDA GPU. Each instance owns its model
    directory so multiple CV fold instances can coexist without conflict.
    """

    def __init__(
        self,
        epochs: int = 15,
        lr: float = 1e-4,
        batch_size: int = 16,
        seed: int = 42,
        save_path: Optional[str] = None,
    ):
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed
        self.save_path = save_path
        self._model_path: Optional[str] = None
        self._train_mean: float = 0.0

    def fit(self, smiles: list, y: np.ndarray) -> "UniMolModel":
        from unimol_tools import MolTrain

        path = self.save_path or tempfile.mkdtemp(prefix="unimol_pxr_")
        Path(path).mkdir(parents=True, exist_ok=True)
        self._model_path = path
        self._train_mean = float(np.mean(y))

        logger.info(f"UniMol: generating conformers for {len(smiles)} compounds...")
        confs = _rdkit_conformers(smiles)
        valid_idx = [i for i, c in enumerate(confs) if c is not None]
        n_fail = len(smiles) - len(valid_idx)
        if n_fail:
            logger.warning(
                f"UniMol: {n_fail}/{len(smiles)} conformer failures — excluded from training"
            )

        data = {
            "atoms": [confs[i][0] for i in valid_idx],
            "coordinates": [np.array(confs[i][1]) for i in valid_idx],
            "target": np.array(y)[valid_idx].tolist(),
        }
        logger.info(f"UniMol fit: {len(valid_idx)} compounds, {self.epochs} epochs → {path}")

        trainer = MolTrain(
            task="regression",
            data_type="molecule",
            epochs=self.epochs,
            learning_rate=self.lr,
            batch_size=self.batch_size,
            seed=self.seed,
            remove_hs=False,
            save_path=path,
        )
        trainer.fit(data)
        return self

    def predict(self, smiles: list) -> np.ndarray:
        from unimol_tools import MolPredict

        if self._model_path is None:
            raise RuntimeError("UniMolModel not fitted. Call fit() first.")

        confs = _rdkit_conformers(smiles)
        valid_idx = [i for i, c in enumerate(confs) if c is not None]
        n_fail = len(smiles) - len(valid_idx)
        if n_fail:
            logger.warning(
                f"UniMol: {n_fail}/{len(smiles)} predict conformer failures — filling with train mean"
            )

        data = {
            "atoms": [confs[i][0] for i in valid_idx],
            "coordinates": [np.array(confs[i][1]) for i in valid_idx],
        }

        predictor = MolPredict(load_model=self._model_path)
        result = predictor.predict(data)

        if isinstance(result, np.ndarray):
            valid_preds = result.ravel().astype(np.float64)
        elif isinstance(result, dict):
            vals = result.get("target", result.get("prediction", next(iter(result.values()))))
            valid_preds = np.array(vals, dtype=np.float64).ravel()
        else:
            valid_preds = np.array(result, dtype=np.float64).ravel()

        preds = np.full(len(smiles), self._train_mean, dtype=np.float64)
        for out_i, in_i in enumerate(valid_idx):
            preds[in_i] = valid_preds[out_i]
        return preds
