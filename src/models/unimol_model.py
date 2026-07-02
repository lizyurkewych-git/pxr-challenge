"""
Uni-Mol fine-tuned regression model for molecular property prediction.

Fine-tunes the Uni-Mol transformer (pretrained on 209M ZINC/PubChem conformations)
on the PXR pEC50 regression task.

Conformers are pre-generated here with RDKit ETKDGv3 rather than delegating to
unimol_tools' ConformerGen, which uses Python multiprocessing and crashes when any
molecule fails embedding. This wrapper handles failures gracefully and passes
atoms+coordinates directly to DataHub, bypassing ConformerGen entirely.

Multi-conformer modes (controlled by UniMolModel config flags):
  n_train_conformers=1, n_infer_conformers=1  → baseline (byte-for-byte identical to original)
  use_conformer_resampling=False, n_infer_conformers>1  → inference averaging only (Option A)
  use_conformer_resampling=True, n_infer_conformers>1   → train resampling + infer avg (Option B)

Note on training resampling: MolTrain.fit() takes a fixed data dict, so true per-epoch
__getitem__ resampling is not possible without a custom training loop. The approximation
selects one random conformer per molecule before calling fit(), seeded by self.seed.

Install on GPU server:
    pip install unimol_tools
"""

from __future__ import annotations

import hashlib
import json
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Baseline single-conformer generator (original, unchanged)
# ---------------------------------------------------------------------------

def _rdkit_conformers(
    smiles_list: List[str],
) -> List[Optional[Tuple[List[str], List]]]:
    """Generate ETKDGv3 conformers with RDKit (single conformer, seed=42).

    Returns list of (atom_symbols, coordinates) per molecule, or None for
    any molecule that fails embedding. Callers must handle None entries.
    Used by the baseline code path (n_train_conformers=1, n_infer_conformers=1).
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


# ---------------------------------------------------------------------------
# Multi-conformer generator (used by ConformerCache)
# ---------------------------------------------------------------------------

def _rdkit_multi_conformers(
    smiles_list: List[str],
    n_conformers: int = 8,
    base_seed: int = 42,
) -> List[Optional[Dict]]:
    """Generate up to n_conformers ETKDGv3 conformers per molecule.

    Returns a list (one per SMILES) of dicts:
      {"atoms": List[str], "conformers": List[List[List[float]]]}  # [n_conf, n_atoms, 3]
    or None if the molecule fails to parse or embed entirely.
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
            atoms = [a.GetSymbol() for a in mol.GetAtoms()]

            ps = AllChem.ETKDGv3()
            ps.randomSeed = base_seed
            conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, params=ps))
            if len(conf_ids) == 0:
                ps2 = AllChem.ETKDG()
                ps2.randomSeed = base_seed
                conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, params=ps2))
            if len(conf_ids) == 0:
                results.append(None)
                continue

            try:
                AllChem.MMFFOptimizeMoleculeConfs(mol)
            except Exception:
                pass

            conformers = [mol.GetConformer(cid).GetPositions().tolist() for cid in conf_ids]
            results.append({"atoms": atoms, "conformers": conformers})
        except Exception:
            results.append(None)
    return results


# ---------------------------------------------------------------------------
# Disk-backed conformer cache
# ---------------------------------------------------------------------------

class ConformerCache:
    """Cache RDKit conformers on disk, keyed by canonical SMILES (MD5 hash)."""

    def __init__(self, cache_dir: str, n_conformers: int = 8, base_seed: int = 42):
        self.cache_dir = Path(cache_dir)
        self.n_conformers = n_conformers
        self.base_seed = base_seed
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _canonical(self, smi: str) -> str:
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                return Chem.MolToSmiles(mol)
        except Exception:
            pass
        return smi

    def _cache_path(self, canonical_smi: str) -> Path:
        key = hashlib.md5(canonical_smi.encode()).hexdigest()
        return self.cache_dir / f"{key}.json"

    def get_or_generate(self, smiles_list: List[str]) -> List[Optional[Dict]]:
        """Return cached conformers for each SMILES, generating missing entries.

        Each result is {"atoms": [...], "conformers": [[coords_0], [coords_1], ...]}
        or None for molecules that fail to embed.
        """
        canonical_list = [self._canonical(s) for s in smiles_list]
        results: List[Optional[Dict]] = [None] * len(smiles_list)
        need_gen: List[Tuple[int, str]] = []

        for i, csmi in enumerate(canonical_list):
            p = self._cache_path(csmi)
            if p.exists():
                try:
                    cached = json.loads(p.read_text())
                    if cached.get("failed"):
                        results[i] = None
                    else:
                        results[i] = {"atoms": cached["atoms"], "conformers": cached["conformers"]}
                except Exception:
                    need_gen.append((i, csmi))
            else:
                need_gen.append((i, csmi))

        if need_gen:
            logger.info(
                f"ConformerCache: generating {self.n_conformers} conformers "
                f"for {len(need_gen)} new molecules (base_seed={self.base_seed})..."
            )
            gen_smiles = [csmi for _, csmi in need_gen]
            gen_results = _rdkit_multi_conformers(
                gen_smiles, n_conformers=self.n_conformers, base_seed=self.base_seed
            )
            for (orig_idx, csmi), res in zip(need_gen, gen_results):
                p = self._cache_path(csmi)
                if res is None:
                    p.write_text(json.dumps({"smiles": csmi, "failed": True}))
                    results[orig_idx] = None
                else:
                    p.write_text(json.dumps({
                        "smiles": csmi,
                        "n_conformers": len(res["conformers"]),
                        "atoms": res["atoms"],
                        "conformers": res["conformers"],
                    }))
                    results[orig_idx] = res

        return results


# ---------------------------------------------------------------------------
# UniMolModel
# ---------------------------------------------------------------------------

class UniMolModel:
    """Uni-Mol fine-tuned regression model.

    Each instance owns its model directory so multiple CV fold instances
    coexist without conflict.

    Config flags for multi-conformer support:
      use_conformer_resampling (bool):  pick a random conformer per molecule before fit()
      n_train_conformers (int):         number of conformers to cache for training selection
      n_infer_conformers (int):         number of conformers to average at inference time

    With all flags at defaults (False / 1 / 1) behaviour is byte-for-byte identical to
    the original single-conformer implementation.
    """

    def __init__(
        self,
        epochs: int = 15,
        lr: float = 1e-4,
        batch_size: int = 16,
        seed: int = 42,
        save_path: Optional[str] = None,
        use_conformer_resampling: bool = False,
        n_train_conformers: int = 1,
        n_infer_conformers: int = 1,
        cache_dir: str = "data/conformer_cache",
    ):
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed
        self.save_path = save_path
        self.use_conformer_resampling = use_conformer_resampling
        self.n_train_conformers = n_train_conformers
        self.n_infer_conformers = n_infer_conformers
        self.cache_dir = cache_dir
        self._model_path: Optional[str] = None
        self._train_mean: float = 0.0

        logger.info(
            f"UniMolModel: use_conformer_resampling={use_conformer_resampling}, "
            f"n_train_conformers={n_train_conformers}, n_infer_conformers={n_infer_conformers}, "
            f"epochs={epochs}, lr={lr}, seed={seed}"
        )

    def fit(self, smiles: list, y: np.ndarray) -> "UniMolModel":
        from unimol_tools import MolTrain

        path = self.save_path or tempfile.mkdtemp(prefix="unimol_pxr_")
        Path(path).mkdir(parents=True, exist_ok=True)
        self._model_path = path
        self._train_mean = float(np.mean(y))

        use_resampling = self.use_conformer_resampling
        n_train = self.n_train_conformers

        if not use_resampling and n_train <= 1:
            # ---- Baseline path: identical to original ----
            logger.info(f"UniMol: generating conformers for {len(smiles)} compounds...")
            confs = _rdkit_conformers(smiles)
            valid_idx = [i for i, c in enumerate(confs) if c is not None]
            n_fail = len(smiles) - len(valid_idx)
            if n_fail:
                logger.warning(f"UniMol: {n_fail}/{len(smiles)} conformer failures — excluded from training")
            atoms_list = [confs[i][0] for i in valid_idx]
            coords_list = [np.array(confs[i][1]) for i in valid_idx]
        else:
            # ---- Multi-conformer path ----
            n_to_cache = max(n_train, self.n_infer_conformers)
            cache = ConformerCache(self.cache_dir, n_conformers=n_to_cache, base_seed=42)
            cached = cache.get_or_generate(smiles)
            valid_idx = [i for i, c in enumerate(cached) if c is not None]
            n_fail = len(smiles) - len(valid_idx)
            if n_fail:
                logger.warning(f"UniMol: {n_fail}/{len(smiles)} conformer failures — excluded from training")

            if use_resampling and n_train > 1:
                # Approximate per-epoch resampling: pick one random conformer per molecule.
                # MolTrain.fit() takes a fixed data dict, so selection is done once before fit().
                rng = np.random.RandomState(self.seed)
                logger.info(
                    f"UniMol: conformer resampling — selecting 1 of up to {n_train} conformers "
                    f"per molecule (seed={self.seed})"
                )
                atoms_list = []
                coords_list = []
                for i in valid_idx:
                    c = cached[i]
                    n_avail = min(len(c["conformers"]), n_train)
                    pick = rng.randint(0, n_avail)
                    atoms_list.append(c["atoms"])
                    coords_list.append(np.array(c["conformers"][pick]))
            else:
                # n_infer_conformers>1 but training resampling off: use conformer 0
                atoms_list = [cached[i]["atoms"] for i in valid_idx]
                coords_list = [np.array(cached[i]["conformers"][0]) for i in valid_idx]

        data = {
            "atoms": atoms_list,
            "coordinates": coords_list,
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

        preds_out = np.full(len(smiles), self._train_mean, dtype=np.float64)
        n_infer = self.n_infer_conformers

        if n_infer <= 1:
            # ---- Baseline path: identical to original ----
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
            valid_preds = self._parse_predict_result(predictor.predict(data))
            for out_i, in_i in enumerate(valid_idx):
                preds_out[in_i] = valid_preds[out_i]
        else:
            # ---- Inference averaging over n_infer conformers ----
            n_to_cache = max(self.n_train_conformers, n_infer)
            cache = ConformerCache(self.cache_dir, n_conformers=n_to_cache, base_seed=42)
            cached = cache.get_or_generate(smiles)

            logger.info(f"UniMol: inference averaging over {n_infer} conformers...")
            acc = np.zeros(len(smiles), dtype=np.float64)
            count = np.zeros(len(smiles), dtype=np.int32)

            predictor = MolPredict(load_model=self._model_path)
            for conf_idx in range(n_infer):
                valid_for_conf = [
                    i for i in range(len(smiles))
                    if cached[i] is not None and conf_idx < len(cached[i]["conformers"])
                ]
                if not valid_for_conf:
                    continue
                data = {
                    "atoms": [cached[i]["atoms"] for i in valid_for_conf],
                    "coordinates": [np.array(cached[i]["conformers"][conf_idx]) for i in valid_for_conf],
                }
                preds_c = self._parse_predict_result(predictor.predict(data))
                for out_i, in_i in enumerate(valid_for_conf):
                    acc[in_i] += preds_c[out_i]
                    count[in_i] += 1

            n_fail = int(np.sum(count == 0))
            if n_fail:
                logger.warning(
                    f"UniMol: {n_fail}/{len(smiles)} predict failures — filling with train mean"
                )
            preds_out = np.where(count > 0, acc / np.maximum(count, 1), self._train_mean)

        return preds_out

    @staticmethod
    def _parse_predict_result(result) -> np.ndarray:
        """Parse MolPredict output into a 1D float64 array."""
        if isinstance(result, np.ndarray):
            return result.ravel().astype(np.float64)
        elif isinstance(result, dict):
            vals = result.get("target", result.get("prediction", next(iter(result.values()))))
            return np.array(vals, dtype=np.float64).ravel()
        else:
            return np.array(result, dtype=np.float64).ravel()
