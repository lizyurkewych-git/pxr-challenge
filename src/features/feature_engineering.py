"""
Feature engineering pipeline for molecular property prediction.

Provides:
  - Morgan / ECFP fingerprints (binary + count)
  - FCFP pharmacophore fingerprints
  - RDKit physicochemical descriptors
  - Mordred 2D descriptors
  - Combined feature matrix for tree-based models

Usage:
    from src.features.feature_engineering import FeaturePipeline
    pipe = FeaturePipeline()
    X_train = pipe.fit_transform(train_smiles)
    X_test  = pipe.transform(test_smiles)
"""

from __future__ import annotations

import logging
import warnings
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem import rdFingerprintGenerator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pre-built generator singletons — created once, reused for every call
# ---------------------------------------------------------------------------
_morgan2_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
_morgan3_gen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)
_fcfp2_gen   = rdFingerprintGenerator.GetMorganGenerator(
    radius=2, fpSize=2048,
    atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen(),
)


def _get_morgan2(n_bits: int, use_chirality: bool) -> "rdFingerprintGenerator.FingerprintGenerator64":
    """Return a (possibly new) generator for the given params."""
    if n_bits == 2048 and use_chirality is False:
        return _morgan2_gen
    return rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=n_bits, includeChirality=use_chirality
    )


def _get_morgan3(n_bits: int, use_chirality: bool) -> "rdFingerprintGenerator.FingerprintGenerator64":
    if n_bits == 2048 and use_chirality is False:
        return _morgan3_gen
    return rdFingerprintGenerator.GetMorganGenerator(
        radius=3, fpSize=n_bits, includeChirality=use_chirality
    )


# ---------------------------------------------------------------------------
# Fingerprint functions
# ---------------------------------------------------------------------------

def smiles_to_mol(smi: str) -> Optional[Chem.Mol]:
    try:
        mol = Chem.MolFromSmiles(smi)
        return mol
    except Exception:
        return None


def ecfp4(smiles: Sequence[str], n_bits: int = 2048, use_chirality: bool = True) -> np.ndarray:
    """Binary ECFP4 (Morgan radius=2) fingerprints."""
    gen = _get_morgan2(n_bits, use_chirality)
    fps = []
    for smi in smiles:
        mol = smiles_to_mol(smi)
        if mol is None:
            fps.append(np.zeros(n_bits, dtype=np.uint8))
        else:
            fps.append(gen.GetFingerprintAsNumPy(mol).astype(np.uint8))
    return np.vstack(fps)


def ecfp6(smiles: Sequence[str], n_bits: int = 2048, use_chirality: bool = True) -> np.ndarray:
    """Binary ECFP6 (Morgan radius=3) fingerprints."""
    gen = _get_morgan3(n_bits, use_chirality)
    fps = []
    for smi in smiles:
        mol = smiles_to_mol(smi)
        if mol is None:
            fps.append(np.zeros(n_bits, dtype=np.uint8))
        else:
            fps.append(gen.GetFingerprintAsNumPy(mol).astype(np.uint8))
    return np.vstack(fps)


def ecfp4_count(smiles: Sequence[str], n_bits: int = 2048) -> np.ndarray:
    """Count-based Morgan (radius=2) fingerprints — more information-dense than binary."""
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
    fps = []
    for smi in smiles:
        mol = smiles_to_mol(smi)
        if mol is None:
            fps.append(np.zeros(n_bits, dtype=np.float32))
        else:
            fps.append(gen.GetCountFingerprintAsNumPy(mol).astype(np.float32))
    return np.vstack(fps)


def fcfp4(smiles: Sequence[str], n_bits: int = 2048) -> np.ndarray:
    """Feature-based (pharmacophore) ECFP4 fingerprints."""
    fps = []
    for smi in smiles:
        mol = smiles_to_mol(smi)
        if mol is None:
            fps.append(np.zeros(n_bits, dtype=np.uint8))
        else:
            fps.append(_fcfp2_gen.GetFingerprintAsNumPy(mol).astype(np.uint8))
    return np.vstack(fps)


# ---------------------------------------------------------------------------
# RDKit physicochemical descriptors
# ---------------------------------------------------------------------------

# Curated set of reliable, non-redundant RDKit descriptors
RDKIT_DESCRIPTOR_NAMES = [
    "MolWt", "ExactMolWt", "HeavyAtomMolWt", "NumHAcceptors", "NumHDonors",
    "NumHeteroatoms", "NumRotatableBonds", "NumAromaticRings", "NumSaturatedRings",
    "NumAliphaticRings", "NumAromaticHeterocycles", "NumSaturatedHeterocycles",
    "NumAliphaticHeterocycles", "NumAromaticCarbocycles", "NumSaturatedCarbocycles",
    "NumAliphaticCarbocycles", "RingCount", "FractionCSP3", "HeavyAtomCount",
    "NHOHCount", "NOCount", "NumValenceElectrons", "NumRadicalElectrons",
    "MolLogP", "MolMR", "TPSA", "LabuteASA",
    "BalabanJ", "BertzCT", "Chi0", "Chi0n", "Chi0v", "Chi1", "Chi1n",
    "Chi1v", "Chi2n", "Chi2v", "Chi3n", "Chi3v", "Chi4n", "Chi4v",
    "Ipc", "Kappa1", "Kappa2", "Kappa3",
    "HallKierAlpha", "MaxAbsPartialCharge", "MinAbsPartialCharge",
    "MaxPartialCharge", "MinPartialCharge",
]

# Build the RDKit descriptor calculator functions once
_RDKIT_FUNCS = {name: func for name, func in Descriptors.descList if name in RDKIT_DESCRIPTOR_NAMES}


def rdkit_descriptors(smiles: Sequence[str]) -> pd.DataFrame:
    """Compute curated RDKit physicochemical descriptors.

    Returns DataFrame with one row per compound, one column per descriptor.
    Missing values (failed calculation) are filled with NaN.
    """
    records = []
    for smi in smiles:
        mol = smiles_to_mol(smi)
        if mol is None:
            records.append({name: np.nan for name in RDKIT_DESCRIPTOR_NAMES})
            continue
        row = {}
        for name in RDKIT_DESCRIPTOR_NAMES:
            func = _RDKIT_FUNCS.get(name)
            if func is None:
                row[name] = np.nan
                continue
            try:
                row[name] = func(mol)
            except Exception:
                row[name] = np.nan
        records.append(row)
    return pd.DataFrame(records, columns=RDKIT_DESCRIPTOR_NAMES)


# ---------------------------------------------------------------------------
# Mordred 2D descriptors
# ---------------------------------------------------------------------------

def mordred_descriptors(
    smiles: Sequence[str],
    ignore_3d: bool = True,
    missing_threshold: float = 0.05,
) -> pd.DataFrame:
    """Compute Mordred 2D descriptors and return clean DataFrame.

    Parameters
    ----------
    missing_threshold : drop descriptors with > this fraction missing values
    """
    try:
        from mordred import Calculator, descriptors as mordred_descs
    except ImportError:
        logger.warning("mordred not installed. Returning empty DataFrame.")
        return pd.DataFrame(index=range(len(smiles)))

    calc = Calculator(mordred_descs, ignore_3D=ignore_3d)
    mols = [smiles_to_mol(smi) for smi in smiles]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # nproc=1: force serial computation — multiprocessing Manager crashes on macOS Python 3.11
        result = calc.pandas(mols, nproc=1)

    # Convert to numeric; non-numeric values → NaN
    result = result.apply(pd.to_numeric, errors="coerce")

    # Drop columns with too many missing values
    missing_frac = result.isna().mean()
    keep_cols = missing_frac[missing_frac <= missing_threshold].index
    result = result[keep_cols]

    # Impute remaining NaNs with column median
    for col in result.columns:
        if result[col].isna().any():
            result[col] = result[col].fillna(result[col].median())

    logger.info(f"Mordred: {result.shape[1]} descriptors retained (from 1826) after filtering.")
    return result


# ---------------------------------------------------------------------------
# Feature pipeline: combines fingerprints + descriptors for tree models
# ---------------------------------------------------------------------------

class FeaturePipeline:
    """Fit/transform pipeline that builds the combined feature matrix for GBM models.

    Features: ECFP4 (2048) + ECFP4_count (2048) + RDKit (50) + Mordred_PCA (300)
    Dimensionality is reduced via VarianceThreshold + PCA for the descriptor block.
    """

    def __init__(
        self,
        n_bits: int = 2048,
        mordred_pca_components: int = 300,
        include_mordred: bool = True,
        include_ecfp6: bool = False,
        include_fcfp4: bool = False,
    ):
        self.n_bits = n_bits
        self.mordred_pca_components = mordred_pca_components
        self.include_mordred = include_mordred
        self.include_ecfp6 = include_ecfp6
        self.include_fcfp4 = include_fcfp4

        self._scaler = None
        self._var_selector = None
        self._pca = None
        self._rdkit_medians: Optional[pd.Series] = None
        self._mordred_cols: Optional[list] = None  # columns retained at fit time
        self._mordred_col_medians: Optional[dict] = None  # per-column medians for imputation
        self._fitted = False

    def _build_fp_block(self, smiles: Sequence[str]) -> np.ndarray:
        parts = [ecfp4(smiles, self.n_bits), ecfp4_count(smiles, self.n_bits)]
        if self.include_ecfp6:
            parts.append(ecfp6(smiles, self.n_bits))
        if self.include_fcfp4:
            parts.append(fcfp4(smiles, self.n_bits))
        return np.hstack(parts).astype(np.float32)

    def _build_rdkit_block(self, smiles: Sequence[str]) -> np.ndarray:
        df = rdkit_descriptors(smiles)
        if self._rdkit_medians is not None:
            df = df.fillna(self._rdkit_medians)
        else:
            df = df.fillna(df.median())
        return df.values.astype(np.float32)

    def fit_transform(self, smiles: Sequence[str]) -> np.ndarray:
        """Fit the descriptor preprocessing pipeline and return transformed features."""
        from sklearn.decomposition import PCA
        from sklearn.feature_selection import VarianceThreshold
        from sklearn.preprocessing import StandardScaler

        fp_block = self._build_fp_block(smiles)

        rdkit_df = rdkit_descriptors(smiles)
        self._rdkit_medians = rdkit_df.median()
        rdkit_block = rdkit_df.fillna(self._rdkit_medians).values.astype(np.float32)

        parts = [fp_block, rdkit_block]

        if self.include_mordred:
            mordred_df = mordred_descriptors(smiles)
            if mordred_df.shape[1] > 0:
                # Save column list and per-column medians so transform() uses identical columns
                self._mordred_cols = list(mordred_df.columns)
                self._mordred_col_medians = mordred_df.median().to_dict()

                mordred_block = mordred_df.values.astype(np.float32)

                self._var_selector = VarianceThreshold(threshold=0.01)
                mordred_block = self._var_selector.fit_transform(mordred_block)

                self._scaler = StandardScaler()
                mordred_block = self._scaler.fit_transform(mordred_block)

                n_comp = min(self.mordred_pca_components, mordred_block.shape[1])
                self._pca = PCA(n_components=n_comp, random_state=42)
                mordred_block = self._pca.fit_transform(mordred_block).astype(np.float32)

                parts.append(mordred_block)

        X = np.hstack(parts)
        self._fitted = True
        logger.info(f"FeaturePipeline fit: feature matrix shape = {X.shape}")
        return X

    def transform(self, smiles: Sequence[str]) -> np.ndarray:
        """Transform new compounds using the already-fitted pipeline."""
        if not self._fitted:
            raise RuntimeError("Pipeline has not been fit. Call fit_transform() first.")

        fp_block = self._build_fp_block(smiles)
        rdkit_block = rdkit_descriptors(smiles).fillna(self._rdkit_medians).values.astype(np.float32)

        parts = [fp_block, rdkit_block]

        if self.include_mordred and self._pca is not None:
            mordred_df = mordred_descriptors(smiles)
            if mordred_df.shape[1] > 0:
                # Align to the exact columns seen at fit time; fill missing with train medians
                mordred_df = mordred_df.reindex(columns=self._mordred_cols)
                for col, med in self._mordred_col_medians.items():
                    if mordred_df[col].isna().any():
                        mordred_df[col] = mordred_df[col].fillna(med)
                mordred_block = mordred_df.values.astype(np.float32)
                mordred_block = self._var_selector.transform(mordred_block)
                mordred_block = self._scaler.transform(mordred_block)
                mordred_block = self._pca.transform(mordred_block).astype(np.float32)
                parts.append(mordred_block)

        return np.hstack(parts)


# ---------------------------------------------------------------------------
# Tanimoto similarity utilities (used by kNN and DeepDelta)
# ---------------------------------------------------------------------------

def tanimoto_matrix(fps_a: np.ndarray, fps_b: np.ndarray) -> np.ndarray:
    """Compute Tanimoto similarity matrix between two sets of binary fingerprints.

    Parameters
    ----------
    fps_a : shape (n, bits) uint8 binary fingerprints
    fps_b : shape (m, bits) uint8 binary fingerprints

    Returns
    -------
    sim : shape (n, m) float32 Tanimoto similarity matrix
    """
    fps_a = fps_a.astype(np.float64)
    fps_b = fps_b.astype(np.float64)

    # Dot product = intersection counts
    # errstate suppresses spurious Accelerate/BLAS warnings on Apple Silicon
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        intersect = fps_a @ fps_b.T  # (n, m)

    # Sum of bits in each fingerprint
    sum_a = fps_a.sum(axis=1, keepdims=True)  # (n, 1)
    sum_b = fps_b.sum(axis=1, keepdims=True).T  # (1, m)

    union = sum_a + sum_b - intersect
    # Safe division: only divide where union > 0 (avoids 0/0 for all-zero fingerprints)
    sim = np.zeros_like(intersect)
    mask = union > 0
    sim[mask] = intersect[mask] / union[mask]
    return sim.astype(np.float32)


def nearest_neighbors(
    query_fps: np.ndarray,
    train_fps: np.ndarray,
    k: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return indices and similarities of k nearest training compounds for each query.

    Returns
    -------
    indices : shape (n_query, k)  int32
    sims    : shape (n_query, k)  float32
    """
    sim_matrix = tanimoto_matrix(query_fps, train_fps)  # (n_query, n_train)
    # argsort descending
    top_k_idx = np.argsort(-sim_matrix, axis=1)[:, :k]
    top_k_sim = sim_matrix[np.arange(len(query_fps))[:, None], top_k_idx]
    return top_k_idx.astype(np.int32), top_k_sim.astype(np.float32)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_smiles = [
        "Cc1ccc(cc1)S(=O)(=O)N",
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "c1ccc(cc1)C(=O)O",
    ]
    fps = ecfp4(test_smiles)
    print("ECFP4 shape:", fps.shape)
    rdkit_df = rdkit_descriptors(test_smiles)
    print("RDKit descriptors shape:", rdkit_df.shape)
    sim = tanimoto_matrix(fps, fps)
    print("Tanimoto self-similarity diagonal:", np.diag(sim))
