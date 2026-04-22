"""
Data loading, canonicalization, and preprocessing for all four dataset tiers.

Usage:
    from src.data.load_data import load_all_tiers, PXRDataset
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column name constants
# ---------------------------------------------------------------------------
COL_SMILES = "SMILES"
COL_PECSO = "pEC50"
COL_EMAX = "Emax_estimate (log2FC vs. baseline)"
COL_EMAX_CTRL = "Emax.vs.pos.ctrl_estimate (dimensionless)"
COL_PECSO_SE = "pEC50_std.error (-log10(molarity))"
COL_EMAX_SE = "Emax_std.error (log2FC vs. baseline)"
COL_ID = "OCNT_ID"
COL_SPLIT = "Split"
COL_SOURCE = "source"

# HTS tier columns
COL_LOG2FC = "log2_fc_estimate"
COL_LOG2FC_SE = "log2_fc_stderr"
COL_T_STAT = "t_statistic"
COL_PVAL = "p_value"
COL_FDR = "fdr_bh"
COL_NEG_LOG_FDR = "neg_log10_fdr"
COL_COHENS_D = "cohens_d"
COL_N_REPS = "n_replicates"
COL_COMPOUND_CLASS = "compound_class"


# ---------------------------------------------------------------------------
# SMILES utilities
# ---------------------------------------------------------------------------

def canonicalize_smiles(smi: str) -> Optional[str]:
    """Return canonical SMILES or None if the SMILES cannot be parsed."""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def _parse_and_clean(df: pd.DataFrame, smiles_col: str = COL_SMILES) -> pd.DataFrame:
    """Canonicalize SMILES and drop un-parseable rows."""
    df = df.copy()
    df[smiles_col] = df[smiles_col].apply(canonicalize_smiles)
    n_bad = df[smiles_col].isna().sum()
    if n_bad > 0:
        logger.warning(f"Dropping {n_bad} rows with unparseable SMILES.")
    df = df.dropna(subset=[smiles_col]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Data loading functions
# ---------------------------------------------------------------------------

def load_primary_drc(cache_dir: Optional[str] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load primary DRC assay (train + test).

    Returns
    -------
    train : DataFrame with pEC50 labels
    test  : DataFrame with SMILES + OCNT_ID only
    """
    from datasets import load_dataset  # lazy import to avoid import-time cost

    ds = load_dataset(
        "openadmet/pxr-challenge-train-test",
        "default",
        cache_dir=cache_dir,
    )
    train = ds["train"].to_pandas()
    test = ds["test"].to_pandas()

    train = _parse_and_clean(train)
    test = _parse_and_clean(test)

    logger.info(f"Primary DRC: {len(train)} train, {len(test)} test compounds loaded.")
    return train, test


def load_counter_assay(cache_dir: Optional[str] = None) -> pd.DataFrame:
    """Load PXR-null counter-assay training data."""
    from datasets import load_dataset

    ds = load_dataset(
        "openadmet/pxr-challenge-train-test",
        "counter_assay",
        cache_dir=cache_dir,
    )
    df = ds["train"].to_pandas()
    df = _parse_and_clean(df)
    logger.info(f"Counter-assay: {len(df)} compounds loaded.")
    return df


def load_hts(cache_dir: Optional[str] = None) -> pd.DataFrame:
    """Load single-concentration HTS screening data."""
    from datasets import load_dataset

    ds = load_dataset(
        "openadmet/pxr-challenge-train-test",
        "single_concentration",
        cache_dir=cache_dir,
    )
    df = ds["train"].to_pandas()
    df = _parse_and_clean(df)
    logger.info(f"HTS screen: {len(df)} compounds loaded.")
    return df


# ---------------------------------------------------------------------------
# Post-load processing
# ---------------------------------------------------------------------------

def add_inverse_variance_weights(df: pd.DataFrame) -> pd.DataFrame:
    """Add sample_weight column = 1 / SE_pEC50^2 (clipped to [0.25, 100]).

    Compounds with very small SE (high-quality measurements) get higher weight.
    Missing SE values are replaced with the median SE.
    """
    df = df.copy()
    se_col = COL_PECSO_SE
    if se_col not in df.columns:
        df["sample_weight"] = 1.0
        return df

    se = df[se_col].copy()
    median_se = se[se > 0].median()
    se = se.fillna(median_se).clip(lower=0.05)  # floor at 0.05 to avoid huge weights
    df["sample_weight"] = (1.0 / se**2).clip(lower=0.25, upper=100.0)
    # Normalize to mean=1 so loss scale is stable
    df["sample_weight"] = df["sample_weight"] / df["sample_weight"].mean()
    return df


def flag_nonspecific_compounds(
    train_pxr: pd.DataFrame,
    counter_assay: pd.DataFrame,
    delta_threshold: float = 1.5,
) -> pd.DataFrame:
    """Flag compounds where PXR pEC50 is not clearly above the null-line pEC50.

    The official tutorial criterion: a compound is 'potent and selective' only when
    pEC50_PXR - pEC50_null >= 1.5. Compounds below this selectivity margin are flagged
    as potentially non-specific and downweighted during training.

    Adds 'is_nonspecific' bool column to train_pxr using OCNT_ID for matching.
    """
    train = train_pxr.copy()
    train["is_nonspecific"] = False

    if COL_ID not in train.columns or COL_ID not in counter_assay.columns:
        return train

    null_map = counter_assay.set_index(COL_ID)[COL_PECSO].to_dict()
    for idx, row in train.iterrows():
        ocnt_id = row.get(COL_ID)
        null_pecso = null_map.get(ocnt_id)
        if null_pecso is not None and not np.isnan(null_pecso):
            delta = row[COL_PECSO] - null_pecso
            if delta < delta_threshold:  # not directional: PXR must exceed null by >= 1.5
                train.at[idx, "is_nonspecific"] = True

    n_flagged = train["is_nonspecific"].sum()
    logger.info(
        f"Flagged {n_flagged} / {len(train)} training compounds as potentially "
        f"non-specific (pEC50_PXR - pEC50_null < {delta_threshold})."
    )
    return train


def merge_counter_assay_columns(
    train_pxr: pd.DataFrame,
    counter_assay: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join counter-assay pEC50/Emax columns onto primary DRC DataFrame.

    Adds columns: pEC50_null, Emax_null (NaN if not matched).
    """
    rename_map = {
        COL_PECSO: "pEC50_null",
        COL_EMAX: "Emax_null",
        COL_PECSO_SE: "pEC50_null_se",
    }
    ca_subset = counter_assay[[COL_ID] + [c for c in rename_map if c in counter_assay.columns]].copy()
    ca_subset = ca_subset.rename(columns=rename_map)
    merged = train_pxr.merge(ca_subset, on=COL_ID, how="left")
    return merged


def compute_hts_pec50_correlation(
    train_pxr: pd.DataFrame,
    hts: pd.DataFrame,
) -> float:
    """Compute Pearson r between neg_log10_fdr (HTS) and pEC50 on shared compounds.

    Note: neg_log10_fdr contains inf values when fdr_bh = 0. These are capped at
    the finite max + 1 before computing correlation rather than dropped, since they
    represent the most confidently active compounds.
    """
    shared = train_pxr[[COL_ID, COL_PECSO]].merge(
        hts[[COL_ID, COL_NEG_LOG_FDR]], on=COL_ID, how="inner"
    )
    if len(shared) < 10:
        logger.warning("Fewer than 10 shared compounds between HTS and primary DRC — cannot compute correlation.")
        return float("nan")

    # Cap inf values to finite_max + 1 (they represent the strongest hits)
    col = shared[COL_NEG_LOG_FDR].copy()
    finite_max = col[np.isfinite(col)].max()
    n_inf = np.isinf(col).sum()
    if n_inf > 0:
        col = col.replace([np.inf, -np.inf], finite_max + 1.0)
        logger.info(f"Capped {n_inf} inf values in neg_log10_fdr to {finite_max + 1.0:.2f}")

    from scipy.stats import pearsonr
    r, p = pearsonr(col, shared[COL_PECSO])
    logger.info(f"HTS neg_log10_fdr ↔ pEC50 Pearson r = {r:.3f} (p = {p:.2e}, n = {len(shared)})")
    return r


# ---------------------------------------------------------------------------
# Master loader
# ---------------------------------------------------------------------------

@dataclass
class PXRDataset:
    """Container for all data tiers."""
    train: pd.DataFrame = field(default_factory=pd.DataFrame)
    test: pd.DataFrame = field(default_factory=pd.DataFrame)
    counter_assay: pd.DataFrame = field(default_factory=pd.DataFrame)
    hts: pd.DataFrame = field(default_factory=pd.DataFrame)


def load_all_tiers(cache_dir: Optional[str] = "data/hf_cache") -> PXRDataset:
    """Load and preprocess all four dataset tiers.

    Parameters
    ----------
    cache_dir : path for HuggingFace dataset cache (keeps data/ local)

    Returns
    -------
    PXRDataset with all tiers loaded and augmented with derived columns.
    """
    train, test = load_primary_drc(cache_dir=cache_dir)
    counter = load_counter_assay(cache_dir=cache_dir)
    hts = load_hts(cache_dir=cache_dir)

    # Enrich training data
    train = add_inverse_variance_weights(train)
    train = flag_nonspecific_compounds(train, counter)
    train = merge_counter_assay_columns(train, counter)

    # Log HTS correlation as a quick sanity check (non-fatal)
    try:
        compute_hts_pec50_correlation(train, hts)
    except Exception as e:
        logger.warning(f"HTS correlation skipped: {e}")

    logger.info(
        f"Dataset loaded — train: {len(train)}, test: {len(test)}, "
        f"counter_assay: {len(counter)}, hts: {len(hts)}"
    )
    return PXRDataset(train=train, test=test, counter_assay=counter, hts=hts)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ds = load_all_tiers()
    print(ds.train.head())
    print(ds.train.columns.tolist())
