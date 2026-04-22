"""
Activity cliff (AC) detection and annotation for the PXR training set.

An Activity Cliff is defined as a pair of structurally similar compounds
(Tanimoto(ECFP4) ≥ similarity_threshold) with a large difference in potency
(|ΔpEC50| ≥ activity_threshold).

Provides:
  - identify_activity_cliffs(): enumerate all AC pairs
  - annotate_cliff_compounds(): add cliff metadata columns to training DataFrame
  - export_mmpdb_smiles(): write SMILES file for mmpdb MMP analysis
  - CliffSampler: yields AC pairs for contrastive loss training

Reference:
  Stumpfe & Bajorath (2012) Exploring Activity Cliffs in Medicinal Chemistry
  J. Med. Chem. 55, 2932-2942
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default thresholds (from MoleculeACE benchmark defaults)
DEFAULT_SIM_THRESHOLD = 0.7    # Tanimoto(ECFP4)
DEFAULT_ACTIVITY_THRESHOLD = 1.0  # |ΔpEC50| in log units


# ---------------------------------------------------------------------------
# Core AC identification
# ---------------------------------------------------------------------------

def identify_activity_cliffs(
    smiles: list[str],
    activities: np.ndarray,
    sim_threshold: float = DEFAULT_SIM_THRESHOLD,
    activity_threshold: float = DEFAULT_ACTIVITY_THRESHOLD,
    n_bits: int = 2048,
    batch_size: int = 500,
) -> pd.DataFrame:
    """Identify all activity cliff pairs in a compound set.

    Parameters
    ----------
    smiles         : list of canonical SMILES
    activities     : pEC50 values (same order as smiles)
    sim_threshold  : minimum Tanimoto similarity to consider as AC pair
    activity_threshold : minimum |ΔpEC50| to flag as AC

    Returns
    -------
    DataFrame with columns: idx_i, idx_j, smiles_i, smiles_j,
                             pec50_i, pec50_j, tanimoto, delta_pec50
    """
    from src.features.feature_engineering import ecfp4, tanimoto_matrix

    n = len(smiles)
    activities = np.array(activities, dtype=np.float64)
    fps = ecfp4(smiles, n_bits=n_bits)

    cliff_pairs = []

    # Process in batches to avoid memory issues for large datasets
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_fps = fps[start:end]
        # Only compute upper triangle: j > i (but we batch over i, compute vs. all j)
        sim_block = tanimoto_matrix(batch_fps, fps)  # (batch, n)

        for local_i, i in enumerate(range(start, end)):
            for j in range(i + 1, n):
                sim = sim_block[local_i, j]
                if sim < sim_threshold:
                    continue
                delta = abs(activities[i] - activities[j])
                if delta < activity_threshold:
                    continue
                cliff_pairs.append({
                    "idx_i": i,
                    "idx_j": j,
                    "smiles_i": smiles[i],
                    "smiles_j": smiles[j],
                    "pec50_i": float(activities[i]),
                    "pec50_j": float(activities[j]),
                    "tanimoto": float(sim),
                    "delta_pec50": float(delta),
                })

    df = pd.DataFrame(cliff_pairs)
    n_pairs = len(df)
    if n_pairs > 0:
        logger.info(
            f"Found {n_pairs} activity cliff pairs "
            f"(Tanimoto ≥ {sim_threshold}, |ΔpEC50| ≥ {activity_threshold}) "
            f"among {n} compounds."
        )
    else:
        logger.info("No activity cliff pairs found with the given thresholds.")
    return df


# ---------------------------------------------------------------------------
# Annotate training compounds
# ---------------------------------------------------------------------------

def annotate_cliff_compounds(
    train_df: pd.DataFrame,
    cliff_pairs: pd.DataFrame,
    smiles_col: str = "SMILES",
    activity_col: str = "pEC50",
) -> pd.DataFrame:
    """Add AC-related columns to the training DataFrame.

    Added columns:
    - is_cliff_member (bool): compound appears in ≥1 AC pair
    - n_cliff_pairs (int): number of AC pairs this compound belongs to
    - max_cliff_delta (float): largest |ΔpEC50| in any AC pair for this compound
    - cliff_neighbor_pec50 (float): pEC50 of most similar AC neighbor
    - cliff_sample_weight (float): 1 + 1.0 * is_cliff_member (for training)
    """
    df = train_df.copy()
    n = len(df)

    is_cliff = np.zeros(n, dtype=bool)
    n_pairs = np.zeros(n, dtype=int)
    max_delta = np.zeros(n, dtype=np.float64)
    neighbor_pec50 = np.full(n, np.nan, dtype=np.float64)

    if len(cliff_pairs) == 0:
        df["is_cliff_member"] = False
        df["n_cliff_pairs"] = 0
        df["max_cliff_delta"] = 0.0
        df["cliff_neighbor_pec50"] = np.nan
        df["cliff_sample_weight"] = 1.0
        return df

    for _, row in cliff_pairs.iterrows():
        i, j = int(row["idx_i"]), int(row["idx_j"])
        delta = row["delta_pec50"]

        for idx, other_pec50 in [(i, row["pec50_j"]), (j, row["pec50_i"])]:
            if idx >= n:
                continue
            is_cliff[idx] = True
            n_pairs[idx] += 1
            if delta > max_delta[idx]:
                max_delta[idx] = delta
                neighbor_pec50[idx] = other_pec50

    df["is_cliff_member"] = is_cliff
    df["n_cliff_pairs"] = n_pairs
    df["max_cliff_delta"] = max_delta
    df["cliff_neighbor_pec50"] = neighbor_pec50
    df["cliff_sample_weight"] = 1.0 + 1.0 * is_cliff.astype(float)

    n_cliff = is_cliff.sum()
    logger.info(
        f"Cliff annotation: {n_cliff} / {n} training compounds are cliff members "
        f"({100*n_cliff/n:.1f}%). Mean cliff_delta = {max_delta[is_cliff].mean():.2f} log units."
    )
    return df


# ---------------------------------------------------------------------------
# MMP database utilities
# ---------------------------------------------------------------------------

def export_mmpdb_smiles(
    df: pd.DataFrame,
    output_path: str = "data/training_smiles.smi",
    smiles_col: str = "SMILES",
    id_col: str = "OCNT_ID",
) -> str:
    """Write SMILES file in the format expected by mmpdb fragment.

    Format: <SMILES>\\t<ID>

    mmpdb usage after export:
        mmpdb fragment data/training_smiles.smi -o data/training_frags.db
        mmpdb index data/training_frags.db -o data/training_mmps.db
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for _, row in df.iterrows():
            f.write(f"{row[smiles_col]}\t{row[id_col]}\n")
    logger.info(f"Exported {len(df)} SMILES to {output_path} for mmpdb.")
    return output_path


# ---------------------------------------------------------------------------
# CliffSampler: yields AC pairs for contrastive loss
# ---------------------------------------------------------------------------

class CliffSampler:
    """Yields (smiles_i, smiles_j, delta_pec50) tuples for contrastive training.

    Used in the AC-aware Chemprop trainer to construct contrastive batches.
    """

    def __init__(
        self,
        cliff_pairs: pd.DataFrame,
        train_smiles: list[str],
        train_activities: np.ndarray,
    ):
        self.pairs = cliff_pairs
        self.smiles = train_smiles
        self.activities = np.array(train_activities)

    def sample(self, n: int, seed: Optional[int] = None) -> list[dict]:
        """Sample n AC pairs (with replacement if n > available pairs)."""
        rng = np.random.default_rng(seed)
        if len(self.pairs) == 0:
            return []
        idx = rng.integers(0, len(self.pairs), size=n)
        samples = []
        for i in idx:
            row = self.pairs.iloc[i]
            samples.append({
                "smiles_i": row["smiles_i"],
                "smiles_j": row["smiles_j"],
                "pec50_i": row["pec50_i"],
                "pec50_j": row["pec50_j"],
                "delta_pec50": row["delta_pec50"],
                "tanimoto": row["tanimoto"],
            })
        return samples

    def __len__(self) -> int:
        return len(self.pairs)


# ---------------------------------------------------------------------------
# Chemical space summary for EDA
# ---------------------------------------------------------------------------

def cliff_summary_stats(cliff_pairs: pd.DataFrame, train_df: pd.DataFrame) -> dict:
    """Return summary statistics about activity cliffs for EDA reporting."""
    if len(cliff_pairs) == 0:
        return {"n_cliff_pairs": 0, "n_cliff_compounds": 0, "cliff_fraction": 0.0}

    n_cliff_compounds = len(
        set(cliff_pairs["idx_i"].tolist() + cliff_pairs["idx_j"].tolist())
    )
    return {
        "n_cliff_pairs": len(cliff_pairs),
        "n_cliff_compounds": n_cliff_compounds,
        "cliff_fraction": n_cliff_compounds / len(train_df),
        "mean_tanimoto": float(cliff_pairs["tanimoto"].mean()),
        "mean_delta_pec50": float(cliff_pairs["delta_pec50"].mean()),
        "max_delta_pec50": float(cliff_pairs["delta_pec50"].max()),
        "min_delta_pec50": float(cliff_pairs["delta_pec50"].min()),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smiles = [
        "Cc1ccc(cc1)S(=O)(=O)N",
        "Cc1ccc(cc1)S(=O)(=O)NC",  # very similar to above
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "c1ccc(cc1)C(=O)O",
    ]
    activities = np.array([6.5, 4.5, 5.2, 4.8])  # first two form an AC pair

    pairs = identify_activity_cliffs(smiles, activities, sim_threshold=0.5, activity_threshold=1.0)
    print(pairs)
