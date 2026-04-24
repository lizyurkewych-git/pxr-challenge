"""
HTS dose-response fitting and pseudo-pEC50 generation for Chemprop pre-training.

Converts the 4-concentration HTS screen data into pseudo-pEC50 estimates by
fitting a Hill sigmoidal model to each compound's concentration-response profile.
The resulting pseudo-pEC50 values are used to pre-train Chemprop on PXR-screened
compounds before fine-tuning on the 4,139 primary DRC compounds.

Usage:
    from src.models.hts_pretrain import prepare_hts_pretrain_data

    pretrain_df = prepare_hts_pretrain_data(hts_df)
    # pretrain_df: DataFrame with SMILES, pseudo_pec50, fit_quality
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

COL_SMILES = "SMILES"
COL_CONC = "concentration_M"
COL_LOG2FC = "log2_fc_estimate"
COL_NEG_LOG_FDR = "neg_log10_fdr"


def _hill(C: np.ndarray, rmax: float, ec50: float, n: float = 1.5) -> np.ndarray:
    Cn = np.power(np.maximum(C, 1e-20), n)
    ec50n = pow(max(ec50, 1e-20), n)
    return rmax * Cn / (ec50n + Cn)


def _fit_single_compound(
    concs: np.ndarray,
    responses: np.ndarray,
    n_fixed: float = 1.5,
    min_r2: float = 0.5,
    pec50_range: tuple[float, float] = (3.5, 9.0),
) -> Optional[float]:
    """Fit 2-parameter Hill model to one compound's dose-response.

    Returns pseudo_pEC50 (float) or None if fit fails or compound is inactive.
    """
    from scipy.optimize import curve_fit

    concs = np.asarray(concs, dtype=np.float64)
    responses = np.asarray(responses, dtype=np.float64)

    valid = np.isfinite(responses) & np.isfinite(concs) & (concs > 0)
    concs, responses = concs[valid], responses[valid]
    if len(concs) < 2:
        return None

    if np.max(responses) < 0.3:
        return None

    try:
        popt, _ = curve_fit(
            lambda C, rmax, ec50: _hill(C, rmax, ec50, n=n_fixed),
            concs, responses,
            p0=[float(np.max(responses)), float(np.median(concs))],
            bounds=([0.01, 1e-11], [50.0, 1e-2]),
            maxfev=10000,
        )
        rmax_fit, ec50_fit = popt

        r_pred = _hill(concs, rmax_fit, ec50_fit, n=n_fixed)
        ss_res = np.sum((responses - r_pred) ** 2)
        ss_tot = np.sum((responses - responses.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0

        if r2 < min_r2:
            return None

        pec50 = -np.log10(ec50_fit)
        if not (pec50_range[0] <= pec50 <= pec50_range[1]):
            return None

        return float(pec50)
    except Exception:
        return None


def prepare_hts_pretrain_data(
    hts_df: pd.DataFrame,
    primary_train: Optional[pd.DataFrame] = None,
    min_r2: float = 0.5,
    pec50_range: tuple[float, float] = (3.5, 9.0),
) -> pd.DataFrame:
    """Convert multi-concentration HTS data to pseudo-pEC50 for Chemprop pre-training.

    Fits a Hill model to each compound's dose-response profile. Compounds where
    the fit fails or potency falls outside pec50_range are dropped rather than
    imputed, keeping the pre-training signal clean.

    If primary_train is provided, compounds already in the primary DRC set are
    replaced with their true pEC50 values (higher quality signal).

    Returns DataFrame with columns: SMILES, pseudo_pec50, fit_quality ('hill' | 'drc')
    """
    if COL_CONC not in hts_df.columns:
        raise ValueError(
            f"'{COL_CONC}' column not found. Available: {list(hts_df.columns)}"
        )

    response_col = COL_LOG2FC if COL_LOG2FC in hts_df.columns else COL_NEG_LOG_FDR
    logger.info(f"HTS response column: {response_col}")

    grouped = hts_df.groupby(COL_SMILES)
    n_total = grouped.ngroups
    logger.info(f"HTS: {len(hts_df)} rows, {n_total} unique compounds")

    results = []
    n_hill = n_inactive = n_failed = 0

    for smiles, group in grouped:
        group = group.sort_values(COL_CONC)
        concs = group[COL_CONC].values
        resps = group[response_col].values.copy().astype(float)

        # Cap ±inf (fdr_bh=0 case → neg_log10_fdr = inf)
        finite_max = np.nanmax(resps[np.isfinite(resps)]) if np.any(np.isfinite(resps)) else 0.0
        resps = np.where(np.isposinf(resps), finite_max + 1.0, resps)
        resps = np.where(np.isneginf(resps), 0.0, resps)

        pec50 = _fit_single_compound(concs, resps, min_r2=min_r2, pec50_range=pec50_range)

        if pec50 is not None:
            results.append({"SMILES": smiles, "pseudo_pec50": pec50, "fit_quality": "hill"})
            n_hill += 1
        elif np.max(resps[np.isfinite(resps)]) < 0.3 if np.any(np.isfinite(resps)) else True:
            n_inactive += 1
        else:
            n_failed += 1

    logger.info(
        f"Hill fit accepted: {n_hill} | inactive (dropped): {n_inactive} | "
        f"active but poor fit (dropped): {n_failed}"
    )

    df = pd.DataFrame(results)
    if df.empty:
        logger.warning("No compounds passed Hill fitting — check HTS data format.")
        return df

    # Replace with true pEC50 for compounds in primary DRC (higher quality)
    if primary_train is not None and "pEC50" in primary_train.columns:
        drc_map = primary_train.set_index(COL_SMILES)["pEC50"].to_dict()
        mask = df[COL_SMILES].isin(drc_map)
        df.loc[mask, "pseudo_pec50"] = df.loc[mask, COL_SMILES].map(drc_map)
        df.loc[mask, "fit_quality"] = "drc"
        logger.info(
            f"Replaced {mask.sum()} HTS compounds with true DRC pEC50 values."
        )

    logger.info(
        f"Pre-training dataset: {len(df)} compounds | "
        f"pEC50 range: [{df['pseudo_pec50'].min():.2f}, {df['pseudo_pec50'].max():.2f}] | "
        f"mean: {df['pseudo_pec50'].mean():.2f}"
    )
    return df
