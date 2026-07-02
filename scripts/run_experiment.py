"""
Hill climb experiment runner.

Usage:
    python scripts/run_experiment.py configs/exp001.json
    python scripts/run_experiment.py configs/exp001.json --dry-run

Always writes logs/{exp_id}_status.json so you can check progress at any time:

    cat logs/exp001_status.json

Status lifecycle:
    {"status": "running", "stage": "cv_fold_2_of_5", ...}
    {"status": "done",    "cv_rae": 0.51, "accepted": true, ...}
    {"status": "failed",  "error": "...", "traceback": "...", ...}

A "failed" status with the full traceback means something errored — the instance
is NOT silently running; check the log file and status file before assuming it's
working.

ntfy push notifications (optional — if NTFY_TOPIC is not set, skipped silently):
    export NTFY_TOPIC=my-pxr-topic
    Sends to https://ntfy.sh/$NTFY_TOPIC on completion or failure.
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "0")

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Reactive electrophile filter (discoverybytes attribution: +0.019 RAE)
# ---------------------------------------------------------------------------

_ELECTROPHILE_SMARTS = [
    "[CH2]=[CH]C(=O)[NX3]",   # acrylamide
    "[CH2]=[CH]C(=O)[OX2]",   # acrylate
    "[CX3H1](=O)[#6]",         # aldehyde (non-formaldehyde)
]


def _filter_reactive_electrophiles(df, smiles_col: str = "SMILES"):
    """Return (filtered_df, n_removed). Removes covalent binders unrelated to PXR activity."""
    from rdkit import Chem
    patterns = [Chem.MolFromSmarts(s) for s in _ELECTROPHILE_SMARTS]
    keep = []
    for smi in df[smiles_col]:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            keep.append(True)
            continue
        keep.append(not any(mol.HasSubstructMatch(p) for p in patterns))
    import numpy as _np_ef
    mask = _np_ef.array(keep)
    return df[mask].reset_index(drop=True), int((~mask).sum())


# ---------------------------------------------------------------------------
# HTChem training augmentation
# ---------------------------------------------------------------------------

# OCNT-2395416 is an exact match with Analog Set 1 holdout (Tc=1.0); exclude unconditionally.
_HTCHEM_HOLDOUT_EXCLUSIONS = {"OCNT-2395416"}


def _load_htchem_augmentation(config: dict, train_smiles_set: set, holdout_smiles_set: set):
    """Load HTChem crude/semi-pure data for fold-time training augmentation.

    Reads from data/htchem_clean.csv — a pre-cleaned file committed to git containing
    440 compounds (347 crude + 93 semi_pure) with canonical SMILES and pEC50 values.
    OCNT-2395416 and null-pEC50 rows were removed at CSV-generation time.

    At training time this function only deduplicates against the current train+holdout sets,
    which is fast (set lookup) and must happen per-run since those sets change per experiment.

    Returns (smiles_list, y_array). Returns empty lists if htchem_augmentation.enabled=False.
    HTChem Emax is on a different scale (0-1 normalized) — treat as NaN for multi-task Task 2.
    """
    import numpy as _np_ht
    htchem_cfg = config.get("htchem_augmentation", {})
    if not htchem_cfg.get("enabled", False):
        return [], _np_ht.array([], dtype=_np_ht.float32)

    import pandas as _pd
    from pathlib import Path as _Path

    subsets = set(htchem_cfg.get("subsets", ["crude", "semi_pure"]))

    csv_path = _Path(config.get("htchem_csv", "data/htchem_clean.csv"))
    if not csv_path.exists():
        raise FileNotFoundError(
            f"HTChem data file not found at {csv_path}. "
            "Ensure data/htchem_clean.csv is present in the repository. "
            "Re-generate with: python scripts/generate_htchem_csv.py"
        )

    df = _pd.read_csv(csv_path)
    df = df[df["subset"].isin(subsets)].copy()

    pec50_min_cutoff = htchem_cfg.get("pec50_min_cutoff", None)
    if pec50_min_cutoff is not None:
        n_before = len(df)
        df = df[df["pec50"] >= float(pec50_min_cutoff)].copy()
        logging.getLogger("runner").info(
            f"HTChem pEC50 cutoff >= {pec50_min_cutoff}: {n_before} → {len(df)} rows retained"
        )

    all_smiles, all_y = [], []
    seen_canon = set(train_smiles_set) | set(holdout_smiles_set)
    n_dedup = 0

    for _, row in df.iterrows():
        canon = row["smiles"]
        if canon in seen_canon:
            n_dedup += 1
            continue
        seen_canon.add(canon)
        all_smiles.append(canon)
        all_y.append(float(row["pec50"]))

    logging.getLogger("runner").info(
        f"HTChem augmentation: {len(df)} candidates from {sorted(subsets)} → "
        f"{n_dedup} deduped against train/holdout → {len(all_smiles)} added"
    )

    if not all_smiles:
        logging.getLogger("runner").warning(
            "HTChem augmentation enabled but 0 compounds survived deduplication. "
            "Training proceeds without augmentation — results are NOT comparable to "
            "a true augmented run. Check for unexpected overlap with train/holdout sets."
        )

    return all_smiles, _np_ht.array(all_y, dtype=_np_ht.float32)


# ---------------------------------------------------------------------------
# PMI + 3D shape features for chemprop_mt x_d (discoverybytes attribution: +0.005–0.010 RAE)
# ---------------------------------------------------------------------------

_PMI_FEATURE_NAMES = [
    "PMI1", "PMI2", "PMI3", "NPR1", "NPR2",
    "Asphericity", "Eccentricity", "InertialShapeFactor",
    "SpherocityIndex", "RadiusOfGyration", "PBF",
]
_N_PMI = len(_PMI_FEATURE_NAMES)


def _compute_pmi_features(smiles: list) -> "np.ndarray":
    """Compute 11 3D shape descriptors via RDKit MMFF conformer. Failures get zeros."""
    import numpy as _np_pmi
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors3D

    features = []
    n_failed = 0
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            features.append(_np_pmi.zeros(_N_PMI, dtype=_np_pmi.float32))
            n_failed += 1
            continue
        try:
            mol_h = Chem.AddHs(mol)
            ps = AllChem.ETKDGv3()
            ps.randomSeed = 42
            result = AllChem.EmbedMolecule(mol_h, ps)
            if result != 0:
                result = AllChem.EmbedMolecule(mol_h, randomSeed=42)
            if result != 0:
                features.append(_np_pmi.zeros(_N_PMI, dtype=_np_pmi.float32))
                n_failed += 1
                continue
            AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200)
            row = _np_pmi.array([
                Descriptors3D.PMI1(mol_h), Descriptors3D.PMI2(mol_h), Descriptors3D.PMI3(mol_h),
                Descriptors3D.NPR1(mol_h), Descriptors3D.NPR2(mol_h),
                Descriptors3D.Asphericity(mol_h), Descriptors3D.Eccentricity(mol_h),
                Descriptors3D.InertialShapeFactor(mol_h), Descriptors3D.SpherocityIndex(mol_h),
                Descriptors3D.RadiusOfGyration(mol_h), Descriptors3D.PBF(mol_h),
            ], dtype=_np_pmi.float32)
            features.append(_np_pmi.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0))
        except Exception:
            features.append(_np_pmi.zeros(_N_PMI, dtype=_np_pmi.float32))
            n_failed += 1

    logging.getLogger("runner").info(f"PMI features: {len(smiles) - n_failed}/{len(smiles)} succeeded")
    return _np_pmi.array(features, dtype=_np_pmi.float32)


# ---------------------------------------------------------------------------
# Status file — atomic writes so you never read a partial file
# ---------------------------------------------------------------------------

class StatusWriter:
    """Writes logs/{exp_id}_status.json atomically at every stage."""

    def __init__(self, path: Path, exp_id: str, config: dict):
        self.path = path
        self._s = {
            "experiment_id": exp_id,
            "status": "running",
            "started_at": _now(),
            "updated_at": _now(),
            "stage": "initializing",
            "description": config.get("description", ""),
            "device": config.get("device", "cpu"),
            "baseline_rae": config.get("baseline_rae"),
            "cv_rae": None,
            "improvement": None,
            "accepted": None,
            "coefs": None,
            "runtime_seconds": None,
            "submission_path": None,
            "error": None,
            "log_file": str(path.parent / f"{exp_id}.log"),
            "results_file": str(path.parent / f"{exp_id}_results.json"),
        }
        self._flush()

    def update(self, **kwargs):
        self._s.update(kwargs)
        self._s["updated_at"] = _now()
        self._flush()

    def _flush(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._s, indent=2))
        tmp.replace(self.path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _elapsed(t0: float) -> float:
    return round(time.time() - t0, 1)


# ---------------------------------------------------------------------------
# ntfy push notification
# ---------------------------------------------------------------------------

def send_ntfy(topic: str, title: str, message: str, priority: str = "default"):
    """POST to ntfy.sh. Silently ignores all errors — notifications are optional."""
    try:
        import urllib.request
        # HTTP headers are ASCII-only; encode title safely and put emoji in body
        safe_title = title.encode("ascii", errors="ignore").decode("ascii").strip()
        req = urllib.request.Request(
            f"https://ntfy.sh/{topic}",
            data=message.encode("utf-8"),
            method="POST",
            headers={
                "Title": safe_title,
                "Priority": priority,
                "Content-Type": "text/plain; charset=utf-8",
            },
        )
        urllib.request.urlopen(req, timeout=10)
        logging.getLogger("runner").info(f"ntfy sent: {title}")
    except Exception as exc:
        logging.getLogger("runner").warning(f"ntfy send skipped: {exc}")


# ---------------------------------------------------------------------------
# OOF / test-prediction checkpoint helpers
# ---------------------------------------------------------------------------

def _load_oof(source: str, model_names: list, train_y_list: list, remap: dict = None):
    """Load reusable OOF predictions. Returns (oof_dict, fold_raes_dict).

    remap maps target key → source key, e.g. {"unimol_s42": "unimol"} loads
    exp008's single "unimol" entry under the new per-seed name.
    """
    import numpy as np
    remap = remap or {}
    p = Path(source)
    if not p.exists():
        logging.getLogger("runner").warning(f"OOF checkpoint not found: {source}")
        return {}, {}
    data = json.loads(p.read_text())
    if data.get("train_y") != train_y_list:
        logging.getLogger("runner").warning(
            f"OOF checkpoint train_y mismatch — recomputing: {source}"
        )
        return {}, {}
    oof, fold_raes = {}, {}
    for m in model_names:
        src_key = remap.get(m, m)
        if src_key in data:
            oof[m] = np.array(data[src_key], dtype=np.float64)
            fold_raes[m] = data.get("fold_raes", {}).get(src_key, [])
            label = f"{m} (remapped from '{src_key}')" if src_key != m else m
            logging.getLogger("runner").info(f"  Reused OOF: {label} ← {source}")
    return oof, fold_raes


def _load_test_cache(source: str, model_names: list, test_smiles: list, remap: dict = None):
    """Load cached test predictions. Returns dict or {} on mismatch.

    remap maps target key → source key, e.g. {"tabpfn": "tabpfn_aug"}.
    """
    import numpy as np
    remap = remap or {}
    p = Path(source)
    if not p.exists():
        return {}
    data = json.loads(p.read_text())
    if data.get("test_smiles") != test_smiles:
        logging.getLogger("runner").warning(f"Test cache SMILES mismatch: {source}")
        return {}
    out = {}
    for m in model_names:
        src_key = remap.get(m, m)
        if src_key in data:
            out[m] = np.array(data[src_key], dtype=np.float32)
            label = f"{m} (remapped from '{src_key}')" if src_key != m else m
            logging.getLogger("runner").info(f"  Reused test preds: {label} ← {source}")
    return out


# ---------------------------------------------------------------------------
# Competition progress log
# ---------------------------------------------------------------------------

COMPETITION_LOG = Path("logs/competition_log.jsonl")


def _append_competition_log(config: dict, sw_state: dict, runtime_hours: float):
    """Append one line to competition_log.jsonl after an experiment finishes."""
    ct = config.get("compute_tracking", {})
    hourly = ct.get("hourly_cost_usd", 0.0)
    coefs = sw_state.get("coefs") or {}

    # Derive model lists from config
    mc = config.get("models", {})
    hts_cfg = mc.get("chemprop_hts", {})
    seeds = hts_cfg.get("seeds", [42, 7, 13])
    all_models = []
    for name in ["delta", "chemprop_scratch"]:
        if mc.get(name, {}).get("enabled", False):
            all_models.append(name)
    if hts_cfg.get("enabled", False):
        all_models.extend(f"chemprop_hts_s{s}" for s in seeds)
    _mt_cfg_log = mc.get("chemprop_mt", {})
    if _mt_cfg_log.get("enabled", False):
        _mt_seeds_log = _mt_cfg_log.get("seeds", [42, 7])
        all_models.extend(f"chemprop_mt_s{s}" for s in _mt_seeds_log)
    if mc.get("tabpfn", {}).get("enabled", False):
        all_models.append("tabpfn")
    if mc.get("rf", {}).get("enabled", False):
        all_models.append("rf")
    if mc.get("gp", {}).get("enabled", False):
        all_models.append("gp")
    _um_cfg = mc.get("unimol", {})
    if _um_cfg.get("enabled", False):
        _um_seeds = _um_cfg.get("seeds")
        if _um_seeds:
            all_models.extend(f"unimol_s{s}" for s in _um_seeds)
        else:
            all_models.append("unimol")
    if mc.get("spherenet", {}).get("enabled", False):
        all_models.append("spherenet")
    if mc.get("tabicl", {}).get("enabled", False):
        all_models.append("tabicl")
    reuse_names = config.get("reuse_oof", {}).get("models", [])
    models_reused = [m for m in reuse_names if m in all_models]
    models_trained = [m for m in all_models if m not in models_reused]

    positive_coefs = {m: v for m, v in coefs.items() if v > 0}
    dominant_model = max(positive_coefs, key=positive_coefs.get) if positive_coefs else None
    models_zeroed = [m for m, v in coefs.items() if v <= 0]

    device = config.get("device", "cpu")
    phase = "gpu" if device == "cuda" else "local"

    entry = {
        "experiment_id": config["experiment_id"],
        "submission_number": None,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "phase": phase,
        "description": config.get("description", ""),
        "hypothesis": ct.get("hypothesis", ""),
        "device": device,
        "instance_type": ct.get("instance_type", "local"),
        "hourly_cost_usd": hourly,
        "runtime_hours": runtime_hours,
        "compute_cost_usd": round(hourly * runtime_hours, 2),
        "models_trained": models_trained,
        "models_reused": models_reused,
        "models_zeroed": models_zeroed,
        "ensemble_size": len(all_models),
        "dominant_model": dominant_model,
        "elasticnet_coefs": coefs,
        "cv_rae": sw_state.get("cv_rae"),
        "lb_rae": None,
        "lb_rank": None,
        "lb_field_size": None,
        "submitted": sw_state.get("submission_path") is not None,
        "accepted_cv": sw_state.get("accepted"),
        "outcome": None,
        "key_insight": None,
        "notes": sw_state.get("status", ""),
    }

    COMPETITION_LOG.parent.mkdir(exist_ok=True)
    with open(COMPETITION_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
    logging.getLogger("runner").info(f"Competition log updated → {COMPETITION_LOG}")


# ---------------------------------------------------------------------------
# Core experiment logic
# ---------------------------------------------------------------------------

def run_experiment(config: dict, sw: StatusWriter, t0: float, dry_run: bool = False, restack: bool = False):
    """Run one hill climb experiment. Raises on unrecoverable error."""
    import numpy as np

    logger = logging.getLogger("runner")
    exp_id = config["experiment_id"]
    device = config.get("device", "cuda")
    baseline_rae = config.get("baseline_rae", 0.5297)
    accept_thresh = config.get("acceptance_threshold", 0.015)
    model_cfg = config.get("models", {})

    if dry_run:
        logger.info("DRY RUN — skipping all model training")
        time.sleep(1)
        sw.update(stage="dry_run_complete", cv_rae=0.9999, accepted=False,
                  runtime_seconds=1.0, status="done")
        return

    # Auto-restack: if the config requests it and a completed OOF checkpoint exists,
    # treat this run as --restack so CV is skipped and full training resumes from where
    # it left off. This makes any retry (RETRY_INSTANCE or manual re-run) fault-tolerant
    # against mid-full-training crashes without repeating expensive CV.
    if not restack and config.get("restack_on_retry", False):
        _oof_chk = Path("logs") / f"{exp_id}_oof_checkpoint.json"
        if _oof_chk.exists():
            logger.info(
                f"restack_on_retry: OOF checkpoint found at {_oof_chk} — "
                "enabling --restack automatically (CV already complete)."
            )
            restack = True

    # ------------------------------------------------------------------
    # GPU health check — fast fail before spending any compute
    # ------------------------------------------------------------------
    if device == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError(
                "device=cuda in config but torch.cuda.is_available() is False. "
                "Check that the venv has a CUDA-enabled PyTorch install."
            )
        gpu_name = torch.cuda.get_device_name(0)
        gpu_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU verified: {gpu_name}, {gpu_gb:.1f} GB VRAM")

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    sw.update(stage="loading_data")
    from src.data.load_data import load_all_tiers, COL_SMILES, COL_PECSO

    logger.info("Loading dataset...")
    include_phase1 = config.get("include_phase1", False)
    ds = load_all_tiers(cache_dir="data/hf_cache", include_phase1=include_phase1)
    train, test, hts = ds.train, ds.test, ds.hts

    if config.get("filter_electrophiles", False):
        n_before = len(train)
        train, n_removed = _filter_reactive_electrophiles(train, COL_SMILES)
        logger.info(f"Electrophile filter: removed {n_removed} / {n_before} compounds → {len(train)} remain")

    # Activity cliffs (needed by delta model)
    delta_cfg = model_cfg.get("delta", {})
    if delta_cfg.get("enabled", False):
        try:
            from src.data.cliff_analysis import (
                identify_activity_cliffs, annotate_cliff_compounds,
            )
            cliff_pairs = identify_activity_cliffs(
                smiles=train[COL_SMILES].tolist(),
                activities=train[COL_PECSO].to_numpy(),
                sim_threshold=0.7, activity_threshold=1.0,
            )
            train = annotate_cliff_compounds(train, cliff_pairs)
            logger.info(f"Activity cliffs: {len(cliff_pairs)} pairs")
        except Exception as e:
            logger.warning(f"Cliff analysis failed, continuing without it: {e}")

    train_smiles = train[COL_SMILES].tolist()
    train_y = train[COL_PECSO].to_numpy()
    test_smiles = test[COL_SMILES].tolist()
    train_mean = float(train_y.mean())
    train_y_list = train_y.tolist()
    logger.info(f"Train: {len(train)} | Test: {len(test)} | HTS: {len(hts)}")

    # When include_phase1=False, load Phase 1 as a held-out evaluation set.
    # These 253 compounds are structurally similar to the Phase 2 test set,
    # so holdout RAE is a better generalization signal than Butina CV.
    holdout_smiles: Optional[list] = None
    holdout_y: Optional[np.ndarray] = None
    if not include_phase1:
        from src.data.load_data import load_phase1_unblinded
        holdout_df = load_phase1_unblinded(cache_dir="data/hf_cache")
        if config.get("filter_electrophiles", False):
            holdout_df, _ = _filter_reactive_electrophiles(holdout_df, COL_SMILES)
        holdout_smiles = holdout_df[COL_SMILES].tolist()
        holdout_y = holdout_df[COL_PECSO].to_numpy()
        logger.info(
            f"Holdout (Analog Set 1): {len(holdout_smiles)} compounds, "
            f"pEC50 {holdout_y.min():.2f}–{holdout_y.max():.2f}"
        )

    # HTChem augmentation data loaded here; fingerprints/PMI computed below after ecfp4.
    htchem_smiles: list = []
    htchem_y: np.ndarray = np.array([], dtype=np.float32)
    htchem_fps = None
    htchem_pmi = None
    _htchem_aug_cfg = config.get("htchem_augmentation", {})
    _htchem_aug_models = set(_htchem_aug_cfg.get("models", ["chemprop_mt", "delta"]))
    _ht_weight: float = float(_htchem_aug_cfg.get("crude_weight", 1.0))
    if _htchem_aug_cfg.get("enabled", False):
        htchem_smiles, htchem_y = _load_htchem_augmentation(
            config, set(train_smiles), set(holdout_smiles or [])
        )
        logger.info(
            f"HTChem augmentation: {len(htchem_smiles)} compounds ready "
            f"(crude_weight={_ht_weight})"
        )

    # Multi-task target matrix for chemprop_mt (pEC50 + pEC50_null + Emax).
    # NaN columns are masked in ChempropModel loss — safe to build unconditionally.
    from src.data.load_data import COL_EMAX
    _y_null = (
        train["pEC50_null"].to_numpy(dtype=np.float64)
        if "pEC50_null" in train.columns
        else np.full(len(train), np.nan, dtype=np.float64)
    )
    _y_emax = (
        train[COL_EMAX].to_numpy(dtype=np.float64)
        if COL_EMAX in train.columns
        else np.full(len(train), np.nan, dtype=np.float64)
    )
    y_mt = np.column_stack([train_y, _y_null, _y_emax]).astype(np.float32)
    logger.info(
        f"Multi-task targets: pEC50={len(train)}, "
        f"pEC50_null={(~np.isnan(_y_null)).sum()} avail, "
        f"Emax={(~np.isnan(_y_emax)).sum()} avail"
    )

    # Sample weights
    w = (
        train["sample_weight"].values.copy()
        if "sample_weight" in train.columns
        else np.ones(len(train))
    )
    if "is_nonspecific" in train.columns:
        w[train["is_nonspecific"].values] *= 0.3
    if config.get("ionizable_downweight", False):
        from src.features.feature_engineering import permeability_flags
        perm = permeability_flags(train_smiles)
        mask = perm[:, 0].astype(bool) & (train_y < 5.0)
        w[mask] *= 0.5
        logger.info(f"Ionizable downweighting: {mask.sum()} compounds → 0.5×")
    sample_weights = w.astype(np.float32)

    # ------------------------------------------------------------------
    # 2. ChEMBL + HTS pretraining
    # ------------------------------------------------------------------
    from src.models.chemprop_model import ChempropModel

    chemprop_hts_cfg = model_cfg.get("chemprop_hts", {})
    pretrain_stages = chemprop_hts_cfg.get("pretrain_stages", ["hts"])

    # Fast path: load pre-computed 3-stage pretrain checkpoint if available.
    # Generated once by `python -m hillclimb --make-checkpoint` (Tox21→ChEMBL→HTS).
    # Saves ~30 min of pretrain per trial; all inline pretrain stages are skipped.
    _pretrain_ckpt_path = config.get("chemprop_mt_pretrain_ckpt")
    hts_state_dict = None
    tox21_state_dict = None
    chembl_state_dict = None
    if _pretrain_ckpt_path:
        if Path(_pretrain_ckpt_path).exists():
            import torch as _torch
            hts_state_dict = _torch.load(_pretrain_ckpt_path, map_location="cpu")
            logger.info(
                f"Pretrain checkpoint loaded from {_pretrain_ckpt_path} "
                "— skipping all inline pretrain stages."
            )
        else:
            raise FileNotFoundError(
                f"chemprop_mt_pretrain_ckpt='{_pretrain_ckpt_path}' not found. "
                "Results without the pretrain checkpoint are not comparable to "
                "hill-climb experiments and must not be used for ablation decisions. "
                "Ensure logs/gpu/hts_pretrain_checkpoint.pt exists locally so the "
                "orchestrator uploads it before the experiment runs. "
                "Re-generate with: python -m hillclimb --make-checkpoint"
            )

    # Stage 0: Tox21 nuclear receptor panel pretraining (skipped if checkpoint loaded)
    if hts_state_dict is not None:
        pass  # checkpoint already provides the initialisation
    else:
        tox21_state_dict = None
    if hts_state_dict is None and chemprop_hts_cfg.get("enabled", False) and "tox21" in pretrain_stages:
        tox21_path = Path(config.get("tox21_data_path", "data/tox21_nr.csv"))
        if tox21_path.exists():
            sw.update(stage="stage0_tox21_pretrain")
            logger.info("\n=== Stage 0: Tox21 nuclear receptor pretraining ===")
            import pandas as pd
            df_tox = pd.read_csv(tox21_path)
            train_set = set(train_smiles)
            df_tox = df_tox[~df_tox["smiles"].isin(train_set)]
            logger.info(f"Tox21 NR after DRC overlap removal: {len(df_tox)} records")
            tm = ChempropModel(
                epochs=chemprop_hts_cfg.get("tox21_epochs", 40),
                hidden_size=300, depth=3, ffn_num_layers=3,
                dropout=0.1, batch_size=64, lr=1e-3, device=device,
                snapshot_epochs=5, extra_features=False, seed=42, n_tasks=1,
            )
            tm.fit(df_tox["smiles"].tolist(), df_tox["activity"].to_numpy(dtype=np.float64))
            tox21_state_dict = tm.get_state_dict()
            del tm; gc.collect()
            logger.info("Stage 0 complete.")
        else:
            raise FileNotFoundError(
                f"Tox21 data not found at {tox21_path}. "
                "Run: python scripts/fetch_tox21.py"
            )

    if hts_state_dict is None and chemprop_hts_cfg.get("enabled", False) and "chembl" in pretrain_stages:
        chembl_path = config.get("chembl_data_path", "data/chembl_pxr.csv")
        if Path(chembl_path).exists():
            sw.update(stage="stage1_chembl_pretrain")
            logger.info("\n=== Stage 1: ChEMBL PXR pretraining ===")
            import pandas as pd
            df_c = pd.read_csv(chembl_path)
            train_set = set(train_smiles)
            df_c = df_c[~df_c["smiles"].isin(train_set)]
            logger.info(f"ChEMBL after DRC overlap removal: {len(df_c)} compounds")
            cm = ChempropModel(
                epochs=chemprop_hts_cfg.get("chembl_epochs", 80),
                hidden_size=300, depth=3, ffn_num_layers=3,
                dropout=0.1, batch_size=64, lr=1e-3, device=device,
                snapshot_epochs=5, extra_features=False, seed=42, n_tasks=1,
            )
            cm.fit(df_c["smiles"].tolist(), df_c["pchembl_value"].to_numpy(dtype=np.float64))
            chembl_state_dict = cm.get_state_dict()
            del cm; gc.collect()
            logger.info("Stage 1 complete.")
        else:
            raise FileNotFoundError(
                f"ChEMBL data not found at {chembl_path}. "
                "Run: python scripts/fetch_chembl_pxr.py"
            )

    # Skip HTS pretraining if all chemprop_hts OOFs will be reused from cache —
    # defer to full training phase so local/CV-only runs don't pay the GPU cost.
    _reuse_cfg = config.get("reuse_oof", {})
    _hts_seeds_check = chemprop_hts_cfg.get("seeds", [42, 7, 13])
    _all_hts_reused = (
        chemprop_hts_cfg.get("enabled", False) and
        len(_hts_seeds_check) > 0 and   # empty seeds → vacuous all() → never skip pretrain
        all(f"chemprop_hts_s{s}" in _reuse_cfg.get("models", []) for s in _hts_seeds_check) and
        Path(_reuse_cfg.get("source", "__missing__")).exists()
    )

    if hts_state_dict is None and chemprop_hts_cfg.get("enabled", False) and "hts" in pretrain_stages and not _all_hts_reused:
        sw.update(stage="stage2_hts_pretrain")
        logger.info("\n=== Stage 2: HTS concentration-aware pretraining ===")
        from src.models.hts_pretrain import prepare_hts_concentration_data
        pt_sm, pt_y, pt_xd = prepare_hts_concentration_data(hts_df=hts, primary_train=train)
        logger.info(f"HTS pretraining on {len(pt_sm)} rows")
        hm = ChempropModel(
            epochs=chemprop_hts_cfg.get("hts_epochs", 60),
            hidden_size=300, depth=3, ffn_num_layers=3,
            dropout=0.1, batch_size=64, lr=1e-3, device=device,
            snapshot_epochs=5, extra_features=False, seed=42, n_tasks=1,
        )
        hm.fit(pt_sm, pt_y, x_d=pt_xd, init_state_dict=tox21_state_dict or chembl_state_dict)
        hts_state_dict = hm.get_state_dict()
        del hm, tox21_state_dict, chembl_state_dict; gc.collect()
        logger.info("Stage 2 complete.")
    elif _all_hts_reused:
        logger.info("Stage 2: HTS pretraining skipped — all chemprop_hts OOFs will be reused.")

    # ------------------------------------------------------------------
    # 3. Feature matrices (RF)
    # ------------------------------------------------------------------
    from src.features.feature_engineering import ecfp4, FeaturePipeline

    rf_cfg = model_cfg.get("rf", {})
    fp_train = ecfp4(train_smiles)
    fp_test = ecfp4(test_smiles)
    fp_holdout = ecfp4(holdout_smiles) if holdout_smiles is not None else None
    if htchem_smiles:
        htchem_fps = ecfp4(htchem_smiles)
    X_train = X_test = X_holdout = X_htchem = None
    tabicl_extra_train: Optional[np.ndarray] = None
    tabicl_extra_test: Optional[np.ndarray] = None
    tabicl_extra_holdout: Optional[np.ndarray] = None
    if rf_cfg.get("enabled", False):
        sw.update(stage="building_features")
        logger.info("\n=== Building feature matrices ===")
        pipeline = FeaturePipeline(
            include_mordred=False,
            include_ecfp6=rf_cfg.get("include_ecfp6", True),
            include_fcfp4=False,
            include_maccs=rf_cfg.get("include_maccs", True),
            include_permeability=rf_cfg.get("include_permeability", False),
            include_caco2=rf_cfg.get("include_caco2", False),
        )
        X_train = pipeline.fit_transform(train_smiles)
        X_test = pipeline.transform(test_smiles)
        if holdout_smiles is not None:
            X_holdout = pipeline.transform(holdout_smiles)
        logger.info(f"Feature matrix: train={X_train.shape}, test={X_test.shape}")

        # Optional scalar features appended to the RF feature matrix
        if rf_cfg.get("include_selectivity_delta", False):
            pec50_null = train["pEC50_null"].to_numpy(dtype=np.float32)
            sel_delta = train_y.astype(np.float32) - pec50_null
            delta_mean = float(np.nanmean(sel_delta))
            sel_delta = np.where(np.isnan(sel_delta), delta_mean, sel_delta)
            sel_delta_test = np.full(len(test_smiles), delta_mean, dtype=np.float32)
            X_train = np.hstack([X_train, sel_delta.reshape(-1, 1)])
            X_test = np.hstack([X_test, sel_delta_test.reshape(-1, 1)])
            n_measured = int(np.isfinite(pec50_null).sum())
            logger.info(
                f"Added selectivity_delta feature: {n_measured} measured, "
                f"{len(train) - n_measured} imputed with mean={delta_mean:.3f}"
            )

        if rf_cfg.get("include_crystal_sim", False):
            ligand_path = rf_cfg.get("crystal_ligand_path", "data/pxr_crystal_ligands.csv")
            if not Path(ligand_path).exists():
                raise FileNotFoundError(
                    f"Crystal ligand file not found at {ligand_path}. "
                    "Run: python scripts/fetch_crystal_ligands.py"
                )
            import pandas as _pd_crys
            from src.features.feature_engineering import crystal_ligand_similarity
            ligand_smiles = _pd_crys.read_csv(ligand_path)["smiles"].tolist()
            sim_train = crystal_ligand_similarity(train_smiles, ligand_smiles)
            sim_test = crystal_ligand_similarity(test_smiles, ligand_smiles)
            X_train = np.hstack([X_train, sim_train.reshape(-1, 1)])
            X_test = np.hstack([X_test, sim_test.reshape(-1, 1)])
            tabicl_extra_train = sim_train.reshape(-1, 1).astype(np.float32)
            tabicl_extra_test = sim_test.reshape(-1, 1).astype(np.float32)
            if holdout_smiles is not None:
                sim_holdout = crystal_ligand_similarity(holdout_smiles, ligand_smiles)
                X_holdout = np.hstack([X_holdout, sim_holdout.reshape(-1, 1)])
                tabicl_extra_holdout = sim_holdout.reshape(-1, 1).astype(np.float32)
            logger.info(
                f"Added crystal_sim feature: {len(ligand_smiles)} reference ligands, "
                f"train mean={sim_train.mean():.3f}, test mean={sim_test.mean():.3f}"
            )

        if rf_cfg.get("include_rdkit2d", False):
            from src.features.feature_engineering import rdkit_descriptors
            rdk_tr = np.nan_to_num(rdkit_descriptors(train_smiles).astype(np.float32),
                                   nan=0.0, posinf=0.0, neginf=0.0)
            rdk_te = np.nan_to_num(rdkit_descriptors(test_smiles).astype(np.float32),
                                   nan=0.0, posinf=0.0, neginf=0.0)
            X_train = np.hstack([X_train, rdk_tr])
            X_test = np.hstack([X_test, rdk_te])
            tabicl_extra_train = (
                np.hstack([tabicl_extra_train, rdk_tr]) if tabicl_extra_train is not None
                else rdk_tr
            )
            tabicl_extra_test = (
                np.hstack([tabicl_extra_test, rdk_te]) if tabicl_extra_test is not None
                else rdk_te
            )
            if holdout_smiles is not None:
                rdk_ho = np.nan_to_num(rdkit_descriptors(holdout_smiles).astype(np.float32),
                                       nan=0.0, posinf=0.0, neginf=0.0)
                X_holdout = np.hstack([X_holdout, rdk_ho])
                tabicl_extra_holdout = (
                    np.hstack([tabicl_extra_holdout, rdk_ho]) if tabicl_extra_holdout is not None
                    else rdk_ho
                )
            logger.info(f"Added RDKit2D descriptors: {rdk_tr.shape[1]} features")

        # HTChem RF feature matrix — same column order as X_train.
        # Must be built here while pipeline, ligand_smiles, etc. are still in scope.
        if htchem_smiles and "rf" in _htchem_aug_models:
            from src.features.feature_engineering import crystal_ligand_similarity as _crys_sim
            X_htchem = pipeline.transform(htchem_smiles)
            if rf_cfg.get("include_selectivity_delta", False):
                # HTChem has no pEC50_null; impute with train mean delta
                X_htchem = np.hstack([
                    X_htchem,
                    np.full((len(htchem_smiles), 1), delta_mean, dtype=np.float32),
                ])
            if rf_cfg.get("include_crystal_sim", False):
                _sim_ht = _crys_sim(htchem_smiles, ligand_smiles)
                X_htchem = np.hstack([X_htchem, _sim_ht.reshape(-1, 1)])
            if rf_cfg.get("include_rdkit2d", False):
                from src.features.feature_engineering import rdkit_descriptors as _rdk
                _rdk_ht = np.nan_to_num(
                    _rdk(htchem_smiles).astype(np.float32),
                    nan=0.0, posinf=0.0, neginf=0.0,
                )
                X_htchem = np.hstack([X_htchem, _rdk_ht])
            logger.info(f"HTChem RF feature matrix: {X_htchem.shape}")

    # ------------------------------------------------------------------
    # Stage 2b: SphereNet ChEMBL pretraining (optional, before CV)
    # ------------------------------------------------------------------
    spherenet_pretrain_sd = None
    sn_cfg = model_cfg.get("spherenet", {})
    _sn_pretrain_cache = Path("logs") / f"{exp_id}_spherenet_pretrain.pt"
    if sn_cfg.get("enabled", False) and sn_cfg.get("chembl_pretrain", False):
        if restack and _sn_pretrain_cache.exists():
            import torch as _torch_pt
            spherenet_pretrain_sd = _torch_pt.load(_sn_pretrain_cache, map_location="cpu")
            logger.info(f"Restack: loaded cached ViSNet pretrain weights ← {_sn_pretrain_cache}")
        else:
            # Prefer combined pretrain file (PXR + CYP3A4 induction + CAR);
            # fall back to PXR-only if fetch_chembl_pxr.py was run before this update.
            chembl_path = Path("data/chembl/chembl_pretrain.csv")
            if not chembl_path.exists():
                chembl_path = Path("data/chembl/chembl_pxr.csv")
            if chembl_path.exists():
                sw.update(stage="stage2b_spherenet_chembl_pretrain")
                logger.info("\n=== Stage 2b: SphereNet ChEMBL PXR pretraining ===")
                import pandas as pd
                chembl_df = pd.read_csv(chembl_path)
                chembl_sm = chembl_df["smiles"].tolist()
                chembl_y = chembl_df["pchembl_value"].to_numpy(dtype=np.float64)
                valid = np.isfinite(chembl_y) & (chembl_y >= 4.0) & (chembl_y <= 11.0)
                chembl_sm = [chembl_sm[i] for i in range(len(chembl_sm)) if valid[i]]
                chembl_y = chembl_y[valid]
                logger.info(f"ChEMBL pretraining: {len(chembl_sm)} PXR compounds")
                from src.models.spherenet_model import SphereNetModel
                sn_pre = SphereNetModel(
                    epochs=sn_cfg.get("epochs_pretrain", 100),
                    lr=sn_cfg.get("lr", 1e-3),
                    batch_size=sn_cfg.get("batch_size", 32),
                    n_conformers=sn_cfg.get("n_conformers", 3),
                    cutoff=sn_cfg.get("cutoff", 5.0),
                    hidden_channels=sn_cfg.get("hidden_channels", 128),
                    num_layers=sn_cfg.get("num_layers", 4),
                    device=device,
                    seed=42,
                )
                sn_pre.fit(chembl_sm, chembl_y)
                spherenet_pretrain_sd = sn_pre.get_state_dict()
                import torch as _torch_pt
                _torch_pt.save(spherenet_pretrain_sd, _sn_pretrain_cache)
                logger.info(f"ViSNet pretrain weights saved → {_sn_pretrain_cache}")
                del sn_pre; gc.collect()
                logger.info("Stage 2b: SphereNet ChEMBL pretraining complete.")
            else:
                logger.warning(
                    "spherenet.chembl_pretrain=True but data/chembl/chembl_pxr.csv not found. "
                    "Run `python scripts/fetch_chembl_pxr.py` first. Training from scratch."
                )

    # ------------------------------------------------------------------
    # 4. Determine model list and which to reuse
    # ------------------------------------------------------------------
    chemprop_seeds = chemprop_hts_cfg.get("seeds", [42, 7, 13])
    chemprop_members = [f"chemprop_hts_s{s}" for s in chemprop_seeds]
    mt_cfg = model_cfg.get("chemprop_mt", {})
    mt_seeds = mt_cfg.get("seeds", [42, 7]) if mt_cfg.get("enabled", False) else []

    all_models = []
    for name in ["delta", "chemprop_scratch"]:
        if model_cfg.get(name, {}).get("enabled", False):
            all_models.append(name)
    if chemprop_hts_cfg.get("enabled", False):
        all_models.extend(chemprop_members)
    if mt_cfg.get("enabled", False):
        all_models.extend(f"chemprop_mt_s{s}" for s in mt_seeds)
    if model_cfg.get("tabpfn", {}).get("enabled", False):
        all_models.append("tabpfn")
    if rf_cfg.get("enabled", False):
        all_models.append("rf")
    if model_cfg.get("gp", {}).get("enabled", False):
        all_models.append("gp")
    unimol_cfg = model_cfg.get("unimol", {})
    unimol_seeds = unimol_cfg.get("seeds") if unimol_cfg.get("enabled", False) else None
    if unimol_cfg.get("enabled", False):
        if unimol_seeds:
            all_models.extend(f"unimol_s{s}" for s in unimol_seeds)
        else:
            all_models.append("unimol")
    if model_cfg.get("spherenet", {}).get("enabled", False):
        all_models.append("spherenet")
    if model_cfg.get("molformer", {}).get("enabled", False):
        all_models.append("molformer")
    if model_cfg.get("tabicl", {}).get("enabled", False):
        all_models.append("tabicl")
    logger.info(f"\nModel members: {all_models}")

    # PMI features for chemprop_mt x_d — compute once, reuse across folds
    pmi_train = pmi_test = pmi_holdout = None
    if mt_cfg.get("enabled", False) and mt_cfg.get("use_pmi", True):
        pmi_npz = Path("logs") / f"{exp_id}_pmi_features.npz"
        if pmi_npz.exists():
            logger.info(f"PMI features: loading from cache {pmi_npz}")
            _pmi = np.load(pmi_npz)
            pmi_train = _pmi["train"]
            pmi_test = _pmi["test"]
            if holdout_smiles is not None:
                pmi_holdout = _pmi["holdout"] if "holdout" in _pmi else None
        else:
            sw.update(stage="computing_pmi_features")
            logger.info(f"Computing PMI features ({_N_PMI} descriptors)...")
            pmi_train = _compute_pmi_features(train_smiles)
            pmi_test = _compute_pmi_features(test_smiles)
            _pmi_save = dict(train=pmi_train, test=pmi_test)
            if holdout_smiles is not None:
                pmi_holdout = _compute_pmi_features(holdout_smiles)
                _pmi_save["holdout"] = pmi_holdout
            np.savez(pmi_npz, **_pmi_save)
            logger.info(f"PMI features cached → {pmi_npz}")
        logger.info(f"PMI feature matrix: train={pmi_train.shape}, test={pmi_test.shape}")
        if htchem_smiles and pmi_train is not None and "chemprop_mt" in _htchem_aug_models:
            htchem_pmi = _compute_pmi_features(htchem_smiles)
            logger.info(f"HTChem PMI features: {htchem_pmi.shape}")

    oof = {m: np.zeros(len(train)) for m in all_models}
    fold_raes_d = {m: [] for m in all_models}

    # Load reusable OOF
    reuse_oof_cfg = config.get("reuse_oof", {})
    if reuse_oof_cfg.get("source") and reuse_oof_cfg.get("models"):
        loaded_oof, loaded_fr = _load_oof(
            reuse_oof_cfg["source"],
            [m for m in reuse_oof_cfg["models"] if m in all_models],
            train_y_list,
            remap=reuse_oof_cfg.get("remap", {}),
        )
        for m, v in loaded_oof.items():
            oof[m] = v
            fold_raes_d[m] = loaded_fr.get(m, [])

    models_to_train = [m for m in all_models if np.all(oof[m] == 0)]
    logger.info(f"Models needing CV training: {models_to_train}")

    # ------------------------------------------------------------------
    # 4b. Restack shortcut: load saved OOF checkpoint, skip CV entirely
    # ------------------------------------------------------------------
    if restack:
        chk_path = Path("logs") / f"{exp_id}_oof_checkpoint.json"
        if not chk_path.exists():
            raise FileNotFoundError(
                f"--restack requires {chk_path} but it does not exist. "
                "Run the full experiment first so the OOF checkpoint is saved."
            )
        chk = json.loads(chk_path.read_text())
        for m in all_models:
            if m in chk:
                oof[m] = np.array(chk[m], dtype=np.float64)
                fold_raes_d[m] = chk.get("fold_raes", {}).get(m, [])
        loaded_train_y = np.array(chk.get("train_y", train_y_list))
        assert len(loaded_train_y) == len(train_y), (
            f"Checkpoint train_y length mismatch: {len(loaded_train_y)} vs {len(train_y)}"
        )
        logger.info(f"Restack: loaded OOF checkpoint from {chk_path}")
        # fall through directly to stacking — skip the CV section below

    # ------------------------------------------------------------------
    # 5. Butina 5-fold CV
    # ------------------------------------------------------------------
    from src.evaluation.validate import ButinaKFold, rae

    n_splits = config.get("cv_splits", 5)
    splitter = ButinaKFold(n_splits=n_splits, tanimoto_threshold=0.4)
    folds = list(splitter.split(train_smiles))

    for fold_idx, (tr_idx, va_idx) in enumerate(folds):
        if not models_to_train or restack:
            break

        sw.update(stage=f"cv_fold_{fold_idx + 1}_of_{n_splits}")
        logger.info(
            f"\n--- Fold {fold_idx + 1}/{n_splits} "
            f"(train={len(tr_idx)}, val={len(va_idx)}) ---"
        )

        sm_tr = [train_smiles[i] for i in tr_idx]
        sm_va = [train_smiles[i] for i in va_idx]
        y_tr, y_va = train_y[tr_idx], train_y[va_idx]
        sw_tr = sample_weights[tr_idx]
        fps_tr, fps_va = fp_train[tr_idx], fp_train[va_idx]
        fold_mean = float(y_tr.mean())

        # HTChem fold augmentation — appended to training partition only, never validation.
        # OOF indices still reference only the primary 3,823 compounds for clean evaluation.
        if htchem_smiles:
            _nht = len(htchem_smiles)
            sm_tr_ht = sm_tr + htchem_smiles
            y_tr_ht = np.concatenate([y_tr, htchem_y])
            sw_tr_ht = np.concatenate([sw_tr, np.full(_nht, _ht_weight, dtype=np.float32)])
            fps_tr_ht = np.vstack([fps_tr, htchem_fps])
            y_mt_tr_ht = np.vstack([
                y_mt[tr_idx],
                np.column_stack([htchem_y, np.full(_nht, np.nan), np.full(_nht, np.nan)]).astype(np.float32),
            ])
            _x_d_tr_base = pmi_train[tr_idx] if pmi_train is not None else None
            x_d_tr_ht = (
                np.vstack([_x_d_tr_base, htchem_pmi])
                if _x_d_tr_base is not None and htchem_pmi is not None
                else _x_d_tr_base
            )
            X_tr_ht = (
                np.vstack([X_train[tr_idx], X_htchem])
                if X_htchem is not None and "rf" in _htchem_aug_models
                else (X_train[tr_idx] if X_train is not None else None)
            )
        else:
            sm_tr_ht = sm_tr
            y_tr_ht = y_tr
            sw_tr_ht = sw_tr
            fps_tr_ht = fps_tr
            y_mt_tr_ht = y_mt[tr_idx]
            x_d_tr_ht = pmi_train[tr_idx] if pmi_train is not None else None
            X_tr_ht = X_train[tr_idx] if X_train is not None else None

        if "delta" in models_to_train:
            from src.models.delta_model import DeltaChempropModel
            logger.info(f"Fold {fold_idx + 1}: Delta...")
            d = DeltaChempropModel(
                epochs=delta_cfg.get("epochs_cv", 40),
                hidden_size=300, depth=3, ffn_num_layers=3,
                dropout=0.1, batch_size=64, lr=delta_cfg.get("lr", 1e-3), device=device,
                snapshot_epochs=5, seed=42, n_pairs_per_epoch=20_000,
                cliff_oversample=3, k_neighbors=delta_cfg.get("k_neighbors", 10),
            )
            _delta_use_ht = htchem_smiles and "delta" in _htchem_aug_models
            d.fit(
                sm_tr_ht if _delta_use_ht else sm_tr,
                y_tr_ht if _delta_use_ht else y_tr,
                fps_train=fps_tr_ht if _delta_use_ht else fps_tr,
                init_state_dict=hts_state_dict,
            )
            delta_series_anchors = None
            if delta_cfg.get("series_aware", False):
                from src.data.cliff_analysis import identify_test_series
                delta_series_anchors = identify_test_series(
                    fps_va, fps_tr,
                    sim_threshold=delta_cfg.get("series_sim_threshold", 0.4),
                )
            _delta_kw = {"series_anchors": delta_series_anchors} if delta_series_anchors is not None else {}
            oof["delta"][va_idx] = d.predict(
                sm_va, sm_tr, y_tr, fps_tr, fps_va, **_delta_kw
            )
            fold_raes_d["delta"].append(
                rae(y_va, oof["delta"][va_idx], y_train_mean=fold_mean)
            )
            logger.info(f"  delta RAE={fold_raes_d['delta'][-1]:.4f}")
            del d; gc.collect()

        if "chemprop_scratch" in models_to_train:
            scratch_cfg = model_cfg.get("chemprop_scratch", {})
            logger.info(f"Fold {fold_idx + 1}: Chemprop scratch...")
            m = ChempropModel(
                epochs=scratch_cfg.get("epochs_cv", 60),
                hidden_size=300, depth=3, ffn_num_layers=3,
                dropout=0.1, batch_size=64, lr=1e-3, device=device,
                snapshot_epochs=5, extra_features=False, seed=42, n_tasks=1,
            )
            m.fit(sm_tr, y_tr, sample_weight=sw_tr)
            oof["chemprop_scratch"][va_idx] = m.predict(sm_va)
            fold_raes_d["chemprop_scratch"].append(
                rae(y_va, oof["chemprop_scratch"][va_idx], y_train_mean=fold_mean)
            )
            logger.info(f"  chemprop_scratch RAE={fold_raes_d['chemprop_scratch'][-1]:.4f}")
            del m; gc.collect()

        for seed in chemprop_seeds:
            mname = f"chemprop_hts_s{seed}"
            if mname in models_to_train:
                logger.info(f"Fold {fold_idx + 1}: chemprop_hts seed={seed}...")
                m = ChempropModel(
                    epochs=chemprop_hts_cfg.get("epochs_cv", 60),
                    hidden_size=300, depth=3, ffn_num_layers=3,
                    dropout=0.1, batch_size=64, lr=5e-4, device=device,
                    snapshot_epochs=5, extra_features=False, seed=seed, n_tasks=1,
                )
                m.fit(sm_tr, y_tr, sample_weight=sw_tr, init_state_dict=hts_state_dict)
                oof[mname][va_idx] = m.predict(sm_va)
                fold_raes_d[mname].append(
                    rae(y_va, oof[mname][va_idx], y_train_mean=fold_mean)
                )
                logger.info(f"  {mname} RAE={fold_raes_d[mname][-1]:.4f}")
                del m; gc.collect()

        for seed in mt_seeds:
            mname = f"chemprop_mt_s{seed}"
            if mname in models_to_train:
                logger.info(f"Fold {fold_idx + 1}: chemprop_mt seed={seed} (n_tasks=3)...")
                x_d_va = pmi_train[va_idx] if pmi_train is not None else None
                m = ChempropModel(
                    epochs=mt_cfg.get("epochs_cv", 80),
                    hidden_size=300, depth=3, ffn_num_layers=3,
                    dropout=0.1, batch_size=64,
                    lr=mt_cfg.get("lr", 5e-4), device=device,
                    snapshot_epochs=5, extra_features=False, seed=seed,
                    n_tasks=3, mask_missing_tasks=True,
                )
                _mt_use_ht = htchem_smiles and "chemprop_mt" in _htchem_aug_models
                m.fit(
                    sm_tr_ht if _mt_use_ht else sm_tr,
                    y_mt_tr_ht if _mt_use_ht else y_mt[tr_idx],
                    sample_weight=sw_tr_ht if _mt_use_ht else sw_tr,
                    init_state_dict=hts_state_dict,
                    x_d=x_d_tr_ht if _mt_use_ht else (pmi_train[tr_idx] if pmi_train is not None else None),
                )
                oof[mname][va_idx] = m.predict(sm_va, x_d=x_d_va)
                fold_raes_d[mname].append(
                    rae(y_va, oof[mname][va_idx], y_train_mean=fold_mean)
                )
                logger.info(f"  {mname} RAE={fold_raes_d[mname][-1]:.4f}")
                del m; gc.collect()

        if "tabpfn" in models_to_train:
            from src.models.tabpfn_model import TabPFNCheMeleonModel
            tabpfn_cfg = model_cfg.get("tabpfn", {})
            n_est = tabpfn_cfg.get("n_estimators", 16)
            logger.info(f"Fold {fold_idx + 1}: TabPFN (n_estimators={n_est})...")
            tf = TabPFNCheMeleonModel(
                n_components=200, n_estimators=n_est, device=device, random_state=42,
            )
            _tabpfn_use_ht = htchem_smiles and "tabpfn" in _htchem_aug_models
            tf.fit(sm_tr_ht if _tabpfn_use_ht else sm_tr, y_tr_ht if _tabpfn_use_ht else y_tr)
            oof["tabpfn"][va_idx] = tf.predict(sm_va)
            fold_raes_d["tabpfn"].append(
                rae(y_va, oof["tabpfn"][va_idx], y_train_mean=fold_mean)
            )
            logger.info(f"  tabpfn RAE={fold_raes_d['tabpfn'][-1]:.4f}")
            del tf; gc.collect()

        if "tabicl" in models_to_train:
            from src.models.tabicl_model import TabICLCheMeleonModel
            tabicl_cfg = model_cfg.get("tabicl", {})
            n_est_ti = tabicl_cfg.get("n_estimators", 8)
            feat_mode = tabicl_cfg.get("features", "chemeleon+rdkit2d+crystal")
            logger.info(
                f"Fold {fold_idx + 1}: TabICL (n_estimators={n_est_ti}, features={feat_mode})..."
            )
            ti = TabICLCheMeleonModel(
                features=feat_mode, n_components=200, n_estimators=n_est_ti,
                device=device, random_state=42,
            )
            ex_tr = tabicl_extra_train[tr_idx] if tabicl_extra_train is not None else None
            ex_va = tabicl_extra_train[va_idx] if tabicl_extra_train is not None else None
            ti.fit(sm_tr, y_tr, extra_features=ex_tr)
            oof["tabicl"][va_idx] = ti.predict(sm_va, extra_features=ex_va)
            fold_raes_d["tabicl"].append(
                rae(y_va, oof["tabicl"][va_idx], y_train_mean=fold_mean)
            )
            logger.info(f"  tabicl RAE={fold_raes_d['tabicl'][-1]:.4f}")
            del ti; gc.collect()

        if "rf" in models_to_train:
            from src.models.gbm_models import RFWrapper
            X_va = X_train[va_idx]
            rf = RFWrapper()
            _rf_use_ht = htchem_smiles and "rf" in _htchem_aug_models and X_tr_ht is not None
            rf.fit(
                X_tr_ht if _rf_use_ht else X_train[tr_idx],
                y_tr_ht if _rf_use_ht else y_tr,
                sample_weight=sw_tr_ht if _rf_use_ht else sw_tr,
            )
            oof["rf"][va_idx] = rf.predict(X_va)
            fold_raes_d["rf"].append(
                rae(y_va, oof["rf"][va_idx], y_train_mean=fold_mean)
            )
            logger.info(f"  rf RAE={fold_raes_d['rf'][-1]:.4f}")

        if "gp" in models_to_train:
            from src.models.local_models import TanimotoGP
            gp_cfg = model_cfg.get("gp", {})
            max_ts = gp_cfg.get("max_train_size", 4000)
            logger.info(f"Fold {fold_idx + 1}: TanimotoGP (max_train_size={max_ts})...")
            gp = TanimotoGP(max_train_size=max_ts)
            gp.fit(fps_tr.astype(np.float32), y_tr)
            oof["gp"][va_idx] = gp.predict(fps_va.astype(np.float32))
            fold_raes_d["gp"].append(
                rae(y_va, oof["gp"][va_idx], y_train_mean=fold_mean)
            )
            logger.info(f"  gp RAE={fold_raes_d['gp'][-1]:.4f}")
            del gp; gc.collect()

        _unimol_train_seeds = unimol_seeds if unimol_seeds else ([42] if "unimol" in models_to_train else [])
        for _um_seed in _unimol_train_seeds:
            _um_key = f"unimol_s{_um_seed}" if unimol_seeds else "unimol"
            if _um_key not in models_to_train:
                continue
            from src.models.unimol_model import UniMolModel
            logger.info(
                f"Fold {fold_idx + 1}: UniMol seed={_um_seed} "
                f"(epochs={unimol_cfg.get('epochs_cv', 10)})..."
            )
            um = UniMolModel(
                epochs=unimol_cfg.get("epochs_cv", 10),
                lr=unimol_cfg.get("lr", 1e-4),
                batch_size=unimol_cfg.get("batch_size", 16),
                seed=_um_seed,
                use_conformer_resampling=unimol_cfg.get("use_conformer_resampling", False),
                n_train_conformers=unimol_cfg.get("n_train_conformers", 1),
                n_infer_conformers=unimol_cfg.get("n_infer_conformers", 1),
            )
            _um_use_ht = htchem_smiles and "unimol" in _htchem_aug_models
            um.fit(sm_tr_ht if _um_use_ht else sm_tr, y_tr_ht if _um_use_ht else y_tr)
            oof[_um_key][va_idx] = um.predict(sm_va)
            fold_raes_d[_um_key].append(
                rae(y_va, oof[_um_key][va_idx], y_train_mean=fold_mean)
            )
            logger.info(f"  {_um_key} RAE={fold_raes_d[_um_key][-1]:.4f}")
            del um; gc.collect()

        if "spherenet" in models_to_train:
            from src.models.spherenet_model import SphereNetModel
            logger.info(
                f"Fold {fold_idx + 1}: SphereNet "
                f"(epochs={sn_cfg.get('epochs_cv', 150)}, "
                f"n_confs={sn_cfg.get('n_conformers', 3)}, "
                f"chembl_pretrain={spherenet_pretrain_sd is not None})..."
            )
            sn = SphereNetModel(
                epochs=sn_cfg.get("epochs_cv", 150),
                lr=sn_cfg.get("lr", 1e-3),
                batch_size=sn_cfg.get("batch_size", 32),
                n_conformers=sn_cfg.get("n_conformers", 3),
                cutoff=sn_cfg.get("cutoff", 5.0),
                hidden_channels=sn_cfg.get("hidden_channels", 128),
                num_layers=sn_cfg.get("num_layers", 4),
                device=device,
                seed=42,
            )
            sn.fit(sm_tr, y_tr, init_state_dict=spherenet_pretrain_sd)
            oof["spherenet"][va_idx] = sn.predict(sm_va)
            fold_raes_d["spherenet"].append(
                rae(y_va, oof["spherenet"][va_idx], y_train_mean=fold_mean)
            )
            logger.info(f"  spherenet RAE={fold_raes_d['spherenet'][-1]:.4f}")
            del sn; gc.collect()

        if "molformer" in models_to_train:
            from src.models.molformer_model import MolFormerModel
            mf_cfg = model_cfg.get("molformer", {})
            logger.info(f"Fold {fold_idx + 1}: MolFormer (frozen + RidgeCV)...")
            mf = MolFormerModel(
                batch_size=mf_cfg.get("batch_size", 64),
                device=mf_cfg.get("device", "cpu"),
                seed=42,
            )
            mf.fit(sm_tr, y_tr)
            oof["molformer"][va_idx] = mf.predict(sm_va)
            fold_raes_d["molformer"].append(
                rae(y_va, oof["molformer"][va_idx], y_train_mean=fold_mean)
            )
            logger.info(f"  molformer RAE={fold_raes_d['molformer'][-1]:.4f}")
            del mf; gc.collect()

    # Checkpoint OOF
    chk = {m: oof[m].tolist() for m in all_models}
    chk["train_y"] = train_y_list
    chk["fold_raes"] = {m: fold_raes_d[m] for m in all_models}
    chk_path = Path("logs") / f"{exp_id}_oof_checkpoint.json"
    chk_path.write_text(json.dumps(chk))
    logger.info(f"OOF checkpoint → {chk_path}")

    # ------------------------------------------------------------------
    # 6. CV summary + ElasticNet stacking
    # ------------------------------------------------------------------
    sw.update(stage="stacking_cv")
    logger.info("\n=== CV Summary ===")
    cv_summary = {}
    for m in all_models:
        rs = fold_raes_d[m]
        if rs:
            mean_r, std_r = float(np.mean(rs)), float(np.std(rs))
        else:
            # Reused model — compute directly from OOF
            res = np.abs(oof[m] - train_y)
            mean_r = float(np.mean(res) / np.mean(np.abs(train_y - train_mean)))
            std_r = 0.0
        cv_summary[m] = {"mean": mean_r, "std": std_r}
        logger.info(f"  {m:<30}: {mean_r:.4f} ± {std_r:.4f}")

    from src.ensemble.stack_and_submit import ElasticNetStacker, RidgeStacker
    oof_matrix = np.column_stack([oof[m] for m in all_models])

    # Impute NaN columns (e.g. from models that diverged on some molecules)
    for col_idx, m in enumerate(all_models):
        col = oof_matrix[:, col_idx]
        nan_mask = ~np.isfinite(col)
        if nan_mask.any():
            col_mean = float(np.nanmean(col)) if np.isfinite(col).any() else train_mean
            oof_matrix[nan_mask, col_idx] = col_mean
            logger.warning(
                f"  NaN imputed in OOF '{m}': {nan_mask.sum()} values → {col_mean:.4f}"
            )

    meta_learner_type = config.get("meta_learner", "ridge")
    if meta_learner_type == "elasticnet":
        _ml_kwargs = config.get("meta_learner_kwargs", {})
        stacker = ElasticNetStacker(l1_ratio=_ml_kwargs.get("l1_ratio", 0.7), cv=5)
    else:
        stacker = RidgeStacker(cv=5)
    stacker.fit(oof_matrix, train_y, model_names=all_models)
    oof_stacked = stacker.predict(oof_matrix)
    stacked_rae = float(rae(train_y, oof_stacked, y_train_mean=train_mean))
    improvement = baseline_rae - stacked_rae
    accepted = improvement >= accept_thresh

    logger.info(f"\n{meta_learner_type.capitalize()} OOF RAE: {stacked_rae:.4f}")
    logger.info(f"Baseline RAE:       {baseline_rae:.4f}")
    logger.info(f"Improvement:        {improvement:+.4f}")
    logger.info(f"Accepted:           {accepted} (threshold={accept_thresh})")
    logger.info(f"Coefs: {stacker.coefs}")

    results = {
        "experiment_id": exp_id,
        "cv_rae": stacked_rae,
        "baseline_rae": baseline_rae,
        "improvement": improvement,
        "accepted": accepted,
        "acceptance_threshold": accept_thresh,
        "cv_summary": cv_summary,
        "stacker_coefs": stacker.coefs,
        "meta_learner": meta_learner_type,
        "device": device,
        "runtime_seconds": _elapsed(t0),
        "submission_path": None,
    }
    results_path = Path("logs") / f"{exp_id}_results.json"
    results_path.write_text(json.dumps(results, indent=2))
    logger.info(f"Results → {results_path}")

    sw.update(
        stage="cv_done",
        cv_rae=stacked_rae,
        improvement=improvement,
        accepted=accepted,
        coefs=stacker.coefs,
    )

    if not accepted:
        logger.info(
            f"\nNOT accepted (improvement {improvement:+.4f} < {accept_thresh}). "
            "No submission generated."
        )
        sw.update(stage="rejected")
        if holdout_smiles is None:
            sw.update(status="done", runtime_seconds=_elapsed(t0))
            return
        logger.info("Holdout eval requested — continuing to full training for Analog Set 1 evaluation.")

    # ------------------------------------------------------------------
    # 7. Full training + test predictions
    # ------------------------------------------------------------------
    sw.update(stage="full_training")
    logger.info("\n=== Full training (all data) ===")
    FINAL_SEEDS = [42, 7, 13]
    test_preds: dict = {}
    holdout_preds: dict = {}

    # Load cached holdout predictions for fixed models (rf, tabpfn, unimol).
    # The cache is written by the first hill-climb trial and reused by all subsequent ones,
    # avoiding redundant retrains for models whose params are frozen in EXCLUDED_DIMS.
    _holdout_cache_path = config.get("holdout_pred_cache")
    if _holdout_cache_path and holdout_smiles is not None and Path(_holdout_cache_path).exists():
        try:
            import json as _json
            with open(_holdout_cache_path) as _hf:
                _hcache = _json.load(_hf)
            for _hm, _hpreds in _hcache.items():
                if _hm in all_models:
                    holdout_preds[_hm] = np.array(_hpreds, dtype=np.float32)
                    logger.info(f"Holdout pred cache: loaded {_hm} ({len(_hpreds)} predictions)")
        except Exception as _exc:
            logger.warning(f"Holdout pred cache load failed ({_exc}); will retrain holdout models")

    # Reuse cached test predictions
    reuse_test_cfg = config.get("reuse_test", {})
    if reuse_test_cfg.get("source") and reuse_test_cfg.get("models"):
        cached = _load_test_cache(
            reuse_test_cfg["source"],
            [m for m in reuse_test_cfg["models"] if m in all_models],
            test_smiles,
            remap=reuse_test_cfg.get("remap", {}),
        )
        test_preds.update(cached)

    # Build full-training augmented arrays once (reused by all augmented models).
    if htchem_smiles:
        _nht_full = len(htchem_smiles)
        _aug = lambda m: m in _htchem_aug_models  # noqa: E731
        _full_sm_delta   = train_smiles + htchem_smiles if _aug("delta") else train_smiles
        _full_y_delta    = np.concatenate([train_y, htchem_y]) if _aug("delta") else train_y
        _full_fps_delta  = np.vstack([fp_train, htchem_fps]) if _aug("delta") else fp_train
        _full_sm_mt      = train_smiles + htchem_smiles if _aug("chemprop_mt") else train_smiles
        _full_sw_mt      = (
            np.concatenate([sample_weights, np.full(_nht_full, _ht_weight, dtype=np.float32)])
            if _aug("chemprop_mt") else sample_weights
        )
        _full_y_mt       = (
            np.vstack([y_mt, np.column_stack([htchem_y, np.full(_nht_full, np.nan), np.full(_nht_full, np.nan)]).astype(np.float32)])
            if _aug("chemprop_mt") else y_mt
        )
        _full_pmi_mt     = (
            np.vstack([pmi_train, htchem_pmi])
            if pmi_train is not None and htchem_pmi is not None and _aug("chemprop_mt")
            else pmi_train
        )
        _full_sm_tabpfn  = train_smiles + htchem_smiles if _aug("tabpfn") else train_smiles
        _full_y_tabpfn   = np.concatenate([train_y, htchem_y]) if _aug("tabpfn") else train_y
        _full_sm_unimol  = train_smiles + htchem_smiles if _aug("unimol") else train_smiles
        _full_y_unimol   = np.concatenate([train_y, htchem_y]) if _aug("unimol") else train_y
        _full_X_rf       = (
            np.vstack([X_train, X_htchem])
            if _aug("rf") and X_train is not None and X_htchem is not None
            else X_train
        )
        _full_y_rf       = np.concatenate([train_y, htchem_y]) if _aug("rf") else train_y
        _full_sw_rf      = (
            np.concatenate([sample_weights, np.full(_nht_full, _ht_weight, dtype=np.float32)])
            if _aug("rf") else sample_weights
        )
    else:
        _full_sm_delta   = train_smiles
        _full_y_delta    = train_y
        _full_fps_delta  = fp_train
        _full_sm_mt      = train_smiles
        _full_sw_mt      = sample_weights
        _full_y_mt       = y_mt
        _full_pmi_mt     = pmi_train
        _full_sm_tabpfn  = train_smiles
        _full_y_tabpfn   = train_y
        _full_sm_unimol  = train_smiles
        _full_y_unimol   = train_y
        _full_X_rf       = X_train
        _full_y_rf       = train_y
        _full_sw_rf      = sample_weights

    if "delta" not in test_preds and "delta" in all_models:
        from src.models.delta_model import DeltaChempropModel
        logger.info("Full training: Delta (3 seeds)...")
        delta_series_anchors_test = None
        delta_series_anchors_holdout = None
        if delta_cfg.get("series_aware", False):
            from src.data.cliff_analysis import identify_test_series
            delta_series_anchors_test = identify_test_series(
                fp_test, fp_train,
                sim_threshold=delta_cfg.get("series_sim_threshold", 0.4),
            )
            if holdout_smiles is not None:
                delta_series_anchors_holdout = identify_test_series(
                    fp_holdout, fp_train,
                    sim_threshold=delta_cfg.get("series_sim_threshold", 0.4),
                )
        preds = []
        holdout_preds_delta = [] if holdout_smiles is not None else None
        for seed in FINAL_SEEDS:
            dm = DeltaChempropModel(
                epochs=delta_cfg.get("epochs_full", 50),
                hidden_size=300, depth=3, ffn_num_layers=3,
                dropout=0.1, batch_size=64, lr=delta_cfg.get("lr", 1e-3), device=device,
                snapshot_epochs=5, seed=seed, n_pairs_per_epoch=20_000,
                cliff_oversample=3, k_neighbors=delta_cfg.get("k_neighbors", 10),
            )
            dm.fit(_full_sm_delta, _full_y_delta, fps_train=_full_fps_delta,
                   init_state_dict=hts_state_dict)
            _delta_kw = {"series_anchors": delta_series_anchors_test} if delta_series_anchors_test is not None else {}
            preds.append(dm.predict(
                test_smiles, _full_sm_delta, _full_y_delta, _full_fps_delta, fp_test, **_delta_kw
            ))
            if holdout_preds_delta is not None:
                _delta_kw_h = {"series_anchors": delta_series_anchors_holdout} if delta_series_anchors_holdout is not None else {}
                holdout_preds_delta.append(dm.predict(
                    holdout_smiles, _full_sm_delta, _full_y_delta, _full_fps_delta, fp_holdout, **_delta_kw_h
                ))
            del dm; gc.collect()
        test_preds["delta"] = np.mean(preds, axis=0)
        if holdout_preds_delta is not None:
            holdout_preds["delta"] = np.mean(holdout_preds_delta, axis=0)

    # Holdout-only Delta: reused test_preds from cache but holdout needs fresh training
    if holdout_smiles is not None and "delta" in all_models and "delta" not in holdout_preds:
        from src.models.delta_model import DeltaChempropModel
        logger.info("Holdout eval: Delta retrain (3 seeds)...")
        holdout_preds_delta_h = []
        for seed in FINAL_SEEDS:
            dm_h = DeltaChempropModel(
                epochs=delta_cfg.get("epochs_full", 50),
                hidden_size=300, depth=3, ffn_num_layers=3,
                dropout=0.1, batch_size=64, lr=delta_cfg.get("lr", 1e-3), device=device,
                snapshot_epochs=5, seed=seed, n_pairs_per_epoch=20_000,
                cliff_oversample=3, k_neighbors=delta_cfg.get("k_neighbors", 10),
            )
            dm_h.fit(_full_sm_delta, _full_y_delta, fps_train=_full_fps_delta,
                     init_state_dict=hts_state_dict)
            holdout_preds_delta_h.append(dm_h.predict(
                holdout_smiles, _full_sm_delta, _full_y_delta, _full_fps_delta, fp_holdout
            ))
            del dm_h; gc.collect()
        holdout_preds["delta"] = np.mean(holdout_preds_delta_h, axis=0)

    if "chemprop_scratch" not in test_preds and "chemprop_scratch" in all_models:
        scratch_cfg = model_cfg.get("chemprop_scratch", {})
        logger.info("Full training: Chemprop scratch (3 seeds)...")
        preds = []
        for seed in FINAL_SEEDS:
            m = ChempropModel(
                epochs=scratch_cfg.get("epochs_full", 80),
                hidden_size=300, depth=3, ffn_num_layers=3,
                dropout=0.1, batch_size=64, lr=1e-3, device=device,
                snapshot_epochs=5, extra_features=False, seed=seed, n_tasks=1,
            )
            m.fit(train_smiles, train_y, sample_weight=sample_weights)
            preds.append(m.predict(test_smiles))
            del m; gc.collect()
        test_preds["chemprop_scratch"] = np.mean(preds, axis=0)

    # Deferred HTS pretraining — runs here only if it was skipped earlier
    # (all CV OOFs were reused) but full training now needs it.
    _needs_hts_full = any(
        f"chemprop_hts_s{s}" not in test_preds and f"chemprop_hts_s{s}" in all_models
        for s in chemprop_seeds
    )
    _needs_mt_full = any(
        f"chemprop_mt_s{s}" not in test_preds and f"chemprop_mt_s{s}" in all_models
        for s in mt_seeds
    )
    _needs_mt_holdout = holdout_smiles is not None and any(
        f"chemprop_mt_s{s}" in all_models and f"chemprop_mt_s{s}" not in holdout_preds
        and f"chemprop_mt_s{s}" in test_preds  # i.e. test was reused, holdout still needed
        for s in mt_seeds
    )
    # Deferred pretrain: also check for pre-computed checkpoint before re-running HTS
    if hts_state_dict is None and _pretrain_ckpt_path and Path(_pretrain_ckpt_path).exists():
        import torch as _torch2
        hts_state_dict = _torch2.load(_pretrain_ckpt_path, map_location="cpu")
        logger.info(f"Deferred pretrain: loaded checkpoint from {_pretrain_ckpt_path}")
    if hts_state_dict is None and (_needs_hts_full or _needs_mt_full or _needs_mt_holdout) and "hts" in pretrain_stages:
        sw.update(stage="stage2_hts_pretrain_deferred")
        logger.info("\n=== Stage 2 (deferred): HTS concentration-aware pretraining ===")
        from src.models.hts_pretrain import prepare_hts_concentration_data
        pt_sm_d, pt_y_d, pt_xd_d = prepare_hts_concentration_data(hts_df=hts, primary_train=train)
        logger.info(f"HTS pretraining on {len(pt_sm_d)} rows")
        hm_d = ChempropModel(
            epochs=chemprop_hts_cfg.get("hts_epochs", 60),
            hidden_size=300, depth=3, ffn_num_layers=3,
            dropout=0.1, batch_size=64, lr=1e-3, device=device,
            snapshot_epochs=5, extra_features=False, seed=42, n_tasks=1,
        )
        hm_d.fit(pt_sm_d, pt_y_d, x_d=pt_xd_d)
        hts_state_dict = hm_d.get_state_dict()
        del hm_d; gc.collect()
        logger.info("Deferred HTS pretraining complete.")

    for seed in chemprop_seeds:
        mname = f"chemprop_hts_s{seed}"
        if mname not in test_preds and mname in all_models:
            logger.info(f"Full training: {mname} (3-stage pretrained)...")
            m = ChempropModel(
                epochs=chemprop_hts_cfg.get("epochs_full", 80),
                hidden_size=300, depth=3, ffn_num_layers=3,
                dropout=0.1, batch_size=64, lr=5e-4, device=device,
                snapshot_epochs=5, extra_features=False, seed=seed, n_tasks=1,
            )
            m.fit(train_smiles, train_y, sample_weight=sample_weights,
                  init_state_dict=hts_state_dict)
            test_preds[mname] = m.predict(test_smiles)
            del m; gc.collect()

    for seed in mt_seeds:
        mname = f"chemprop_mt_s{seed}"
        if mname not in test_preds and mname in all_models:
            logger.info(f"Full training: {mname} (3-task, HTS pretrained)...")
            m = ChempropModel(
                epochs=mt_cfg.get("epochs_full", 100),
                hidden_size=300, depth=3, ffn_num_layers=3,
                dropout=0.1, batch_size=64,
                lr=mt_cfg.get("lr", 5e-4), device=device,
                snapshot_epochs=5, extra_features=False, seed=seed,
                n_tasks=3, mask_missing_tasks=True,
            )
            m.fit(_full_sm_mt, _full_y_mt, sample_weight=_full_sw_mt,
                  init_state_dict=hts_state_dict, x_d=_full_pmi_mt)
            test_preds[mname] = m.predict(test_smiles, x_d=pmi_test)
            if holdout_smiles is not None:
                holdout_preds[mname] = m.predict(holdout_smiles, x_d=pmi_holdout)
            del m; gc.collect()

    # Holdout-only chemprop_mt: reused test_preds from cache but holdout needs fresh training
    for seed in mt_seeds:
        mname = f"chemprop_mt_s{seed}"
        if holdout_smiles is not None and mname in all_models and mname not in holdout_preds:
            logger.info(f"Holdout eval: {mname} retrain...")
            m_h = ChempropModel(
                epochs=mt_cfg.get("epochs_full", 100),
                hidden_size=300, depth=3, ffn_num_layers=3,
                dropout=0.1, batch_size=64,
                lr=mt_cfg.get("lr", 5e-4), device=device,
                snapshot_epochs=5, extra_features=False, seed=seed,
                n_tasks=3, mask_missing_tasks=True,
            )
            m_h.fit(_full_sm_mt, _full_y_mt, sample_weight=_full_sw_mt,
                    init_state_dict=hts_state_dict, x_d=_full_pmi_mt)
            holdout_preds[mname] = m_h.predict(holdout_smiles, x_d=pmi_holdout)
            del m_h; gc.collect()

    if hts_state_dict is not None:
        del hts_state_dict; gc.collect()

    if "tabpfn" not in test_preds and "tabpfn" in all_models:
        from src.models.tabpfn_model import TabPFNCheMeleonModel
        tabpfn_cfg = model_cfg.get("tabpfn", {})
        n_est = tabpfn_cfg.get("n_estimators", 16)
        logger.info(f"Full training: TabPFN (n_estimators={n_est})...")
        tf = TabPFNCheMeleonModel(
            n_components=200, n_estimators=n_est, device=device, random_state=42,
        )
        tf.fit(_full_sm_tabpfn, _full_y_tabpfn)
        test_preds["tabpfn"] = tf.predict(test_smiles)
        if holdout_smiles is not None:
            holdout_preds["tabpfn"] = tf.predict(holdout_smiles)
        del tf; gc.collect()

    # Holdout-only TabPFN: reused test_preds from cache but holdout needs fresh training
    if holdout_smiles is not None and "tabpfn" in all_models and "tabpfn" not in holdout_preds:
        from src.models.tabpfn_model import TabPFNCheMeleonModel
        tabpfn_cfg = model_cfg.get("tabpfn", {})
        n_est = tabpfn_cfg.get("n_estimators", 16)
        logger.info(f"Holdout eval: TabPFN retrain (n_estimators={n_est})...")
        tf_h = TabPFNCheMeleonModel(
            n_components=200, n_estimators=n_est, device=device, random_state=42,
        )
        tf_h.fit(_full_sm_tabpfn, _full_y_tabpfn)
        holdout_preds["tabpfn"] = tf_h.predict(holdout_smiles)
        del tf_h; gc.collect()

    if "tabicl" not in test_preds and "tabicl" in all_models:
        from src.models.tabicl_model import TabICLCheMeleonModel
        tabicl_cfg = model_cfg.get("tabicl", {})
        n_est_ti = tabicl_cfg.get("n_estimators", 8)
        feat_mode = tabicl_cfg.get("features", "chemeleon+rdkit2d+crystal")
        logger.info(f"Full training: TabICL (n_estimators={n_est_ti}, features={feat_mode})...")
        ti_f = TabICLCheMeleonModel(
            features=feat_mode, n_components=200, n_estimators=n_est_ti,
            device=device, random_state=42,
        )
        ti_f.fit(train_smiles, train_y, extra_features=tabicl_extra_train)
        test_preds["tabicl"] = ti_f.predict(test_smiles, extra_features=tabicl_extra_test)
        if holdout_smiles is not None:
            holdout_preds["tabicl"] = ti_f.predict(holdout_smiles, extra_features=tabicl_extra_holdout)
        del ti_f; gc.collect()

    # Holdout-only TabICL: reused test_preds from cache but holdout needs fresh training
    if holdout_smiles is not None and "tabicl" in all_models and "tabicl" not in holdout_preds:
        from src.models.tabicl_model import TabICLCheMeleonModel
        tabicl_cfg = model_cfg.get("tabicl", {})
        n_est_ti = tabicl_cfg.get("n_estimators", 8)
        feat_mode = tabicl_cfg.get("features", "chemeleon+rdkit2d+crystal")
        logger.info(f"Holdout eval: TabICL retrain (n_estimators={n_est_ti})...")
        ti_h = TabICLCheMeleonModel(
            features=feat_mode, n_components=200, n_estimators=n_est_ti,
            device=device, random_state=42,
        )
        ti_h.fit(train_smiles, train_y, extra_features=tabicl_extra_train)
        holdout_preds["tabicl"] = ti_h.predict(holdout_smiles, extra_features=tabicl_extra_holdout)
        del ti_h; gc.collect()

    if "rf" not in test_preds and "rf" in all_models:
        from src.models.gbm_models import RFWrapper
        logger.info("Full training: RF...")
        rf_f = RFWrapper()
        rf_f.fit(_full_X_rf if _full_X_rf is not None else X_train, _full_y_rf, sample_weight=_full_sw_rf)
        test_preds["rf"] = rf_f.predict(X_test)
        if holdout_smiles is not None and X_holdout is not None:
            holdout_preds["rf"] = rf_f.predict(X_holdout)

    # Holdout-only RF: reused test_preds from cache but holdout needs fresh training
    if holdout_smiles is not None and X_holdout is not None and "rf" in all_models and "rf" not in holdout_preds:
        from src.models.gbm_models import RFWrapper
        logger.info("Holdout eval: RF retrain...")
        rf_h = RFWrapper()
        rf_h.fit(_full_X_rf if _full_X_rf is not None else X_train, _full_y_rf, sample_weight=_full_sw_rf)
        holdout_preds["rf"] = rf_h.predict(X_holdout)
        del rf_h

    if "gp" not in test_preds and "gp" in all_models:
        from src.models.local_models import TanimotoGP
        gp_cfg = model_cfg.get("gp", {})
        max_ts = gp_cfg.get("max_train_size", 4000)
        logger.info(f"Full training: TanimotoGP (max_train_size={max_ts})...")
        gp_f = TanimotoGP(max_train_size=max_ts)
        gp_f.fit(fp_train.astype(np.float32), train_y)
        test_preds["gp"] = gp_f.predict(fp_test.astype(np.float32))
        del gp_f; gc.collect()

    _unimol_full_seeds = unimol_seeds if unimol_seeds else (
        [42] if "unimol" in all_models else []
    )
    for _um_seed in _unimol_full_seeds:
        _um_key = f"unimol_s{_um_seed}" if unimol_seeds else "unimol"
        if _um_key not in test_preds and _um_key in all_models:
            from src.models.unimol_model import UniMolModel
            stable_path = str(Path("logs") / f"{exp_id}_unimol_s{_um_seed}_full")
            logger.info(
                f"Full training: UniMol seed={_um_seed} "
                f"(epochs={unimol_cfg.get('epochs_full', 20)})..."
            )
            um_f = UniMolModel(
                epochs=unimol_cfg.get("epochs_full", 20),
                lr=unimol_cfg.get("lr", 1e-4),
                batch_size=unimol_cfg.get("batch_size", 16),
                seed=_um_seed,
                save_path=stable_path,
                use_conformer_resampling=unimol_cfg.get("use_conformer_resampling", False),
                n_train_conformers=unimol_cfg.get("n_train_conformers", 1),
                n_infer_conformers=unimol_cfg.get("n_infer_conformers", 1),
            )
            um_f.fit(_full_sm_unimol, _full_y_unimol)
            test_preds[_um_key] = um_f.predict(test_smiles)
            if holdout_smiles is not None:
                holdout_preds[_um_key] = um_f.predict(holdout_smiles)
            del um_f; gc.collect()

    # Holdout-only UniMol: reused test_preds from cache but holdout needs fresh training
    _um_holdout_key = f"unimol_s{_unimol_full_seeds[0]}" if unimol_seeds else "unimol"
    if (holdout_smiles is not None and "unimol" in all_models
            and _um_holdout_key not in holdout_preds and _unimol_full_seeds):
        from src.models.unimol_model import UniMolModel
        _seed_h = _unimol_full_seeds[0]
        stable_path_h = str(Path("logs") / f"{exp_id}_unimol_holdout")
        logger.info(f"Holdout eval: UniMol retrain (epochs={unimol_cfg.get('epochs_full', 20)})...")
        um_h = UniMolModel(
            epochs=unimol_cfg.get("epochs_full", 20),
            lr=unimol_cfg.get("lr", 1e-4),
            batch_size=unimol_cfg.get("batch_size", 16),
            seed=_seed_h,
            save_path=stable_path_h,
            use_conformer_resampling=unimol_cfg.get("use_conformer_resampling", False),
            n_train_conformers=unimol_cfg.get("n_train_conformers", 1),
            n_infer_conformers=unimol_cfg.get("n_infer_conformers", 1),
        )
        um_h.fit(_full_sm_unimol, _full_y_unimol)
        holdout_preds[_um_holdout_key] = um_h.predict(holdout_smiles)
        del um_h; gc.collect()

    # Save holdout predictions for fixed models (rf, tabpfn, unimol) so subsequent
    # hill-climb trials can skip their holdout retrains entirely.
    if _holdout_cache_path and holdout_smiles is not None:
        _reuse_models = set(config.get("reuse_oof", {}).get("models", []))
        _cache_to_save = {
            m: holdout_preds[m].tolist()
            for m in _reuse_models
            if m in holdout_preds
        }
        if _cache_to_save:
            try:
                import json as _json2
                Path(_holdout_cache_path).parent.mkdir(parents=True, exist_ok=True)
                with open(_holdout_cache_path, "w") as _cf:
                    _json2.dump(_cache_to_save, _cf)
                logger.info(f"Holdout pred cache: saved {sorted(_cache_to_save)} → {_holdout_cache_path}")
            except Exception as _exc2:
                logger.warning(f"Holdout pred cache save failed: {_exc2}")

    if "spherenet" not in test_preds and "spherenet" in all_models:
        from src.models.spherenet_model import SphereNetModel
        sn_save = str(Path("logs") / f"{exp_id}_spherenet_full.pt")
        logger.info(f"Full training: SphereNet (epochs={sn_cfg.get('epochs_full', 200)})...")
        sn_f = SphereNetModel(
            epochs=sn_cfg.get("epochs_full", 200),
            lr=sn_cfg.get("lr", 1e-3),
            batch_size=sn_cfg.get("batch_size", 32),
            n_conformers=sn_cfg.get("n_conformers", 3),
            cutoff=sn_cfg.get("cutoff", 5.0),
            hidden_channels=sn_cfg.get("hidden_channels", 128),
            num_layers=sn_cfg.get("num_layers", 4),
            device=device,
            seed=42,
            save_path=sn_save,
        )
        sn_f.fit(train_smiles, train_y, init_state_dict=spherenet_pretrain_sd)
        test_preds["spherenet"] = sn_f.predict(test_smiles)
        del sn_f; gc.collect()

    if "molformer" not in test_preds and "molformer" in all_models:
        from src.models.molformer_model import MolFormerModel
        mf_cfg = model_cfg.get("molformer", {})
        logger.info("Full training: MolFormer (frozen embeddings + RidgeCV)...")
        mf_f = MolFormerModel(
            batch_size=mf_cfg.get("batch_size", 64),
            device=mf_cfg.get("device", "cpu"),
            seed=42,
        )
        mf_f.fit(train_smiles, train_y)
        test_preds["molformer"] = mf_f.predict(test_smiles)
        del mf_f; gc.collect()

    # Cache test preds
    cache_out = {"test_smiles": test_smiles}
    cache_out.update({m: test_preds[m].tolist() for m in all_models if m in test_preds})
    cache_path = Path("logs") / f"{exp_id}_test_pred_cache.json"
    cache_path.write_text(json.dumps(cache_out))
    logger.info(f"Test preds cached → {cache_path}")

    # ------------------------------------------------------------------
    # 7b. Holdout evaluation (Analog Set 1)
    # ------------------------------------------------------------------
    if holdout_smiles is not None:
        logger.info("\n=== Holdout Evaluation (Analog Set 1) ===")
        missing_holdout = [m for m in all_models if m not in holdout_preds]
        if missing_holdout:
            logger.warning(
                f"Holdout preds missing for: {missing_holdout} — imputing with train_mean"
            )
            for m in missing_holdout:
                holdout_preds[m] = np.full(len(holdout_smiles), train_mean, dtype=np.float32)
        holdout_matrix = np.column_stack([holdout_preds[m] for m in all_models])
        holdout_stacked = stacker.predict(holdout_matrix)
        holdout_rae = float(rae(holdout_y, holdout_stacked, y_train_mean=train_mean))
        cv_holdout_gap = holdout_rae - stacked_rae
        logger.info(f"Holdout RAE (n={len(holdout_smiles)}): {holdout_rae:.4f}")
        logger.info(f"CV RAE:                                {stacked_rae:.4f}")
        logger.info(f"CV → Holdout gap:                      {cv_holdout_gap:+.4f}")
        results["holdout_rae"] = holdout_rae
        results["holdout_n"] = len(holdout_smiles)
        results["cv_holdout_gap"] = cv_holdout_gap
        results_path.write_text(json.dumps(results, indent=2))
        sw.update(holdout_rae=holdout_rae, cv_holdout_gap=cv_holdout_gap)

    # Skip submission if experiment was not accepted by CV RAE threshold
    if not accepted:
        sw.update(status="done", stage="holdout_eval_done", runtime_seconds=_elapsed(t0))
        return

    # ------------------------------------------------------------------
    # 8. Submission
    # ------------------------------------------------------------------
    sw.update(stage="building_submission")
    test_matrix = np.column_stack([test_preds[m] for m in all_models])
    final_preds = stacker.predict(test_matrix)
    logger.info(f"Ensemble: mean={final_preds.mean():.3f}, std={final_preds.std():.3f}")

    from src.ensemble.stack_and_submit import make_submission, validate_submission
    sub_path = make_submission(
        predictions=final_preds,
        test_df=test,
        output_dir="submissions",
        description=f"{exp_id} CV_RAE={stacked_rae:.4f}",
        val_rae=stacked_rae,
    )
    logger.info(f"Submission → {sub_path}")

    ok = validate_submission(str(sub_path), test)
    if not ok:
        logger.error("Submission validation FAILED — check above for details.")

    results["submission_path"] = str(sub_path)
    results["runtime_seconds"] = _elapsed(t0)
    results_path.write_text(json.dumps(results, indent=2))

    sw.update(
        status="done",
        stage="complete",
        runtime_seconds=_elapsed(t0),
        submission_path=str(sub_path),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Hill climb experiment runner")
    parser.add_argument("config", help="Path to experiment config JSON")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip all model training (test plumbing only)")
    parser.add_argument("--restack", action="store_true",
                        help="Skip CV; load OOF from logs/{exp_id}_oof_checkpoint.json "
                             "and redo stacking + full training only")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    config = json.loads(config_path.read_text())
    exp_id = config.get("experiment_id")
    if not exp_id:
        print("ERROR: config must have 'experiment_id'", file=sys.stderr)
        sys.exit(1)

    Path("logs").mkdir(exist_ok=True)
    log_path = Path("logs") / f"{exp_id}.log"

    # Dual logging: stdout + file
    handlers = [logging.StreamHandler(sys.stdout)]
    if not args.dry_run:
        handlers.append(logging.FileHandler(log_path, mode="w"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )
    logger = logging.getLogger("runner")
    logger.info(f"=== Experiment: {exp_id} ===")
    logger.info(f"Config: {config_path}")
    logger.info(f"Status file: logs/{exp_id}_status.json  (check this to see progress)")
    logger.info(f"Log file:    logs/{exp_id}.log")

    status_path = Path("logs") / f"{exp_id}_status.json"
    sw = StatusWriter(status_path, exp_id, config)
    ntfy_topic = os.environ.get("NTFY_TOPIC", "")
    if not ntfy_topic:
        logger.warning("NTFY_TOPIC not set — you will NOT receive completion notifications")
    t0 = time.time()

    try:
        run_experiment(config, sw, t0, dry_run=args.dry_run, restack=args.restack)
        elapsed = _elapsed(t0)
        cv_rae = sw._s.get("cv_rae")
        accepted = sw._s.get("accepted")

        logger.info(f"\n{'=' * 60}")
        logger.info(f"COMPLETE: {exp_id}")
        logger.info(f"  CV RAE:   {cv_rae}")
        logger.info(f"  Accepted: {accepted}")
        logger.info(f"  Runtime:  {elapsed:.0f}s ({elapsed / 3600:.1f}h)")
        logger.info(f"  Status:   logs/{exp_id}_status.json")
        logger.info(f"{'=' * 60}")

        if not args.dry_run:
            _append_competition_log(config, sw._s, elapsed / 3600)

        if ntfy_topic:
            emoji = "✅" if accepted else "🔶"
            send_ntfy(
                ntfy_topic,
                title=f"PXR {exp_id} done",
                message=(
                    f"{emoji} CV RAE={cv_rae:.4f} | "
                    f"{'ACCEPTED' if accepted else 'rejected'} | "
                    f"{elapsed / 3600:.1f}h\n"
                    f"Check: logs/{exp_id}_status.json"
                ),
                priority="high" if accepted else "default",
            )

    except Exception:
        elapsed = _elapsed(t0)
        tb = traceback.format_exc()
        err_line = tb.strip().split("\n")[-1]

        logger.error(f"\n{'!' * 60}")
        logger.error(f"FAILED: {exp_id}")
        logger.error(tb)
        logger.error(f"Runtime before failure: {elapsed:.0f}s")
        logger.error(f"Status file: logs/{exp_id}_status.json")
        logger.error(f"{'!' * 60}")

        sw.update(
            status="failed",
            stage="ERROR",
            error=err_line,
            traceback=tb,
            runtime_seconds=elapsed,
        )

        if ntfy_topic:
            send_ntfy(
                ntfy_topic,
                title=f"PXR {exp_id} FAILED",
                message=(
                    f"❌ {err_line}\n"
                    f"Runtime: {elapsed:.0f}s\n"
                    f"Check: logs/{exp_id}_status.json"
                ),
                priority="urgent",
            )

        sys.exit(1)


if __name__ == "__main__":
    main()
