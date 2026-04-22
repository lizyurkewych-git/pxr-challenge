# PXR Induction Activity Prediction

OpenADMET PXR Blind Challenge — Activity Prediction Track

**Target:** Predict pEC50 values for 513 blinded compounds from the Pregnane X Receptor (PXR/NR1I2) induction assay.  
**Metric:** Relative Absolute Error (RAE) — lower is better.  
**Timeline:** Phase 1 closes May 25, 2026 · Phase 2 closes July 1, 2026  
**Challenge portal:** huggingface.co/spaces/openadmet/pxr-challenge

---

## Submissions

| # | Script | Models | CV RAE | Leaderboard RAE |
|---|--------|--------|--------|-----------------|
| 1 | `baseline_submission.py` | kNN + LightGBM | ~0.76 | 0.7999 |
| 2 | `submission2_gbm_ensemble.py` | kNN + LGBM + XGBoost + RF (inv-RAE ensemble) | 0.6508 | TBD |

---

## Approach

### Submission 1 — kNN + LightGBM baseline

Ensemble of Tanimoto k-nearest-neighbor (kNN) and LightGBM regression using:
- ECFP4 binary fingerprints (2048 bits, radius=2)
- Count-based Morgan fingerprints
- RDKit physicochemical descriptors (~50)
- Mordred 2D descriptors (PCA-compressed to 200 dimensions)

### Submission 2 — 4-model inverse-RAE weighted ensemble

Adds XGBoost and Random Forest to the ensemble, with weights proportional to inverse CV RAE (stronger models get more vote). New features:
- ECFP6 fingerprints (radius=3) added to feature matrix
- Activity cliff sample reweighting: compounds in cliff pairs (Tanimoto ≥ 0.7, |ΔpEC50| ≥ 1.0) get 2× training weight
- Non-specific compound downweighting: compounds flagged by counter-assay get 0.3× weight
- Inverse-variance weights from experimental SE (measurement quality)

Model selection uses scaffold-stratified 5-fold cross-validation (Murcko scaffolds), which better reflects the analog-series structure of the test set than random splits.

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
brew install libomp  # required for LightGBM on macOS
```

### 2. Set HuggingFace token (optional, suppresses rate-limit warnings)

```bash
export HF_TOKEN=your_token_here
```

### 3. Generate a submission

Submission 1 baseline:
```bash
cd pxr-challenge-public
python scripts/baseline_submission.py
```

Submission 2 ensemble:
```bash
cd pxr-challenge-public
python scripts/submission2_gbm_ensemble.py
```

Both scripts will:
- Download all four data tiers from HuggingFace (cached to `data/hf_cache/`)
- Compute fingerprints and descriptors
- Run scaffold-stratified 5-fold CV and print RAE per fold
- Train on the full training set
- Save a validated submission CSV to `submissions/`

---

## Project Structure

```
pxr-challenge-public/
├── scripts/
│   ├── baseline_submission.py         # Submission 1: kNN + LightGBM
│   └── submission2_gbm_ensemble.py    # Submission 2: kNN + LGBM + XGBoost + RF ensemble
├── src/
│   ├── data/
│   │   ├── load_data.py               # HuggingFace loading, SMILES canonicalization,
│   │   │                                inverse-variance weights, counter-assay flagging
│   │   └── cliff_analysis.py          # Activity cliff detection and annotation
│   ├── features/
│   │   └── feature_engineering.py     # ECFP4/6, FCFP4, RDKit, Mordred, Tanimoto utils
│   ├── models/
│   │   ├── local_models.py            # TanimotoKNN, TanimotoGP
│   │   └── gbm_models.py              # LightGBM, XGBoost, RandomForest wrappers
│   ├── evaluation/
│   │   └── validate.py                # RAE metric, bootstrap CI, ScaffoldKFold CV
│   └── ensemble/
│       └── stack_and_submit.py        # Weighted ensemble, submission CSV generation
├── data/                              # Downloaded datasets (gitignored)
├── submissions/                       # Output submission CSVs (gitignored)
└── requirements.txt
```

---

## Data Tiers

All data loaded automatically from `openadmet/pxr-challenge-train-test` on HuggingFace.

| Tier | n | Key columns | Usage |
|------|---|-------------|-------|
| Primary DRC | 4,139 | pEC50 (1.61–7.55), Emax, SE | Primary training target |
| Counter-assay | 2,859 | pEC50_null | Non-specific compound detection |
| HTS screen | 21,003 | neg_log10_fdr, concentration_M | Auxiliary signal |
| Test | 513 | SMILES, Molecule Name | Submission target |

## Submission Format

```
SMILES | Molecule Name | pEC50
```

513 rows, validated against the official checker before upload.
