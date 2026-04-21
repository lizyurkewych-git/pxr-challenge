# PXR Induction Activity Prediction

OpenADMET PXR Blind Challenge — Activity Prediction Track

**Target:** Predict pEC50 values for 513 blinded compounds from the Pregnane X Receptor (PXR/NR1I2) induction assay.  
**Metric:** Relative Absolute Error (RAE) — lower is better.  
**Timeline:** Phase 1 closes May 25, 2026 · Phase 2 closes July 1, 2026  
**Challenge portal:** huggingface.co/spaces/openadmet/pxr-challenge

---

## Approach

Ensemble of Tanimoto k-nearest-neighbor (kNN) and LightGBM regression using:
- ECFP4 binary fingerprints (2048 bits, radius=2)
- Count-based Morgan fingerprints
- RDKit physicochemical descriptors (~50)
- Mordred 2D descriptors (PCA-compressed to 200 dimensions)

Model selection uses scaffold-stratified 5-fold cross-validation (Murcko scaffolds), which better reflects the analog-series structure of the test set than random splits.

**Baseline scaffold CV result:** kNN RAE ≈ 0.760, ensemble RAE ≈ 0.74–0.78

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

```bash
cd pxr-challenge-public
python scripts/baseline_submission.py
```

This will:
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
│   └── baseline_submission.py    # End-to-end runner: load → features → CV → submit
├── src/
│   ├── data/
│   │   └── load_data.py          # HuggingFace loading, SMILES canonicalization,
│   │                               inverse-variance weights, counter-assay flagging
│   ├── features/
│   │   └── feature_engineering.py  # ECFP4/6, FCFP4, RDKit, Mordred, Tanimoto utils
│   ├── models/
│   │   ├── local_models.py       # TanimotoKNN, TanimotoGP
│   │   └── gbm_models.py         # LightGBM, XGBoost wrappers
│   ├── evaluation/
│   │   └── validate.py           # RAE metric, bootstrap CI, ScaffoldKFold CV
│   └── ensemble/
│       └── stack_and_submit.py   # Weighted ensemble, submission CSV generation
├── data/                         # Downloaded datasets (gitignored)
├── submissions/                  # Output submission CSVs (gitignored)
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
