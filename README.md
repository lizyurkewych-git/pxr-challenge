# PXR Induction Activity Prediction

OpenADMET PXR Blind Challenge — Activity Prediction Track

**Target:** Predict pEC50 values for 513 blinded compounds (two analog sets) from the Pregnane X Receptor (PXR/NR1I2) induction assay.  
**Metric:** Relative Absolute Error (RAE) — lower is better.  
**Timeline:** Phase 1 closes May 25, 2026 · Phase 2 closes July 1, 2026

---

## Quickstart

### 1. Set up environment

```bash
conda env create -f environment.yml
conda activate pxr
```

Or with pip:

```bash
pip install -r requirements.txt
```

Verify MPS (Apple M4 GPU):

```python
import torch
print(torch.backends.mps.is_available())  # should print True
```

### 2. Run Week 1 baseline (kNN + LightGBM)

```bash
cd pxr-challenge
python scripts/baseline_submission.py
```

This will:
- Download the dataset from HuggingFace (cached to `data/hf_cache/`)
- Run scaffold-stratified 5-fold CV to validate internally
- Generate a submission CSV in `submissions/`
- Log the submission to `submissions/submissions_log.csv`

### 3. EDA notebook

```bash
jupyter notebook notebooks/01_eda.ipynb
```

---

## Project Structure

```
pxr-challenge/
├── notebooks/
│   └── 01_eda.ipynb              # EDA: target distributions, UMAP, AC analysis
├── src/
│   ├── data/
│   │   ├── load_data.py          # Dataset loading, preprocessing, tier merging
│   │   ├── cliff_analysis.py     # Activity cliff detection and annotation
│   │   └── external_data.py      # ChEMBL / TDC / PubChem data fetching (Week 3+)
│   ├── features/
│   │   └── feature_engineering.py  # ECFP4/6, FCFP4, RDKit, Mordred, Tanimoto utils
│   ├── models/
│   │   ├── local_models.py       # TanimotoKNN (M8), TanimotoGP (M9)
│   │   ├── gbm_models.py         # LightGBM (M6), XGBoost (M7), RandomForest
│   │   ├── chemprop_multitask.py # Chemprop D-MPNN models M1–M4 (Week 3+)
│   │   └── deepdelta.py          # Pairwise delta-learning model M5 (Week 3+)
│   ├── ensemble/
│   │   └── stack_and_submit.py   # WeightedEnsemble, CaruanaEnsemble, submission
│   └── evaluation/
│       └── validate.py           # RAE metric, bootstrap CI, ScaffoldKFold CV
├── scripts/
│   └── baseline_submission.py    # Week 1: kNN + LightGBM baseline runner
├── data/                         # Downloaded datasets (gitignored)
├── models/checkpoints/           # Saved model checkpoints (gitignored)
├── submissions/                  # Submission CSVs + log
├── requirements.txt
└── environment.yml
```

---

## Implementation Roadmap

| Week | Target | Models | Expected RAE |
|------|--------|--------|-------------|
| 1 (≤Apr 22) | First leaderboard submission | kNN + LightGBM | 0.75–0.85 |
| 2 (≤Apr 29) | Full feature matrix + XGBoost | GBM ensemble | 0.65–0.75 |
| 3–5 (≤May 18) | CheMeleon Chemprop + DeepDelta | GNN + pairwise delta | 0.55–0.70 |
| 6 (≤May 25) | GP gating + Caruana ensemble | Full 10-model ensemble | 0.50–0.65 |
| 7–10 (≤Jul 1) | Retrain with Set 1 data + stacking | Phase 2 meta-learner | 0.35–0.50 |

---

## Key Design Decisions

**Why activity cliff-specific methods?**  
The test set is explicitly designed as lead-optimization analog expansions (~513 analogs of 46 potent compounds). Activity cliffs are common in this regime. Standard global QSAR models systematically fail on cliffs because small structural changes (e.g., single methyl group) cause >1 log-unit potency shifts that are invisible to global descriptors.

**Why Chemprop + DeepDelta together?**  
Chemprop (D-MPNN) learns global structure-activity relationships; DeepDelta exploits the known pEC50 of structurally similar training compounds to predict the potency *difference*. For analog series prediction, DeepDelta is often the strongest individual model.

**Why scaffold-stratified CV?**  
Random CV overestimates performance for analog-series prediction. Scaffold-stratified holdout forces the model to generalize to structurally distinct chemical series — the same challenge posed by the test set.

---

## Data Tiers

| Tier | n | Key columns | Usage |
|------|---|-------------|-------|
| Primary DRC | 4,140 | pEC50 (1.61–7.55), Emax, SE | Primary training target |
| Counter-assay | 2,860 | pEC50_null, Emax_null | Auxiliary task; non-specific compound flagging |
| HTS screen | 21,000 | neg_log10_fdr, cohens_d | Pretraining / auxiliary task (weight 0.1×) |
| Test | 513 | SMILES, OCNT_ID only | Submission |

---

## Submission Log

See `submissions/submissions_log.csv` for all submissions with internal RAE, leaderboard RAE, and model descriptions.

After getting a leaderboard score, update the `leaderboard_rae` column manually.
