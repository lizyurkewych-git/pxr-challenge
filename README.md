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
| 2 | `submission2_gbm_ensemble.py` | kNN + LGBM + XGBoost + RF (inv-RAE ensemble) | 0.6508 | 0.7962 |
| 3 | `submission3_chemprop.py` | Chemprop D-MPNN + kNN + LGBM + XGBoost + RF (inv-RAE ensemble) | 0.6249 | 0.7511 |
| 4 | `submission4_foundation_models.py` | + CheMeleon + ChemBERTa foundation embeddings (two GBM tracks) | 0.6437 | 0.7643 |
| 5 | `submission5_hts_pretrain.py` | HTS-pretrained Chemprop + scratch Chemprop + kNN + LGBM + RF (ElasticNet stacking) | 0.5609 | pending |

---

## Approach

### Submission 5 — HTS pre-training + ElasticNet stacking + Butina CV

Key changes from Sub 4:

- **HTS pre-training**: Chemprop is first pre-trained on ~5,500 PXR HTS compounds (21,003 rows at 4 concentrations → Hill sigmoid fit → pseudo-pEC50), then fine-tuned on the 4,139 primary DRC compounds. This gives the graph network a structural prior on PXR-active chemotypes before seeing precise activity data.
- **Hill sigmoid fitting** (`src/models/hts_pretrain.py`): fits `R(C) = Rmax × Cⁿ / (EC50ⁿ + Cⁿ)` (n fixed at 1.5) per compound. R² ≥ 0.5 and pEC50 ∈ [3.5, 9.0] are required; inactive and poorly-fit compounds are dropped rather than imputed.
- **Two Chemprop variants**: scratch (random init) and HTS-pretrained (lower fine-tuning LR = 5×10⁻⁴). Both use 2 seeds in CV, 3 seeds for final training, with predictions averaged.
- **ElasticNet stacking** (`ElasticNetStacker` in `stack_and_submit.py`): replaces the hand-tuned inverse-RAE weighting. A `StandardScaler + ElasticNetCV` meta-learner is trained on out-of-fold predictions from all five base models. The L1 component automatically zeros out weak contributors.
- **Butina cluster CV** (`ButinaKFold` in `validate.py`): replaces Murcko scaffold CV. Clusters compounds by ECFP4 Tanimoto similarity (threshold=0.4) using the Butina algorithm, then assigns entire clusters to folds. This gives a more conservative and realistic estimate of generalization to structurally novel analog series.
- **Foundation embeddings dropped**: CheMeleon and ChemBERTa were tested in Sub 4 and did not improve leaderboard RAE. Removed from Sub 5 to reduce noise.

CV RAE of 0.5609 is the best OOF result to date. The consistent ~0.12 unit CV-to-leaderboard gap across Subs 3–4 suggests an expected leaderboard RAE around 0.68.

### Submission 1 — kNN + LightGBM baseline

Ensemble of Tanimoto k-nearest-neighbor (kNN) and LightGBM regression using:
- ECFP4 binary fingerprints (2048 bits, radius=2)
- Count-based Morgan fingerprints
- RDKit physicochemical descriptors (~50)
- Mordred 2D descriptors (PCA-compressed to 200 dimensions)

### Submission 4 — Foundation model embeddings (CheMeleon + ChemBERTa)

Adds two pretrained molecular embedding models as additional feature blocks:
- **CheMeleon** (2048-dim): pretrained Chemprop D-MPNN fingerprints, checkpoint downloaded automatically from Zenodo
- **ChemBERTa** (384-dim): `DeepChem/ChemBERTa-77M-MTR` SMILES-based BERT, mean-pooled token embeddings

Both embedding blocks are PCA-compressed to 200 components and used to train a separate "foundation" GBM track alongside the traditional ECFP4+RDKit GBM track. Embeddings are cached to `data/embed_cache/` — the second run is instant.

Ensemble: Chemprop + kNN + LGBM_traditional + LGBM_foundation + XGB_foundation + RF_foundation (inverse-RAE weights).

Requires Python 3.11 (`chemprop>=2.1.0`, `transformers>=4.x`).

### Submission 3 — Chemprop D-MPNN + GBM ensemble

Adds a Chemprop v2 message-passing neural network (D-MPNN) trained directly on molecular graphs:
- Chemprop D-MPNN (hidden_size=300, depth=3, 100 epochs) with snapshot ensembling (last 5 epoch checkpoints averaged)
- MPS (Apple Silicon) acceleration
- Same kNN + LightGBM + XGBoost + RF models from Submission 2
- All five models combined via inverse-RAE weights
- Chemprop must be trained *before* Mordred multiprocessing to avoid a semaphore crash on macOS Python 3.11

Requires Python 3.11 (`chemprop>=2.1.0`).

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

Submission 3 (Chemprop + GBM ensemble, requires Python 3.11):
```bash
cd pxr-challenge-public
.venv311/bin/python scripts/submission3_chemprop.py
```

Submission 4 (foundation model embeddings, requires Python 3.11):
```bash
cd pxr-challenge-public
.venv311/bin/python scripts/submission4_foundation_models.py
```

Submission 5 (HTS pre-training + ElasticNet stacking, requires Python 3.11):
```bash
cd pxr-challenge-public
.venv311/bin/python scripts/submission5_hts_pretrain.py
```

All scripts will:
- Download all four data tiers from HuggingFace (cached to `data/hf_cache/`)
- Compute fingerprints and descriptors
- Run cross-validation (Butina cluster CV for Sub 5, scaffold CV for Subs 1–4) and print RAE per fold
- Train on the full training set
- Save a validated submission CSV to `submissions/`

---

## Project Structure

```
pxr-challenge-public/
├── scripts/
│   ├── baseline_submission.py           # Submission 1: kNN + LightGBM
│   ├── submission2_gbm_ensemble.py      # Submission 2: kNN + LGBM + XGBoost + RF ensemble
│   ├── submission3_chemprop.py          # Submission 3: Chemprop D-MPNN + GBM ensemble
│   ├── submission4_foundation_models.py # Submission 4: + CheMeleon + ChemBERTa embeddings
│   └── submission5_hts_pretrain.py      # Submission 5: HTS pre-training + ElasticNet stacking
├── src/
│   ├── data/
│   │   ├── load_data.py               # HuggingFace loading, SMILES canonicalization,
│   │   │                                inverse-variance weights, counter-assay flagging
│   │   └── cliff_analysis.py          # Activity cliff detection and annotation
│   ├── features/
│   │   └── feature_engineering.py     # ECFP4/6, FCFP4, RDKit, Mordred, Tanimoto utils
│   ├── models/
│   │   ├── local_models.py            # TanimotoKNN, TanimotoGP
│   │   ├── gbm_models.py              # LightGBM, XGBoost, RandomForest wrappers
│   │   ├── chemprop_model.py          # Chemprop v2 D-MPNN; transfer learning via init_state_dict
│   │   ├── hts_pretrain.py            # Hill sigmoid fitting → pseudo-pEC50 for HTS pre-training
│   │   └── foundation_embeddings.py   # CheMeleon + ChemBERTa pretrained embedders (Sub 4)
│   ├── evaluation/
│   │   └── validate.py                # RAE metric, bootstrap CI, ScaffoldKFold, ButinaKFold CV
│   └── ensemble/
│       └── stack_and_submit.py        # WeightedEnsemble, ElasticNetStacker, submission pipeline
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
