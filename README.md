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
| 7 | `submission7_tabpfn.py` | TabPFN+CheMeleon + Delta + HTS-pretrained Chemprop + kNN + RF (ElasticNet) | 0.5297 | 0.6358 |
| 6 | `submission6_delta.py` | Delta + HTS-pretrained Chemprop + kNN + LGBM + RF (ElasticNet) | 0.5481 | 0.6583 |
| 5 | `submission5_hts_pretrain.py` | HTS-pretrained Chemprop + scratch Chemprop + kNN + LGBM + RF (ElasticNet) | 0.5609 | 0.6615 |
| 4 | `submission4_foundation_models.py` | + CheMeleon + ChemBERTa foundation embeddings (two GBM tracks) | 0.6437 | 0.7643 |
| 3 | `submission3_chemprop.py` | Chemprop D-MPNN + kNN + LGBM + XGBoost + RF (inv-RAE ensemble) | 0.6249 | 0.7511 |
| 2 | `submission2_gbm_ensemble.py` | kNN + LGBM + XGBoost + RF (inv-RAE ensemble) | 0.6508 | 0.7962 |
| 1 | `baseline_submission.py` | kNN + LightGBM | ~0.76 | 0.7999 |

---

## Approach

### Submission 7 — TabPFN + CheMeleon in-context learning

Key changes from Sub 6:

- **TabPFN + CheMeleon** (`src/models/tabpfn_model.py`): a TabPFN v2 regressor backed by frozen 2048-dim CheMeleon embeddings compressed via PCA(200). TabPFN performs *in-context learning* — there is no gradient training step. At prediction time, the full `(X_train, y_train)` is passed through a transformer pretrained on millions of synthetic tabular datasets, which adaptively weights each training compound's contribution to each test prediction. This is fundamentally different from Sub 4's CheMeleon + LGBM approach: LGBM fits a fixed decision tree and forgets the training labels at test time; TabPFN sees all 4,139 training labels during every prediction. Motivated by Ben Hicham et al. (2025), which showed TabPFN + CheMeleon achieves up to 100% win rate on the MoleculeACE activity cliff benchmark — structurally identical to our analog-series test set.
- **Installation note**: `tabpfn==2.2.1` must be installed with `--no-deps` to avoid a `huggingface-hub` version conflict with `transformers` (TabPFN 2.x pins `huggingface-hub<1` but transformers requires `>=1.5.0`; the `--no-deps` flag keeps the existing compatible version).
- **ElasticNet result**: `tabpfn` was the dominant contributor (coef=0.42), surpassing `chemprop_hts` (0.33). `chemprop_scratch` received a small negative coefficient (−0.12), acting as a bias corrector. `lgbm` was zeroed out.

ElasticNet OOF RAE of **0.5297** is the best result to date (improved from 0.5481 in Sub 6).

---

### Submission 6 — Pairwise delta learning + concentration-aware HTS pre-training

Key changes from Sub 5:

- **Pairwise delta learning** (`src/models/delta_model.py`): a Chemprop D-MPNN is trained on all pairwise activity differences — input is (SMILES_i, SMILES_j), target is pEC50_i − pEC50_j. At inference, each test compound is anchored to its 10 nearest training neighbors: `pred(t) = mean(y_n + Δ(t, n))`. This directly optimizes for relative activity within scaffold families, which is exactly what the analog-series test set requires. Activity cliff pairs (Tanimoto ≥ 0.7, |Δ pEC50| ≥ 1.0) are oversampled 3× per epoch. Antisymmetry averaging at inference: `Δ(t,n) = 0.5 × (forward − reverse)`.
- **Concentration-aware HTS pre-training** (`prepare_hts_concentration_data` in `hts_pretrain.py`): instead of Hill-fitting 4 concentrations into a single pseudo-pEC50, all 4 dose-response points per compound are kept as separate training rows, with `log10[concentration_M]` passed as a molecule-level descriptor (`x_d`) to the Chemprop FFN. This gives ~21K training rows (4× more than Hill-fitting) with no R² acceptance filter that previously discarded borderline-active compounds.
- **Encoder-only transfer**: only `message_passing.*` weights are transferred from HTS pretraining to fine-tuning. The FFN is always re-initialized, preventing size mismatches when the pre-training uses `x_d` but fine-tuning does not.
- **ElasticNet result**: `chemprop_scratch` was assigned a zero coefficient and effectively dropped. The strongest contributors were `chemprop_hts` (0.38), `rf` (0.29), `delta` (0.20), `knn` (0.06), `lgbm` (0.01).

ElasticNet OOF RAE of 0.5481 is the best result to date (improved from 0.5609 in Sub 5).

---

### Submission 5 — HTS pre-training + ElasticNet stacking + Butina CV

Key changes from Sub 4:

- **HTS pre-training**: Chemprop is first pre-trained on ~5,500 PXR HTS compounds (21,003 rows at 4 concentrations → Hill sigmoid fit → pseudo-pEC50), then fine-tuned on the 4,139 primary DRC compounds.
- **Hill sigmoid fitting** (`src/models/hts_pretrain.py`): fits `R(C) = Rmax × Cⁿ / (EC50ⁿ + Cⁿ)` (n fixed at 1.5) per compound. R² ≥ 0.5 and pEC50 ∈ [3.5, 9.0] required; inactive and poorly-fit compounds are dropped.
- **Two Chemprop variants**: scratch (random init) and HTS-pretrained (lower fine-tuning LR = 5×10⁻⁴). Both use 2 seeds in CV, 3 seeds for final training, with predictions averaged.
- **ElasticNet stacking** (`ElasticNetStacker` in `stack_and_submit.py`): replaces hand-tuned inverse-RAE weighting. A `StandardScaler + ElasticNetCV` meta-learner is trained on out-of-fold predictions from all five base models.
- **Butina cluster CV** (`ButinaKFold` in `validate.py`): replaces Murcko scaffold CV. Clusters by ECFP4 Tanimoto similarity (threshold=0.4) using the Butina algorithm; entire clusters are held out per fold.
- **Foundation embeddings dropped**: CheMeleon and ChemBERTa (Sub 4) did not improve leaderboard RAE.

Largest single-submission leaderboard improvement to date: 0.7643 → 0.6615 (rank 109 → 64).

---

### Submission 4 — Foundation model embeddings (CheMeleon + ChemBERTa)

Adds two pretrained molecular embedding models as additional feature blocks:
- **CheMeleon** (2048-dim): pretrained Chemprop D-MPNN fingerprints, checkpoint downloaded from Zenodo
- **ChemBERTa** (384-dim): `DeepChem/ChemBERTa-77M-MTR` SMILES-based BERT, mean-pooled token embeddings

Both blocks are PCA-compressed to 200 components and used to train a separate "foundation" GBM track alongside the traditional ECFP4+RDKit track. Embeddings are cached to `data/embed_cache/`.

Ensemble: Chemprop + kNN + LGBM_traditional + LGBM_foundation + XGB_foundation + RF_foundation (inverse-RAE weights).

Result: foundation embeddings added noise rather than signal (leaderboard RAE increased vs Sub 3). Dropped in Sub 5.

---

### Submission 3 — Chemprop D-MPNN + GBM ensemble

Adds a Chemprop v2 message-passing neural network (D-MPNN) trained directly on molecular graphs:
- Chemprop D-MPNN (hidden_size=300, depth=3, 100 epochs) with snapshot ensembling (last 5 epoch checkpoints averaged)
- Same kNN + LightGBM + XGBoost + RF models from Submission 2
- All five models combined via inverse-RAE weights

Requires Python 3.11 (`chemprop>=2.1.0`).

---

### Submission 2 — 4-model inverse-RAE weighted ensemble

Adds XGBoost and Random Forest to the ensemble, with weights proportional to inverse CV RAE. New features:
- ECFP6 fingerprints (radius=3) added to the feature matrix
- Activity cliff reweighting: compounds in cliff pairs (Tanimoto ≥ 0.7, |ΔpEC50| ≥ 1.0) get 2× training weight
- Non-specific compound downweighting: counter-assay flagged compounds get 0.3× weight
- Inverse-variance weights from experimental SE

Uses scaffold-stratified 5-fold CV (Murcko scaffolds).

---

### Submission 1 — kNN + LightGBM baseline

Ensemble of Tanimoto k-nearest-neighbor (kNN) and LightGBM regression using:
- ECFP4 binary fingerprints (2048 bits, radius=2)
- Count-based Morgan fingerprints
- RDKit physicochemical descriptors (~50)
- Mordred 2D descriptors (PCA-compressed to 200 dimensions)

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

Submission 7 (TabPFN + CheMeleon + full Sub 6 stack, requires Python 3.11):
```bash
pip install "tabpfn==2.2.1" --no-deps
.venv311/bin/python scripts/submission7_tabpfn.py
```

Submission 6 (pairwise delta learning + concentration-aware HTS pre-training, requires Python 3.11):
```bash
.venv311/bin/python scripts/submission6_delta.py
```

Submission 5 (HTS pre-training + ElasticNet stacking, requires Python 3.11):
```bash
.venv311/bin/python scripts/submission5_hts_pretrain.py
```

Submission 4 (foundation model embeddings, requires Python 3.11):
```bash
.venv311/bin/python scripts/submission4_foundation_models.py
```

Submission 3 (Chemprop + GBM ensemble, requires Python 3.11):
```bash
.venv311/bin/python scripts/submission3_chemprop.py
```

Submission 2 ensemble:
```bash
python scripts/submission2_gbm_ensemble.py
```

Submission 1 baseline:
```bash
python scripts/baseline_submission.py
```

All scripts will:
- Download all four data tiers from HuggingFace (cached to `data/hf_cache/`)
- Compute fingerprints and descriptors
- Run cross-validation (Butina cluster CV for Subs 5–7, scaffold CV for Subs 1–4) and print RAE per fold
- Train on the full training set
- Save a validated submission CSV to `submissions/`

---

## Project Structure

```
pxr-challenge-public/
├── scripts/
│   ├── submission7_tabpfn.py            # Submission 7: TabPFN+CheMeleon in-context learning
│   ├── submission6_delta.py             # Submission 6: delta learning + conc-aware HTS pretrain
│   ├── submission5_hts_pretrain.py      # Submission 5: HTS pre-training + ElasticNet stacking
│   ├── submission4_foundation_models.py # Submission 4: + CheMeleon + ChemBERTa embeddings
│   ├── submission3_chemprop.py          # Submission 3: Chemprop D-MPNN + GBM ensemble
│   ├── submission2_gbm_ensemble.py      # Submission 2: kNN + LGBM + XGBoost + RF ensemble
│   └── baseline_submission.py           # Submission 1: kNN + LightGBM
├── src/
│   ├── data/
│   │   ├── load_data.py               # HuggingFace loading, SMILES canonicalization,
│   │   │                                inverse-variance weights, counter-assay flagging
│   │   └── cliff_analysis.py          # Activity cliff detection and annotation
│   ├── features/
│   │   └── feature_engineering.py     # ECFP4/6, FCFP4, RDKit, Mordred, Tanimoto utils
│   ├── models/
│   │   ├── tabpfn_model.py            # TabPFN v2 + CheMeleon PCA; in-context learning (Sub 7)
│   │   ├── delta_model.py             # Pairwise Δ pEC50 Chemprop; kNN-anchored inference
│   │   ├── chemprop_model.py          # Chemprop v2 D-MPNN; x_d support; encoder-only transfer
│   │   ├── hts_pretrain.py            # Hill fitting + concentration-aware HTS data prep
│   │   ├── local_models.py            # TanimotoKNN, TanimotoGP
│   │   ├── gbm_models.py              # LightGBM, XGBoost, RandomForest wrappers
│   │   └── foundation_embeddings.py   # CheMeleon + ChemBERTa pretrained embedders (Sub 4)
│   ├── evaluation/
│   │   └── validate.py                # RAE metric, bootstrap CI, ScaffoldKFold, ButinaKFold
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
| HTS screen | 21,003 | neg_log10_fdr, concentration_M | HTS pre-training signal |
| Test | 513 | SMILES, Molecule Name | Submission target |

## Submission Format

```
SMILES | Molecule Name | pEC50
```

513 rows, validated against the official checker before upload.
