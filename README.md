# PXR Induction Activity Prediction

OpenADMET PXR Blind Challenge — Activity Prediction Track

**Target:** Predict pEC50 values for 513 blinded compounds from the Pregnane X Receptor (PXR/NR1I2) induction assay.  
**Metric:** Relative Absolute Error (RAE) — lower is better.  
**Timeline:** Phase 1 closes May 25, 2026 · Phase 2 closes July 1, 2026  
**Challenge portal:** huggingface.co/spaces/openadmet/pxr-challenge

---

## Model Report | Phase 2 Submission

### Overview

The final submission is a five-model stacked ensemble trained on all 4,392 available PXR activity measurements (primary DRC + Phase 1 unblinded compounds). The ensemble is blended via an ElasticNet meta-learner fit on out-of-fold predictions. The key Phase 2 advance was Uni-Mol multi-conformer inference averaging, which produced the largest single improvement of the phase.

**Analog Set 1 held-out RAE: 0.5104** (253 Phase 1 compounds, held out throughout Phase 2 development)

---

### Data

**Primary DRC** (4,392 compounds after adding Phase 1 unblinded): Main training target. 316 reactive electrophiles (acrylamides, acrylates, aldehydes by SMARTS filter) were removed — these are assay artifacts rather than true PXR binders. PAINS and REOS compounds were intentionally retained; removing them hurt performance by degrading calibration at the inactive end of the activity landscape.

**HTS Screen** (21,003 compounds): Used exclusively for Chemprop pre-training via a 3-stage transfer learning pipeline (Tox21 → ChEMBL → HTS). Concentration was included as an input feature during pre-training.

**Counter-assay** (2,859 compounds, pEC50_null): Used as an auxiliary training *target* in multi-task Chemprop, not as an input feature. This distinction is critical — see the leakage incident below.

**HTChem crude assay** (external): Additional PXR activity measurements from crude cell extracts, augmented into all model training at sample_weight=0.5. Ablation confirmed that downweighting (vs. full weight) improves holdout generalization.

**Validation strategy**: Phase 1 used Butina scaffold 5-fold CV. From Phase 2 onward, Analog Set 1 (253 Phase 1 unblinded compounds) was held out as a fixed test set. These compounds are structurally closer to the Phase 2 analog series than to the Butina CV folds, making holdout RAE a substantially better generalization signal. Scaffold CV consistently underestimated true performance by ~0.024 RAE units.

---

### Ensemble Architecture

| Model | Ensemble Weight |
|-------|----------------|
| Uni-Mol 3D Fine-tuning | 0.413 |
| DeepDelta | 0.181 |
| TabPFN + CheMeleon | 0.170 |
| Random Forest | 0.111 |
| Multi-task Chemprop (seed 7) | 0.035 |
| Multi-task Chemprop (seed 42) | 0.027 |

Weights are ElasticNet stacker coefficients from the `phase2_final` run (trained on all 4,392 compounds).

**1. Uni-Mol 3D Fine-tuning (~0.41 ensemble weight)**

Pre-trained Uni-Mol transformer fine-tuned on PXR pEC50. RDKit ETKDGv3 conformers are generated locally to avoid Uni-Mol's built-in ConformerGen, which crashes on embedding failures. The critical Phase 2 innovation: *inference averaging over 8 independently-seeded conformers per molecule*. Each forward pass uses a different 3D geometry; predictions are averaged before stacking. This improved Analog Set 1 holdout RAE from 0.5163 → 0.5104 — the largest Phase 2 gain. The stacker responded by increasing Uni-Mol's weight from 0.37 → 0.44 in the held-out development run; the final retrain on 4,392 compounds settled at 0.41, confirming genuine quality improvement rather than variance reduction.

**2. Multi-task Chemprop D-MPNN (~0.06 ensemble weight)**

CheMeleon is a molecular foundation model pre-trained on large-scale chemical data that provides the initial atom and bond representations for this Chemprop architecture. The CheMeleon-initialized D-MPNN was then pre-trained in three stages — Tox21 → ChEMBL → HTS screen — before fine-tuning on PXR data. This staged transfer approach progressively narrows the domain from general toxicity to broad bioactivity to PXR-specific HTS signal, with concentration included as an input feature during the HTS stage.

Fine-tuning used three output heads: pEC50 (primary), pEC50_null (counter-assay selectivity), and Emax (effect size). Two random seeds (42, 7) ensembled. MAE loss used throughout. The stacker assigned modest weight to this component in Phase 2 — Uni-Mol dominated on the analog test set — but Chemprop contributed meaningfully to ensemble diversity in earlier phases.

**3. DeepDelta (~0.18 ensemble weight)**

Pairwise delta model. For each query compound, the k=5 nearest neighbors by Tanimoto similarity are identified and a model is trained on activity differences. Particularly well-suited to tight analog series where small structural perturbations produce predictable activity changes.

**4. TabPFN + CheMeleon Embeddings (~0.17 ensemble weight)**

CheMeleon embeddings are used here independently of Chemprop — the foundation model is applied directly to each molecule to produce a fixed-length representation, which is passed to TabPFN as input features. TabPFN is a transformer-based in-context learner that treats the entire training set as context at inference time, making it particularly effective for small-to-medium tabular datasets. This combination of chemistry-aware embeddings with an in-context learner was a strong standalone performer (CV RAE ~0.57). TabICL was evaluated as a replacement for TabPFN but showed consistent holdout regression across multiple trials and was excluded.

**5. Random Forest (~0.11 ensemble weight)**

RDKit2D physicochemical descriptors combined with PXR crystal ligand Tanimoto similarity features — similarity scores to co-crystallized ligands from known PXR crystal structures. The crystal features provided a consistent small gain by anchoring predictions to structurally validated active geometries. ECFP4/ECFP6 were evaluated but provided less signal than RDKit2D for this target.

**Meta-learner**: ElasticNet stacking (l1_ratio=0.7, 5-fold inner CV). Extensive l1_ratio tuning confirmed 0.7 was already near-optimal — none of four alternatives (0.2, 0.4, 0.6, 0.9) improved holdout RAE.

---

### What Didn't Work

**Counter-assay delta as input feature (critical leakage)**: One submission used pEC50_null as a direct RF/TabPFN input feature. This achieved a deceptive CV RAE of 0.470 but a leaderboard RAE of 0.845 — the worst result of the competition run. The counter-assay data is available for ~64% of training compounds but 0% of the 513 blinded test compounds; all test predictions received delta=0, collapsing the model. Multi-task Chemprop uses pEC50_null as a training target rather than an input, which is fundamentally safe. This incident is shared in detail because the failure mode was subtle and others may encounter it.

**TabICL**: Improved CV RAE by ~0.005 but hurt holdout RAE by ~0.008 across multiple trials. Excluded.

**Series-aware DeepDelta**: Modification to make the delta model aware of analog series membership. Three independent trials all showed holdout regression. Excluded.

**Training conformer resampling (Uni-Mol)**: Selecting one random conformer per molecule before each fit() call — an approximation of true per-epoch resampling — introduced training noise without benefit. Holdout 0.5128 vs inference-averaging-only 0.5104. Zero-cost inference averaging was strictly superior.

---

### Acknowledgements

Thank you to the OpenADMET organizers for designing a well-structured challenge with meaningful data across multiple assay tiers and a clean validation framework. Thank you also to the community members who shared their insights along the way — with special thanks to [discoverybytes](https://github.com/discoverybytes/openadmet-pxr-blind-challenge/tree/main/activity-prediction), dargason, and [JacksonBurn](https://gist.github.com/JacksonBurns/94cb5e7dda4d72bd876c947df92c5147). I enjoyed learning from everyone!

---

*Phase 2 submission generated by `scripts/run_experiment.py` with config `phase2_final.json`. Submission validated against the official OpenADMET validator prior to upload.*

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
