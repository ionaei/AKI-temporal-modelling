# GOSH model development and evaluation

This folder contains the pipeline used to build the AKI cohort at Great Ormond Street Hospital (GOSH), train and select the models, evaluate them on the held-out test set and the six-month temporal hold-out, run the feature ablations and compute the explainability analyses. It runs inside the GOSH Digital Research Environment (DRE), where the raw extract is held.

All site-specific details (file paths, column names, code lists, units) are in `configs/sites/gosh.yaml`. Everything after extraction is site-agnostic, so the same code runs on any dataset mapped to the canonical tables described below.

## Installation

```bash
pip install -r requirements.txt
```

The paper used Python 3.10, PyTorch 2.5.1 and scikit-learn 1.7.2. Models train on CPU or a single GPU.

Check the installation without real data:

```bash
pytest -q                      # unit tests
bash tools/run_synthetic.sh    # whole pipeline on synthetic data, a few minutes on CPU
```

## Running the analyses

```bash
bash run_gosh.sh               # data preparation, all models, hold-out, ablations, explainability
bash run_paper_analyses.sh     # tables and figures for the selected model
```

Every command accepts `--resume`, which skips work whose outputs already exist and continues an interrupted training run from its last completed epoch. Every command also writes a log to `logs/<command>_<site>_<timestamp>.log`.

## Pipeline

All stages are sub-commands of `python -m aki.cli`. Each stage reads the previous stage's output from `data/<site>/` and writes its own, so stages can be re-run independently.

| Stage | Command | Output |
|---|---|---|
| 1. Extract | `extract --site gosh` | `data/gosh/canonical/{episodes,labs,vitals,meds,rrt}.parquet`, `unit_report.csv` |
| 2. Label | `label --site gosh` | `data/gosh/labelled/stays_labelled.parquet`, `label_summary.json` |
| 3. Windows | `windows --site gosh` | `data/gosh/windows/windows.parquet`, `feature_groups.json`, `availability.csv` |
| 4. Split | `split --site gosh` | `data/gosh/labelled/split.csv`, `split_meta.json` |
| 5. Train | `train --site gosh --model transformer --loss focal --name <run> --seeds 0 1 2 3 4` | `runs/<run>/seed_<n>/` |
| 6. Evaluate | `evaluate --run runs/<run> --site gosh --split test` | `runs/<run>/seed_<n>/eval_gosh_test/` |
| 7. Aggregate | `aggregate --run runs/<run>` | `runs/<run>/summary_across_seeds.csv` (mean and SD over seeds) |
| 8. Temporal hold-out | `deploy --run runs/<run>/seed_<n> --site gosh --split holdout` | `runs/<run>/seed_<n>/deploy_gosh_holdout/` |
| 9. Explain | `explain --run runs/<run>/seed_<n> --site gosh --split test` | `runs/<run>/seed_<n>/explain_gosh_test/` |

Other commands: `thresholds` re-derives the operating points of trained runs without retraining, and `core` writes the set of features observed at every listed site (`configs/core_features_<sites>.json`, optional `--core` argument of `train`).

### 1. Extraction (`aki/adapters/`)

A site adapter reads the raw extract and writes five canonical tables:

| Table | Columns |
|---|---|
| episodes | subject_id, start, end, dob, sex, unit |
| labs | subject_id, time, concept, value, unit, code |
| vitals | subject_id, time, concept, value, unit, code |
| meds | subject_id, start, end, name, route |
| rrt | subject_id, time |

`adapters/gosh.py` reads the GOSH extract; `adapters/mimic_like.py` reads databases with the MIMIC-III schema (PIC, MIMIC-III). Codes are mapped only through the explicit lists in the site YAML (no substring or fuzzy matching). Units are converted to one canonical unit per concept and implausible values are dropped (`aki/concepts.py`). Every dropped or converted row is counted by concept and source code in `unit_report.csv`. At GOSH, episodes are the ICU ward stays (`ward_stays` block); vital signs are not available in the extract, so no vital-sign features are built.

`tools/discover_codes.py --site gosh` lists every laboratory code with its unit, count and value centiles, and the most frequent medication names, to audit the mappings.

### 2. Cohort and AKI labels (`aki/cohort.py`, `aki/labelling.py`)

- ICU episodes of the same patient separated by 4 hours or less are merged into one stay. Stays of patients aged 0 to < 18 years at admission with at least 6 hours in ICU are kept.
- AKI onset is the time of the first creatinine result in the stay that meets either KDIGO criterion:
  1. a rise of at least 26.5 µmol/L within 48 hours;
  2. a value at least 1.5 times the reference value.
- Reference value: the lowest creatinine in the preceding 7 days, including results from the ward before ICU admission and from earlier encounters. If there is none, the lowest value in the preceding 90 days (30 days for samples taken before 1 year of age). If there is no earlier value, the ratio criterion cannot be met: population reference values are not imputed (`allow_population_baseline: false`).
- Stage: peak KDIGO stage within 7 days of onset. Stage 3 also at creatinine ≥ 353.6 µmol/L or renal replacement therapy (RRT). Severe AKI = stage 2 or 3.
- Neonatal-modified KDIGO for samples taken before 28 days of age: the reference is the lowest previous value, stage 3 at ≥ 221 µmol/L, and samples from the first 72 hours of life cannot trigger onset.
- Exclusions (kept in `stays_labelled.parquet` with `exclude_reason`):
  - `aki_before_admission`: KDIGO criteria met in the 7 days before ICU admission;
  - `aki_on_arrival`: onset within 24 hours of ICU admission;
  - `rrt_before_onset`, `recent_prior_rrt`: RRT before onset or in the 90 days before admission;
  - `fewer_than_2_creatinine`: no creatinine in the stay, or fewer than two in total including possible references before admission.

`label_summary.json` reports counts by exclusion reason, AKI stage, trigger and reference source.

### 3. Windows and features (`aki/windows.py`)

Each stay is divided into non-overlapping 6-hour windows from ICU admission. Window *k* covers [admission + 6k h, admission + 6(k+1) h) and its features use only events timestamped before the end of the window. The label of a window is 1 if AKI onset occurs within the following 24 hours. Windows stop at onset (AKI stays) or discharge, and stays are capped at 1,200 windows (300 days).

Feature names start with a prefix that defines the feature group used for ablations:

| Prefix | Group | Content |
|---|---|---|
| `lab_creatinine_*`, `renal_*` | creatinine | last, min, max and change of creatinine; ratio of the latest in-stay value to the lowest value in the preceding 7 days; rise over the preceding 48 hours |
| `lab_*` | lab | the same summaries for the other laboratory and blood-gas analytes |
| `med_*` | med | per drug group: exposure in the window, days exposed in the last 72 h, hours since last dose; number of concurrent nephrotoxic groups; high nephrotoxin exposure |
| `proc_*` | process | number of measurements of each analyte in the window and hours since its last measurement |
| `vit_*`, `sup_*` | vital, support | vital signs and organ support (not available at GOSH) |
| `static_*` | static | log age, paediatric age band, sex |
| `time_*` | time | log hours since ICU admission |

Values are carried forward within a stay for an analyte-specific number of hours (`carry_h` in `aki/concepts.py`). Medication groups, their nephrotoxicity, carry-over duration and excluded routes are defined in `aki/concepts.py` (`MED_GROUPS`).

### 4. Splits (`aki/split.py`)

- Temporal hold-out: stays admitted in the six months before the last eligible ICU admission form the `holdout` split, which is never used during development. Development stays still in ICU at the cutoff are dropped.
- Development stays are split by patient into train (70%), validation (15%) and test (15%), stratified by whether the patient ever had AKI (seed 42). A patient never appears in more than one split.

`split_meta.json` records the cutoff, the number of hold-out stays and AKI stays, and the split summary.

### 5. Training (`aki/features.py`, `aki/models.py`, `aki/train.py`)

- Preprocessing is fitted on the training split and saved with each run (`preprocessor.joblib`): age-dependent physiology (creatinine, urea, phosphate, haemoglobin, vital signs) is normalised against age-band medians and robust standard deviations, all features are standardised, clipped at ±10 SD, and missing values set to 0 (the training mean).
- Models (`configs/common.yaml`, `model` and `training`):
  - GRU and LSTM: input projection, two recurrent layers (hidden size 128), classification head at every window;
  - Transformer: causal encoder, two layers, four heads, sinusoidal encoding of the window index since admission, attention limited to the preceding 120 windows (30 days), trained on random crops of up to 240 windows;
  - logistic regression (L2, C = 0.1) and histogram gradient boosting (up to 400 iterations, learning rate 0.05, early stopping on 10% of the training data), both with balanced class weights and window-level features.
- Loss: weighted binary cross-entropy (positive weight = square root of the negative-to-positive window ratio, `--loss bce`) or focal loss (α = 0.75, γ = 2, `--loss focal`).
- AdamW (learning rate 5 × 10⁻⁴, weight decay 10⁻⁴), learning rate halved after 4 epochs without improvement in validation AUPRC, gradient clipping at 1.0, up to 60 epochs with early stopping after 10 epochs; the checkpoint with the highest validation AUPRC is kept.
- After training, predicted risks are recalibrated on the validation set (Platt scaling, `platt.json`) and every operating point is chosen on the validation set (`thresholds.json`).

Each seed folder contains `run_config.json` (all settings and the git commit), `model.pt` and `model_meta.json` (sequence models) or `model.joblib` (baselines), `preprocessor.joblib`, `platt.json`, `thresholds.json`, `val_threshold_sweep.csv` and `history.csv`.

### 6. Evaluation (`aki/metrics.py`)

`evaluate` applies a trained run, with its calibration and thresholds unchanged, to a split and writes:

| File | Content |
|---|---|
| `summary.csv` | one column per operating point; all metrics below |
| `bootstrap_<operating point>.csv` | 95% CIs from a patient-level bootstrap (`--n_boot`, default 500) |
| `horizon_auroc.csv` | AUROC of windows 0-6, 6-12, 12-24, 24-48, 48-72 and 72-120 h before onset against all windows of non-AKI stays |
| `reliability.csv`, `decision_curve.csv` | calibration by decile of predicted risk; net benefit across threshold probabilities |
| `threshold_sweep.csv` | alert metrics across thresholds |
| `subgroups.csv` | results by age band (first operating point in `common.yaml`) |
| `predictions.parquet` | window-level predictions with pseudonymised IDs (keep in the DRE) |

Metric names used in the output files and their meaning in the paper:

| Column | Paper |
|---|---|
| `auroc`, `auprc`, `prevalence` | window-level AUROC, AUPRC and AKI prevalence (AUPRC× = `auprc / prevalence`) |
| `brier`, `cal_slope`, `cal_intercept` | Brier score, calibration slope and calibration-in-the-large |
| `sens@24h`, `sens@48h` | sensitivity within 24 / 48 h: AKI stays with at least one alert in the 24 / 48 h before onset |
| `precision@24h`, `precision@48h` | precision within 24 / 48 h: alerts followed by onset within 24 / 48 h |
| `legacy_sens_ever` | stay-level sensitivity (detection rate): AKI stays with at least one alert before onset |
| `legacy_spec_stay` | stay-level specificity: non-AKI stays never alerted |
| `lead@48h_median` | lead time: median hours before onset of the earliest alert within 48 h |
| `alerts_per_100_patient_days` | alert burden |
| `sens_severe@48h` | sensitivity within 48 h for severe AKI (stage 2-3) |

Operating points (`configs/common.yaml`, all chosen on the validation set):

| Name | Rule |
|---|---|
| `orig_spec_at_sens0.75` | **primary**: maximum stay-level specificity with stay-level sensitivity ≥ 0.75 |
| `orig_youden` | maximum stay-level Youden's J |
| `sens48_0.80` | sensitivity within 48 h ≥ 0.80 |
| `alerts_10_per_100pd` | at most 10 alerts per 100 patient-days |

### 7. Temporal hold-out (`deploy`)

`deploy` evaluates a frozen run (weights, calibration and thresholds unchanged) on the `holdout` split and also writes `monthly.csv` (metrics per month of ICU admission at the operating point `--op`, default `orig_spec_at_sens0.75`) and `overlap.json` (hold-out stays of patients seen during training).

### 8. Explainability (`aki/explain.py`)

Integrated gradients (32 steps, all-zero baseline, i.e. the training mean after preprocessing) explain the logit of the last window before onset for 200 AKI stays and of a random window for 200 non-AKI stays from the test set, summed over all preceding windows. Outputs: `ig_by_feature_signed.csv` (signed mean attribution per class, mean absolute attribution per class and `importance_diff` = AKI minus non-AKI), `ig_share_by_group.csv` and `ig_by_lag.csv`. `tools/explain_all.py` runs this for every sequence-model run and seed.

## Paper tables and figures

`run_paper_analyses.sh` runs these for the selected model (`gosh_transformer_focal_all`). All outputs go to `results/`.

| Paper | Source |
|---|---|
| Table 1 | `tools/table1.py`, `tools/ethnicity_table.py` |
| Table 2, Supplementary Tables S3, S4 | `evaluate` + `aggregate`; collected by `tools/compare_runs.py` |
| Table 3 | `tools/stage_performance.py` |
| Figure 2 | `tools/alert_curves.py` |
| Figure 3 | `tools/traffic_light.py` |
| Figure 4, Supplementary Table S5 | `deploy` (`monthly.csv`), `tools/holdout_demographics.py`, `tools/month_diagnostics.py` |
| Figure 5 | `explain` (`importance_diff`); collected by `tools/compare_runs.py` |
| Calibration and decision curves (Supplementary Figure S4) | `evaluate` (`reliability.csv`, `decision_curve.csv`) |
| Comparison with the Renal Angina Index | `tools/renal_angina.py` |
| Comparison with logistic regression | `tools/paired_compare.py` |
| Label validation (Supplementary Figure S2) | `tools/label_validation.py` |
| Excluded stays (Supplementary Table S2) | `tools/excluded_profile.py` |
| Clustering (Supplementary Figure S3) | `tools/cluster_dataset.py` |

## Configuration

- `configs/common.yaml`: cohort, labelling, windows, features, splits, model, training and evaluation settings shared by all sites.
- `configs/sites/gosh.yaml`: raw file paths, column names and code lists for GOSH. A site file can override any common setting under `common_overrides:` and can inherit from another site with `extends:`.
- `configs/sites/synth_*.yaml`: synthetic data used by `tools/run_synthetic.sh`.

To run the pipeline on another dataset, write a site YAML for it. If the data follow the MIMIC-III schema, use `adapter: mimic_like` (see `../external_validation/configs/sites/`); otherwise add an adapter in `aki/adapters/` that writes the five canonical tables.

## Outputs that can leave the DRE

`data/` and `predictions.parquet` contain pseudonymised identifiers and stay in the DRE. `label_summary.json`, `split_meta.json`, `unit_report.csv`, the evaluation tables listed above, `monthly.csv`, the explainability tables and the files written by `tools/` to `results/` are aggregates. Check small cells against the DRE output rules before export. A trained run can be exported for external validation without patient data: `model.pt`, `model_meta.json` or `model.joblib`, `preprocessor.joblib`, `platt.json`, `thresholds.json` and `run_config.json`.
