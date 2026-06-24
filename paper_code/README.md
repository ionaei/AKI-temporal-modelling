# paper_code

Full research pipeline for the AKI temporal modelling study, from
raw EHR extraction through to model evaluation, explainability, and
silent deployment. Organised into four stages, run in order:

```
paper_code/
├── data_exploration/   Cohort and feature-space exploratory analysis
├── preprocessing/      Raw EHR → AKI-labelled, windowed, matched dataset
├── training/           Model architectures, training, evaluation, explainability
└── scripts/            Shell wrappers that chain the above end-to-end
```

---

## `data_exploration/`

Exploratory analyses run on the processed cohort, independent of any
specific model.

| File | Purpose |
|---|---|
| `demographics_exploration.ipynb` | Cohort demographic and clinical characteristics (age, sex, AKI stage, ethnicity) for GOSH and PIC, used to build Table 1. |
| `umap_tsne_pca.ipynb` | UMAP, t-SNE, and PCA of admission-level feature vectors comparing GOSH vs PIC distributions — generates the cross-site separation figure used in the discussion of zero-shot transfer failure. |

---

## `preprocessing/`

Converts raw GOSH and PIC EHR exports into the model-ready windowed
dataset. **Run in numbered order** — each step depends on the output
of the previous one.

| File | Purpose |
|---|---|
| `1_aki_labelling_kdigo.py` | Applies KDIGO creatinine criteria to timestamped lab records to derive AKI onset timestamps and stage. Implements the baseline cascade (7-day → 7–90-day → first-admission → population-normal) and on-arrival AKI exclusion. |
| `2_window_builder.py` | Segments each admission into non-overlapping 6-hour windows and computes per-window summary features (mean/min/max/last/trend/count/alert) for all labs, medications, and procedures. |
| `3_age_sex_match.py` | Constructs the 1:2 age-band and sex-matched case-control cohort. Controls are drawn exclusively from patients who never develop AKI across any admission. |
| `4_split.py` | Patient-level stratified train/validation/test split (70/15/15), ensuring no patient's admissions cross splits. |
| `clinical_codes.py` | Reference dictionaries for nephrotoxin tiers, drug class groupings, GPI codes, and OPCS-4 procedure groupings. Imported by `feature_engineering.py`. |
| `lab_config.py` | Canonical lab synonym mapping (raw `component_basename` → canonical analyte name), basename exclusion list (urine tests, hepatitis screens, non-serum components), and unit-harmonisation logic (creatinine µmol/L conversion, haematocrit decimal/percentage correction). |
| `feature_engineering.py` | Core GOSH feature computation: loads raw labs/medications/procedures, applies `lab_config.py` mappings and unit fixes, and produces the per-window feature matrix consumed by `2_window_builder.py`. |
| `pic_preprocessing.py` | PIC-specific equivalent pipeline: cohort construction from `ADMISSIONS`/`PATIENTS`/`ICUSTAYS`, KDIGO labelling from `LABEVENTS`, lab/medication feature extraction, 6-hour windowing, and alignment to the GOSH `feature_cols.csv` schema for downstream zero-shot and retraining experiments. |

**Output:** `windows_matched.csv`, `split_index_matched.csv` (GOSH) and
the PIC equivalents, ready for `training/dataset.py`.

---

## `training/`

Model definitions, training loop, evaluation, explainability, and all
downstream analyses (traffic light stratification, stage stratification,
silent deployment, bootstrap CIs).

### Core pipeline

| File | Purpose |
|---|---|
| `dataset.py` | `AKISequenceDataset` and `collate_fn`: loads windowed CSVs, applies forward-fill and scaling, pads variable-length admissions into batches via `pack_padded_sequence`-compatible tensors. |
| `models.py` | All model architectures: `AKI_LSTM`, `AKI_LSTM_Attention`, `AKI_GRU`, `AKI_GRU_Attention`, `AKI_Transformer` (causal, with sinusoidal positional encoding), and the `build_model()` factory. |
| `losses.py` | Focal loss and weighted binary cross-entropy implementations for handling window-level class imbalance. |
| `train.py` | Training loop: AdamW optimiser, reduce-on-plateau scheduling on validation AUPRC, gradient clipping, checkpointing of the best model by validation AUPRC. |
| `baseline_models.py` | Non-temporal baselines (XGBoost, Random Forest) with patient-level 5-fold cross-validated hyperparameter search, admission-level threshold selection (max specificity subject to sensitivity ≥ 0.75), and comparison against the temporal models. |

### Evaluation and explainability

| File | Purpose |
|---|---|
| `evaluate.py` | Primary evaluation script. Run with `--split val` to compute and save the decision threshold (max specificity subject to a minimum sensitivity constraint), then `--split test` to apply that fixed threshold and automatically run admission-level bootstrap confidence intervals. |
| `bootstrap_ci.py` | Standalone admission-level bootstrap CI computation at a fixed threshold, for reproducing or re-running CIs without repeating full evaluation. |
| `gradient_explain.py` | Integrated Gradients attribution analysis. Computes per-feature, per-timestep signed attributions on the test set; outputs population-level differential importance (AKI vs non-AKI) and per-admission temporal attribution heatmaps. |
| `plot_feature_importance.py` | Generates the differential and directional IG importance bar charts from `gradient_explain.py` output. |
| `stage_analysis.py` | AKI severity stage (1/2/3) stratified performance: detection rate, lead time distribution, AUROC/AUPRC per stage, with bootstrap CIs. |
| `traffic_light_analysis.py` | Green/Amber/Red risk stratification: threshold optimisation (Youden-J and three alternative methods — AKI-percentile, capacity-based, AKI-coverage), Amber transition analysis, and comparison plots across all threshold methods. |
| `silent_deployment_monthly_evalu...py` | Monthly evaluation of the frozen, best-performing model on prospectively collected silent deployment data, with per-month bootstrap CIs and longitudinal performance plots. |
| `zero_shot_gosh_pic.py` | Applies the GOSH-trained model directly to PIC data without retraining, using the GOSH validation-set threshold unmodified, to test cross-site generalisability. |

**Run order for a full model evaluation:**
```bash
python train.py --model transformer --loss focal --name transformer_focal
python evaluate.py --run transformer_focal --split val
python evaluate.py --run transformer_focal --split test   # auto-runs bootstrap
python gradient_explain.py --run transformer_focal --split test
python stage_analysis.py --predictions data/models/transformer_focal/test_predictions.csv --threshold <val_threshold>
python traffic_light_analysis.py --run transformer_focal
```

---

## `scripts/`

Shell wrappers chaining the Python scripts above for full reproducibility.

| File | Purpose |
|---|---|
| `run.sh` | End-to-end pipeline: preprocessing → training → evaluation for all model configurations. |
| `run_bootstrap.sh` | Runs `evaluate.py` (val then test) for every model, producing bootstrap CIs for all runs in one pass. |
| `evaluate_min_sensitivity.sh` | Runs evaluation across a range of `--min_sensitivity` constraints to compare resulting sensitivity/specificity operating points. |

---

## Reproducing the paper's results

```bash
# 1. Preprocessing (GOSH)
cd preprocessing
python 1_aki_labelling_kdigo.py
python 2_window_builder.py
python 3_age_sex_match.py
python 4_split.py

# 2. Train all GOSH model variants
cd ../training
bash ../scripts/run.sh

# 3. Evaluate with bootstrap CIs
bash ../scripts/run_bootstrap.sh

# 4. PIC external validation
python ../preprocessing/pic_preprocessing.py
python zero_shot_gosh_pic.py --run transformer_focal
# then retrain on PIC using train.py with PIC windows/split_index

# 5. Explainability, stage analysis, traffic light, silent deployment
python gradient_explain.py --run transformer_focal --split test
python stage_analysis.py --predictions <path_to_test_predictions> --threshold <val_threshold>
python traffic_light_analysis.py --run transformer_focal
python silent_deployment_monthly_evalu...py --run transformer_focal
```

See inline docstrings in each script for full argument lists and
expected input/output formats.