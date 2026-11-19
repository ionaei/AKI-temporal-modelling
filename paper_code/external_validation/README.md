# External validation on PIC and MIMIC-III neonates

This folder evaluates models trained at GOSH, with their weights frozen, on two external cohorts:

- **PIC** (Paediatric Intensive Care database v1.1.0): Children's Hospital of Zhejiang University School of Medicine, Hangzhou, China; admissions 2010-2018.
- **MIMIC-III neonates** (MIMIC-III v1.4): neonatal intensive care unit of the Beth Israel Deaconess Medical Center, Boston, USA.

Both databases use the MIMIC-III table structure and are read by the same adapter (`aki/adapters/mimic_like.py`). The cohort definition, KDIGO labelling, windows and features are those described in [`../gosh_development/README.md`](../gosh_development/README.md); this README covers what is specific to the external cohorts.

## Requirements

1. PhysioNet credentialed access to PIC (<https://physionet.org/content/picdb/1.1.0/>) and MIMIC-III (<https://physionet.org/content/mimiciii/1.4/>).
2. The trained GOSH runs, exported from the GOSH DRE, in `runs/`:

   | Run | Model |
   |---|---|
   | `gosh_transformer_focal_all` | selected model: Transformer, focal loss, all features |
   | `gosh_logreg_all` | logistic regression, all features |
   | `gosh_transformer_focal_no_process` | Transformer without measurement-process features |
   | `gosh_logreg_no_process` | logistic regression without measurement-process features |

   Each `seed_<n>/` folder needs `run_config.json`, `preprocessor.joblib`, `platt.json`, `thresholds.json` and either `model.pt` + `model_meta.json` or `model.joblib`. These files contain model parameters and training-set summary statistics only, no patient data.
3. Python packages: `pip install -r requirements.txt`. Use the scikit-learn version that trained the GOSH runs (1.7.2 in the paper), since preprocessing and baseline models are stored with joblib.

## Running

Set `raw_dir` in `configs/sites/pic.yaml` and `configs/sites/mimic3.yaml` to the folders holding the PhysioNet `.csv.gz` files, then:

```bash
bash run_external.sh
```

This prepares both cohorts (`extract`, `label`, `windows`, `split`) and evaluates each GOSH run in three ways:

| Evaluation folder | Calibration and threshold | Stays |
|---|---|---|
| `eval_<site>_all` | from GOSH, unchanged | all eligible stays |
| `eval_<site>_test` | from GOSH, unchanged | external test partition |
| `eval_<site>_test_recal` | re-derived on the external validation partition (same operating-point rule), model weights still frozen | external test partition |

The external cohorts are split by patient into 70/15/15 partitions as at GOSH; the training partition is not used. Features that a GOSH model expects but that are absent at the external site are set to the GOSH training mean.

Before the first run, the code lists can be audited with:

```bash
python tools/inspect_raw.py --site pic                       # headers, care units, units, routes (aggregates only)
python tools/discover_codes.py --site pic --skip-chartevents # candidate ITEMIDs with counts, units and medians
```

## Outputs and paper

| Paper | Source |
|---|---|
| Table 4 | `evaluate` (`summary.csv`, `bootstrap_*.csv`) for `gosh_transformer_focal_all`; `tools/seed_summary.py` |
| Supplementary Table S6 (logistic regression, no-process models) | the same, for the other three runs |
| Comparison of the Transformer with logistic regression | `tools/paired_compare.py` |
| Shift of measurement-process and laboratory features relative to GOSH | `tools/shift_diagnostics.py` |
| Table 1 (PIC and MIMIC-III columns) | `tools/table1.py`, `tools/ethnicity_table.py` |
| Supplementary Table S2 (excluded stays) | `tools/excluded_profile.py` |
| Supplementary Figure S3 (clustering) | `tools/cluster_dataset.py` |

`tools/export_stay_features.py` writes one de-identified, averaged feature vector per sampled stay for the pooled comparison of cohorts. Check the PhysioNet data use agreement before moving that file.

## Site configuration

### PIC (`configs/sites/pic.yaml`)

- ICU stays from all care units except the NICU (PICU, SICU, CICU, general ICU).
- Laboratory ITEMIDs chosen from `D_LABITEMS`; temperature-corrected blood-gas items, standard bicarbonate and base excess, and non-serum samples are not used. Blood gases are converted from mmHg to kPa.
- Vital signs are not used, as at GOSH.
- Prescriptions: `DRUG_FORM` is used as the route; where the form does not identify the route, it is read from the drug name (`route_value_map`, `route_name_rules`). Rows without an end time are single administrations.
- RRT: only peritoneal dialysis is identifiable (prescription names); continuous RRT cannot be detected in PIC.

### MIMIC-III neonates (`configs/sites/mimic3.yaml`)

- NICU stays only (`include_careunits: [NICU]`), age below 1 year at admission. Labels use the neonatal-modified KDIGO definition for samples taken before 28 days of age.
- Date of birth has no time of day; when the first NICU admission is on the birth date, the admission time is used as the time of birth. Hours of life are then underestimated rather than overestimated, so samples from the first 72 hours of life cannot trigger onset.
- Prescriptions have dates only; they become available at the end of the day (`date_only_policy: end_of_day`) so no future information enters a window. Vasoactive drugs and some nephrotoxins are added from `INPUTEVENTS_CV`.
- RRT: dialysate and replacement-fluid prescriptions (`ROUTE` DIALYS / REPLACE) and matching drug names.

## Differences from `gosh_development/aki`

The `aki` package in this folder is the version used for the external cohorts. It has the same cohort, labelling, windowing, preprocessing, model and metric code as `gosh_development/aki`, and in addition:

- `adapters/mimic_like.py`: route mapping from form or route columns and drug names, text results where `VALUENUM` is blank, date-of-birth handling for neonates, single-administration prescriptions and RRT from prescriptions.
- `concepts.py`: wider plausibility ranges for creatinine (5-2,500 µmol/L), urea (0.2-120 mmol/L), sodium (90-190 mmol/L), phosphate (0.1-15 mmol/L), ALT (1-30,000 U/L), CRP (0-1,000 mg/L) and white cell count (0-1,500 × 10⁹/L); additional drug-name spellings found in PIC (norvancomycin, ioversol, iopamidol, diatrizoate, PIC spellings of iohexol and phenylephrine); additional non-systemic routes excluded for some drug groups; and unit strings written with '×' or '℃'. In the GOSH extract no route is recorded, so the route exclusions have no effect there.
- `adapters/gosh.py`: optional route inference from GOSH drug names (`route_from_name`), not enabled in `configs/sites/gosh.yaml`.

`gosh_development/aki` additionally writes the AKI-minus-non-AKI importance difference in `explain` and accepts `--op` in `deploy`. `configs/sites/gosh.yaml` here is the GOSH configuration, included for reference and for the synthetic tests.
