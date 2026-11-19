# Temporal deep learning for early prediction of paediatric acute kidney injury

Code for the paper:

> **Identifying acute kidney injury risk in paediatric intensive care from routine electronic health record data hours before current diagnostic criteria.**
> Biggart I, Reed M, Fogel A, Jodrell T, Zakka K, Patel B, Ng C, Ashrafian H, Geranmayeh F, Fitzgerald F, Marks S, Sebire N, Barnaghi P.

The models estimate, every 6 hours, the risk that a child in intensive care will meet the KDIGO creatinine criteria for acute kidney injury (AKI) within the next 24 hours. They use routinely collected laboratory results, blood gases, medication administrations and demographics. Models were developed at Great Ormond Street Hospital (GOSH), evaluated on a six-month temporal hold-out at GOSH, and validated without retraining on two external cohorts: the Paediatric Intensive Care database (PIC) and the neonatal cohort of MIMIC-III.

## Repository layout

```
AKI-temporal-modelling/
├── gosh_development/        model development and evaluation at GOSH (run inside the GOSH DRE)
│   ├── aki/                 pipeline package: extraction, labelling, windows, models, metrics
│   ├── configs/             study settings (common.yaml) and site mappings (sites/*.yaml)
│   ├── tools/               analyses for the tables and figures, synthetic data, audits
│   ├── tests/               unit tests
│   ├── run_gosh.sh          full GOSH workflow
│   └── run_paper_analyses.sh  tables and figures computed at GOSH
└── external_validation/     zero-shot evaluation on PIC and MIMIC-III neonates
    ├── aki/  configs/  tools/  tests/
    └── run_external.sh      external validation workflow
```

The two folders are self-contained. GOSH data can only be processed inside the GOSH Digital Research Environment (DRE), whereas PIC and MIMIC-III are processed wherever the PhysioNet data are held, so each analysis was run from its own copy of the pipeline. Both copies share the same design (site adapters, KDIGO labelling, windowing, preprocessing, models and metrics); `external_validation/README.md` lists where its `aki` package differs from `gosh_development/aki`. Trained GOSH models (weights, preprocessing, calibration and thresholds, with no patient data) are exported from the DRE and evaluated in `external_validation`.

Start with [`gosh_development/README.md`](gosh_development/README.md) for the method and the pipeline, then [`external_validation/README.md`](external_validation/README.md).

## Quick start (synthetic data, no access needed)

Each folder contains a synthetic data generator and a smoke test that runs the whole pipeline on CPU in a few minutes:

```bash
cd gosh_development            # or external_validation
pip install -r requirements.txt
pytest -q                      # unit tests: KDIGO labelling, units, medication routes, no-leakage windows
bash tools/run_synthetic.sh    # extract -> label -> windows -> split -> train -> evaluate -> explain -> deploy
```

The synthetic data only test that the code runs end to end; results on them have no clinical meaning.

## Overview of the method

| Step | Summary | Code |
|---|---|---|
| Cohort | ICU stays of patients aged 0 to 18 years; contiguous ICU episodes merged | `aki/cohort.py` |
| AKI labels | KDIGO serum-creatinine criteria applied to timestamped results; neonatal-modified definition below 28 days of age; urine output not used | `aki/labelling.py` |
| Exclusions | AKI before or within 24 h of ICU admission; renal replacement therapy before onset or in the 90 days before admission; stays where AKI could not be detected | `aki/labelling.py` |
| Windows | Non-overlapping 6-hour windows; label = onset within the next 24 h; windows after onset removed | `aki/windows.py` |
| Features | Laboratory and blood-gas values, creatinine ratio and rise, medication exposure, measurement process, age, sex, time since admission | `aki/windows.py`, `aki/concepts.py` |
| Preprocessing | Age-band normalisation, standardisation and clipping, fitted on the training split only | `aki/features.py` |
| Splits | Patient-level 70/15/15, stratified by AKI; last six months held out | `aki/split.py` |
| Models | LSTM, GRU and causal Transformer; logistic regression and gradient boosting baselines; weighted BCE or focal loss | `aki/models.py`, `aki/train.py` |
| Calibration and thresholds | Platt recalibration and operating points chosen on validation and applied unchanged | `aki/train.py`, `aki/metrics.py` |
| Evaluation | AUROC, AUPRC, calibration, sensitivity and precision within 24/48 h, lead time, alert burden, patient-level bootstrap | `aki/metrics.py` |
| Explainability | Integrated gradients | `aki/explain.py` |

## Data access

- **GOSH**: the data are not publicly available. Access to anonymised data can be requested from the GOSH Data Research, Innovation and Virtual Environments (DRIVE) unit, subject to research approvals: <https://www.gosh.nhs.uk/our-research/drive-unit-for-digital-innovation/a-secure-environment-fpr-digital-research/>.
- **PIC v1.1.0**: <https://physionet.org/content/picdb/1.1.0/>
- **MIMIC-III v1.4**: <https://physionet.org/content/mimiciii/1.4/>

PIC and MIMIC-III require PhysioNet credentialing, completion of the CITI "Data or Specimens Only Research" course and agreement to the PhysioNet Credentialed Health Data Use Agreement. No data, trained models or patient-level outputs are included in this repository.

## Data protection

Stage outputs under `data/` and `predictions.parquet` files under `runs/` contain pseudonymised identifiers and must stay inside the environment where the data are held. Summary tables written by `evaluate`, `aggregate` and the scripts in `tools/` contain aggregates only; apply the small-number rules of your environment before exporting them. `.gitignore` excludes `data/`, `runs/`, `results/`, `logs/` and `export/`.

## Citation

If you use this code, please cite the paper above (full reference to be added on publication).

## Contact

Iona Biggart (iona.biggart23@imperial.ac.uk) and Payam Barnaghi (p.barnaghi@imperial.ac.uk), Imperial College London.
