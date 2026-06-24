# AKI Temporal Modelling

Temporal deep learning framework for early prediction of acute kidney
injury (AKI) in paediatric intensive care, developed at Great Ormond
Street Hospital (GOSH) and externally validated on the Paediatric
Intensive Care (PIC) database.

The model predicts KDIGO-defined AKI onset within a 24-hour horizon,
using 6-hourly clinical windows constructed from routinely collected
EHR data (labs, medications, procedures, diagnoses, demographics,
blood gases). The best-performing architecture (causal Transformer)
achieves a median early-warning lead time of 58.5 hours before
KDIGO-defined onset, with independent replication on PIC and a
twelve-month silent deployment evaluation.

## Repository structure

```
AKI-temporal-modelling/
├── aki_demo/        Interactive clinical dashboard (Streamlit prototype)
└── paper_code/      Full research pipeline — data, models, evaluation
```

See [`paper_code/README.md`](paper_code/README.md) for the full
research pipeline (preprocessing, training, evaluation, explainability,
silent deployment) and [`aki_demo/README.md`](aki_demo/README.md)
for the clinical dashboard prototype.

## Quick start

```bash
git clone https://github.com/ionaei/AKI-temporal-modelling.git
cd AKI-temporal-modelling

# Research pipeline
cd paper_code
pip install -r requirements.txt

# Clinical dashboard demo (synthetic patient data only)
cd ../aki_demo
pip install -r requirements.txt
streamlit run app.py
```

## Data availability

This repository contains code only. The GOSH clinical dataset used
for model development is not publicly available due to patient
privacy and institutional governance restrictions; access can be
requested through the GOSH Data Research, Innovation and Virtual
Environments (DRE) unit. The PIC database used for external
validation is publicly available via PhysioNet
(https://physionet.org/content/picdb/1.1.0/) under a standard data
use agreement. See `paper_code/README.md` for details.


## Ethics and governance

This study was approved by the Health Research Authority and
Research Ethics Committee as secondary use of data
(HRA/REC reference 21/LO/0646; GOSH R&D reference 24PE36). All
analyses involving GOSH data were conducted within the GOSH Digital
Research Environment; no patient data is included in this repository.

## Licence

This project is licensed under the MIT Licence — see [LICENSE](LICENSE) for details.

## Contact

iona.biggart23@imperial.ac.uk
p.barnaghi@imperial.ac.uk
