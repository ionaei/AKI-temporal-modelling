# AKI Temporal Modelling

By Iona Biggart et al.

Imperial College London and Great Ormond Street Hospital for Children

Temporal deep learning framework for early prediction of acute kidney
injury (AKI) in paediatric intensive care, developed at Great Ormond
Street Hospital (GOSH) and externally validated on the Paediatric
Intensive Care (PIC) and MIMIC-III datasets.

The model predicts KDIGO-defined AKI onset within a 24-hour horizon,
using 6-hourly clinical windows constructed from routinely collected
EHR data (labs, medications, demographics,
blood gases).

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
