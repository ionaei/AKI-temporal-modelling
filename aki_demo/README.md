# AKI Early Warning System — GOSH PICU Demo

A Streamlit demonstration of continuous AKI risk prediction in a paediatric
ICU setting. Uses a pre-trained LSTM model and entirely synthetic patient data.

> ⚠️ **This is a research demo only. All patient data is synthetic.
> Not for clinical use.**

---

## What it shows

- **Ward overview** — all 30 synthetic patients displayed with traffic light
  risk status (🔴 Red / 🟡 Amber / 🟢 Green)
- **Patient detail view** — click any patient to see:
  - Risk trajectory over time, coloured by traffic light state
  - Temporal attention heatmap — which time steps drove the prediction
  - Current lab values with abnormality flags
  - **Add new observations** — enter a lab result and watch the risk update
- **Timeline control** — step forward through the admission to simulate
  real-time monitoring

## Run locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/aki-demo.git
cd aki-demo

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add model weights (copy from your training server)
# Place these in model_weights/:
#   best_model.pt
#   scaler.pkl
#   feature_cols.csv
#   config.json

# 4. Run
streamlit run app.py
```

## Deploy to Streamlit Cloud (free)

1. Push this repo to GitHub (model weights included — they are safe to share,
   see note below)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set main file to `app.py`
5. Deploy — you get a public URL instantly

## Model weights

The `model_weights/` folder should contain:

| File | Description |
|------|-------------|
| `best_model.pt` | Trained LSTM weights (~1.4 MB) |
| `scaler.pkl` | Feature normalisation scaler (~28 KB) |
| `feature_cols.csv` | Ordered list of feature column names |
| `config.json` | Model architecture hyperparameters |

**Are the weights safe to share publicly?**
Yes. LSTM weights trained on tabular EHR data encode statistical patterns,
not individual patient records. They cannot be reverse-engineered to recover
any patient's data. This is consistent with published clinical ML papers
including the DeepMind AKI paper (Tomašev et al., Nature 2019).

## Project structure

```
aki_demo/
├── app.py                  # Streamlit application
├── src/
│   ├── fake_data.py        # Synthetic patient generator
│   └── model.py            # Model loading and inference
├── model_weights/          # Add your weights here (not in repo by default)
│   ├── best_model.pt
│   ├── scaler.pkl
│   ├── feature_cols.csv
│   └── config.json
├── requirements.txt
└── README.md
```

## Research context

This demo accompanies the paper:

> *Temporal prediction of acute kidney injury in critically ill children:
> a multi-modal deep learning framework using routine electronic health records*
>
> Great Ormond Street Hospital · R&D reference 24PE36

## Thresholds

Default traffic light thresholds (adjustable in the sidebar):
- **Green** (low risk): probability < 0.30
- **Amber** (intermediate): 0.30 ≤ probability < 0.60
- **Red** (high risk): probability ≥ 0.60

These were optimised on the validation set to maximise Youden's J index
while constraining the RED alert rate to 10–20% of admissions.