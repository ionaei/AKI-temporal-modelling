"""
generate_placeholders.py

Generates placeholder best_model.pt and scaler.pkl matched exactly to
the real feature_cols.csv (633 features, 128 hidden, 2 layers, attention).

Run from aki_demo/ directory:
    python generate_placeholders.py

Then start the app:
    streamlit run app.py

When you have the real weights, replace:
    model_weights/best_model.pt   ← from training server
    model_weights/scaler.pkl      ← from training server
(feature_cols.csv and config.json stay as-is)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.preprocessing import StandardScaler
import joblib

sys.path.insert(0, str(Path(__file__).parent / 'src'))
from inference.fake_data import generate_demo_cohort

WEIGHTS_DIR = Path(__file__).parent / 'model_weights'
WEIGHTS_DIR.mkdir(exist_ok=True)

# ── Load real feature columns ─────────────────────────────
feat_path  = WEIGHTS_DIR / 'feature_cols.csv'
feat_cols  = pd.read_csv(feat_path, header=None)[0].tolist()
INPUT_SIZE = len(feat_cols)
print(f"Feature columns loaded: {INPUT_SIZE}")


# ── Generate synthetic data ───────────────────────────────
print("Generating synthetic cohort...")
windows, meta = generate_demo_cohort(n_patients=30, seed=42)

# Align to exact feature_cols order — fill missing with 0
X_df = pd.DataFrame(index=windows.index)
for col in feat_cols:
    X_df[col] = windows[col].fillna(0) if col in windows.columns else 0.0
X = X_df.values.astype(np.float32)
print(f"Data matrix: {X.shape}")


# ── Fit and save scaler ───────────────────────────────────
print("Fitting scaler...")
scaler = StandardScaler()
scaler.fit(X)
joblib.dump(scaler, WEIGHTS_DIR / 'scaler.pkl')
print(f"  Saved scaler.pkl  ({(WEIGHTS_DIR/'scaler.pkl').stat().st_size/1024:.1f} KB)")


# ── Model architecture ────────────────────────────────────

class AKI_LSTM_Attention(nn.Module):
    """Exact architecture — must match src/model.py."""
    def __init__(self, input_size, hidden_size=128,
                 n_layers=2, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size  = hidden_size,
            hidden_size = hidden_size,
            num_layers  = n_layers,
            batch_first = True,
            dropout     = dropout if n_layers > 1 else 0.0,
        )
        self.norm      = nn.LayerNorm(hidden_size)
        self.dropout   = nn.Dropout(dropout)
        self.attention = nn.Linear(hidden_size, 1)
        self.head      = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x, lengths, return_attention=False):
        x      = self.input_proj(x)
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=True
        )
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True)
        out    = self.norm(out)
        out    = self.dropout(out)
        logits = self.head(out).squeeze(-1)
        if return_attention:
            raw  = self.attention(out).squeeze(-1)
            mask = torch.zeros_like(raw, dtype=torch.bool)
            for i, L in enumerate(lengths):
                if L < raw.shape[1]:
                    mask[i, L:] = True
            raw  = raw.masked_fill(mask, -1e9)
            attn = torch.softmax(raw, dim=-1)
            return logits, attn
        return logits


# ── Build model ───────────────────────────────────────────
HIDDEN   = 128
N_LAYERS = 2
DROPOUT  = 0.3

print(f"\nBuilding model (input={INPUT_SIZE}, hidden={HIDDEN}, layers={N_LAYERS})...")
torch.manual_seed(42)
model = AKI_LSTM_Attention(INPUT_SIZE, HIDDEN, N_LAYERS, DROPOUT)

n_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {n_params:,}")


# ── Brief training on synthetic data ──────────────────────
# 15 epochs — enough for the demo to show differentiation
# between AKI and non-AKI trajectories
print("Training placeholder model (15 epochs)...")
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Positive class weight to handle imbalance
labels = windows['label'].values
pos_w  = (labels == 0).sum() / max((labels == 1).sum(), 1)
criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([pos_w], dtype=torch.float32)
)

X_scaled = scaler.transform(X)

for epoch in range(15):
    epoch_loss = 0.0
    for pid in meta['patient_id']:
        mask = windows['patient_id'] == pid
        w    = windows[mask].sort_values('step_index')
        if len(w) < 2:
            continue

        # Build input tensor aligned to feat_cols
        x_np = X_scaled[mask][w.index - w.index[0]
                               if w.index[0] > 0 else w.index]

        # Safer: reindex from X_scaled using iloc positions
        pos     = windows.index.get_indexer(w.index)
        x_np    = X_scaled[pos]
        y_np    = w['label'].values.astype(np.float32)

        x_t  = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0)
        y_t  = torch.tensor(y_np, dtype=torch.float32).unsqueeze(0)
        lens = torch.tensor([len(w)])

        optimizer.zero_grad()
        logits = model(x_t, lens)
        loss   = criterion(logits, y_t)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item()

    if (epoch + 1) % 5 == 0:
        print(f"  Epoch {epoch+1:2d}/15 — loss: "
              f"{epoch_loss/len(meta):.4f}")


# ── Save checkpoint in same format as train.py ────────────
model.eval()
ckpt = {
    'epoch':       15,
    'model_state': model.state_dict(),
    'val_auprc':   0.0,
    'val_auroc':   0.0,
    'args': {
        'model':     'lstm',
        'hidden':    HIDDEN,
        'layers':    N_LAYERS,
        'dropout':   DROPOUT,
        'attention': True,
    },
    'feat_cols':    feat_cols,
    '_placeholder': True,
}
ckpt_path = WEIGHTS_DIR / 'best_model.pt'
torch.save(ckpt, ckpt_path)
print(f"\n  Saved best_model.pt  "
      f"({ckpt_path.stat().st_size/1024:.0f} KB)")


# ── Verification ──────────────────────────────────────────
print("\nVerifying...")
ckpt_v  = torch.load(ckpt_path, map_location='cpu')
model_v = AKI_LSTM_Attention(INPUT_SIZE, HIDDEN, N_LAYERS, DROPOUT)
model_v.load_state_dict(ckpt_v['model_state'])
model_v.eval()

x_test = torch.zeros(1, 5, INPUT_SIZE)
l_test = torch.tensor([5])
with torch.no_grad():
    logits, attn = model_v(x_test, l_test, return_attention=True)

scaler_v = joblib.load(WEIGHTS_DIR / 'scaler.pkl')

assert logits.shape  == (1, 5),         f"Bad logits shape: {logits.shape}"
assert attn.shape    == (1, 5),         f"Bad attn shape: {attn.shape}"
assert scaler_v.n_features_in_ == INPUT_SIZE, \
    f"Scaler/model feature mismatch: {scaler_v.n_features_in_} vs {INPUT_SIZE}"

print(f"  feature_cols.csv : {INPUT_SIZE} features  ✓")
print(f"  scaler.pkl       : {scaler_v.n_features_in_} features  ✓")
print(f"  best_model.pt    : logits {tuple(logits.shape)}, "
      f"attn {tuple(attn.shape)}  ✓")

print(f"""
╔══════════════════════════════════════════════════╗
║  Placeholder files ready                         ║
║                                                  ║
║  model_weights/                                  ║
║    ✓ feature_cols.csv  {INPUT_SIZE} features            ║
║    ✓ config.json       architecture config       ║
║    ✓ scaler.pkl        fitted on synthetic data  ║
║    ✓ best_model.pt     placeholder LSTM weights  ║
║                                                  ║
║  Run the app:                                    ║
║    streamlit run app.py                          ║
║                                                  ║
║  When real weights are available:                ║
║    Replace best_model.pt and scaler.pkl only     ║
║    feature_cols.csv and config.json stay as-is   ║
╚══════════════════════════════════════════════════╝
""")