"""
src/model.py

Load trained LSTM model and run inference for the Streamlit demo app.
Handles all model loading, feature alignment, scaling, and prediction.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import joblib


# ─────────────────────────────────────────────────────────
#  Model architecture — must match training code exactly
# ─────────────────────────────────────────────────────────

class AKI_LSTM_Attention(nn.Module):
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
            attn_raw = self.attention(out).squeeze(-1)
            mask     = torch.zeros_like(attn_raw, dtype=torch.bool)
            for i, L in enumerate(lengths):
                if L < attn_raw.shape[1]:
                    mask[i, L:] = True
            attn_raw     = attn_raw.masked_fill(mask, -1e9)
            attn_weights = torch.softmax(attn_raw, dim=-1)
            return logits, attn_weights

        return logits


class AKI_LSTM(nn.Module):
    """Standard LSTM without attention — fallback if attention not in config."""
    def __init__(self, input_size, hidden_size=128,
                 n_layers=2, dropout=0.3, bidirectional=False):
        super().__init__()
        self.n_dir      = 2 if bidirectional else 1
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size    = hidden_size,
            hidden_size   = hidden_size,
            num_layers    = n_layers,
            batch_first   = True,
            dropout       = dropout if n_layers > 1 else 0.0,
            bidirectional = bidirectional,
        )
        lstm_out   = hidden_size * self.n_dir
        self.norm  = nn.LayerNorm(lstm_out)
        self.drop  = nn.Dropout(dropout)
        self.head  = nn.Sequential(
            nn.Linear(lstm_out, 64),
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
        out    = self.drop(out)
        return self.head(out).squeeze(-1)


# ─────────────────────────────────────────────────────────
#  Model loader
# ─────────────────────────────────────────────────────────

class AKIPredictor:
    """
    Wraps the trained LSTM model for inference in the Streamlit app.

    Usage:
        predictor = AKIPredictor('model_weights/')
        prob, attn = predictor.predict(patient_windows_df)
    """

    def __init__(self, weights_dir: str = 'model_weights'):
        self.weights_dir = Path(weights_dir)
        self.device      = torch.device('cpu')   # demo runs on CPU
        self.model       = None
        self.scaler      = None
        self.feat_cols   = None
        self.config      = None
        self._load()

    def _load(self):
        """Load model, scaler, and feature columns."""
        wdir = self.weights_dir

        # Config
        cfg_path = wdir / 'config.json'
        if cfg_path.exists():
            with open(cfg_path) as f:
                self.config = json.load(f)
        else:
            # Defaults matching training
            self.config = {
                'model': 'lstm', 'hidden': 128, 'layers': 2,
                'dropout': 0.3, 'attention': True,
            }

        # Feature columns
        feat_path = wdir / 'feature_cols.csv'
        if feat_path.exists():
            self.feat_cols = pd.read_csv(feat_path,
                                          header=None)[0].tolist()
        else:
            raise FileNotFoundError(
                f"feature_cols.csv not found in {wdir}. "
                "Copy from your training run."
            )

        # Scaler
        scaler_path = wdir / 'scaler.pkl'
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
        else:
            raise FileNotFoundError(
                f"scaler.pkl not found in {wdir}."
            )

        # Build model
        use_attention = self.config.get('attention', False)
        input_size    = len(self.feat_cols)
        hidden        = self.config.get('hidden', 128)
        n_layers      = self.config.get('layers', 2)
        dropout       = self.config.get('dropout', 0.3)

        if use_attention:
            self.model = AKI_LSTM_Attention(
                input_size, hidden, n_layers, dropout
            )
        else:
            self.model = AKI_LSTM(
                input_size, hidden, n_layers, dropout
            )

        # Load weights
        ckpt_path = wdir / 'best_model.pt'
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"best_model.pt not found in {wdir}."
            )
        ckpt = torch.load(ckpt_path, map_location=self.device)
        if 'model_state' in ckpt:
            state = ckpt['model_state']
        else:
            state = ckpt
        self.model.load_state_dict(state)
        self.model.eval()
        print(f"Model loaded: {input_size} features, "
              f"hidden={hidden}, attention={use_attention}")

    def _align_features(self, windows_df: pd.DataFrame) -> np.ndarray:
        """
        Align synthetic data columns to the model's expected feature set.
        Missing columns are filled with 0, extra columns are dropped.
        """
        X = pd.DataFrame(index=windows_df.index)
        for col in self.feat_cols:
            if col in windows_df.columns:
                X[col] = windows_df[col].fillna(0)
            else:
                X[col] = 0.0   # missing feature → zero

        # Scale
        X_scaled = self.scaler.transform(X.values.astype(np.float32))
        return X_scaled

    def predict(
        self,
        windows_df: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Run inference on a sequence of windows for one patient.

        Parameters
        ----------
        windows_df : DataFrame sorted by step_index, one row per 6h window

        Returns
        -------
        probs      : (T,) array of AKI probabilities, one per step
        attn       : (T,) attention weights or None
        """
        windows_df = windows_df.sort_values('step_index').reset_index(drop=True)
        X          = self._align_features(windows_df)
        T          = len(X)

        x_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0)  # (1, T, F)
        lengths  = torch.tensor([T])

        with torch.no_grad():
            use_attn = self.config.get('attention', False)
            if use_attn:
                try:
                    logits, attn_w = self.model(
                        x_tensor, lengths, return_attention=True
                    )
                    probs = torch.sigmoid(logits[0, :T]).numpy()
                    attn  = attn_w[0, :T].numpy()
                except Exception:
                    logits = self.model(x_tensor, lengths)
                    probs  = torch.sigmoid(logits[0, :T]).numpy()
                    attn   = None
            else:
                logits = self.model(x_tensor, lengths)
                probs  = torch.sigmoid(logits[0, :T]).numpy()
                attn   = None

        return probs, attn

    def predict_step(
        self,
        windows_df: pd.DataFrame,
        up_to_step: int,
    ) -> tuple[float, np.ndarray | None]:
        """
        Predict risk for steps 0..up_to_step (for real-time simulation).
        Returns (current_prob, attn_weights_so_far).
        """
        sub        = windows_df[
            windows_df['step_index'] <= up_to_step
        ].copy()
        probs, attn = self.predict(sub)
        return float(probs[-1]), attn


def traffic_light(prob: float,
                   amber_t: float = 0.30,
                   red_t:   float = 0.60) -> str:
    """Classify a probability into GREEN / AMBER / RED."""
    if prob >= red_t:
        return 'RED'
    elif prob >= amber_t:
        return 'AMBER'
    return 'GREEN'


TL_COLOURS = {
    'GREEN': '#27AE60',
    'AMBER': '#F39C12',
    'RED':   '#C0392B',
}

TL_EMOJI = {
    'GREEN': '🟢',
    'AMBER': '🟡',
    'RED':   '🔴',
}