"""
src/models.py

Temporal models for AKI onset prediction.
All models output a risk logit at every time step.

Models
------
AKI_LSTM        : multi-layer LSTM with input projection + LayerNorm
AKI_GRU         : GRU variant, fewer parameters than LSTM
AKI_Transformer : causal transformer (no future leakage)

Usage
-----
    from models import build_model
    model = build_model('lstm', input_size=427, hidden_size=128,
                        n_layers=2, dropout=0.3)
"""

import math
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# ─────────────────────────────────────────────────────────
#  LSTM
# ─────────────────────────────────────────────────────────

class AKI_LSTM(nn.Module):
    def __init__(
        self,
        input_size:    int,
        hidden_size:   int   = 128,
        n_layers:      int   = 2,
        dropout:       float = 0.3,
        bidirectional: bool  = False,
    ):
        super().__init__()
        self.n_dir = 2 if bidirectional else 1

        # Project raw features into hidden space before LSTM
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

        lstm_out = hidden_size * self.n_dir
        self.norm    = nn.LayerNorm(lstm_out)
        self.dropout = nn.Dropout(dropout)

        self.head = nn.Sequential(
            nn.Linear(lstm_out, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor,
                lengths: torch.Tensor) -> torch.Tensor:
        """
        x       : (B, T, input_size)
        lengths : (B,) actual lengths (descending)
        Returns : (B, T) logits
        """
        x      = self.input_proj(x)
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=True
        )
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True)
        out    = self.norm(out)
        out    = self.dropout(out)
        return self.head(out).squeeze(-1)


# ─────────────────────────────────────────────────────────
#  GRU
# ─────────────────────────────────────────────────────────

class AKI_GRU(nn.Module):
    def __init__(
        self,
        input_size:  int,
        hidden_size: int   = 128,
        n_layers:    int   = 2,
        dropout:     float = 0.3,
    ):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.gru = nn.GRU(
            input_size  = hidden_size,
            hidden_size = hidden_size,
            num_layers  = n_layers,
            batch_first = True,
            dropout     = dropout if n_layers > 1 else 0.0,
        )

        self.norm    = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor,
                lengths: torch.Tensor) -> torch.Tensor:
        x      = self.input_proj(x)
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=True
        )
        out, _ = self.gru(packed)
        out, _ = pad_packed_sequence(out, batch_first=True)
        out    = self.norm(out)
        out    = self.dropout(out)
        return self.head(out).squeeze(-1)


# ─────────────────────────────────────────────────────────
#  Causal Transformer
# ─────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


class AKI_Transformer(nn.Module):
    """
    Causal (decoder-style) transformer — step t cannot attend to t+1.
    Suitable for online prediction as it mirrors what the LSTM does.
    """
    def __init__(
        self,
        input_size:  int,
        hidden_size: int   = 128,
        n_heads:     int   = 4,
        n_layers:    int   = 2,
        dropout:     float = 0.3,
        max_len:     int   = 512,
    ):
        super().__init__()
        assert hidden_size % n_heads == 0, \
            f"hidden_size {hidden_size} must be divisible by n_heads {n_heads}"

        self.input_proj = nn.Linear(input_size, hidden_size)
        self.pos_enc    = PositionalEncoding(hidden_size, dropout, max_len)

        layer = nn.TransformerEncoderLayer(
            d_model         = hidden_size,
            nhead           = n_heads,
            dim_feedforward = hidden_size * 2,
            dropout         = dropout,
            batch_first     = True,
            norm_first      = True,   # Pre-LN — more stable training
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm    = nn.LayerNorm(hidden_size)

        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor,
                lengths: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        x = self.pos_enc(self.input_proj(x))

        # Causal mask: position i cannot attend to position j > i
        causal = torch.triu(
            torch.ones(T, T, device=x.device), diagonal=1
        ).bool()

        # Padding mask: True = ignore this position
        pad_mask = torch.zeros(B, T, device=x.device, dtype=torch.bool)
        for i, L in enumerate(lengths):
            if L < T:
                pad_mask[i, L:] = True

        out = self.encoder(x, mask=causal,
                           src_key_padding_mask=pad_mask)
        out = self.norm(out)
        return self.head(out).squeeze(-1)

# ------------------------------
# LSTM with Attention for Explainability
class AKI_LSTM_Attention(nn.Module):
    """
    LSTM with temporal attention.
    The attention weight at each step tells you how much that
    time point contributed to the final risk prediction.
    """
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
        self.norm    = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        # Attention: learned weight per time step
        self.attention = nn.Linear(hidden_size, 1)

        self.head = nn.Sequential(
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

        # Per-step risk score (what evaluate.py uses)
        logits = self.head(out).squeeze(-1)  # (B, T)

        if return_attention:
            # Attention over time — which steps matter most?
            # Mask padding before softmax
            attn_raw = self.attention(out).squeeze(-1)  # (B, T)
            mask = torch.zeros_like(attn_raw, dtype=torch.bool)
            for i, L in enumerate(lengths):
                mask[i, L:] = True
            attn_raw = attn_raw.masked_fill(mask, -1e9)
            attn_weights = torch.softmax(attn_raw, dim=-1)  # (B, T)
            return logits, attn_weights

        return logits

# GRU with attention for explainability

class AKI_GRU_Attention(nn.Module):
    """
    GRU with temporal attention.
    Identical structure to AKI_LSTM_Attention but uses GRU cell —
    fewer parameters, often comparable performance on clinical data.
    """
    def __init__(self, input_size, hidden_size=128,
                 n_layers=2, dropout=0.3):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(
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
        out, _ = self.gru(packed)
        out, _ = pad_packed_sequence(out, batch_first=True)
        out    = self.norm(out)
        out    = self.dropout(out)

        logits = self.head(out).squeeze(-1)  # (B, T)

        if return_attention:
            attn_raw = self.attention(out).squeeze(-1)  # (B, T)
            mask = torch.zeros_like(attn_raw, dtype=torch.bool)
            for i, L in enumerate(lengths):
                mask[i, L:] = True
            attn_raw     = attn_raw.masked_fill(mask, -1e9)
            attn_weights = torch.softmax(attn_raw, dim=-1)
            return logits, attn_weights

        return logits

# ─────────────────────────────────────────────────────────
#  Factory
# ─────────────────────────────────────────────────────────

def build_model(
    architecture: str,
    input_size:   int,
    hidden_size:  int   = 128,
    n_layers:     int   = 2,
    dropout:      float = 0.3,
    **kwargs,
) -> nn.Module:
    arch      = architecture.lower()
    attention = kwargs.get('attention', False)

    if arch == 'lstm':
        cls = AKI_LSTM_Attention if attention else AKI_LSTM
        return cls(
            input_size  = input_size,
            hidden_size = hidden_size,
            n_layers    = n_layers,
            dropout     = dropout,
            **({} if attention else
               {'bidirectional': kwargs.get('bidirectional', False)})
        )
    elif arch == 'gru':
        cls = AKI_GRU_Attention if attention else AKI_GRU
        return cls(
            input_size  = input_size,
            hidden_size = hidden_size,
            n_layers    = n_layers,
            dropout     = dropout,
        )
    elif arch == 'transformer':
        return AKI_Transformer(
            input_size  = input_size,
            hidden_size = hidden_size,
            n_heads     = kwargs.get('n_heads', 4),
            n_layers    = n_layers,
            dropout     = dropout,
        )
    else:
        raise ValueError(
            f"Unknown architecture '{arch}'. "
            "Choose: lstm | gru | transformer"
        )
