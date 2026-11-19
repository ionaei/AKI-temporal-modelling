"""Sequence models and losses.

All models output one logit per 6h window and are strictly causal: the
prediction at window k uses windows <= k only.

GRU / LSTM: input projection (linear, layer norm, ReLU, dropout), stacked
recurrent layers and a two-layer classification head applied at every window.

Transformer: causal attention over the last `context` windows only (banded
mask), with sinusoidal encoding of the ABSOLUTE window index since
admission. Training crops and full-length inference therefore see exactly
the same input/position semantics, and no truncation depends on when the
stay ends.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def _proj(d_in, d, p):
    return nn.Sequential(nn.Linear(d_in, d), nn.LayerNorm(d), nn.ReLU(), nn.Dropout(p))


def _head(d, p):
    return nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Dropout(p), nn.Linear(64, 1))


class RNNModel(nn.Module):
    def __init__(self, n_features, hidden=128, layers=2, dropout=0.3, cell="gru"):
        super().__init__()
        self.proj = _proj(n_features, hidden, dropout)
        rnn = nn.GRU if cell == "gru" else nn.LSTM
        self.rnn = rnn(hidden, hidden, num_layers=layers, batch_first=True,
                       dropout=dropout if layers > 1 else 0.0)
        self.norm = nn.LayerNorm(hidden)
        self.head = _head(hidden, dropout)

    def forward(self, x, lengths, pos=None):
        h = self.proj(x)
        packed = pack_padded_sequence(h, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.rnn(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.shape[1])
        return self.head(self.norm(out)).squeeze(-1)


class TransformerModel(nn.Module):
    def __init__(self, n_features, hidden=128, layers=2, dropout=0.3, heads=4,
                 context=120, max_pos=2048):
        super().__init__()
        self.proj = nn.Linear(n_features, hidden)
        self.context = context
        self.heads = heads
        pe = torch.zeros(max_pos, hidden)
        p = torch.arange(max_pos).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, hidden, 2).float() * (-math.log(10000.0) / hidden))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(p * div), torch.cos(p * div)
        self.register_buffer("pe", pe)
        layer = nn.TransformerEncoderLayer(hidden, heads, hidden * 2, dropout,
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)
        self.head = _head(hidden, dropout)

    def _mask(self, T, lengths, device):
        i = torch.arange(T, device=device)
        rel = i.view(-1, 1) - i.view(1, -1)                     # query - key
        band = (rel >= 0) & (rel < self.context)                 # causal, limited lookback
        valid_k = i.view(1, 1, -1) < lengths.to(device).view(-1, 1, 1)
        allowed = band.unsqueeze(0) & (valid_k | torch.eye(T, dtype=torch.bool,
                                                           device=device).unsqueeze(0))
        # (B*heads, T, T); True = blocked. Diagonal always allowed -> no all-masked rows
        return (~allowed).repeat_interleave(self.heads, dim=0)

    def forward(self, x, lengths, pos):
        B, T, _ = x.shape
        h = self.drop(self.proj(x) + self.pe[pos.clamp(max=self.pe.shape[0] - 1)])
        out = self.enc(h, mask=self._mask(T, lengths, x.device))
        return self.head(self.norm(out)).squeeze(-1)


def build_model(kind: str, n_features: int, mc: dict) -> nn.Module:
    if kind in ("gru", "lstm"):
        return RNNModel(n_features, mc["hidden"], mc["layers"], mc["dropout"], cell=kind)
    if kind == "transformer":
        return TransformerModel(n_features, mc["hidden"], mc["layers"], mc["dropout"],
                                mc["heads"], mc["context_windows"], mc["max_windows"] + 1)
    raise ValueError(kind)


# ── losses (computed on valid positions only — never multiply NaNs by 0) ──
def masked_loss(logits, y, mask, kind: str, pos_weight: float = 1.0,
                alpha: float = 0.75, gamma: float = 2.0):
    lg, t = logits[mask], y[mask]
    if kind == "bce":
        pw = torch.tensor(pos_weight, device=lg.device, dtype=lg.dtype)
        return F.binary_cross_entropy_with_logits(lg, t, pos_weight=pw)
    if kind == "focal":
        # alpha is the weight of the POSITIVE (rare) class; 0.5 = no reweighting.
        bce = F.binary_cross_entropy_with_logits(lg, t, reduction="none")
        p = torch.sigmoid(lg)
        pt = torch.where(t == 1, p, 1 - p)
        a = torch.where(t == 1, torch.full_like(p, alpha), torch.full_like(p, 1 - alpha))
        return (a * (1 - pt) ** gamma * bce).mean()
    raise ValueError(kind)
