"""Integrated gradients for the sequence models.

Explains the logit at a chosen window t with respect to every input in
windows 0..t (baseline = all zeros = training mean after preprocessing),
then sums over time. Returns SIGNED attributions (positive pushes towards
AKI-within-24h, negative pushes away) and the mean absolute attribution per
class, whose difference (AKI minus non-AKI) is the feature-importance
difference shown in the paper.

Also returns the attribution mass by lag (how far back the evidence sits)
and by feature group, e.g. how much of the prediction is carried by the
measurement-process (`process`) group.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch


def integrated_gradients(model, x: np.ndarray, k: np.ndarray, t: int, device,
                         steps: int = 32) -> np.ndarray:
    """x: (T, F) one stay; returns (t+1, F) attributions for the logit at t."""
    model.eval()
    xs = torch.from_numpy(np.array(x[: t + 1])).to(device)
    pos = torch.from_numpy(np.array(k[: t + 1])).to(device).unsqueeze(0).repeat(steps, 1)
    alphas = torch.linspace(1.0 / steps, 1.0, steps, device=device).view(-1, 1, 1)
    inp = (alphas * xs.unsqueeze(0)).requires_grad_(True)
    lengths = torch.full((steps,), t + 1, dtype=torch.long)
    with torch.backends.cudnn.flags(enabled=False):
        out = model(inp, lengths, pos)[:, t].sum()
        grad, = torch.autograd.grad(out, inp)
    return (xs * grad.mean(0)).detach().cpu().numpy()


def explain_sample(model, seqs, columns, groups, device, n_pos=200, n_neg=200, seed=0,
                   steps=32, max_lag=40):
    """Explain the last window before onset (positives) and random non-AKI windows."""
    rng = np.random.default_rng(seed)
    pos_targets, neg_targets = [], []
    for i, (x, y, k) in enumerate(seqs.items):
        if y.max() == 1:
            pos_targets.append((i, len(y) - 1))
        else:
            neg_targets.append((i, int(rng.integers(0, len(y)))))
    rng.shuffle(pos_targets); rng.shuffle(neg_targets)
    feat_rows, lag_rows = [], []
    for cls, targets in (("aki_window", pos_targets[:n_pos]), ("non_aki", neg_targets[:n_neg])):
        for i, t in targets:
            x, y, k = seqs.items[i]
            att = integrated_gradients(model, x, k, t, device, steps)
            feat_rows.append(pd.Series(att.sum(0), index=columns).rename(cls))
            lag = att.sum(1)[::-1][:max_lag]                 # lag 0 = window t
            lag_rows.append(dict(cls=cls, **{f"lag{j}": float(v) for j, v in enumerate(lag)}))
    F = pd.DataFrame(feat_rows)
    F["cls"] = F.index
    by_feat = F.groupby("cls").agg(["mean"]).T.droplevel(1)
    by_feat["abs_mean_aki"] = pd.DataFrame(feat_rows).loc["aki_window"].abs().mean() \
        if "aki_window" in F.index else np.nan
    by_feat["group"] = by_feat.index.map(groups)
    share = (pd.DataFrame(feat_rows).abs().T.groupby(groups).sum().T
             .div(pd.DataFrame(feat_rows).abs().sum(1), axis=0))
    share["cls"] = share.index
    by_group = share.groupby("cls").mean().T
    return by_feat.sort_values("aki_window", ascending=False), by_group, pd.DataFrame(lag_rows)
