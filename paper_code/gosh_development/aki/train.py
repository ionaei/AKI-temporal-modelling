"""Training, Platt recalibration and prediction for sequence and tabular models."""
from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from .models import build_model, masked_loss

log = logging.getLogger(__name__)
META = ["stay_id", "subject_id", "k", "window_end", "hours_since_admit", "hours_to_onset",
        "label", "aki", "stage_peak", "age_days"]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ── data ──────────────────────────────────────────────────────────────────
class Sequences:
    """Holds one (X, y, k) array triple per stay, in grid order."""

    def __init__(self, win: pd.DataFrame, X: np.ndarray):
        win = win.reset_index(drop=True)
        self.meta = win[[c for c in META if c in win.columns]]
        bounds = np.flatnonzero(win["stay_id"].to_numpy()[1:] != win["stay_id"].to_numpy()[:-1]) + 1
        starts = np.r_[0, bounds]
        ends = np.r_[bounds, len(win)]
        y = win["label"].to_numpy(np.float32)
        k = win["k"].to_numpy(np.int64)
        self.items = [(X[s:e], y[s:e], k[s:e]) for s, e in zip(starts, ends)]
        self.row_slices = list(zip(starts, ends))

    def __len__(self):
        return len(self.items)


def batches(seqs: Sequences, max_tokens: int, shuffle: bool, crop: int | None, rng):
    """Length-bucketed batches with a token budget; optional random crops."""
    order = np.argsort([len(it[0]) for it in seqs.items])
    groups, cur, cur_max = [], [], 0
    for i in order:
        L = len(seqs.items[i][0]) if crop is None else min(len(seqs.items[i][0]), crop)
        if cur and max(cur_max, L) * (len(cur) + 1) > max_tokens:
            groups.append(cur)
            cur, cur_max = [], 0
        cur.append(i)
        cur_max = max(cur_max, L)
    if cur:
        groups.append(cur)
    if shuffle:
        rng.shuffle(groups)
    for g in groups:
        yield g


def collate(seqs: Sequences, idx, crop: int | None, context: int, rng):
    xs, ys, ks, ms = [], [], [], []
    for i in idx:
        x, y, k = seqs.items[i]
        m = np.ones(len(x), bool)
        if crop is not None and len(x) > crop:
            e = rng.integers(crop, len(x) + 1)
            s = e - crop
            x, y, k = x[s:e], y[s:e], k[s:e]
            m = np.ones(len(x), bool)
            if s > 0:                       # first positions lack full context
                m[: max(0, context - 1)] = False
        xs.append(x); ys.append(y); ks.append(k); ms.append(m)
    T = max(len(x) for x in xs)
    B, F = len(xs), xs[0].shape[1]
    X = np.zeros((B, T, F), np.float32)
    Y = np.zeros((B, T), np.float32)
    K = np.zeros((B, T), np.int64)
    M = np.zeros((B, T), bool)
    for b, (x, y, k, m) in enumerate(zip(xs, ys, ks, ms)):
        L = len(x)
        X[b, :L], Y[b, :L], K[b, :L], M[b, :L] = x, y, k, m
    lengths = torch.tensor([len(x) for x in xs])
    return (torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(K),
            torch.from_numpy(M), lengths)


@torch.no_grad()
def predict_logits(model, seqs: Sequences, tc: dict, device) -> np.ndarray:
    model.eval()
    out = np.zeros(sum(len(it[0]) for it in seqs.items), np.float32)
    rng = np.random.default_rng(0)
    for idx in batches(seqs, tc["max_tokens_eval"], False, None, rng):
        X, Y, K, M, L = collate(seqs, idx, None, tc["context_windows"], rng)
        lg = model(X.to(device), L, K.to(device)).cpu().numpy()
        for b, i in enumerate(idx):
            s, e = seqs.row_slices[i]
            out[s:e] = lg[b, : e - s]
    return out


def _metrics(y, logit):
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    return roc_auc_score(y, logit), average_precision_score(y, logit)


# ── sequence training ─────────────────────────────────────────────────────
def train_sequence(kind, tr: Sequences, va: Sequences, n_features, cfg, out_dir: Path, seed,
                   resume: bool = False):
    """Train with a checkpoint written after EVERY epoch (checkpoint.pt).

    resume=True and a checkpoint exists -> continue from the next epoch with the
    same model, optimiser, scheduler, early-stopping and random-number state.
    """
    tc = cfg["training"]
    mc = {**cfg["model"], "context_windows": tc["context_windows"],
          "max_windows": cfg["windows"]["max_windows"]}
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(kind, n_features, mc).to(device)
    y_tr = np.concatenate([it[1] for it in tr.items])
    pos_w = tc["pos_weight"]
    if pos_w == "auto":
        pos_w = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    elif pos_w == "sqrt":
        pos_w = float(np.sqrt((y_tr == 0).sum() / max((y_tr == 1).sum(), 1)))
    log.info("%s | params %d | loss %s | pos_weight %.2f | focal alpha %.2f",
             kind, sum(p.numel() for p in model.parameters()), tc["loss"], float(pos_w),
             tc["focal_alpha"])
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=4)
    crop = tc["train_crop_windows"] if kind == "transformer" else None
    rng = np.random.default_rng(seed)
    y_va = np.concatenate([it[1] for it in va.items])
    best, best_ep, wait, hist, start = -1.0, 0, 0, [], 1
    ckpt_path = out_dir / "checkpoint.pt"
    if resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        best, best_ep, wait, hist = ck["best"], ck["best_ep"], ck["wait"], ck["hist"]
        rng.bit_generator.state = ck["np_rng"]
        torch.set_rng_state(ck["torch_rng"])
        start = ck["epoch"] + 1
        log.info("resuming %s seed %d from epoch %d (best val AUPRC %.3f at epoch %d)",
                 kind, seed, start, best, best_ep)
        if ck.get("stopped"):
            start = tc["epochs"] + 1          # early stopping had already triggered
    for ep in range(start, tc["epochs"] + 1):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for idx in batches(tr, tc["max_tokens_train"], True, crop, rng):
            X, Y, K, M, L = collate(tr, idx, crop, tc["context_windows"], rng)
            X, Y, K, M = X.to(device), Y.to(device), K.to(device), M.to(device)
            if not M.any():
                continue
            loss = masked_loss(model(X, L, K), Y, M, tc["loss"], pos_w,
                               tc["focal_alpha"], tc["focal_gamma"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            opt.step()
            tot += loss.item(); nb += 1
        auroc, auprc = _metrics(y_va, predict_logits(model, va, tc, device))
        sched.step(auprc)
        hist.append(dict(epoch=ep, train_loss=tot / max(nb, 1), val_auroc=auroc,
                         val_auprc=auprc, lr=opt.param_groups[0]["lr"],
                         sec=time.time() - t0))
        log.info("ep %3d loss %.4f val AUROC %.3f AUPRC %.3f", ep, tot / max(nb, 1), auroc, auprc)
        if auprc > best:
            best, best_ep, wait = auprc, ep, 0
            torch.save(model.state_dict(), out_dir / "model.pt")
        else:
            wait += 1
        stop = wait >= tc["patience"]
        # write to a temp file then rename, so a crash mid-save cannot corrupt it
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": ep, "best": best,
                    "best_ep": best_ep, "wait": wait, "hist": hist, "stopped": stop,
                    "np_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state()},
                   out_dir / "checkpoint.tmp")
        (out_dir / "checkpoint.tmp").replace(ckpt_path)
        if stop:
            break
    pd.DataFrame(hist).to_csv(out_dir / "history.csv", index=False)
    model.load_state_dict(torch.load(out_dir / "model.pt", map_location=device, weights_only=True))
    json.dump({"kind": kind, "n_features": n_features, "model_cfg": mc, "best_epoch": best_ep,
               "best_val_auprc": best, "seed": seed, "pos_weight": pos_w},
              open(out_dir / "model_meta.json", "w"), indent=1)
    return model, device


def load_sequence_model(run_dir: Path):
    meta = json.load(open(run_dir / "model_meta.json"))
    model = build_model(meta["kind"], meta["n_features"], meta["model_cfg"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device, weights_only=True))
    return model.to(device), device


# ── tabular baselines (window-independent: no temporal state) ─────────────
def train_tabular(kind, X_tr, y_tr, X_va, y_va, seed):
    if kind == "logreg":
        m = LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced")
    elif kind == "hgb":
        m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                           early_stopping=True, validation_fraction=0.1,
                                           class_weight="balanced", random_state=seed)
    else:
        raise ValueError(kind)
    m.fit(X_tr, y_tr)
    log.info("%s val AUROC %.3f AUPRC %.3f", kind, *_metrics(y_va, tabular_logits(m, X_va)))
    return m


def tabular_logits(m, X):
    p = np.clip(m.predict_proba(X)[:, 1], 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


# ── calibration ───────────────────────────────────────────────────────────
class Platt:
    """p = sigmoid(a * logit + b), fitted on validation windows at TRUE prevalence."""

    def fit(self, logit, y):
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit.reshape(-1, 1), y)
        self.a, self.b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
        return self

    def __call__(self, logit):
        return 1 / (1 + np.exp(-(self.a * logit + self.b)))

    def save(self, path):
        json.dump({"a": self.a, "b": self.b}, open(path, "w"))

    @classmethod
    def load(cls, path):
        d = json.load(open(path))
        p = cls(); p.a, p.b = d["a"], d["b"]
        return p