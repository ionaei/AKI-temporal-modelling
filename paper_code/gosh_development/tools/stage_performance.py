"""Performance of one run by AKI severity stage (peak stage), at one operating point.

    python tools/stage_performance.py --run gosh_transformer_focal_all --evals eval_gosh_test deploy_gosh_holdout --n_boot 1000

Reads runs/<run>/seed_*/<evaluation>/predictions.parquet (nothing is re-trained or re-evaluated).
Needs tools/paired_compare.py next to it (loading and alignment helpers).

Strata: Stage 1, Stage 2, Stage 3, Severe (stage 2-3), All AKI. Per stratum:
  n                 AKI stays in the stratum
  detection         share of the stratum's stays with >= 1 alert before onset (stay-level sensitivity)
  sens48            share with >= 1 alert in the 48 h before onset
  lead              median hours before onset of the earliest alert inside that 48 h, among those stays
  auroc             window-level AUROC: windows labelled positive (onset within 24 h) of the stratum's
                    stays against ALL windows of non-AKI stays (the same controls for every stratum)
AUPRC is not given per stage: its baseline (prevalence) changes with the stratum, so it is not
comparable across strata.

Each seed uses its own validation-chosen threshold; values are the mean over seeds, and the 95% CI
is the 2.5-97.5 percentile of the patient-level bootstrap of that mean ('All AKI' reproduces the
stay-level sensitivity, sensitivity within 48 h and lead time of the main results).
Output: results/stage_performance/<run>__<evaluation>__<op>.csv (aggregates only).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paired_compare import align, load_run  # noqa: E402

STRATA = [("Stage 1", lambda s: s == 1), ("Stage 2", lambda s: s == 2),
          ("Stage 3", lambda s: s == 3), ("Severe (stage 2-3)", lambda s: s >= 2),
          ("All AKI", lambda s: s >= 1)]
FIELDS = ["detection", "sens48", "lead", "auroc"]


def strata_metrics(prob, thr, y, aki_w, stage_w, h2o, key, K):
    alert = prob >= thr
    any_alert = np.bincount(key, weights=alert, minlength=K) > 0
    in48 = alert & (h2o <= 48)                               # NaN (non-AKI windows) -> False
    any48 = np.bincount(key, weights=in48, minlength=K) > 0
    earliest = np.full(K, np.nan)
    np.fmax.at(earliest, key[in48], h2o[in48])
    stage_k = np.zeros(K)
    stage_k[key] = stage_w
    aki_k = np.zeros(K, bool)
    aki_k[key] = aki_w
    neg = ~aki_w
    out = {}
    for name, f in STRATA:
        sel = aki_k & f(stage_k)
        n = int(sel.sum())
        if n == 0:
            out[name] = dict(n=0, **{k: np.nan for k in FIELDS})
            continue
        pos = (y == 1) & sel[key]
        auc = np.nan
        if pos.any() and neg.any():
            auc = roc_auc_score(np.r_[np.ones(pos.sum()), np.zeros(neg.sum())],
                                np.r_[prob[pos], prob[neg]])
        hit = sel & any48
        out[name] = dict(n=n, detection=float(any_alert[sel].mean()),
                         sens48=float(any48[sel].mean()),
                         lead=float(np.nanmedian(earliest[hit])) if hit.any() else np.nan,
                         auroc=auc)
    return out


def seed_mean(models_probs, thrs, *args):
    per = [strata_metrics(p, t, *args) for p, t in zip(models_probs, thrs)]
    out = {}
    for name, _ in STRATA:
        out[name] = dict(n=per[0][name]["n"])
        for k in FIELDS:
            v = np.array([m[name][k] for m in per], float)
            out[name][k] = float(np.nanmean(v)) if not np.isnan(v).all() else np.nan
    return out


def run_one(run, tag, op, n_boot, seed):
    models = load_run(run, tag, op)
    if not models:
        print(f"[{tag}] skipped: no predictions under runs/{run}/seed_*/{tag}/")
        return None
    base = models[0]["df"]
    probs = [align(base, m["df"], f"{run}/{m['seed']}") for m in models]
    thrs = [m["thr"] for m in models]
    y = base["label"].to_numpy(float)
    aki_w = base["aki"].to_numpy(bool)
    stage_w = base["stage_peak"].to_numpy(float)
    h2o = base["hours_to_onset"].to_numpy(float)
    stay_codes = pd.factorize(base["stay_id"])[0].astype(np.int64)
    n_stays = int(stay_codes.max()) + 1
    subj = pd.factorize(base["subject_id"])[0]
    order, counts = np.argsort(subj, kind="stable"), np.bincount(subj)
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    rows = [order[s:s + c] for s, c in zip(starts, counts)]

    def on(idx, key):
        return seed_mean([p[idx] for p in probs], thrs, y[idx], aki_w[idx], stage_w[idx],
                         h2o[idx], key, int(key.max()) + 1)

    est = on(np.arange(len(base)), stay_codes)
    rng = np.random.default_rng(seed)
    draws = {n: {k: [] for k in FIELDS} for n, _ in STRATA}
    t0 = time.time()
    for i in range(n_boot):
        d = rng.integers(0, len(rows), len(rows))
        idx = np.concatenate([rows[j] for j in d])
        occ = np.repeat(np.arange(len(d)), counts[d])
        key = pd.factorize(occ.astype(np.int64) * n_stays + stay_codes[idx])[0]
        m = on(idx, key)
        for n, _ in STRATA:
            for k in FIELDS:
                draws[n][k].append(m[n][k])
        if (i + 1) % 100 == 0:
            print(f"   [{tag}] {i + 1}/{n_boot} resamples ({time.time() - t0:.0f}s)", flush=True)
    out = []
    for n, _ in STRATA:
        row = dict(stratum=n, n_stays=est[n]["n"])
        for k in FIELDS:
            v = np.array(draws[n][k], float)
            v = v[~np.isnan(v)]
            row[k] = est[n][k]
            row[k + "_lo"], row[k + "_hi"] = (np.percentile(v, [2.5, 97.5]) if len(v) >= 20
                                              else (np.nan, np.nan))
        out.append(row)
    return pd.DataFrame(out), len(models)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--evals", nargs="+", default=["eval_gosh_test"])
    ap.add_argument("--op", default="orig_spec_at_sens0.75")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results" / "stage_performance"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    for tag in a.evals:
        r = run_one(a.run, tag, a.op, a.n_boot, a.seed)
        if r is None:
            continue
        df, n_seeds = r
        df.to_csv(out / f"{a.run}__{tag}__{a.op}.csv", index=False)
        show = pd.DataFrame({"stratum": df.stratum, "n": df.n_stays})
        for k, nm, dec in [("detection", "detection", 3), ("sens48", "sens within 48h", 3),
                           ("lead", "lead (h)", 1), ("auroc", "AUROC", 3)]:
            show[nm] = [("" if pd.isna(v) else f"{v:.{dec}f} ({lo:.{dec}f}-{hi:.{dec}f})")
                        for v, lo, hi in zip(df[k], df[k + "_lo"], df[k + "_hi"])]
        print(f"\n=== {a.run} | {tag} | {a.op} | {n_seeds} seed(s), {a.n_boot} patient resamples")
        print(show.to_string(index=False))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()