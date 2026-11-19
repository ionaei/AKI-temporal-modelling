"""Paired patient-level bootstrap comparison of two trained runs on the SAME stays.

    python tools/paired_compare.py --a gosh_transformer_focal_all --b gosh_logreg_all \
        --evals eval_gosh_test deploy_gosh_holdout --n_boot 1000

    # external sites (run where those evaluations exist), fewer resamples:
    python tools/paired_compare.py --a gosh_transformer_focal_all --b gosh_logreg_all \
        --evals eval_mimic3_all eval_pic_all eval_mimic3_test_recal --n_boot 200

For each evaluation folder (eval_<site>_<split>[_recal] or deploy_<site>_<split>) it reads
runs/<run>/seed_*/<folder>/predictions.parquet, i.e. NOTHING is re-trained or re-evaluated.
Each model uses its own validation-chosen threshold for the chosen operating point
(thresholds.json in the evaluation folder if it exists, else in the seed folder).

Method
  * patients are resampled with replacement; the SAME patient draw is applied to both models,
    so the difference removes the variation that comes from which patients were drawn;
  * for a model with several seeds, each metric is the mean over seeds within the draw
    (matching how results are reported); logistic regression typically has one;
  * difference = A - B; 95% CI = 2.5th-97.5th percentile of the bootstrap differences;
    p = two-sided bootstrap p-value 2*min(P(d<=0), P(d>=0)).

Metrics: window-level AUROC, AUPRC, Brier; stay-level sensitivity / specificity / Youden J;
sensitivity and precision within 24 and 48 h; alerts per 100 patient-days; lead time.
For Brier and alerts per 100 patient-days LOWER is better.

Outputs: results/paired_compare/<A>_vs_<B>__<evaluation>__<op>.csv  (aggregates only).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aki import metrics as M  # noqa: E402

METRICS = ["auroc", "auprc", "brier", "legacy_sens_ever", "legacy_spec_stay",
           "legacy_youden_stay", "sens@24h", "sens@48h", "precision@24h", "precision@48h",
           "alerts_per_100_patient_days", "lead@48h_median"]
LOWER_IS_BETTER = {"brier", "alerts_per_100_patient_days"}
NICE = {"legacy_sens_ever": "stay sensitivity", "legacy_spec_stay": "stay specificity",
        "legacy_youden_stay": "stay Youden J"}
COLS = ["stay_id", "subject_id", "k", "label", "aki", "stage_peak", "hours_to_onset", "prob"]


def load_run(run: str, tag: str, op: str) -> list[dict]:
    models = []
    for sd in sorted((ROOT / "runs" / run).glob("seed_*")):
        f = sd / tag / "predictions.parquet"
        if not f.exists():
            continue
        tf = sd / tag / "thresholds.json"
        if not tf.exists():
            tf = sd / "thresholds.json"
        thr = json.loads(tf.read_text())
        if op not in thr:
            raise SystemExit(f"operating point '{op}' not in {tf} (has {list(thr)})")
        step = json.loads((sd / "run_config.json").read_text())["common"]["windows"]["step_h"]
        models.append(dict(seed=sd.name, df=pd.read_parquet(f, columns=COLS), thr=thr[op],
                           step=step))
    return models


def align(base: pd.DataFrame, df: pd.DataFrame, what: str) -> np.ndarray:
    """Probabilities of `df` in the row order of `base`; refuse if the windows differ."""
    if len(df) == len(base) and (df["stay_id"].to_numpy() == base["stay_id"].to_numpy()).all() \
            and (df["k"].to_numpy() == base["k"].to_numpy()).all():
        return df["prob"].to_numpy(float)
    r = df.set_index(["stay_id", "k"])["prob"].reindex(pd.MultiIndex.from_frame(base[["stay_id", "k"]]))
    if r.isna().any():
        raise SystemExit(f"{what}: windows differ from the reference model "
                         f"({int(r.isna().sum())} missing) - the two runs were not evaluated "
                         f"on the same stays")
    return r.to_numpy(float)


def metrics_one(prob, y, d, thr, step):
    d["prob"] = prob
    a = M.alert_metrics(M.AlertArrays(d, step), thr)
    out = {"auroc": roc_auc_score(y, prob), "auprc": average_precision_score(y, prob),
           "brier": float(np.mean((prob - y) ** 2))}
    for k in METRICS[3:]:
        out[k] = a.get(k, np.nan)
    return out


def mean_over_seeds(probs, thrs, step, y, d):
    ms = [metrics_one(p, y, d, t, step) for p, t in zip(probs, thrs)]
    return {k: float(np.nanmean([m[k] for m in ms])) if not all(np.isnan(m[k]) for m in ms)
            else np.nan for k in METRICS}


def compare(a_name, b_name, tag, op, n_boot, seed):
    A, B = load_run(a_name, tag, op), load_run(b_name, tag, op)
    if not A or not B:
        print(f"[{tag}] skipped: predictions missing for "
              f"{a_name if not A else b_name} (looked for seed_*/{tag}/predictions.parquet)")
        return None
    base = A[0]["df"]
    probs_a = [align(base, m["df"], f"{a_name}/{m['seed']}") for m in A]
    probs_b = [align(base, m["df"], f"{b_name}/{m['seed']}") for m in B]
    thr_a, thr_b = [m["thr"] for m in A], [m["thr"] for m in B]
    step = A[0]["step"]
    y = base["label"].to_numpy(float)
    aki = base["aki"].to_numpy(bool)
    sp = base["stage_peak"].to_numpy()
    h2o = base["hours_to_onset"].to_numpy(float)
    stay_codes = pd.factorize(base["stay_id"])[0].astype(np.int64)
    n_stays = int(stay_codes.max()) + 1
    subj_codes = pd.factorize(base["subject_id"])[0]
    order = np.argsort(subj_codes, kind="stable")
    counts = np.bincount(subj_codes)
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    rows = [order[s:s + c] for s, c in zip(starts, counts)]

    def frame(idx, key):
        return pd.DataFrame({"stay_id": key, "hours_to_onset": h2o[idx], "aki": aki[idx],
                             "stage_peak": sp[idx], "prob": 0.0})

    def run_on(idx, key):
        yy, d = y[idx], frame(idx, key)
        if yy.min() == yy.max():
            return None
        a = mean_over_seeds([p[idx] for p in probs_a], thr_a, step, yy, d)
        b = mean_over_seeds([p[idx] for p in probs_b], thr_b, step, yy, d)
        return a, b

    allidx = np.arange(len(base))
    est_a, est_b = run_on(allidx, stay_codes)
    rng = np.random.default_rng(seed)
    diffs = {k: [] for k in METRICS}
    t0 = time.time()
    for i in range(n_boot):
        draw = rng.integers(0, len(rows), len(rows))
        idx = np.concatenate([rows[j] for j in draw])
        occ = np.repeat(np.arange(len(draw)), counts[draw])
        key = pd.factorize(occ.astype(np.int64) * n_stays + stay_codes[idx])[0]
        r = run_on(idx, key)
        if r is None:
            continue
        for k in METRICS:
            diffs[k].append(r[0][k] - r[1][k])
        if (i + 1) % 100 == 0:
            print(f"   [{tag}] {i + 1}/{n_boot} resamples ({time.time() - t0:.0f}s)", flush=True)

    rows_out = []
    for k in METRICS:
        v = np.array(diffs[k], float)
        v = v[~np.isnan(v)]
        d0 = est_a[k] - est_b[k]
        if len(v) < 20:
            lo = hi = p = np.nan
        else:
            lo, hi = np.percentile(v, [2.5, 97.5])
            p = min(1.0, 2 * min(np.mean(v <= 0), np.mean(v >= 0)))
        excl = (not np.isnan(lo)) and (lo > 0 or hi < 0)
        better = ""
        if excl:
            a_wins = (d0 < 0) if k in LOWER_IS_BETTER else (d0 > 0)
            better = "A" if a_wins else "B"
        rows_out.append(dict(metric=NICE.get(k, k), A=est_a[k], B=est_b[k], diff_A_minus_B=d0,
                             ci_lo=lo, ci_hi=hi, p=p, ci_excludes_0=excl,
                             better=better if k != "lead@48h_median" else
                             (("A longer" if d0 > 0 else "B longer") if excl else ""),
                             n_valid_draws=len(v)))
    res = pd.DataFrame(rows_out)
    info = dict(n_windows=len(base), n_stays=n_stays, n_patients=len(rows),
                n_aki_stays=int(pd.Series(aki).groupby(stay_codes).max().sum()),
                seeds_A=len(A), seeds_B=len(B))
    return res, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="run name A (e.g. the selected model)")
    ap.add_argument("--b", required=True, help="run name B (e.g. logistic regression)")
    ap.add_argument("--evals", nargs="+", default=["eval_gosh_test"],
                    help="evaluation folders, e.g. eval_gosh_test deploy_gosh_holdout eval_pic_all")
    ap.add_argument("--op", default="orig_spec_at_sens0.75")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results" / "paired_compare"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    for tag in a.evals:
        r = compare(a.a, a.b, tag, a.op, a.n_boot, a.seed)
        if r is None:
            continue
        res, info = r
        res.to_csv(out / f"{a.a}_vs_{a.b}__{tag}__{a.op}.csv", index=False)
        print(f"\n=== {tag}  |  A = {a.a}, B = {a.b}  |  {a.op}")
        print(f"    {info['n_patients']} patients, {info['n_stays']} stays "
              f"({info['n_aki_stays']} with AKI), {info['n_windows']} windows; "
              f"seeds A/B: {info['seeds_A']}/{info['seeds_B']}; {a.n_boot} patient resamples")
        show = res.copy()
        for c in ["A", "B", "diff_A_minus_B", "ci_lo", "ci_hi"]:
            show[c] = show[c].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
        show["p"] = show["p"].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
        print(show.drop(columns=["n_valid_draws", "ci_excludes_0"]).to_string(index=False))
    print(f"\nwritten to {out}   (difference = A - B; lower is better for brier and alerts)")


if __name__ == "__main__":
    main()