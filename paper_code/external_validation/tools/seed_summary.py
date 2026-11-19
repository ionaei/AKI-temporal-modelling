"""One number per metric across seeds: mean, SD across seeds, and seed-averaged 95% CI.

    python tools/seed_summary.py --runs gosh_transformer_focal_all gosh_logreg_all gosh_transformer_focal_no_process gosh_logreg_no_process --op orig_spec_at_sens0.75

For every run x evaluation folder (eval_<site>_<split>[_recal], deploy_...):
  mean      mean of the metric over seeds
  sd        SD over seeds (training randomness; 0 for deterministic models)
  ci_lo/hi  mean over seeds of each seed's patient-bootstrap 95% CI (sampling uncertainty)
            -> only where evaluate was run with --n_boot > 0
Writes results/seed_summary_<op>.csv (long) and results/seed_summary_<op>_table.csv
(formatted 'mean (95% CI lo-hi); SD x' per metric, one row per run x evaluation).
Aggregates only - safe to export.s
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRICS = ["auroc", "auprc", "prevalence", "brier", "cal_slope", "cal_intercept",
                   "threshold", "legacy_sens_ever", "legacy_spec_stay", "legacy_youden_stay",
                   "sens@24h", "sens@48h", "sens_severe@48h", "precision@24h",
                   "precision@48h", "lead@48h_median", "alerts_per_100_patient_days"]
DECIMALS = {"lead@48h_median": 1, "alerts_per_100_patient_days": 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run names under runs/")
    ap.add_argument("--op", default="orig_spec_at_sens0.75", help="operating point column")
    ap.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    ap.add_argument("--out", default=str(ROOT / "results"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for run in a.runs:
        folders = sorted((ROOT / "runs" / run).glob("seed_*/*/summary.csv"))
        if not folders:
            print(f"warning: no summary.csv under runs/{run}")
        for f in folders:
            ev, seed = f.parent.name, f.parent.parent.name
            s = pd.read_csv(f, index_col=0)
            if a.op not in s.columns:
                print(f"warning: {a.op} not in {f} (has {list(s.columns)})")
                continue
            s = pd.to_numeric(s[a.op], errors="coerce")
            b = f.parent / f"bootstrap_{a.op}.csv"
            ci = pd.read_csv(b, index_col=0) if b.exists() else None
            for m in a.metrics:
                rows.append(dict(run=run, evaluation=ev, seed=seed, metric=m,
                                 value=s.get(m),
                                 lo=ci.loc[m, "lo"] if ci is not None and m in ci.index else None,
                                 hi=ci.loc[m, "hi"] if ci is not None and m in ci.index else None))
    if not rows:
        raise SystemExit("nothing found")
    df = pd.DataFrame(rows)
    for c in ("value", "lo", "hi"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    g = (df.groupby(["run", "evaluation", "metric"], sort=False)
           .agg(n_seeds=("seed", "nunique"), mean=("value", "mean"), sd=("value", "std"),
                ci_lo=("lo", "mean"), ci_hi=("hi", "mean"))
           .reset_index())
    g["sd"] = g["sd"].fillna(0.0)
    g.to_csv(out / f"seed_summary_{a.op}.csv", index=False)

    def fmt(r):
        d = DECIMALS.get(r["metric"], 3)
        if pd.isna(r["mean"]):
            return ""
        txt = f"{r['mean']:.{d}f}"
        if pd.notna(r["ci_lo"]) and pd.notna(r["ci_hi"]):
            txt += f" ({r['ci_lo']:.{d}f}-{r['ci_hi']:.{d}f})"
        if r["n_seeds"] > 1:
            txt += f"; SD {r['sd']:.{d}f}"
        return txt

    g["formatted"] = g.apply(fmt, axis=1)
    t = g.pivot_table(index=["run", "evaluation"], columns="metric", values="formatted",
                      aggfunc="first", sort=False)
    t = t[[m for m in a.metrics if m in t.columns]]
    t.insert(0, "n_seeds", g.groupby(["run", "evaluation"], sort=False)["n_seeds"].first())
    t.to_csv(out / f"seed_summary_{a.op}_table.csv")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 40)
    show = [c for c in ["n_seeds", "auroc", "auprc", "brier", "cal_slope", "legacy_sens_ever",
                        "legacy_spec_stay", "precision@48h", "alerts_per_100_patient_days"]
            if c in t.columns]
    print(t[show].to_string())
    print(f"\nwritten: {out / f'seed_summary_{a.op}.csv'}\n         "
          f"{out / f'seed_summary_{a.op}_table.csv'}")


if __name__ == "__main__":
    main()