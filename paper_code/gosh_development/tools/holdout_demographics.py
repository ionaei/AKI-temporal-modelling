"""Age and sex distribution per month over the temporal hold-out (Supplementary Table S5).

    python tools/holdout_demographics.py --site gosh

Uses the eligible hold-out stays: data/<site>/labelled/stays_labelled.parquet joined to split.csv
(split == holdout). Month = month of ICU admission, as in the monthly performance figure.
Per month: stays, AKI stays, age at admission in years (mean (SD) and median (IQR)), female n (%).
Small-number rule: any count below --min_cell (default 3) is shown as '<min_cell'; a month with fewer than
--min_cell stays is suppressed completely; percentages and age summaries that would reveal a
suppressed count are dropped. Only the suppressed table is written (safe to export):
    results/holdout_demographics.csv   results/holdout_demographics.tex
Add --exact to also print the unsuppressed table to the console (not saved).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def summarise(g, mc):
    n = len(g)
    n_aki = int(g["aki"].sum())
    age = g["age_years"]
    female = int((g["sex"].astype(str).str.upper().str[0] == "F").sum())
    q1, med, q3 = age.quantile([.25, .5, .75])
    exact = dict(stays=n, aki_stays=n_aki, age_mean=age.mean(), age_sd=age.std(), age_median=med,
                 age_q1=q1, age_q3=q3, female_n=female, female_pct=100 * female / n)
    sup = {k: v for k, v in exact.items()}
    if n < mc:                                                   # whole month suppressed
        return exact, {k: (f"<{mc}" if k == "stays" else "") for k in exact}
    if 0 < n_aki < mc:
        sup["aki_stays"] = f"<{mc}"
    if female < mc or (n - female) < mc:                         # percentage would reveal a small count
        sup["female_n"] = f"<{mc}" if female < mc else female
        sup["female_pct"] = ""
    return exact, sup


def tx(v):
    v = str(v)
    return "$<$" + v[1:] if v.startswith("<") else v


def fmt(v, d=1):
    return v if isinstance(v, str) else ("" if pd.isna(v) else f"{v:.{d}f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="gosh")
    ap.add_argument("--split", default="holdout")
    ap.add_argument("--min_cell", type=int, default=3)
    ap.add_argument("--exact", action="store_true", help="also print the unsuppressed table")
    ap.add_argument("--out", default=str(ROOT / "results"))
    a = ap.parse_args()
    d = ROOT / "data" / a.site / "labelled"
    lab = pd.read_parquet(d / "stays_labelled.parquet")
    need = {"stay_id", "admit", "age_days", "sex", "aki"}
    if need - set(lab.columns):
        raise SystemExit(f"stays_labelled.parquet lacks columns {sorted(need - set(lab.columns))}")
    sp = pd.read_csv(d / "split.csv", dtype=str)[["stay_id", "split"]]
    s = lab.merge(sp, on="stay_id")
    s = s[s["split"] == a.split].copy()
    if s.empty:
        raise SystemExit(f"no stays with split == '{a.split}'")
    s["aki"] = s["aki"].astype(bool)
    s["age_years"] = s["age_days"].astype(float) / 365.25
    s["month"] = pd.to_datetime(s["admit"]).dt.to_period("M").astype(str)

    ex_rows, su_rows = [], []
    for m, g in list(s.groupby("month")) + [("Total", s)]:
        e, u = summarise(g, a.min_cell)
        ex_rows.append(dict(month=m, **e)); su_rows.append(dict(month=m, **u))
    ex, su = pd.DataFrame(ex_rows), pd.DataFrame(su_rows)
    if a.exact:
        pd.set_option("display.width", 200)
        print("EXACT (console only, do not export):\n", ex.round(2).to_string(index=False), "\n")

    out = su.copy()
    for c in ["age_mean", "age_sd", "age_median", "age_q1", "age_q3"]:
        out[c] = out[c].map(lambda v: fmt(v, 1))
    out["female_pct"] = out["female_pct"].map(lambda v: fmt(v, 1))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    out.to_csv(Path(a.out) / "holdout_demographics.csv", index=False)

    rows = []
    for r in out.itertuples():
        if r.stays == "<5":
            rows.append(f"        {r.month} & {tx(r.stays)} & & & & \\\\")
            continue
        age_ms = f"{r.age_mean} ({r.age_sd})"
        age_iqr = f"{r.age_median} ({r.age_q1}--{r.age_q3})"
        fem = f"{r.female_n} ({r.female_pct}\\%)" if r.female_pct != "" else tx(r.female_n)
        pre = "\\midrule " if r.month == "Total" else ""
        rows.append(f"        {pre}{r.month} & {r.stays} & {tx(r.aki_stays)} & {age_ms} & {age_iqr} & {fem} \\\\")
    tex = ("\\begin{tabular}{lccccc}\n        \\toprule\n"
           "        \\textbf{Month} & \\textbf{Stays} & \\textbf{AKI stays} & \\textbf{Age, years: mean (SD)} & "
           "\\textbf{median (IQR)} & \\textbf{Female, n (\\%)} \\\\\n        \\midrule\n"
           + "\n".join(rows) + "\n        \\bottomrule\n\\end{tabular}\n")
    (Path(a.out) / "holdout_demographics.tex").write_text(tex)
    print(out.to_string(index=False))
    print(f"\nwritten to {a.out}: holdout_demographics.csv, holdout_demographics.tex "
          f"(counts below {a.min_cell} suppressed)")


if __name__ == "__main__":
    main()