"""Numbers for the cohort table, one column per cohort/split. Aggregates only.

    python tools/table1.py gosh:train,val gosh:test pic:all mimic3:all

Each argument is  <site>:<splits>  (splits = comma-separated train/val/test/holdout, or 'all').
Reads data/<site>/labelled/{stays_labelled.parquet,split.csv} and windows/windows.parquet,
i.e. only stays that survived every exclusion. Prints the rows in table order and writes
results/table1.csv. Cells are formatted for LaTeX (\\% and $\\pm$).
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def pct(k, tot):
    return f"{k:,} ({100 * k / tot:.1f}\\%)" if tot else "---"


def column(spec: str) -> dict:
    site, sp = spec.split(":")
    d = ROOT / "data" / site
    lab = pd.read_parquet(d / "labelled" / "stays_labelled.parquet")
    split = pd.read_csv(d / "labelled" / "split.csv", dtype=str)
    win = pd.read_parquet(d / "windows" / "windows.parquet", columns=["stay_id", "label"])
    s = lab.merge(split[["stay_id", "split"]], on="stay_id")      # eligible stays only
    if sp != "all":
        s = s[s["split"].isin(sp.split(","))]
    w = win[win["stay_id"].isin(s["stay_id"])]
    n = len(s)
    known = int(s["sex"].isin(["M", "F"]).sum())
    age = s["age_days"] / 365.25
    aki = s[s["aki"]]
    return {
        "ICU stays": f"{n:,}",
        "Unique patients": f"{s['subject_id'].nunique():,}",
        "Windows": f"{len(w):,}",
        "Windows labelled AKI-onset": pct(int(w["label"].sum()), len(w)),
        "Male": pct(int((s["sex"] == "M").sum()), known),
        "Female": pct(int((s["sex"] == "F").sum()), known),
        "Age min (years)": f"{age.min():.1f}",
        "Age max (years)": f"{age.max():.1f}",
        "Age mean +- SD (years)": f"{age.mean():.2f} $\\pm$ {age.std():.2f}",
        "Age < 28 days": pct(int((s["age_days"] < 28).sum()), n),
        "AKI onset": pct(len(aki), n),
        "No AKI": pct(n - len(aki), n),
        "Stage 1 (peak)": f"{int((aki['stage_peak'] == 1).sum()):,}",
        "Stage 2 (peak)": f"{int((aki['stage_peak'] == 2).sum()):,}",
        "Stage 3 (peak)": f"{int((aki['stage_peak'] == 3).sum()):,}",
        "Severe AKI (stage 2-3)": f"{int(aki['severe'].sum()):,}",
    }


if __name__ == "__main__":
    specs = sys.argv[1:]
    if not specs:
        raise SystemExit(__doc__)
    t = pd.DataFrame({sp: column(sp) for sp in specs})
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    t.to_csv(out / "table1.csv")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 40)
    print(t.to_string())
    print(f"\nwritten: {out / 'table1.csv'}")