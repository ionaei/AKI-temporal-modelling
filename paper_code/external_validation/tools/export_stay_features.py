"""Compact, de-identified stay-level feature file for the pooled cohort comparison.

Run where PIC / MIMIC-III live (NOT in the GOSH DRE):

    python tools/export_stay_features.py --site pic    --name PIC
    python tools/export_stay_features.py --site mimic3 --name MIMIC-III

Reads data/<site>/windows/windows.parquet + feature_groups.json (written by `aki.cli windows`).
Writes export/<site>_stay_features.parquet (+ .csv.gz): ONE ROW PER ICU STAY, columns
    dataset, aki, severe, <features>
and nothing else. To keep it as small as possible:
  * one row per stay (mean of the stay's 6-hour windows), at most --max_stays stays (random sample);
  * only the groups in --groups (default creatinine lab med static process);
  * for lab and med features the derived variants in --drop_suffix are dropped
    (default _min _max _delta _hsince _days72h), leaving one column per analyte / drug;
  * features observed in < --min_observed of stays, or constant, are dropped;
  * no identifiers, no dates, rows shuffled, values rounded to 3 decimals, float32.
Check the PhysioNet data use agreement and your DRE import rules before moving the file: it is
derived from credentialed data, although it contains no identifiers and only a sample of stays.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    ap.add_argument("--name", help="dataset name in the file (default: the site name)")
    ap.add_argument("--groups", nargs="+", default=["creatinine", "lab", "med", "static", "process"])
    ap.add_argument("--drop_suffix", nargs="*", default=["_min", "_max", "_delta", "_hsince", "_days72h"])
    ap.add_argument("--max_stays", type=int, default=2000)
    ap.add_argument("--min_observed", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "export"))
    a = ap.parse_args()
    name = a.name or a.site
    wdir = ROOT / "data" / a.site / "windows"
    fg = json.loads((wdir / "feature_groups.json").read_text())

    def keep(f, g):
        if g not in a.groups:
            return False
        return not (g in ("lab", "med") and f.endswith(tuple(a.drop_suffix)))

    cols = sorted(f for f, g in fg.items() if keep(f, g))
    w = pd.read_parquet(wdir / "windows.parquet", columns=["stay_id", "aki", "stage_peak"] + cols)
    grp = w.groupby("stay_id")
    X = grp[cols].mean()
    meta = grp[["aki", "stage_peak"]].first()
    ok = [c for c in cols if X[c].notna().mean() >= a.min_observed and X[c].nunique() > 1]
    X = X[ok]
    sel = np.random.default_rng(a.seed).permutation(len(X))[:a.max_stays]      # sample + shuffle
    out = pd.concat([
        pd.DataFrame({"dataset": name, "aki": meta["aki"].astype(int).to_numpy()[sel],
                      "severe": (meta["stage_peak"].to_numpy()[sel] >= 2).astype(int)}),
        X.iloc[sel].reset_index(drop=True).round(3).astype("float32")], axis=1)

    d = Path(a.out)
    d.mkdir(parents=True, exist_ok=True)
    p1, p2 = d / f"{a.site}_stay_features.parquet", d / f"{a.site}_stay_features.csv.gz"
    out.to_parquet(p1, index=False)
    out.to_csv(p2, index=False, compression="gzip")
    print(f"{name}: {len(out):,} stays ({int(out.aki.sum()):,} with AKI), {len(ok)} features "
          f"(of {len(cols)} candidates)")
    print(f"written {p1} ({p1.stat().st_size / 1e6:.2f} MB) and {p2} ({p2.stat().st_size / 1e6:.2f} MB)")
    print("columns:", ", ".join(out.columns[:8]), "...")


if __name__ == "__main__":
    main()