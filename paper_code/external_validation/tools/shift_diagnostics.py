"""Why does a model trained at one site fail at another? Feature-level diagnostics.

    python tools/shift_diagnostics.py --run runs/gosh_logreg_all/seed_0 --site pic
    python tools/shift_diagnostics.py --run runs/gosh_logreg_all/seed_0 --site gosh   # reference

For every model feature, using the TRAINING site's preprocessing:
  mean_z        mean value at this site in training-site SD units (0 = same as training;
                |mean_z| > 2 = strongly shifted). At the training site's own test set
                this is close to 0 by construction.
  univ_auroc    AUROC of that single feature for the 24h label AT THIS SITE
                (> 0.5 = higher values mean more AKI)
  coef          logistic-regression coefficient (logreg runs only)
  contrib_shift coef x mean_z: how far that feature alone moves every prediction's
                log-odds at this site (large = it swamps the others)
  sign_agrees   does the model's direction match the direction AKI actually takes here?

Also prints label prevalence and the univariate AUROC of the creatinine features,
which should carry signal at any site if labels and features are built correctly.
Outputs are aggregates only -> <run>/shift_<site>_<split>.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aki.cli import _subset, load_site_windows  # noqa: E402
from aki.features import Preprocessor  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="a seed directory, e.g. runs/x/seed_0")
    ap.add_argument("--site", required=True)
    ap.add_argument("--split", default="all")
    ap.add_argument("--max_rows", type=int, default=400_000)
    a = ap.parse_args()
    sd = Path(a.run)
    rc = json.loads((sd / "run_config.json").read_text())
    _, win, groups = load_site_windows(a.site)
    df = _subset(win, a.split)
    if len(df) > a.max_rows:
        df = df.sample(a.max_rows, random_state=0)
    pre = Preprocessor.load(sd / "preprocessor.joblib")
    X = pd.DataFrame(pre.transform(df), columns=pre.columns, index=df.index)
    y = df["label"].to_numpy()
    raw_missing = {c: float(df[c].isna().mean()) if c in df else 1.0 for c in pre.columns}

    rows = []
    for c in pre.columns:
        x = X[c].to_numpy()
        try:
            au = roc_auc_score(y, x) if np.std(x) > 0 and len(np.unique(y)) == 2 else np.nan
        except ValueError:
            au = np.nan
        rows.append(dict(feature=c, group=groups.get(c, "?"), mean_z=float(x.mean()),
                         sd_z=float(x.std()), missing_here=raw_missing[c], univ_auroc=au))
    t = pd.DataFrame(rows)
    if (sd / "model.joblib").exists() and rc["model"] == "logreg":
        m = joblib.load(sd / "model.joblib")
        t["coef"] = m.coef_.ravel()
        t["contrib_shift"] = t["coef"] * t["mean_z"]
        t["sign_agrees"] = np.sign(t["coef"]) == np.sign(t["univ_auroc"] - 0.5)
        t = t.reindex(t["contrib_shift"].abs().sort_values(ascending=False).index)
    else:
        t = t.reindex(t["mean_z"].abs().sort_values(ascending=False).index)
    out = sd / f"shift_{a.site}_{a.split}.csv"
    t.round(4).to_csv(out, index=False)

    pd.set_option("display.width", 200)
    print(f"\n{a.site}/{a.split}: {len(df):,} windows, {df['stay_id'].nunique():,} stays, "
          f"label prevalence {y.mean():.4f}, AKI stays {df.loc[df['aki'], 'stay_id'].nunique()}")
    cr = t[t["group"] == "creatinine"].sort_values("univ_auroc", ascending=False)
    print("\ncreatinine features (should be > 0.5 wherever labels are right):")
    print(cr[["feature", "univ_auroc", "mean_z", "missing_here"]].to_string(index=False))
    print("\nmean |shift| by group (training-site SD units):")
    print(t.groupby("group")["mean_z"].apply(lambda s: s.abs().mean()).round(2).to_string())
    print("\nfeatures with |mean_z| > 2:", int((t["mean_z"].abs() > 2).sum()), "of", len(t))
    cols = ["feature", "group", "mean_z", "univ_auroc", "missing_here"] + \
        (["coef", "contrib_shift", "sign_agrees"] if "coef" in t else [])
    print("\ntop 25 features by impact at this site:")
    print(t[cols].head(25).round(3).to_string(index=False))
    if "contrib_shift" in t:
        g = t.groupby("group")["contrib_shift"].sum().round(2)
        print("\nsum of coef x mean_z by group (log-odds shift applied to EVERY window):")
        print(g.to_string())
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()