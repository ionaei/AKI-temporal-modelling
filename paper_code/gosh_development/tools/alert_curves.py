"""Aggregates for the alert-performance figure (run in the DRE; output is safe to export).

    python tools/alert_curves.py --runs gosh_transformer_focal_all gosh_logreg_all \
        --evals eval_gosh_test deploy_gosh_holdout

Reads runs/<run>/seed_*/<evaluation>/{predictions.parquet,threshold_sweep.csv,thresholds.json}.
Writes results/alert_curves/alert_curves.csv, one tidy table with three kinds of rows:
  lookback   share of AKI stays with >= 1 alert in the final H hours before onset, at the primary
             operating point (all stays and severe AKI)
  sweep      the threshold sweep: alerts per 100 patient-days, sensitivity and precision within
             24 / 48 h at every threshold
  op         the same quantities at each named operating point
No patient identifiers are written.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aki import metrics as M  # noqa: E402

HGRID = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 96, 120]
SWEEP = {"alerts_per_100_patient_days": "alerts", "sens@24h": "sens24", "sens@48h": "sens48",
         "precision@24h": "prec24", "precision@48h": "prec48"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--evals", nargs="+", default=["eval_gosh_test"])
    ap.add_argument("--primary", default="orig_spec_at_sens0.75", help="operating point for panel a")
    ap.add_argument("--out", default=str(ROOT / "results" / "alert_curves"))
    a = ap.parse_args()
    rows = []
    for run in a.runs:
        for tag in a.evals:
            found = 0
            for sd in sorted((ROOT / "runs" / run).glob("seed_*")):
                ev = sd / tag
                if not (ev / "predictions.parquet").exists():
                    continue
                found += 1
                tf = ev / "thresholds.json" if (ev / "thresholds.json").exists() else sd / "thresholds.json"
                thr = json.loads(tf.read_text())
                step = json.loads((sd / "run_config.json").read_text())["common"]["windows"]["step_h"]
                df = pd.read_parquet(ev / "predictions.parquet",
                                     columns=["stay_id", "aki", "stage_peak", "hours_to_onset", "prob"])
                base = dict(run=run, eval=tag, seed=sd.name)
                key = pd.factorize(df["stay_id"])[0]
                K = int(key.max()) + 1
                aki_k = np.zeros(K, bool)
                aki_k[key] = df["aki"].to_numpy(bool)
                sev_k = np.zeros(K, bool)
                sev_k[key] = df["stage_peak"].to_numpy() >= 2
                alert = df["prob"].to_numpy() >= thr[a.primary]
                h2o = df["hours_to_onset"].to_numpy(float)
                for H in HGRID:
                    hit = np.bincount(key, weights=alert & (h2o <= H), minlength=K) > 0
                    rows.append(dict(base, kind="lookback", H=H, sens=hit[aki_k].mean(),
                                     sens_severe=hit[aki_k & sev_k].mean() if (aki_k & sev_k).any()
                                     else np.nan))
                sf = ev / "threshold_sweep.csv"
                if sf.exists():
                    sw = pd.read_csv(sf)[["threshold"] + list(SWEEP)].rename(columns=SWEEP)
                    rows += [dict(base, kind="sweep", **r) for r in sw.to_dict("records")]
                arr = M.AlertArrays(df, step)
                for op, t in thr.items():
                    m = M.alert_metrics(arr, t)
                    rows.append(dict(base, kind="op", op=op, threshold=t,
                                     alerts=m["alerts_per_100_patient_days"], sens24=m["sens@24h"],
                                     sens48=m["sens@48h"], prec24=m["precision@24h"],
                                     prec48=m["precision@48h"], stay_sens=m["legacy_sens_ever"],
                                     stay_spec=m["legacy_spec_stay"]))
            print(f"{run} / {tag}: {found} seed(s)")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "alert_curves.csv", index=False)
    print(f"written {out / 'alert_curves.csv'}  ({len(rows)} rows) - export this file")


if __name__ == "__main__":
    main()