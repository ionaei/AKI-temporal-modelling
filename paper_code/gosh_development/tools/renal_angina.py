"""Renal Angina Index (RAI) at GOSH, and a head-to-head comparison with the AKI model.

    python tools/renal_angina.py --site gosh --run runs/gosh_transformer_focal_all --splits test holdout

RAI (Basu et al., Kidney Int 2014;85:659-667, doi:10.1038/ki.2013.349), computed 12 h
after ICU admission:
  risk     1 = ICU admission
           3 = stem-cell or solid-organ transplant (history)
           5 = invasive ventilation AND >= 1 vasoactive drug
  injury   worse of eCrCl decrease and % fluid overload:
           1 = no decrease, 2 = decrease 0-25 %, 4 = 25-50 %, 8 = > 50 %
           eCrCl decrease = 1 - baseline SCr / current SCr  (height cancels in Schwartz)
  RAI      risk x injury (1-40); RAI >= 8 = renal angina fulfilled
  outcome  severe AKI (SCr >= 2 x baseline, KDIGO stage 2-3) on day 3 (48-72 h)
Fluid-balance and weight data are not available at GOSH, so the injury score uses
creatinine only (eCrCl decrease).

Analysis A ('as published'): all ICU stays with >= 1 creatinine in the first 12 h and
  >= 1 creatinine on day 3 (so the outcome is known); includes AKI present on arrival.
Analysis B ('head to head'): the model's unseen stays (--splits, e.g. test holdout; pooled
  and per split); the model's risk at the window ending
  12 h after admission (mean of seeds) vs the RAI, for severe AKI with onset <= 72 h
  (onset >= 24 h by the cohort definition). Paired patient bootstrap for the AUROC difference.

Settings in the site YAML under `rai:` (see bottom of this file for an example).
Outputs in results/rai/: summary.json and CSV tables (aggregates only; safe to export)
plus rai_scores.parquet (stay-level, contains IDs: keep inside the DRE).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aki.concepts import MED_GROUPS, med_groups_for, route_class  # noqa: E402
from aki.config import load_config  # noqa: E402

H = pd.Timedelta(hours=1)
VASOACTIVE = {"vasopressor", "inotrope"}


def injury_score(dec):
    if pd.isna(dec):
        return np.nan
    return 1 if dec <= 0 else 2 if dec <= 0.25 else 4 if dec <= 0.50 else 8


def code_events(spec, codes):
    """Rows of a coded file (procedures / diagnoses) whose code starts with any prefix."""
    if not spec or not spec.get("path") or not codes:
        return pd.DataFrame(columns=["subject_id", "time"])
    c = spec["columns"]
    cols = [c["subject"], c["code"]] + ([c["time"]] if c.get("time") else [])
    df = pd.read_csv(spec["path"], usecols=cols, dtype=str)
    code = df[c["code"]].astype(str).str.upper().str.replace(".", "", regex=False)
    hit = code.str.startswith(tuple(str(x).upper().replace(".", "") for x in codes))
    out = pd.DataFrame({"subject_id": df.loc[hit, c["subject"]].str.strip()})
    out["time"] = pd.to_datetime(df.loc[hit, c["time"]], errors="coerce") if c.get("time") \
        else pd.NaT
    return out


def compute_rai(stays, creat, meds, vent, transplant, p, older_days, older_days_infant):
    cr = {s: g.sort_values("time") for s, g in creat.groupby("subject_id")}
    vt = {s: g["time"].to_numpy() for s, g in vent.groupby("subject_id")}
    tx = {s: g["time"] for s, g in transplant.groupby("subject_id")}
    md = {s: g for s, g in meds.groupby("subject_id")}
    rows = []
    for s in stays.itertuples(index=False):
        a, t12 = s.admit, s.admit + 12 * H
        g = cr.get(s.subject_id)
        first12 = g[(g.time >= a) & (g.time <= t12)] if g is not None else pd.DataFrame()
        day3 = g[(g.time >= a + 48 * H) & (g.time <= a + 72 * H)] if g is not None else pd.DataFrame()
        # baseline: lowest pre-admission value (7 d, then age-appropriate older window),
        # otherwise the lowest value in the first 12 h
        base, src = np.nan, None
        if g is not None:
            older = older_days_infant if s.age_days < 365 else older_days
            for lo, name in ((7, "pre_7d"), (older, f"pre_{older}d")):
                pre = g[(g.time < a) & (g.time >= a - pd.Timedelta(days=lo))]
                if len(pre):
                    base, src = pre.value.min(), name
                    break
        if np.isnan(base) and len(first12):
            base, src = first12.value.min(), "first_12h"
        cur = first12.value.max() if len(first12) else np.nan
        dec = 1 - base / cur if len(first12) and base > 0 else np.nan
        # risk stratum
        v = vt.get(s.subject_id, np.array([], dtype="datetime64[ns]"))
        ventilated = bool(((v >= np.datetime64(a - p["vent_lookback_h"] * H)) &
                           (v <= np.datetime64(t12))).any()) if len(v) else False
        m = md.get(s.subject_id)
        vaso = bool(len(m) and ((m.start <= t12) & (m.end_eff >= a)).any()) if m is not None else False
        txs = tx.get(s.subject_id)
        transplanted = bool(txs is not None and (txs.isna() | (txs <= t12)).any())
        risk = 5 if (ventilated and vaso) else 3 if transplanted else 1
        inj = injury_score(dec)
        severe_d3 = bool((day3.value >= 2 * base).any()) if len(day3) and base > 0 else np.nan
        rows.append(dict(stay_id=s.stay_id, subject_id=s.subject_id, risk=risk, injury=inj,
                         rai=risk * inj if not np.isnan(inj) else np.nan,
                         ventilated=ventilated, vasoactive=vaso, transplant=transplanted,
                         baseline_source=src, ecrcl_decrease=dec,
                         n_creat_12h=len(first12), n_creat_day3=len(day3),
                         severe_aki_day3=severe_d3))
    return pd.DataFrame(rows)


def stats_at(y, positive):
    y, positive = np.asarray(y, bool), np.asarray(positive, bool)
    tp, fp = (positive & y).sum(), (positive & ~y).sum()
    fn, tn = (~positive & y).sum(), (~positive & ~y).sum()
    return dict(sensitivity=tp / max(tp + fn, 1), specificity=tn / max(tn + fp, 1),
                ppv=tp / max(tp + fp, 1), npv=tn / max(tn + fn, 1),
                n_positive=int(positive.sum()))


def bootstrap_auc_diff(df, a, b, y, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    idx = df.groupby("subject_id").indices
    subj = np.array(list(idx))
    diffs, aa, bb = [], [], []
    for _ in range(n):
        rows = np.concatenate([idx[s] for s in rng.choice(subj, len(subj))])
        d = df.iloc[rows]
        if d[y].nunique() < 2:
            continue
        ra, rb = roc_auc_score(d[y], d[a]), roc_auc_score(d[y], d[b])
        aa.append(ra); bb.append(rb); diffs.append(ra - rb)
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return dict(auroc_ci_first=q(aa), auroc_ci_second=q(bb), difference_ci=q(diffs),
                difference_p_le_0=float(np.mean(np.array(diffs) <= 0)), n_boot=len(diffs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="gosh")
    ap.add_argument("--run", help="model run for analysis B, e.g. runs/gosh_transformer_focal_all")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--splits", nargs="+", default=["test"],
                    help="model-unseen stays to use for analysis B, e.g. test holdout")
    a = ap.parse_args()
    cfg = load_config(a.site)
    site, C = cfg["site"], cfg["common"]
    p = {"vent_lookback_h": 24, **(site.get("rai") or {})}
    d = site["dirs"]
    out = ROOT / "results" / "rai"
    out.mkdir(parents=True, exist_ok=True)

    stays = pd.read_parquet(Path(d["labelled"]) / "stays_labelled.parquet")
    # RAI population: ICU stays >= 12 h, no renal replacement before/around admission
    keep = (stays["los_h"] >= 12) & ~stays["exclude_reason"].isin(
        ["rrt_before_onset", "recent_prior_rrt"])
    stays = stays[keep]
    labs = pd.read_parquet(Path(d["canonical"]) / "labs.parquet")
    creat = labs[labs.concept == "creatinine"][["subject_id", "time", "value"]].astype(
        {"value": float})
    meds = pd.read_parquet(Path(d["canonical"]) / "meds.parquet")
    names = pd.Series(meds["name"].unique())
    groups = dict(zip(names, names.map(med_groups_for)))
    meds = meds.assign(group=meds["name"].map(groups)).explode("group")
    meds = meds[meds["group"].isin(VASOACTIVE)]
    meds = meds[~meds.apply(lambda r: route_class(r["route"]) in
                            MED_GROUPS[r["group"]].exclude_routes, axis=1)]
    meds["end_eff"] = np.maximum(meds["end"], meds["start"]) + H * 1
    vent = code_events(site.get("procedures"), p.get("ventilation_opcs"))
    tx = pd.concat([code_events(site.get("procedures"), p.get("transplant_opcs")),
                    code_events(p.get("diagnoses"), p.get("transplant_icd10"))])
    lab_p = C["labelling"]
    r = compute_rai(stays, creat, meds, vent, tx, p, lab_p["older_baseline_days"],
                    lab_p.get("older_baseline_days_infant", lab_p["older_baseline_days"]))
    r.to_parquet(out / "rai_scores.parquet", index=False)

    summ = {"definition": "creatinine-only RAI at 12 h (no fluid-overload data)",
            "stays_considered": len(r),
            "with_creatinine_first_12h": int((r.n_creat_12h > 0).sum()),
            "risk_strata": r["risk"].value_counts().sort_index().to_dict(),
            "ventilated_12h": int(r.ventilated.sum()), "vasoactive_12h": int(r.vasoactive.sum()),
            "transplant_history": int(r.transplant.sum()),
            "baseline_source": r["baseline_source"].value_counts().to_dict()}

    # ── A: as published ──
    A = r[r.rai.notna() & r.severe_aki_day3.notna()].copy()
    A["y"] = A.severe_aki_day3.astype(bool)
    if A["y"].nunique() == 2:
        summ["A_as_published"] = dict(
            n_stays=len(A), n_severe_day3=int(A.y.sum()),
            auroc=float(roc_auc_score(A.y, A.rai)),
            rai_ge_8=stats_at(A.y, A.rai >= 8))
        (A.groupby("rai").y.agg(stays="size", severe_day3="sum").reset_index()
          .to_csv(out / "A_rai_distribution.csv", index=False))

    # ── B: head to head with the model, on test and/or hold-out stays ──
    if a.run:
        sp = pd.read_csv(Path(d["labelled"]) / "split.csv", dtype=str)
        lab = pd.read_parquet(Path(d["labelled"]) / "stays_labelled.parquet")
        lab = lab[~lab.exclude]
        frames = []
        for split in a.splits:
            # where each split's predictions live: evaluate -> eval_<site>_<split>,
            # deploy -> deploy_<site>_<split>
            files = []
            for folder in (f"eval_{a.site}_{split}", f"deploy_{a.site}_{split}"):
                files = sorted(Path(a.run).glob(f"seed_*/{folder}/predictions.parquet"))
                if files:
                    break
            if not files:
                raise SystemExit(f"no predictions for split '{split}' under {a.run} "
                                 f"(looked for eval_{a.site}_{split} and deploy_{a.site}_{split})")
            preds = []
            for f in files:
                q = pd.read_parquet(f, columns=["stay_id", "k", "prob"])
                preds.append(q[q.k == 1].set_index("stay_id")["prob"])   # window ending 12 h
            model = pd.concat(preds, axis=1).mean(axis=1).rename("model_risk_12h")
            ids = set(sp.loc[sp.split == split, "stay_id"])
            frames.append(lab[lab.stay_id.isin(ids)].merge(model, left_on="stay_id",
                                                           right_index=True)
                          .assign(split=split, n_seed_files=len(files)))
        L = pd.concat(frames, ignore_index=True)
        L = L.merge(r[["stay_id", "rai"]], on="stay_id")
        L = L[L.rai.notna()].reset_index(drop=True)
        outcomes = {
            "severe_aki_onset_le_72h (primary)": L.aki & (L.stage_peak >= 2) & (L.onset_h <= 72),
            "any_aki_onset_le_72h (secondary)": L.aki & (L.onset_h <= 72),
        }
        groups = {"+".join(a.splits): L}
        if len(a.splits) > 1:
            groups.update({s: L[L.split == s] for s in a.splits})
        summ["B_head_to_head"] = {"splits": a.splits,
                                  "seed_files_per_split": L.groupby("split")["n_seed_files"]
                                  .first().to_dict()}
        for oname, y in outcomes.items():
            for gname, G in groups.items():
                G = G.assign(y=y.loc[G.index].astype(bool).to_numpy()).reset_index(drop=True)
                key = f"{oname} | {gname}"
                if G["y"].nunique() < 2:
                    summ["B_head_to_head"][key] = "fewer than 2 outcome classes"
                    continue
                res = dict(n_stays=len(G), n_events=int(G.y.sum()),
                           auroc_model=float(roc_auc_score(G.y, G.model_risk_12h)),
                           auroc_rai=float(roc_auc_score(G.y, G.rai)),
                           rai_ge_8=stats_at(G.y, G.rai >= 8))
                spec = res["rai_ge_8"]["specificity"]
                thr = np.quantile(G.loc[~G.y, "model_risk_12h"], spec)
                res["model_at_same_specificity"] = stats_at(G.y, G.model_risk_12h > thr)
                # paired bootstrap only for the pooled set (the reported comparison)
                if gname == "+".join(a.splits):
                    res["paired_bootstrap"] = bootstrap_auc_diff(G, "model_risk_12h", "rai",
                                                                 "y", a.n_boot)
                summ["B_head_to_head"][key] = res

    (out / "summary.json").write_text(json.dumps(summ, indent=1, default=float))
    print(json.dumps(summ, indent=1, default=float))
    print(f"\nwritten to {out}  (rai_scores.parquet has IDs: keep it in the DRE)")


if __name__ == "__main__":
    main()

# Example site-YAML block (gosh.yaml):
#
# rai:
#   ventilation_opcs: [E851, X562]     # invasive ventilation; endotracheal intubation
#   vent_lookback_h: 24                # coded procedures are often dated, not timed
#   transplant_opcs: []                # optional: transplant procedure prefixes (verify first)
#   transplant_icd10: [Z94, T86]       # transplanted-organ status; complications of transplant
#   diagnoses:                         # file holding ICD-10 codes (used ONLY for RAI risk)
#     path: /path/to/gosh/diagnoses.csv
#     columns: {subject: project_id, code: diag_nat_code, time: start_datetime}