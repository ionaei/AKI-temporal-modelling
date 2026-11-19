"""Label validation with gradient boosting (Supplementary Figure S2).

    python tools/label_validation.py --site gosh
    python tools/label_validation.py --site gosh --groups lab med static --tag no_creatinine

Question: are the algorithmic KDIGO labels predictable from clinically coherent features measured
BEFORE onset? (A check that the labels are coherent, not proof that they are right.)

Design (one row per ICU stay, eligible stays only, temporal hold-out excluded)
  AKI stays      features of the last window before onset
  control stays  features of one window drawn at the same position (window index) as the AKI stays'
                 onset windows, so time in ICU cannot separate the classes
  features       window features of --groups (default creatinine lab med static process; NOT 'time');
                 missing values are left missing (XGBoost handles them natively)
  split          your patient-level split: train fits, val early-stops (AUPRC) and sets the threshold
                 (Youden J), test is the held-out evaluation
  CV             5-fold patient-level cross-validation over train+val+test (AUPRC / AUROC per fold)
  importance     XGBoost weight, gain and cover
Outputs in results/label_validation/<tag>/ (aggregates only; export these):
  metrics.json  roc.csv  pr.csv  cv.csv  calibration.csv  importance.csv
Needs xgboost >= 1.6. Run once with all groups and once without creatinine (--groups lab med static)
to show the labels are not just a creatinine artefact.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]


def make_model(seed, n_estimators=600, early=None):
    return XGBClassifier(n_estimators=n_estimators, learning_rate=0.05, max_depth=4, subsample=0.8,
                         colsample_bytree=0.8, min_child_weight=3, tree_method="hist",
                         eval_metric="aucpr", early_stopping_rounds=early, random_state=seed,
                         n_jobs=4)


def thin(x, n=200):
    idx = np.unique(np.linspace(0, len(x) - 1, min(n, len(x))).astype(int))
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    ap.add_argument("--groups", nargs="+", default=["creatinine", "lab", "med", "static", "process"])
    ap.add_argument("--tag", default="all_features")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--n_boot", type=int, default=200)
    a = ap.parse_args()
    out = ROOT / "results" / "label_validation" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    wdir = ROOT / "data" / a.site / "windows"

    cols = sorted(f for f, g in json.loads((wdir / "feature_groups.json").read_text()).items()
                  if g in a.groups)
    w = pd.read_parquet(wdir / "windows.parquet", columns=["stay_id", "subject_id", "k", "aki"] + cols)
    split = pd.read_csv(ROOT / "data" / a.site / "labelled" / "split.csv", dtype=str)[["stay_id", "split"]]
    w = w.merge(split, on="stay_id")
    w = w[w["split"].isin(["train", "val", "test"])].copy()
    w["subject_id"] = w["subject_id"].astype(str)
    w["aki"] = w["aki"].astype(bool)

    # index rows: last window before onset (AKI) / a position-matched window (controls)
    last = w[w["aki"]].sort_values("k").groupby("stay_id").tail(1)
    aki_k = last["k"].to_numpy()
    ctrl = w[~w["aki"]]
    rng = np.random.default_rng(a.seed)
    sel = []
    for s, m in ctrl.groupby("stay_id")["k"].max().items():
        eligible = aki_k[aki_k <= m]
        sel.append((s, int(rng.choice(eligible)) if len(eligible) else int(m)))
    cidx = ctrl.merge(pd.DataFrame(sel, columns=["stay_id", "k"]), on=["stay_id", "k"])
    D = pd.concat([last.assign(y=1), cidx.assign(y=0)], ignore_index=True)
    keep = [c for c in cols if D[c].notna().any() and D[c].nunique() > 1]
    X, y, grp = D[keep].astype(float), D["y"].to_numpy(), D["subject_id"].to_numpy()
    sp = D["split"].to_numpy()
    n = {s: (int((sp == s).sum()), int(y[sp == s].sum())) for s in ("train", "val", "test")}
    print(f"{a.site}: {len(D):,} stays ({int(y.sum()):,} AKI); {len(keep)} features "
          f"(groups {a.groups}); stays/AKI by split: {n}")

    tr, va, te = sp == "train", sp == "val", sp == "test"
    model = make_model(a.seed, early=30)
    model.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
    pva, pte = model.predict_proba(X[va])[:, 1], model.predict_proba(X[te])[:, 1]
    fpr, tpr, thr = roc_curve(y[va], pva)
    cut = float(thr[np.argmax(tpr - fpr)])                     # Youden J on validation
    yt = y[te]
    sens = float(((pte >= cut) & (yt == 1)).sum() / max((yt == 1).sum(), 1))
    spec = float(((pte < cut) & (yt == 0)).sum() / max((yt == 0).sum(), 1))

    # patient-level bootstrap of the test metrics
    gte = grp[te]
    by = pd.Series(np.arange(len(gte))).groupby(gte).apply(np.array)
    bs = []
    for i in range(a.n_boot):
        d = np.concatenate(by.sample(len(by), replace=True, random_state=a.seed + i).to_list())
        if 0 < yt[d].sum() < len(d):
            bs.append((roc_auc_score(yt[d], pte[d]), average_precision_score(yt[d], pte[d])))
    bs = np.array(bs)
    ci = lambda v: [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]
    metrics = dict(site=a.site, groups=a.groups, n_features=len(keep), split_counts=n,
                   auroc=round(float(roc_auc_score(yt, pte)), 3), auroc_ci=ci(bs[:, 0]),
                   auprc=round(float(average_precision_score(yt, pte)), 3), auprc_ci=ci(bs[:, 1]),
                   prevalence_test=round(float(yt.mean()), 3), brier=round(float(brier_score_loss(yt, pte)), 3),
                   threshold_val_youden=round(cut, 3), sensitivity=round(sens, 3),
                   specificity=round(spec, 3), best_iteration=int(getattr(model, "best_iteration", 0)))

    f, t, _ = roc_curve(yt, pte)
    i = thin(f)
    pd.DataFrame({"fpr": f[i], "tpr": t[i]}).to_csv(out / "roc.csv", index=False)
    pr, rc, _ = precision_recall_curve(yt, pte)
    i = thin(pr)
    pd.DataFrame({"recall": rc[i], "precision": pr[i]}).to_csv(out / "pr.csv", index=False)

    # patient-level cross-validation over all development stays (no early stopping: fixed rounds)
    rounds = max(int(getattr(model, "best_iteration", 200)) + 1, 20)
    cv = []
    for fold, (a_i, b_i) in enumerate(StratifiedGroupKFold(a.folds, shuffle=True, random_state=a.seed)
                                      .split(X, y, grp), 1):
        m = make_model(a.seed, n_estimators=rounds).fit(X.iloc[a_i], y[a_i], verbose=False)
        p = m.predict_proba(X.iloc[b_i])[:, 1]
        cv.append(dict(fold=fold, n_stays=len(b_i), n_aki=int(y[b_i].sum()),
                       auprc=average_precision_score(y[b_i], p), auroc=roc_auc_score(y[b_i], p)))
    cv = pd.DataFrame(cv)
    cv.to_csv(out / "cv.csv", index=False)
    metrics["cv_auprc_mean"] = round(float(cv["auprc"].mean()), 3)
    metrics["cv_auprc_sd"] = round(float(cv["auprc"].std()), 3)

    # calibration (deciles of predicted risk on the test set; model trained without class weights)
    q = pd.qcut(pte, 10, duplicates="drop")
    cal = (pd.DataFrame({"p": pte, "y": yt, "bin": q}).groupby("bin", observed=True)
           .agg(mean_predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size")).reset_index(drop=True))
    cal.to_csv(out / "calibration.csv", index=False)

    # importance: weight, gain, cover
    booster = model.get_booster()
    imp = pd.DataFrame({t: pd.Series(booster.get_score(importance_type=t)) for t in ("weight", "gain", "cover")})
    imp = imp.reindex(keep).fillna(0.0)
    fg = json.loads((wdir / "feature_groups.json").read_text())
    imp.insert(0, "group", imp.index.map(fg))
    imp.sort_values("gain", ascending=False).to_csv(out / "importance.csv", index_label="feature")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))

    print(json.dumps({k: v for k, v in metrics.items() if k not in ("groups", "split_counts")}, indent=1))
    print("\ncross-validation:\n" + cv.round(3).to_string(index=False))
    print("\ntop features by gain:")
    print(imp.sort_values("gain", ascending=False).head(10).round(2).to_string())
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()