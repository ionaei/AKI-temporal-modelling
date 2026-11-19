"""Evaluation metrics, all computed at the TRUE (unmatched) prevalence.

Window level   AUROC, AUPRC, Brier, calibration intercept/slope, reliability table
Horizon        AUROC of AKI windows k hours before onset vs. all windows of non-AKI
               stays, for bins inside AND beyond the 24h training horizon
Alert level    at a threshold fixed on validation:
                 precision@H    share of alert windows followed by onset within H h
                 sens@H         share of AKI stays with >=1 alert in the H h before onset
                 lead@H         hours before onset of the earliest alert inside that H
                 alerts / 100 patient-days, non-AKI stays never alerted
               stay level: AKI stays with any alert before onset (legacy_sens_ever)
               and non-AKI stays never alerted (legacy_spec_stay)
Decision curve net benefit for the 24h label
Bootstrap      resampling PATIENTS (stays from one patient are not independent)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

HORIZONS = (24, 48, 72)


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


# ── window level ──────────────────────────────────────────────────────────
def discrimination(y, p) -> dict:
    if len(np.unique(y)) < 2:
        return dict(auroc=np.nan, auprc=np.nan, prevalence=float(np.mean(y)))
    return dict(auroc=roc_auc_score(y, p), auprc=average_precision_score(y, p),
                prevalence=float(np.mean(y)))


def calibration(y, p) -> dict:
    if len(np.unique(y)) < 2:       # e.g. a month or subgroup with no positive windows
        return dict(brier=brier_score_loss(y, p, labels=[0, 1]), cal_intercept=np.nan,
                    cal_slope=np.nan, mean_pred=float(np.mean(p)), observed=float(np.mean(y)))
    lp = _logit(p)
    slope = LogisticRegression(C=1e6, max_iter=1000).fit(lp.reshape(-1, 1), y).coef_[0, 0]
    # calibration-in-the-large: intercept with logit(p) as offset (1-D Newton)
    a = 0.0
    for _ in range(50):
        q = 1 / (1 + np.exp(-np.clip(lp + a, -50, 50)))
        g, h = np.sum(y - q), np.sum(q * (1 - q))
        a += g / max(h, 1e-9)
        if abs(g) < 1e-8:
            break
    return dict(brier=brier_score_loss(y, p), cal_intercept=float(a), cal_slope=float(slope),
                mean_pred=float(np.mean(p)), observed=float(np.mean(y)))


def reliability_table(y, p, n_bins=10) -> pd.DataFrame:
    q = pd.qcut(p, n_bins, duplicates="drop")
    return (pd.DataFrame({"y": y, "p": p, "bin": q}).groupby("bin", observed=True)
              .agg(n=("y", "size"), mean_pred=("p", "mean"), observed=("y", "mean"))
              .reset_index(drop=True))


def horizon_discrimination(df: pd.DataFrame, bins=((0, 6), (6, 12), (12, 24), (24, 48),
                                                    (48, 72), (72, 120))) -> pd.DataFrame:
    neg = df.loc[~df["aki"], "prob"].to_numpy()
    rows = []
    for lo, hi in bins:
        pos = df.loc[df["aki"] & (df["hours_to_onset"] >= lo) & (df["hours_to_onset"] < hi),
                     "prob"].to_numpy()
        if len(pos) < 5 or len(neg) < 5:
            continue
        y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
        rows.append(dict(hours_before_onset=f"{lo}-{hi}", n_pos_windows=len(pos),
                         auroc=roc_auc_score(y, np.r_[pos, neg])))
    return pd.DataFrame(rows)


# ── alert level ───────────────────────────────────────────────────────────
class AlertArrays:
    """Pre-factorised arrays so thresholds/bootstraps are cheap."""

    def __init__(self, df: pd.DataFrame, step_h: float):
        self.stay, stays = pd.factorize(df["stay_id"])
        self.n_stays = len(stays)
        self.p = df["prob"].to_numpy(float)
        self.h2o = df["hours_to_onset"].to_numpy(float)
        st_aki = np.zeros(self.n_stays, bool)
        st_aki[self.stay] = df["aki"].to_numpy(bool)
        st_sev = np.zeros(self.n_stays, bool)
        st_sev[self.stay] = df["stage_peak"].to_numpy() >= 2
        self.st_aki, self.st_sev = st_aki, st_sev
        self.step_h = step_h


def alert_metrics(a: AlertArrays, thr: float) -> dict:
    alert = a.p >= thr
    h = np.where(np.isnan(a.h2o), np.inf, a.h2o)
    n_alerts = int(alert.sum())
    pdays = len(a.p) * a.step_h / 24
    out = dict(threshold=thr, n_alerts=n_alerts,
               alerts_per_100_patient_days=100 * n_alerts / max(pdays, 1e-9))
    n_aki = a.st_aki.sum()
    for H in HORIZONS:
        within = alert & (h <= H)
        out[f"precision@{H}h"] = within.sum() / n_alerts if n_alerts else np.nan
        det = np.zeros(a.n_stays, bool)
        det[a.stay[within]] = True
        out[f"sens@{H}h"] = det[a.st_aki].mean() if n_aki else np.nan
        out[f"sens_severe@{H}h"] = det[a.st_sev].mean() if a.st_sev.any() else np.nan
        earliest = np.full(a.n_stays, np.nan)
        hh = h[within]
        if len(hh):
            np.fmax.at(earliest, a.stay[within], hh)
        e = earliest[det]
        out[f"lead@{H}h_median"] = float(np.median(e)) if len(e) else np.nan
    # stay-level sensitivity: any alert at any time before onset
    ever = np.zeros(a.n_stays, bool)
    ever[a.stay[alert & np.isfinite(h)]] = True
    first_any = np.full(a.n_stays, np.nan)
    sel = alert & np.isfinite(h)
    if sel.any():
        np.fmax.at(first_any, a.stay[sel], h[sel])
    out["legacy_sens_ever"] = ever[a.st_aki].mean() if n_aki else np.nan
    out["legacy_first_alert_median_h"] = float(np.nanmedian(first_any[ever])) if ever.any() else np.nan
    any_alert = np.zeros(a.n_stays, bool)
    any_alert[a.stay[alert]] = True
    out["nonaki_stays_never_alerted"] = (~any_alert[~a.st_aki]).mean()
    # stay-level definitions reported in the paper:
    #   sensitivity = AKI stays with any alert before onset; specificity = non-AKI stays never alerted
    out["legacy_spec_stay"] = out["nonaki_stays_never_alerted"]
    out["legacy_youden_stay"] = out["legacy_sens_ever"] + out["legacy_spec_stay"] - 1
    na = ~np.isfinite(h)
    out["nonaki_window_alert_rate"] = alert[na].mean() if na.any() else np.nan
    return out


def threshold_sweep(a: AlertArrays, thresholds) -> pd.DataFrame:
    return pd.DataFrame([alert_metrics(a, t) for t in thresholds])


def choose_threshold(sweep: pd.DataFrame, rule: dict) -> float:
    """Pick a threshold from a validation sweep.

    Constraint rules
      {metric: 'sens@48h', min: 0.80}                  -> highest thr meeting it
      {metric: 'precision@48h', min: 0.30}             -> lowest thr meeting it
      {metric: 'alerts_per_100_patient_days', max: 10} -> lowest thr meeting it
    Optimisation rules
      {maximise: 'legacy_youden_stay'}                 -> thr with the largest value
      {maximise: 'legacy_spec_stay', subject_to: {metric: 'legacy_sens_ever', min: 0.75}}
                                                       -> largest value among thresholds
                                                          meeting the constraint
    Ties are broken towards the HIGHER threshold (fewer alerts).
    """
    if "maximise" in rule:
        sw = sweep
        c = rule.get("subject_to")
        if c:
            ok = sw[c["metric"]] >= c["min"] if "min" in c else sw[c["metric"]] <= c["max"]
            if not ok.any():
                return float(sw["threshold"].min())
            sw = sw[ok]
        best = sw[rule["maximise"]].max()
        return float(sw.loc[sw[rule["maximise"]] >= best - 1e-12, "threshold"].max())
    m = rule["metric"]
    if "min" in rule and m.startswith("sens"):
        ok = sweep[sweep[m] >= rule["min"]]
        return float(ok["threshold"].max()) if len(ok) else float(sweep["threshold"].min())
    if "min" in rule:
        ok = sweep[sweep[m] >= rule["min"]]
        return float(ok["threshold"].min()) if len(ok) else float(sweep["threshold"].max())
    ok = sweep[sweep[m] <= rule["max"]]
    return float(ok["threshold"].min()) if len(ok) else float(sweep["threshold"].max())


def threshold_grid(df: pd.DataFrame, n: int = 300) -> np.ndarray:
    """Candidate thresholds: upper window-level quantiles plus the quantiles of each
    stay's maximum risk (the values that matter for stay-level 'any alert' rules)."""
    w = np.quantile(df["prob"], np.linspace(0.5, 0.9995, n))
    s = np.quantile(df.groupby("stay_id")["prob"].max(), np.linspace(0, 1, n))
    return np.unique(np.r_[w, s])


# ── decision curve ────────────────────────────────────────────────────────
def decision_curve(y, p, pts=np.arange(0.01, 0.51, 0.01)) -> pd.DataFrame:
    n, prev = len(y), np.mean(y)
    rows = []
    for t in pts:
        pred = p >= t
        tp = np.sum(pred & (y == 1)); fp = np.sum(pred & (y == 0))
        rows.append(dict(threshold=t, net_benefit=tp / n - fp / n * t / (1 - t),
                         treat_all=prev - (1 - prev) * t / (1 - t), treat_none=0.0))
    return pd.DataFrame(rows)


# ── everything at once + patient-cluster bootstrap ────────────────────────
def summary(df: pd.DataFrame, thr: float, step_h: float) -> dict:
    y, p = df["label"].to_numpy(), df["prob"].to_numpy()
    out = {**discrimination(y, p), **calibration(y, p)}
    out.update(alert_metrics(AlertArrays(df, step_h), thr))
    return out


def bootstrap(df: pd.DataFrame, thr: float, step_h: float, n_boot: int, seed: int = 0,
              keys=("auroc", "auprc", "brier", "cal_slope", "precision@24h", "precision@48h",
                    "sens@24h", "sens@48h", "sens_severe@48h", "lead@48h_median",
                    "alerts_per_100_patient_days", "nonaki_stays_never_alerted",
                    "legacy_sens_ever", "legacy_spec_stay", "legacy_youden_stay")) -> pd.DataFrame:
    if n_boot <= 0:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    subj_rows = df.groupby("subject_id").indices
    subjects = np.array(list(subj_rows))
    vals = {k: [] for k in keys}
    for b in range(n_boot):
        draw = rng.choice(subjects, len(subjects), replace=True)
        rows = [subj_rows[s] for s in draw]
        d = df.iloc[np.concatenate(rows)].copy()
        # duplicated patients must count as separate stays
        rep = np.repeat(np.arange(len(rows)), [len(r) for r in rows])
        d["stay_id"] = d["stay_id"].astype(str) + "#" + rep.astype(str)
        try:
            s = summary(d, thr, step_h)
        except ValueError:
            continue
        for k in keys:
            vals[k].append(s.get(k, np.nan))
    return pd.DataFrame({k: dict(lo=np.nanpercentile(v, 2.5), hi=np.nanpercentile(v, 97.5),
                                 n=len(v)) for k, v in vals.items()}).T
