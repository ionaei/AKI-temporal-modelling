"""KDIGO serum-creatinine AKI labelling (paediatric + neonatal-modified).

Labelling rules
---------------
* Criterion (i): absolute rise >= `abs_rise_umol` within 48 hours.
* Criterion (ii): the ratio criterion uses a reference value from the PRECEDING 7 DAYS
  (rolling minimum, including values from earlier encounters). If nothing
  exists in the 7-day window, the lowest value in the preceding
  `older_baseline_days` is used and its source is recorded. A population
  normal is available only if `allow_population_baseline` is true (it is
  false for all reported analyses).
* Stage is reported at onset AND as the PEAK stage within `stage_window_days`
  after onset (KDIGO stages by maximum severity). `severe` = peak stage >= 2.
* AKI-evaluable cohort: stays with < `min_creatinine_in_stay` creatinine values
  (in-stay plus possible pre-admission references; at least one in-stay) are
  excluded rather than labelled 'no AKI'.
* Infants (< 1 year at the sample) use `older_baseline_days_infant` for the
  fallback reference window.
* Neonates (< `neonatal_max_age_days` at the time of the sample) use the
  neonatal-modified definition: reference = lowest previous value, no
  population fallback, stage 3 at SCr >= 221 umol/L (2.5 mg/dL), and samples
  in the first `neonatal_ignore_first_h` hours of life cannot trigger onset
  (maternal creatinine).
* AKI criteria already met in the `pre_admission_aki_lookback_days` before
  admission (e.g. on the ward before ICU transfer) -> `aki_before_admission`,
  excluded: this is not ICU-acquired AKI even if no new trigger occurs in ICU.
* Optional renal replacement therapy (RRT) events set stage 3 and drive
  exclusions.
* Diagnostic fields describing HOW onset was found (trigger, reference source,
  whether onset was the first in-stay creatinine) are kept for reporting.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .concepts import population_creatinine

log = logging.getLogger(__name__)
H = np.timedelta64(1, "h")


def _stage(ratio: float, value: float, neonatal: bool, p: dict) -> int:
    abs3 = p["neonatal_stage3_umol"] if neonatal else p["stage3_abs_umol"]
    if (not np.isnan(ratio) and ratio >= 3.0) or value >= abs3:
        return 3
    if not np.isnan(ratio) and ratio >= 2.0:
        return 2
    return 1


def older_days(age_days: float, p: dict) -> int:
    """Fallback-reference window: shorter for infants, whose creatinine changes fast."""
    if age_days < 365 and p.get("older_baseline_days_infant") is not None:
        return int(p["older_baseline_days_infant"])
    return int(p["older_baseline_days"])


def _first_trigger(t, v, idx, dob, sex, p):
    """Scan samples `idx` in time order; return (position in idx, i, trigger, ref, src)."""
    for k, i in enumerate(idx):
        ti, vi = t[i], v[i]
        age_d = (ti - dob) / np.timedelta64(1, "D")
        older = np.timedelta64(int(older_days(age_d, p) * 24), "h")
        neonatal = age_d < p["neonatal_max_age_days"]
        if neonatal and age_d * 24 < p["neonatal_ignore_first_h"]:
            continue
        prior = t < ti
        w48 = prior & (t >= ti - 48 * H)
        w7 = prior & (t >= ti - 168 * H)
        # reference value for the ratio criterion
        if w7.any():
            ref, src = v[w7].min(), "rolling_7d"
        elif not neonatal and (prior & (t >= ti - older)).any():
            ref, src = v[prior & (t >= ti - older)].min(), "older_<=%dd" % older_days(age_d, p)
        elif not neonatal and p["allow_population_baseline"]:
            ref, src = population_creatinine(age_d / 365.25, sex), "population"
        else:
            ref, src = np.nan, "none"
        if w48.any() and vi - v[w48].min() >= p["abs_rise_umol"]:
            return k, i, "abs_rise_48h", ref, src, neonatal
        if not np.isnan(ref) and ref > 0 and vi / ref >= p["ratio"]:
            return k, i, f"ratio_vs_{src}", ref, src, neonatal
    return None


def label_stay(t: np.ndarray, v: np.ndarray, admit, discharge, dob, sex: str,
               p: dict, rrt_t: np.ndarray | None = None) -> dict:
    """t, v: ALL creatinine times/values for the subject (sorted, any encounter/ward).

    Samples from before admission (e.g. the ward before ICU) are used only as
    REFERENCE values and to detect AKI that was already present before
    admission (`aki_before_admission`, excluded downstream). Onset itself must
    fall inside [admit, discharge].
    """
    admit, discharge, dob = (np.datetime64(x) for x in (admit, discharge, dob))
    in_stay = np.where((t >= admit) & (t <= discharge))[0]
    # values that could serve to DETECT AKI in this stay: in-stay samples plus any
    # possible reference before admission (within the age-appropriate fallback window)
    age_admit = (admit - dob) / np.timedelta64(1, "D")
    look_ref = np.timedelta64(int(older_days(age_admit, p) * 24), "h")
    n_evaluable = int(((t >= admit - look_ref) & (t <= discharge)).sum())
    look = np.timedelta64(int(p.get("pre_admission_aki_lookback_days", 7) * 24), "h")
    pre = np.where((t >= admit - look) & (t < admit))[0]
    res = dict(aki=False, onset_dt=pd.NaT, trigger=None, ref_source=None, ref_value=np.nan,
               stage_onset=0, stage_peak=0, n_creat_stay=len(in_stay),
               n_creat_evaluable=n_evaluable,
               n_creat_pre_onset=len(in_stay), onset_first_in_stay=False,
               aki_before_admission=_first_trigger(t, v, pre, dob, sex, p) is not None)
    hit = _first_trigger(t, v, in_stay, dob, sex, p)
    if hit is None:
        return res
    k, i, trig, ref, src, neonatal = hit
    ti, vi = t[i], v[i]
    res.update(aki=True, onset_dt=pd.Timestamp(ti), trigger=trig, ref_source=src,
               ref_value=float(ref), n_creat_pre_onset=k, onset_first_in_stay=(k == 0))
    res["stage_onset"] = _stage(vi / ref if ref > 0 else np.nan, vi, neonatal, p)
    horizon = ti + np.timedelta64(int(p["stage_window_days"] * 24), "h")
    after = (t >= ti) & (t <= min(horizon, discharge))
    peak = max(_stage(vv / ref if ref > 0 else np.nan, vv, neonatal, p) for vv in v[after])
    if rrt_t is not None and len(rrt_t) and ((rrt_t >= ti) & (rrt_t <= horizon)).any():
        peak = 3
    res["stage_peak"] = int(max(peak, res["stage_onset"]))
    return res


def label_cohort(stays: pd.DataFrame, labs: pd.DataFrame, rrt: pd.DataFrame,
                 cfg: dict) -> pd.DataFrame:
    p = cfg["labelling"]
    creat = labs[labs["concept"] == "creatinine"][["subject_id", "time", "value"]]
    creat = creat.dropna().sort_values(["subject_id", "time"])
    # identical duplicates (same sample reported twice) would create 0-length windows
    creat = creat.drop_duplicates(["subject_id", "time"])
    by_subj = {s: (g["time"].to_numpy("datetime64[ns]"), g["value"].to_numpy(float))
               for s, g in creat.groupby("subject_id")}
    rrt_by = {s: g["time"].sort_values().to_numpy("datetime64[ns]")
              for s, g in rrt.groupby("subject_id")} if len(rrt) else {}
    empty_t = np.array([], dtype="datetime64[ns]")

    rows = []
    for s in stays.itertuples(index=False):
        t, v = by_subj.get(s.subject_id, (empty_t, np.array([])))
        r = label_stay(t, v, s.admit, s.discharge, s.dob, s.sex, p, rrt_by.get(s.subject_id))
        r["stay_id"] = s.stay_id
        rows.append(r)
    lab = stays.merge(pd.DataFrame(rows), on="stay_id")
    lab["onset_h"] = (lab["onset_dt"] - lab["admit"]).dt.total_seconds() / 3600
    lab["severe"] = lab["stage_peak"] >= 2

    # ── exclusions (kept in the table with a reason, filtered later) ──
    reason = pd.Series("", index=lab.index)
    if p.get("exclude_aki_before_admission", True):
        reason[lab["aki_before_admission"]] = "aki_before_admission"
    reason[lab["aki"] & (lab["onset_h"] < p["exclude_onset_before_h"]) & (reason == "")] = \
        "aki_on_arrival"
    if len(rrt):
        rr = rrt.merge(lab[["stay_id", "subject_id", "admit", "discharge", "onset_dt"]],
                       on="subject_id")
        prior_window = pd.Timedelta(days=p["prior_rrt_exclusion_days"])
        prior = rr[(rr["time"] < rr["admit"]) & (rr["time"] >= rr["admit"] - prior_window)]
        before_onset = rr[(rr["time"] >= rr["admit"]) &
                          (rr["time"] < rr["onset_dt"].fillna(rr["discharge"]))]
        reason[lab["stay_id"].isin(before_onset["stay_id"]) & (reason == "")] = "rrt_before_onset"
        reason[lab["stay_id"].isin(prior["stay_id"]) & (reason == "")] = "recent_prior_rrt"
    # AKI-evaluable cohort: >= 1 creatinine in the stay AND >= min_creatinine_in_stay values
    # counting possible references before admission. Stays where AKI could not have been
    # detected are excluded instead of being labelled 'no AKI' (surveillance confounding).
    min_cr = p.get("min_creatinine_in_stay", 1 if p.get("require_creatinine_in_stay") else 0)
    if min_cr > 0:
        not_eval = (lab["n_creat_stay"] < 1) | (lab["n_creat_evaluable"] < min_cr)
        reason[not_eval & (reason == "")] = f"fewer_than_{min_cr}_creatinine"
    lab["exclude_reason"] = reason
    lab["exclude"] = reason != ""

    inc = lab[~lab["exclude"]]
    log.info("stays %d | excluded %d (%s) | included %d | AKI %d (%.1f%%) | severe %d",
             len(lab), lab["exclude"].sum(),
             lab.loc[lab["exclude"], "exclude_reason"].value_counts().to_dict(),
             len(inc), inc["aki"].sum(), 100 * inc["aki"].mean(), inc["severe"].sum())
    log.info("onset at first in-stay creatinine: %d of %d AKI",
             inc["onset_first_in_stay"].sum(), inc["aki"].sum())
    return lab