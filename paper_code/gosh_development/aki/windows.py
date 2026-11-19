"""Vectorised 6-hourly window construction and feature engineering.

One row per (stay, window k). Window k covers [admit + k*step, admit + (k+1)*step)
and its features use ONLY events timestamped strictly before window_end.
Rows stop at AKI onset. label = onset within [window_end, window_end + horizon].

Feature naming (prefix -> group used for ablations / transfer sets):
    static_*   age, sex                                    -> static
    time_*     hours since admission                       -> time
    lab_*      laboratory and blood-gas values / deltas    -> lab (creatinine -> creatinine)
    renal_*    creatinine ratio to 7-day reference, 48h rise -> creatinine
    vit_*      vital signs (no count features)             -> vital
    sup_*      organ support (FiO2, invasive ventilation)  -> support
    med_*      medication exposure history                 -> med
    proc_*     measurement process: counts, hours since last test -> process

Values are left as NaN where unknown; imputation happens after the split
(features.py) so nothing is fitted on test data.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .concepts import (AGE_BAND_NAMES, AGE_BANDS_DAYS, CONCEPTS, LAB_CONCEPTS,
                       MED_GROUPS, SUPPORT_CONCEPTS, VITAL_CONCEPTS, age_band,
                       med_groups_for, route_class)

log = logging.getLogger(__name__)


# ── grid ──────────────────────────────────────────────────────────────────
def build_grid(stays: pd.DataFrame, w: dict) -> pd.DataFrame:
    step = pd.Timedelta(hours=w["step_h"])
    limit = stays["onset_dt"].where(stays["aki"], stays["discharge"])
    n = np.floor((limit - stays["admit"]) / step).astype(int).clip(lower=0)
    n = n.clip(upper=w["max_windows"])
    st = stays.assign(n_windows=n)
    st = st[st["n_windows"] > 0]
    grid = st.loc[st.index.repeat(st["n_windows"]),
                  ["stay_id", "subject_id", "admit", "onset_dt", "aki", "age_days", "sex",
                   "stage_peak"]].copy()
    grid["k"] = grid.groupby("stay_id").cumcount()
    grid["window_end"] = grid["admit"] + (grid["k"] + 1) * step
    grid["hours_since_admit"] = (grid["k"] + 1) * w["step_h"]
    h2o = (grid["onset_dt"] - grid["window_end"]).dt.total_seconds() / 3600
    grid["hours_to_onset"] = h2o.where(grid["aki"])
    grid["label"] = (grid["aki"] & h2o.between(0, w["horizon_h"])).astype(np.int8)
    return grid.reset_index(drop=True), st[["stay_id", "subject_id", "admit", "n_windows"]]


def assign_to_stays(ev: pd.DataFrame, stays_n: pd.DataFrame, step_h: float,
                    time_col: str = "time") -> pd.DataFrame:
    """Attach stay_id and window index k to events (subject-level -> stay-level)."""
    if ev.empty:
        return ev.assign(stay_id=pd.Series(dtype=str), k=pd.Series(dtype=int))
    s = stays_n.sort_values("admit")
    e = ev.sort_values(time_col)
    e = pd.merge_asof(e, s, left_on=time_col, right_on="admit", by="subject_id",
                      direction="backward")
    e = e[e["stay_id"].notna()]
    k = np.floor((e[time_col] - e["admit"]) / pd.Timedelta(hours=step_h))
    e = e.assign(k=k)
    e = e[(e["k"] >= 0) & (e["k"] < e["n_windows"])]
    e["k"] = e["k"].astype(int)
    return e.drop(columns=["admit", "n_windows"])


# ── measured concepts (labs, gases, vitals, support) ──────────────────────
def _aggregate_measures(ev: pd.DataFrame, prefix: str, stats: list[str]) -> pd.DataFrame:
    ev = ev.sort_values("time")
    agg = ev.groupby(["stay_id", "k", "concept"])["value"].agg(stats)
    wide = agg.unstack("concept")
    wide.columns = [f"{prefix}{c}_{s}" for s, c in wide.columns]
    return wide


def measure_features(grid: pd.DataFrame, ev: pd.DataFrame, concepts: list[str],
                     prefix: str, stats: list[str], step_h: float,
                     with_process: bool, with_delta: bool, cap_h: float) -> pd.DataFrame:
    ev = ev[ev["concept"].isin(concepts)]
    idx = pd.MultiIndex.from_frame(grid[["stay_id", "k"]])
    if ev.empty:
        return pd.DataFrame(index=idx)
    if with_delta:
        ev = ev.sort_values(["stay_id", "concept", "time"])
        prev_v = ev.groupby(["stay_id", "concept"])["value"].shift()
        prev_t = ev.groupby(["stay_id", "concept"])["time"].shift()
        gap_h = (ev["time"] - prev_t).dt.total_seconds() / 3600
        ev = ev.assign(delta=(ev["value"] - prev_v).where(gap_h <= 168))
    wide = _aggregate_measures(ev, prefix, stats)
    if with_delta:
        d = ev.sort_values("time").groupby(["stay_id", "k", "concept"])["delta"].last()
        d = d.unstack("concept")
        d.columns = [f"{prefix}{c}_delta" for c in d.columns]
        wide = wide.join(d, how="outer")
    wide = wide.reindex(idx)

    # forward fill each concept's columns, limited by its carry-forward horizon
    for c in ev["concept"].unique():
        cols = [x for x in wide.columns if x.startswith(f"{prefix}{c}_")]
        lim = max(1, int(np.ceil(CONCEPTS[c].carry_h / step_h)))
        wide[cols] = wide[cols].groupby(level="stay_id").ffill(limit=lim)

    if with_process:
        n = ev.groupby(["stay_id", "k", "concept"]).size().unstack("concept")
        n = n.reindex(idx).fillna(0)
        n.columns = [f"proc_{c}_n" for c in n.columns]
        last_t = ev.groupby(["stay_id", "k", "concept"])["time"].max().unstack("concept")
        last_t = last_t.reindex(idx).groupby(level="stay_id").ffill()
        wend = pd.Series(grid["window_end"].to_numpy(), index=idx)
        hs = last_t.apply(lambda col: (wend - col).dt.total_seconds() / 3600)
        hs = hs.fillna(cap_h).clip(upper=cap_h)
        hs.columns = [f"proc_{c}_hsince" for c in hs.columns]
        wide = wide.join(n).join(hs)
    return wide


# ── creatinine reference features ─────────────────────────────────────────
# current value: in-stay samples only; reference minimum: any prior sample
# (including the ward before ICU), mirroring the KDIGO labelling.
def renal_features(grid: pd.DataFrame, labs: pd.DataFrame) -> pd.DataFrame:
    cr = labs[labs["concept"] == "creatinine"].sort_values(["subject_id", "time"])
    by = {s: (g["time"].to_numpy("datetime64[ns]"), g["value"].to_numpy(float))
          for s, g in cr.groupby("subject_id")}
    ratio = np.full(len(grid), np.nan)
    rise = np.full(len(grid), np.nan)
    H = np.timedelta64(1, "h")
    ends = grid["window_end"].to_numpy("datetime64[ns]")
    admits = grid["admit"].to_numpy("datetime64[ns]")
    for subj, rows in grid.groupby("subject_id").indices.items():
        if subj not in by:
            continue
        t, v = by[subj]
        for r in rows:
            e = ends[r]
            j = np.searchsorted(t, e, side="left")      # values strictly before window end
            if j == 0:
                continue
            last_t, last_v = t[j - 1], v[j - 1]
            if e - last_t > 72 * H or last_t < admits[r]:
                # too old, or measured before admission (ward): the CURRENT value must be
                # an in-stay sample; earlier samples are used only as the reference
                continue
            i7 = np.searchsorted(t, last_t - 168 * H, side="left")
            prior7 = v[i7:j - 1]
            if len(prior7):
                ratio[r] = last_v / prior7.min()
            i48 = np.searchsorted(t, last_t - 48 * H, side="left")
            prior48 = v[i48:j - 1]
            if len(prior48):
                rise[r] = last_v - prior48.min()
    return pd.DataFrame({"renal_creat_ratio_7d": ratio, "renal_creat_rise_48h": rise},
                        index=pd.MultiIndex.from_frame(grid[["stay_id", "k"]]))


# ── medications ───────────────────────────────────────────────────────────
def med_features(grid: pd.DataFrame, meds: pd.DataFrame, stays_n: pd.DataFrame,
                 w: dict) -> pd.DataFrame:
    idx = pd.MultiIndex.from_frame(grid[["stay_id", "k"]])
    out = pd.DataFrame(index=idx)
    if meds.empty:
        return out
    step = pd.Timedelta(hours=w["step_h"])
    names = pd.Series(meds["name"].unique())
    name2groups = dict(zip(names, names.map(med_groups_for)))
    m = meds.assign(group=meds["name"].map(name2groups)).explode("group")
    m = m[m["group"].notna()]
    m["route_cls"] = m["route"].map(route_class) if "route" in m else "unknown"
    excl = m.apply(lambda r: r["route_cls"] in MED_GROUPS[r["group"]].exclude_routes, axis=1)
    log.info("medication rows mapped to groups: %d (%d dropped for route)", len(m), excl.sum())
    m = m[~excl]
    if m.empty:
        return out
    carry = m["group"].map(lambda g: pd.Timedelta(hours=MED_GROUPS[g].carry_h))
    m["end_eff"] = np.maximum(m["end"], m["start"]) + carry

    # an interval may start before admission and run into the stay: join on subject
    j = m.merge(stays_n, on="subject_id")
    limit = j["admit"] + j["n_windows"] * step
    j = j[(j["end_eff"] > j["admit"]) & (j["start"] < limit)]
    if not w.get("pre_admission_med_carry", False):
        # only exposure that actually continues into the stay counts; a dose given on
        # the ward before ICU admission does not 'carry' into ICU windows
        j = j[j["end"] >= j["admit"]]
    if j.empty:
        return out
    k0 = np.floor((j["start"] - j["admit"]) / step).clip(lower=0).astype(int)
    k1 = np.floor((j["end_eff"] - j["admit"] - pd.Timedelta(seconds=1)) / step)
    k1 = np.minimum(k1, j["n_windows"] - 1).astype(int)
    k1 = np.maximum(k1, k0)
    j = j.assign(k0=k0.to_numpy(), k1=k1.to_numpy())
    j["k"] = [np.arange(a, b + 1) for a, b in zip(j["k0"], j["k1"])]
    ex = j[["stay_id", "group", "k"]].explode("k")
    ex["k"] = ex["k"].astype(int)
    flags = (ex.drop_duplicates().assign(v=1)
               .pivot_table(index=["stay_id", "k"], columns="group", values="v",
                            aggfunc="max"))
    flags = flags.reindex(idx).fillna(0)
    look = int(round(72 / w["step_h"]))
    cap = w["hours_since_cap_h"]
    for g in flags.columns:
        f = flags[g]
        out[f"med_{g}"] = f
        out[f"med_{g}_days72h"] = (f.groupby(level="stay_id")
                                     .rolling(look, min_periods=1).sum()
                                     .droplevel(0) * w["step_h"] / 24)
        kk = pd.Series(np.where(f.to_numpy() > 0, idx.get_level_values("k"), np.nan), index=idx)
        last_k = kk.groupby(level="stay_id").ffill()
        out[f"med_{g}_hsince"] = ((idx.get_level_values("k") - last_k) * w["step_h"]) \
            .fillna(cap).clip(upper=cap)
    nephro = [g for g in flags.columns if MED_GROUPS[g].nephrotoxin]
    out["med_nephrotoxin_n_concurrent"] = flags[nephro].sum(axis=1) if nephro else 0
    ag = out.get("med_aminoglycoside_days72h", 0)
    out["med_nephrotoxin_high_exposure"] = ((out["med_nephrotoxin_n_concurrent"] >= 3) |
                                            (pd.Series(ag, index=idx) >= 3)).astype(float)
    return out


# ── assemble ──────────────────────────────────────────────────────────────
def build_windows(stays: pd.DataFrame, labs: pd.DataFrame, vitals: pd.DataFrame,
                  meds: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    w = cfg["windows"]
    grid, stays_n = build_grid(stays, w)
    log.info("grid: %d windows, %d stays, %.2f%% positive", len(grid),
             grid["stay_id"].nunique(), 100 * grid["label"].mean())

    lab_ev = assign_to_stays(labs, stays_n, w["step_h"])
    vit_ev = assign_to_stays(vitals, stays_n, w["step_h"])
    sup_ev = pd.concat([lab_ev[lab_ev["concept"].isin(SUPPORT_CONCEPTS)],
                        vit_ev[vit_ev["concept"].isin(SUPPORT_CONCEPTS)]])
    cap = w["hours_since_cap_h"]
    parts = [
        measure_features(grid, lab_ev, LAB_CONCEPTS, "lab_", ["last", "min", "max"],
                         w["step_h"], with_process=True, with_delta=True, cap_h=cap),
        measure_features(grid, vit_ev, VITAL_CONCEPTS, "vit_", ["mean", "min", "max", "last"],
                         w["step_h"], with_process=False, with_delta=False, cap_h=cap),
        measure_features(grid, sup_ev, SUPPORT_CONCEPTS, "sup_", ["max"],
                         w["step_h"], with_process=False, with_delta=False, cap_h=cap),
        renal_features(grid, labs),
        med_features(grid, meds, stays_n, w),
    ]
    feats = pd.concat(parts, axis=1)
    feats = feats.reset_index(drop=True)
    win = pd.concat([grid.reset_index(drop=True), feats], axis=1)

    # static and time features
    age_now = win["age_days"] + win["hours_since_admit"] / 24
    win["static_age_log_days"] = np.log1p(age_now)
    bands = age_band(age_now, AGE_BANDS_DAYS, AGE_BAND_NAMES)
    for b in AGE_BAND_NAMES:
        win[f"static_age_{b}"] = (bands == b).astype(float)
    win["static_sex_male"] = (win["sex"] == "M").astype(float)
    win["time_log_hours_since_admit"] = np.log1p(win["hours_since_admit"])

    groups = feature_groups(win.columns)
    log.info("features: %d (%s)", len(groups),
             pd.Series(groups).value_counts().to_dict())
    return win, groups


def feature_groups(columns) -> dict:
    g = {}
    for c in columns:
        if c.startswith("lab_creatinine_") or c.startswith("renal_"):
            g[c] = "creatinine"
        elif c.startswith("lab_"):
            g[c] = "lab"
        elif c.startswith("vit_"):
            g[c] = "vital"
        elif c.startswith("sup_"):
            g[c] = "support"
        elif c.startswith("med_"):
            g[c] = "med"
        elif c.startswith("proc_"):
            g[c] = "process"
        elif c.startswith("static_"):
            g[c] = "static"
        elif c.startswith("time_"):
            g[c] = "time"
    return g


def availability(win: pd.DataFrame, groups: dict) -> pd.DataFrame:
    """Per-feature coverage at this site; used to define the cross-site 'core' set."""
    rows = []
    for c in groups:
        x = win[c]
        rows.append(dict(feature=c, group=groups[c], non_null=float(x.notna().mean()),
                         nonzero=float((x.fillna(0) != 0).mean()),
                         n_unique=int(x.nunique(dropna=True))))
    return pd.DataFrame(rows)
