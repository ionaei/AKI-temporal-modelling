"""Temporal hold-out + patient-level development splits. No matching.

Hold-out (silent deployment): cutoff = admission time of the LAST valid stay
minus `holdout.months` (or a fixed `holdout.date`). Stays admitted on/after
the cutoff -> split 'holdout', never touched during development. Stays
admitted before the cutoff -> train / val / test (method below). With
`exclude_straddling: true` (default) development stays must also be
DISCHARGED before the cutoff, so no development data come from the
hold-out period; straddling stays are dropped and counted.

method: patient_random  – stratified by 'patient ever had AKI'
        temporal        – test = stays admitted on/after `test_start`;
                          train/val = earlier stays, split by patient.
                          Patients seen in train/val can optionally be
                          removed from test (`drop_seen_patients_from_test`).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _patient_split(st: pd.DataFrame, fracs: dict, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    ever = st.groupby("subject_id")["aki"].max()
    assign = {}
    for _, subj in ever.groupby(ever):
        ids = subj.index.to_numpy().copy()
        rng.shuffle(ids)
        n = len(ids)
        n_test = int(round(n * fracs.get("test", 0)))
        n_val = int(round(n * fracs["val"]))
        for i, s in enumerate(ids):
            assign[s] = "test" if i < n_test else ("val" if i < n_test + n_val else "train")
    return st["subject_id"].map(assign)


def holdout_cutoff(st: pd.DataFrame, hc: dict) -> pd.Timestamp:
    if hc.get("date"):
        return pd.Timestamp(hc["date"])
    return st["admit"].max() - pd.DateOffset(months=int(hc.get("months", 6)))


def make_split(labels: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    sc = cfg["split"]
    st = labels[~labels["exclude"]].copy()
    st["split"] = None
    meta = {}
    hc = sc.get("holdout") or {}
    dev = pd.Series(True, index=st.index)
    if hc.get("enabled"):
        cut = holdout_cutoff(st, hc)
        hold = st["admit"] >= cut
        straddle = (~hold) & (st["discharge"] >= cut)
        st.loc[hold, "split"] = "holdout"
        if hc.get("exclude_straddling", True):
            st = st[~straddle]
            hold, dev = hold[st.index], ~hold[st.index]
        else:
            dev = ~hold
        dev_subj = set(st.loc[dev, "subject_id"])
        meta = dict(cutoff=str(cut), last_admission=str(labels.loc[~labels["exclude"], "admit"].max()),
                    holdout_stays=int(hold.sum()), holdout_aki=int(st.loc[hold, "aki"].sum()),
                    dropped_straddling=int(straddle.sum()) if hc.get("exclude_straddling", True) else 0,
                    holdout_stays_of_dev_patients=int(st.loc[hold, "subject_id"].isin(dev_subj).sum()))
        log.info("hold-out cutoff %s: %s", cut.date(), meta)
    d = st[dev]
    if sc["method"] == "patient_random":
        st.loc[dev, "split"] = _patient_split(d, sc["fractions"], sc["seed"])
    elif sc["method"] == "temporal":
        t0 = pd.Timestamp(sc["test_start"])
        if hc.get("enabled") and t0 >= pd.Timestamp(meta["cutoff"]):
            raise ValueError("split.test_start must be before the hold-out cutoff")
        early = d["admit"] < t0
        st.loc[d.index[~early], "split"] = "test"
        st.loc[d.index[early], "split"] = _patient_split(d[early], {"val": sc["fractions"]["val"]},
                                                          sc["seed"])
        if sc.get("drop_seen_patients_from_test"):
            seen = set(d.loc[early, "subject_id"])
            drop = d.index[(~early) & d["subject_id"].isin(seen)]
            log.info("temporal split: dropping %d test stays of patients seen in train/val", len(drop))
            st = st.drop(index=drop)
    else:
        raise ValueError(sc["method"])
    # leakage guard: within development, one patient -> one split
    dv = st[st["split"] != "holdout"]
    if sc["method"] == "patient_random" and (dv.groupby("subject_id")["split"].nunique() > 1).any():
        raise RuntimeError("patient appears in more than one development split")
    summ = st.groupby("split").agg(stays=("stay_id", "size"), patients=("subject_id", "nunique"),
                                   aki=("aki", "sum"), severe=("severe", "sum"),
                                   first_admit=("admit", "min"), last_admit=("admit", "max"))
    log.info("split summary:\n%s", summ.to_string())
    meta["summary"] = summ.astype(str).to_dict(orient="index")
    return st[["stay_id", "subject_id", "split"]], meta
