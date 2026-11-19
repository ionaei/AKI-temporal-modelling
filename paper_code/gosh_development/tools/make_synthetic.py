"""Generate small synthetic datasets in GOSH and MIMIC-III/PIC raw formats.

Used by tests/ and for checking the pipeline end-to-end anywhere (no real
data). AKI cases have a creatinine rise preceded by rising lactate, falling
MAP and nephrotoxin exposure, so a working pipeline should learn something.

    python tools/make_synthetic.py        # writes data/_synthetic/{gosh,mimic}
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "_synthetic"


def _patients(rng, n, neonatal=False):
    rows = []
    t0 = pd.Timestamp("2018-01-01")
    for i in range(n):
        age_d = rng.uniform(0, 20) if neonatal else rng.uniform(1, 17.5 * 365)
        first = t0 + pd.Timedelta(days=float(rng.uniform(0, 1500)))
        rows.append(dict(pid=f"P{i:05d}", sex=rng.choice(["M", "F"]),
                         dob=first - pd.Timedelta(days=float(age_d)), first=first))
    return pd.DataFrame(rows)


def _simulate_stay(rng, pid, admit, los_h, age_y, aki, stay_no):
    """Return labs, vitals, meds rows (generic) for one stay."""
    base_cr = 25 + 3.2 * age_y + rng.normal(0, 3)
    onset_h = rng.uniform(36, max(40, los_h - 12)) if aki else np.inf
    labs, vit, meds = [], [], []
    t = rng.uniform(1, 8)
    while t < los_h:
        pre = np.clip((t - (onset_h - 48)) / 48, 0, 1) if aki else 0
        cr = base_cr * (1 + 1.2 * pre) + rng.normal(0, 2)
        labs += [(pid, admit + pd.Timedelta(hours=t), "CREAT", cr),
                 (pid, admit + pd.Timedelta(hours=t), "UREA", 4 + 6 * pre + rng.normal(0, .5)),
                 (pid, admit + pd.Timedelta(hours=t), "NA1", 140 + rng.normal(0, 2))]
        t += rng.uniform(8, 30) if not aki or t < onset_h - 60 else rng.uniform(6, 14)
    t = rng.uniform(0, 3)
    while t < los_h:
        risk = np.clip((t - (onset_h - 60)) / 60, 0, 1) if aki else 0
        labs += [(pid, admit + pd.Timedelta(hours=t), "POCLAC", 1.2 + 3 * risk + abs(rng.normal(0, .4))),
                 (pid, admit + pd.Timedelta(hours=t), "POCPH", 7.38 - .12 * risk + rng.normal(0, .02))]
        t += rng.uniform(3, 10)
    t = 0.5
    while t < los_h:
        risk = np.clip((t - (onset_h - 60)) / 60, 0, 1) if aki else 0
        hr = 160 - 4 * age_y + 25 * risk + rng.normal(0, 8)
        vit += [(pid, admit + pd.Timedelta(hours=t), "HR", hr),
                (pid, admit + pd.Timedelta(hours=t), "MAP", 45 + 2 * age_y - 12 * risk + rng.normal(0, 5)),
                (pid, admit + pd.Timedelta(hours=t), "TEMP", 37 + rng.normal(0, .4))]
        t += 1.0
    if aki or rng.random() < 0.25:
        s = (onset_h - 72) if aki else rng.uniform(0, los_h)
        s = max(0.0, s)
        for d in range(4):
            st = admit + pd.Timedelta(hours=s + 24 * d)
            meds.append((pid, st, st + pd.Timedelta(hours=1), "Vancomycin 250mg IV", "IV", ""))
            meds.append((pid, st, st + pd.Timedelta(hours=1), "vanc oral", "Oral", ""))
    if aki and rng.random() < 0.6:
        st = admit + pd.Timedelta(hours=max(0, onset_h - 30))
        meds.append((pid, st, st + pd.Timedelta(hours=30), "Noradrenaline infusion", "IV", ""))
    if rng.random() < 0.3:
        st = admit + pd.Timedelta(hours=rng.uniform(0, los_h))
        meds.append((pid, st, st, "", "IV", "2007170123"))     # furosemide via GPI only
    return labs, vit, meds


def make_gosh(n=500, seed=1):
    rng = np.random.default_rng(seed)
    out = OUT / "gosh"
    out.mkdir(parents=True, exist_ok=True)
    pts = _patients(rng, n)
    adm, labs, vit, meds, procs, wards = [], [], [], [], [], []
    for p in pts.itertuples():
        t = p.first
        for s in range(rng.integers(1, 4)):
            los = float(rng.uniform(30, 400))
            age_y = (t - p.dob).days / 365.25
            aki = rng.random() < 0.15
            end = t + pd.Timedelta(hours=los)
            pre = t - pd.Timedelta(hours=float(rng.uniform(12, 72)))   # ward time before ICU
            adm.append((p.pid, pre, end + pd.Timedelta(hours=24)))       # hospital admission
            wards.append((p.pid, "WARD_A", 0, pre, t))
            if rng.random() < 0.3:     # ICU bed move: two contiguous ICU ward stays
                cut = t + pd.Timedelta(hours=los / 2)
                wards += [(p.pid, "PICU", 1, t, cut),
                          (p.pid, "CICU", 1, cut + pd.Timedelta(hours=1), end)]
            else:
                wards.append((p.pid, "PICU", 1, t, end))
            wards.append((p.pid, "WARD_B", 0, end, end + pd.Timedelta(hours=24)))
            # ward-only admissions (never ICU) must not enter the cohort
            if rng.random() < 0.1:
                w0 = end + pd.Timedelta(days=2)
                adm.append((p.pid, w0, w0 + pd.Timedelta(days=3)))
                wards.append((p.pid, "WARD_C", 0, w0, w0 + pd.Timedelta(days=3)))
            l, v, m = _simulate_stay(rng, p.pid, t, los, age_y, aki, s)
            labs += l; vit += v; meds += m
            if rng.random() < 0.01:
                procs.append((p.pid, "X40.3", t + pd.Timedelta(hours=10)))
            procs.append((p.pid, "L91.1", t + pd.Timedelta(hours=2)))
            t = t + pd.Timedelta(hours=los) + pd.Timedelta(days=float(rng.uniform(5, 200)))
    pd.DataFrame(adm, columns=["project_id", "start_datetime", "end_datetime"]).to_csv(
        out / "hospital_admissions.csv", index=False)
    pd.DataFrame(wards, columns=["project_id", "ward_name", "icu_ward_stay", "ward_start",
                                 "ward_end"]).to_csv(out / "ward_stays.csv", index=False)
    pd.DataFrame({"project_id": pts.pid, "birth_date": pts.dob.dt.date,
                  "sex_name": pts.sex.map({"M": "Male", "F": "Female"})}).to_csv(
        out / "demographics.csv", index=False)
    L = pd.DataFrame(labs, columns=["project_id", "end_datetime", "component_basename",
                                    "NumericValue"])
    L["Unit"] = L["component_basename"].map({"CREAT": "umol/L", "UREA": "mmol/L",
                                             "NA1": "mmol/L", "POCLAC": "mmol/L", "POCPH": ""})
    L.to_csv(out / "lab_components.csv", index=False)
    pd.DataFrame(vit, columns=["project_id", "time", "code", "value"]).to_csv(
        out / "vitals.csv", index=False)
    pd.DataFrame(meds, columns=["project_id", "start_datetime", "end_datetime", "drug_name",
                                "route", "drug_gpi"]).to_csv(out / "medication_admins.csv",
                                                             index=False)
    pd.DataFrame(procs, columns=["project_id", "proc_local_code", "start_datetime"]).to_csv(
        out / "procedures.csv", index=False)
    return out


def make_mimic(n=300, seed=2):
    """MIMIC-III/PIC-shaped tables (creatinine in mg/dL, BUN, dates-only prescriptions)."""
    rng = np.random.default_rng(seed)
    out = OUT / "mimic"
    out.mkdir(parents=True, exist_ok=True)
    pts = _patients(rng, n, neonatal=True)
    icu, lab, chart, rx = [], [], [], []
    for i, p in enumerate(pts.itertuples()):
        sid = str(100 + i)
        t = p.first
        los = float(rng.uniform(40, 300))
        aki = rng.random() < 0.15
        unit = "NICU" if rng.random() < 0.9 else "PICU"
        icu.append((sid, f"{i}", unit, t, t + pd.Timedelta(hours=los)))
        l, v, m = _simulate_stay(rng, sid, t, los, 0.02, aki, 0)
        codes = {"CREAT": 50912, "UREA": 51006, "NA1": 50983, "POCLAC": 50813, "POCPH": 50820}
        for s, tt, c, val in l:
            if c == "CREAT":
                val = val / 88.42; u = "mg/dL"
            elif c == "UREA":
                val = val / 0.357; u = "mg/dL"
            else:
                u = {"NA1": "mEq/L", "POCLAC": "mmol/L", "POCPH": "units"}[c]
            lab.append((s, codes[c], tt, val, u))
        vc = {"HR": 211, "MAP": 52, "TEMP": 678}
        for s, tt, c, val in v:
            if c == "TEMP":
                val = val * 9 / 5 + 32; u = "Deg. F"
            else:
                u = "BPM" if c == "HR" else "mmHg"
            chart.append((s, vc[c], tt, val, u))
        for s, st, en, name, route, gpi in m:
            if name:
                rx.append((s, st.normalize(), en.normalize(), name, route))
    pd.DataFrame({"SUBJECT_ID": [str(100 + i) for i in range(n)], "GENDER": pts.sex,
                  "DOB": pts.dob.dt.date}).to_csv(out / "PATIENTS.csv", index=False)
    pd.DataFrame(icu, columns=["SUBJECT_ID", "ICUSTAY_ID", "FIRST_CAREUNIT", "INTIME",
                               "OUTTIME"]).to_csv(out / "ICUSTAYS.csv", index=False)
    pd.DataFrame(lab, columns=["SUBJECT_ID", "ITEMID", "CHARTTIME", "VALUENUM",
                               "VALUEUOM"]).to_csv(out / "LABEVENTS.csv", index=False)
    pd.DataFrame(chart, columns=["SUBJECT_ID", "ITEMID", "CHARTTIME", "VALUENUM",
                                 "VALUEUOM"]).to_csv(out / "CHARTEVENTS.csv", index=False)
    pd.DataFrame(rx, columns=["SUBJECT_ID", "STARTDATE", "ENDDATE", "DRUG",
                              "ROUTE"]).to_csv(out / "PRESCRIPTIONS.csv", index=False)
    pd.DataFrame({"ITEMID": [211, 52, 678], "LABEL": ["Heart Rate", "Arterial BP Mean",
                                                      "Temperature F"]}).to_csv(
        out / "D_ITEMS.csv", index=False)
    pd.DataFrame({"ITEMID": [50912, 51006, 50983, 50813, 50820],
                  "LABEL": ["Creatinine", "Urea Nitrogen", "Sodium", "Lactate", "pH"],
                  "FLUID": ["Blood"] * 5}).to_csv(out / "D_LABITEMS.csv", index=False)
    return out


if __name__ == "__main__":
    print(make_gosh(), make_mimic(), file=sys.stderr)
