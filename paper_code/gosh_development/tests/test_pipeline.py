"""Unit tests: run `pytest -q` from the project root (no real data needed)."""
import numpy as np
import pandas as pd
import pytest

from aki.concepts import med_groups_for, route_class, to_canonical
from aki.labelling import label_stay
from aki.windows import build_windows

P = dict(abs_rise_umol=26.5, ratio=1.5, older_baseline_days=90, allow_population_baseline=True,
         stage3_abs_umol=353.6, stage_window_days=7, neonatal_max_age_days=28,
         neonatal_ignore_first_h=48, neonatal_stage3_umol=221, exclude_onset_before_h=24,
         prior_rrt_exclusion_days=90, require_creatinine_in_stay=False)
T0 = pd.Timestamp("2020-01-01")


def _lab(hours, values):
    t = np.array([T0 + pd.Timedelta(hours=h) for h in hours], dtype="datetime64[ns]")
    return t, np.asarray(values, float)


def _label(hours, values, age_days=3000, los_h=500, sex="M"):
    t, v = _lab(hours, values)
    return label_stay(t, v, T0, T0 + pd.Timedelta(hours=los_h),
                      T0 - pd.Timedelta(days=age_days), sex, P)


# ── labelling ────────────────────────────────────────────────────────────
def test_abs_rise_within_48h():
    r = _label([10, 40], [40, 70])
    assert r["aki"] and r["trigger"] == "abs_rise_48h"
    assert r["onset_dt"] == T0 + pd.Timedelta(hours=40)


def test_abs_rise_outside_48h_not_triggered_but_ratio_is():
    r = _label([10, 70], [40, 61])           # rise 21 (<26.5) but ratio 1.53 within 7 d
    assert r["aki"] and r["trigger"].startswith("ratio_vs_rolling_7d")


def test_ratio_needs_reference_within_7_days():
    # low value 10 days ago, nothing since: falls back to the 'older' baseline, and says so
    r = _label([-240, 30], [30, 50])
    assert r["aki"] and "older" in r["ref_source"]


def test_peak_stage_not_onset_stage():
    r = _label([10, 40, 80, 120], [40, 70, 100, 130])
    assert r["stage_onset"] == 1 and r["stage_peak"] == 3


def test_population_baseline_uses_sex():
    # 14-year-old: male normal 62, female 53 -> 85 umol/L is AKI only for a female
    m = _label([30], [85], age_days=14 * 365.25, sex="M")
    f = _label([30], [85], age_days=14 * 365.25, sex="F")
    assert not m["aki"] and f["aki"] and f["ref_source"] == "population"


def test_neonate_first_48h_ignored_and_no_population_baseline():
    r = _label([10, 30], [60, 95], age_days=0.1)        # maternal creatinine period
    assert not r["aki"]
    r = _label([100], [200], age_days=5)                # single value, no reference
    assert not r["aki"]


def test_no_aki_stable():
    assert not _label([10, 30, 60, 90], [40, 42, 41, 43])["aki"]


# ── units / meds ─────────────────────────────────────────────────────────
def test_unit_conversion_and_unknown_units_dropped():
    df = pd.DataFrame({"concept": ["creatinine", "creatinine", "creatinine", "temp", "pco2"],
                       "value": [1.0, 88.0, 5.0, 98.6, 40.0],
                       "unit": ["mg/dL", "µmol/L", "furlongs", "Deg. F", "mm Hg"]})
    rep = []
    out = to_canonical(df, report=rep)
    cr = out[out.concept == "creatinine"]["value"].round(1).tolist()
    assert sorted(cr) == [88.0, 88.4]
    assert abs(out[out.concept == "temp"]["value"].iloc[0] - 37.0) < 0.01
    assert abs(out[out.concept == "pco2"]["value"].iloc[0] - 5.33) < 0.01
    assert any("furlongs" in r["issue"] for r in rep)


def test_med_matching_whole_word_and_routes():
    assert med_groups_for("Noradrenaline 4mg/50ml") == ["vasopressor"]
    assert "vasopressor" in med_groups_for("adrenaline infusion")
    assert med_groups_for("vancomycin oral") == ["vancomycin"]
    assert route_class("Oral") == "oral" and route_class("Nebulised") == "inhaled"
    assert route_class("IV") == "systemic"


# ── windows: no future information, labels correct ───────────────────────
def test_windows_no_leakage_and_labels():
    stays = pd.DataFrame({"stay_id": ["s1"], "subject_id": ["p1"], "admit": [T0],
                          "discharge": [T0 + pd.Timedelta(hours=96)], "aki": [True],
                          "onset_dt": [T0 + pd.Timedelta(hours=50)], "age_days": [2000.0],
                          "sex": ["F"], "stage_peak": [1]})
    labs = pd.DataFrame({"subject_id": "p1", "concept": "lactate", "unit": "mmol/l", "code": "x",
                         "time": [T0 + pd.Timedelta(hours=h) for h in (5.9, 6.0, 13)],
                         "value": [1.0, 9.0, 3.0]})
    empty = pd.DataFrame(columns=["subject_id", "time", "concept", "value"])
    meds = pd.DataFrame(columns=["subject_id", "start", "end", "name", "route"])
    cfg = {"windows": dict(step_h=6, horizon_h=24, max_windows=1000, hours_since_cap_h=168)}
    w, _ = build_windows(stays, labs, empty.astype({"value": float}), meds, cfg)
    assert w["window_end"].max() <= T0 + pd.Timedelta(hours=50)      # stops at onset
    assert len(w) == 8
    # value at exactly 6.0h belongs to window 1, not window 0
    assert w.loc[w.k == 0, "lab_lactate_last"].item() == 1.0
    assert w.loc[w.k == 1, "lab_lactate_last"].item() == 9.0
    # label: onset within [end, end+24h]
    exp = [(50 - 6 * (k + 1)) <= 24 for k in range(8)]
    assert w.sort_values("k")["label"].astype(bool).tolist() == exp


# ── ICU restriction ──────────────────────────────────────────────────────
def test_aki_before_admission_flagged():
    # creatinine rose on the ward (-30h -> -5h) and stays high in ICU
    r = _label([-30, -5, 10], [40, 80, 82])
    assert r["aki_before_admission"]
    r2 = _label([-30, 10, 40], [40, 41, 75])        # rise only inside the stay
    assert not r2["aki_before_admission"] and r2["aki"]


def test_ward_data_not_in_icu_windows():
    stays = pd.DataFrame({"stay_id": ["s1"], "subject_id": ["p1"], "admit": [T0],
                          "discharge": [T0 + pd.Timedelta(hours=48)], "aki": [False],
                          "onset_dt": [pd.NaT], "age_days": [2000.0], "sex": ["F"],
                          "stage_peak": [0]})
    labs = pd.DataFrame({"subject_id": "p1", "concept": ["lactate", "creatinine", "creatinine"],
                         "unit": "x", "code": "x", "value": [5.0, 40.0, 60.0],
                         "time": [T0 - pd.Timedelta(hours=3), T0 - pd.Timedelta(hours=20),
                                  T0 + pd.Timedelta(hours=8)]})
    meds = pd.DataFrame({"subject_id": "p1", "name": ["gentamicin", "noradrenaline"],
                         "route": "IV",
                         "start": [T0 - pd.Timedelta(hours=5), T0 - pd.Timedelta(hours=2)],
                         "end": [T0 - pd.Timedelta(hours=4), T0 + pd.Timedelta(hours=7)]})
    empty = pd.DataFrame(columns=["subject_id", "time", "concept", "value"]).astype({"value": float})
    cfg = {"windows": dict(step_h=6, horizon_h=24, max_windows=1000, hours_since_cap_h=168)}
    w, _ = build_windows(stays, labs, empty, meds, cfg)
    w = w.sort_values("k")
    assert "lab_lactate_last" not in w or w["lab_lactate_last"].isna().all()   # ward lactate unused
    assert np.isnan(w.loc[w.k == 0, "renal_creat_ratio_7d"].item())            # no ICU creat yet
    assert w.loc[w.k == 1, "renal_creat_ratio_7d"].item() == pytest.approx(1.5) # ward value = reference
    assert "med_aminoglycoside" not in w or w["med_aminoglycoside"].sum() == 0  # ward dose: no carry
    assert w.loc[w.k <= 1, "med_vasopressor"].tolist() == [1.0, 1.0]           # infusion running into ICU


# ── temporal hold-out ────────────────────────────────────────────────────
def test_holdout_split():
    from aki.split import make_split
    rng = np.random.default_rng(0)
    n = 400
    admit = pd.Timestamp("2018-01-01") + pd.to_timedelta(rng.uniform(0, 1500, n), unit="D")
    lab = pd.DataFrame({"stay_id": [f"s{i}" for i in range(n)],
                        "subject_id": [f"p{i // 2}" for i in range(n)],
                        "admit": admit, "discharge": admit + pd.Timedelta(days=10),
                        "aki": rng.random(n) < 0.2, "severe": False, "exclude": False})
    cfg = {"split": {"holdout": {"enabled": True, "months": 6, "exclude_straddling": True},
                     "method": "patient_random", "fractions": {"val": .15, "test": .15},
                     "seed": 1}}
    sp, meta = make_split(lab, cfg)
    cut = lab["admit"].max() - pd.DateOffset(months=6)
    m = lab.merge(sp, on=["stay_id", "subject_id"])
    assert (m.loc[m.split == "holdout", "admit"] >= cut).all()
    dev = m[m.split != "holdout"]
    assert (dev["discharge"] < cut).all()                      # nothing from hold-out period
    assert set(dev["split"]) == {"train", "val", "test"}
    assert meta["dropped_straddling"] == int(((lab.admit < cut) & (lab.discharge >= cut)).sum())


def test_route_inferred_from_drug_name():
    from aki.concepts import MED_GROUPS
    def counted(name):
        r = route_class(name)
        return [g for g in med_groups_for(name) if r not in MED_GROUPS[g].exclude_routes]
    assert counted("Vancomycin 125mg oral solution") == []
    assert counted("Vancomycin 500mg IV infusion") == ["vancomycin"]
    assert counted("Tobramycin 300mg/5ml nebuliser solution") == []
    assert counted("Gentamicin 0.3% eye drops") == []
    assert counted("Tacrolimus 0.1% ointment") == []
    assert counted("Tacrolimus 1mg capsules") == ["calcineurin_inhibitor"]
    assert counted("Methotrexate 12mg intrathecal") == []
    assert counted("Ibuprofen 100mg/5ml oral suspension") == ["nsaid"]


def test_censored_counts_and_report():
    from aki.concepts import norm_unit
    assert norm_unit("x10*9/l") == norm_unit("10^9/L") == "10^9/l"
    df = pd.DataFrame({"concept": ["crp", "crp", "crp", "haemoglobin"],
                       "code": ["CRP", "CRP", "CRP", "POCCTHB"],
                       "value": ["<5", "12", "haemolysed", "12.1"],
                       "unit": ["mg/L", "", "mg/L", ""]})
    rep = []
    out = to_canonical(df, {"crp": "mg/L", "haemoglobin": "g/L"}, rep)
    assert sorted(out["value"]) == [5.0, 12.0]          # censored kept at bound, text dropped
    r = pd.DataFrame(rep)
    assert {"non-numeric value"} <= set(r["issue"])
    hb = r[r.code == "POCCTHB"].iloc[0]
    assert hb["issue"].startswith("outside plausible") and hb["median"] == 12.1


def test_original_operating_point_rules():
    from aki.metrics import choose_threshold
    sw = pd.DataFrame({"threshold": [0.1, 0.2, 0.3, 0.4, 0.5],
                       "legacy_sens_ever": [0.95, 0.90, 0.80, 0.74, 0.60],
                       "legacy_spec_stay": [0.40, 0.55, 0.70, 0.85, 0.95]})
    sw["legacy_youden_stay"] = sw["legacy_sens_ever"] + sw["legacy_spec_stay"] - 1
    # Youden: 0.35, 0.45, 0.50, 0.59, 0.55 -> 0.4
    assert choose_threshold(sw, {"maximise": "legacy_youden_stay"}) == 0.4
    # max specificity with sensitivity >= 0.75 -> 0.3 (0.4 has sens 0.74)
    rule = {"maximise": "legacy_spec_stay",
            "subject_to": {"metric": "legacy_sens_ever", "min": 0.75}}
    assert choose_threshold(sw, rule) == 0.3


# ── revised labelling rules ──────────────────────────────────────────────
def test_no_population_baseline_needs_measured_reference():
    p = {**P, "allow_population_baseline": False}
    t, v = _lab([30], [85])          # single value, no reference -> cannot be AKI
    r = label_stay(t, v, T0, T0 + pd.Timedelta(hours=500), T0 - pd.Timedelta(days=5000), "F", p)
    assert not r["aki"]


def test_infant_older_baseline_window():
    p = {**P, "allow_population_baseline": False, "older_baseline_days_infant": 30}
    dob = T0 - pd.Timedelta(days=200)                     # infant
    t, v = _lab([-45 * 24, 30], [20, 40])                  # reference 45 days earlier
    r = label_stay(t, v, T0, T0 + pd.Timedelta(hours=500), dob, "M", p)
    assert not r["aki"]                                    # outside the 30-day infant window
    t, v = _lab([-20 * 24, 30], [20, 40])                  # 20 days earlier -> usable
    r = label_stay(t, v, T0, T0 + pd.Timedelta(hours=500), dob, "M", p)
    assert r["aki"] and r["ref_source"] == "older_<=30d"
    old = T0 - pd.Timedelta(days=3000)                     # older child: 90-day window
    t, v = _lab([-45 * 24, 30], [30, 60])
    r = label_stay(t, v, T0, T0 + pd.Timedelta(hours=500), old, "M", p)
    assert r["aki"] and r["ref_source"] == "older_<=90d"


def test_min_creatinine_evaluable_cohort():
    from aki.labelling import label_cohort
    stays = pd.DataFrame({
        "stay_id": ["a", "b", "c", "d"], "subject_id": ["a", "b", "c", "d"],
        "admit": [T0] * 4, "discharge": [T0 + pd.Timedelta(hours=200)] * 4,
        "dob": [T0 - pd.Timedelta(days=3000)] * 4, "sex": ["F"] * 4, "age_days": [3000.0] * 4})
    rows = [("a", 10, 40), ("a", 60, 42),          # 2 in stay -> evaluable
            ("b", 10, 40),                          # 1 in stay only -> excluded
            ("c", -48, 40), ("c", 30, 41),         # ward reference + 1 in stay -> evaluable
            ("d", -48, 40)]                         # none in stay -> excluded
    labs = pd.DataFrame([dict(subject_id=s, concept="creatinine", value=float(x),
                              time=T0 + pd.Timedelta(hours=h)) for s, h, x in rows])
    cfg = {"labelling": {**P, "allow_population_baseline": False, "min_creatinine_in_stay": 2}}
    lab = label_cohort(stays, labs, pd.DataFrame(columns=["subject_id", "time"]), cfg)
    ex = dict(zip(lab.stay_id, lab.exclude_reason))
    assert ex["a"] == "" and ex["c"] == ""
    assert ex["b"] == "fewer_than_2_creatinine" and ex["d"] == "fewer_than_2_creatinine"