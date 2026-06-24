import pandas as pd
import numpy as np
from pathlib import Path

# ── Constants (µmol/L) ────────────────────────────────────
ACUTE_RISE_UMOL            = 26.5
STAGE1_RATIO               = 1.5
STAGE2_RATIO               = 2.0
STAGE3_RATIO               = 3.0
STAGE3_ABS_UMOL            = 354.0
ACUTE_WINDOW_H             = 48
MIN_ONSET_HOURS_FROM_ADMIT = 24

POPULATION_NORMALS = {
    (0,   1,  "M"): 27,  (0,   1,  "F"): 27,
    (1,   3,  "M"): 27,  (1,   3,  "F"): 27,
    (3,   5,  "M"): 35,  (3,   5,  "F"): 35,
    (5,   10, "M"): 44,  (5,   10, "F"): 44,
    (10,  13, "M"): 53,  (10,  13, "F"): 53,
    (13,  16, "M"): 62,  (13,  16, "F"): 53,
    (16,  18, "M"): 74,  (16,  18, "F"): 60,
}

def get_population_normal(age_years: float, sex: str) -> float:
    sex = sex.upper()[0] if sex else "M"
    for (lo, hi, s), val in POPULATION_NORMALS.items():
        if s == sex and lo <= age_years < hi:
            return float(val)
    return 53.0


def get_baseline_creatinine(
    all_creat_times:  pd.Series,
    all_creat_values: pd.Series,
    admit_dt:         pd.Timestamp,
    age_years:        float,
    sex:              str,
    recent_days:      int = 7,
    extended_days:    int = 90,
) -> tuple[float, str]:
    """
    Priority cascade for baseline creatinine:
      1. Lowest value in 7 days before admission
      2. Lowest value in 7–90 days before admission
      3. First value in current admission
      4. Age/sex population normal
    """
    # Reset index to avoid misaligned .loc calls on filtered Series
    times  = all_creat_times.reset_index(drop=True)
    values = all_creat_values.reset_index(drop=True)

    mask_recent = (
        (times >= admit_dt - pd.Timedelta(days=recent_days)) &
        (times <  admit_dt) &
        (values.notna())
    )
    mask_extended = (
        (times >= admit_dt - pd.Timedelta(days=extended_days)) &
        (times <  admit_dt - pd.Timedelta(days=recent_days)) &
        (values.notna())
    )
    mask_current = (
        (times >= admit_dt) &
        (values.notna())
    )

    if mask_recent.any():
        val = values[mask_recent].min()
        return float(val), f"prior_7d (n={mask_recent.sum()})"

    if mask_extended.any():
        val = values[mask_extended].min()
        return float(val), f"prior_7_90d (n={mask_extended.sum()})"

    if mask_current.any():
        idx = times[mask_current].idxmin()
        val = values.loc[idx]
        return float(val), "first_in_admission"

    val = get_population_normal(age_years, sex)
    return float(val), f"population_normal (age={age_years:.1f}, sex={sex})"


def _get_stage(ratio: float, current_val: float, min_recent: float) -> int:
    if ratio >= STAGE3_RATIO:
        return 3
    if current_val >= STAGE3_ABS_UMOL and (current_val - min_recent) >= ACUTE_RISE_UMOL:
        return 3
    if ratio >= STAGE2_RATIO:
        return 2
    return 1


def detect_aki_onset(
    creatinine_df:        pd.DataFrame,
    precomputed_baseline: float,          
) -> dict:
    """
    Scan creatinine results for one admission and return the earliest
    timestamp where KDIGO criteria are met.

    creatinine_df columns: result_dt, value, admission_id
    precomputed_baseline : baseline creatinine in µmol/L from get_baseline_creatinine()
    """
    df = creatinine_df.sort_values("result_dt").reset_index(drop=True)

    if len(df) < 1:
        return {"aki_detected": False, "onset_dt": pd.NaT,
                "stage": 0, "trigger": "insufficient data"}

    for _, row in df.iterrows():
        t_current = row["result_dt"]
        v_current = row["value"]

        if pd.isna(v_current):
            continue

        # Prior values within 48h of this result (within this admission only)
        mask_48h = (
            (df["result_dt"] >= t_current - pd.Timedelta(hours=ACUTE_WINDOW_H)) &
            (df["result_dt"] <  t_current) &
            (df["value"].notna())
        )
        prior_48h       = df.loc[mask_48h, "value"]
        min_48h         = prior_48h.min() if len(prior_48h) > 0 else np.nan
        ratio_vs_base   = v_current / precomputed_baseline if precomputed_baseline > 0 else 1.0

        # ── Criterion 1: absolute rise ≥ 26.5 µmol/L within 48h ──
        if len(prior_48h) > 0 and (v_current - min_48h) >= ACUTE_RISE_UMOL:
            stage = _get_stage(ratio_vs_base, v_current, min_48h)
            return {
                "aki_detected": True,
                "onset_dt":     t_current,
                "stage":        stage,
                "trigger":      f"48h absolute rise {v_current - min_48h:.1f} µmol/L"
            }

        # ── Criterion 2: ≥ 1.5× baseline within 7 days ────────────
        if ratio_vs_base >= STAGE1_RATIO:
            # BUG FIX: guard against empty prior_48h before calling min()
            min_recent = min_48h if not np.isnan(min_48h) else v_current
            stage = _get_stage(ratio_vs_base, v_current, min_recent)
            return {
                "aki_detected": True,
                "onset_dt":     t_current,
                "stage":        stage,
                "trigger":      f"7d ratio {ratio_vs_base:.2f}x baseline ({precomputed_baseline:.1f} µmol/L)"
            }

    return {"aki_detected": False, "onset_dt": pd.NaT,
            "stage": 0, "trigger": "no criteria met"}


def build_aki_labels(
    admissions:   pd.DataFrame,
    labs:         pd.DataFrame,
    demographics: pd.DataFrame,
) -> pd.DataFrame:
    """
    admissions  : admission_id, subject_id, admit_dt, discharge_dt
    labs        : admission_id, component_base_name, value, result_dt
    demographics: subject_id, dob, sex
    """
    admissions = admissions.merge(
        demographics[["subject_id", "dob", "sex"]], on="subject_id", how="left"
    )

    # ── Creatinine identification ─────────────────────────
    # Keep only serum creatinine test names.
    # Exclude urine creatinine variants (prefix U = urine,
    # ratio units such as mg/mmol and umol/mmol Creatinine).
    SERUM_CREAT_NAMES = {
        'creat', 'ucreat1', 'ukcreat', 'ucreatenz',
    }
    URINE_CREAT_PREFIXES = (
        'unacreat', 'ualbucreat', 'ucreatmet', 'ohcreat',
        'ugaacreat', 'hex4creat', 'fcarncreat', 'pgaacreat',
        'tucreat', 'createx', 'creatsyn',
    )

    # Unit conversion factors to µmol/L
    # None = ratio/urine unit — exclude entirely
    UNIT_CONVERSION = {
        'umol/l':               1.0,
        'µmol/l':               1.0,
        'μmol/l':               1.0,
        'umol':                 1.0,
        'mmol/l':               1000.0,
        'mmol':                 1000.0,
        # Ratio units are urine creatinine — exclude
        'mg/mmol':              None,
        'umol/mmol creatinine': None,
        'umol/mmol':            None,
        'mmol/mmol creatinine': None,
    }

    name_col = labs["component_base_name"].str.casefold().str.strip()
    print(labs.columns)
    unit_col  = labs["Unit"].str.casefold().str.strip() \
                if "Unit" in labs.columns \
                else pd.Series("umol/l", index=labs.index)

    # Step 1: keep only serum creatinine test names
    serum_mask = name_col.isin(SERUM_CREAT_NAMES)
    urine_mask = name_col.str.startswith(URINE_CREAT_PREFIXES, na=False)
    creat = labs[serum_mask & ~urine_mask].copy()

    n_raw = len(creat)

    # Step 2: convert units to µmol/L, drop ratio/urine units
    creat["value"] = pd.to_numeric(creat["value"], errors="coerce")

    if "Unit" in creat.columns:
        creat["_unit_lower"] = creat["Unit"].str.casefold().str.strip()
        print(creat['Unit'].value_counts())
        creat["_conversion"] = creat["_unit_lower"].map(UNIT_CONVERSION)

        # Rows with no unit recorded — assume µmol/L (most common at GOSH)
        creat["_conversion"] = creat["_conversion"].fillna(1.0)

        # Drop ratio units (conversion mapped to None → stays NaN after fillna
        # only if we don't fillna them — handle separately)
        creat["_conversion"] = creat["Unit"].str.casefold().str.strip().map(
            lambda u: UNIT_CONVERSION.get(u, 1.0)
            if UNIT_CONVERSION.get(u, 1.0) is not None
            else np.nan
        )

        n_ratio = creat["_conversion"].isna().sum()
        if n_ratio > 0:
            print(f"  Dropped {n_ratio} creatinine rows with ratio/urine units "
                  f"(mg/mmol or umol/mmol Creatinine)")
        creat = creat[creat["_conversion"].notna()].copy()

        creat["value"] = creat["value"] * creat["_conversion"]
        creat = creat.drop(columns=["_unit_lower", "_conversion"])
    else:
        print("  WARNING: no 'unit' column found in labs — "
              "assuming all creatinine values are in µmol/L")

    # Step 3: plausibility filter — paediatric serum creatinine range
    CREAT_MIN_UMOL = 10.0
    CREAT_MAX_UMOL = 800.0
    n_before = len(creat)
    creat = creat[
        creat["value"].between(CREAT_MIN_UMOL, CREAT_MAX_UMOL, inclusive="both")
    ].copy()
    n_dropped = n_before - len(creat)

    creat = creat.dropna(subset=["value"])

    print(f"\nCreatinine extraction summary:")
    print(f"  Raw rows (serum names only):   {n_raw:,}")
    print(f"  After unit conversion:         {n_before:,}")
    print(f"  After plausibility filter:     {len(creat):,} "
          f"(dropped {n_dropped} outside {CREAT_MIN_UMOL}–{CREAT_MAX_UMOL} µmol/L)")
    if "Unit" in labs.columns:
        print(f"  Unit distribution after filter:")
        print(creat["Unit"].value_counts().to_string())

    adm_to_subj = admissions.set_index("admission_id")["subject_id"].to_dict()
    creat["subject_id"] = creat["admission_id"].map(adm_to_subj)

    results = []
    for _, adm in admissions.iterrows():
        age_years = (adm["admit_dt"] - adm["dob"]).days / 365.25
        sex       = "M" if str(adm.get("sex", "")).strip().lower() == 'Male' else 'F'

        all_creat = creat[
            creat["subject_id"] == adm["subject_id"]
        ].sort_values("result_dt")

        baseline_val, baseline_src = get_baseline_creatinine(
            all_creat_times  = all_creat["result_dt"],
            all_creat_values = all_creat["value"],
            admit_dt         = adm["admit_dt"],
            age_years        = age_years,
            sex              = sex,
        )

        adm_creat = all_creat[
            (all_creat["result_dt"] >= adm["admit_dt"]) &
            (all_creat["result_dt"] <= adm["discharge_dt"])
        ]

        # BUG FIX: pass precomputed_baseline — detect_aki_onset no longer
        # calls get_baseline_creatinine internally
        result = detect_aki_onset(adm_creat, precomputed_baseline=baseline_val)

        onset_hours = (
            (result["onset_dt"] - adm["admit_dt"]).total_seconds() / 3600
            if pd.notna(result["onset_dt"]) else np.nan
        )
        results.append({
            "admission_id":           adm["admission_id"],
            "aki_detected":           result["aki_detected"],
            "onset_dt":               result["onset_dt"],
            "aki_stage":              result["stage"],
            "aki_trigger":            result["trigger"],
            "baseline_creatinine":    baseline_val,
            "baseline_source":        baseline_src,
            "onset_hours_from_admit": onset_hours,
            "exclude_early_onset": (
                result["aki_detected"] and
                not np.isnan(onset_hours) and
                onset_hours < MIN_ONSET_HOURS_FROM_ADMIT
            ),
        })

    labels_df = pd.DataFrame(results)

    print("\nBaseline source distribution:")
    print(labels_df["baseline_source"].str.split(" ").str[0].value_counts())
    print(f"\nAKI detected: {labels_df['aki_detected'].sum()} / {len(labels_df)}")
    print(f"Excluded (early onset): {labels_df['exclude_early_onset'].sum()}")

    return admissions.merge(labels_df, on="admission_id", how="left")

def main():
    # ── 1. EDIT THESE PATHS ───────────────────────────────
    raw_dir  = Path("/")       # folder containing your raw files (training and test)
    
    proc_dir = Path("data/reprocessed") # where outputs are saved for train and test
        
    admissions_file  = raw_dir / "hospital_admissions.csv"   # admission file for train test
    demographics_file= raw_dir / "demographics.csv" # raw denographics file for train test
    
    labs_file = raw_dir / "lab_components.csv" # raw lab results file for train test
    
    # ── 2. EDIT THESE COLUMN NAME MAPPINGS ───────────────
    # Left side  = what the code expects
    # Right side = what your actual column is called
    # If your column is already named correctly, you can delete that line
    print('loading admissions')
    admissions = pd.read_csv(admissions_file, parse_dates=["start_datetime", "end_datetime"])

    admissions = admissions.assign(admission_id = np.nan)
    pr_counts = admissions.project_id.value_counts()
    for pr in admissions.project_id.unique():
        admissions.loc[admissions['project_id'] == pr,'admission_id'] = list(range(pr_counts.loc[pr]))
        
    admissions['admission_id'] = admissions['admission_id'].astype(int) + 1
    admissions['admission_id'] = admissions['project_id'].astype(str) +'_'+ admissions['admission_id'].astype(str)

    admissions = admissions.rename(columns={
        "start_datetime":  "admit_dt",     
        "end_datetime":  "discharge_dt",
        "project_id":      "subject_id",
    })
    print('loading demographics')
    demographics = pd.read_csv(demographics_file, parse_dates=["birth_date"])
    demographics = demographics.rename(columns={
        "project_id":    "subject_id",
        "birth_date": "dob",
        "sex_name":        "sex",
    })

    print('loading labs')
    LABS_COLS_NEEDED = ['project_id', 'component_basename', 'end_datetime', 'NumericValue', 'flag', 'Unit']
    labs = pd.read_csv(labs_file, usecols = LABS_COLS_NEEDED, parse_dates=["end_datetime"])
    
    labs = labs.rename(columns={
        "project_id":       "subject_id",   
        "component_basename":        "component_base_name",
        "NumericValue":     "value",
        "end_datetime": "result_dt"
    })

    labs['flag'] = labs['flag'].fillna('Normal')
    labs['alert_flag'] = labs['flag'].where(labs['flag'].isin(['Abnormal', 'High', 'Low']), other = np.nan)

    labs = labs.merge(admissions[['subject_id', 'admission_id', 'admit_dt', 'discharge_dt']], on = 'subject_id',  
    how='inner')
    labs = labs[(labs['result_dt'] >= labs['admit_dt']) & (labs['result_dt'] <= labs['discharge_dt'])]
    labs = labs.drop(columns = ['admit_dt', 'discharge_dt'])

    print("admissions", admissions.shape)
    print(admissions[['admission_id', 'subject_id', 'admit_dt', 'discharge_dt']].head(3))
    print("labs", labs.shape)
    print(labs[['admission_id', 'subject_id', 'component_base_name', 'value']].head(3))
    print(labs.component_base_name.value_counts().head(20))

    print('------running labelling-------')
    cohort = build_aki_labels(admissions, labs, demographics)
    output_path = proc_dir / "cohort.parquet"
    cohort.to_parquet(output_path, index = False)
    print(f"saved to {output_path}")
    print("total admissions", len(cohort))
    print("AKI detected:", cohort['aki_detected'].sum())

if __name__ == "__main__":
    main()

    

