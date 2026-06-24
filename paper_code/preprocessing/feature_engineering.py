"""
src/feature_engineering.py

Builds a flat feature matrix from the processed cohort + raw modality tables.

Inputs:
    data/processed/cohort.parquet      – from aki_labelling.py
                                         must contain: admission_id, subject_id,
                                         admit_dt, discharge_dt, age_years, sex,
                                         aki_detected, onset_dt, aki_stage,
                                         exclude_early_onset, baseline_creatinine,
                                         baseline_source
    data/raw/lab_components.csv        – component_name, end_datetime, project_id,
                                         NumericValue, flag
    data/raw/medication_admins.csv     – drug_name, drug_gpi, end_datetime,
                                         project_id
    data/raw/procedures.csv            – proc_local_code, start_datetime, project_id
    data/raw/diagnoses.csv             – diag_local_code, start_datetime, project_id
    data/raw/demographics.csv          – project_id, ethnicity_name

Output:
    data/processed/features.parquet   – one row per admission, all features merged

Usage:
    python src/feature_engineering.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ── Optional fuzzy matching (pip install rapidfuzz) ───────
try:
    from rapidfuzz import fuzz, process as rf_process
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    warnings.warn(
        "rapidfuzz not installed – falling back to exact+substring matching. "
        "Run: pip install rapidfuzz",
        stacklevel=2,
    )

# ── Local config imports ──────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab_config import (
    LAB_SYNONYMS, EXCLUDE_BASENAMES,
    build_reverse_map, validate_units,
)

# Build the reverse lookup and canonical list from lab_config
_REVERSE_MAP  = build_reverse_map(LAB_SYNONYMS)
RAW_TO_CANONICAL = {k.lower(): v for k, v in _REVERSE_MAP.items()}
LAB_SYNONYM_MAP  = LAB_SYNONYMS   # kept for backwards compatibility
from clinical_codes import (
    NEPHROTOXINS, DRUG_CLASSES,
    GPI_NEPHROTOXINS, GPI_DRUG_CLASSES,
    PROCEDURE_GROUPS,
)

# ── Canonical lab list (drives feature columns) ───────────
CANONICAL_LABS = list(LAB_SYNONYM_MAP.keys())

# ── Suffixes to strip before lab name matching ────────────
LAB_NAME_SUFFIXES_TO_STRIP = [
    " poc", " (poc)", " blood gas", " (blood gas)",
    " arterial", " venous", " capillary",
    " whole blood", " plasma", " serum",
    " (serum)", " (plasma)", " (blood)",
    " blood", " random", " fasting",
]

# ── Ethnicity grouping (GOSH categories) ─────────────────
ETHNICITY_GROUPS: dict[str, list[str]] = {
    "white": [
        "white british",
        "other white background",
        "white irish",
    ],
    "black": [
        "black african",
        "black caribbean",
        "other black background",
    ],
    "asian": [
        "asian pakistani",
        "asian indian",
        "other asian background",
        "asian bangladeshi",
        "other chinese",
        "chinese",
    ],
    "mixed": [
        "mixed white and asian",
        "mixed white and black caribbean",
        "mixed white and black african",
        "other mixed background",
    ],
    "other": [
        "other ethnic backgrounds",
        "other ethnic background",
        "prefer not to say",
        "n/a",
    ],
    "not_available": [
        "not available",
        "unknown",
        "not stated",
        "",
    ],
}


# ─────────────────────────────────────────────────────────
#  1.  Lab name normalisation
# ─────────────────────────────────────────────────────────

def normalise_lab_name(raw_name: str, fuzzy_threshold: int = 80) -> str | None:
    """
    Map a raw lab string to its canonical form.

    Strategy
    --------
    1. Exact lookup on original clean string.
    2. Strip known suffixes (POC, blood, serum etc.) and try exact again.
    3. Substring match on stripped name.
    4. Fuzzy match via rapidfuzz if installed.
    5. Return None if no match — row will be dropped.
    """
    if not isinstance(raw_name, str) or not raw_name.strip():
        return None

    clean = raw_name.lower().strip()

    # 1. Exact match
    if clean in RAW_TO_CANONICAL:
        return RAW_TO_CANONICAL[clean]

    # 2. Strip suffix and try exact match again
    stripped = clean
    for suffix in LAB_NAME_SUFFIXES_TO_STRIP:
        if stripped.endswith(suffix):
            stripped = stripped[: -len(suffix)].strip()
            break

    if stripped != clean and stripped in RAW_TO_CANONICAL:
        return RAW_TO_CANONICAL[stripped]

    # 3. Substring match on stripped name
    for known_raw, canonical in RAW_TO_CANONICAL.items():
        if known_raw in stripped:
            return canonical

    # 4. Fuzzy match
    if RAPIDFUZZ_AVAILABLE:
        match, score, _ = rf_process.extractOne(
            stripped,
            RAW_TO_CANONICAL.keys(),
            scorer=fuzz.token_sort_ratio,
        )
        if score >= fuzzy_threshold:
            return RAW_TO_CANONICAL[match]

    return None

def fix_haematocrit(labs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Haematocrit is reported in decimal (e.g. 0.35) or percentage (35%).
    Convert all decimal values to percentage using a value threshold.
    Threshold 1.5 is safe — no physiological Hct exceeds 75% (0.75 decimal).
    """
    hct_mask    = labs_df['canonical_name'] == 'haematocrit_poc'
    decimal_mask = hct_mask & (labs_df['value'] < 1.5)
    n_decimal    = decimal_mask.sum()
    if n_decimal > 0:
        labs_df.loc[decimal_mask, 'value'] *= 100
        print(f"  Haematocrit: converted {n_decimal:,} decimal values "
              f"to percentage (×100)")
    return labs_df

def fix_creatinine_units(labs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise all serum creatinine values to µmol/L.
    Converts mmol/L → µmol/L (×1000).
    Drops ratio/urine units (mmol/mmol etc.).
    Flags values outside plausible paediatric range (10–800 µmol/L).
    """
    creat_mask = labs_df['canonical_name'] == 'creatinine'
    if 'unit' not in labs_df.columns:
        print("  WARNING: no unit column — cannot fix creatinine units")
        return labs_df

    unit = labs_df.loc[creat_mask, 'unit'].str.strip().str.lower()

    # Convert mmol/L → µmol/L
    mmol_mask = creat_mask & unit.isin(['mmol/l', 'mmol'])
    n_mmol    = mmol_mask.sum()
    if n_mmol > 0:
        labs_df.loc[mmol_mask, 'value'] *= 1000
        print(f"  Creatinine: converted {n_mmol:,} mmol/L → µmol/L (×1000)")

    # Drop ratio/urine units
    ratio_units = {
        'mmol/mmol', 'mmol/mmol creatinine',
        'umol/mmol', 'umol/mmol creatinine',
        'mg/mmol',
    }
    ratio_mask = creat_mask & unit.isin(ratio_units)
    n_ratio    = ratio_mask.sum()
    if n_ratio > 0:
        labs_df = labs_df[~ratio_mask].copy()
        print(f"  Creatinine: dropped {n_ratio:,} ratio unit rows "
              f"(urine creatinine excluded)")

    # Plausibility filter after conversion
    creat_mask2 = labs_df['canonical_name'] == 'creatinine'
    out_of_range = creat_mask2 & ~labs_df['value'].between(10, 800)
    n_oor = out_of_range.sum()
    if n_oor > 0:
        print(f"  Creatinine: {n_oor:,} values outside 10–800 µmol/L "
              f"after conversion — check unit mapping")
        print(labs_df[out_of_range][
            ['component_base_name', 'value', 'unit']
        ].head(10).to_string())

    return labs_df


def normalise_labs(labs: pd.DataFrame) -> pd.DataFrame:
    """
    Add a `canonical_name` column and drop unmapped rows.
    Prints audit of top unmapped names.

    Expected columns: component_base_name, value, result_dt,
                      admission_id, alert_flag
    """
    # Guard against wrong schema arriving
    required = {"component_base_name", "value", "result_dt", "admission_id", "unit"}
    missing  = required - set(labs.columns)
    if missing:
        raise ValueError(
            f"normalise_labs: missing columns {missing}\n"
            f"Got: {labs.columns.tolist()}"
        )

    labs = labs.copy()

    # ── Drop excluded basenames (urine tests, hepatitis screens etc) ──
    if "component_base_name" in labs.columns:
        n_before = len(labs)
        labs = labs[
            ~labs["component_base_name"].str.upper().str.strip()
            .isin(EXCLUDE_BASENAMES)
        ].copy()
        n_dropped = n_before - len(labs)
        if n_dropped > 0:
            print(f"  Dropped {n_dropped:,} rows with excluded basenames "
                  f"(urine tests, hepatitis screens etc.)")

    labs["value"] = pd.to_numeric(labs["value"], errors="coerce")
    labs["canonical_name"] = labs["component_base_name"].map(
        lambda x: normalise_lab_name(str(x)) if pd.notna(x) else None
    )

    unmapped = (
        labs[labs["canonical_name"].isna()]["component_base_name"]
        .value_counts()
    )
    if len(unmapped):
        print(
            f"\n[normalise_labs] {len(unmapped)} unmapped lab name(s) "
            f"({labs['canonical_name'].isna().sum()} rows dropped)."
        )
        print("Top unmapped (add to lab_config.py if needed):")
        print(unmapped.head(20).to_string())

    labs = labs[labs["canonical_name"].notna()].copy()

    # ── Fix unit issues before returning ─────────────────────
    print("  Fixing creatinine units...")
    labs = fix_creatinine_units(labs)
    print("  Fixing haematocrit decimal/percentage...")
    labs = fix_haematocrit(labs)

    return labs


# ─────────────────────────────────────────────────────────
#  2.  Per-admission feature extractors
# ─────────────────────────────────────────────────────────

def extract_lab_features(lab_window: pd.DataFrame) -> dict:
    """
    Summarise labs in a single lookback window.

    Per analyte: _mean, _min, _max, _last, _trend, _n, _missing, _alert
    """
    feats: dict = {}

    # Guard — return all-missing if empty or wrong schema
    if lab_window.empty or "canonical_name" not in lab_window.columns:
        for analyte in CANONICAL_LABS:
            feats[f"{analyte}_mean"]    = np.nan
            feats[f"{analyte}_min"]     = np.nan
            feats[f"{analyte}_max"]     = np.nan
            feats[f"{analyte}_last"]    = np.nan
            feats[f"{analyte}_trend"]   = np.nan
            feats[f"{analyte}_n"]       = 0
            feats[f"{analyte}_missing"] = 1
            feats[f"{analyte}_alert"]   = 0
        return feats

    for analyte in CANONICAL_LABS:
        sub  = lab_window[lab_window["canonical_name"] == analyte]
        vals = sub["value"].dropna()

        if len(vals) == 0:
            feats[f"{analyte}_mean"]    = np.nan
            feats[f"{analyte}_min"]     = np.nan
            feats[f"{analyte}_max"]     = np.nan
            feats[f"{analyte}_last"]    = np.nan
            feats[f"{analyte}_trend"]   = np.nan
            feats[f"{analyte}_n"]       = 0
            feats[f"{analyte}_missing"] = 1
            feats[f"{analyte}_alert"]   = 0
        else:
            feats[f"{analyte}_mean"]    = float(vals.mean())
            feats[f"{analyte}_min"]     = float(vals.min())
            feats[f"{analyte}_max"]     = float(vals.max())
            feats[f"{analyte}_last"]    = float(vals.iloc[-1])
            feats[f"{analyte}_trend"]   = (
                float(vals.iloc[-1] - vals.iloc[0]) if len(vals) > 1 else 0.0
            )
            feats[f"{analyte}_n"]       = len(vals)
            feats[f"{analyte}_missing"] = 0
            feats[f"{analyte}_alert"]   = int(sub["alert_flag"].notna().any())

    return feats


def extract_medication_features(med_window: pd.DataFrame) -> dict:
    """
    Summarise medications in a single lookback window.

    Uses GPI codes where available (more reliable than name matching),
    falls back to name string matching.

    Features:
        nephrotoxin_<drug>         – binary flag per tier-1 nephrotoxin
        nephrotoxin_<drug>_count   – administration count
        nephrotoxin_any            – 1 if ANY nephrotoxin given
        nephrotoxin_total_count    – total nephrotoxin administrations
        drug_class_<class>         – binary flag per tier-2 class
        n_unique_drugs             – regimen complexity proxy
    """
    feats: dict = {}

    if med_window.empty:
        text      = pd.Series([], dtype=str)
        gpi_codes = pd.Series([], dtype=str)
    else:
        text      = med_window["drug_description"].str.lower().fillna("")
        gpi_codes = (
            med_window["drug_gpi"].astype(str).str.strip()
            if "drug_gpi" in med_window.columns
            else pd.Series([""] * len(med_window), dtype=str)
        )

    has_gpi = gpi_codes.str.len().gt(0).any()

    # ── Tier 1: individual nephrotoxins ───────────────────
    for drug in NEPHROTOXINS:
        if has_gpi and drug in GPI_NEPHROTOXINS:
            # GPI match — preferred
            prefixes = GPI_NEPHROTOXINS[drug]
            mask = gpi_codes.apply(
                lambda g: isinstance(g, str) and any(g.startswith(p) for p in prefixes))
        else:
            # Name match — fallback
            pattern = "|".join(NEPHROTOXINS[drug])
            mask = text.str.contains(pattern, na=False, regex=True)

        feats[f"nephrotoxin_{drug}"]       = int(mask.any())
        feats[f"nephrotoxin_{drug}_count"] = int(mask.sum())

    feats["nephrotoxin_any"] = int(
        any(feats[f"nephrotoxin_{d}"] for d in NEPHROTOXINS)
    )
    feats["nephrotoxin_total_count"] = sum(
        feats[f"nephrotoxin_{d}_count"] for d in NEPHROTOXINS
    )

    # ── Tier 2: drug classes ──────────────────────────────
    for cls in DRUG_CLASSES:
        if has_gpi and cls in GPI_DRUG_CLASSES:
            prefixes = GPI_DRUG_CLASSES[cls]
            mask = gpi_codes.apply(
                lambda g: isinstance(g, str) and any(g.startswith(p) for p in prefixes))
            
        else:
            pattern = "|".join(DRUG_CLASSES[cls])
            mask = text.str.contains(pattern, na=False, regex=True)

        feats[f"drug_class_{cls}"] = int(mask.any())

    feats["n_unique_drugs"] = (
        med_window["drug_description"].nunique()
        if not med_window.empty else 0
    )
    return feats


def extract_procedure_features(proc_window: pd.DataFrame) -> dict:
    """
    Summarise procedures in a single lookback window.
    Features: proc_<group> binary flags, n_procedures count.
    """
    feats: dict = {}

    codes = (
        proc_window["opcs_code"].str.upper().fillna("")
        if not proc_window.empty
        else pd.Series([], dtype=str)
    )

    for group, prefixes in PROCEDURE_GROUPS.items():
        feats[f"proc_{group}"] = int(
            codes.str.startswith(tuple(prefixes)).any()
        )

    feats["n_procedures"] = len(proc_window)
    return feats


def extract_static_features(adm: pd.Series) -> dict:
    """
    Extract time-invariant features from a single admission row.

    Expects on the row:
        age_years   – pre-computed in main() from dob + admit_dt
        sex         – "Male" / "Female"
        ethnicity   – free text, mapped to GOSH groups
    """
    age = adm.get("age_years", np.nan)

    # Handle NaN age safely
    def age_flag(lo, hi):
        if pd.isna(age):
            return 0
        return int(lo <= age < hi)

    feats: dict = {
        "age_years":      float(age) if not pd.isna(age) else np.nan,
        "age_neonate":    int(age < 0.083) if not pd.isna(age) else 0,
        "age_infant":     age_flag(0.083, 1),
        "age_toddler":    age_flag(1, 3),
        "age_child":      age_flag(3, 12),
        "age_adolescent": age_flag(12, 18),
        # BUG FIX: was comparing to "Male" (capital) after .lower() — always False
        "sex_male":       int(str(adm.get("sex", "")).strip().lower() == "male"),
    }

    eth_raw     = str(adm.get("ethnicity", "")).lower().strip()
    eth_encoded = "not_available"
    for group, terms in ETHNICITY_GROUPS.items():
        if any(t in eth_raw for t in terms):
            eth_encoded = group
            break
    for group in ETHNICITY_GROUPS:
        feats[f"ethnicity_{group}"] = int(eth_encoded == group)

    return feats


# ─────────────────────────────────────────────────────────
#  3.  Main feature matrix builder
# ─────────────────────────────────────────────────────────

def build_feature_matrix(
    cohort:       pd.DataFrame,
    labs:         pd.DataFrame,
    medications:  pd.DataFrame,
    procedures:   pd.DataFrame,
    demographics: pd.DataFrame,
    diagnoses:    pd.DataFrame,
) -> pd.DataFrame:
    """
    Receives clean, already-linked tables from main().
    All renaming and project_id → admission_id linking done before this call.

    cohort       : admission_id, subject_id, admit_dt, discharge_dt,
                   age_years, sex, onset_dt, aki_detected, aki_stage,
                   exclude_early_onset, baseline_creatinine, baseline_source
    labs         : admission_id, component_base_name, result_dt, value,
                   alert_flag  (canonical_name added here)
    medications  : admission_id, drug_description, drug_gpi, given_dt
    procedures   : admission_id, opcs_code, proc_date
    demographics : subject_id, ethnicity  (age + sex already on cohort)
    diagnoses    : admission_id, icd10_code, diag_date
    """

    # ── Merge ethnicity onto cohort ───────────────────────
    # age_years and sex are already on cohort from aki_labelling step
    # Only ethnicity comes from demographics here
    cohort = cohort.merge(
        demographics[["subject_id", "ethnicity"]],
        on="subject_id", how="left",
    )
    missing_eth = cohort["ethnicity"].isna().sum()
    if missing_eth > 0:
        print(f"  WARNING: {missing_eth} admissions missing ethnicity "
              f"— will encode as 'not_available'")
        cohort["ethnicity"] = cohort["ethnicity"].fillna("not_available")

    # ── Encode diagnoses at ICD chapter level ─────────────
    # Single letter → ~20 groups, avoids hundreds of sparse columns
    # e.g. "N18.3" → "N" (genitourinary), "J96" → "J" (respiratory)
    diagnoses = diagnoses.copy()
    diagnoses["icd_chapter"] = diagnoses["icd10_code"].str[0].str.upper()

    diag_dummies = (
        pd.get_dummies(
            diagnoses[["admission_id", "icd_chapter"]].drop_duplicates(),
            columns=["icd_chapter"],
            prefix="diag",
        )
        .groupby("admission_id").max()
        .reset_index()
    )
    cohort    = cohort.merge(diag_dummies, on="admission_id", how="left")
    diag_cols = [c for c in cohort.columns if c.startswith("diag_")]
    cohort[diag_cols] = cohort[diag_cols].fillna(0).astype(int)
    print(f"Diagnosis chapters encoded: "
          f"{[c.replace('diag_', '') for c in diag_cols]}")

    # ── Normalise lab names ────────────────────────────────
    print("Normalising lab names...")
    labs_norm = normalise_labs(labs)

    # ── Index all tables by admission_id for O(1) lookup ──
    labs_by_adm  = {aid: grp for aid, grp in labs_norm.groupby("admission_id")}
    meds_by_adm  = {aid: grp for aid, grp in medications.groupby("admission_id")}
    procs_by_adm = {aid: grp for aid, grp in procedures.groupby("admission_id")}

    # BUG FIX: empty fallback DataFrames must have correct columns so that
    # filtered slices always carry canonical_name and other expected columns
    EMPTY_LABS  = pd.DataFrame(columns=labs_norm.columns)
    EMPTY_MEDS  = pd.DataFrame(columns=medications.columns)
    EMPTY_PROCS = pd.DataFrame(columns=procedures.columns)

    records = []

    for _, adm in cohort.iterrows():
        adm_id   = adm["admission_id"]
        admit_dt = adm["admit_dt"]
        dis_dt   = adm["discharge_dt"]

        # Window ends at onset for AKI cases, discharge for controls
        if pd.notna(adm.get("onset_dt")) and adm.get("aki_detected", False):
            window_end = adm["onset_dt"]
        else:
            window_end = dis_dt

        # ── Labs in window ────────────────────────────────
        adm_labs = labs_by_adm.get(adm_id, EMPTY_LABS)
        if len(adm_labs) > 0:
            lab_win = adm_labs[
                (adm_labs["result_dt"] >= admit_dt) &
                (adm_labs["result_dt"] <  window_end)
            ]
        else:
            lab_win = EMPTY_LABS   # BUG FIX: was pd.DataFrame() — no columns

        # ── Medications in window ─────────────────────────
        adm_meds = meds_by_adm.get(adm_id, EMPTY_MEDS)
        if len(adm_meds) > 0:
            med_win = adm_meds[
                (adm_meds["given_dt"] >= admit_dt) &
                (adm_meds["given_dt"] <  window_end)
            ]
        else:
            med_win = EMPTY_MEDS

        # ── Procedures in window ──────────────────────────
        adm_procs = procs_by_adm.get(adm_id, EMPTY_PROCS)
        if len(adm_procs) > 0:
            proc_win = adm_procs[
                (adm_procs["proc_date"] >= admit_dt) &
                (adm_procs["proc_date"] <  window_end)
            ]
        else:
            proc_win = EMPTY_PROCS

        # ── Assemble row ──────────────────────────────────
        row: dict = {
            "admission_id":           adm_id,
            "subject_id":             adm["subject_id"],
            "admit_dt":               admit_dt,
            "discharge_dt":           dis_dt,
            "window_end":             window_end,
            "label":                  int(adm.get("aki_detected", 0)),
            "aki_stage":              int(adm.get("aki_stage", 0)),
            "onset_dt":               adm.get("onset_dt", pd.NaT),
            "onset_hours_from_admit": adm.get("onset_hours_from_admit", np.nan),
            "exclude_early_onset":    bool(adm.get("exclude_early_onset", False)),
            "baseline_creatinine":    adm.get("baseline_creatinine", np.nan),
            "baseline_source":        adm.get("baseline_source", "unknown"),
        }

        # Diagnosis flags (merged onto cohort above, picked up per row)
        for col in diag_cols:
            row[col] = int(adm.get(col, 0))

        row.update(extract_static_features(adm))
        row.update(extract_lab_features(lab_win))
        row.update(extract_medication_features(med_win))
        row.update(extract_procedure_features(proc_win))

        records.append(row)

    features = pd.DataFrame(records)

    # ── Drop early-onset exclusions ───────────────────────
    n_before = len(features)
    features = features[~features["exclude_early_onset"]].copy()
    print(f"\nDropped {n_before - len(features)} early-onset admissions.")
    print(f"Final feature matrix: {len(features)} admissions, "
          f"{features.shape[1]} columns.")
    print(f"AKI prevalence: {features['label'].mean():.3f} "
          f"({features['label'].sum()} / {len(features)})")

    # ── Missingness audit ─────────────────────────────────
    missing_pct  = (features.isna().mean() * 100).sort_values(ascending=False)
    high_missing = missing_pct[missing_pct > 50]
    if len(high_missing):
        print(f"\nWARNING – {len(high_missing)} features >50% missing:")
        print(high_missing.to_string())

    return features


# ─────────────────────────────────────────────────────────
#  4.  Entry point
# ─────────────────────────────────────────────────────────

def main():
    raw_dir  = Path("/home/workspace/files/PICU_Data_Upload_20241108/")
    proc_dir = Path("data/processed")
    proc_dir.mkdir(parents=True, exist_ok=True)

    # ── Load cohort ───────────────────────────────────────
    # cohort already has: age_years, sex, admission_id, subject_id,
    # admit_dt, discharge_dt, aki_detected, onset_dt, aki_stage,
    # exclude_early_onset, baseline_creatinine, baseline_source
    print("Loading cohort...")
    cohort = pd.read_parquet(proc_dir / "cohort_with_age.parquet")
    cohort = cohort[~cohort["exclude_early_onset"]].copy()
    print(f"Cohort: {len(cohort)} admissions "
          f"({cohort['aki_detected'].sum()} AKI)")
    print(f"  Age range: {cohort['age_years'].min():.1f} – "
          f"{cohort['age_years'].max():.1f} years")
    print(f"  Sex distribution:\n{cohort['sex'].value_counts().to_string()}")

    # ── Load demographics (ethnicity only) ────────────────
    print("\nLoading demographics...")
    demographics = pd.read_csv(raw_dir / "demographics.csv")
    demographics = demographics.rename(columns={
        "project_id":     "subject_id",
        "ethnicity_name": "ethnicity",
    })
    # Only keep what we need
    demographics = demographics[["subject_id", "ethnicity"]].drop_duplicates(
        subset="subject_id"
    )
    print(f"  Ethnicity distribution:\n"
          f"{demographics['ethnicity'].value_counts().head(10).to_string()}")

    # ── Load labs (chunked) ───────────────────────────────
    print("\nLoading labs...")
    LAB_COLS = ["project_id", "component_basename", "end_datetime",
                "NumericValue", "flag", "unit"]
    chunks = []
    for chunk in pd.read_csv(
        raw_dir / "lab_components.csv",
        usecols=LAB_COLS,
        dtype={"project_id": "str", "component_name": "str",
               "NumericValue": "float32", "flag": "str"},
        chunksize=200_000,
    ):
        chunks.append(chunk)
    labs = pd.concat(chunks, ignore_index=True)
    print(f"  Labs loaded: {len(labs)} rows")

    labs["end_datetime"] = pd.to_datetime(labs["end_datetime"])
    labs = labs.rename(columns={
        "component_basename": "component_base_name",
        "end_datetime":   "result_dt",
        "project_id":     "subject_id",
        "NumericValue":   "value",
    })
    labs["flag"]       = labs["flag"].fillna("Normal")
    labs["alert_flag"] = labs["flag"].where(
        labs["flag"].isin(["Abnormal", "High", "Low"]), other=np.nan
    )
    labs = labs.merge(
        cohort[["admission_id", "subject_id", "admit_dt", "discharge_dt"]],
        on="subject_id", how="inner",
    )
    labs = labs[
        (labs["result_dt"] >= labs["admit_dt"]) &
        (labs["result_dt"] <= labs["discharge_dt"])
    ].drop(columns=["admit_dt", "discharge_dt"])
    print(f"  Labs after admission filter: {len(labs)} rows")

    # ── Load medications ──────────────────────────────────
    print("\nLoading medications...")
    medications = pd.read_csv(raw_dir / "medication_admins.csv")
    medications["end_datetime"] = pd.to_datetime(medications["end_datetime"])
    medications = medications.rename(columns={
        "drug_name":    "drug_description",
        "end_datetime": "given_dt",
        "project_id":   "subject_id",
        # drug_gpi kept as-is — already correct column name
    })
    medications = medications.merge(
        cohort[["admission_id", "subject_id", "admit_dt", "discharge_dt"]],
        on="subject_id", how="inner",
    )
    medications = medications[
        (medications["given_dt"] >= medications["admit_dt"]) &
        (medications["given_dt"] <= medications["discharge_dt"])
    ].drop(columns=["admit_dt", "discharge_dt"])
    print(f"  Medications after admission filter: {len(medications)} rows")

    # Audit GPI coverage
    if "drug_gpi" in medications.columns:
        gpi_coverage = medications["drug_gpi"].notna().mean() * 100
        print(f"  GPI code coverage: {gpi_coverage:.1f}%")

    # ── Load procedures ───────────────────────────────────
    print("\nLoading procedures...")
    procedures = pd.read_csv(raw_dir / "procedures.csv")
    procedures["start_datetime"] = pd.to_datetime(procedures["start_datetime"])
    procedures = procedures.rename(columns={
        "proc_local_code": "opcs_code",
        "start_datetime":  "proc_date",
        "project_id":      "subject_id",
    })
    procedures = procedures.merge(
        cohort[["admission_id", "subject_id", "admit_dt", "discharge_dt"]],
        on="subject_id", how="inner",
    )
    procedures = procedures[
        (procedures["proc_date"] >= procedures["admit_dt"]) &
        (procedures["proc_date"] <= procedures["discharge_dt"])
    ].drop(columns=["admit_dt", "discharge_dt"])
    print(f"  Procedures after admission filter: {len(procedures)} rows")

    # ── Load diagnoses ────────────────────────────────────
    print("\nLoading diagnoses...")
    diagnoses = pd.read_csv(raw_dir / "diagnoses.csv")
    diagnoses["start_datetime"] = pd.to_datetime(diagnoses["start_datetime"])
    diagnoses = diagnoses.rename(columns={
        "diag_local_code": "icd10_code",
        "start_datetime":  "diag_date",
        "project_id":      "subject_id",
    })
    diagnoses = diagnoses.merge(
        cohort[["admission_id", "subject_id", "admit_dt", "discharge_dt"]],
        on="subject_id", how="inner",
    )
    diagnoses = diagnoses[
        (diagnoses["diag_date"] >= diagnoses["admit_dt"]) &
        (diagnoses["diag_date"] <= diagnoses["discharge_dt"])
    ].drop(columns=["admit_dt", "discharge_dt"])
    print(f"  Diagnoses after admission filter: {len(diagnoses)} rows")

    # ── Build feature matrix ──────────────────────────────
    print("\nBuilding feature matrix...")
    features = build_feature_matrix(
        cohort       = cohort,
        labs         = labs,
        medications  = medications,
        procedures   = procedures,
        demographics = demographics,
        diagnoses    = diagnoses,
    )

    # ── Save ──────────────────────────────────────────────
    out_path = proc_dir / "features.parquet"
    features.to_parquet(out_path, index=False)
    print(f"\nSaved → {out_path}")

    # ── Feature column summary ────────────────────────────
    non_feature_cols = {
        "admission_id", "subject_id", "admit_dt", "discharge_dt",
        "window_end", "label", "aki_stage", "onset_dt",
        "onset_hours_from_admit", "exclude_early_onset",
        "baseline_creatinine", "baseline_source",
    }
    feature_cols = [c for c in features.columns if c not in non_feature_cols]
    print(f"\nFeature columns ({len(feature_cols)}):")
    groups = {
        "static":      [c for c in feature_cols
                        if c.startswith(("age_", "sex_", "ethnicity_"))],
        "diagnosis":   [c for c in feature_cols if c.startswith("diag_")],
        "lab":         [c for c in feature_cols
                        if any(c.startswith(f"{a}_") for a in CANONICAL_LABS)],
        "nephrotoxin": [c for c in feature_cols if c.startswith("nephrotoxin_")],
        "drug_class":  [c for c in feature_cols if c.startswith("drug_class_")],
        "procedure":   [c for c in feature_cols if c.startswith("proc_")],
        "other":       [c for c in feature_cols if not any(
                            c.startswith(p) for p in
                            ("age_", "sex_", "ethnicity_", "diag_",
                             "nephrotoxin_", "drug_class_", "proc_")
                        ) and not any(
                            c.startswith(f"{a}_") for a in CANONICAL_LABS)],
    }
    for grp, cols in groups.items():
        print(f"  {grp:12s}: {len(cols):3d} columns")


if __name__ == "__main__":
    main()
