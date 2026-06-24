"""
src/window_builder.py

Generates sliding time windows for temporal (LSTM) modelling of AKI onset.

Design
------
- One row per 6-hour step per admission
- Each row summarises ONLY the data in that 6-hour period (not cumulative)
- The LSTM accumulates history across rows via its hidden state
- Rows STOP being generated once AKI onset has occurred
- Label = 1 if AKI onset falls within the next HORIZON_H hours from window_end

Window layout per admission
---------------------------
    admit_dt
    │
    ├── step 0: hours 0–6
    ├── step 1: hours 6–12
    ├── step 2: hours 12–18
    │    ...
    ├── step N: last window before onset  ← label may be 1 here
    └── STOP — no rows generated at or after onset_dt

Output
------
    data/processed/windows.parquet
        one row per (admission_id, step_index)
        columns: admission_id, subject_id, step_index, window_start,
                 window_end, hours_since_admit, label, aki_stage,
                 onset_dt, [diag_cols], [static_feats], [lab_feats],
                 [med_feats], [proc_feats]

Usage
-----
    python src/window_builder.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Local imports ─────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from feature_engineering import (
    normalise_labs,
    extract_lab_features,
    extract_medication_features,
    extract_procedure_features,
    extract_static_features,
    CANONICAL_LABS,
)

# ── Window parameters ─────────────────────────────────────
STEP_H    = 6    # hours between rows — one row every 6 hours
HORIZON_H = 24   # predict AKI within this many hours ahead
MIN_OBS_H = 6    # minimum hours into admission before first window
                 # avoids near-empty windows right at admission start


def build_windows(
    cohort:       pd.DataFrame,
    labs_norm:    pd.DataFrame,
    medications:  pd.DataFrame,
    procedures:   pd.DataFrame,
    diag_cols:    list[str],
) -> pd.DataFrame:
    """
    Build one row per 6-hour step per admission.

    Parameters
    ----------
    cohort       : one row per admission, must contain:
                   admission_id, subject_id, admit_dt, discharge_dt,
                   aki_detected, onset_dt, aki_stage, age_years, sex,
                   ethnicity, exclude_early_onset, [diag_cols]
    labs_norm    : normalised labs with canonical_name column,
                   already filtered to admission windows
    medications  : medications already filtered to admission windows
    procedures   : procedures already filtered to admission windows
    diag_cols    : list of diag_* column names already on cohort

    Returns
    -------
    pd.DataFrame with one row per (admission, 6h step)
    """

    # ── Build fast lookup dicts ───────────────────────────
    labs_by_adm  = {
        aid: grp for aid, grp in labs_norm.groupby("admission_id")
    }
    meds_by_adm  = {
        aid: grp for aid, grp in medications.groupby("admission_id")
    }
    procs_by_adm = {
        aid: grp for aid, grp in procedures.groupby("admission_id")
    }

    # Empty fallback DataFrames with correct columns
    # — prevents KeyError on canonical_name when admission has no labs
    EMPTY_LABS  = pd.DataFrame(columns=labs_norm.columns)
    EMPTY_MEDS  = pd.DataFrame(columns=medications.columns)
    EMPTY_PROCS = pd.DataFrame(columns=procedures.columns)

    records = []
    n_admissions_with_windows = 0
    n_admissions_no_windows   = 0

    for _, adm in tqdm(cohort.iterrows(), total=len(cohort),
                       desc="Building windows"):

        adm_id   = adm["admission_id"]
        admit_dt = pd.Timestamp(adm["admit_dt"])
        dis_dt   = pd.Timestamp(adm["discharge_dt"])
        has_aki  = bool(adm.get("aki_detected", False))
        onset_dt = (
            pd.Timestamp(adm["onset_dt"])
            if has_aki and pd.notna(adm.get("onset_dt"))
            else pd.NaT
        )

        # Retrieve this admission's data
        adm_labs  = labs_by_adm.get(adm_id,  EMPTY_LABS)
        adm_meds  = meds_by_adm.get(adm_id,  EMPTY_MEDS)
        adm_procs = procs_by_adm.get(adm_id, EMPTY_PROCS)

        # First window ends after MIN_OBS_H hours
        window_end = admit_dt + pd.Timedelta(hours=MIN_OBS_H)
        step_idx   = 0

        while window_end <= dis_dt:

            # ── Stop once AKI has occurred ────────────────
            # We never want rows describing the post-onset period.
            # A window ending exactly at onset is still valid —
            # it describes the last period BEFORE onset.
            if has_aki and pd.notna(onset_dt) and window_end > onset_dt:
                break

            window_start = window_end - pd.Timedelta(hours=STEP_H)

            # ── Label ─────────────────────────────────────
            # 1 if AKI onset falls within (window_end, window_end + HORIZON_H]
            # i.e. "will AKI happen in the next 24 hours?"
            predict_end = window_end + pd.Timedelta(hours=HORIZON_H)
            if has_aki and pd.notna(onset_dt):
                label = int(window_end <= onset_dt <= predict_end)
            else:
                label = 0

            # Hours until onset — not a feature, used for analysis only
            hours_to_onset = (
                (onset_dt - window_end).total_seconds() / 3600
                if has_aki and pd.notna(onset_dt)
                else np.nan
            )

            # ── Extract features for this 6h period only ─
            # window_start to window_end — NOT cumulative from admission start
            if len(adm_labs) > 0:
                lab_win = adm_labs[
                    (adm_labs["result_dt"] >= window_start) &
                    (adm_labs["result_dt"] <  window_end)
                ]
            else:
                lab_win = EMPTY_LABS

            if len(adm_meds) > 0:
                med_win = adm_meds[
                    (adm_meds["given_dt"] >= window_start) &
                    (adm_meds["given_dt"] <  window_end)
                ]
            else:
                med_win = EMPTY_MEDS

            if len(adm_procs) > 0:
                proc_win = adm_procs[
                    (adm_procs["proc_date"] >= window_start) &
                    (adm_procs["proc_date"] <  window_end)
                ]
            else:
                proc_win = EMPTY_PROCS

            # ── Assemble row ──────────────────────────────
            row: dict = {
                # identifiers and temporal position
                "admission_id":      adm_id,
                "subject_id":        adm["subject_id"],
                "step_index":        step_idx,
                "window_start":      window_start,
                "window_end":        window_end,
                "hours_since_admit": round(
                    (window_end - admit_dt).total_seconds() / 3600, 1
                ),
                # label and metadata
                "label":             label,
                "aki_stage":         int(adm.get("aki_stage", 0)),
                "onset_dt":          onset_dt if pd.notna(onset_dt) else pd.NaT,
                "hours_to_onset":    round(hours_to_onset, 1)
                                     if not np.isnan(hours_to_onset)
                                     else np.nan,
            }

            # Diagnosis flags — static per admission, repeated on every row
            for col in diag_cols:
                row[col] = int(adm.get(col, 0))

            # Static features — age, sex, ethnicity
            # These are repeated on every row (static covariates for LSTM)
            row.update(extract_static_features(adm))

            # Temporal features — only data in this 6h window
            row.update(extract_lab_features(lab_win))
            row.update(extract_medication_features(med_win))
            row.update(extract_procedure_features(proc_win))

            records.append(row)

            # Advance
            step_idx   += 1
            window_end += pd.Timedelta(hours=STEP_H)

        if step_idx > 0:
            n_admissions_with_windows += 1
        else:
            n_admissions_no_windows += 1

    windows = pd.DataFrame(records)

    # ── Summary ───────────────────────────────────────────
    print(f"\n── Window builder summary ───────────────────────")
    print(f"Total windows:       {len(windows):,}")
    print(f"Positive (label=1):  {windows['label'].sum():,} "
          f"({windows['label'].mean()*100:.2f}%)")
    print(f"Admissions with windows:    {n_admissions_with_windows}")
    print(f"Admissions without windows: {n_admissions_no_windows} "
          f"(admission shorter than {MIN_OBS_H}h)")

    adm_counts = windows.groupby("admission_id").size()
    print(f"Steps per admission — "
          f"mean: {adm_counts.mean():.1f}, "
          f"median: {adm_counts.median():.1f}, "
          f"max: {adm_counts.max()}")

    # Positive window timing
    pos = windows[windows["label"] == 1]["hours_to_onset"].dropna()
    if len(pos) > 0:
        print(f"\nPositive windows — hours to onset:")
        print(f"  mean:   {pos.mean():.1f}h")
        print(f"  median: {pos.median():.1f}h")
        print(f"  min:    {pos.min():.1f}h")
        print(f"  max:    {pos.max():.1f}h")

    return windows


def main():
    raw_dir  = Path("/") # folder with raw hosptal data 
    proc_dir = Path("data/processed") # where outputs are saved

    proc_dir.mkdir(parents=True, exist_ok=True)

    # ── Load cohort ───────────────────────────────────────
    print("Loading cohort...")
    cohort = pd.read_parquet(proc_dir / "cohort.parquet") # cohort from aki_labelling_kdigo.py
    cohort = cohort[~cohort["exclude_early_onset"]].copy()
    print(f"  Admissions: {len(cohort)} "
          f"({cohort['aki_detected'].sum()} AKI)")

    # ── Load demographics for ethnicity ───────────────────
    print("Loading demographics...")
    demographics = pd.read_csv(raw_dir / "demographics.csv")

    demographics = demographics.rename(columns={
        "project_id":     "subject_id",
        "ethnicity_name": "ethnicity",
    })[["subject_id", "ethnicity"]].drop_duplicates(subset="subject_id")

    cohort = cohort.merge(demographics, on="subject_id", how="left")
    cohort["ethnicity"] = cohort["ethnicity"].fillna("not_available")

    # ── Load and encode diagnoses ─────────────────────────
    print("Loading diagnoses...")
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

    diagnoses["icd_chapter"] = diagnoses["icd10_code"].str[0].str.upper()
    diag_dummies = (
        pd.get_dummies(
            diagnoses[["admission_id", "icd_chapter"]].drop_duplicates(),
            columns=["icd_chapter"], prefix="diag",
        )
        .groupby("admission_id").max()
        .reset_index()
    )
    cohort    = cohort.merge(diag_dummies, on="admission_id", how="left")
    diag_cols = [c for c in cohort.columns if c.startswith("diag_")]
    cohort[diag_cols] = cohort[diag_cols].fillna(0).astype(int)
    print(f"  Diagnosis chapters: {len(diag_cols)}")

    # ── Load labs (chunked) ───────────────────────────────
    print("Loading labs...")
    LAB_COLS = ["project_id", "component_basename", "end_datetime",
                "NumericValue", "flag", "Unit"]
    chunks = []
    for chunk in pd.read_csv(
        raw_dir/ "lab_components.csv", # update for train test or silent
        usecols=LAB_COLS,
        dtype={"project_id": "str", "component_name": "str",
               "NumericValue": "float32", "flag": "str"},
        chunksize=200_000,
    ):
        chunks.append(chunk)
    labs = pd.concat(chunks, ignore_index=True)
    print(f"  Labs loaded: {len(labs):,} rows")

    labs["end_datetime"] = pd.to_datetime(labs["end_datetime"])
    labs = labs.rename(columns={
        "component_basename": "component_base_name",
        "end_datetime":   "result_dt",
        "project_id":     "subject_id",
        "NumericValue":   "value",
        "Unit": "unit"
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
    print(f"  Labs after admission filter: {len(labs):,} rows")

    # Normalise lab names once here — passed to build_windows
    print("  Normalising lab names...")
    labs_norm = normalise_labs(labs)
    print(f"  Labs after normalisation: {len(labs_norm):,} rows")

    # ── Load medications ──────────────────────────────────
    print("Loading medications...")
    medications = pd.read_csv(raw_dir / "medication_admins.csv") # update for train test or silent
    medications["end_datetime"] = pd.to_datetime(medications["end_datetime"])
    medications = medications.rename(columns={
        "drug_name":    "drug_description",
        "end_datetime": "given_dt",
        "project_id":   "subject_id",
    })
    medications = medications.merge(
        cohort[["admission_id", "subject_id", "admit_dt", "discharge_dt"]],
        on="subject_id", how="inner",
    )
    medications = medications[
        (medications["given_dt"] >= medications["admit_dt"]) &
        (medications["given_dt"] <= medications["discharge_dt"])
    ].drop(columns=["admit_dt", "discharge_dt"])
    print(f"  Medications after admission filter: {len(medications):,} rows")

    # ── Load procedures ───────────────────────────────────
    print("Loading procedures...")
    procedures = pd.read_csv(raw_dir / "icu_2024_procedures.csv") # update for train test or silent
    #procedures = pd.read_csv(raw_dir / "procedures.csv")
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
    print(f"  Procedures after admission filter: {len(procedures):,} rows")

    # ── Build windows ─────────────────────────────────────
    print(f"\nBuilding windows "
          f"(step={STEP_H}h, horizon={HORIZON_H}h, "
          f"min_obs={MIN_OBS_H}h)...")
    print("Rows stop at AKI onset — no post-onset windows generated.\n")

    windows = build_windows(
        cohort       = cohort,
        labs_norm    = labs_norm,
        medications  = medications,
        procedures   = procedures,
        diag_cols    = diag_cols,
    )

    # ── Save ──────────────────────────────────────────────
    out_path = proc_dir / "windows.parquet"
    windows.to_parquet(out_path, index=False)
    print(f"\nSaved → {out_path}")
    print(f"Shape: {windows.shape}")

    # ── Verify no post-onset rows ─────────────────────────
    aki_windows = windows[windows["onset_dt"].notna()].copy()
    if len(aki_windows) > 0:
        post_onset = aki_windows[
            aki_windows["window_end"] > aki_windows["onset_dt"]
        ]
        if len(post_onset) > 0:
            print(f"\nWARNING: {len(post_onset)} post-onset rows found — "
                  f"investigate build_windows logic")
        else:
            print("\nVerification passed — no post-onset rows in dataset")

    # ── Column summary ────────────────────────────────────
    non_feat = {
        "admission_id", "subject_id", "step_index",
        "window_start", "window_end", "hours_since_admit",
        "label", "aki_stage", "onset_dt", "hours_to_onset",
    }
    feat_cols = [c for c in windows.columns if c not in non_feat]
    print(f"\nFeature columns: {len(feat_cols)}")

    from feature_engineering import CANONICAL_LABS as CL
    groups = {
        "static":      [c for c in feat_cols
                        if c.startswith(("age_", "sex_", "ethnicity_"))],
        "diagnosis":   [c for c in feat_cols if c.startswith("diag_")],
        "lab":         [c for c in feat_cols
                        if any(c.startswith(f"{a}_") for a in CL)],
        "nephrotoxin": [c for c in feat_cols if c.startswith("nephrotoxin_")],
        "drug_class":  [c for c in feat_cols if c.startswith("drug_class_")],
        "procedure":   [c for c in feat_cols if c.startswith("proc_")],
    }
    for grp, cols in groups.items():
        print(f"  {grp:12s}: {len(cols):3d} columns")


if __name__ == "__main__":
    main()