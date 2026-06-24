"""
pic_pipeline.py

Process PIC database into 6h windows compatible with the GOSH-trained LSTM.

Confirmed from exploration:
  - Creatinine ITEMIDs: 5041, 6954, 5032 (blood, µmol/L only)
  - Diagnoses: ICD10_CODE_CN in DIAGNOSES_ICD and ADMISSIONS
  - Medications: DRUG_NAME_EN column with English names
  - ICU units: keep PICU, CICU, General ICU — exclude NICU, SICU
  - HADM_ID null rate in LABEVENTS: 0% — perfect linkage
  - Timestamps: 'YYYY-MM-DD HH:MM:SS' format

Usage
-----
    python pic_pipeline.py \\
        --pic_dir    /path/to/PIC/data \\
        --out_dir    data/pic_processed \\
        --feat_cols  model_weights/feature_cols.csv
    python pic_pipeline.py --pic_dir /home/jovyan/iona-shared/PIC_data/files/picdb/1.1.0 --out_dir /home/jovyan/AKI_PIC/pic_processed --feat_cols /home/jovyan/AKI_PIC/model_weights/feature_cols.csv
Runtime: ~20-40 min (LABEVENTS is 10M rows)
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


# ─────────────────────────────────────────────────────────
#  Configuration — confirmed from exploration
# ─────────────────────────────────────────────────────────

# Blood serum creatinine only — confirmed LOINC 2160-0
CREATININE_ITEMIDS = {5041, 6954, 5032}

# Valid units for creatinine µmol/L — confirmed in VALUEUOM
CREAT_UMOL_UNITS = {'μmol/L', 'µmol/L', 'umol/L', 'umol', 'μmol'}

# Sanity bounds for paediatric serum creatinine (µmol/L)
CREAT_MIN = 5.0
CREAT_MAX = 2000.0

# ICU units to include — matching GOSH PICU/nephrology setting
INCLUDE_UNITS = {'PICU', 'CICU', 'General ICU'}

# KDIGO thresholds
KDIGO_ABS_UMOL  = 26.5   # absolute rise within 48h
KDIGO_REL_MULT  = 1.5    # relative rise within 7d
STAGE2_MULT     = 2.0
STAGE3_MULT     = 3.0

# Windowing
STEP_H      = 6
HORIZON_H   = 24
MIN_STAY_H  = 6
MAX_STEPS   = 100

# Baseline lookback
PRIOR_7D    = 7
PRIOR_90D   = 90


# ─────────────────────────────────────────────────────────
#  Lab item map — ITEMIDs confirmed from D_LABITEMS
# ─────────────────────────────────────────────────────────

LAB_ITEMID_MAP = {
    # Creatinine (blood)
    5041: 'creatinine', 6954: 'creatinine', 5032: 'creatinine',
    # Urea
    5033: 'urea', 6477: 'urea',
    # Electrolytes
    5062: 'sodium',       5230: 'sodium',
    5226: 'potassium',
    5224: 'bicarbonate',  5248: 'bicarbonate',
    5218: 'chloride',
    5059: 'magnesium',
    5034: 'calcium',      5215: 'calcium',
    5072: 'phosphate',
    # Liver
    5026: 'alt',  5195: 'alt',
    5031: 'ast',
    5075: 'bilirubin', 5255: 'bilirubin',
    5024: 'albumin',   6465: 'albumin',
    5078: 'protein',
    # Haematology
    5099: 'haemoglobin', 5257: 'haemoglobin',
    5129: 'platelets',
    5141: 'wbc',
    5094: 'neutrophils',
    5110: 'lymphocytes', 5111: 'lymphocytes',
    5097: 'hct',         5225: 'hct',
    5114: 'mchc',
    5113: 'mch',
    5115: 'mcv',
    5132: 'rbc',
    5136: 'rdw',
    5120: 'mpv',
    # Blood gas
    5237: 'ph',  5238: 'ph',
    5235: 'pco2', 5236: 'pco2',
    5239: 'po2',  5244: 'po2',
    5252: 'spo2',
    5227: 'lactate',
    5211: 'base_excess', 5249: 'base_excess',
    5212: 'anion_gap',   5213: 'anion_gap',
    5220: 'tao2',
    5232: 'p50',
    5253: 'temperature',
    # Metabolic
    5047: 'glucose', 5223: 'glucose',
    # CRP / inflammation
    5626: 'crp', 5821: 'crp',
    # Renal
    6261: 'egfr',
    # Cardiac
    6192: 'troponin', 6525: 'troponin',
    # Coagulation
    5174: 'inr',
    # FIO2
    5222: 'fio2',
}

# Unique canonical lab names
CANONICAL_LABS = sorted(set(LAB_ITEMID_MAP.values()))


# ─────────────────────────────────────────────────────────
#  Medication mapping — confirmed from DRUG_NAME_EN
# ─────────────────────────────────────────────────────────

NEPHROTOXIN_MAP = {
    'vancomycin':      'nephrotoxin_vancomycin',
    'gentamicin':      'nephrotoxin_gentamicin',
    'tobramycin':      'nephrotoxin_tobramycin',
    'amikacin':        'nephrotoxin_amikacin',
    'colistin':        'nephrotoxin_colistin',
    'amphotericin':    'nephrotoxin_amphotericin',
    'cyclosporin':     'nephrotoxin_ciclosporin',
    'ciclosporin':     'nephrotoxin_ciclosporin',
    'cyclosporine':    'nephrotoxin_ciclosporin',
    'tacrolimus':      'nephrotoxin_tacrolimus',
    'ibuprofen':       'nephrotoxin_ibuprofen',
    'indomethacin':    'nephrotoxin_indomethacin',
    'indometacin':     'nephrotoxin_indomethacin',
    'contrast':        'nephrotoxin_contrast',
    'aciclovir':       'nephrotoxin_aciclovir',
    'acyclovir':       'nephrotoxin_aciclovir',
    'methotrexate':    'nephrotoxin_methotrexate',
    'cisplatin':       'nephrotoxin_cisplatin',
    'carboplatin':     'nephrotoxin_carboplatin',
    'ifosfamide':      'nephrotoxin_ifosfamide',
    'sirolimus':       'nephrotoxin_sirolimus',
}

DRUG_CLASS_MAP = {
    'furosemide':       'drug_class_diuretic',
    'frusemide':        'drug_class_diuretic',
    'spironolactone':   'drug_class_diuretic',
    'dopamine':         'drug_class_vasopressor',
    'dobutamine':       'drug_class_vasopressor',
    'norepinephrine':   'drug_class_vasopressor',
    'adrenaline':       'drug_class_vasopressor',
    'epinephrine':      'drug_class_vasopressor',
    'meropenem':        'drug_class_antibiotic',
    'cefotaxime':       'drug_class_antibiotic',
    'ceftriaxone':      'drug_class_antibiotic',
    'ampicillin':       'drug_class_antibiotic',
    'penicillin':       'drug_class_antibiotic',
    'fluconazole':      'drug_class_antifungal',
    'voriconazole':     'drug_class_antifungal',
    'methylprednisolone':'drug_class_immunosuppressant',
    'prednisolone':     'drug_class_immunosuppressant',
    'dexamethasone':    'drug_class_immunosuppressant',
    'midazolam':        'drug_class_sedation_analgesia',
    'fentanyl':         'drug_class_sedation_analgesia',
    'morphine':         'drug_class_sedation_analgesia',
    'heparin':          'drug_class_anticoagulant',
    'omeprazole':       'drug_class_ppi_h2',
    'pantoprazole':     'drug_class_ppi_h2',
    'ranitidine':       'drug_class_ppi_h2',
}

NEPHROTOXIN_COLS = [
    'nephrotoxin_vancomycin','nephrotoxin_gentamicin','nephrotoxin_tobramycin',
    'nephrotoxin_amikacin','nephrotoxin_colistin','nephrotoxin_amphotericin',
    'nephrotoxin_ciclosporin','nephrotoxin_tacrolimus','nephrotoxin_ibuprofen',
    'nephrotoxin_indomethacin','nephrotoxin_contrast','nephrotoxin_aciclovir',
    'nephrotoxin_methotrexate','nephrotoxin_cisplatin','nephrotoxin_carboplatin',
    'nephrotoxin_ifosfamide','nephrotoxin_sirolimus','nephrotoxin_any',
]

DRUG_CLASS_COLS = [
    'drug_class_diuretic','drug_class_ace_inhibitor','drug_class_arb',
    'drug_class_vasopressor','drug_class_antibiotic','drug_class_antifungal',
    'drug_class_immunosuppressant','drug_class_antiviral','drug_class_ppi_h2',
    'drug_class_anticoagulant','drug_class_sedation_analgesia',
]

PROC_COLS = [
    'proc_cardiac_surgery','proc_renal_procedure','proc_transplant',
    'proc_bypass_circuit','proc_dialysis_rrt','proc_imaging_contrast',
    'proc_mechanical_vent','proc_central_line','proc_abdominal_surgery',
    'proc_neurosurgery','proc_orthopaedic','proc_bronchoscopy_scope',
]


# ─────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────

def gz(pic_dir, name):
    return Path(pic_dir) / f'{name}.csv.gz'


def age_band(a):
    if a < 0.083:   return 'neonate'
    elif a < 1:     return 'infant'
    elif a < 3:     return 'toddler'
    elif a < 12:    return 'child'
    return 'adolescent'


def pop_creatinine_normal(age_years, sex):
    """Age/sex-adjusted population creatinine reference (µmol/L)."""
    if age_years < 0.083:  return 30.0
    elif age_years < 1:    return 25.0
    elif age_years < 3:    return 28.0
    elif age_years < 6:    return 35.0
    elif age_years < 12:   return 45.0
    elif str(sex) == 'M':  return 65.0
    return 55.0


def classify_drug(drug_en):
    """Map DRUG_NAME_EN to nephrotoxin/drug class flags."""
    if pd.isna(drug_en):
        return []
    name = str(drug_en).lower()
    flags = []
    for kw, col in NEPHROTOXIN_MAP.items():
        if kw in name:
            flags.append(col)
    for kw, col in DRUG_CLASS_MAP.items():
        if kw in name:
            flags.append(col)
    return flags


# ─────────────────────────────────────────────────────────
#  Step 1 — Build cohort
# ─────────────────────────────────────────────────────────

def build_cohort(pic_dir):
    print("\n── Step 1: Building cohort ──────────────────────────")

    adm = pd.read_csv(gz(pic_dir, 'ADMISSIONS'),
                      parse_dates=['ADMITTIME', 'DISCHTIME'])
    pat = pd.read_csv(gz(pic_dir, 'PATIENTS'),
                      parse_dates=['DOB'])
    icu = pd.read_csv(gz(pic_dir, 'ICUSTAYS'),
                      parse_dates=['INTIME', 'OUTTIME'])

    print(f"  Raw ICU stays:  {len(icu):,}")
    print(f"  Care unit counts:")
    print(icu['FIRST_CAREUNIT'].value_counts().to_string())

    # Filter to PICU / CICU / General ICU
    icu = icu[icu['FIRST_CAREUNIT'].isin(INCLUDE_UNITS)].copy()
    print(f"\n  After unit filter: {len(icu):,} stays")

    # Merge admissions, patients, ICU stays
    print("ADMISSIONS columns:", adm.columns.tolist())
    print("PATIENTS columns:", pat.columns.tolist())
    print("ICUSTAYS columns:", icu.columns.tolist())
    
    cohort = (
        icu
        .merge(
            adm[['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME',
                 'ICD10_CODE_CN', 'HOSPITAL_EXPIRE_FLAG']],
            on=['SUBJECT_ID', 'HADM_ID'],
            how='inner'
        )
        .merge(
            pat[['SUBJECT_ID', 'GENDER', 'DOB']],
            on='SUBJECT_ID',
            how='left'
        )
    )
    # Use ICU INTIME as admission start (more precise than hospital ADMITTIME)
    cohort['admit_dt']    = cohort['INTIME']
    cohort['discharge_dt']= cohort['OUTTIME']

    # Age at ICU admission
    cohort['age_years'] = (
        (cohort['admit_dt'] - cohort['DOB']).dt.days / 365.25
    )

    # Restrict to 0–18 years
    cohort = cohort[
        (cohort['age_years'] >= 0) &
        (cohort['age_years'] <= 18)
    ].copy()

    # Minimum stay
    cohort['los_h'] = (
        (cohort['discharge_dt'] - cohort['admit_dt'])
        .dt.total_seconds() / 3600
    )
    cohort = cohort[cohort['los_h'] >= MIN_STAY_H].copy()

    # Unique admission identifier (use HADM_ID as string)
    cohort['admission_id'] = cohort['HADM_ID'].astype(str)

    # Demographic features
    cohort['age_band_label'] = cohort['age_years'].apply(age_band)
    cohort['sex']            = cohort['GENDER']

    # Primary ICD-10 chapter (first character of ICD10_CODE_CN)
    cohort['icd_chapter'] = (
        cohort['ICD10_CODE_CN']
        .fillna('Z')
        .str[0]
        .str.upper()
    )

    print(f"\n  Final cohort: {len(cohort):,} admissions, "
          f"{cohort['SUBJECT_ID'].nunique():,} patients")
    print(f"  Age: median {cohort['age_years'].median():.1f}y  "
          f"IQR {cohort['age_years'].quantile(0.25):.1f}–"
          f"{cohort['age_years'].quantile(0.75):.1f}y")
    print(f"  Sex: {(cohort['sex']=='M').mean()*100:.1f}% male")
    print(f"  ICD chapters (top 10):")
    print(cohort['icd_chapter'].value_counts().head(10).to_string())

    return cohort.reset_index(drop=True)


# ─────────────────────────────────────────────────────────
#  Step 2 — Load creatinine
# ─────────────────────────────────────────────────────────

def load_creatinine(pic_dir, hadm_ids):
    print("\n── Step 2: Loading creatinine ───────────────────────")

    lab = pd.read_csv(
        gz(pic_dir, 'LABEVENTS'),
        usecols=['SUBJECT_ID','HADM_ID','ITEMID',
                 'CHARTTIME','VALUENUM','VALUEUOM'],
        parse_dates=['CHARTTIME'],
    )

    # Filter to creatinine ITEMIDs and cohort admissions
    cr = lab[
        lab['ITEMID'].isin(CREATININE_ITEMIDS) &
        lab['HADM_ID'].isin(hadm_ids)
    ].copy()
    print(f"  Creatinine rows (all units): {len(cr):,}")

    # Keep µmol/L only — exclude CK contamination in U/L
    cr_umol = cr[cr['VALUEUOM'].isin(CREAT_UMOL_UNITS)].copy()
    print(f"  After µmol/L filter:         {len(cr_umol):,}")

    # Sanity bounds
    cr_clean = cr_umol[
        (cr_umol['VALUENUM'] >= CREAT_MIN) &
        (cr_umol['VALUENUM'] <= CREAT_MAX) &
        cr_umol['VALUENUM'].notna()
    ].copy()
    print(f"  After sanity bounds:         {len(cr_clean):,}")
    print(f"  Admissions with creatinine:  "
          f"{cr_clean['HADM_ID'].nunique():,} / {len(hadm_ids):,}")
    print(f"  Median {cr_clean['VALUENUM'].median():.0f} µmol/L  "
          f"IQR {cr_clean['VALUENUM'].quantile(0.25):.0f}–"
          f"{cr_clean['VALUENUM'].quantile(0.75):.0f}")

    cr_clean = cr_clean.sort_values(
        ['HADM_ID','CHARTTIME']
    ).reset_index(drop=True)

    return cr_clean


# ─────────────────────────────────────────────────────────
#  Step 3 — KDIGO labelling
# ─────────────────────────────────────────────────────────

def derive_baseline(cr_adm_win, admit_dt, age_years, sex, cr_subj):
    """
    Baseline creatinine cascade matching GOSH pipeline:
    1. Lowest in 7 days prior
    2. Lowest in 7–90 days prior
    3. First admission measurement
    4. Age/sex population normal
    """
    prior = cr_subj[cr_subj['CHARTTIME'] < admit_dt]

    w7 = prior[prior['CHARTTIME'] >= admit_dt - pd.Timedelta(days=PRIOR_7D)]
    if len(w7) > 0:
        return w7['VALUENUM'].min(), '7d_prior'

    w90 = prior[
        (prior['CHARTTIME'] >= admit_dt - pd.Timedelta(days=PRIOR_90D)) &
        (prior['CHARTTIME'] <  admit_dt - pd.Timedelta(days=PRIOR_7D))
    ]
    if len(w90) > 0:
        return w90['VALUENUM'].min(), '7_90d_prior'

    if len(cr_adm_win) > 0:
        return cr_adm_win['VALUENUM'].iloc[0], 'first_admission'

    return pop_creatinine_normal(age_years, sex), 'population_normal'


def label_aki(cohort, creatinine):
    print("\n── Step 3: KDIGO labelling ───────────────────────────")

    cr_by_hadm  = {h: g.sort_values('CHARTTIME')
                   for h, g in creatinine.groupby('HADM_ID')}
    cr_by_subj  = {s: g.sort_values('CHARTTIME')
                   for s, g in creatinine.groupby('SUBJECT_ID')}

    results = []
    n_no_cr = 0

    for _, row in tqdm(cohort.iterrows(), total=len(cohort),
                       desc='  Labelling AKI'):
        hadm_id  = int(row['HADM_ID'])
        subj_id  = int(row['SUBJECT_ID'])
        admit_dt = row['admit_dt']
        dis_dt   = row['discharge_dt']

        cr_adm  = cr_by_hadm.get(hadm_id,
                                  pd.DataFrame(columns=['CHARTTIME','VALUENUM']))
        cr_subj = cr_by_subj.get(subj_id,
                                  pd.DataFrame(columns=['CHARTTIME','VALUENUM']))

        # Creatinine within admission window
        cr_win = cr_adm[
            (cr_adm['CHARTTIME'] >= admit_dt) &
            (cr_adm['CHARTTIME'] <= dis_dt)
        ].sort_values('CHARTTIME')

        if len(cr_win) == 0:
            n_no_cr += 1
            results.append({
                'admission_id':        str(hadm_id),
                'aki_detected':        False,
                'onset_dt':            pd.NaT,
                'aki_stage':           0,
                'baseline_cr':         np.nan,
                'baseline_source':     'no_creatinine',
                'exclude_early_onset': False,
            })
            continue

        baseline, bl_src = derive_baseline(
            cr_win, admit_dt, row['age_years'], row['sex'], cr_subj
        )

        aki_detected = False
        onset_dt     = pd.NaT
        aki_stage    = 0

        for _, meas in cr_win.iterrows():
            val  = meas['VALUENUM']
            t    = meas['CHARTTIME']

            # Criterion 1: absolute rise ≥26.5 µmol/L within 48h
            w48 = cr_win[
                (cr_win['CHARTTIME'] >= t - pd.Timedelta(hours=48)) &
                (cr_win['CHARTTIME'] <  t)
            ]
            abs_rise = val - (w48['VALUENUM'].min() if len(w48) > 0
                               else baseline)

            # Criterion 2: relative rise ≥1.5× within 7 days
            w7d = cr_win[
                (cr_win['CHARTTIME'] >= t - pd.Timedelta(days=7)) &
                (cr_win['CHARTTIME'] <  t)
            ]
            ref       = w7d['VALUENUM'].min() if len(w7d) > 0 else baseline
            rel_ratio = val / (ref + 1e-9)

            if abs_rise >= KDIGO_ABS_UMOL or rel_ratio >= KDIGO_REL_MULT:
                aki_detected = True
                onset_dt     = t
                ratio_bl     = val / (baseline + 1e-9)
                if ratio_bl >= STAGE3_MULT or val >= 354:
                    aki_stage = 3
                elif ratio_bl >= STAGE2_MULT:
                    aki_stage = 2
                else:
                    aki_stage = 1
                break

        # Exclude on-arrival AKI (onset within 24h of admission)
        exclude = (
            aki_detected and
            pd.notna(onset_dt) and
            (onset_dt - admit_dt).total_seconds() / 3600 <= 24
        )

        results.append({
            'admission_id':        str(hadm_id),
            'aki_detected':        aki_detected,
            'onset_dt':            onset_dt,
            'aki_stage':           aki_stage,
            'baseline_cr':         float(baseline),
            'baseline_source':     bl_src,
            'exclude_early_onset': exclude,
        })

    label_df = pd.DataFrame(results)
    cohort   = cohort.copy()
    cohort['admission_id'] = cohort['HADM_ID'].astype(str)
    cohort   = cohort.merge(label_df, on='admission_id', how='left')
    cohort   = cohort[~cohort['exclude_early_onset']].copy()

    n_total  = len(cohort)
    n_aki    = cohort['aki_detected'].sum()
    print(f"\n  After on-arrival exclusion: {n_total:,} admissions")
    print(f"  AKI:     {n_aki:,} ({n_aki/n_total*100:.1f}%)")
    print(f"  Non-AKI: {n_total-n_aki:,}")
    print(f"  No creatinine: {n_no_cr:,}")
    print(f"  AKI stage distribution:")
    print(
        cohort[cohort['aki_detected']]['aki_stage']
        .value_counts().sort_index().to_string()
    )
    print(f"\n  Baseline source distribution:")
    print(cohort['baseline_source'].value_counts().to_string())

    return cohort.reset_index(drop=True)


# ─────────────────────────────────────────────────────────
#  Step 4 — Load all lab events
# ─────────────────────────────────────────────────────────

def load_labs(pic_dir, hadm_ids):
    print("\n── Step 4: Loading all lab events ───────────────────")

    chunks = []
    reader = pd.read_csv(
        gz(pic_dir, 'LABEVENTS'),
        usecols=['HADM_ID','ITEMID','CHARTTIME','VALUENUM'],
        parse_dates=['CHARTTIME'],
        chunksize=500_000,
    )
    for chunk in tqdm(reader, desc='  Reading LABEVENTS'):
        sub = chunk[
            chunk['HADM_ID'].isin(hadm_ids) &
            chunk['ITEMID'].isin(LAB_ITEMID_MAP.keys()) &
            chunk['VALUENUM'].notna()
        ]
        if len(sub) > 0:
            chunks.append(sub)

    labs = pd.concat(chunks, ignore_index=True)
    labs['canonical'] = labs['ITEMID'].map(LAB_ITEMID_MAP)
    labs['HADM_ID']   = labs['HADM_ID'].astype(str)

    print(f"  Mapped lab events: {len(labs):,}")
    present = labs['canonical'].value_counts()
    print(f"  Labs present ({len(present)}):")
    print(present.to_string())

    return labs


# ─────────────────────────────────────────────────────────
#  Step 5 — Load prescriptions
# ─────────────────────────────────────────────────────────

def load_rx(pic_dir, hadm_ids):
    print("\n── Step 5: Loading prescriptions ────────────────────")

    rx = pd.read_csv(
        gz(pic_dir, 'PRESCRIPTIONS'),
        usecols=['HADM_ID','STARTDATE','ENDDATE','DRUG_NAME_EN'],
        parse_dates=['STARTDATE','ENDDATE'],
    )
    rx = rx[rx['HADM_ID'].isin(hadm_ids)].copy()
    rx['HADM_ID']   = rx['HADM_ID'].astype(str)
    rx['given_dt']  = rx['ENDDATE'].fillna(rx['STARTDATE'])
    rx['drug_flags']= rx['DRUG_NAME_EN'].apply(classify_drug)

    print(f"  Prescriptions: {len(rx):,}")

    # Summary of nephrotoxin/drug class hits
    from collections import Counter
    all_flags = [f for flags in rx['drug_flags'] for f in flags]
    if all_flags:
        print(f"  Drug flag counts (top 15):")
        for flag, n in Counter(all_flags).most_common(15):
            print(f"    {flag:<40} {n:,}")

    return rx


# ─────────────────────────────────────────────────────────
#  Step 6 — Feature engineering per window
# ─────────────────────────────────────────────────────────

def lab_features(lab_win, canonical_labs):
    row = {}

    # Handle admissions/windows with no labs
    if lab_win is None or len(lab_win) == 0 or 'canonical' not in lab_win.columns:
        for lab in canonical_labs:
            for s in ['mean', 'min', 'max', 'last', 'trend', 'n']:
                row[f'{lab}_{s}'] = np.nan
            row[f'{lab}_missing'] = 1
            row[f'{lab}_alert'] = 0
        return row

    for lab in canonical_labs:
        sub = lab_win.loc[lab_win['canonical'] == lab, 'VALUENUM'].values

        if len(sub) == 0:
            for s in ['mean', 'min', 'max', 'last', 'trend', 'n']:
                row[f'{lab}_{s}'] = np.nan
            row[f'{lab}_missing'] = 1
            row[f'{lab}_alert'] = 0
        else:
            row[f'{lab}_mean'] = float(np.mean(sub))
            row[f'{lab}_min'] = float(np.min(sub))
            row[f'{lab}_max'] = float(np.max(sub))
            row[f'{lab}_last'] = float(sub[-1])
            row[f'{lab}_trend'] = float(sub[-1] - sub[0]) if len(sub) > 1 else 0.0
            row[f'{lab}_n'] = int(len(sub))
            row[f'{lab}_missing'] = 0
            row[f'{lab}_alert'] = 0

    return row


def med_features(rx_win):
    row = {c: 0 for c in NEPHROTOXIN_COLS + DRUG_CLASS_COLS}
    row.update({'n_unique_drugs': 0, 'nephrotoxin_total_count': 0})
    if len(rx_win) == 0:
        return row

    all_flags = set(f for flags in rx_win['drug_flags'] for f in flags)
    for flag in all_flags:
        if flag in row:
            row[flag] = 1

    row['nephrotoxin_any'] = int(
        any(row.get(c, 0) for c in NEPHROTOXIN_COLS if c != 'nephrotoxin_any')
    )
    row['nephrotoxin_total_count'] = sum(
        row.get(c, 0)
        for c in NEPHROTOXIN_COLS if c != 'nephrotoxin_any'
    )
    row['n_unique_drugs'] = int(
        rx_win['DRUG_NAME_EN'].dropna().nunique()
    )
    return row


# ─────────────────────────────────────────────────────────
#  Step 7 — Build 6h windows
# ─────────────────────────────────────────────────────────

def build_windows(cohort, labs, rx):
    print("\n── Step 6: Building 6-hour windows ─────────────────")

    labs_by_adm = {h: g for h, g in labs.groupby('HADM_ID')}
    rx_by_adm   = {h: g for h, g in rx.groupby('HADM_ID')}

    # ICD chapter dummies
    all_chapters = sorted(set(
        c.upper() for c in
        'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    ))

    records  = []
    n_ok     = 0

    for _, adm in tqdm(cohort.iterrows(), total=len(cohort),
                       desc='  Building windows'):
        hid      = adm['admission_id']
        admit_dt = pd.Timestamp(adm['admit_dt'])
        dis_dt   = pd.Timestamp(adm['discharge_dt'])
        has_aki  = bool(adm['aki_detected'])
        onset_dt = (
            pd.Timestamp(adm['onset_dt'])
            if has_aki and pd.notna(adm['onset_dt'])
            else pd.NaT
        )

        empty_labs = pd.DataFrame(columns=['HADM_ID', 'ITEMID', 'CHARTTIME', 'VALUENUM', 'canonical'])
        empty_rx   = pd.DataFrame(columns=['HADM_ID', 'STARTDATE', 'ENDDATE', 'DRUG_NAME_EN', 'given_dt', 'drug_flags'])

        empty_labs = pd.DataFrame(columns=['HADM_ID', 'ITEMID', 'CHARTTIME', 'VALUENUM', 'canonical'])

        adm_labs = labs_by_adm.get(hid, empty_labs)
        adm_rx   = rx_by_adm.get(hid,  empty_rx)

        # Static features
        ab  = adm['age_band_label']
        ch  = adm['icd_chapter']
        sex = adm['sex']

        static = {
            'age_years':       float(adm['age_years']),
            'age_neonate':     int(ab == 'neonate'),
            'age_infant':      int(ab == 'infant'),
            'age_toddler':     int(ab == 'toddler'),
            'age_child':       int(ab == 'child'),
            'age_adolescent':  int(ab == 'adolescent'),
            'sex_male':        int(str(sex) == 'M'),
            # Ethnicity: PIC is a Chinese hospital
            # All patients assumed Han Chinese — map to ethnicity_asian
            'ethnicity_white':         0,
            'ethnicity_black':         0,
            'ethnicity_asian':         1,
            'ethnicity_mixed':         0,
            'ethnicity_other':         0,
            'ethnicity_not_available': 0,
            # Procedures: not available in PIC — fill 0
            **{c: 0 for c in PROC_COLS},
            'n_procedures': 0,
            # ICD chapter dummies
            **{f'diag_{c}': (1 if ch == c else 0) for c in all_chapters},
        }

        window_end = admit_dt + pd.Timedelta(hours=STEP_H)
        step_idx   = 0

        while window_end <= dis_dt and step_idx < MAX_STEPS:

            # Stop generating after onset
            if has_aki and pd.notna(onset_dt) and window_end > onset_dt:
                break

            window_start = window_end - pd.Timedelta(hours=STEP_H)

            # Label: AKI onset within next 24h
            if has_aki and pd.notna(onset_dt):
                label = int(window_end <= onset_dt <=
                             window_end + pd.Timedelta(hours=HORIZON_H))
                h2o   = (onset_dt - window_end).total_seconds() / 3600
            else:
                label = 0
                h2o   = np.nan

            row = {
                'admission_id':       hid,
                'subject_id':         str(adm['SUBJECT_ID']),
                'step_index':         step_idx,
                'hours_since_admit':  round(
                    (window_end - admit_dt).total_seconds() / 3600, 1
                ),
                'label':              label,
                'aki_stage':          int(adm['aki_stage']),
                'hours_to_onset':     round(h2o, 1)
                                      if not np.isnan(h2o) else np.nan,
                'has_aki':            int(has_aki),
            }
            row.update(static)

            # Lab features for this window
            if len(adm_labs) > 0 and 'canonical' in adm_labs.columns:
                lw = adm_labs[
                    (adm_labs['CHARTTIME'] >= window_start) &
                    (adm_labs['CHARTTIME'] <  window_end)
                ]
            else:
                lw = pd.DataFrame(columns=['HADM_ID', 'ITEMID', 'CHARTTIME', 'VALUENUM', 'canonical'])

            row.update(lab_features(lw, CANONICAL_LABS))

            # Medication features
            if len(adm_rx) > 0:
                rw = adm_rx[
                    (adm_rx['given_dt'] >= window_start) &
                    (adm_rx['given_dt'] <  window_end)
                ]
            else:
                rw = pd.DataFrame()
            row.update(med_features(rw))

            records.append(row)
            step_idx   += 1
            window_end += pd.Timedelta(hours=STEP_H)

        if step_idx > 0:
            n_ok += 1

    windows = pd.DataFrame(records)
    print(f"\n  Admissions with windows: {n_ok:,}")
    print(f"  Total windows:           {len(windows):,}")
    print(f"  Positive (label=1):      "
          f"{windows['label'].sum():,} "
          f"({windows['label'].mean()*100:.2f}%)")

    return windows


# ─────────────────────────────────────────────────────────
#  Step 8 — Align to GOSH feature columns
# ─────────────────────────────────────────────────────────

def align_features(windows, feat_cols):
    print("\n── Step 7: Aligning to GOSH feature set ────────────")

    # Add _missing_flag columns for lab stats that are NaN
    lab_stat_cols = [
        c for c in windows.columns
        if any(c.endswith(s)
               for s in ['_mean','_min','_max','_last','_trend'])
        and not c.endswith('_missing_flag')
    ]
    for col in lab_stat_cols:
        flag = f'{col}_missing_flag'
        if flag not in windows.columns:
            windows[flag] = windows[col].isna().astype(float)

    meta_cols = {
        'admission_id','subject_id','step_index','hours_since_admit',
        'label','aki_stage','hours_to_onset','has_aki',
    }

    missing = [c for c in feat_cols if c not in windows.columns]
    extra   = [c for c in windows.columns
               if c not in feat_cols and c not in meta_cols]

    print(f"  GOSH features:           {len(feat_cols)}")
    print(f"  PIC features present:    {len(feat_cols) - len(missing)}")
    print(f"  Missing (filled 0):      {len(missing)}")
    print(f"  Extra (dropped):         {len(extra)}")

    # Group missing by type
    for grp in ['proc_','nephrotoxin_','drug_class_','diag_','_missing_flag']:
        grp_m = [c for c in missing if grp in c]
        if grp_m:
            print(f"    {grp}: {len(grp_m)} missing columns")

    # Add missing as 0
    for col in missing:
        windows[col] = 0.0

    # Forward-fill within admission then fill NaN with 0
    fill_cols = feat_cols
    windows = windows.sort_values(['admission_id','step_index'])
    windows[fill_cols] = (
        windows.groupby('admission_id')[fill_cols]
        .transform(lambda x: x.ffill())
        .fillna(0)
    )

    remaining_nan = windows[feat_cols].isna().sum().sum()
    print(f"  NaN remaining after fill: {remaining_nan}")

    return windows


# ─────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description='Process PIC database for GOSH LSTM zero-shot transfer'
    )
    p.add_argument('--pic_dir',   required=True)
    p.add_argument('--out_dir',   default='data/pic_processed')
    p.add_argument('--feat_cols', default='model_weights/feature_cols.csv')
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load GOSH feature columns
    feat_cols = pd.read_csv(args.feat_cols, header=None)[0].tolist()
    print(f"GOSH feature columns: {len(feat_cols)}")

    # Run pipeline
    cohort    = build_cohort(args.pic_dir)
    hadm_ids  = set(cohort['HADM_ID'].astype(int))
    creatinine= load_creatinine(args.pic_dir, hadm_ids)
    cohort    = label_aki(cohort, creatinine)
    labs      = load_labs(args.pic_dir, set(cohort['HADM_ID'].astype(int)))
    rx        = load_rx(args.pic_dir,   set(cohort['HADM_ID'].astype(int)))
    windows   = build_windows(cohort, labs, rx)
    windows   = align_features(windows, feat_cols)

    # Save
    cohort_path  = out_dir / 'cohort_pic.csv'
    windows_path = out_dir / 'windows_pic.csv'
    cohort.to_csv(cohort_path,   index=False)
    windows.to_csv(windows_path, index=False)

    # Summary
    n_total  = len(cohort)
    n_aki    = cohort['aki_detected'].sum()
    print(f"\n{'='*55}")
    print("PIC PIPELINE COMPLETE")
    print(f"{'='*55}")
    print(f"  Cohort:          {n_total:,} admissions")
    print(f"  AKI:             {n_aki:,} ({n_aki/n_total*100:.1f}%)")
    print(f"  Total windows:   {len(windows):,}")
    print(f"  Positive rate:   {windows['label'].mean()*100:.2f}%")
    print(f"\n  Cohort  → {cohort_path}")
    print(f"  Windows → {windows_path}")
    print(f"\n  Next step:")
    print(f"    python pic_inference.py \\")
    print(f"        --windows   {windows_path} \\")
    print(f"        --weights_dir model_weights \\")
    print(f"        --out_dir   data/pic_results")


if __name__ == '__main__':
    main()