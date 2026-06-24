"""
src/fake_data.py

Generate synthetic PICU patient data with exactly the same feature columns
as the real GOSH AKI prediction pipeline (633 features from feature_cols.csv).

All data is entirely synthetic. No real patient information is used.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

# ── Exact lab bases from feature_cols.csv ─────────────────
CANONICAL_LABS = [
    'creatinine', 'urea', 'potassium', 'sodium', 'bicarbonate',
    'haemoglobin', 'platelets', 'alt', 'ast', 'ph', 'lactate',
    'egfr', 'albumin', 'crp', 'wbc', 'neutrophils', 'phosphate',
    'calcium', 'magnesium', 'glucose', 'bilirubin', 'inr',
    'pco2', 'po2', 'troponin', 'chloride', 'base_excess',
    'osmolality', 'lymphocytes', 'mcv', 'rbc', 'hct', 'fio2',
    'spo2', 'temperature', 'anion_gap', 'mchc', 'mch',
    'p50', 'tao2', 'mpv',
]

# Normal ranges: (low, high, unit)
LAB_NORMALS = {
    'creatinine':  (25,   70,    'µmol/L'),
    'urea':        (2.5,  7.0,   'mmol/L'),
    'potassium':   (3.5,  5.0,   'mmol/L'),
    'sodium':      (135,  145,   'mmol/L'),
    'bicarbonate': (22,   29,    'mmol/L'),
    'haemoglobin': (100,  145,   'g/L'),
    'platelets':   (150,  400,   'x10⁹/L'),
    'alt':         (7,    45,    'U/L'),
    'ast':         (10,   40,    'U/L'),
    'ph':          (7.35, 7.45,  ''),
    'lactate':     (0.5,  2.0,   'mmol/L'),
    'egfr':        (60,   120,   'mL/min/1.73m²'),
    'albumin':     (35,   50,    'g/L'),
    'crp':         (0,    10,    'mg/L'),
    'wbc':         (4.0,  12.0,  'x10⁹/L'),
    'neutrophils': (1.8,  7.5,   'x10⁹/L'),
    'phosphate':   (0.8,  1.5,   'mmol/L'),
    'calcium':     (2.1,  2.6,   'mmol/L'),
    'magnesium':   (0.7,  1.1,   'mmol/L'),
    'glucose':     (3.5,  7.0,   'mmol/L'),
    'bilirubin':   (3,    21,    'µmol/L'),
    'inr':         (0.8,  1.2,   ''),
    'pco2':        (4.5,  6.0,   'kPa'),
    'po2':         (10,   13,    'kPa'),
    'troponin':    (0,    14,    'ng/L'),
    'chloride':    (98,   107,   'mmol/L'),
    'base_excess': (-2,   2,     'mmol/L'),
    'osmolality':  (275,  295,   'mOsm/kg'),
    'lymphocytes': (1.0,  4.0,   'x10⁹/L'),
    'mcv':         (70,   90,    'fL'),
    'rbc':         (3.8,  5.2,   'x10¹²/L'),
    'hct':         (0.33, 0.43,  ''),
    'fio2':        (0.21, 0.40,  ''),
    'spo2':        (95,   100,   '%'),
    'temperature': (36.5, 37.5,  '°C'),
    'anion_gap':   (8,    16,    'mmol/L'),
    'mchc':        (310,  360,   'g/L'),
    'mch':         (27,   33,    'pg'),
    'p50':         (25,   30,    'mmHg'),
    'tao2':        (8,    12,    'mL/dL'),
    'mpv':         (7.5,  12.5,  'fL'),
}

# Exact nephrotoxin binary cols
NEPHROTOXIN_BINARY = [
    'nephrotoxin_vancomycin', 'nephrotoxin_gentamicin',
    'nephrotoxin_tobramycin', 'nephrotoxin_amikacin',
    'nephrotoxin_colistin', 'nephrotoxin_amphotericin',
    'nephrotoxin_ciclosporin', 'nephrotoxin_tacrolimus',
    'nephrotoxin_ibuprofen', 'nephrotoxin_indomethacin',
    'nephrotoxin_contrast', 'nephrotoxin_aciclovir',
    'nephrotoxin_methotrexate', 'nephrotoxin_cisplatin',
    'nephrotoxin_carboplatin', 'nephrotoxin_ifosfamide',
    'nephrotoxin_sirolimus', 'nephrotoxin_any',
]

DRUG_CLASS_COLS = [
    'drug_class_diuretic', 'drug_class_ace_inhibitor',
    'drug_class_arb', 'drug_class_vasopressor',
    'drug_class_antibiotic', 'drug_class_antifungal',
    'drug_class_immunosuppressant', 'drug_class_antiviral',
    'drug_class_ppi_h2', 'drug_class_anticoagulant',
    'drug_class_sedation_analgesia',
]

PROC_COLS = [
    'proc_cardiac_surgery', 'proc_renal_procedure',
    'proc_transplant', 'proc_bypass_circuit',
    'proc_dialysis_rrt', 'proc_imaging_contrast',
    'proc_mechanical_vent', 'proc_central_line',
    'proc_abdominal_surgery', 'proc_neurosurgery',
    'proc_orthopaedic', 'proc_bronchoscopy_scope',
]

DIAG_COLS = [
    'diag_A','diag_B','diag_C','diag_D','diag_E','diag_F',
    'diag_G','diag_H','diag_I','diag_J','diag_K','diag_L',
    'diag_M','diag_N','diag_O','diag_P','diag_Q','diag_R',
    'diag_S','diag_T','diag_U','diag_W','diag_X','diag_Y','diag_Z',
]

# Missing flag columns (one per lab × {mean,min,max,last,trend})
MISSING_FLAG_COLS = [
    f'{lab}_{stat}_missing_flag'
    for lab in CANONICAL_LABS
    for stat in ['mean','min','max','last','trend']
]


def _lab_row(lab: str, value: float,
             aki_pre: bool = False) -> dict:
    """
    Generate all feature columns for one lab in one 6h window.
    Columns: mean, min, max, last, trend, n, missing, alert,
             plus missing_flag columns.
    """
    # Measurement probability — labs not always taken every window
    p_measured = 0.55 if lab in ['creatinine','sodium','potassium'] else 0.30
    measured   = RNG.random() < p_measured

    row = {}
    if measured:
        noise = RNG.normal(0, abs(value) * 0.04)
        v     = value + noise
        lo, hi, _ = LAB_NORMALS[lab]
        alert = 1 if (v < lo or v > hi) else 0
        n_obs = int(RNG.choice([1, 2, 3], p=[0.6, 0.3, 0.1]))
        trend = RNG.normal(0, abs(value) * 0.02)
        row[f'{lab}_mean']    = round(v, 3)
        row[f'{lab}_min']     = round(v - abs(noise)*0.3, 3)
        row[f'{lab}_max']     = round(v + abs(noise)*0.3, 3)
        row[f'{lab}_last']    = round(v + RNG.normal(0, abs(value)*0.02), 3)
        row[f'{lab}_trend']   = round(trend, 4)
        row[f'{lab}_n']       = n_obs
        row[f'{lab}_missing'] = 0
        row[f'{lab}_alert']   = alert
        # Missing flags = 0 (value present)
        for stat in ['mean','min','max','last','trend']:
            row[f'{lab}_{stat}_missing_flag'] = 0
    else:
        # Not measured this window
        row[f'{lab}_mean']    = 0.0
        row[f'{lab}_min']     = 0.0
        row[f'{lab}_max']     = 0.0
        row[f'{lab}_last']    = 0.0
        row[f'{lab}_trend']   = 0.0
        row[f'{lab}_n']       = 0
        row[f'{lab}_missing'] = 1
        row[f'{lab}_alert']   = 0
        for stat in ['mean','min','max','last','trend']:
            row[f'{lab}_{stat}_missing_flag'] = 1

    return row


def _static_row(patient: dict) -> dict:
    """One-hot encode all static features."""
    row = {}

    # Diagnoses
    for col in DIAG_COLS:
        row[col] = 1 if col in patient.get('diagnoses', []) else 0

    # Age
    row['age_years']     = patient['age_years']
    row['age_neonate']   = 1 if patient['age_band'] == 'neonate'    else 0
    row['age_infant']    = 1 if patient['age_band'] == 'infant'     else 0
    row['age_toddler']   = 1 if patient['age_band'] == 'toddler'    else 0
    row['age_child']     = 1 if patient['age_band'] == 'child'      else 0
    row['age_adolescent']= 1 if patient['age_band'] == 'adolescent' else 0

    # Sex (only sex_male in feature_cols — no sex_female)
    row['sex_male'] = 1 if patient['sex'] == 'male' else 0

    # Ethnicity
    for col in [
        'ethnicity_white','ethnicity_black','ethnicity_asian',
        'ethnicity_mixed','ethnicity_other','ethnicity_not_available',
    ]:
        key = col.replace('ethnicity_', '')
        row[col] = 1 if key == patient['ethnicity'] else 0

    # Procedures (static — same every step)
    for col in PROC_COLS:
        row[col] = 1 if col in patient.get('procedures', []) else 0

    return row


def generate_patient(
    patient_id:   str,
    n_steps:      int,
    trajectory:   str,
    age_years:    float,
    sex:          str,
    cardiac:      bool,
    nephrotoxins: list,
) -> tuple[pd.DataFrame, dict]:
    """
    Generate a synthetic patient as a sequence of 6h windows
    with exactly 633 feature columns matching feature_cols.csv.
    """
    # Age band
    if age_years < 0.083:    age_band = 'neonate'
    elif age_years < 1:      age_band = 'infant'
    elif age_years < 3:      age_band = 'toddler'
    elif age_years < 12:     age_band = 'child'
    else:                    age_band = 'adolescent'

    # Baseline creatinine (age-adjusted)
    if age_years < 1:    base_cr = RNG.uniform(18, 38)
    elif age_years < 5:  base_cr = RNG.uniform(25, 42)
    elif age_years < 12: base_cr = RNG.uniform(32, 58)
    else:                base_cr = RNG.uniform(48, 78)

    # Diagnoses
    diagnoses = []
    if cardiac:
        diagnoses += ['diag_I', 'diag_Q']
    if RNG.random() > 0.5:
        diagnoses.append(RNG.choice(['diag_J','diag_E','diag_D','diag_Z']))

    # Procedures
    procedures = list(PROC_COLS[:0])
    if cardiac:
        procedures += ['proc_cardiac_surgery', 'proc_bypass_circuit']
    if RNG.random() > 0.45:
        procedures.append('proc_mechanical_vent')
    if RNG.random() > 0.55:
        procedures.append('proc_central_line')

    patient_meta = {
        'patient_id': patient_id,
        'age_years':  age_years,
        'age_band':   age_band,
        'sex':        sex,
        'ethnicity':  RNG.choice(
            ['white','black','asian','mixed','other','not_available'],
            p=[0.45,0.20,0.15,0.08,0.07,0.05]
        ),
        'diagnoses':  diagnoses,
        'procedures': procedures,
        'trajectory': trajectory,
        'has_aki':    'aki' in trajectory,
        'cardiac':    cardiac,
    }

    # AKI onset step
    if trajectory == 'aki_gradual':
        onset_step = int(n_steps * RNG.uniform(0.55, 0.75))
        cr_mult    = 1.04
    elif trajectory == 'aki_moderate':
        onset_step = int(n_steps * RNG.uniform(0.50, 0.70))
        cr_mult    = 1.07
    elif trajectory == 'aki_severe':
        onset_step = int(n_steps * RNG.uniform(0.40, 0.60))
        cr_mult    = 1.12
    elif trajectory == 'aki_late':
        onset_step = int(n_steps * RNG.uniform(0.75, 0.90))
        cr_mult    = 1.05
    elif trajectory == 'borderline':
        onset_step = n_steps + 1
        cr_mult    = 1.015
    else:
        onset_step = n_steps + 1
        cr_mult    = 1.00

    static = _static_row(patient_meta)
    rows   = []
    creatinine = base_cr

    for step in range(n_steps):
        if step >= onset_step:
            break

        hours_since_admit = step * 6.0
        aki_pre           = 'aki' in trajectory and step >= onset_step - 5

        # Creatinine trajectory
        if aki_pre:
            creatinine *= cr_mult * RNG.uniform(0.99, 1.01)
        elif trajectory == 'borderline' and step > n_steps // 3:
            creatinine *= RNG.uniform(1.00, 1.01)
        else:
            creatinine *= RNG.uniform(0.99, 1.01)
        creatinine = max(creatinine, 10)

        # Label
        steps_to_onset = onset_step - step
        label          = 1 if 0 < steps_to_onset <= 4 else 0
        h2o            = steps_to_onset * 6.0 if 'aki' in trajectory else np.nan

        row = {
            'patient_id':        patient_id,
            'step_index':        step,
            'hours_since_admit': hours_since_admit,
            'label':             label,
            'hours_to_onset':    h2o,
            'aki_stage':         0,
            'trajectory':        trajectory,
            'has_aki':           patient_meta['has_aki'],
        }

        # Static features (same every step)
        row.update(static)

        # Lab features
        for lab in CANONICAL_LABS:
            lo, hi, _ = LAB_NORMALS[lab]
            base_val   = RNG.uniform(lo, hi)

            if lab == 'creatinine':
                base_val = creatinine
            elif aki_pre:
                # Worsen relevant labs
                if lab in ['ph','bicarbonate','base_excess']:
                    base_val = lo - RNG.uniform(0, (hi-lo)*0.4)
                elif lab in ['potassium','chloride']:
                    base_val = hi + RNG.uniform(0, (hi-lo)*0.3)
                elif lab in ['lactate','crp','troponin']:
                    base_val = hi + RNG.uniform(0, hi*0.6)
                elif lab in ['po2','spo2','tao2']:
                    base_val = lo - RNG.uniform(0, (hi-lo)*0.4)
                elif lab in ['urea','egfr']:
                    if lab == 'urea':
                        base_val = hi + RNG.uniform(0, hi*0.4)
                    else:
                        base_val = lo - RNG.uniform(0, lo*0.3)

            row.update(_lab_row(lab, base_val, aki_pre))

        # Nephrotoxins
        active_drugs = set(nephrotoxins)
        for col in NEPHROTOXIN_BINARY:
            drug = col.replace('nephrotoxin_', '')
            if col == 'nephrotoxin_any':
                row[col] = 1 if any(
                    row.get(f'nephrotoxin_{d}', 0) for d in
                    [c.replace('nephrotoxin_','') for c in NEPHROTOXIN_BINARY
                     if c != 'nephrotoxin_any']
                ) else 0
            else:
                row[col] = 1 if drug in active_drugs else int(RNG.random() < 0.03)

        # Nephrotoxin counts
        for col in NEPHROTOXIN_BINARY:
            if col == 'nephrotoxin_any':
                continue
            drug = col.replace('nephrotoxin_', '')
            is_active = drug in active_drugs or row.get(col, 0)
            row[f'nephrotoxin_{drug}_count'] = (
                int(RNG.integers(1, 4)) if is_active else 0
            )
        row['nephrotoxin_total_count'] = sum(
            row.get(f'{c}_count', 0) for c in NEPHROTOXIN_BINARY
            if c != 'nephrotoxin_any'
        )

        # Drug classes
        p_drug = 0.65 if cardiac else 0.35
        for col in DRUG_CLASS_COLS:
            row[col] = int(RNG.random() < p_drug)

        # Summary counts
        row['n_unique_drugs'] = int(
            sum(row[c] for c in DRUG_CLASS_COLS) +
            sum(row[c] for c in NEPHROTOXIN_BINARY if c != 'nephrotoxin_any')
        )
        row['n_procedures'] = len(procedures)

        rows.append(row)

    df = pd.DataFrame(rows)
    return df, patient_meta


def generate_demo_cohort(n_patients: int = 30,
                          seed: int = 42) -> tuple:
    """Generate 30 synthetic PICU patients."""
    global RNG
    RNG = np.random.default_rng(seed)

    specs = [
        # (pid, n_steps, trajectory, cardiac, age, sex, nephrotoxins)
        ('P001', 20, 'aki_severe',   True,  7.2,  'male',   ['vancomycin']),
        ('P002', 16, 'aki_moderate', True,  5.1,  'female', []),
        ('P003', 24, 'aki_gradual',  False, 9.3,  'male',   ['gentamicin','vancomycin']),
        ('P004', 12, 'aki_severe',   True,  4.5,  'female', ['contrast']),
        ('P005', 18, 'aki_late',     False, 14.2, 'male',   []),
        ('P006', 10, 'aki_moderate', False, 2.1,  'female', ['tacrolimus']),
        ('P007', 22, 'aki_gradual',  True,  11.0, 'male',   []),
        ('P008', 14, 'aki_severe',   True,  8.7,  'female', ['vancomycin','contrast']),
        ('P009', 16, 'aki_moderate', False, 13.5, 'male',   ['ibuprofen']),
        ('P010', 20, 'aki_late',     True,  6.0,  'female', []),
        ('P011', 18, 'borderline',   True,  7.8,  'male',   ['vancomycin']),
        ('P012', 14, 'borderline',   False, 12.1, 'female', []),
        ('P013', 22, 'borderline',   True,  5.5,  'male',   ['gentamicin']),
        ('P014', 16, 'borderline',   False, 10.2, 'female', []),
        ('P015', 12, 'borderline',   False, 3.3,  'male',   []),
        ('P016', 24, 'stable',       False, 15.0, 'female', []),
        ('P017', 16, 'stable',       False, 8.0,  'male',   []),
        ('P018', 20, 'stable',       False, 11.5, 'female', []),
        ('P019', 14, 'stable',       True,  6.2,  'male',   []),
        ('P020', 18, 'stable',       False, 14.0, 'female', []),
        ('P021', 12, 'stable',       False, 4.5,  'male',   []),
        ('P022', 22, 'stable',       True,  9.0,  'female', []),
        ('P023', 16, 'stable',       False, 7.3,  'male',   []),
        ('P024', 20, 'stable',       False, 12.8, 'female', []),
        ('P025', 14, 'stable',       False, 5.1,  'male',   []),
        ('P026', 18, 'stable',       True,  10.5, 'female', ['vancomycin']),
        ('P027', 16, 'stable',       False, 8.9,  'male',   []),
        ('P028', 12, 'stable',       False, 2.8,  'female', []),
        ('P029', 22, 'stable',       True,  16.0, 'male',   []),
        ('P030', 20, 'stable',       False, 13.2, 'female', []),
    ][:n_patients]

    all_windows, all_meta = [], []

    for pid, n_steps, traj, cardiac, age, sex, drugs in specs:
        windows_df, meta = generate_patient(
            patient_id   = pid,
            n_steps      = n_steps,
            trajectory   = traj,
            age_years    = age,
            sex          = sex,
            cardiac      = cardiac,
            nephrotoxins = drugs,
        )
        all_windows.append(windows_df)
        all_meta.append({
            'patient_id':  pid,
            'name':        _fake_name(pid),
            'dob':         _fake_dob(age),
            'nhs_number':  _fake_nhs(pid),
            'trajectory':  traj,
            'has_aki':     'aki' in traj,
            'cardiac':     cardiac,
            'age_years':   age,
            'sex':         sex,
            'nephrotoxins':drugs,
            'n_steps':     len(windows_df),
        })

    windows_df = pd.concat(all_windows, ignore_index=True)
    meta_df    = pd.DataFrame(all_meta)

    print(f"Generated {len(meta_df)} patients, "
          f"{len(windows_df)} windows, "
          f"{windows_df.shape[1]} columns")
    return windows_df, meta_df


# ── Fake identifiers ─────────────────────────────────────

_FIRST = ['Oliver','Amelia','George','Isla','Harry','Poppy','Noah','Ava',
           'Jack','Lily','Charlie','Grace','Freddie','Sophia','Archie','Emily',
           'Leo','Ella','Oscar','Scarlett','Henry','Mia','Theo','Chloe',
           'Alfie','Layla','Arthur','Isabella','Tommy','Evie']
_LAST  = ['Smith','Jones','Williams','Taylor','Brown','Davies','Evans','Wilson',
           'Thomas','Roberts','Johnson','Walker','Wright','Robinson','Thompson',
           'White','Hughes','Edwards','Green','Hall']


def _fake_name(pid):
    i = int(pid[1:]) - 1
    return f'{_FIRST[i % len(_FIRST)]} {_LAST[i % len(_LAST)]}'

def _fake_dob(age):
    import datetime
    today = datetime.date(2024, 6, 1)
    return (today - datetime.timedelta(days=int(age*365.25))).strftime('%d/%m/%Y')

def _fake_nhs(pid):
    i = int(pid[1:])
    return f'999 {i:03d} {(i*7)%1000:04d}'