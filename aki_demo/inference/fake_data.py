"""
src/fake_data.py

Generate synthetic PICU patient data with the same feature-engineered
columns as the GOSH AKI prediction pipeline.

Each patient has:
- A sequence of 6-hour windows
- Lab values, medication flags, procedure flags, demographics
- A realistic trajectory (AKI, non-AKI, or borderline/amber)

This data is entirely synthetic and does not contain any real patient information.
"""

import numpy as np
import pandas as pd
from pathlib import Path


# ── Seed for reproducibility ──────────────────────────────
RNG = np.random.default_rng(42)


# ── Feature column names (must match your trained model) ─────────────
# These are loaded from feature_cols.csv at runtime.
# The generator below produces values for all of these.

LAB_STATS = ['mean', 'min', 'max', 'last', 'trend', 'n']

CANONICAL_LABS = [
    'creatinine', 'urea', 'potassium', 'sodium', 'bicarbonate',
    'ph', 'lactate', 'albumin', 'glucose', 'haemoglobin',
    'wbc', 'platelets', 'bilirubin', 'chloride',
    'po2', 'pco2', 'spo2', 'fio2', 'tao2',
    'hct', 'mchc', 'mcv', 'rdw',
    'anion_gap', 'ast', 'alt',
]

NEPHROTOXIN_COLS = [
    'nephrotoxin_vancomycin',
    'nephrotoxin_gentamicin',
    'nephrotoxin_tacrolimus',
    'nephrotoxin_ciclosporin',
    'nephrotoxin_contrast',
    'nephrotoxin_nsaid',
    'nephrotoxin_acyclovir',
]

DRUG_CLASS_COLS = [
    'drug_class_diuretic',
    'drug_class_vasopressor',
    'drug_class_antibiotic',
    'drug_class_antifungal',
    'drug_class_immunosuppressant',
    'drug_class_analgesic',
    'drug_class_sedative',
]

PROC_COLS = [
    'proc_cardiac_surgery',
    'proc_bypass',
    'proc_dialysis_rrt',
    'proc_mechanical_ventilation',
    'proc_central_line',
    'proc_arterial_line',
    'proc_ecmo',
    'proc_orthopaedic',
]

DIAG_COLS = [
    'diag_A', 'diag_C', 'diag_D', 'diag_E',
    'diag_I', 'diag_J', 'diag_K', 'diag_N',
    'diag_Q', 'diag_R', 'diag_Z',
]

AGE_BAND_COLS = [
    'age_neonate', 'age_infant', 'age_toddler',
    'age_child', 'age_adolescent',
]

SEX_COLS   = ['sex_male', 'sex_female']
ETHN_COLS  = [
    'ethnicity_white', 'ethnicity_black', 'ethnicity_asian',
    'ethnicity_mixed', 'ethnicity_other', 'ethnicity_not_available',
]


# ── Normal ranges (paediatric, approximate) ──────────────
LAB_NORMALS = {
    'creatinine': (30,  70,   'µmol/L'),   # age-dependent, using child range
    'urea':       (2.5, 7.0,  'mmol/L'),
    'potassium':  (3.5, 5.0,  'mmol/L'),
    'sodium':     (135, 145,  'mmol/L'),
    'bicarbonate':(22,  29,   'mmol/L'),
    'ph':         (7.35,7.45, ''),
    'lactate':    (0.5, 2.0,  'mmol/L'),
    'albumin':    (35,  50,   'g/L'),
    'glucose':    (3.5, 7.0,  'mmol/L'),
    'haemoglobin':(100, 145,  'g/L'),
    'wbc':        (4.0, 12.0, 'x10⁹/L'),
    'platelets':  (150, 400,  'x10⁹/L'),
    'bilirubin':  (3,   21,   'µmol/L'),
    'chloride':   (98,  107,  'mmol/L'),
    'po2':        (10,  13,   'kPa'),
    'pco2':       (4.5, 6.0,  'kPa'),
    'spo2':       (95,  100,  '%'),
    'fio2':       (0.21,0.40, ''),
    'tao2':       (8,   12,   'mL/dL'),
    'hct':        (0.33,0.43, ''),
    'mchc':       (310, 360,  'g/L'),
    'mcv':        (70,  90,   'fL'),
    'rdw':        (11,  15,   '%'),
    'anion_gap':  (8,   16,   'mmol/L'),
    'ast':        (10,  40,   'U/L'),
    'alt':        (7,   45,   'U/L'),
}

# AKI trajectory: creatinine multiplier per step
AKI_CREATININE_RISE = {
    'gradual':  1.04,   # slow rise — Stage 1
    'moderate': 1.07,   # faster rise — Stage 2
    'severe':   1.12,   # rapid rise — Stage 3
    'stable':   1.00,   # no rise
    'dip_rise': None,   # dips then rises (tricky case)
}


def _lab_row(lab: str, value: float,
              noise_scale: float = 0.05) -> dict:
    """Generate one row of lab stats for a single lab in one window."""
    noise  = RNG.normal(0, abs(value) * noise_scale)
    trend  = RNG.normal(0, abs(value) * 0.03)
    n_obs  = int(RNG.choice([0, 0, 1, 1, 2, 3], p=[0.25,0.15,0.25,0.15,0.15,0.05]))
    row    = {}
    for stat in LAB_STATS:
        if stat == 'mean':
            row[f'{lab}_mean']  = round(value + noise, 3)
        elif stat == 'min':
            row[f'{lab}_min']   = round(value + noise - abs(noise)*0.5, 3)
        elif stat == 'max':
            row[f'{lab}_max']   = round(value + noise + abs(noise)*0.5, 3)
        elif stat == 'last':
            row[f'{lab}_last']  = round(value + noise*0.5, 3)
        elif stat == 'trend':
            row[f'{lab}_trend'] = round(trend, 4)
        elif stat == 'n':
            row[f'{lab}_n']     = n_obs
    return row


def _static_features(patient: dict) -> dict:
    """One-hot encode static features."""
    row = {}
    # Age band
    for col in AGE_BAND_COLS:
        row[col] = 1 if col == f'age_{patient["age_band"]}' else 0
    # Sex
    row['sex_male']   = 1 if patient['sex'] == 'male'   else 0
    row['sex_female'] = 1 if patient['sex'] == 'female' else 0
    # Ethnicity
    for col in ETHN_COLS:
        key = col.replace('ethnicity_', '')
        row[col] = 1 if key == patient['ethnicity'] else 0
    # Diagnoses
    for col in DIAG_COLS:
        row[col] = 1 if col in patient.get('diagnoses', []) else 0
    # Procedures
    for col in PROC_COLS:
        row[col] = 1 if col in patient.get('procedures', []) else 0
    return row


def generate_patient(
    patient_id:   str,
    n_steps:      int,
    trajectory:   str = 'stable',
    age_years:    float = None,
    sex:          str   = None,
    cardiac:      bool  = False,
    nephrotoxins: list  = None,
) -> pd.DataFrame:
    """
    Generate a synthetic patient admission as a sequence of 6h windows.

    trajectory : 'aki_gradual' | 'aki_moderate' | 'aki_severe'
                 | 'stable' | 'borderline' | 'aki_late'

    Returns DataFrame with one row per 6h step.
    """
    if age_years is None:
        age_years = float(RNG.uniform(0.5, 17))
    if sex is None:
        sex = RNG.choice(['male', 'female'])
    if nephrotoxins is None:
        nephrotoxins = []

    # Age band
    if age_years < 0.083:      age_band = 'neonate'
    elif age_years < 1:        age_band = 'infant'
    elif age_years < 3:        age_band = 'toddler'
    elif age_years < 12:       age_band = 'child'
    else:                      age_band = 'adolescent'

    # Creatinine baseline (age-adjusted)
    if age_years < 1:     base_cr = RNG.uniform(20, 40)
    elif age_years < 5:   base_cr = RNG.uniform(25, 45)
    elif age_years < 12:  base_cr = RNG.uniform(35, 60)
    else:                 base_cr = RNG.uniform(50, 80)

    # Diagnoses
    diagnoses = []
    if cardiac:
        diagnoses.append('diag_I')   # circulatory
        diagnoses.append('diag_Q')   # congenital
    if RNG.random() > 0.6:
        diagnoses.append(RNG.choice(['diag_J','diag_E','diag_D','diag_Z']))

    # Procedures
    procedures = []
    if cardiac:
        procedures.extend(['proc_cardiac_surgery', 'proc_bypass'])
    if RNG.random() > 0.5:
        procedures.append('proc_mechanical_ventilation')
    if RNG.random() > 0.7:
        procedures.append('proc_central_line')

    patient_meta = {
        'patient_id':  patient_id,
        'age_years':   age_years,
        'age_band':    age_band,
        'sex':         sex,
        'ethnicity':   RNG.choice(
            ['white','black','asian','mixed','other','not_available'],
            p=[0.45,0.20,0.15,0.08,0.07,0.05]
        ),
        'diagnoses':   diagnoses,
        'procedures':  procedures,
        'trajectory':  trajectory,
        'has_aki':     'aki' in trajectory,
        'cardiac':     cardiac,
    }

    # ── Determine AKI onset step ──────────────────────────
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
        cr_mult    = 1.06
    elif trajectory == 'borderline':
        onset_step = n_steps + 1   # no onset — high risk but no AKI
        cr_mult    = 1.02
    else:   # stable
        onset_step = n_steps + 1
        cr_mult    = 1.00

    rows = []
    creatinine = base_cr

    for step in range(n_steps):
        hours_since_admit = step * 6.0

        # Stop generating after onset (matches window_builder)
        if step >= onset_step:
            break

        # ── Creatinine trajectory ─────────────────────────
        if 'aki' in trajectory and step >= onset_step - 4:
            # Rising in the pre-onset window
            creatinine *= cr_mult * RNG.uniform(0.98, 1.02)
        elif trajectory == 'borderline' and step > n_steps // 2:
            creatinine *= RNG.uniform(1.00, 1.015)
        else:
            creatinine *= RNG.uniform(0.99, 1.01)  # small random walk

        creatinine = max(creatinine, 10)

        # ── Label ─────────────────────────────────────────
        # 1 if onset is within the next 24h (4 steps)
        steps_to_onset = onset_step - step
        label = 1 if 0 < steps_to_onset <= 4 else 0
        hours_to_onset = steps_to_onset * 6.0 if 'aki' in trajectory else np.nan

        row = {
            'patient_id':        patient_id,
            'step_index':        step,
            'hours_since_admit': hours_since_admit,
            'label':             label,
            'hours_to_onset':    hours_to_onset if 'aki' in trajectory else np.nan,
            'aki_stage':         0,
            'trajectory':        trajectory,
            'has_aki':           patient_meta['has_aki'],
        }

        # ── Lab features ──────────────────────────────────
        # Creatinine
        row.update(_lab_row('creatinine', creatinine, 0.04))

        # Other labs — perturb normals
        for lab in CANONICAL_LABS:
            if lab == 'creatinine':
                continue
            lo, hi, _ = LAB_NORMALS[lab]
            base_val   = RNG.uniform(lo, hi)

            # Worsen some labs in AKI trajectory pre-onset
            if 'aki' in trajectory and step >= onset_step - 6:
                if lab in ['ph', 'bicarbonate']:
                    base_val = lo - RNG.uniform(0, (hi-lo)*0.3)
                elif lab in ['potassium']:
                    base_val = hi + RNG.uniform(0, (hi-lo)*0.2)
                elif lab in ['chloride']:
                    base_val = hi + RNG.uniform(0, (hi-lo)*0.3)
                elif lab in ['lactate']:
                    base_val = hi + RNG.uniform(0, hi*0.5)
                elif lab in ['po2', 'spo2', 'tao2']:
                    base_val = lo - RNG.uniform(0, (hi-lo)*0.4)
                elif lab in ['urea']:
                    base_val = hi + RNG.uniform(0, hi*0.3)

            row.update(_lab_row(lab, base_val))

        # ── Medications ───────────────────────────────────
        for col in NEPHROTOXIN_COLS:
            drug = col.replace('nephrotoxin_', '')
            row[col] = 1 if drug in nephrotoxins else int(RNG.random() < 0.05)

        for col in DRUG_CLASS_COLS:
            p_drug = 0.6 if cardiac else 0.3
            row[col] = int(RNG.random() < p_drug)

        # ── Static features ───────────────────────────────
        row.update(_static_features(patient_meta))

        rows.append(row)

    df              = pd.DataFrame(rows)
    df['patient_id']= patient_id
    df['age_years'] = age_years
    df['sex']       = sex
    df['age_band']  = age_band
    df['cardiac']   = int(cardiac)
    df['onset_step']= onset_step if 'aki' in trajectory else -1

    return df, patient_meta


def generate_demo_cohort(n_patients: int = 30,
                          seed: int = 42) -> tuple:
    """
    Generate a cohort of synthetic PICU patients.
    Mix of AKI and non-AKI cases with varying trajectories.

    Returns (all_windows_df, patients_meta_df)
    """
    global RNG
    RNG = np.random.default_rng(seed)

    # Define patient mix
    patient_specs = [
        # AKI cases — various severity and timing
        ('P001', 20, 'aki_severe',   True,  7.2, 'male',   ['vancomycin']),
        ('P002', 16, 'aki_moderate', True,  5.1, 'female', []),
        ('P003', 24, 'aki_gradual',  False, 9.3, 'male',   ['gentamicin', 'vancomycin']),
        ('P004', 12, 'aki_severe',   True,  4.5, 'female', ['contrast']),
        ('P005', 18, 'aki_late',     False, 14.2,'male',   []),
        ('P006', 10, 'aki_moderate', False, 2.1, 'female', ['tacrolimus']),
        ('P007', 22, 'aki_gradual',  True,  11.0,'male',   []),
        ('P008', 14, 'aki_severe',   True,  8.7, 'female', ['vancomycin','contrast']),
        ('P009', 16, 'aki_moderate', False, 13.5,'male',   ['nsaid']),
        ('P010', 20, 'aki_late',     True,  6.0, 'female', []),
        # Borderline — amber zone
        ('P011', 18, 'borderline',   True,  7.8, 'male',   ['vancomycin']),
        ('P012', 14, 'borderline',   False, 12.1,'female', []),
        ('P013', 22, 'borderline',   True,  5.5, 'male',   ['gentamicin']),
        ('P014', 16, 'borderline',   False, 10.2,'female', []),
        ('P015', 12, 'borderline',   False, 3.3, 'male',   []),
        # Stable non-AKI
        ('P016', 24, 'stable',       False, 15.0,'female', []),
        ('P017', 16, 'stable',       False, 8.0, 'male',   []),
        ('P018', 20, 'stable',       False, 11.5,'female', []),
        ('P019', 14, 'stable',       True,  6.2, 'male',   []),
        ('P020', 18, 'stable',       False, 14.0,'female', []),
        ('P021', 12, 'stable',       False, 4.5, 'male',   []),
        ('P022', 22, 'stable',       True,  9.0, 'female', []),
        ('P023', 16, 'stable',       False, 7.3, 'male',   []),
        ('P024', 20, 'stable',       False, 12.8,'female', []),
        ('P025', 14, 'stable',       False, 5.1, 'male',   []),
        ('P026', 18, 'stable',       True,  10.5,'female', ['vancomycin']),
        ('P027', 16, 'stable',       False, 8.9, 'male',   []),
        ('P028', 12, 'stable',       False, 2.8, 'female', []),
        ('P029', 22, 'stable',       True,  16.0,'male',   []),
        ('P030', 20, 'stable',       False, 13.2,'female', []),
    ]

    # Trim to n_patients
    patient_specs = patient_specs[:n_patients]

    all_windows = []
    all_meta    = []

    for pid, n_steps, traj, cardiac, age, sex, drugs in patient_specs:
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
            'trajectory':  traj,
            'has_aki':     'aki' in traj,
            'cardiac':     cardiac,
            'age_years':   age,
            'sex':         sex,
            'nephrotoxins':drugs,
            'n_steps':     len(windows_df),
            'name':        _fake_name(pid),
            'dob':         _fake_dob(age),
            'nhs_number':  _fake_nhs(pid),
        })

    windows_df = pd.concat(all_windows, ignore_index=True)
    meta_df    = pd.DataFrame(all_meta)

    print(f"Generated {len(meta_df)} patients, "
          f"{len(windows_df)} total windows")
    print(f"  AKI:        {meta_df['has_aki'].sum()}")
    print(f"  Borderline: {(meta_df['trajectory']=='borderline').sum()}")
    print(f"  Stable:     {(meta_df['trajectory']=='stable').sum()}")

    return windows_df, meta_df


# ── Fake identifiers (clearly synthetic) ─────────────────

_FIRST_NAMES = [
    'Oliver','Amelia','George','Isla','Harry','Poppy','Noah','Ava',
    'Jack','Lily','Charlie','Grace','Freddie','Sophia','Archie','Emily',
    'Leo','Ella','Oscar','Scarlett','Henry','Mia','Theo','Chloe',
    'Alfie','Layla','Arthur','Isabella','Tommy','Evie',
]
_LAST_NAMES = [
    'Smith','Jones','Williams','Taylor','Brown','Davies','Evans','Wilson',
    'Thomas','Roberts','Johnson','Walker','Wright','Robinson','Thompson',
    'White','Hughes','Edwards','Green','Hall',
]


def _fake_name(pid: str) -> str:
    idx   = int(pid[1:]) - 1
    first = _FIRST_NAMES[idx % len(_FIRST_NAMES)]
    last  = _LAST_NAMES[idx % len(_LAST_NAMES)]
    return f'{first} {last}'


def _fake_dob(age_years: float) -> str:
    import datetime
    today     = datetime.date(2024, 6, 1)  # fixed reference date
    days_old  = int(age_years * 365.25)
    dob       = today - datetime.timedelta(days=days_old)
    return dob.strftime('%d/%m/%Y')


def _fake_nhs(pid: str) -> str:
    """Generate a clearly fake NHS number."""
    idx = int(pid[1:])
    return f'999 {idx:03d} {(idx*7)%1000:04d}'


if __name__ == '__main__':
    windows, meta = generate_demo_cohort(30)
    print(windows.head())
    print(meta[['patient_id','name','trajectory','age_years','sex']].to_string())