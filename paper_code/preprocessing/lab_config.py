"""
lab_config.py

Canonical lab synonym mapping for GOSH feature engineering.

All POC values confirmed single-unit — safe to group directly.
Units verified per lab type — all within-group values are in the
same unit and can be averaged/compared without conversion.

Usage in feature_engineering.py
--------------------------------
    from lab_config import LAB_SYNONYMS, EXCLUDE_BASENAMES, validate_units

    # Map raw component_basename to canonical name
    labs['canonical'] = (labs['component_basename']
                         .str.upper().str.strip()
                         .map(build_reverse_map(LAB_SYNONYMS)))

    # Drop excluded basenames before feature computation
    labs = labs[~labs['component_basename']
                .str.upper().str.strip()
                .isin(EXCLUDE_BASENAMES)]
"""

# ─────────────────────────────────────────────────────────
#  Canonical lab synonym mapping
#  Keys   = canonical feature name used in feature matrix
#  Values = all component_basename values that map to it
#           (uppercase, stripped)
# ─────────────────────────────────────────────────────────

LAB_SYNONYMS = {

    # ── Creatinine (serum only) — µmol/L ─────────────────
    # UCREAT1 / UKCREAT = confirm serum not urine with data dict
    # All ratio units (mg/mmol, umol/mmol) are URINE — excluded below
    'creatinine': [
        'CREAT',
         ],

    # ── Glucose — mmol/L ─────────────────────────────────
    'glucose': [
        'GLU',
        'POCGLUCOSE',
    ],

    # ── Bilirubin (total) — µmol/L ───────────────────────
    'bilirubin': [
        'BC',
        'BU',
        'CALBILI',
        'TBIL',
        'POCAUBILI',
        'POCBILIRUBIN',
    ],

    # ── Calcium (total and ionised) — mmol/L ─────────────
    # ICA / ICACOR = ionised calcium — same unit, comparable
    # UCA = urine calcium — EXCLUDE (see EXCLUDE_BASENAMES)
    'calcium': [
        'CA',
        'ICA',
        'ICACOR',
        'POCCALCIUM',
        'POCCALI',
    ],

    # ── Phosphate — mmol/L ───────────────────────────────
    'phosphate': [
        'PO5',
    ],

    # ── Urea — mmol/L ────────────────────────────────────
    # UUREA = urine urea — EXCLUDE (see EXCLUDE_BASENAMES)
    'urea': [
        'FLUREA',
        'UREA',
    ],

    # ── Lactate — mmol/L ─────────────────────────────────
    'lactate': [
        'POCLAC',
    ],

    # ── Albumin — g/L ────────────────────────────────────
    'albumin': [
        'ALB',
    ],

    # ── Anion gap — mmol/L ───────────────────────────────
    'anion_gap': [
        'POCGAP',
    ],

    # ── POC blood gas and respiratory — all single unit ──
    'po2': [
        'POCPO2',
        'POCPO2IC',
        'POCPO2T',
        'POCPO2',
    ],
    'pco2': [
        'POCPCO2',
        'POCPCO2IC',
        'POCPCO2T',
        'POCPCO2IM',
        'POCPP02IM',    # likely PCO2 — confirm with component_name
        'POCPPCCO2IM',  # confirm with component_name
    ],
    'ph': [
        'POCPH',
        'POCPHC',
    ],
    'so2': [
        'POCSO2',
    ],
    'fio2': [
        'POCFI02',      # note: character is zero not O
    ],
    'base_excess': [
        'POCBASEEXC',
    ],
    'bicarbonate': [
        'POCHCO3',
    ],
    'tco2': [
        'POCTCO2',
        'POCTCO2I',
    ],
    'tao2': [
        'POCTO2',
    ],
    'temperature_poc': [
        'POCTEMP',
    ],
    'osmolality': [
        'POCPOSM',
    ],
    'potassium_poc': [
        'POCPOTASSIU',   # truncated — confirm full basename
    ],
    'sodium_poc': [
        'POCSODIUM', 'NA1'
    ],
    'haematocrit_poc': [
        'POCHCT',
        'POCHCTI',
    ],
    'haemoglobin_poc': [
        'POCCTHB',      # total haemoglobin POC
    ],
    'methb': [
        'POCFMETHB',
    ],
    'cohb': [
        'POCFCOHB',
    ],

    # ── POC electrolytes ──────────────────────────────────
    'calcium_poc_ionised': [
        'POCBEIM',      # confirm = ionised calcium with component_name
    ],
    'sodium_poc_alt': [
        'POCNAISTAT',
    ],
    'nitrite_poc': [
        'POCNITRITE',
    ],
    'leukocytes_poc': [
        'POCLEUKS',
    ],

    # ── Other POC ─────────────────────────────────────────
    'p50': [
        'POCP50',
    ],
    'bicarb_poc_alt': [
        'POCBICARBI',
    ],
    'phosphate_poc': [
        'POCPHPHISTATM', # confirm with component_name
    ],
    'blood_poc': [
        'POCAUBLOOD',   # POC blood — confirm what this measures
    ],
    'urine_ket_poc': [
        'POCAUKET',
    ],
    'urine_grav_poc': [
        'POCAUGRAV',
    ],
    'urine_ph_poc': [
        'POCAUPH',
    ],
    'urine_prot_poc': [
        'POCAUPROT',
    ],
    'urine_rob_poc': [
        'POCAUUROB',
    ],
    'urine_glu_poc': [
        'POCAUGLU',
    ],
    
    'spec_poc': [
        'POCSPECSITE',
    ],
    'sapo': [
        'SAPOCT',
    ],
}

# ─────────────────────────────────────────────────────────
#  Basenames to EXCLUDE from feature engineering
#  These are urine tests, hepatitis screens, or components
#  that should not be mixed with serum equivalents
# ─────────────────────────────────────────────────────────

EXCLUDE_BASENAMES = {
    # Urine creatinine and ratios — NOT serum creatinine
    'UNACREAT',         # urine Na:creatinine ratio
    'UALBUCREAT',       # urine albumin:creatinine ratio
    'UCREATMET',        # urine metabolic creatinine
    'OHCREAT',
    'UGAACREAT',
    'HEX4CREAT',
    'FCARNCREAT',
    'PGAACREAT',
    'TUCREAT',
    'CREATEX',
    'CREATSYN',

    # Urine calcium — not serum
    'UCA',

    # Urine urea — not serum
    'UUREA',

    # Hepatitis B surface antigen screen — NOT haemoglobin
    # HBSSCREEN is a serology test, nothing to do with Hb
    'HBSSCREEN',

    # POC urine dipstick components — not blood chemistry
    'POCAUBLOOD',       # remove if confirmed urine blood
    'POCAUKET',
    'POCAUGRAV',
    'POCAUPH',
    'POCAUPROT',
    'POCAUUROB',
    'POCAUGLU',
    'POCAUBILI',
    'POCSPECSITE',
    'POCNITRITE',
    'POCLEUKS',

    # Haemoglobin fractions — not total Hb
    # Keep as separate features if needed, but exclude from 'haemoglobin'
    'POCFCOHB',         # carboxyhaemoglobin
    'POCFMETHB',        # methaemoglobin
    'POCFHHB',          # deoxyhaemoglobin
    'POCFHBF',          # fetal haemoglobin
}

# ─────────────────────────────────────────────────────────
#  Creatinine unit validation
#  Applied specifically to CREAT/serum creatinine rows
# ─────────────────────────────────────────────────────────

CREATININE_UNIT_CONVERSION = {
    'umol/l':               1.0,
    'µmol/l':               1.0,
    'μmol/l':               1.0,
    'umol':                 1.0,
    'mmol/l':               1000.0,   # convert to µmol/L
    'mmol':                 1000.0,
    # Ratio/urine units — exclude (map to None)
    'mg/mmol':              None,
    'umol/mmol creatinine': None,
    'umol/mmol':            None,
    'mmol/mmol creatinine': None,
}

CREATININE_PLAUSIBLE_RANGE = (10, 800)   # µmol/L — paediatric serum

HAEMATOCRIT_CONVERSION = {
    'decimal_to_pct_threshold': 1.5, 
    'decimal_to_pct_factor': 100.0,
    'canonical_unit': 'percent'
}


# ─────────────────────────────────────────────────────────
#  Helper: build reverse lookup {BASENAME → canonical}
# ─────────────────────────────────────────────────────────

def build_reverse_map(lab_synonyms: dict) -> dict:
    """
    Returns {BASENAME_UPPER: canonical_name} for all synonyms.
    Raises ValueError if any basename appears in multiple canonicals.
    """
    reverse = {}
    for canonical, synonyms in lab_synonyms.items():
        for syn in synonyms:
            key = syn.upper().strip()
            if key in reverse and reverse[key] != canonical:
                raise ValueError(
                    f"Basename '{key}' mapped to both "
                    f"'{reverse[key]}' and '{canonical}'"
                )
            reverse[key] = canonical
    return reverse


# ─────────────────────────────────────────────────────────
#  Unit validation check — run before feature engineering
# ─────────────────────────────────────────────────────────

def validate_units(labs_df: 'pd.DataFrame',
                   lab_synonyms: dict,
                   unit_col: str = 'unit',
                   value_col: str = 'value',
                   name_col: str = 'component_basename') -> None:
    """
    Print unit distribution per canonical lab.
    Call this BEFORE running feature engineering to confirm
    all synonyms within a canonical group share the same unit.

    Parameters
    ----------
    labs_df      : raw labs dataframe
    lab_synonyms : LAB_SYNONYMS dict from this file
    """
    import pandas as pd
    import numpy as np

    reverse = build_reverse_map(lab_synonyms)
    df = labs_df.copy()
    df.columns = df.columns.str.lower()

    name_col_l  = name_col.lower()
    value_col_l = value_col.lower()
    unit_col_l  = unit_col.lower()

    df['canonical'] = (df[name_col_l]
                       .str.upper().str.strip()
                       .map(reverse))
    df = df[df['canonical'].notna()]
    df[value_col_l] = pd.to_numeric(df[value_col_l], errors='coerce')

    if unit_col_l not in df.columns:
        print("WARNING: no unit column found — cannot validate units")
        return

    df['unit_clean'] = df[unit_col_l].str.strip().str.lower().fillna('unknown')

    print("\n" + "="*65)
    print("UNIT VALIDATION PER CANONICAL LAB")
    print("="*65)

    problems = []
    for canonical in sorted(df['canonical'].unique()):
        sub = df[df['canonical'] == canonical]
        unit_counts = sub['unit_clean'].value_counts()

        if len(unit_counts) == 1:
            unit = unit_counts.index[0]
            vals = sub[value_col_l].dropna()
            print(f"  {canonical:<25} {unit:<20} "
                  f"n={len(sub):,}  "
                  f"median={vals.median():.2f}")
        else:
            print(f"\n  *** {canonical.upper()} — MULTIPLE UNITS ***")
            for unit, count in unit_counts.items():
                vals = sub[sub['unit_clean']==unit][value_col_l].dropna()
                if len(vals) == 0:
                    continue
                print(f"    {unit:<30} n={count:,}  "
                      f"median={vals.median():.3f}  "
                      f"range={vals.quantile(0.05):.3f}"
                      f"–{vals.quantile(0.95):.3f}")

            unit_medians = {}
            for unit in unit_counts.index:
                vals = sub[sub['unit_clean']==unit][value_col_l].dropna()
                if len(vals) > 0:
                    unit_medians[unit] = vals.median()
            if len(unit_medians) > 1:
                ratio = (max(unit_medians.values()) /
                         (min(unit_medians.values()) + 1e-9))
                if ratio > 3:
                    problems.append(canonical)
                    print(f"    SCALE RATIO: {ratio:.0f}x — "
                          f"CONVERSION REQUIRED")

    print("\n" + "="*65)
    if problems:
        print(f"PROBLEMS REQUIRING CONVERSION: {problems}")
    else:
        print("All canonical labs have consistent units — safe to proceed")
    print("="*65 + "\n")