"""Canonical clinical concepts shared by every site.

Each site adapter maps its own codes (GOSH basenames, PIC/MIMIC ITEMIDs) to
these concept names. Values are converted to ONE canonical unit here, so the
model never sees site-specific units.

Nothing in this file is site-specific. If you add a concept, add it here and
then map it in each site YAML.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Concept:
    name: str
    unit: str                       # canonical unit
    lo: float                       # plausible range in canonical unit
    hi: float
    kind: str                       # 'lab' | 'gas' | 'vital' | 'support'
    carry_h: float                  # max hours a value is carried forward
    conversions: dict = field(default_factory=dict)  # normalised unit -> factor


# ── unit strings are normalised with _norm_unit before lookup ─────────────
_SAME = 1.0
CONCEPTS: dict[str, Concept] = {c.name: c for c in [
    # renal / chemistry (carry 72h)
    Concept("creatinine", "umol/l", 5, 1500, "lab", 72,
            {"umol/l": _SAME, "mg/dl": 88.42, "mmol/l": 1000.0}),
    Concept("urea", "mmol/l", 0.2, 80, "lab", 72,
            {"mmol/l": _SAME, "mg/dl": 0.357}),           # MIMIC BUN mg/dL -> urea mmol/L
    Concept("sodium", "mmol/l", 100, 190, "lab", 72, {"mmol/l": _SAME, "meq/l": _SAME}),
    Concept("potassium", "mmol/l", 1, 10, "lab", 72, {"mmol/l": _SAME, "meq/l": _SAME}),
    Concept("chloride", "mmol/l", 60, 150, "lab", 72, {"mmol/l": _SAME, "meq/l": _SAME}),
    Concept("bicarbonate", "mmol/l", 2, 60, "lab", 72, {"mmol/l": _SAME, "meq/l": _SAME}),
    Concept("glucose", "mmol/l", 0.5, 60, "lab", 24, {"mmol/l": _SAME, "mg/dl": 0.0555}),
    Concept("calcium_total", "mmol/l", 0.5, 5, "lab", 72, {"mmol/l": _SAME, "mg/dl": 0.2495}),
    Concept("calcium_ionised", "mmol/l", 0.2, 3, "gas", 24, {"mmol/l": _SAME, "mg/dl": 0.2495}),
    Concept("phosphate", "mmol/l", 0.1, 8, "lab", 72, {"mmol/l": _SAME, "mg/dl": 0.3229}),
    Concept("magnesium", "mmol/l", 0.1, 5, "lab", 72, {"mmol/l": _SAME, "mg/dl": 0.4114,
                                                       "meq/l": 0.5}),
    Concept("albumin", "g/l", 5, 70, "lab", 72, {"g/l": _SAME, "g/dl": 10.0}),
    Concept("bilirubin_total", "umol/l", 0, 1000, "lab", 72, {"umol/l": _SAME, "mg/dl": 17.1}),
    Concept("alt", "u/l", 1, 10000, "lab", 72, {"u/l": _SAME, "iu/l": _SAME}),
    Concept("crp", "mg/l", 0, 600, "lab", 72, {"mg/l": _SAME, "mg/dl": 10.0}),
    # haematology
    Concept("haemoglobin", "g/l", 20, 250, "lab", 48, {"g/l": _SAME, "g/dl": 10.0}),
    Concept("haematocrit", "%", 5, 80, "lab", 48, {"%": _SAME, "l/l": 100.0}),
    Concept("wbc", "10^9/l", 0, 500, "lab", 72, {"10^9/l": _SAME, "k/ul": _SAME}),
    Concept("platelets", "10^9/l", 0, 2000, "lab", 72, {"10^9/l": _SAME, "k/ul": _SAME}),
    # blood gas (carry 12h)
    Concept("lactate", "mmol/l", 0.1, 40, "gas", 12, {"mmol/l": _SAME, "mg/dl": 0.111}),
    Concept("ph", "", 6.5, 8.0, "gas", 12, {"": _SAME, "units": _SAME}),
    Concept("pco2", "kpa", 1, 30, "gas", 12, {"kpa": _SAME, "mmhg": 0.133322}),
    Concept("po2", "kpa", 1, 80, "gas", 12, {"kpa": _SAME, "mmhg": 0.133322}),
    Concept("base_excess", "mmol/l", -40, 40, "gas", 12, {"mmol/l": _SAME, "meq/l": _SAME}),
    Concept("fio2", "fraction", 0.21, 1.0, "support", 12, {"fraction": _SAME, "%": 0.01}),
    # vital signs (carry 6h; no measurement-count features are built for these)
    Concept("hr", "/min", 20, 300, "vital", 6, {"/min": _SAME, "bpm": _SAME}),
    Concept("rr", "/min", 2, 150, "vital", 6, {"/min": _SAME, "insp/min": _SAME, "bpm": _SAME}),
    Concept("spo2", "%", 30, 100, "vital", 6, {"%": _SAME}),
    Concept("temp", "c", 25, 45, "vital", 12, {"c": _SAME, "f": None}),   # F handled below
    Concept("sbp", "mmhg", 20, 250, "vital", 6, {"mmhg": _SAME}),
    Concept("dbp", "mmhg", 10, 200, "vital", 6, {"mmhg": _SAME}),
    Concept("map", "mmhg", 15, 220, "vital", 6, {"mmhg": _SAME}),
    # organ support (binary, charted)
    Concept("vent_invasive", "flag", 0, 1, "support", 12, {"flag": _SAME}),
]}

LAB_CONCEPTS = [c for c, d in CONCEPTS.items() if d.kind in ("lab", "gas")]
VITAL_CONCEPTS = [c for c, d in CONCEPTS.items() if d.kind == "vital"]
SUPPORT_CONCEPTS = [c for c, d in CONCEPTS.items() if d.kind == "support"]

# Age bands used for one-hot static features and for age-band normalisation
AGE_BANDS_DAYS = [0, 28, 365, 365 * 3, 365 * 12, 365 * 19]
AGE_BAND_NAMES = ["neonate", "infant", "toddler", "child", "adolescent"]
# finer bands for normalising age-dependent physiology
NORM_BANDS_DAYS = [0, 28, 91, 182, 365, 730, 1826, 4383, 6940]


def age_band(age_days, edges=AGE_BANDS_DAYS, names=None):
    names = names or [f"b{i}" for i in range(len(edges) - 1)]
    return pd.cut(np.asarray(age_days, float), bins=edges, labels=names,
                  right=False, include_lowest=True)


# ── units ─────────────────────────────────────────────────────────────────
_UNIT_ALIASES = {
    "µmol/l": "umol/l", "μmol/l": "umol/l", "umol": "umol/l",
    "mmol": "mmol/l", "meq/l": "meq/l", "mequiv/l": "meq/l",
    "x10^9/l": "10^9/l", "10*9/l": "10^9/l", "x10e9/l": "10^9/l", "10e9/l": "10^9/l",
    "k/µl": "k/ul", "k/μl": "k/ul", "thou/ul": "k/ul", "x10^3/ul": "k/ul",
    "mm hg": "mmhg", "mmhg": "mmhg", "kpa": "kpa",
    "beats/min": "bpm", "bpm": "bpm", "breaths/min": "/min", "/min": "/min",
    "°c": "c", "deg c": "c", "degc": "c", "celsius": "c", "°f": "f", "deg f": "f",
    "degf": "f", "fahrenheit": "f", "iu/l": "iu/l", "u/l": "u/l", "units/l": "u/l",
    "percent": "%", "fraction": "fraction", "": "",
}

_COUNT_UNIT = re.compile(r"^x?\s*10\s*(\^|\*|e|\*\*)?\s*9\s*/\s*l$")   # x10*9/L, 10^9/L, 10e9/L ...


def norm_unit(u) -> str:
    if u is None or (isinstance(u, float) and np.isnan(u)):
        return ""
    s = str(u).strip().lower()
    if s in ("nan", "none", "null"):
        return ""
    if s in _UNIT_ALIASES:
        return _UNIT_ALIASES[s]
    s = s.replace("µ", "u").replace("μ", "u")
    if s in _UNIT_ALIASES:
        return _UNIT_ALIASES[s]
    s2 = s.replace(" ", "")
    if _COUNT_UNIT.match(s2):
        return "10^9/l"
    return _UNIT_ALIASES.get(s2, s2)


def parse_values(raw: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Text/number -> float. '<5' -> 5 and '>1000' -> 1000 (flagged as censored),
    '1,234' -> 1234. Anything else non-numeric ('haemolysed', '') -> NaN."""
    s = raw.astype(str).str.strip().str.replace(",", "", regex=False)
    censored = s.str.match(r"^[<>]=?")
    num = pd.to_numeric(s.str.replace(r"^[<>]=?\s*", "", regex=True), errors="coerce")
    return num.astype(float), censored & num.notna()


def _report(report, concept, g, mask, issue, values=None):
    """Append one aggregate row per source code: counts and value centiles only."""
    if report is None or not mask.any():
        return
    sub = g[mask]
    codes = sub["code"] if "code" in sub else pd.Series("", index=sub.index)
    for code, idx in codes.groupby(codes).groups.items():
        row = dict(concept=concept, code=code, issue=issue, n_rows=len(idx))
        if values is not None:
            v = pd.Series(values, index=g.index).loc[idx].dropna()
            if len(v):
                row.update(p01=float(v.quantile(.01)), median=float(v.median()),
                           p99=float(v.quantile(.99)))
        report.append(row)


def to_canonical(df: pd.DataFrame, default_units: dict | None = None,
                 report: list | None = None) -> pd.DataFrame:
    """Convert a long table [concept, value, unit, code] to canonical units.

    Dropped and reported (per concept AND source code, with value centiles):
      'non-numeric value'          blank / free text that cannot be parsed
      "unit 'x' not convertible"   unit unknown for this concept
      'outside plausible ...'      converted value outside the plausible range
    Reported but KEPT: 'censored (<x or >x), bound used'.
    `default_units` (site YAML) supplies the unit only where the row has none.
    """
    default_units = default_units or {}
    out = []
    for concept, g in df.groupby("concept", sort=False):
        if concept not in CONCEPTS:
            _report(report, concept, g, pd.Series(True, index=g.index), "UNKNOWN_CONCEPT")
            continue
        c = CONCEPTS[concept]
        g = g.copy()
        units = g["unit"].map(norm_unit) if "unit" in g else pd.Series("", index=g.index)
        if concept in default_units:
            units = units.where(units != "", norm_unit(default_units[concept]))
        if c.unit == "":
            units = units.where(units.isin(c.conversions), "")
        vals, censored = parse_values(g["value"])
        nonnum = vals.isna()
        _report(report, concept, g, nonnum, "non-numeric value")
        _report(report, concept, g, censored, "censored (<x or >x), bound used (kept)", vals)
        vals = vals.to_numpy()

        # special cases that depend on value as well as unit
        if concept == "temp":
            is_f = units.str.contains("f", regex=False).to_numpy() | (vals > 50)
            vals = np.where(is_f, (vals - 32) * 5 / 9, vals)
            units = pd.Series("c", index=g.index)
        if concept == "haematocrit":
            vals = np.where(vals < 1.5, vals * 100, vals)
            units = pd.Series("%", index=g.index)
        if concept == "fio2":
            vals = np.where(vals > 1.5, vals / 100, vals)
            units = pd.Series("fraction", index=g.index)
        if concept == "spo2":
            units = pd.Series("%", index=g.index)

        factor = units.map(lambda u: c.conversions.get(u, np.nan)).astype(float)
        bad_unit = factor.isna() & ~nonnum
        for u in units[bad_unit].unique():
            _report(report, concept, g, bad_unit & (units == u),
                    f"unit '{u}' not convertible", vals)
        conv = vals * factor.to_numpy()
        implausible = ~nonnum & ~bad_unit & ~pd.Series(conv, index=g.index).between(c.lo, c.hi)
        _report(report, concept, g, implausible,
                f"outside plausible {c.lo}-{c.hi} {c.unit} (values shown converted)", conv)
        keep = ~(nonnum | bad_unit | implausible)
        g["value"] = conv
        g["unit"] = c.unit
        out.append(g[keep.to_numpy()])
    if not out:
        return df.iloc[0:0].assign(value=pd.Series(dtype=float))
    return pd.concat(out, ignore_index=True)
    
# ── medications ───────────────────────────────────────────────────────────
@dataclass(frozen=True)
class MedGroup:
    name: str
    patterns: tuple            # whole-word regex fragments matched on lowercased name
    nephrotoxin: bool
    carry_h: float             # 'on therapy' duration after each dose/interval end
    exclude_routes: tuple = ()  # route classes that do NOT count for this group


NON_SYSTEMIC = ("oral_nonabsorbed", "inhaled", "topical", "ophthalmic", "otic", "lock")

MED_GROUPS: dict[str, MedGroup] = {g.name: g for g in [
    MedGroup("aminoglycoside", (r"gentamicin", r"gentamycin", r"tobramycin", r"amikacin"),
             True, 24, ("inhaled", "topical", "ophthalmic", "otic", "oral", "lock")),
    MedGroup("vancomycin", (r"vancomycin",), True, 24, ("oral", "inhaled", "topical", "lock")),
    MedGroup("polymyxin", (r"colistin", r"colistimethate", r"polymyxin"), True, 24,
             ("inhaled", "topical", "ophthalmic")),
    MedGroup("amphotericin", (r"amphotericin", r"ambisome", r"fungizone", r"abelcet"), True, 24,
             ("topical", "oral", "inhaled")),
    MedGroup("calcineurin_inhibitor", (r"ciclosporin", r"cyclosporin", r"cyclosporine",
                                       r"tacrolimus", r"sandimmun", r"neoral", r"prograf",
                                       r"advagraf", r"adoport"), True, 24,
             ("topical", "ophthalmic")),
    MedGroup("nsaid", (r"ibuprofen", r"indometacin", r"indomethacin", r"ketorolac",
                       r"diclofenac", r"naproxen"), True, 12, ("topical", "ophthalmic")),
    MedGroup("antiviral_nephrotoxic", (r"aciclovir", r"acyclovir", r"ganciclovir", r"foscarnet",
                                       r"cidofovir"), True, 24,
             ("topical", "ophthalmic", "oral")),
    MedGroup("chemo_nephrotoxic", (r"cisplatin", r"carboplatin", r"ifosfamide",
                                   r"methotrexate"), True, 72,
             ("intrathecal", "topical")),
    MedGroup("contrast", (r"iohexol", r"iodixanol", r"iopromide", r"omnipaque", r"visipaque",
                          r"iodinated contrast"), True, 48, ()),
    # non-nephrotoxic but physiologically relevant
    MedGroup("diuretic", (r"furosemide", r"frusemide", r"bumetanide", r"spironolactone",
                          r"hydrochlorothiazide", r"chlorothiazide", r"metolazone",
                          r"torasemide", r"torsemide", r"acetazolamide"), False, 12,
             ("topical", "ophthalmic")),
    MedGroup("acei_arb", (r"captopril", r"enalapril", r"lisinopril", r"ramipril",
                          r"perindopril", r"losartan", r"valsartan", r"candesartan",
                          r"irbesartan"), False, 24, ()),
    MedGroup("vasopressor", (r"noradrenaline", r"norepinephrine", r"adrenaline", r"epinephrine",
                             r"vasopressin", r"terlipressin", r"phenylephrine",
                             r"metaraminol", r"dopamine"), False, 1,
             ("topical", "inhaled", "local")),
    MedGroup("inotrope", (r"dobutamine", r"milrinone", r"levosimendan"), False, 1, ()),
]}

# Compile once: whole-word match so 'adrenaline' does not match 'noradrenaline'
_MED_RE = {name: re.compile(r"\b(?:" + "|".join(g.patterns) + r")\b")
           for name, g in MED_GROUPS.items()}

# keywords ending in '*' match as word PREFIXES (neb* -> nebulised); others are whole words
ROUTE_KEYWORDS = {
    "inhaled": ("neb*", "inhal*", "inh"),
    "topical": ("topical*", "top", "cutaneous", "skin", "cream", "oint*"),
    "ophthalmic": ("eye*", "ophth*", "ocular"),
    "otic": ("ear", "ears", "otic"),
    "intrathecal": ("intrathec*", "it", "intraventric*"),
    "lock": ("lock", "flush*"),
    "oral": ("oral*", "po", "enteral", "ng", "nj", "peg", "gastrostom*", "nasogast*",
             "jejun*", "subling*", "buccal", "tube"),
    "local": ("local", "infiltrat*"),
}
_ROUTE_RE = {cls: re.compile("|".join(
    rf"(?<![a-z]){re.escape(k[:-1])}" if k.endswith("*") else rf"(?<![a-z]){re.escape(k)}(?![a-z])"
    for k in keys)) for cls, keys in ROUTE_KEYWORDS.items()}


def route_class(route) -> str:
    if route is None or (isinstance(route, float) and np.isnan(route)):
        return "unknown"
    r = str(route).lower().strip()
    if not r:
        return "unknown"
    for cls, rx in _ROUTE_RE.items():
        if rx.search(r):
            return cls
    return "systemic"


def med_groups_for(name: str) -> list[str]:
    s = str(name).lower()
    return [g for g, rx in _MED_RE.items() if rx.search(s)]


# ── population creatinine normals (fallback baseline only) ────────────────
# (lo_years, hi_years, sex) -> umol/L. Approximate values, used only when
# labelling.allow_population_baseline is true. Disabled in configs/common.yaml
# and NOT used for any reported analysis: AKI requires a measured reference.
POP_CREAT = {
    (0, 1, "M"): 27, (0, 1, "F"): 27, (1, 3, "M"): 27, (1, 3, "F"): 27,
    (3, 5, "M"): 35, (3, 5, "F"): 35, (5, 10, "M"): 44, (5, 10, "F"): 44,
    (10, 13, "M"): 53, (10, 13, "F"): 53, (13, 16, "M"): 62, (13, 16, "F"): 53,
    (16, 19, "M"): 74, (16, 19, "F"): 60,
}


def population_creatinine(age_years: float, sex: str) -> float:
    s = sex if sex in ("M", "F") else "F"
    for (lo, hi, sx), v in POP_CREAT.items():
        if sx == s and lo <= age_years < hi:
            return float(v)
    return float("nan")
