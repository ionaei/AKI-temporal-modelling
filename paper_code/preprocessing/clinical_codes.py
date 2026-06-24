"""
config/clinical_codes.py

Medication and procedure groupings for feature engineering.

Medications
-----------
NEPHROTOXINS      : tier-1 drugs flagged individually by name string match.
DRUG_CLASSES      : tier-2 drug classes flagged by name string match.
GPI_NEPHROTOXINS  : tier-1 nephrotoxins mapped by GPI code prefix (more robust).
GPI_DRUG_CLASSES  : tier-2 drug classes mapped by GPI code prefix.

GPI code structure:
    First 2 digits = drug class
    First 4 digits = drug group
    First 6 digits = drug subgroup
    Full 14 digits = unique product

Use GPI mapping where drug_gpi column is available — it is more reliable
than string matching on drug names which vary by formulary.

Procedures
----------
PROCEDURE_GROUPS : OPCS-4 prefix → clinical group name.
"""

# ── Tier 1: individually nephrotoxic drugs (name match) ──
NEPHROTOXINS: dict[str, list[str]] = {
    "vancomycin":     ["vancomycin", "vanco"],
    "gentamicin":     ["gentamicin", "genta", "gentamycin"],
    "tobramycin":     ["tobramycin"],
    "amikacin":       ["amikacin"],
    "colistin":       ["colistin", "polymyxin e", "polymyxin b"],
    "amphotericin":   ["amphotericin", "ambisome", "fungizone",
                       "amphotericin b"],
    "ciclosporin":    ["ciclosporin", "cyclosporine", "cyclosporin",
                       "sandimmun", "neoral"],
    "tacrolimus":     ["tacrolimus", "prograf", "advagraf", "adoport"],
    "ibuprofen":      ["ibuprofen", "nurofen", "brufen"],
    "indomethacin":   ["indomethacin", "indometacin", "indocin"],
    "contrast":       ["contrast", "iodinated contrast", "omnipaque",
                       "visipaque", "iohexol", "iopromide", "iodixanol"],
    "aciclovir":      ["aciclovir", "acyclovir", "zovirax"],
    "methotrexate":   ["methotrexate", "mtx"],
    "cisplatin":      ["cisplatin", "platinol"],
    "carboplatin":    ["carboplatin", "paraplatin"],
    "ifosfamide":     ["ifosfamide", "ifex", "mitoxana"],
    "sirolimus":      ["sirolimus", "rapamycin", "rapamune"],
}

# ── Tier 2: drug classes (name match) ────────────────────
DRUG_CLASSES: dict[str, list[str]] = {
    "diuretic":          ["furosemide", "frusemide", "bumetanide",
                          "spironolactone", "hydrochlorothiazide",
                          "metolazone", "chlorothiazide", "torsemide"],
    "ace_inhibitor":     ["captopril", "enalapril", "lisinopril",
                          "ramipril", "perindopril", "quinapril",
                          "fosinopril"],
    "arb":               ["losartan", "valsartan", "irbesartan",
                          "candesartan", "olmesartan", "telmisartan"],
    "vasopressor":       ["noradrenaline", "norepinephrine", "adrenaline",
                          "epinephrine", "dopamine", "vasopressin",
                          "dobutamine", "milrinone", "terlipressin"],
    "antibiotic":        ["penicillin", "amoxicillin", "piperacillin",
                          "meropenem", "imipenem", "ceftriaxone",
                          "cefotaxime", "ceftazidime", "cefuroxime",
                          "ciprofloxacin", "flucloxacillin", "teicoplanin",
                          "azithromycin", "clarithromycin", "trimethoprim",
                          "co-amoxiclav", "tazobactam"],
    "antifungal":        ["fluconazole", "voriconazole", "caspofungin",
                          "micafungin", "posaconazole", "itraconazole"],
    "immunosuppressant": ["mycophenolate", "mycophenolic acid", "azathioprine",
                          "prednisolone", "methylprednisolone",
                          "dexamethasone", "hydrocortisone",
                          "basiliximab", "rituximab"],
    "antiviral":         ["ganciclovir", "valganciclovir", "foscarnet",
                          "cidofovir", "oseltamivir"],
    "ppi_h2":            ["omeprazole", "lansoprazole", "esomeprazole",
                          "ranitidine", "pantoprazole", "famotidine"],
    "anticoagulant":     ["heparin", "enoxaparin", "dalteparin",
                          "tinzaparin", "fondaparinux", "warfarin",
                          "apixaban", "rivaroxaban"],
    "sedation_analgesia":["morphine", "fentanyl", "midazolam", "ketamine",
                          "propofol", "clonidine", "dexmedetomidine",
                          "lorazepam", "diazepam"],
}

# ── GPI-based nephrotoxin mapping ─────────────────────────
# Keys match NEPHROTOXINS above — values are GPI prefixes (str.startswith)
# GPI codes are more reliable than name matching across formularies.
# Extend this as you audit your specific drug_gpi values.
GPI_NEPHROTOXINS: dict[str, list[str]] = {
    "vancomycin":     ["0280002010"],          # glycopeptide antibiotics
    "gentamicin":     ["0104070010"],          # aminoglycosides
    "tobramycin":     ["0104070020"],
    "amikacin":       ["0104070004"],
    "colistin":       ["0104090010"],          # polymyxins
    "amphotericin":   ["0622010010"],          # polyene antifungals
    "ciclosporin":    ["6000001010"],          # calcineurin inhibitors
    "tacrolimus":     ["6000001020"],
    "ibuprofen":      ["6600200510"],          # NSAIDs
    "indomethacin":   ["6600200310"],
    "methotrexate":   ["2110002010"],          # antimetabolites
    "cisplatin":      ["2116000010"],          # platinum compounds
    "carboplatin":    ["2116000020"],
    "ifosfamide":     ["2102000620"],          # alkylating agents
    "sirolimus":      ["6000001030"],          # mTOR inhibitors
    "aciclovir":      ["0202010010"],          # antivirals
}

# ── GPI-based drug class mapping ─────────────────────────
# Values are GPI 2-digit class prefixes unless more specificity needed
GPI_DRUG_CLASSES: dict[str, list[str]] = {
    "diuretic":          ["3620"],             # loop + thiazide diuretics
    "ace_inhibitor":     ["3610060"],          # ACE inhibitors
    "arb":               ["3610065"],          # angiotensin receptor blockers
    "vasopressor":       ["1210", "1212"],     # adrenergic agents
    "antibiotic":        ["0101", "0102",      # penicillins, cephalosporins
                          "0104", "0107",      # aminoglycosides, macrolides
                          "0110", "0113"],     # fluoroquinolones, carbapenems
    "antifungal":        ["0622"],             # antifungals
    "immunosuppressant": ["6000",              # immunosuppressants
                          "0510060"],          # systemic corticosteroids
    "antiviral":         ["0202"],             # antivirals
    "ppi_h2":            ["4920010",           # PPIs
                          "4920020"],          # H2 antagonists
    "anticoagulant":     ["3900"],             # anticoagulants
    "sedation_analgesia":["6500",              # opioid analgesics
                          "2820"],             # benzodiazepines
}

# ── OPCS-4 procedure groups ───────────────────────────────
PROCEDURE_GROUPS: dict[str, list[str]] = {
    "cardiac_surgery":    ["K", "L"],
    "renal_procedure":    ["M"],
    "transplant":         ["X5"],
    "bypass_circuit":     ["K61", "K62"],
    "dialysis_rrt":       ["X40", "M01", "X39"],
    "imaging_contrast":   ["U17", "U18"],
    "mechanical_vent":    ["E85", "E86"],
    "central_line":       ["X36"],
    "abdominal_surgery":  ["H", "J", "G"],
    "neurosurgery":       ["A", "B"],
    "orthopaedic":        ["V", "W"],
    "bronchoscopy_scope": ["E48", "E49", "G16"],
}
