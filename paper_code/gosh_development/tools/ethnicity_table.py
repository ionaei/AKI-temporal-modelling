"""Ethnicity rows for Table 1: White / Asian / Black / Mixed / Other / Not available.

    # GOSH (in the DRE): point it at the raw demographics file and name its columns
    python tools/ethnicity_table.py --site gosh --file /path/to/gosh/demographics.csv \
        --subject_col project_id --eth_col ethnicity_name --columns "train,val" "test"

    # MIMIC-III (where the data are): ethnicity is in ADMISSIONS
    python tools/ethnicity_table.py --site mimic3 \
        --file /path/to/mimiciii/1.4/ADMISSIONS.csv.gz \
        --subject_col SUBJECT_ID --eth_col ETHNICITY --columns all

Counts ICU STAYS that survived every exclusion (data/<site>/labelled/stays_labelled.parquet joined to
split.csv), each stay taking its patient's ethnicity, so repeat stays are counted repeatedly, as in the
rest of Table 1. A patient with several recorded values gets the most frequent known one; 'Not available'
only if none is known. Percentages are of all stays in the column (Not available included).

Raw text -> category (first match wins, case-insensitive): Not available (unknown, declined, unable,
not stated, NHS code Z) > Mixed > specific 'Other' groups (Hispanic/Latino, American Indian, Pacific
Islander, Middle Eastern, South American, Portuguese) > Black > Asian > White > Other.
ALWAYS check the printed mapping table: unusual raw values are the usual source of error.
Override any value with --map_file (CSV with columns raw,category).

Outputs: results/ethnicity_<site>.csv (counts) and results/ethnicity_<site>_mapping.csv (raw value ->
category, number of rows). Aggregates only; counts below 5 are shown as '<5'.
"""
import argparse
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CATS = ["White", "Asian", "Black", "Mixed", "Other", "Not available"]
RULES = [
    ("Not available", r"^$|^(z|99|nan|none|null)$|not stated|not known|unknown|unable|declin|refus|"
                      r"not recorded|not specified|undeclared|not provided|missing|no data|prefer not"),
    ("Mixed", r"mixed|multi|dual|white (and|&|/) ?(black|asian)|^(d|e|f|g)$"),
    ("Other", r"american indian|alaska|hawaiian|pacific islander|hispanic|latino|latin american|"
              r"middle east|arab|south american|portuguese"),
    ("Black", r"black|african|caribbean|^(m|n|p)$"),
    ("Asian", r"asian|indian|pakistani|bangladeshi|chinese|sri ?lankan|nepal|^(h|j|k|l|r)$"),
    ("White", r"white|british|irish|english|welsh|scottish|gypsy|traveller|european|^(a|b|c)$"),
]


def categorise(raw, overrides):
    s = "" if pd.isna(raw) else str(raw).strip().lower()
    if s in overrides:
        return overrides[s]
    for cat, rx in RULES:
        if re.search(rx, s):
            return cat
    return "Other"


def cell(n, tot, min_cell=5):
    shown = f"<{min_cell}" if 0 < n < min_cell else f"{n:,}"
    return f"{shown} ({100 * n / tot:.1f}\\%)" if tot else "---"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    ap.add_argument("--file", required=True, help="raw file with patient id and ethnicity")
    ap.add_argument("--subject_col", required=True)
    ap.add_argument("--eth_col", required=True)
    ap.add_argument("--columns", nargs="+", default=["all"],
                    help="table columns as comma-separated splits, e.g. \"train,val\" \"test\"; or all")
    ap.add_argument("--map_file", help="CSV with columns raw,category to override the rules")
    ap.add_argument("--sep", default=None, help="separator (default: auto)")
    a = ap.parse_args()

    overrides = {}
    if a.map_file:
        m = pd.read_csv(a.map_file, dtype=str)
        overrides = {r.lower().strip(): c for r, c in zip(m["raw"], m["category"])}
    raw = pd.read_csv(a.file, dtype=str, sep=a.sep, engine="python" if a.sep is None else "c",
                      usecols=[a.subject_col, a.eth_col])
    raw[a.subject_col] = raw[a.subject_col].str.strip()
    raw["cat"] = raw[a.eth_col].map(lambda v: categorise(v, overrides))

    d = ROOT / "data" / a.site / "labelled"
    lab = pd.read_parquet(d / "stays_labelled.parquet")
    split = pd.read_csv(d / "split.csv", dtype=str)
    s = lab.merge(split[["stay_id", "split"]], on="stay_id")            # eligible stays only
    s["subject_id"] = s["subject_id"].astype(str).str.strip()

    # one category per patient: most frequent known value, else 'Not available'
    known = raw[raw["cat"] != "Not available"]
    mode = known.groupby(a.subject_col)["cat"].agg(lambda x: x.value_counts().index[0])
    s["ethnicity"] = s["subject_id"].map(mode).fillna("Not available")
    unmatched = int((~s["subject_id"].isin(raw[a.subject_col])).sum())
    if unmatched:
        print(f"note: {unmatched} stays have a patient id absent from {Path(a.file).name} "
              f"(counted as Not available); check --subject_col")

    # mapping table for review (rows of the raw file belonging to included patients)
    inc = raw[raw[a.subject_col].isin(set(s["subject_id"]))]
    mp = (inc.groupby([a.eth_col, "cat"], dropna=False).size().reset_index(name="n_rows")
          .sort_values(["cat", "n_rows"], ascending=[True, False]))
    mp["n_rows"] = mp["n_rows"].map(lambda n: f"<5" if 0 < n < 5 else str(n))
    print("\nraw value -> category (check this!):")
    print(mp.to_string(index=False))

    out = {}
    for spec in a.columns:
        sel = s if spec == "all" else s[s["split"].isin(spec.split(","))]
        vc = sel["ethnicity"].value_counts()
        out[spec] = {c: cell(int(vc.get(c, 0)), len(sel)) for c in CATS}
        out[spec]["(n stays)"] = f"{len(sel):,}"
    t = pd.DataFrame(out)
    res = ROOT / "results"
    res.mkdir(exist_ok=True)
    t.to_csv(res / f"ethnicity_{a.site}.csv")
    mp.rename(columns={a.eth_col: "raw", "cat": "category"}).to_csv(
        res / f"ethnicity_{a.site}_mapping.csv", index=False)
    pd.set_option("display.width", 200)
    print("\nTable 1 rows (LaTeX-ready):")
    print(t.to_string())
    print(f"\nwritten: results/ethnicity_{a.site}.csv, results/ethnicity_{a.site}_mapping.csv")


if __name__ == "__main__":
    main()