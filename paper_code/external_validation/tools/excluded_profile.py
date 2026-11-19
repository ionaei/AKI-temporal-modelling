"""Who was excluded because AKI was already present on arrival? Demographics and a short clinical profile.

    python tools/excluded_profile.py --site gosh
    python tools/excluded_profile.py --site gosh --eth_file <demographics.csv> --subject_col project_id --eth_col <col>
    python tools/excluded_profile.py --site pic

Reads data/<site>/labelled/stays_labelled.parquet, which keeps EVERY stay with its exclusion reason, and
profiles four groups side by side:

    AKI before admission      exclude_reason == aki_before_admission   (community-acquired / pre-existing AKI)
    AKI within 24 h           exclude_reason == aki_on_arrival          (onset in the first 24 h of the stay)
    Both excluded groups      the union (GOSH 725 = 387 + 338; PIC 1,244 = 269 + 975)
    Eligible stays            the stays kept for modelling (for comparison)

Rows: stays and patients; sex; age (mean +- SD, median [IQR], range, age bands); ICU length of stay; creatinine
measurements per stay; AKI detected during the ICU stay, peak KDIGO stage and hours from admission to onset
(only stays with an in-stay AKI have these; for 'AKI before admission' stays this is AKI that was ALSO found
after admission, as the pre-admission episode itself is not staged); ethnicity if --eth_file is given.

Cells are formatted for LaTeX (\\% and $\\pm$), as in table1.py. Aggregates only: counts below 5 are shown as '<5';
a group with fewer than 5 stays is suppressed completely, and any percentage whose denominator is below 5 is '---'.
Writes results/excluded_profile_<site>.csv and prints the table.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
MIN_CELL = 5
BANDS = [("< 28 days", 0, 28), ("28 days to 1 year", 28, 365), ("1 to 3 years", 365, 3 * 365),
         ("3 to 12 years", 3 * 365, 12 * 365), ("12 to 18 years", 12 * 365, 19 * 365)]   # edges as in the pipeline
CATS = ["White", "Asian", "Black", "Mixed", "Other", "Not available"]


def cnt(k, tot):
    if tot < MIN_CELL:                       # denominators below 5 would reveal individual stays
        return "---"
    if 0 < k < MIN_CELL:
        return f"<{MIN_CELL}"
    return f"{k:,} ({100 * k / tot:.1f}\\%)"


def msd(x, d=2):
    x = pd.Series(x).dropna()
    return f"{x.mean():.{d}f} $\\pm$ {x.std():.{d}f}" if len(x) >= MIN_CELL else "---"


def med(x, d=1):
    x = pd.Series(x).dropna()
    if len(x) < MIN_CELL:
        return "---"
    q1, m, q3 = x.quantile([.25, .5, .75])
    return f"{m:.{d}f} ({q1:.{d}f}--{q3:.{d}f})"


def rng(x, d=1):
    x = pd.Series(x).dropna()
    return f"{x.min():.{d}f}--{x.max():.{d}f}" if len(x) >= MIN_CELL else "---"


def patient_ethnicity(path, subject_col, eth_col, sep=None):
    """One category per patient (most frequent known value), using the rules of ethnicity_table.py."""
    try:
        from ethnicity_table import categorise
    except ImportError:
        raise SystemExit("ethnicity_table.py must be in the same folder (tools/) to categorise ethnicity")
    raw = pd.read_csv(path, dtype=str, sep=sep, engine="python" if sep is None else "c",
                      usecols=[subject_col, eth_col])
    raw[subject_col] = raw[subject_col].str.strip()
    raw["cat"] = raw[eth_col].map(lambda v: categorise(v, {}))

    def pick(s):
        known = s[s != "Not available"]
        return known.mode().iloc[0] if len(known) else "Not available"
    return raw.groupby(subject_col)["cat"].agg(pick)


def profile(s: pd.DataFrame, eth: pd.Series | None, extra: list[str]) -> dict:
    n = len(s)
    if 0 < n < MIN_CELL:                     # a group this small is suppressed completely
        return {"ICU stays": f"<{MIN_CELL}", "Unique patients": f"<{MIN_CELL}"}
    out = {"ICU stays": f"{n:,}", "Unique patients": f"{s['subject_id'].nunique():,}"}
    if n == 0:
        return out
    known = int(s["sex"].isin(["M", "F"]).sum())
    out["Female"] = cnt(int((s["sex"] == "F").sum()), known)
    out["Male"] = cnt(int((s["sex"] == "M").sum()), known)
    age = s["age_days"] / 365.25
    out["Age, years: mean $\\pm$ SD"] = msd(age)
    out["Age, years: median (IQR)"] = med(age)
    out["Age, years: range"] = rng(age)
    for label, lo, hi in BANDS:
        out[f"Age {label}"] = cnt(int(s["age_days"].between(lo, hi, inclusive="left").sum()), n)
    if "los_h" in s:
        out["ICU length of stay, days: median (IQR)"] = med(s["los_h"] / 24)
    if "n_creat_stay" in s:
        out["Creatinine measurements in the stay: median (IQR)"] = med(s["n_creat_stay"])
    if "aki" in s:
        a = s[s["aki"].astype(bool)]
        out["AKI also detected during the ICU stay"] = cnt(len(a), n)
        if "stage_peak" in s:
            for k in (1, 2, 3):
                out[f"  Peak stage {k} (of stays with in-stay AKI)"] = cnt(int((a["stage_peak"] == k).sum()), len(a))
            out["  Severe (stage 2-3)"] = cnt(int((a["stage_peak"] >= 2).sum()), len(a))
        if "onset_h" in s:
            out["  Hours from admission to onset: median (IQR)"] = med(a["onset_h"])
        if "trigger" in s:
            out["  Onset by rise within 48 h (absolute)"] = cnt(int(a["trigger"].astype(str).str.startswith("abs").sum()), len(a))
    if eth is not None:
        e = s["subject_id"].map(eth).fillna("Not available")
        for c in CATS:
            out[f"Ethnicity: {c}"] = cnt(int((e == c).sum()), n)
    for col in extra:
        if col in s:
            for v, k in s[col].astype(str).value_counts().items():
                out[f"{col}: {v}"] = cnt(int(k), n)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True, help="gosh | pic | mimic3 (folder under data/)")
    ap.add_argument("--eth_file", help="raw demographics file with patient id and ethnicity (GOSH, MIMIC-III ADMISSIONS)")
    ap.add_argument("--subject_col")
    ap.add_argument("--eth_col")
    ap.add_argument("--sep", default=None)
    ap.add_argument("--extra", nargs="*", default=[], help="further categorical columns of stays_labelled to tabulate")
    a = ap.parse_args()

    lab = pd.read_parquet(ROOT / "data" / a.site / "labelled" / "stays_labelled.parquet")
    need = {"stay_id", "subject_id", "age_days", "sex", "exclude", "exclude_reason"}
    if need - set(lab.columns):
        raise SystemExit(f"stays_labelled.parquet lacks {sorted(need - set(lab.columns))}")
    reason = lab["exclude_reason"].fillna("").astype(str)
    groups = {
        "AKI before admission": lab[reason == "aki_before_admission"],
        "AKI within 24 h of admission": lab[reason == "aki_on_arrival"],
        "Both excluded groups": lab[reason.isin(["aki_before_admission", "aki_on_arrival"])],
        "Eligible stays": lab[~lab["exclude"].astype(bool)],
    }
    eth = None
    if a.eth_file:
        if not (a.subject_col and a.eth_col):
            raise SystemExit("--eth_file needs --subject_col and --eth_col")
        eth = patient_ethnicity(a.eth_file, a.subject_col, a.eth_col, a.sep)

    t = pd.DataFrame({name: profile(g, eth, a.extra) for name, g in groups.items()}).fillna("---")
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    t.to_csv(out / f"excluded_profile_{a.site}.csv")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 48)
    print(f"[{a.site}] stays in the file: {len(lab):,}; exclusion reasons: {reason[reason != ''].value_counts().to_dict()}")
    print(f"check: AKI before admission + within 24 h = {len(groups['Both excluded groups']):,} stays\n")
    print(t.to_string())
    print(f"\nwritten: {out / f'excluded_profile_{a.site}.csv'}")


if __name__ == "__main__":
    main()