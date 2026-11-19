"""First look at a MIMIC-schema database (PIC, MIMIC-III) before mapping anything.

    python tools/inspect_raw.py --site pic

Prints (and saves to data/<site>/discover/inspect.txt) AGGREGATES ONLY:
file names and sizes, column headers, care-unit counts, whether time stamps
carry a time of day, unit strings, route labels, and the short text tokens
found where VALUENUM is blank (e.g. '<14'). No patient-level rows are printed,
so the output can be shared.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aki.config import load_config  # noqa: E402

OUT = []


def say(*a):
    line = " ".join(str(x) for x in a)
    print(line)
    OUT.append(line)


def header(path):
    return pd.read_csv(path, nrows=0).columns.tolist()


def time_of_day_share(series: pd.Series) -> str:
    t = pd.to_datetime(series, errors="coerce").dropna()
    if t.empty:
        return "unparseable"
    has_time = ((t.dt.hour != 0) | (t.dt.minute != 0) | (t.dt.second != 0)).mean()
    return f"{100 * has_time:.1f}% have a time of day (0% = dates only)"


def find(cols, *names):
    for n in names:
        for c in cols:
            if c.upper() == n.upper():
                return c
    for n in names:
        for c in cols:
            if n.upper() in c.upper():
                return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    ap.add_argument("--lab_rows", type=int, default=5_000_000,
                    help="max LABEVENTS rows to scan for units / text tokens")
    a = ap.parse_args()
    cfg = load_config(a.site)["site"]
    raw = Path(cfg["raw_dir"])
    out = Path(cfg["dirs"]["root"]) / "discover"
    out.mkdir(parents=True, exist_ok=True)

    say(f"== files in {raw}")
    for f in sorted(raw.iterdir()):
        if f.is_file():
            say(f"  {f.name:40s} {f.stat().st_size / 1e6:10.1f} MB")

    say("\n== headers")
    heads = {}
    for f in sorted(raw.glob("*.csv*")):
        try:
            heads[f.name] = header(f)
            say(f"  {f.name}: {heads[f.name]}")
        except Exception as e:  # noqa: BLE001
            say(f"  {f.name}: could not read ({e})")

    def path_of(stem):
        for f in raw.glob("*.csv*"):
            if f.name.upper().startswith(stem.upper() + "."):
                return f
        return None

    p = path_of("ICUSTAYS")
    if p:
        d = pd.read_csv(p, dtype=str)
        cu = find(d.columns, "FIRST_CAREUNIT", "CAREUNIT")
        say("\n== ICUSTAYS")
        say(f"  rows {len(d)}, subjects {d[find(d.columns, 'SUBJECT_ID')].nunique()}")
        if cu:
            say("  care units:\n" + d[cu].value_counts(dropna=False).to_string())
        it = find(d.columns, "INTIME")
        if it:
            say(f"  INTIME: {time_of_day_share(d[it])}")
        lo = find(d.columns, "LOS")
        if lo:
            say("  LOS (days) quartiles: " +
                str(pd.to_numeric(d[lo], errors="coerce").quantile([.25, .5, .75]).round(2).tolist()))

    p = path_of("PATIENTS")
    if p:
        d = pd.read_csv(p, dtype=str)
        say("\n== PATIENTS")
        g = find(d.columns, "GENDER", "SEX")
        if g:
            say("  gender values:\n" + d[g].value_counts(dropna=False).to_string())
        dob = find(d.columns, "DOB")
        if dob:
            say(f"  DOB: {time_of_day_share(d[dob])}")

    p = path_of("PRESCRIPTIONS")
    if p:
        d = pd.read_csv(p, dtype=str)
        say("\n== PRESCRIPTIONS")
        say(f"  rows {len(d)}")
        for c in d.columns:
            if re.search(r"START|END|TIME|DATE", c, re.I):
                say(f"  {c}: {time_of_day_share(d[c])}; missing {100 * d[c].isna().mean():.1f}%")
        r = find(d.columns, "ROUTE")
        if r:
            say("  route labels (top 40):\n" + d[r].value_counts(dropna=False).head(40).to_string())
        else:
            say("  no ROUTE column")
        for c in d.columns:
            if re.search(r"DRUG|NAME", c, re.I):
                say(f"  {c}: {d[c].nunique()} distinct values, missing {100 * d[c].isna().mean():.1f}%")

    p = path_of("LABEVENTS")
    if p:
        say(f"\n== LABEVENTS (first {a.lab_rows:,} rows)")
        cols = header(p)
        vn, v, u = find(cols, "VALUENUM"), find(cols, "VALUE"), find(cols, "VALUEUOM")
        use = [c for c in (vn, v, u, find(cols, "ITEMID"), find(cols, "CHARTTIME")) if c]
        d = pd.read_csv(p, usecols=use, dtype=str, nrows=a.lab_rows)
        say(f"  rows scanned {len(d):,}; VALUENUM blank {100 * d[vn].isna().mean():.1f}%")
        ct = find(cols, "CHARTTIME")
        if ct:
            say(f"  CHARTTIME: {time_of_day_share(d[ct].sample(min(len(d), 100000), random_state=0))}")
        if u:
            say("  unit strings (top 50):\n" + d[u].value_counts(dropna=False).head(50).to_string())
        if v:
            t = d.loc[d[vn].isna(), v].fillna("<empty>")
            t = t[t.str.len() <= 20].value_counts()
            say("  text where VALUENUM is blank (n >= 20, top 40):\n" +
                t[t >= 20].head(40).to_string())

    for stem in ("D_LABITEMS", "D_ITEMS"):
        p = path_of(stem)
        if p:
            d = pd.read_csv(p, dtype=str)
            say(f"\n== {stem}: {len(d)} items; columns {d.columns.tolist()}")
            for c in ("FLUID", "CATEGORY", "LINKSTO", "DBSOURCE"):
                cc = find(d.columns, c)
                if cc:
                    say(f"  {cc}:\n" + d[cc].value_counts(dropna=False).head(25).to_string())

    (out / "inspect.txt").write_text("\n".join(OUT), encoding="utf-8")
    print(f"\nsaved to {out / 'inspect.txt'}")


if __name__ == "__main__":
    main()
