"""Site adapters turn raw site extracts into identical canonical tables.

Canonical tables (written to data/<site>/canonical/*.parquet):

  episodes : subject_id, start, end, dob, sex ('M'/'F'/'U'), unit
             -- raw care episodes; cohort.py merges contiguous ones into stays
  labs     : subject_id, time, concept, value, unit, code
  vitals   : subject_id, time, concept, value, unit, code   (includes support flags)
  meds     : subject_id, start, end, name, route
  rrt      : subject_id, time                                (renal replacement events)

Everything downstream of this file is site-agnostic. All codes are mapped
through explicit lists in the site YAML: there is NO substring or fuzzy
matching anywhere in the pipeline.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..concepts import to_canonical

log = logging.getLogger(__name__)

CANONICAL_COLUMNS = {
    "episodes": ["subject_id", "start", "end", "dob", "sex", "unit"],
    "labs": ["subject_id", "time", "concept", "value", "unit", "code"],
    "vitals": ["subject_id", "time", "concept", "value", "unit", "code"],
    "meds": ["subject_id", "start", "end", "name", "route"],
    "rrt": ["subject_id", "time"],
}


def empty(table: str) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in CANONICAL_COLUMNS[table]})


def require_columns(df_cols, needed: dict, where: str):
    missing = [v for v in needed.values() if v and v not in df_cols]
    if missing:
        raise KeyError(
            f"[{where}] columns {missing} not found. Available: {list(df_cols)}.\n"
            f"Fix the `columns:` block for this table in the site YAML.")


def invert_map(mapping: dict) -> dict:
    """{concept: [codes]} -> {code: concept}; raises if a code maps twice."""
    inv = {}
    for concept, codes in (mapping or {}).items():
        for code in codes or []:
            key = str(code).strip().upper()
            if key in inv and inv[key] != concept:
                raise ValueError(f"code {code} mapped to both {inv[key]} and {concept}")
            inv[key] = concept
    return inv


def read_mapped_events(spec: dict, where: str, default_units: dict | None = None,
                       chunksize: int = 500_000, report: list | None = None) -> pd.DataFrame:
    """Read a long event file, keep only explicitly mapped codes, convert units.

    spec keys: path, columns{subject,time,code,value,unit?}, map{concept:[codes]},
               optional sep, encoding, filter{column: [allowed values]}
    """
    path = spec.get("path")
    if not path:
        log.warning("[%s] no path configured — table skipped", where)
        return pd.DataFrame(columns=CANONICAL_COLUMNS["labs"])
    cols = spec["columns"]
    code2concept = invert_map(spec.get("map", {}))
    if not code2concept:
        log.warning("[%s] empty concept map — table skipped", where)
        return pd.DataFrame(columns=CANONICAL_COLUMNS["labs"])
    usecols = [c for c in {cols["subject"], cols["time"], cols["code"], cols["value"],
                           cols.get("unit"), cols.get("value_text")} if c]
    usecols += [c for c in (spec.get("filter") or {}) if c not in usecols]
    header = pd.read_csv(path, nrows=0, sep=spec.get("sep", ","),
                         encoding=spec.get("encoding"))
    require_columns(header.columns, {k: v for k, v in cols.items() if v}, where)

    parts = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize, dtype=str,
                             sep=spec.get("sep", ","), encoding=spec.get("encoding")):
        for fcol, allowed in (spec.get("filter") or {}).items():
            chunk = chunk[chunk[fcol].isin([str(a) for a in allowed])]
        key = chunk[cols["code"]].astype(str).str.strip().str.upper()
        # ITEMIDs may be read as '50912' or '50912.0'
        key = key.str.replace(r"\.0$", "", regex=True)
        concept = key.map(code2concept)
        keep = concept.notna()
        if not keep.any():
            continue
        c = chunk[keep]
        value = c[cols["value"]]
        if cols.get("value_text"):
            # numeric column blank (e.g. '<5' stored only as text) -> use the text result
            blank = value.isna() | (value.astype(str).str.strip() == "")
            value = value.where(~blank, c[cols["value_text"]])
        parts.append(pd.DataFrame({
            "subject_id": c[cols["subject"]].astype(str).str.strip(),
            "time": pd.to_datetime(c[cols["time"]], errors="coerce"),
            "concept": concept[keep].to_numpy(),
            "value": value.to_numpy(),
            "unit": c[cols["unit"]].to_numpy() if cols.get("unit") else "",
            "code": key[keep].to_numpy(),
        }))
    if not parts:
        log.warning("[%s] no rows matched the concept map", where)
        return pd.DataFrame(columns=CANONICAL_COLUMNS["labs"])
    df = pd.concat(parts, ignore_index=True)
    df = df[df["time"].notna()]
    # presence-only concepts (e.g. a ventilator-mode row means 'ventilated')
    for fc in spec.get("flag_concepts", []) or []:
        is_f = df["concept"] == fc
        df.loc[is_f, "value"] = "1"
        df.loc[is_f, "unit"] = "flag"
    # per-CODE default units (one concept can mix codes in different units, e.g.
    # laboratory Hb in g/L and blood-gas Hb in g/dL); used only where unit is blank
    code_units = {str(k).strip().upper(): v
                  for k, v in (spec.get("default_units_by_code") or {}).items()}
    if code_units:
        u = df["unit"].astype(str).str.strip()
        blank = df["unit"].isna() | u.isin(["", "nan", "None"])
        df.loc[blank, "unit"] = df.loc[blank, "code"].map(code_units).fillna("").to_numpy()
    n0 = len(df)
    df = to_canonical(df, default_units=default_units, report=report)
    log.info("[%s] %d mapped rows -> %d after unit/plausibility checks", where, n0, len(df))
    return df[CANONICAL_COLUMNS["labs"]]


def map_sex(series: pd.Series, mapping: dict) -> pd.Series:
    m = {str(k).strip().lower(): v for k, v in (mapping or {}).items()}
    return series.astype(str).str.strip().str.lower().map(m).fillna("U")


def date_only_shift(ts: pd.Series, policy: str) -> pd.Series:
    """For sources with dates but no times (e.g. MIMIC-III PRESCRIPTIONS).

    'end_of_day' (default) makes the event available only from the END of
    that day, so a drug started at 18:00 can never appear in a 00:00-06:00
    window. 'start_of_day' leaks up to 24h of future information.
    """
    ts = pd.to_datetime(ts, errors="coerce")
    is_midnight = (ts.dt.hour == 0) & (ts.dt.minute == 0) & (ts.dt.second == 0)
    if policy == "end_of_day":
        return ts.where(~is_midnight, ts + pd.Timedelta(hours=24))
    return ts


class Adapter:
    """Subclasses implement the five readers; `run` writes canonical parquet."""

    def __init__(self, site_cfg: dict):
        self.cfg = site_cfg
        self.unit_report: list = []

    def episodes(self) -> pd.DataFrame: raise NotImplementedError
    def labs(self) -> pd.DataFrame: raise NotImplementedError
    def vitals(self) -> pd.DataFrame: return empty("vitals")
    def meds(self) -> pd.DataFrame: return empty("meds")
    def rrt(self) -> pd.DataFrame: return empty("rrt")

    def run(self) -> dict:
        out_dir = Path(self.cfg["dirs"]["canonical"])
        tables = {}
        for name in ["episodes", "labs", "vitals", "meds", "rrt"]:
            log.info("extracting %s", name)
            df = getattr(self, name)()
            df = df[[c for c in CANONICAL_COLUMNS[name] if c in df.columns]]
            df.to_parquet(out_dir / f"{name}.parquet", index=False)
            tables[name] = df
            log.info("  %s: %d rows", name, len(df))
        cols = ["concept", "code", "issue", "n_rows", "p01", "median", "p99"]
        rep = pd.DataFrame(self.unit_report).reindex(columns=cols)
        rep = rep.sort_values(["concept", "n_rows"], ascending=[True, False])
        rep.to_csv(out_dir / "unit_report.csv", index=False)
        if len(rep):
            log.warning("unit/value issues by concept and code (see unit_report.csv):\n%s",
                        rep.to_string(index=False, max_rows=200))
        return tables
