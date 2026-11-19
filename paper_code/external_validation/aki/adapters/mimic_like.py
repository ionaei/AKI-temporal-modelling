"""Adapter for MIMIC-III-schema databases: used for both PIC (v1.1.0) and MIMIC-III.

The two databases share table structure (PATIENTS, ICUSTAYS, LABEVENTS,
CHARTEVENTS, PRESCRIPTIONS, INPUTEVENTS*, OUTPUTEVENTS, D_ITEMS, D_LABITEMS).
Differences (file names, drug-name column, care units, units, date-only
fields) are handled entirely in the site YAML.

ITEMID lists in the YAML are audited with tools/discover_codes.py.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .base import (Adapter, CANONICAL_COLUMNS, date_only_shift, empty, map_sex,
                   read_mapped_events, require_columns)

log = logging.getLogger(__name__)


def route_for(route_col, name: pd.Series, m: dict):
    """Route text for route_class(), in priority order:
      1. route_name_rules: [[regex, route], ...] matched on the drug name
         (e.g. 'eye drops' -> ophthalmic even when the form column says 'water aqua');
      2. route_value_map: form/route column value -> route word; the special value
         '@name' means 'use the drug name itself' (as GOSH's route_from_name);
      3. the raw route column value.
    """
    lname = name.astype(str).str.lower()
    out = pd.Series(None, index=name.index, dtype=object)
    if route_col is not None:
        raw = route_col.astype(str).str.strip()
        rmap = {str(k).strip().lower(): v for k, v in (m.get("route_value_map") or {}).items()}
        mapped = raw.str.lower().map(rmap)
        use_name = mapped == "@name"
        out = mapped.where(~use_name, lname).fillna(raw).astype(object)
    for rx, rt in (m.get("route_name_rules") or []):
        hit = lname.str.contains(rx, regex=True)
        out[hit] = rt
    return out


class MimicLikeAdapter(Adapter):

    def _path(self, table: str) -> str:
        files = self.cfg["files"]
        p = Path(self.cfg["raw_dir"]) / files[table]
        if not p.exists():
            raise FileNotFoundError(f"{table}: {p} not found (check raw_dir/files in site YAML)")
        return str(p)

    def episodes(self) -> pd.DataFrame:
        c = self.cfg["columns"]
        icu = pd.read_csv(self._path("icustays"), dtype=str)
        pat = pd.read_csv(self._path("patients"), dtype=str)
        require_columns(icu.columns, {k: c[k] for k in ("subject", "careunit", "intime",
                                                         "outtime")}, "icustays")
        require_columns(pat.columns, {k: c[k] for k in ("subject", "dob", "gender")},
                        "patients")
        units = self.cfg.get("include_careunits")
        if units:
            icu = icu[icu[c["careunit"]].isin(units)]
        excl = self.cfg.get("exclude_careunits")
        if excl:
            icu = icu[~icu[c["careunit"]].isin(excl)]
        ep = pd.DataFrame({
            "subject_id": icu[c["subject"]].str.strip(),
            "start": pd.to_datetime(icu[c["intime"]], errors="coerce"),
            "end": pd.to_datetime(icu[c["outtime"]], errors="coerce"),
            "unit": icu[c["careunit"]],
        })
        pat = pd.DataFrame({
            "subject_id": pat[c["subject"]].str.strip(),
            "dob": pd.to_datetime(pat[c["dob"]], errors="coerce"),
            "sex": map_sex(pat[c["gender"]], self.cfg.get("sex_map")),
        }).drop_duplicates("subject_id")
        ep = ep.merge(pat, on="subject_id", how="left").dropna(subset=["start", "end", "dob"])
        if self.cfg.get("dob_from_same_day_admission"):
            # DOB has no time of day (MIMIC-III). If the first ICU admission is on the birth
            # date, use that admission time as the birth time: hours of life are then
            # under- rather than over-estimated, so early (maternal) creatinine is never used.
            ep["dob"] = ep["dob"].astype("datetime64[ns]")
            first = ep.groupby("subject_id")["start"].transform("min").astype("datetime64[ns]")
            same_day = first.dt.normalize() == ep["dob"].dt.normalize()
            ep["dob"] = ep["dob"].where(~same_day, first)
            log.info("DOB set to first ICU admission time for %d of %d stays (same-day admission)",
                     int(same_day.sum()), len(ep))
        return ep[ep["end"] > ep["start"]][CANONICAL_COLUMNS["episodes"]]

    def _events(self, table: str, spec_key: str) -> pd.DataFrame:
        spec = dict(self.cfg.get(spec_key) or {})
        if not spec.get("map"):
            log.warning("%s: no ITEMID map configured — skipped", spec_key)
            return empty("labs")
        c = self.cfg["columns"]
        spec["path"] = self._path(table)
        present = set(pd.read_csv(spec["path"], nrows=0).columns)
        vt = c.get("value") if spec.get("use_value_text", True) else None
        if vt and vt not in present:
            log.info("%s: no '%s' text column in %s — numeric values only", spec_key, vt, table)
            vt = None
        spec["columns"] = {"subject": c["subject"], "time": c["charttime"],
                           "code": c["itemid"], "value": c["valuenum"],
                           "unit": c.get("valueuom"),
                           # text result (e.g. '<14') used where VALUENUM is blank — as GOSH
                           "value_text": vt}
        return read_mapped_events(spec, spec_key, spec.get("default_units"),
                                  report=self.unit_report)

    def labs(self) -> pd.DataFrame:
        return self._events("labevents", "labs")

    def vitals(self) -> pd.DataFrame:
        return self._events("chartevents", "vitals")

    def meds(self) -> pd.DataFrame:
        m = self.cfg.get("prescriptions") or {}
        parts = []
        if m:
            mc = m["columns"]
            usecols = [v for v in mc.values() if v]
            df = pd.read_csv(self._path("prescriptions"), usecols=usecols, dtype=str)
            # 'end_of_day' only for sources with dates but no times (MIMIC-III); 'none' when
            # the start time is real (PIC), so a genuine 00:00 start is not moved.
            policy = m.get("date_only_policy", "end_of_day")
            start = date_only_shift(df[mc["start"]], policy)
            end = date_only_shift(df[mc["end"]], policy) if mc.get("end") \
                else pd.Series(pd.NaT, index=df.index)
            # missing end: rows that are single administrations (PIC) -> point event, like the
            # GOSH administration rows (missing_end_hours: 0); orders without an end (MIMIC)
            # -> one day (24). The drug group's 'on therapy' carry is added later either way.
            end = end.fillna(start + pd.Timedelta(hours=float(m.get("missing_end_hours", 24))))
            name = df[mc["name"]].fillna("")
            if mc.get("name_alt"):
                name = name + " " + df[mc["name_alt"]].fillna("")
            route = route_for(df[mc["route"]] if mc.get("route") else None, name, m)
            parts.append(pd.DataFrame({
                "subject_id": df[mc["subject"]].str.strip(), "start": start, "end": end,
                "name": name.str.lower(), "route": route}))
        for spec in self.cfg.get("inputevents") or []:
            parts.append(self._inputevents(spec))
        if not parts:
            return empty("meds")
        out = pd.concat(parts, ignore_index=True).dropna(subset=["start", "end"])
        return out[CANONICAL_COLUMNS["meds"]]

    def _inputevents(self, spec: dict) -> pd.DataFrame:
        """Infusions (e.g. vasoactives) from INPUTEVENTS_CV / _MV via D_ITEMS labels."""
        c = self.cfg["columns"]
        d = pd.read_csv(self._path("d_items"), dtype=str, usecols=[c["itemid"], "LABEL"])
        label = dict(zip(d[c["itemid"]].str.strip(), d["LABEL"].str.lower()))
        cols = [c["subject"], c["itemid"], spec["start"]] + ([spec["end"]] if spec.get("end")
                                                            else [])
        parts = []
        for ch in pd.read_csv(self._path(spec["table"]), usecols=cols, dtype=str,
                              chunksize=1_000_000):
            ch = ch[ch[c["itemid"]].isin([str(i) for i in spec["itemids"]])]
            if ch.empty:
                continue
            st = pd.to_datetime(ch[spec["start"]], errors="coerce")
            en = pd.to_datetime(ch[spec["end"]], errors="coerce") if spec.get("end") else st
            parts.append(pd.DataFrame({"subject_id": ch[c["subject"]].str.strip(),
                                       "start": st, "end": en,
                                       "name": ch[c["itemid"]].map(label).fillna(""),
                                       "route": "iv"}))
        return pd.concat(parts, ignore_index=True) if parts else empty("meds")

    def rrt(self) -> pd.DataFrame:
        """Renal replacement events from charted ITEMIDs and/or drug names
        (e.g. dialysate, haemofiltration replacement fluid) when no procedure table exists."""
        r = self.cfg.get("rrt") or {}
        c = self.cfg["columns"]
        out = []
        if (r.get("drug_name_regex") or r.get("route_values")) and self.cfg.get("prescriptions"):
            mc = self.cfg["prescriptions"]["columns"]
            cols = [mc["subject"], mc["start"], mc["name"]] + \
                ([mc["name_alt"]] if mc.get("name_alt") else []) + \
                ([mc["route"]] if mc.get("route") and r.get("route_values") else [])
            rx = pd.read_csv(self._path("prescriptions"), usecols=cols, dtype=str)
            name = rx[mc["name"]].fillna("")
            if mc.get("name_alt"):
                name = name + " " + rx[mc["name_alt"]].fillna("")
            hit = pd.Series(False, index=rx.index)
            if r.get("drug_name_regex"):
                hit |= name.str.lower().str.contains(r["drug_name_regex"], regex=True)
            if r.get("route_values") and mc.get("route"):
                # e.g. MIMIC-III ROUTE 'DIALYS' (dialysate) / 'REPLACE' (CRRT replacement fluid)
                hit |= rx[mc["route"]].astype(str).str.strip().str.upper().isin(
                    [str(v).upper() for v in r["route_values"]])
            log.info("rrt: %d prescription rows match the drug-name / route rules", hit.sum())
            if mc.get("start") and self.cfg["prescriptions"].get("date_only_policy") == "end_of_day":
                rx[mc["start"]] = date_only_shift(rx[mc["start"]], "end_of_day").astype(str)
            out.append(pd.DataFrame({"subject_id": rx.loc[hit, mc["subject"]].str.strip(),
                                     "time": pd.to_datetime(rx.loc[hit, mc["start"]],
                                                            errors="coerce")}))
        if not r.get("itemids"):
            return pd.concat(out, ignore_index=True).dropna() if out else empty("rrt")
        for table in r.get("tables", ["chartevents"]):
            for ch in pd.read_csv(self._path(table), usecols=[c["subject"], c["itemid"],
                                                               c["charttime"]],
                                  dtype=str, chunksize=1_000_000):
                ch = ch[ch[c["itemid"]].isin([str(i) for i in r["itemids"]])]
                out.append(pd.DataFrame({"subject_id": ch[c["subject"]].str.strip(),
                                         "time": pd.to_datetime(ch[c["charttime"]],
                                                                errors="coerce")}))
        return pd.concat(out, ignore_index=True).dropna() if out else empty("rrt")