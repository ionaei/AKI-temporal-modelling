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
        return ep[ep["end"] > ep["start"]][CANONICAL_COLUMNS["episodes"]]

    def _events(self, table: str, spec_key: str) -> pd.DataFrame:
        spec = dict(self.cfg.get(spec_key) or {})
        if not spec.get("map"):
            log.warning("%s: no ITEMID map configured — skipped", spec_key)
            return empty("labs")
        c = self.cfg["columns"]
        spec["path"] = self._path(table)
        spec["columns"] = {"subject": c["subject"], "time": c["charttime"],
                           "code": c["itemid"], "value": c["valuenum"],
                           "unit": c.get("valueuom")}
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
            policy = m.get("date_only_policy", "end_of_day")
            start = date_only_shift(df[mc["start"]], policy)
            end = date_only_shift(df[mc["end"]], policy)
            # missing end -> one day of therapy
            end = end.fillna(start + pd.Timedelta(hours=24))
            name = df[mc["name"]].fillna("")
            if mc.get("name_alt"):
                name = name + " " + df[mc["name_alt"]].fillna("")
            parts.append(pd.DataFrame({
                "subject_id": df[mc["subject"]].str.strip(), "start": start, "end": end,
                "name": name.str.lower(),
                "route": df[mc["route"]] if mc.get("route") else None}))
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
        r = self.cfg.get("rrt") or {}
        if not r.get("itemids"):
            return empty("rrt")
        c = self.cfg["columns"]
        out = []
        for table in r.get("tables", ["chartevents"]):
            for ch in pd.read_csv(self._path(table), usecols=[c["subject"], c["itemid"],
                                                               c["charttime"]],
                                  dtype=str, chunksize=1_000_000):
                ch = ch[ch[c["itemid"]].isin([str(i) for i in r["itemids"]])]
                out.append(pd.DataFrame({"subject_id": ch[c["subject"]].str.strip(),
                                         "time": pd.to_datetime(ch[c["charttime"]],
                                                                errors="coerce")}))
        return pd.concat(out, ignore_index=True).dropna() if out else empty("rrt")
