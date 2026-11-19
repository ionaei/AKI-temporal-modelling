"""Great Ormond Street Hospital (DRE extract) adapter.

All file names, column names and code lists live in configs/sites/gosh.yaml.
"""
from __future__ import annotations

import logging

import pandas as pd

from .base import (Adapter, CANONICAL_COLUMNS, empty, map_sex, read_mapped_events,
                   require_columns)

log = logging.getLogger(__name__)


class GOSHAdapter(Adapter):

    def _icu_ward_stays(self) -> pd.DataFrame:
        """ICU time only: ward-stay rows flagged as ICU become the care episodes.

        Contiguous ICU ward stays (e.g. PICU -> CICU bed moves) are merged into
        one ICU stay later by cohort.build_stays. No admission_id is built
        from row order, so the result does not depend on how the file is sorted.
        """
        w = self.cfg["ward_stays"]
        c = w["columns"]
        ws = pd.read_csv(w["path"], dtype=str)
        require_columns(ws.columns, {k: v for k, v in c.items() if v}, "ward_stays")
        n0 = len(ws)
        is_icu = pd.Series(False, index=ws.index)
        if c.get("icu_flag"):
            ok = {str(x).strip().lower() for x in w.get("icu_values", [1, "1", "true", "y"])}
            is_icu |= ws[c["icu_flag"]].astype(str).str.strip().str.lower().isin(ok)
        if w.get("icu_wards") and c.get("ward"):
            is_icu |= ws[c["ward"]].isin(w["icu_wards"])
        ws = ws[is_icu]
        log.info("ward stays: %d rows -> %d ICU ward stays (%d patients)",
                 n0, len(ws), ws[c["subject"]].nunique())
        return pd.DataFrame({
            "subject_id": ws[c["subject"]].str.strip(),
            "start": pd.to_datetime(ws[c["start"]], errors="coerce"),
            "end": pd.to_datetime(ws[c["end"]], errors="coerce"),
            "unit": ws[c["ward"]] if c.get("ward") else "ICU",
        })

    def episodes(self) -> pd.DataFrame:
        d = self.cfg["demographics"]
        if (self.cfg.get("ward_stays") or {}).get("path"):
            ep = self._icu_ward_stays()
        else:
            log.warning("no ward_stays configured: using whole hospital admissions "
                        "(includes non-ICU time)")
            a = self.cfg["admissions"]
            adm = pd.read_csv(a["path"], dtype=str)
            require_columns(adm.columns, a["columns"], "admissions")
            c = a["columns"]
            for col, allowed in (a.get("filter") or {}).items():
                adm = adm[adm[col].isin(allowed)]
            ep = pd.DataFrame({
                "subject_id": adm[c["subject"]].str.strip(),
                "start": pd.to_datetime(adm[c["start"]], errors="coerce"),
                "end": pd.to_datetime(adm[c["end"]], errors="coerce"),
                "unit": adm[c["unit"]] if c.get("unit") else "",
            })
        dem = pd.read_csv(d["path"], dtype=str)
        dc = d["columns"]
        require_columns(dem.columns, dc, "demographics")
        dem = pd.DataFrame({
            "subject_id": dem[dc["subject"]].str.strip(),
            "dob": pd.to_datetime(dem[dc["dob"]], errors="coerce"),
            "sex": map_sex(dem[dc["sex"]], d.get("sex_map")),
        }).drop_duplicates("subject_id")
        ep = ep.merge(dem, on="subject_id", how="left")
        n0 = len(ep)
        ep = ep.dropna(subset=["start", "end", "dob"])
        ep = ep[ep["end"] > ep["start"]]
        log.info("episodes: %d -> %d after dropping missing/negative times", n0, len(ep))
        return ep[CANONICAL_COLUMNS["episodes"]]

    def labs(self) -> pd.DataFrame:
        return read_mapped_events(self.cfg["labs"], "labs",
                                  self.cfg["labs"].get("default_units"),
                                  report=self.unit_report)

    def vitals(self) -> pd.DataFrame:
        spec = self.cfg.get("vitals") or {}
        if not spec.get("path"):
            log.warning("GOSH vitals not configured — vital-sign features will be absent")
            return empty("vitals")
        return read_mapped_events(spec, "vitals", spec.get("default_units"),
                                  report=self.unit_report)

    def meds(self) -> pd.DataFrame:
        m = self.cfg["medications"]
        c = m["columns"]
        usecols = [v for v in c.values() if v]
        hdr = pd.read_csv(m["path"], nrows=0)
        require_columns(hdr.columns, {k: v for k, v in c.items() if v}, "medications")
        parts = []
        for ch in pd.read_csv(m["path"], usecols=usecols, dtype=str, chunksize=500_000):
            start_col = c.get("start") or c["end"]
            parts.append(pd.DataFrame({
                "subject_id": ch[c["subject"]].str.strip(),
                "start": pd.to_datetime(ch[start_col], errors="coerce"),
                "end": pd.to_datetime(ch[c["end"]], errors="coerce"),
                "name": ch[c["name"]].fillna("").str.lower(),
                "route": ch[c["route"]] if c.get("route") else None,
                "gpi": ch[c["gpi"]] if c.get("gpi") else None,
            }))
        df = pd.concat(parts, ignore_index=True)
        # Optional: GOSH local GPI codes -> an ingredient name the group regex understands
        gpi_map = {str(k): v for k, v in (m.get("gpi_to_name") or {}).items()}
        if gpi_map and df["gpi"].notna().any():
            def _gpi_name(g):
                g = str(g)
                for prefix, name in gpi_map.items():
                    if g.startswith(prefix):
                        return name
                return None
            extra = df["gpi"].map(_gpi_name)
            df["name"] = (df["name"] + " " + extra.fillna("")).str.strip()
        # No route column (or empty): optionally infer the route from the drug-name text,
        # e.g. "Vancomycin 125mg oral solution", "Tobramycin nebuliser". Rows whose name
        # carries no route word stay 'systemic' (counted), same as with route unknown.
        if m.get("route_from_name"):
            missing = df["route"].isna() | (df["route"].astype(str).str.strip() == "")
            df.loc[missing, "route"] = df.loc[missing, "name"]
            log.info("route inferred from drug name for %d of %d rows", missing.sum(), len(df))
        df = df.dropna(subset=["start", "end"])
        return df[CANONICAL_COLUMNS["meds"]]

    def rrt(self) -> pd.DataFrame:
        p = self.cfg.get("procedures") or {}
        codes = tuple(str(x).upper() for x in (p.get("rrt_code_prefixes") or []))
        if not p.get("path") or not codes:
            return empty("rrt")
        c = p["columns"]
        df = pd.read_csv(p["path"], usecols=[c["subject"], c["code"], c["time"]], dtype=str)
        code = df[c["code"]].astype(str).str.upper().str.replace(".", "", regex=False)
        hit = code.str.startswith(tuple(x.replace(".", "") for x in codes))
        out = pd.DataFrame({"subject_id": df.loc[hit, c["subject"]].str.strip(),
                            "time": pd.to_datetime(df.loc[hit, c["time"]], errors="coerce")})
        return out.dropna()
