"""Propose / audit code -> concept mappings for a site. Read-only.

GOSH : lists every lab basename with unit, count, median and 5-95th centile,
       plus which concept it is currently mapped to; top medication names with
       their group and route values; procedure-code prefixes.
PIC / MIMIC-III : searches D_LABITEMS / D_ITEMS labels with the regexes below,
       then counts the candidate ITEMIDs in LABEVENTS / CHARTEVENTS with units
       and medians.

Outputs go to data/<site>/discover/. Review them by hand and copy the codes
you accept into the site YAML. Nothing is mapped automatically.

    python tools/discover_codes.py --site gosh
    python tools/discover_codes.py --site pic --skip-chartevents
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aki.adapters.base import invert_map  # noqa: E402
from aki.concepts import med_groups_for, route_class  # noqa: E402
from aki.config import load_config  # noqa: E402

# (include regex, exclude regex) on lower-cased labels
LAB_PATTERNS = {
    "creatinine": (r"\bcreatinine\b", r"urine|clearance|ratio|fluid|ascites|pleural|csf"),
    "urea": (r"\burea\b|\bbun\b|urea nitrogen", r"urine|fluid"),
    "sodium": (r"\bsodium\b|^na\b", r"urine|fluid|stool"),
    "potassium": (r"\bpotassium\b|^k\b", r"urine|fluid|stool"),
    "chloride": (r"\bchloride\b", r"urine|fluid|sweat"),
    "bicarbonate": (r"bicarbonate|\bhco3\b|total co2", r"urine|fluid"),
    "glucose": (r"\bglucose\b", r"urine|csf|fluid|pleural"),
    "calcium_total": (r"\bcalcium\b", r"urine|ionized|ionised|free"),
    "calcium_ionised": (r"(ioni[sz]ed|free) calcium|calcium,? (ioni[sz]ed|free)", r"urine"),
    "phosphate": (r"phosph", r"urine|alkaline|fluid"),
    "magnesium": (r"magnesium", r"urine"),
    "albumin": (r"\balbumin\b", r"urine|ratio|fluid|csf|pleural|ascites|micro"),
    "bilirubin_total": (r"bilirubin,? total|total bilirubin", r"urine|fluid"),
    "alt": (r"\balt\b|alanine aminotransferase|\bsgpt\b|\bgpt\b", r""),
    "crp": (r"c-reactive|\bcrp\b", r""),
    "haemoglobin": (r"h(a)?emoglobin|\bhgb\b|\bhb\b", r"a1c|urine|carboxy|met|fetal|free"),
    "haematocrit": (r"h(a)?ematocrit|\bhct\b", r"fluid|ascites|pleural"),
    "wbc": (r"white blood|\bwbc\b|leukocyte", r"urine|fluid|csf|stool|pleural|ascites"),
    "platelets": (r"platelet", r"function|aggreg|antibod"),
    "lactate": (r"lactate|lactic", r"dehydrogenase|ldh|csf"),
    "ph": (r"^ph$|\bph\b", r"urine|stool|fluid|pleural"),
    "pco2": (r"pco2|p co2|partial pressure.*carbon", r""),
    "po2": (r"\bpo2\b|p o2|partial pressure.*oxygen", r"so2"),
    "base_excess": (r"base excess|\bbe\b", r""),
    "fio2": (r"fio2|inspired o2|oxygen fraction|o2 fraction", r""),
}
RRT_PATTERN = (r"crrt|cvvh|cvvhd|cvvhdf|dialys|haemofil|hemofil|ultrafiltr|replacement fluid"
               r"|dialysate|\bpd\b|peritoneal|\bhd\b|plasma exchange|ecmo", r"")
VITAL_PATTERNS = {
    "hr": (r"heart rate|^hr\b|pulse", r"alarm|target|rhythm"),
    "rr": (r"resp(iratory)? rate|^rr\b", r"alarm|set|vent|spont"),
    "spo2": (r"spo2|o2 sat|oxygen saturation|pulse ?ox", r"alarm|venous|sv02|svo2"),
    "temp": (r"temp", r"alarm|skin|site"),
    "sbp": (r"systolic|\bsbp\b|bp.*sys", r"alarm|pulmonary|pap"),
    "dbp": (r"diastolic|\bdbp\b|bp.*dia", r"alarm|pulmonary|pap"),
    "map": (r"\bmean\b.*(arterial|bp|pressure)|\bmap\b", r"alarm|pulmonary|pap|airway"),
    "fio2": (r"fio2|inspired o2|o2 fraction", r"alarm"),
    "vent_invasive": (r"vent(ilator)? mode|vent type|\bpeep\b", r"non-?invasive|bipap|cpap"),
}


def _match(labels: pd.Series, inc, exc):
    import warnings
    warnings.filterwarnings("ignore", message="This pattern is interpreted as a regular")
    s = labels.astype(str).str.lower()
    m = s.str.contains(inc, regex=True)
    if exc:
        m &= ~s.str.contains(exc, regex=True)
    return m


def _summarise_events(path, item_col, val_col, uom_col, items, chunksize=2_000_000):
    agg = []
    for ch in pd.read_csv(path, usecols=[item_col, val_col, uom_col], chunksize=chunksize,
                          dtype={item_col: str, uom_col: str}):
        ch = ch[ch[item_col].isin(items)]
        if len(ch):
            agg.append(ch)
    if not agg:
        return pd.DataFrame()
    df = pd.concat(agg)
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    return (df.groupby([item_col, uom_col], dropna=False)[val_col]
              .agg(n="size", median="median", p05=lambda x: x.quantile(.05),
                   p95=lambda x: x.quantile(.95)).reset_index())


def discover_mimic_like(cfg, out: Path, skip_chart: bool):
    s = cfg["site"]
    raw = Path(s["raw_dir"])
    c = s["columns"]
    # renal replacement / extracorporeal items (for the RRT exclusion), chart + input items
    d_items = pd.read_csv(raw / s["files"]["d_items"], dtype=str)
    lab = s.get("discover", {}).get("d_items_label_col", "LABEL")
    r = d_items[_match(d_items[lab], *RRT_PATTERN)]
    r.to_csv(out / "rrt_candidates.csv", index=False)
    print(f"rrt: {len(r)} candidate D_ITEMS rows -> {out / 'rrt_candidates.csv'}")
    # infusion items (e.g. INPUTEVENTS_CV vasoactives) that match a medication group
    link = "LINKSTO" if "LINKSTO" in d_items.columns else None
    inf = d_items.copy()
    inf["groups"] = inf[lab].fillna("").str.lower().map(lambda x: ",".join(med_groups_for(x)))
    inf = inf[inf["groups"] != ""]
    if link:
        inf = inf[inf[link].astype(str).str.contains("input", case=False, na=False)]
    inf.to_csv(out / "infusion_item_candidates.csv", index=False)
    print(f"infusions: {len(inf)} D_ITEMS input items matching a medication group -> "
          f"{out / 'infusion_item_candidates.csv'}")
    if len(inf):
        print(inf[[c for c in ("ITEMID", lab, link, "DBSOURCE", "groups") if c]]
              .to_string(index=False))
    # medications: names, mapped groups, route classes (counts only)
    m = s.get("prescriptions") or {}
    if m:
        mc = m["columns"]
        cols = [v for v in (mc.get("name"), mc.get("name_alt"), mc.get("route")) if v]
        rx = pd.read_csv(raw / s["files"]["prescriptions"], usecols=cols, dtype=str)
        name = rx[mc["name"]].fillna("")
        if mc.get("name_alt"):
            name = name + " " + rx[mc["name_alt"]].fillna("")
        t = pd.DataFrame({"name": name.str.lower().str.strip()})
        if mc.get("route"):
            t["route"] = rx[mc["route"]]
        top = t.value_counts().reset_index(name="n")
        top["groups"] = top["name"].map(lambda x: ",".join(med_groups_for(x)))
        from aki.adapters.mimic_like import route_for
        rt = route_for(top["route"] if "route" in top else None, top["name"], m)
        top["route_class"] = (rt if "route" in top or m.get("route_name_rules")
                              else top["name"]).map(route_class)
        if mc.get("route"):
            print("route column values:\n" + rx[mc["route"]].value_counts(dropna=False)
                  .to_string())
        rr = (s.get("rrt") or {}).get("drug_name_regex") or RRT_PATTERN[0]
        rrt_names = top[top["name"].str.contains(rr, regex=True)]
        rrt_names.to_csv(out / "rrt_drug_names.csv", index=False)
        print(f"rrt-like drug names: {len(rrt_names)} -> {out / 'rrt_drug_names.csv'}")
        top.to_csv(out / "medication_names.csv", index=False)
        g = top[top["groups"] != ""]
        print(f"meds: {len(top)} name/route combinations -> {out / 'medication_names.csv'}")
        print(g.groupby(["groups", "route_class"])["n"].sum().sort_values(ascending=False)
                .head(40).to_string())
    for kind, dtable, etable, patterns in (
            ("labs", "d_labitems", "labevents", LAB_PATTERNS),
            ("vitals", "d_items", "chartevents", VITAL_PATTERNS)):
        if kind == "vitals" and skip_chart:
            continue
        d = pd.read_csv(raw / s["files"][dtable], dtype=str)
        label_col = s.get("discover", {}).get(f"{dtable}_label_col", "LABEL")
        rows = []
        for concept, (inc, exc) in patterns.items():
            hit = d[_match(d[label_col], inc, exc)]
            for _, r in hit.iterrows():
                rows.append(dict(concept=concept, itemid=str(r[c["itemid"]]),
                                 label=r[label_col],
                                 **{k: r[k] for k in ("FLUID", "CATEGORY", "LOINC_CODE",
                                                      "DBSOURCE", "UNITNAME") if k in r}))
        cand = pd.DataFrame(rows)
        if cand.empty:
            continue
        ev = _summarise_events(raw / s["files"][etable], c["itemid"], c["valuenum"],
                               c["valueuom"], set(cand["itemid"]))
        if len(ev):
            cand = cand.merge(ev, left_on="itemid", right_on=c["itemid"], how="left")
        cand.to_csv(out / f"{kind}_candidates.csv", index=False)
        print(f"{kind}: {len(cand)} candidate rows -> {out / f'{kind}_candidates.csv'}")


def discover_gosh(cfg, out: Path):
    s = cfg["site"]
    L = s["labs"]
    c = L["columns"]
    code2concept = invert_map(L["map"])
    parts = []
    for ch in pd.read_csv(L["path"], usecols=[c["code"], c["value"], c["unit"]],
                          chunksize=1_000_000, dtype={c["code"]: str, c["unit"]: str}):
        ch[c["value"]] = pd.to_numeric(ch[c["value"]], errors="coerce")
        parts.append(ch)
    df = pd.concat(parts)
    tab = (df.groupby([c["code"], c["unit"]], dropna=False)[c["value"]]
             .agg(n="size", median="median", p05=lambda x: x.quantile(.05),
                  p95=lambda x: x.quantile(.95)).reset_index())
    tab["mapped_to"] = tab[c["code"]].astype(str).str.upper().map(code2concept)
    # suggestions only — never used by the pipeline
    for concept, (inc, exc) in LAB_PATTERNS.items():
        sug = _match(tab[c["code"]].astype(str), inc, exc) & tab["mapped_to"].isna()
        tab.loc[sug, "regex_suggests"] = concept
    tab.sort_values("n", ascending=False).to_csv(out / "lab_basenames.csv", index=False)
    print(f"labs: {len(tab)} basename/unit combinations -> {out / 'lab_basenames.csv'}")

    m = s["medications"]
    mc = m["columns"]
    cols = [v for v in (mc["name"], mc.get("route")) if v]
    meds = pd.read_csv(m["path"], usecols=cols, dtype=str)
    top = meds.value_counts().reset_index(name="n")
    top["groups"] = top[mc["name"]].map(lambda x: ",".join(med_groups_for(x)))
    if mc.get("route"):
        top["route_class"] = top[mc["route"]].map(route_class)
    else:   # what route_from_name would infer
        top["route_class_from_name"] = top[mc["name"]].map(route_class)
    top.to_csv(out / "medication_names.csv", index=False)
    print(f"meds: {len(top)} name/route combinations -> {out / 'medication_names.csv'}")
    grouped = top[top["groups"] != ""]
    print(grouped.groupby("groups")["n"].sum().sort_values(ascending=False).to_string())

    p = s.get("procedures") or {}
    if p.get("path"):
        pr = pd.read_csv(p["path"], usecols=[p["columns"]["code"]], dtype=str)
        code = pr[p["columns"]["code"]].str.upper().str.replace(".", "", regex=False)
        code.str[:3].value_counts().head(200).rename("n").to_csv(out / "procedure_prefixes.csv")
        print(f"procedures: prefix counts -> {out / 'procedure_prefixes.csv'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    ap.add_argument("--skip-chartevents", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.site)
    out = Path(cfg["site"]["dirs"]["root"]) / "discover"
    out.mkdir(parents=True, exist_ok=True)
    if cfg["site"]["adapter"] == "gosh":
        discover_gosh(cfg, out)
    else:
        discover_mimic_like(cfg, out, a.skip_chartevents)


if __name__ == "__main__":
    main()