"""Collect every run's results into grouped comparison tables and one HTML report.

    python tools/compare_runs.py                 # everything under runs/
    python tools/compare_runs.py --site gosh     # only evaluations on that site

Writes to results/:
  main_table_<operating point>.csv   one row per run x evaluation: key metrics, mean ± SD over seeds
  horizon_auroc.csv                  AUROC by hours before onset (rows = runs)
  subgroups_<operating point>.csv    age-band results, mean over seeds
  explain_group_share.csv            share of attribution per feature group (rows = runs)
  explain_top_features.csv           top features per run, mean over seeds
  explain_differential.csv           importance difference AKI minus non-AKI (feature-importance figure)
  deploy_monthly.csv                 silent-deployment metrics per month
  all_metrics_long.csv               every number, one row per run/eval/op-point/seed/metric
  report.html                        all of the above grouped, with figures

Runs are grouped automatically from run_config.json:
  1 Main models     all feature groups, sequence models
  2 Baselines       logistic regression / gradient boosting
  3 Ablations       a subset of feature groups (labelled 'no process', 'only process', ...)
  4 Cross-site      restricted to the cross-site core feature set
and evaluations are split into test-set, hold-out deployment, and other-site results.

All outputs are aggregates over thousands of windows; still apply the usual
small-number check before exporting from the DRE.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FULL_GROUPS = {"static", "time", "creatinine", "lab", "vital", "support", "med", "process"}
SEQ = {"gru", "lstm", "transformer"}
KEY_METRICS = ["auroc", "auprc", "prevalence", "brier", "cal_slope", "cal_intercept",
               "threshold", "sens@24h", "sens@48h", "sens_severe@48h", "precision@24h",
               "precision@48h", "lead@48h_median", "alerts_per_100_patient_days",
               "nonaki_stays_never_alerted", "legacy_sens_ever",
               "legacy_spec_stay", "legacy_youden_stay", "legacy_first_alert_median_h"]
GROUP_ORDER = ["1 Main models", "2 Baselines", "3 Ablations", "4 Cross-site (core features)"]
EVAL_ORDER = ["test", "hold-out deployment", "other site", "other"]


# ── discovery ─────────────────────────────────────────────────────────────
def describe_run(rc: dict, available_groups: set) -> tuple[str, str]:
    """-> (category, short label for the feature set)."""
    groups = set(rc.get("groups") or [])
    full = FULL_GROUPS & available_groups if available_groups else FULL_GROUPS
    if rc.get("core"):
        cat = "4 Cross-site (core features)"
    elif rc.get("model") not in SEQ:
        cat = "2 Baselines"
    elif groups >= full:
        cat = "1 Main models"
    else:
        cat = "3 Ablations"
    missing = sorted(full - groups)
    kept = sorted(groups - {"static", "time"})
    if not missing:
        feat = "all features"
    elif len(kept) <= 2:
        feat = "only " + "+".join(kept)
    else:
        feat = "no " + "+".join(missing)
    if rc.get("core"):
        feat += " (core)"
    return cat, feat


def eval_kind(tag: str, train_site: str) -> tuple[str, str, str]:
    """eval_<site>_<split>[_recal] / deploy_<site>_<split> -> (kind, site, split)."""
    m = re.match(r"(eval|deploy)_(.+)_(train|val|test|dev|holdout|all)(_recal)?$", tag)
    if not m:
        return "other", "?", "?"
    what, site, split, recal = m.groups()
    split = split + ("+recal" if recal else "")
    if what == "deploy" or split.startswith("holdout"):
        return "hold-out deployment", site, split
    if site != train_site:
        return "other site", site, split
    return ("test" if split.startswith("test") else "other"), site, split


def collect(site_filter: str | None):
    rows, avail = [], set()
    for f in (ROOT / "data").glob("*/windows/feature_groups.json"):
        avail |= set(json.loads(f.read_text()).values())
    for rc_path in sorted((ROOT / "runs").glob("*/seed_*/run_config.json")):
        sd = rc_path.parent
        rc = json.loads(rc_path.read_text())
        cat, feat = describe_run(rc, avail)
        for ev in sorted(p for p in sd.iterdir() if p.is_dir()
                         and p.name.startswith(("eval_", "deploy_"))):
            kind, site, split = eval_kind(ev.name, rc["site"])
            if site_filter and site != site_filter:
                continue
            loss = (rc.get("common", {}).get("training", {}).get("loss", "?")
                    if rc.get("model") in SEQ else "-")
            rows.append(dict(run=sd.parent.name, seed=rc.get("seed"), model=rc.get("model"),
                             loss=loss,
                             train_site=rc["site"], category=cat, features=feat,
                             eval_kind=kind, eval_site=site, eval_split=split, path=ev,
                             seed_dir=sd))
    return pd.DataFrame(rows)


# ── tables ────────────────────────────────────────────────────────────────
def _fmt(mean, sd, n, metric):
    if pd.isna(mean):
        return ""
    d = 1 if metric in ("lead@48h_median", "alerts_per_100_patient_days",
                        "legacy_first_alert_median_h") else 3
    if n > 1 and pd.notna(sd):
        return f"{mean:.{d}f} ± {sd:.{d}f}"
    return f"{mean:.{d}f}"


def long_metrics(idx: pd.DataFrame) -> pd.DataFrame:
    out = []
    for r in idx.itertuples():
        f = r.path / "summary.csv"
        if not f.exists():
            continue
        s = pd.read_csv(f, index_col=0)
        for op in s.columns:
            for metric, v in pd.to_numeric(s[op], errors="coerce").items():
                out.append(dict(run=r.run, category=r.category, features=r.features,
                                model=r.model, loss=r.loss, eval_kind=r.eval_kind, eval_site=r.eval_site,
                                eval_split=r.eval_split, op=op, seed=r.seed, metric=metric,
                                value=v))
    return pd.DataFrame(out)


def main_tables(lm: pd.DataFrame) -> dict:
    keys = ["category", "run", "model", "loss", "features", "eval_kind", "eval_site",
            "eval_split"]
    tables = {}
    for op, d in lm.groupby("op"):
        g = d.groupby(keys + ["metric"])["value"].agg(["mean", "std", "count"]).reset_index()
        g["txt"] = [_fmt(m, s, n, k) for m, s, n, k in zip(g["mean"], g["std"], g["count"],
                                                            g["metric"])]
        w = g.pivot_table(index=keys, columns="metric", values="txt", aggfunc="first")
        n = d.groupby(keys)["seed"].nunique().rename("n_seeds")
        w = w.join(n)
        cols = ["n_seeds"] + [m for m in KEY_METRICS if m in w.columns]
        w = w[cols].reset_index()
        w["_c"] = w["category"].map({c: i for i, c in enumerate(GROUP_ORDER)})
        w["_e"] = w["eval_kind"].map({c: i for i, c in enumerate(EVAL_ORDER)})
        tables[op] = w.sort_values(["_e", "_c", "model", "loss", "run", "eval_site",
                                    "eval_split"]) \
                      .drop(columns=["_c", "_e"])
    return tables


def mean_over_seeds(idx, fname, keys, index_col=None):
    parts = []
    for r in idx.itertuples():
        f = r.path / fname
        if f.exists():
            d = pd.read_csv(f, index_col=index_col)
            parts.append(d.assign(run=r.run, eval_kind=r.eval_kind, eval_site=r.eval_site,
                                  eval_split=r.eval_split, seed=r.seed))
    if not parts:
        return pd.DataFrame()
    d = pd.concat(parts, ignore_index=index_col is None)
    num = d.select_dtypes("number").columns.drop(["seed"] + keys, errors="ignore")
    return d.groupby(["run", "eval_kind", "eval_site", "eval_split"] + keys)[list(num)] \
            .mean().reset_index()


def horizon_table(idx):
    h = mean_over_seeds(idx, "horizon_auroc.csv", ["hours_before_onset"])
    if h.empty:
        return h
    order = ["0-6", "6-12", "12-24", "24-48", "48-72", "72-120"]
    w = h.pivot_table(index=["run", "eval_kind", "eval_site", "eval_split"],
                      columns="hours_before_onset", values="auroc")
    return w[[c for c in order if c in w.columns]].round(3).reset_index()


def explain_tables(seed_dirs: pd.DataFrame):
    shares, feats = [], []
    for r in seed_dirs.itertuples():
        for ex in r.seed_dir.glob("explain_*"):
            f1, f2 = ex / "ig_share_by_group.csv", ex / "ig_by_feature_signed.csv"
            if f1.exists():
                s = pd.read_csv(f1, index_col=0)
                for cls in s.columns:
                    shares.append(s[cls].rename("share").to_frame().reset_index()
                                  .rename(columns={"index": "group"})
                                  .assign(run=r.run, features=r.features, data=ex.name,
                                          cls=cls, seed=r.seed))
            if f2.exists():
                d = pd.read_csv(f2, index_col=0)
                d.index.name = "feature"
                feats.append(d.reset_index().assign(run=r.run, data=ex.name, seed=r.seed))
    sh = pd.concat(shares) if shares else pd.DataFrame()
    if len(sh):
        sh = (sh.groupby(["run", "features", "data", "cls", "group"])["share"].mean()
                .unstack("group").round(3).reset_index())
    top, diff = pd.DataFrame(), pd.DataFrame()
    if feats:
        f = pd.concat(feats)
        agg = f.groupby(["run", "data", "feature", "group"])[
            [c for c in ["aki_window", "non_aki", "abs_mean_aki", "abs_mean_non_aki",
                         "importance_diff", "importance_ratio"] if c in f.columns]].mean()
        top = (agg.reset_index().sort_values(["run", "abs_mean_aki"], ascending=[True, False])
                  .groupby(["run", "data"]).head(15))
        top["direction_in_aki"] = np.where(top["aki_window"] > 0, "towards AKI", "away from AKI")
        if "importance_diff" in agg.columns:      # feature-importance figure: AKI minus non-AKI
            allf = agg.reset_index().sort_values(["run", "data", "importance_diff"],
                                                 ascending=[True, True, False])
            pos = allf[allf["importance_diff"] > 0].groupby(["run", "data"]).head(25)
            neg = allf[allf["importance_diff"] < 0].groupby(["run", "data"]).tail(10)
            diff = pd.concat([pos, neg]).sort_values(["run", "data", "importance_diff"],
                                                     ascending=[True, True, False])
    return sh, top, diff


# ── figures ───────────────────────────────────────────────────────────────
def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return '<img src="data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode() + '">'


def figures(idx, horizon, primary_op):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return {}
    figs = {}
    test = idx[idx["eval_kind"] == "test"]
    # horizon AUROC
    h = horizon[horizon["eval_kind"] == "test"] if len(horizon) else horizon
    if len(h):
        fig, ax = plt.subplots(figsize=(7, 4))
        cols = [c for c in h.columns if re.match(r"^\d", str(c))]
        for _, r in h.iterrows():
            ax.plot(cols, r[cols].astype(float), marker="o", label=r["run"])
        ax.axhline(0.5, ls=":", c="grey")
        ax.set_xlabel("hours before AKI onset"); ax.set_ylabel("AUROC")
        ax.set_title("Discrimination by time before onset (test, mean over seeds)")
        ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
        figs["horizon"] = _png(fig)
    # trade-off: sensitivity@48h vs alert burden, and calibration
    for name, fname, x, y, xl, yl in [
            ("tradeoff", "threshold_sweep.csv", "alerts_per_100_patient_days", "sens@48h",
             "alerts per 100 patient-days", "sensitivity within 48 h"),
            ("precision", "threshold_sweep.csv", "sens@48h", "precision@48h",
             "sensitivity within 48 h", "precision within 48 h"),
            ("calibration", "reliability.csv", "mean_pred", "observed",
             "mean predicted risk", "observed rate"),
            ("decision", "decision_curve.csv", "threshold", "net_benefit",
             "decision threshold probability", "net benefit")]:
        if not any((p / fname).exists() for p in test["path"]):
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        for run, g in test.groupby("run"):
            parts = [pd.read_csv(p / fname) for p in g["path"] if (p / fname).exists()]
            if not parts:
                continue
            if name in ("tradeoff", "precision"):
                p0 = parts[0]   # sweeps differ per seed: show seed with lowest number
                ax.plot(p0[x], p0[y], label=run)
            elif name == "calibration":
                c = pd.concat(parts).groupby(level=0).mean()
                ax.plot(c[x], c[y], marker="o", label=run)
            else:
                c = pd.concat(parts).groupby(x).mean().reset_index()
                ax.plot(c[x], c[y], label=run)
        if name == "calibration":
            lim = ax.get_xlim()[1]
            ax.plot([0, lim], [0, lim], ls=":", c="grey")
        if name == "decision":
            dc = pd.read_csv(test["path"].iloc[0] / fname)
            ax.plot(dc["threshold"], dc["treat_all"], ls="--", c="grey", label="alert all")
            ax.axhline(0, ls=":", c="black", label="alert none")
            ax.set_ylim(min(-0.01, ax.get_ylim()[0] / 4), ax.get_ylim()[1])
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
        figs[name] = _png(fig)
    return figs


# ── report ────────────────────────────────────────────────────────────────
CSS = """body{font-family:Arial,sans-serif;margin:24px;color:#222}
h1{font-size:20px}h2{font-size:17px;margin-top:32px;border-bottom:1px solid #ccc}
h3{font-size:14px;margin:18px 0 4px}table{border-collapse:collapse;font-size:12px;margin:6px 0}
th,td{border:1px solid #ddd;padding:3px 6px;text-align:right}th{background:#f2f2f2}
td:first-child,th:first-child{text-align:left}.note{font-size:12px;color:#555;max-width:900px}
img{max-width:900px}"""


def _html_table(df):
    return df.to_html(index=False, na_rep="", border=0)


def report(tables, horizon, subgroups, shares, top, monthly, figs, out, primary, diff=None):
    h = [f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>",
         "<h1>AKI early warning — comparison of all runs</h1>",
         "<p class='note'>Values are mean ± SD over seeds (one value if one seed). "
         "Threshold-based metrics use thresholds chosen on validation. "
         "sens@48h = AKI stays with ≥1 alert in the 48 h before onset; precision@48h = alerts "
         "followed by onset within 48 h; lead@48h = hours before onset of the earliest alert "
         "within that 48 h; legacy_* = stay-level definitions (AKI stays with any alert before onset; non-AKI stays never alerted).</p>"]
    for op, t in tables.items():
        h.append(f"<h2>Main results — operating point <code>{op}</code>"
                 + (" (primary)" if op == primary else "") + "</h2>")
        for ek in EVAL_ORDER:
            sub = t[t["eval_kind"] == ek]
            if sub.empty:
                continue
            h.append(f"<h3>Evaluated on: {ek}</h3>")
            for cat in GROUP_ORDER:
                s = sub[sub["category"] == cat]
                if len(s):
                    h.append(f"<p><b>{cat}</b></p>" + _html_table(s.drop(columns=["category",
                                                                                   "eval_kind"])))
    if len(horizon):
        h.append("<h2>AUROC by hours before onset</h2>"
                 "<p class='note'>AKI windows in each time bin vs all windows of non-AKI stays. "
                 "0.5 = chance.</p>" + _html_table(horizon))
    for k, title in [("horizon", "Horizon AUROC"), ("tradeoff", "Sensitivity vs alert burden"),
                     ("precision", "Precision vs sensitivity"),
                     ("calibration", "Calibration (test)"), ("decision", "Decision curves")]:
        if k in figs:
            h.append(f"<h3>{title}</h3>{figs[k]}")
    if len(subgroups):
        h.append(f"<h2>Age subgroups — {primary}, mean over seeds</h2>" + _html_table(subgroups))
    if len(shares):
        h.append("<h2>Explainability: share of attribution by feature group</h2>"
                 "<p class='note'>Fraction of total |integrated gradients| per group, for the "
                 "last pre-onset window (aki_window) and random non-AKI windows. A large "
                 "'process' share means test-ordering patterns drive the prediction; compare "
                 "with the 'no process' and 'only process' ablations above.</p>"
                 + _html_table(shares))
    if len(top):
        h.append("<h2>Explainability: top 15 features per run (mean over seeds)</h2>"
                 "<p class='note'>Ranked by mean |attribution| in AKI windows; aki_window and "
                 "non_aki are signed means (positive pushes towards AKI).</p>"
                 + _html_table(top.round(4)))
    if diff is not None and len(diff):
        h.append("<h2>Explainability: differential importance (AKI minus non-AKI)</h2>"
                 "<p class='note'>Mean |integrated gradients| in AKI windows minus non-AKI windows, "
                 "as in the paper's feature-importance figure. Up to 25 most positive and 10 most negative "
                 "features per run (mean over seeds).</p>" + _html_table(diff.round(4)))
    if len(monthly):
        h.append("<h2>Temporal hold-out by month (mean over seeds)</h2>" + _html_table(monthly))
    h.append("</body></html>")
    (out / "report.html").write_text("\n".join(h), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", help="only evaluations on this data site")
    ap.add_argument("--out", default=str(ROOT / "results"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    idx = collect(a.site)
    if idx.empty:
        raise SystemExit("no evaluated runs found under runs/")
    lm = long_metrics(idx)
    lm.to_csv(out / "all_metrics_long.csv", index=False)
    tables = main_tables(lm)
    ops = list(tables)
    primary = "orig_spec_at_sens0.75" if "orig_spec_at_sens0.75" in ops else ops[0]
    for op, t in tables.items():
        t.to_csv(out / f"main_table_{op}.csv", index=False)

    horizon = horizon_table(idx)
    horizon.to_csv(out / "horizon_auroc.csv", index=False)

    sg = mean_over_seeds(idx, "subgroups.csv", ["subgroup"])
    if len(sg):
        keep = ["run", "eval_kind", "eval_site", "eval_split", "subgroup", "auroc", "auprc",
                "prevalence", "cal_intercept", "cal_slope", "sens@48h", "precision@48h",
                "alerts_per_100_patient_days"]
        sg = sg[[c for c in keep if c in sg.columns]].round(3)
        sg.to_csv(out / f"subgroups_{primary}.csv", index=False)

    seed_dirs = idx.drop_duplicates("seed_dir")
    shares, top, diff = explain_tables(seed_dirs)
    shares.to_csv(out / "explain_group_share.csv", index=False)
    top.to_csv(out / "explain_top_features.csv", index=False)
    diff.to_csv(out / "explain_differential.csv", index=False)

    dep = idx[idx["eval_kind"] == "hold-out deployment"]
    monthly = pd.DataFrame()
    parts = []
    for r in dep.itertuples():
        f = r.path / "monthly.csv"
        if f.exists():
            parts.append(pd.read_csv(f).assign(run=r.run, seed=r.seed))
    if parts:
        m = pd.concat(parts)
        cols = [c for c in ["stays", "aki_stays", "auroc", "auprc", "prevalence", "sens@48h",
                            "precision@48h", "alerts_per_100_patient_days", "cal_intercept"]
                if c in m.columns]
        monthly = m.groupby(["run", "month"])[cols].mean().round(3).reset_index()
        monthly.to_csv(out / "deploy_monthly.csv", index=False)

    figs = figures(idx, horizon, primary)
    report(tables, horizon, sg, shares, top, monthly, figs, out, primary, diff)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    t = tables[primary]
    show = ["run", "loss", "features", "eval_kind", "eval_site", "n_seeds", "auroc", "auprc",
            "sens@48h", "precision@48h", "lead@48h_median", "alerts_per_100_patient_days"]
    print(f"\n=== {primary} ===")
    for cat in GROUP_ORDER:
        s = t[t["category"] == cat]
        if len(s):
            print(f"\n{cat}")
            print(s[[c for c in show if c in s.columns]].to_string(index=False))
    print(f"\nwritten to {out}/  (open report.html in a browser)")


if __name__ == "__main__":
    main()