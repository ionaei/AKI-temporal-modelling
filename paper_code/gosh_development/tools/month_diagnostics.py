"""Why does the model do badly in one month of the hold-out?  (default: February 2026)

    python tools/month_diagnostics.py --run runs/gosh_transformer_focal_all --site gosh --month 2026-02
    python tools/month_diagnostics.py --run runs/gosh_transformer_focal_all --site gosh --month 2026-02 --no_windows

Reads, for every seed_* folder of the run, <seed>/deploy_gosh_holdout/predictions.parquet and thresholds.json
(threshold of the operating point --op, default the primary orig_spec_at_sens0.75), plus data/<site>/labelled/{stays_labelled.parquet,split.csv} and,
for the drift checks, data/<site>/windows/windows.parquet (skip it with --no_windows). Everything is evaluated
per seed with that seed's own threshold and then averaged over seeds, as in the paper.

Four questions, in order:
  1  Is the month unusual at all?  The month has only a handful of AKI stays, so AUROC and sensitivity are very
     noisy. The month's value is compared with (a) a patient-level bootstrap interval and (b) 2,000 random draws of
     the same number of AKI and non-AKI stays from the OTHER hold-out months: how often does a random draw look
     as bad as the month?
  2  Which AKI stays were missed?  A per-stay table (console only, never saved) and a comparison of the month's AKI
     stays with the other months' on age, onset time, trigger, reference creatinine, severity and creatinine history,
     plus detection rate by those characteristics across the whole hold-out.
  3  Did the non-AKI side change?  Mean predicted risk and alert rate on non-AKI windows, calibration-in-the-large,
     case mix and alerts per 100 patient-days, month by month.
  4  Did the data change?  Creatinine sampling by month and the standardised shift of every feature in the month
     relative to the other hold-out months and to the development data (needs windows.parquet).

Writes results/month_diagnostics_<month>/*.csv and a 6-panel figure. Aggregates only: counts below 5 are shown as
'<5' in the saved CSVs; the console output and the per-stay table are not suppressed and must stay in the DRE.
"""
import argparse
import json
import string
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
MIN_CELL = 5
FEATURE_PREFIXES = ("lab_", "proc_", "med_", "renal_", "static_", "time_", "vit_", "sup_")


# ── loading ───────────────────────────────────────────────────────────────
def load(run, site, tag, op="orig_spec_at_sens0.75"):
    lab = pd.read_parquet(ROOT / "data" / site / "labelled" / "stays_labelled.parquet")
    lab["month"] = pd.to_datetime(lab["admit"]).dt.to_period("M").astype(str)
    seeds = []
    for s in sorted(Path(run).glob("seed_*")):
        p, t = s / tag / "predictions.parquet", s / "thresholds.json"
        if p.exists() and t.exists():
            df = pd.read_parquet(p)
            df["month"] = df["stay_id"].map(lab.set_index("stay_id")["month"])
            seeds.append((s.name, df, float(json.loads(t.read_text())[op])))
    if not seeds:
        raise SystemExit(f"no <seed>/{tag}/predictions.parquet + thresholds.json under {run}")
    return lab, seeds


def stay_table(df, thr):
    """Per stay: detected within 48 h / before onset, highest score, number of alerts."""
    d = df[["stay_id", "aki", "hours_to_onset", "prob"]].copy()
    h = d["hours_to_onset"].where(d["hours_to_onset"].notna(), np.inf)
    d["alert"] = d["prob"] >= thr
    d["w48"] = d["alert"] & (h <= 48)
    d["wpre"] = d["alert"] & np.isfinite(h)
    d["p48"] = d["prob"].where(h <= 48)
    g = d.groupby("stay_id")
    return pd.DataFrame({"aki": g["aki"].first().astype(bool), "n_windows": g.size(),
                         "max_prob": g["prob"].max(), "max_prob_48h": g["p48"].max(),
                         "det48": g["w48"].any(), "det_pre": g["wpre"].any(), "n_alerts": g["alert"].sum()})


def window_metrics(df, thr, step_h):
    y, p = df["label"].to_numpy(), df["prob"].to_numpy()
    both = len(np.unique(y)) == 2
    st = stay_table(df, thr)
    alert = p >= thr
    h = df["hours_to_onset"].where(df["hours_to_onset"].notna(), np.inf).to_numpy()
    non = ~df["aki"].to_numpy(bool)
    return dict(
        stays=int(st.shape[0]), aki_stays=int(st["aki"].sum()), windows=int(len(df)),
        auroc=roc_auc_score(y, p) if both else np.nan, auprc=average_precision_score(y, p) if both else np.nan,
        sens48=st.loc[st["aki"], "det48"].mean() if st["aki"].any() else np.nan,
        precision48=(alert & (h <= 48)).sum() / alert.sum() if alert.any() else np.nan,
        alerts_per_100pd=100 * alert.sum() / (len(df) * step_h / 24),
        nonaki_mean_prob=p[non].mean() if non.any() else np.nan,
        nonaki_p95_prob=np.quantile(p[non], .95) if non.any() else np.nan,
        nonaki_window_alert_rate=alert[non].mean() if non.any() else np.nan,
        observed_prevalence=y.mean(), mean_predicted=p.mean(),
        stay_auroc=(roc_auc_score(st["aki"], st["max_prob_48h"].fillna(st["max_prob"]))
                    if st["aki"].nunique() == 2 else np.nan))


# ── 1  chance ─────────────────────────────────────────────────────────────
class Evaluator:
    """Fast repeated evaluation of arbitrary sets of stays (duplicates allowed) across all seeds."""

    def __init__(self, seeds):
        self.s = []
        for _, df, thr in seeds:
            st = stay_table(df, thr)
            self.s.append(dict(idx=df.groupby("stay_id").indices, y=df["label"].to_numpy(),
                               p=df["prob"].to_numpy(), det=st["det48"].to_dict()))
        self.aki = seeds[0][1].groupby("stay_id")["aki"].first().astype(bool).to_dict()

    def __call__(self, stay_ids):
        au, se = [], []
        for s in self.s:
            rows = np.concatenate([s["idx"][i] for i in stay_ids])
            y, p = s["y"][rows], s["p"][rows]
            au.append(roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan)
            a = [s["det"][i] for i in stay_ids if self.aki[i]]
            se.append(np.mean(a) if a else np.nan)
        return np.nanmean(au) if not np.all(np.isnan(au)) else np.nan, \
            np.nanmean(se) if not np.all(np.isnan(se)) else np.nan


def chance_analysis(seeds, lab, month, n_draws, rng):
    ev = Evaluator(seeds)
    sid = seeds[0][1].groupby("stay_id").agg(subject=("subject_id", "first"), month=("month", "first"))
    sid["aki"] = sid.index.map(ev.aki)
    in_m = sid["month"] == month
    feb, rest = sid[in_m], sid[~in_m]
    n_a, n_n = int(feb["aki"].sum()), int((~feb["aki"]).sum())
    obs_au, obs_se = ev(list(feb.index))
    # (a) patient-level bootstrap of the month
    subj = feb.reset_index().groupby("subject")["stay_id"].apply(list).to_dict()      # works on any pandas
    keys = list(subj)
    boot = []
    for _ in range(n_draws):
        ids = [i for k in rng.choice(keys, len(keys)) for i in subj[k]]
        boot.append(ev(ids))
    boot = np.array(boot, float)
    # (b) random draws of the same size from the other months
    pa, pn = rest.index[rest["aki"]].to_numpy(), rest.index[~rest["aki"]].to_numpy()
    draws = []
    for _ in range(n_draws):
        ids = list(rng.choice(pa, n_a, replace=False)) + list(rng.choice(pn, n_n, replace=len(pn) < n_n))
        draws.append(ev(ids))
    draws = np.array(draws, float)
    res = {"month_stays": len(feb), "month_aki_stays": n_a, "month_non_aki_stays": n_n,
           "other_months_aki_stays": len(pa), "other_months_non_aki_stays": len(pn),
           "month_auroc": obs_au, "month_sens48": obs_se,
           "bootstrap_auroc_ci": f"{np.nanpercentile(boot[:, 0], 2.5):.3f}-{np.nanpercentile(boot[:, 0], 97.5):.3f}",
           "bootstrap_sens48_ci": f"{np.nanpercentile(boot[:, 1], 2.5):.3f}-{np.nanpercentile(boot[:, 1], 97.5):.3f}",
           "random_draws_auroc_median": np.nanmedian(draws[:, 0]),
           "random_draws_auroc_5th_95th": f"{np.nanpercentile(draws[:, 0], 5):.3f}-{np.nanpercentile(draws[:, 0], 95):.3f}",
           "share_of_draws_auroc_le_month": np.nanmean(draws[:, 0] <= obs_au + 1e-9),
           "random_draws_sens48_median": np.nanmedian(draws[:, 1]),
           "share_of_draws_sens48_le_month": np.nanmean(draws[:, 1] <= obs_se + 1e-9)}
    return res, draws, ev


# ── 2  which AKI stays were missed ────────────────────────────────────────
def cnt(k, n):
    if n < MIN_CELL:
        return "---"
    return "<5" if 0 < k < MIN_CELL else f"{k} ({100 * k / n:.0f}%)"


def aki_stay_frame(seeds, lab):
    per = [stay_table(df, thr) for _, df, thr in seeds]
    t = pd.concat([x[["aki", "max_prob_48h", "det48", "det_pre"]].add_suffix(f"_{i}") for i, x in enumerate(per)], axis=1)
    out = pd.DataFrame({"aki": per[0]["aki"],
                        "max_prob_48h": t.filter(like="max_prob_48h_").mean(axis=1),
                        "det48_share_seeds": t.filter(like="det48_").mean(axis=1),
                        "det_pre_share_seeds": t.filter(like="det_pre_").mean(axis=1)})
    out["thr"] = np.mean([thr for *_, thr in seeds])
    out = out[out["aki"]].join(lab.set_index("stay_id")[
        [c for c in ["month", "age_days", "onset_h", "trigger", "ref_source", "stage_peak", "n_creat_pre_onset",
                     "n_creat_stay", "onset_first_in_stay"] if c in lab.columns]])
    out["age_years"] = out["age_days"] / 365.25
    out["detected48"] = out["det48_share_seeds"] >= .5
    return out


def characteristics(a):
    c = pd.DataFrame(index=a.index)
    c["Age < 28 days"] = a["age_days"] < 28
    c["Age 28 days to 1 year"] = a["age_days"].between(28, 365, inclusive="left")
    c["Age 1 year or older"] = a["age_days"] >= 365
    if "onset_h" in a:
        c["Onset in the first 48 h of the stay"] = a["onset_h"] < 48
        c["Onset after day 5"] = a["onset_h"] >= 120
    if "trigger" in a:
        c["Triggered by 48 h absolute rise"] = a["trigger"].astype(str).str.startswith("abs")
    if "ref_source" in a:
        c["Reference not the rolling 7-day value"] = a["ref_source"].astype(str) != "rolling_7d"
    if "stage_peak" in a:
        c["Severe (stage 2-3)"] = a["stage_peak"] >= 2
    if "n_creat_pre_onset" in a:
        c["2 or fewer creatinine values before onset"] = a["n_creat_pre_onset"] <= 2
    if "onset_first_in_stay" in a:
        c["Onset at the first creatinine of the stay"] = a["onset_first_in_stay"].astype(bool)
    return c


def compare_aki_stays(a, month):
    c = characteristics(a)
    m = a["month"] == month
    rows = []
    for col in c.columns:
        rows.append({"characteristic": col, f"{month} AKI stays": cnt(int(c.loc[m, col].sum()), int(m.sum())),
                     "other months' AKI stays": cnt(int(c.loc[~m, col].sum()), int((~m).sum()))})
    rows.append({"characteristic": "Detected within 48 h of onset (majority of seeds)",
                 f"{month} AKI stays": cnt(int(a.loc[m, "detected48"].sum()), int(m.sum())),
                 "other months' AKI stays": cnt(int(a.loc[~m, "detected48"].sum()), int((~m).sum()))})
    return pd.DataFrame(rows)


def detection_by_characteristic(a):
    c = characteristics(a)
    rows = []
    for col in c.columns:
        for flag, name in ((True, "yes"), (False, "no")):
            sel = c[col] == flag
            n = int(sel.sum())
            rows.append({"characteristic": col, "group": name, "AKI stays": n if n >= MIN_CELL else f"<{MIN_CELL}",
                         "detected within 48 h": cnt(int(a.loc[sel, "detected48"].sum()), n)})
    return pd.DataFrame(rows)


# ── 3  non-AKI side, case mix ─────────────────────────────────────────────
def monthly_overview(seeds, lab, step_h):
    rows = []
    months = sorted(set().union(*[set(df["month"].dropna()) for _, df, _ in seeds]))
    for mth in months:
        ms = [window_metrics(df[df["month"] == mth], thr, step_h) for _, df, thr in seeds
              if (df["month"] == mth).any()]
        if not ms:
            continue
        r = pd.DataFrame(ms).mean(numeric_only=True)
        r["month"] = mth
        s = lab[(lab["month"] == mth) & lab["stay_id"].isin(seeds[0][1]["stay_id"])]
        r["median_age_years"] = (s["age_days"] / 365.25).median()
        r["share_under_28_days"] = (s["age_days"] < 28).mean()
        if "los_h" in s:
            r["median_los_days"] = (s["los_h"] / 24).median()
        rows.append(r)
    return pd.DataFrame(rows).set_index("month")


# ── 4  data drift ─────────────────────────────────────────────────────────
def drift(site, lab, seeds, month):
    wp = ROOT / "data" / site / "windows" / "windows.parquet"
    sp = ROOT / "data" / site / "labelled" / "split.csv"
    if not (wp.exists() and sp.exists()):
        print("drift checks skipped: windows.parquet or split.csv not found")
        return None, None
    w = pd.read_parquet(wp)
    feats = [c for c in w.columns if c.startswith(FEATURE_PREFIXES) and pd.api.types.is_numeric_dtype(w[c])]
    split = pd.read_csv(sp, dtype=str).set_index("stay_id")["split"]
    w["split"] = w["stay_id"].map(split)
    w["month"] = w["stay_id"].map(lab.set_index("stay_id")["month"])
    hold = w[w["split"] == "holdout"]
    dev = w[w["split"].isin(["train", "val"])]
    is_m = hold["month"] == month
    cm, co = hold.loc[is_m, feats], hold.loc[~is_m, feats]
    sd_o = np.sqrt((cm.var() + co.var()) / 2).replace(0, np.nan)
    sd_d = np.sqrt((cm.var() + dev[feats].var()) / 2).replace(0, np.nan)
    shift = pd.DataFrame({"smd_vs_other_holdout_months": (cm.mean() - co.mean()) / sd_o,
                          "smd_vs_development": (cm.mean() - dev[feats].mean()) / sd_d})
    shift["abs_max"] = shift.abs().max(axis=1)
    shift = shift.sort_values("abs_max", ascending=False).drop(columns="abs_max")
    samp = []
    for mth, g in hold.groupby("month"):
        r = {"month": mth, "windows": len(g)}
        if "proc_creatinine_n" in g:
            r["creatinine_measurements_per_window"] = g["proc_creatinine_n"].mean()
            r["share_windows_with_creatinine"] = (g["proc_creatinine_n"] > 0).mean()
        if "proc_creatinine_hsince" in g:
            r["median_hours_since_creatinine"] = g["proc_creatinine_hsince"].median()
        samp.append(r)
    if "proc_creatinine_n" in dev:
        samp.append({"month": "development", "windows": len(dev),
                     "creatinine_measurements_per_window": dev["proc_creatinine_n"].mean(),
                     "share_windows_with_creatinine": (dev["proc_creatinine_n"] > 0).mean(),
                     **({"median_hours_since_creatinine": dev["proc_creatinine_hsince"].median()}
                        if "proc_creatinine_hsince" in dev else {})})
    return shift, pd.DataFrame(samp).set_index("month")


# ── figure ────────────────────────────────────────────────────────────────
def figure(res, draws, overview, a, month, shift, samp, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e6e6e6", "axes.axisbelow": True})
    B, R = "#3b4cc0", "#d1495b"
    fig, axs = plt.subplots(3, 2, figsize=(7.4, 8.4))
    ax = axs[0, 0]
    ax.hist(draws[:, 0][~np.isnan(draws[:, 0])], bins=30, color="#c9c9c9")
    ax.axvline(res["month_auroc"], color=R, lw=2)
    ax.set_xlabel(f"AUROC in random draws of {res['month_aki_stays']} AKI stays")
    ax.set_ylabel("Draws")
    ax.set_title(f"{month}: {res['share_of_draws_auroc_le_month']:.0%} of draws are as low", fontsize=8, loc="left")
    ax = axs[0, 1]
    v = draws[:, 1][~np.isnan(draws[:, 1])]
    ax.hist(v, bins=np.linspace(0, 1, 21), color="#c9c9c9")
    ax.axvline(res["month_sens48"], color=R, lw=2)
    ax.set_xlabel("Sensitivity within 48 h in the same draws")
    ax.set_title(f"{res['share_of_draws_sens48_le_month']:.0%} of draws are as low", fontsize=8, loc="left")
    ax = axs[1, 0]
    x = pd.PeriodIndex(overview.index, freq="M").to_timestamp()
    ax.plot(x, 100 * overview["nonaki_window_alert_rate"], marker="o", color=B)
    ax.axvline(pd.Period(month, "M").to_timestamp(), color=R, lw=6, alpha=.15)
    ax.set_ylabel("Non-AKI windows with an alert (%)")
    ax.tick_params(axis="x", rotation=30)
    ax = axs[1, 1]
    other = a[a["month"] != month]
    rng = np.random.default_rng(0)
    ax.scatter(1 + rng.normal(0, .06, len(other)), other["max_prob_48h"], s=10, color="#9a9a9a", alpha=.7)
    this = a[a["month"] == month]
    ax.scatter(2 + rng.normal(0, .04, len(this)), this["max_prob_48h"], s=22, color=R)
    ax.axhline(a["thr"].iloc[0], color="#555", ls="--", lw=1)
    ax.set_xticks([1, 2], ["Other months", month])
    ax.set_ylabel("Highest score in the 48 h before onset")
    ax.set_title("AKI stays (dashed line: alert threshold)", fontsize=8, loc="left")
    ax = axs[2, 0]
    if samp is not None and "creatinine_measurements_per_window" in samp:
        s = samp.drop(index="development", errors="ignore")
        ax.plot(pd.PeriodIndex(s.index, freq="M").to_timestamp(), s["creatinine_measurements_per_window"], marker="o", color=B)
        if "development" in samp.index:
            ax.axhline(samp.loc["development", "creatinine_measurements_per_window"], color="#888", ls=":", label="development")
            ax.legend(frameon=False)
        ax.axvline(pd.Period(month, "M").to_timestamp(), color=R, lw=6, alpha=.15)
        ax.set_ylabel("Creatinine measurements per window")
        ax.tick_params(axis="x", rotation=30)
    else:
        ax.axis("off")
    ax = axs[2, 1]
    if shift is not None:
        top = shift.head(10).iloc[::-1]
        ax.barh(range(len(top)), top["smd_vs_other_holdout_months"], color=B, height=.6)
        ax.set_yticks(range(len(top)), top.index, fontsize=6)
        ax.axvline(0, color="#333", lw=.7)
        ax.set_xlabel(f"Standardised shift of {month} vs other hold-out months")
    else:
        ax.axis("off")
    for ax_, ch in zip(axs.ravel(), string.ascii_lowercase):
        if ax_.axison:
            ax_.text(-0.02, 1.15, ch, transform=ax_.transAxes, fontsize=11, fontweight="bold", va="bottom")
    fig.tight_layout()
    fig.savefig(outdir / "month_diagnostics.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="folder containing seed_0 ... seed_4")
    ap.add_argument("--site", default="gosh")
    ap.add_argument("--tag", default="deploy_gosh_holdout")
    ap.add_argument("--month", default="2026-02")
    ap.add_argument("--n_draws", type=int, default=2000)
    ap.add_argument("--step_h", type=float, default=6.0)
    ap.add_argument("--no_windows", action="store_true", help="skip the drift checks")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--op", default="orig_spec_at_sens0.75", help="operating point (key in thresholds.json)")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = ROOT / "results" / f"month_diagnostics_{a.month}"
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)

    lab, seeds = load(a.run, a.site, a.tag, a.op)
    print(f"{len(seeds)} seeds | thresholds {[round(t, 4) for *_, t in seeds]}\n")

    print("=" * 78 + f"\n1  Is {a.month} unusual, given how few AKI stays it has?\n" + "=" * 78)
    res, draws, ev = chance_analysis(seeds, lab, a.month, a.n_draws, rng)
    pd.Series(res).to_csv(out / "1_chance.csv", header=["value"])
    print(pd.Series(res).to_string(float_format=lambda x: f"{x:.3f}"))

    print("\n" + "=" * 78 + "\n2  Which AKI stays were missed?\n" + "=" * 78)
    aki = aki_stay_frame(seeds, lab)
    cmp_ = compare_aki_stays(aki, a.month)
    cmp_.to_csv(out / "2_month_vs_other_aki_stays.csv", index=False)
    det = detection_by_characteristic(aki)
    det.to_csv(out / "2_detection_by_characteristic.csv", index=False)
    print(cmp_.to_string(index=False))
    print("\nDetection within 48 h by characteristic, whole hold-out:\n" + det.to_string(index=False))
    cases = aki[aki["month"] == a.month].sort_values("max_prob_48h")
    show = [c for c in ["age_years", "onset_h", "trigger", "ref_source", "stage_peak", "n_creat_pre_onset",
                        "max_prob_48h", "det48_share_seeds"] if c in cases.columns]
    print(f"\nPer-stay table for {a.month} AKI stays (console only: row-level, keep inside the DRE;"
          f" alert threshold = {aki['thr'].iloc[0]:.3f}):")
    print(cases[show].round(3).to_string())

    print("\n" + "=" * 78 + "\n3  Non-AKI side, calibration and case mix by month\n" + "=" * 78)
    ov = monthly_overview(seeds, lab, a.step_h)
    ov.to_csv(out / "3_monthly_overview.csv")
    print(ov.round(3).T.to_string())

    shift = samp = None
    if not a.no_windows:
        print("\n" + "=" * 78 + "\n4  Did the data change?\n" + "=" * 78)
        shift, samp = drift(a.site, lab, seeds, a.month)
        if shift is not None:
            shift.head(40).to_csv(out / "4_feature_shift.csv")
            samp.to_csv(out / "4_creatinine_sampling_by_month.csv")
            print("Creatinine sampling:\n" + samp.round(3).to_string())
            print(f"\nLargest standardised shifts of {a.month}:\n" + shift.head(15).round(2).to_string())
    figure(res, draws, ov, aki, a.month, shift, samp, out)
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()