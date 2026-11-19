"""Green / Amber / Red risk stratification for ONE chosen model.

    python tools/traffic_light.py --run runs/gosh_transformer_focal_all   # picks best seed on validation
    python tools/traffic_light.py --run runs/gosh_transformer_focal_all --apply test holdout
    python tools/traffic_light.py --run runs/gosh_transformer_focal_all --resolution 0.05  # absolute 5% grid

Method (paper, Methods "Risk stratification framework")
  * Each stay gets one score: its MAXIMUM predicted risk over the stay
    (windows stop at onset, so for AKI stays this is the pre-onset maximum).
  * Two thresholds tau_amber < tau_red: Green < tau_amber <= Amber < tau_red <= Red.
  * All pairs on a grid are tried ON VALIDATION; the pair maximising Youden's J
    computed on Green vs Red only (Amber excluded) is chosen, subject to
    10-20% of stays Red and >= 50% Green.
  * The pair is applied unchanged to the test set (and the hold-out, if asked).

Grid and scale
  * Probabilities are Platt-recalibrated at true prevalence, so most scores
    are small. The default grid is every 2nd centile of the validation stay
    scores; `--resolution 0.05` uses an absolute 5% grid instead.
  * The cohort is unmatched (true prevalence).

Outputs (aggregate only) in <seed dir>/traffic_light_<site>/:
  thresholds.json, val_grid.csv (J for every pair), distribution_<split>.csv,
  performance_<split>.csv, amber_transitions_<split>.csv, figures *.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aki.cli import _predict, _subset, load_site_windows  # noqa: E402
from aki.train import Platt  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

STATES = ["Green", "Amber", "Red"]


def pick_seed(run: Path, seed: int | None) -> Path:
    if seed is not None:
        return run / f"seed_{seed}" if (run / f"seed_{seed}").exists() else run
    if (run / "run_config.json").exists():
        return run
    cands = []
    for sd in sorted(run.glob("seed_*")):
        meta = sd / "model_meta.json"
        if meta.exists():
            cands.append((json.loads(meta.read_text())["best_val_auprc"], sd))
        elif (sd / "val_threshold_sweep.csv").exists():
            cands.append((np.nan, sd))
    if not cands:
        raise SystemExit(f"no trained seeds under {run}")
    scored = [c for c in cands if not np.isnan(c[0])]
    best = max(scored)[1] if scored else cands[0][1]
    print("validation AUPRC by seed: " + ", ".join(f"{sd.name}={v:.3f}" for v, sd in cands))
    print(f"-> using {best.name} (highest validation AUPRC; test results not used to choose)")
    return best


def predictions(sd: Path, rc: dict, win: pd.DataFrame, split: str) -> pd.DataFrame:
    df = _subset(win, split)
    lg = _predict(sd, rc["model"], df, rc["common"])
    return df.assign(prob=Platt.load(sd / "platt.json")(lg))


def stay_scores(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("stay_id").agg(score=("prob", "max"), aki=("aki", "first"),
                                     severe=("stage_peak", lambda s: s.iloc[0] >= 2))


def classify(score, ta, tr):
    return np.where(score >= tr, "Red", np.where(score >= ta, "Amber", "Green"))


def grid_search(st: pd.DataFrame, resolution, red_frac, min_green):
    if resolution == "auto":
        cand = np.unique(np.quantile(st["score"], np.linspace(0.02, 0.98, 49)))
    else:
        cand = np.round(np.arange(resolution, 1.0, float(resolution)), 6)
    s, y = st["score"].to_numpy(), st["aki"].to_numpy(bool)
    rows = []
    for i, ta in enumerate(cand):
        for tr in cand[i + 1:]:
            red, green = s >= tr, s < ta
            tp, fn = (red & y).sum(), (green & y).sum()
            tn, fp = (green & ~y).sum(), (red & ~y).sum()
            sens = tp / (tp + fn) if tp + fn else np.nan
            spec = tn / (tn + fp) if tn + fp else np.nan
            rows.append(dict(tau_amber=ta, tau_red=tr, red_frac=red.mean(),
                             green_frac=green.mean(), amber_frac=1 - red.mean() - green.mean(),
                             sens_red_vs_green=sens, spec_red_vs_green=spec,
                             youden_j=sens + spec - 1))
    g = pd.DataFrame(rows)
    g["feasible"] = g["red_frac"].between(*red_frac) & (g["green_frac"] >= min_green)
    return g


def distribution(st: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, m in [("non-AKI", ~st["aki"]), ("AKI (all)", st["aki"]),
                    ("AKI stage 1", st["aki"] & ~st["severe"]),
                    ("AKI stage 2-3", st["aki"] & st["severe"]), ("all stays", st["aki"] | True)]:
        d = st[m]
        row = dict(group=name, n_stays=len(d))
        for s in STATES:
            row[f"{s}_n"] = int((d["state"] == s).sum())
            row[f"{s}_%"] = round(100 * (d["state"] == s).mean(), 1) if len(d) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def performance(st: pd.DataFrame, df: pd.DataFrame, ta, tr) -> pd.DataFrame:
    y = st["aki"]
    out = {}
    for s in STATES:
        d = st[st["state"] == s]
        out[f"AKI rate in {s} (%)"] = round(100 * d["aki"].mean(), 1) if len(d) else np.nan
    red, green = st["state"] == "Red", st["state"] == "Green"
    out["Red vs Green sensitivity"] = (red & y).sum() / max(((red | green) & y).sum(), 1)
    out["Red vs Green specificity"] = (green & ~y).sum() / max(((red | green) & ~y).sum(), 1)
    out["Red vs Green Youden J"] = out["Red vs Green sensitivity"] + \
        out["Red vs Green specificity"] - 1
    # timing of the FIRST red window in AKI stays (how early 'Red' appears)
    a = df[df["aki"] & (df["prob"] >= tr)].sort_values("k").groupby("stay_id")["hours_to_onset"].first()
    out["AKI stays ever Red: median h before onset of first Red"] = \
        float(a.median()) if len(a) else np.nan
    out["AKI stays ever Red: first Red within 48 h of onset (%)"] = \
        round(100 * (a <= 48).mean(), 1) if len(a) else np.nan
    out["window AUPRC (for reference)"] = average_precision_score(df["label"], df["prob"]) \
        if df["label"].nunique() == 2 else np.nan
    return pd.Series(out).round(3).to_frame("value")


def amber_transitions(df: pd.DataFrame, ta, tr) -> pd.DataFrame:
    """Window-level course of stays that are Amber at some point."""
    d = df.sort_values(["stay_id", "k"]).assign(state=lambda x: classify(x["prob"], ta, tr))
    rows = []
    for sid, g in d.groupby("stay_id"):
        st = g["state"].to_numpy()
        if "Amber" not in st:
            continue
        first_amber = np.argmax(st == "Amber")
        before_red = "Red" in st[:first_amber]
        after = st[first_amber:]
        if "Red" in after:
            fate = "escalated to Red"
        elif after[-1] == "Green":
            fate = "de-escalated to Green"
        else:
            fate = "remained Amber"
        rows.append(dict(aki=bool(g["aki"].iloc[0]), fate=fate, red_before_amber=before_red))
    if not rows:
        return pd.DataFrame()
    t = pd.DataFrame(rows)
    out = (t.groupby(["aki", "fate"]).size().rename("n").reset_index())
    out["% of ever-Amber stays"] = (100 * out["n"] / out.groupby("aki")["n"].transform("sum")).round(1)
    out["aki"] = out["aki"].map({True: "AKI", False: "non-AKI"})
    return out


def figures(grid, choice, test_st, ta, tr, out: Path, split: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    # (1) J over threshold pairs on validation
    piv = grid[grid["feasible"]].pivot_table(index="tau_amber", columns="tau_red",
                                             values="youden_j")
    if len(piv):
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(piv.to_numpy(), origin="lower", aspect="auto", cmap="RdYlGn")
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels([f"{c:.3f}" for c in piv.columns], rotation=90, fontsize=6)
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels([f"{c:.3f}" for c in piv.index], fontsize=6)
        r = list(piv.columns).index(choice["tau_red"]); c = list(piv.index).index(choice["tau_amber"])
        ax.plot(r, c, "k*", ms=14)
        ax.set_xlabel("red threshold"); ax.set_ylabel("amber threshold")
        ax.set_title("Validation: Youden J (Green vs Red), feasible pairs")
        fig.colorbar(im)
        fig.savefig(out / "val_youden_grid.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    # (2) score histograms with thresholds
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, test_st["score"].quantile(0.995), 40)
    ax.hist(test_st.loc[~test_st["aki"], "score"], bins, alpha=.6, density=True, label="non-AKI")
    ax.hist(test_st.loc[test_st["aki"], "score"], bins, alpha=.6, density=True, label="AKI")
    ax.axvline(ta, c="orange", ls="--", label=f"amber {ta:.3f}")
    ax.axvline(tr, c="red", ls="--", label=f"red {tr:.3f}")
    ax.set_xlabel("maximum predicted risk per stay"); ax.set_ylabel("density"); ax.legend()
    fig.savefig(out / f"score_distribution_{split}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    # (3) stacked bars
    fig, ax = plt.subplots(figsize=(5, 4))
    groups = [("non-AKI", ~test_st["aki"]), ("AKI", test_st["aki"])]
    bottom = np.zeros(2)
    for s, col in zip(STATES, ["#2ecc71", "#f39c12", "#e74c3c"]):
        v = np.array([100 * (test_st.loc[m, "state"] == s).mean() for _, m in groups])
        ax.bar([g for g, _ in groups], v, bottom=bottom, color=col, label=s)
        for i, (b, x) in enumerate(zip(bottom, v)):
            if x > 3:
                ax.text(i, b + x / 2, f"{x:.1f}%", ha="center", va="center", fontsize=9)
        bottom += v
    ax.set_ylabel("% of stays"); ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    fig.savefig(out / f"traffic_light_{split}.png", dpi=130, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="runs/<name> (best seed chosen) or a seed dir")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--site", help="data site (default: the run's training site)")
    ap.add_argument("--apply", nargs="+", default=["test"], help="splits to apply to")
    ap.add_argument("--resolution", default="auto", help="'auto' or a step such as 0.05")
    ap.add_argument("--red_frac", nargs=2, type=float, default=[0.10, 0.20])
    ap.add_argument("--min_green", type=float, default=0.50)
    a = ap.parse_args()
    res = a.resolution if a.resolution == "auto" else float(a.resolution)

    sd = pick_seed(Path(a.run), a.seed)
    rc = json.loads((sd / "run_config.json").read_text())
    site = a.site or rc["site"]
    _, win, _ = load_site_windows(site)
    out = sd / f"traffic_light_{site}"
    out.mkdir(exist_ok=True)

    val = predictions(sd, rc, win, "val")
    vst = stay_scores(val)
    grid = grid_search(vst, res, a.red_frac, a.min_green)
    grid.round(4).to_csv(out / "val_grid.csv", index=False)
    feas = grid[grid["feasible"]]
    if feas.empty:
        raise SystemExit("no threshold pair meets the Red/Green constraints on validation; "
                         "try --resolution auto or relax --red_frac / --min_green")
    choice = feas.sort_values(["youden_j", "green_frac"], ascending=False).iloc[0]
    ta, tr = float(choice["tau_amber"]), float(choice["tau_red"])
    json.dump(dict(seed_dir=str(sd), tau_amber=ta, tau_red=tr, resolution=a.resolution,
                   red_frac_constraint=a.red_frac, min_green=a.min_green,
                   val_youden_j=float(choice["youden_j"]),
                   val_red_frac=float(choice["red_frac"]),
                   val_green_frac=float(choice["green_frac"])),
              open(out / "thresholds.json", "w"), indent=1)
    print(f"\nchosen on validation: amber >= {ta:.4f}, red >= {tr:.4f} "
          f"(J={choice['youden_j']:.3f}, Red {100 * choice['red_frac']:.1f}%, "
          f"Green {100 * choice['green_frac']:.1f}%)")

    for split in a.apply:
        df = predictions(sd, rc, win, split)
        if df.empty:
            print(f"[{split}] no data — skipped")
            continue
        st = stay_scores(df)
        st["state"] = classify(st["score"], ta, tr)
        dist, perf = distribution(st), performance(st, df, ta, tr)
        trans = amber_transitions(df, ta, tr)
        dist.to_csv(out / f"distribution_{split}.csv", index=False)
        perf.to_csv(out / f"performance_{split}.csv")
        trans.to_csv(out / f"amber_transitions_{split}.csv", index=False)
        figures(grid, choice, st, ta, tr, out, split)
        pd.set_option("display.width", 200)
        print(f"\n=== {split} ===")
        print(dist[["group", "n_stays", "Green_%", "Amber_%", "Red_%"]].to_string(index=False))
        print(perf.to_string())
        if len(trans):
            print(trans.to_string(index=False))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
