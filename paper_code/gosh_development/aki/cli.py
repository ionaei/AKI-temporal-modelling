"""Pipeline entry point.

    python -m aki.cli extract  --site gosh
    python -m aki.cli label    --site gosh
    python -m aki.cli windows  --site gosh
    python -m aki.cli split    --site gosh
    python -m aki.cli core     --sites gosh pic mimic3
    python -m aki.cli train    --site gosh --model transformer --loss focal \
                               --name gosh_transformer_focal_all --seeds 0 1 2 3 4
    python -m aki.cli evaluate --run runs/gosh_transformer_focal_all --site gosh --split test
    python -m aki.cli evaluate --run runs/gosh_transformer_focal_all --site pic --split all  # zero-shot
    python -m aki.cli explain  --run runs/gosh_transformer_focal_all/seed_0 --site gosh --split test
    python -m aki.cli deploy   --run runs/gosh_transformer_focal_all/seed_0 --site gosh   # temporal hold-out
    python -m aki.cli aggregate --run runs/gosh_transformer_focal_all

Every stage reads the previous stage's parquet from data/<site>/ and writes
its own, so stages can be re-run independently.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import metrics as M
from .adapters import get_adapter
from .cohort import build_stays
from .config import ROOT, load_config
from .features import ALL_GROUPS, Preprocessor, core_feature_set, select_features
from .labelling import label_cohort
from .split import make_split
from .windows import availability, build_windows

log = logging.getLogger("aki")


def _git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


# ── data stages ───────────────────────────────────────────────────────────
def _done(a, path: Path, what: str) -> bool:
    """With --resume, skip a stage whose final output already exists."""
    if getattr(a, "resume", False) and Path(path).exists():
        log.info("--resume: %s already done (%s exists) — skipping", what, path)
        return True
    return False


def cmd_extract(a):
    cfg = load_config(a.site)
    if _done(a, Path(cfg["site"]["dirs"]["canonical"]) / "unit_report.csv", "extract"):
        return
    get_adapter(cfg["site"]).run()


def cmd_label(a):
    cfg = load_config(a.site)
    d = cfg["site"]["dirs"]
    if _done(a, Path(d["labelled"]) / "label_summary.json", "label"):
        return
    ep = pd.read_parquet(Path(d["canonical"]) / "episodes.parquet")
    labs = pd.read_parquet(Path(d["canonical"]) / "labs.parquet")
    rrt = pd.read_parquet(Path(d["canonical"]) / "rrt.parquet")
    stays = build_stays(ep, cfg["common"]["cohort"])
    lab = label_cohort(stays, labs, rrt, cfg["common"])
    lab.to_parquet(Path(d["labelled"]) / "stays_labelled.parquet", index=False)
    # aggregate-only summary safe to export
    inc = lab[~lab["exclude"]]
    summ = {
        "stays_total": len(lab), "stays_included": len(inc),
        "excluded_by_reason": lab.loc[lab["exclude"], "exclude_reason"].value_counts().to_dict(),
        "aki": int(inc["aki"].sum()), "aki_pct": float(100 * inc["aki"].mean()),
        "peak_stage": inc.loc[inc["aki"], "stage_peak"].value_counts().sort_index().to_dict(),
        "severe": int(inc["severe"].sum()),
        "trigger": inc.loc[inc["aki"], "trigger"].value_counts().to_dict(),
        "onset_first_in_stay_creatinine": int(inc["onset_first_in_stay"].sum()),
        "stays_without_creatinine": int((inc["n_creat_stay"] == 0).sum()),
        "creatinine_per_stay_median": float(inc["n_creat_stay"].median()),
        "ref_source": inc.loc[inc["aki"], "ref_source"].value_counts().to_dict(),
    }
    (Path(d["labelled"]) / "label_summary.json").write_text(json.dumps(summ, indent=1, default=int))
    log.info(json.dumps(summ, indent=1, default=int))


def cmd_windows(a):
    cfg = load_config(a.site)
    d = cfg["site"]["dirs"]
    if _done(a, Path(d["windows"]) / "availability.csv", "windows"):
        return
    lab = pd.read_parquet(Path(d["labelled"]) / "stays_labelled.parquet")
    stays = lab[~lab["exclude"]]
    tabs = {t: pd.read_parquet(Path(d["canonical"]) / f"{t}.parquet")
            for t in ("labs", "vitals", "meds")}
    for t in ("labs", "vitals"):
        tabs[t]["value"] = tabs[t]["value"].astype(float)
    win, groups = build_windows(stays, tabs["labs"], tabs["vitals"], tabs["meds"], cfg["common"])
    win.to_parquet(Path(d["windows"]) / "windows.parquet", index=False)
    (Path(d["windows"]) / "feature_groups.json").write_text(json.dumps(groups, indent=1))
    av = availability(win, groups)
    av.to_csv(Path(d["windows"]) / "availability.csv", index=False)
    const = av[av["n_unique"] <= 1]["feature"].tolist()
    log.info("%d windows written; %d constant features at this site: %s",
             len(win), len(const), const[:20])


def cmd_split(a):
    cfg = load_config(a.site)
    d = cfg["site"]["dirs"]
    if _done(a, Path(d["labelled"]) / "split_meta.json", "split"):
        return
    lab = pd.read_parquet(Path(d["labelled"]) / "stays_labelled.parquet")
    sp, meta = make_split(lab, cfg["common"])
    sp.to_csv(Path(d["labelled"]) / "split.csv", index=False)
    (Path(d["labelled"]) / "split_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    # feature availability (used for the cross-site core set) from DEVELOPMENT stays only,
    # so nothing about the hold-out period influences feature selection
    wpath = Path(d["windows"]) / "windows.parquet"
    if wpath.exists():
        win = pd.read_parquet(wpath)
        groups = json.loads((Path(d["windows"]) / "feature_groups.json").read_text())
        dev_ids = set(sp.loc[sp["split"] != "holdout", "stay_id"])
        availability(win[win["stay_id"].isin(dev_ids)], groups).to_csv(
            Path(d["windows"]) / "availability.csv", index=False)


def cmd_core(a):
    files = [Path(load_config(s)["site"]["dirs"]["windows"]) / "availability.csv"
             for s in a.sites]
    fc = load_config(a.sites[0])["common"]["features"]
    core = core_feature_set(files, fc["core_min_non_null"], fc["core_min_nonzero_binary"])
    out = ROOT / "configs" / f"core_features_{'_'.join(a.sites)}.json"
    out.write_text(json.dumps(core, indent=1))
    log.info("wrote %s (%d features)", out, len(core))


# ── helpers shared by train / evaluate ────────────────────────────────────
def load_site_windows(site: str):
    cfg = load_config(site)
    d = cfg["site"]["dirs"]
    win = pd.read_parquet(Path(d["windows"]) / "windows.parquet")
    groups = json.loads((Path(d["windows"]) / "feature_groups.json").read_text())
    sp_path = Path(d["labelled"]) / "split.csv"
    if sp_path.exists():
        sp = pd.read_csv(sp_path, dtype={"stay_id": str})
        win = win.merge(sp[["stay_id", "split"]], on="stay_id", how="inner")
    else:
        win["split"] = "all"
    win = win.sort_values(["stay_id", "k"]).reset_index(drop=True)
    return cfg, win, groups


def _subset(win, split):
    """'dev' = train+val+test; 'all' = everything incl. hold-out (only for other sites)."""
    if split == "all":
        return win
    if split == "dev":
        return win[win["split"] != "holdout"].reset_index(drop=True)
    return win[win["split"] == split].reset_index(drop=True)


def _predict(run_dir: Path, kind: str, df: pd.DataFrame, cfg: dict):
    pre = Preprocessor.load(run_dir / "preprocessor.joblib")
    X = pre.transform(df)
    if kind in ("gru", "lstm", "transformer"):
        from .train import Sequences, load_sequence_model, predict_logits
        model, device = load_sequence_model(run_dir)
        return predict_logits(model, Sequences(df, X), cfg["training"], device)
    from .train import tabular_logits
    return tabular_logits(joblib.load(run_dir / "model.joblib"), X)


# ── operating points ──────────────────────────────────────────────────────
def choose_operating_points(vdf: pd.DataFrame, C: dict, out: Path) -> dict:
    """Sweep thresholds on VALIDATION predictions and pick every operating point
    defined in common.yaml; writes val_threshold_sweep.csv and thresholds.json."""
    sweep = M.threshold_sweep(M.AlertArrays(vdf, C["windows"]["step_h"]), M.threshold_grid(vdf))
    sweep.to_csv(out / "val_threshold_sweep.csv", index=False)
    thresholds = {name: M.choose_threshold(sweep, rule)
                  for name, rule in C["evaluation"]["operating_points"].items()}
    (out / "thresholds.json").write_text(json.dumps(thresholds, indent=1))
    return thresholds


def cmd_thresholds(a):
    """Re-derive thresholds for already-trained runs (no retraining), using the
    operating points currently in common.yaml. Re-run `evaluate` (without
    --resume) afterwards so test results include the new operating points."""
    from .train import Platt
    run = Path(a.run)
    seeds = [run] if (run / "run_config.json").exists() else sorted(run.glob("seed_*"))
    for sd in seeds:
        rc = json.loads((sd / "run_config.json").read_text())
        C = load_config(a.site or rc["site"])["common"]
        cfg, win, _ = load_site_windows(rc["site"])
        va = _subset(win, "val")
        lg = _predict(sd, rc["model"], va, rc["common"])
        vdf = va.assign(logit=lg, prob=Platt.load(sd / "platt.json")(lg))
        rc["common"]["evaluation"] = C["evaluation"]          # record the new definitions
        th = choose_operating_points(vdf, rc["common"], sd)
        (sd / "run_config.json").write_text(json.dumps(rc, indent=1, default=str))
        log.info("%s/%s thresholds: %s", sd.parent.name, sd.name,
                 {k: round(v, 4) for k, v in th.items()})


# ── training ──────────────────────────────────────────────────────────────
def cmd_train(a):
    from .train import Platt, Sequences, predict_logits, train_sequence, train_tabular, tabular_logits
    cfg, win, groups = load_site_windows(a.site)
    C = cfg["common"]
    # per-run loss override (common.yaml stays the default); recorded in run_config.json
    if a.loss:
        C["training"]["loss"] = a.loss
    if a.pos_weight:
        C["training"]["pos_weight"] = a.pos_weight if a.pos_weight in ("auto", "sqrt") \
            else float(a.pos_weight)
    if a.model in ("gru", "lstm", "transformer"):
        tc = C["training"]
        log.info("loss: %s (%s)", tc["loss"],
                 f"pos_weight={tc['pos_weight']}" if tc["loss"] == "bce"
                 else f"alpha={tc['focal_alpha']}, gamma={tc['focal_gamma']}")
    include = a.groups or C["features"]["include_groups"]
    core = json.loads(Path(a.core).read_text()) if a.core else None
    cols = select_features(groups, include, core, C["features"].get("drop", []))
    if not cols:
        raise SystemExit("no features selected")
    tr, va = _subset(win, "train"), _subset(win, "val")
    base = ROOT / "runs" / a.name
    base.mkdir(parents=True, exist_ok=True)
    for seed in a.seeds:
        out = base / f"seed_{seed}"
        if _done(a, out / "run_config.json", f"train {a.name} seed {seed}"):
            continue
        out.mkdir(exist_ok=True)
        pre = Preprocessor(cols, C["features"]["age_normalise_prefixes"],
                           C["features"]["clip"]).fit(tr)
        pre.save(out / "preprocessor.joblib")
        X_tr, X_va = pre.transform(tr), pre.transform(va)
        if a.model in ("gru", "lstm", "transformer"):
            model, device = train_sequence(a.model, Sequences(tr, X_tr), Sequences(va, X_va),
                                           len(cols), C, out, seed,
                                           resume=getattr(a, "resume", False))
            lg_va = predict_logits(model, Sequences(va, X_va), C["training"], device)
        else:
            m = train_tabular(a.model, X_tr, tr["label"].to_numpy(), X_va,
                              va["label"].to_numpy(), seed)
            joblib.dump(m, out / "model.joblib")
            lg_va = tabular_logits(m, X_va)
        platt = Platt().fit(lg_va, va["label"].to_numpy())
        platt.save(out / "platt.json")
        thresholds = choose_operating_points(va.assign(logit=lg_va, prob=platt(lg_va)), C, out)
        run_cfg = dict(site=a.site, model=a.model, seed=seed, groups=include,
                       core=a.core, n_features=len(cols), git=_git_hash(), common=C)
        (out / "run_config.json").write_text(json.dumps(run_cfg, indent=1, default=str))
        log.info("seed %d done: thresholds %s", seed, thresholds)


# ── evaluation ────────────────────────────────────────────────────────────
def evaluate_run(run_dir: Path, site: str, split: str, n_boot: int,
                 recalibrate_on_target_val: bool = False, tag: str | None = None):
    from .train import Platt
    rc = json.loads((run_dir / "run_config.json").read_text())
    cfg, win, _ = load_site_windows(site)
    C = rc["common"]
    step = C["windows"]["step_h"]
    df = _subset(win, split)
    logit = _predict(run_dir, rc["model"], df, C)
    platt = Platt.load(run_dir / "platt.json")
    out = run_dir / (tag or f"eval_{site}_{split}" + ("_recal" if recalibrate_on_target_val else ""))
    out.mkdir(exist_ok=True)
    thresholds = json.loads((run_dir / "thresholds.json").read_text())
    if recalibrate_on_target_val:
        # local adaptation without retraining: refit calibration AND re-choose every
        # operating point on the TARGET site's validation stays (model weights frozen).
        # The training site's thresholds are on the training-site probability scale and no longer apply.
        va = _subset(win, "val")
        lg_va = _predict(run_dir, rc["model"], va, C)
        platt = Platt().fit(lg_va, va["label"].to_numpy())
        platt.save(out / "platt_target.json")
        thresholds = choose_operating_points(va.assign(logit=lg_va, prob=platt(lg_va)), C, out)
        log.info("recalibrated on %s val; thresholds re-chosen there: %s", site,
                 {k: round(v, 4) for k, v in thresholds.items()})
    df = df.assign(logit=logit, prob=platt(logit))
    df[["stay_id", "subject_id", "k", "hours_since_admit", "hours_to_onset", "label", "aki",
        "stage_peak", "age_days", "logit", "prob"]].to_parquet(out / "predictions.parquet",
                                                                index=False)
    res = {}
    for name, thr in thresholds.items():
        res[name] = M.summary(df, thr, step)
        b = M.bootstrap(df, thr, step, n_boot)
        if len(b):
            b.to_csv(out / f"bootstrap_{name}.csv")
    pd.DataFrame(res).to_csv(out / "summary.csv")
    M.horizon_discrimination(df).to_csv(out / "horizon_auroc.csv", index=False)
    M.reliability_table(df["label"].to_numpy(), df["prob"].to_numpy()).to_csv(
        out / "reliability.csv", index=False)
    M.decision_curve(df["label"].to_numpy(), df["prob"].to_numpy()).to_csv(
        out / "decision_curve.csv", index=False)
    arr = M.AlertArrays(df, step)
    grid = np.unique(np.quantile(df["prob"], np.linspace(0.5, 0.999, 100)))
    M.threshold_sweep(arr, grid).to_csv(out / "threshold_sweep.csv", index=False)
    # severity / age subgroups at the primary operating point
    prim = list(thresholds)[0]
    sub = []
    for name, mask in {"age_<28d": df["age_days"] < 28, "age_28d-1y": df["age_days"].between(28, 365),
                       "age_1-12y": df["age_days"].between(365, 4383),
                       "age_>=12y": df["age_days"] >= 4383}.items():
        d = df[mask]
        if d["label"].sum() >= 5:
            sub.append(dict(subgroup=name, **M.summary(d, thresholds[prim], step)))
    pd.DataFrame(sub).to_csv(out / "subgroups.csv", index=False)
    (out / "DONE").write_text("complete\n")
    log.info("%s -> %s\n%s", run_dir.name, out,
             pd.DataFrame(res).loc[["auroc", "auprc", "brier", "cal_slope", "precision@48h",
                                    "sens@48h", "lead@48h_median",
                                    "alerts_per_100_patient_days"]].round(3).to_string())
    return df, res


def cmd_evaluate(a):
    seeds = sorted(Path(a.run).glob("seed_*")) if not (Path(a.run) / "run_config.json").exists() \
        else [Path(a.run)]
    for sd in seeds:
        tag = f"eval_{a.site}_{a.split}" + ("_recal" if a.recalibrate_on_target_val else "")
        # DONE marker, or subgroups.csv (last file written by older versions)
        if _done(a, sd / tag / "DONE", f"evaluate {sd.parent.name}/{sd.name}") or \
                _done(a, sd / tag / "subgroups.csv", f"evaluate {sd.parent.name}/{sd.name}"):
            continue
        evaluate_run(sd, a.site, a.split, a.n_boot, a.recalibrate_on_target_val)


def cmd_aggregate(a):
    rows = []
    for f in Path(a.run).glob("seed_*/eval_*/summary.csv"):
        s = pd.read_csv(f, index_col=0)
        for op in s.columns:
            rows.append(s[op].rename(f"{f.parent.name}|{op}|{f.parent.parent.name}"))
    if not rows:
        raise SystemExit("no summaries found")
    df = pd.DataFrame(rows)
    df.index = pd.MultiIndex.from_tuples([tuple(i.split("|")) for i in df.index],
                                         names=["eval", "operating_point", "seed"])
    agg = df.apply(pd.to_numeric, errors="coerce").groupby(level=[0, 1]).agg(["mean", "std"])
    agg.to_csv(Path(a.run) / "summary_across_seeds.csv")
    log.info("\n%s", agg.T.round(3).to_string())


def cmd_explain(a):
    from .explain import explain_sample
    from .train import Sequences, load_sequence_model
    run = Path(a.run)
    if _done(a, run / f"explain_{a.site}_{a.split}" / "ig_by_lag.csv", "explain"):
        return
    rc = json.loads((run / "run_config.json").read_text())
    cfg, win, groups = load_site_windows(a.site)
    df = _subset(win, a.split)
    pre = Preprocessor.load(run / "preprocessor.joblib")
    seqs = Sequences(df, pre.transform(df))
    model, device = load_sequence_model(run)
    by_feat, by_group, lags = explain_sample(model, seqs, pre.columns, groups, device,
                                             a.n, a.n)
    out = run / f"explain_{a.site}_{a.split}"
    out.mkdir(exist_ok=True)
    by_feat.to_csv(out / "ig_by_feature_signed.csv")
    by_group.to_csv(out / "ig_share_by_group.csv")
    lags.groupby("cls").mean().T.to_csv(out / "ig_by_lag.csv")
    log.info("attribution share by group:\n%s", by_group.round(3).to_string())


def cmd_deploy(a):
    """Silent deployment: frozen model + frozen calibration + frozen thresholds."""
    run = Path(a.run)
    rc = json.loads((run / "run_config.json").read_text())
    tag = f"deploy_{a.site}_{a.split}"
    if _done(a, run / tag / "DEPLOY_DONE", f"deploy {run.parent.name}/{run.name}") or \
            _done(a, run / tag / "monthly.csv", f"deploy {run.parent.name}/{run.name}"):
        return
    df, _ = evaluate_run(run, a.site, a.split, a.n_boot, tag=tag)
    out = run / tag
    # patient overlap with the development data
    dev = pd.read_csv(Path(load_config(rc["site"])["site"]["dirs"]["labelled"]) / "split.csv",
                      dtype=str)
    seen = set(dev.loc[dev["split"].isin(["train", "val"]), "subject_id"])
    overlap = df.groupby("stay_id")["subject_id"].first().isin(seen)
    lab = pd.read_parquet(Path(load_config(a.site)["site"]["dirs"]["labelled"])
                          / "stays_labelled.parquet")
    month = lab.set_index("stay_id")["admit"].dt.to_period("M").astype(str)
    df["month"] = df["stay_id"].map(month)
    thresholds = json.loads((run / "thresholds.json").read_text())
    step = rc["common"]["windows"]["step_h"]
    rows = []
    for mth, d in df.groupby("month"):
        if d["aki"].any() and (~d["aki"]).any():
            rows.append(dict(month=mth, stays=d["stay_id"].nunique(),
                             aki_stays=d.loc[d["aki"], "stay_id"].nunique(),
                             **M.summary(d, thresholds[a.op], step)))
    pd.DataFrame(rows).to_csv(out / "monthly.csv", index=False)
    rep = dict(stays=int(len(overlap)), stays_of_patients_in_training=int(overlap.sum()))
    (out / "overlap.json").write_text(json.dumps(rep, indent=1))
    log.info("deployment: %s", rep)
    if overlap.any():
        d_new = df[~df["subject_id"].isin(seen)]
        rep["new_patient_stays"] = int(d_new["stay_id"].nunique())
        rep["new_patient_aki_stays"] = int(d_new.loc[d_new["aki"], "stay_id"].nunique())
        (out / "overlap.json").write_text(json.dumps(rep, indent=1))
        pd.Series(M.summary(d_new, thresholds[a.op], step)).to_csv(
            out / "summary_new_patients_only.csv")
    (out / "DEPLOY_DONE").write_text("complete\n")


def main():
    p = argparse.ArgumentParser(prog="aki")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("extract", "label", "windows", "split"):
        s = sub.add_parser(name); s.add_argument("--site", required=True)
    s = sub.add_parser("core"); s.add_argument("--sites", nargs="+", required=True)
    s = sub.add_parser("train")
    s.add_argument("--site", required=True)
    s.add_argument("--model", choices=["gru", "lstm", "transformer", "hgb", "logreg"],
                   default="gru")
    s.add_argument("--name", required=True)
    s.add_argument("--seeds", type=int, nargs="+", default=[0])
    s.add_argument("--groups", nargs="+", choices=ALL_GROUPS,
                   help="feature groups to include (default: common.yaml)")
    s.add_argument("--core", help="path to core_features_*.json to restrict features")
    s.add_argument("--loss", choices=["bce", "focal"],
                   help="override training.loss in common.yaml (sequence models only)")
    s.add_argument("--pos_weight", help="override training.pos_weight: auto | sqrt | number")
    s = sub.add_parser("evaluate")
    s.add_argument("--run", required=True); s.add_argument("--site", required=True)
    s.add_argument("--op", default="orig_spec_at_sens0.75",
                   help="operating point used for the monthly table and the new-patients-only summary")
    s.add_argument("--split", default="test",
                   choices=["train", "val", "test", "dev", "holdout", "all"])
    s.add_argument("--n_boot", type=int, default=500)
    s.add_argument("--recalibrate_on_target_val", action="store_true")
    s = sub.add_parser("explain")
    s.add_argument("--run", required=True); s.add_argument("--site", required=True)
    s.add_argument("--split", default="test"); s.add_argument("--n", type=int, default=200)
    s = sub.add_parser("deploy")
    s.add_argument("--run", required=True); s.add_argument("--site", required=True)
    s.add_argument("--op", default="orig_spec_at_sens0.75",
                   help="operating point used for the monthly table and the new-patients-only summary")    
    s.add_argument("--split", default="holdout", choices=["holdout", "all"],
                   help="holdout = temporal hold-out of the training site; "
                        "all = a separate deployment extract with its own site config")
    s.add_argument("--n_boot", type=int, default=200)
    s = sub.add_parser("aggregate"); s.add_argument("--run", required=True)
    s = sub.add_parser("thresholds", help="re-derive operating points for trained runs")
    s.add_argument("--run", required=True)
    s.add_argument("--site", help="site whose config to use (default: the run's own site)")
    for sp in sub.choices.values():
        sp.add_argument("--resume", action="store_true",
                        help="skip work whose outputs already exist; continue interrupted "
                             "training from its last epoch checkpoint")
    a = p.parse_args()
    # Every run also writes logs/<command>_<site>_<timestamp>.log (no shell redirection needed)
    import datetime
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    who = getattr(a, "site", None) or "_".join(getattr(a, "sites", []) or []) \
        or Path(getattr(a, "run", "") or "").name
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{a.cmd}_{who}_{stamp}.log"
    fmt = "%(asctime)s %(levelname)s %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt,
                        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)])
    log.info("command: %s | log file: %s", " ".join(sys.argv), log_file)
    try:
        globals()[f"cmd_{a.cmd}"](a)
    except Exception:
        log.exception("FAILED — full traceback above (also in %s)", log_file)
        raise


if __name__ == "__main__":
    main()