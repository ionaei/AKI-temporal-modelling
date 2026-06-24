"""
bootstrap_ci.py

Bootstrap confidence intervals for all reported model metrics.
Operates on saved *_predictions.csv files — no retraining needed.

Metrics with CIs:
  Window-level:  AUROC, AUPRC
  Admission-level: sensitivity, specificity, PPV, NPV, FA ratio
  Lead time:     median, IQR, pct ≥24h, pct ≥48h

Bootstrap is at ADMISSION level (not window level) to respect the
clustered structure of the data — windows within the same admission
are not independent observations.

Usage
-----
    # GOSH best model
    python bootstrap_ci.py \\
        --predictions data/models/lstm_focal_attention/test_predictions.csv \\
        --out         data/models/lstm_focal_attention/bootstrap_ci.json \\
        --n_boot      1000

    # PIC best model
    python bootstrap_ci.py \\
        --predictions pic_models/pic_lstm_bce_attention/test_predictions.csv \\
        --out         pic_models/pic_lstm_bce_attention/bootstrap_ci.json \\
        --n_boot      1000

    # All models at once — see bottom of script
    python bootstrap_ci.py --all_gosh
    python bootstrap_ci.py --all_pic
    
    python bootstrap_ci.py --predictions data/models/gru_focal_h128_l2_attention/test_predictions.csv --threshold 0.21 --n_boot 1000
    

Output
------
    bootstrap_ci.json — all metrics with mean, 2.5th, 97.5th percentile CIs
    bootstrap_ci_table.tex — LaTeX table ready to paste into paper
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score
)


# ─────────────────────────────────────────────────────────
#  Admission-level metrics at a given threshold
# ─────────────────────────────────────────────────────────

def admission_metrics(preds_df: pd.DataFrame,
                       all_ids:  list,
                       aki_ids:  set,
                       threshold: float) -> dict:
    tp = fp = fn = tn = 0
    leads = []

    for adm_id in all_ids:
        adm    = preds_df[preds_df['admission_id'] == adm_id]
        is_aki = adm_id in aki_ids

        if is_aki:
            pre   = adm[adm['hours_to_onset'].notna()]
            fired = (pre['prob'] >= threshold).any() if len(pre) > 0 else False
            if fired:
                tp += 1
                first = pre[pre['prob'] >= threshold].iloc[0]
                leads.append(float(first['hours_to_onset']))
            else:
                fn += 1
                leads.append(np.nan)
        else:
            fired = (adm['prob'] >= threshold).any()
            if fired: fp += 1
            else:     tn += 1

    n   = tp + fp + fn + tn
    eps = 1e-9
    sens   = tp / (tp + fn + eps)
    spec   = tn / (tn + fp + eps)
    ppv    = tp / (tp + fp + eps)
    npv    = tn / (tn + fn + eps)
    fa_r   = fp / (tp + eps)

    lead_s  = pd.Series(leads).dropna()
    total   = len(aki_ids)
    detected= len(lead_s)

    return {
        'sensitivity':      sens,
        'specificity':      spec,
        'ppv':              ppv,
        'npv':              npv,
        'fa_ratio':         fa_r,
        'detection_rate':   detected / total if total > 0 else 0,
        'lead_median_h':    float(lead_s.median())           if len(lead_s) > 0 else np.nan,
        'lead_iqr_25':      float(lead_s.quantile(0.25))     if len(lead_s) > 0 else np.nan,
        'lead_iqr_75':      float(lead_s.quantile(0.75))     if len(lead_s) > 0 else np.nan,
        'pct_detected_24h': float((lead_s >= 24).sum()/total) if total > 0 else 0,
        'pct_detected_48h': float((lead_s >= 48).sum()/total) if total > 0 else 0,
    }


def window_metrics(preds_df: pd.DataFrame) -> dict:
    y    = preds_df['label'].values
    prob = preds_df['prob'].values
    if len(np.unique(y)) < 2:
        return {'auroc': np.nan, 'auprc': np.nan}
    return {
        'auroc': float(roc_auc_score(y, prob)),
        'auprc': float(average_precision_score(y, prob)),
    }


def youden_threshold(preds_df: pd.DataFrame,
                      all_ids:  list,
                      aki_ids:  set,
                      min_t:    float = 0.05) -> float:
    """Find Youden-J optimal threshold from these predictions."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thrs = roc_curve(preds_df['label'], preds_df['prob'])
    j       = tpr - fpr
    best    = float(thrs[np.argmax(j)])
    return max(best, min_t)


# ─────────────────────────────────────────────────────────
#  Bootstrap
# ─────────────────────────────────────────────────────────

def bootstrap_ci(preds_path: str,
                  n_boot:     int   = 1000,
                  alpha:      float = 0.05,
                  min_t:      float = 0.05,
                  seed:       int   = 42,
                  threshold:  float = None) -> dict:
    """
    Bootstrap CIs at admission level.

    Algorithm:
    1. Get list of unique admission IDs
    2. On each bootstrap iteration:
       a. Resample admissions WITH replacement
       b. Reconstruct window-level predictions for resampled admissions
       c. Find Youden-J threshold on the bootstrap sample
       d. Compute all metrics
    3. Report 2.5th and 97.5th percentile across iterations
    """
    rng   = np.random.default_rng(seed)
    preds = pd.read_csv(preds_path)
    preds.columns = preds.columns.str.lower()

    all_ids = list(preds['admission_id'].unique())
    aki_ids = set(preds[preds['label'] == 1]['admission_id'].unique())

    print(f"  {len(all_ids)} admissions  "
          f"({len(aki_ids)} AKI, {len(all_ids)-len(aki_ids)} non-AKI)")

    # Point estimates on full data
    if threshold is not None:
            threshold_pt = threshold
            print(f"  Using fixed threshold: {threshold_pt:.4f}")
    else:
        val_pred_path = Path(preds_path).parent / 'val_predictions.csv'
        if val_pred_path.exists():
            val_preds = pd.read_csv(val_pred_path)
            val_preds.columns = val_preds.columns.str.lower()
            threshold_pt = youden_threshold(
                val_preds,
                list(val_preds['admission_id'].unique()),
                set(val_preds[val_preds['label']==1]['admission_id'].unique()),
                min_t
            )
            print(f"  Threshold from val_predictions.csv: {threshold_pt:.4f}")
        else:
            threshold_pt = youden_threshold(preds, all_ids, aki_ids, min_t)
            print(f"  WARNING: using test-set threshold {threshold_pt:.4f} "
                    f"— val_predictions.csv not found")
    wm_pt        = window_metrics(preds)
    am_pt        = admission_metrics(preds, all_ids, aki_ids, threshold_pt)
    point_est    = {**wm_pt, **am_pt, 'threshold': threshold_pt}

    print(f"  Point estimates: AUROC={wm_pt['auroc']:.3f}  "
          f"AUPRC={wm_pt['auprc']:.3f}  "
          f"Sens={am_pt['sensitivity']:.3f}  "
          f"Spec={am_pt['specificity']:.3f}  "
          f"Threshold={threshold_pt:.3f}")
    print(f"  Running {n_boot} bootstrap iterations...")

    boot_results = {k: [] for k in point_est.keys()}

    for i in range(n_boot):
        # Resample admissions with replacement
        boot_ids     = rng.choice(all_ids, size=len(all_ids), replace=True)
        boot_aki_ids = set(bid for bid in boot_ids if bid in aki_ids)

        if len(boot_aki_ids) == 0:
            continue   # skip degenerate samples

        # Reconstruct window predictions for resampled admissions
        # Use pd.concat to handle duplicate admission IDs correctly
        boot_frames = []
        for bid in boot_ids:
            boot_frames.append(preds[preds['admission_id'] == bid])
        boot_df = pd.concat(boot_frames, ignore_index=True)

        # Unique IDs for this bootstrap (may have duplicates — use index)
        # Relabel admission IDs to avoid confusion with duplicate sampling
        id_map = {}
        new_frames = []
        for j, bid in enumerate(boot_ids):
            frame = preds[preds['admission_id'] == bid].copy()
            frame['admission_id'] = f'{bid}_{j}'   # make unique
            new_frames.append(frame)
        boot_df_unique = pd.concat(new_frames, ignore_index=True)

        boot_unique_ids  = list(boot_df_unique['admission_id'].unique())
        boot_unique_aki  = set(f'{bid}_{j}'
                               for j, bid in enumerate(boot_ids)
                               if bid in aki_ids)

        if len(boot_unique_aki) == 0:
            continue

        try:
            t_boot = threshold_pt
            wm_b   = window_metrics(boot_df_unique)
            am_b   = admission_metrics(
                boot_df_unique, boot_unique_ids, boot_unique_aki, t_boot
            )
            boot_sample = {**wm_b, **am_b, 'threshold': t_boot}
            for k in boot_results:
                if k in boot_sample:
                    boot_results[k].append(boot_sample[k])
        except Exception:
            continue   # skip failed iterations silently

        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{n_boot} iterations complete...")

    # Compute CIs
    lo = alpha / 2 * 100
    hi = (1 - alpha / 2) * 100

    results = {}
    for k, vals in boot_results.items():
        vals_clean = [v for v in vals if not np.isnan(v)]
        if len(vals_clean) < 10:
            results[k] = {
                'mean':  point_est.get(k, np.nan),
                'lower': np.nan,
                'upper': np.nan,
                'n_valid_boots': len(vals_clean),
            }
        else:
            results[k] = {
                'mean':  float(np.mean(vals_clean)),
                'lower': float(np.percentile(vals_clean, lo)),
                'upper': float(np.percentile(vals_clean, hi)),
                'point_estimate': float(point_est.get(k, np.nan)),
                'n_valid_boots':  len(vals_clean),
            }

    print(f"\n  Bootstrap complete ({n_boot} iterations)")
    print(f"\n  {'Metric':<25} {'Point':>8} {'95% CI':>20}")
    print(f"  {'-'*55}")
    key_metrics = ['auroc','auprc','sensitivity','specificity',
                   'ppv','fa_ratio','lead_median_h',
                   'pct_detected_24h','pct_detected_48h']
    for k in key_metrics:
        if k not in results:
            continue
        r = results[k]
        pt = r.get('point_estimate', r['mean'])
        lo_v = r['lower']
        hi_v = r['upper']
        if np.isnan(lo_v):
            ci_str = 'insufficient data'
        else:
            ci_str = f"({lo_v:.3f} – {hi_v:.3f})"
        print(f"  {k:<25} {pt:>8.3f} {ci_str:>20}")

    return results


# ─────────────────────────────────────────────────────────
#  LaTeX table output
# ─────────────────────────────────────────────────────────

def format_ci(r: dict, decimals: int = 3, pct: bool = False) -> str:
    """Format a metric dict as 'value (lower–upper)' string."""
    pt   = r.get('point_estimate', r['mean'])
    lo_v = r['lower']
    hi_v = r['upper']
    mult = 100 if pct else 1
    fmt  = f'.{decimals}f'
    if np.isnan(lo_v):
        return f'{pt*mult:{fmt}}'
    return f'{pt*mult:{fmt}} ({lo_v*mult:{fmt}}--{hi_v*mult:{fmt}})'


def make_latex_table(results: dict, run_name: str, out_path: Path):
    """Generate a LaTeX table row with CIs for the paper."""
    def f(k, decimals=3, pct=False):
        return format_ci(results[k], decimals, pct) if k in results else '---'

    lines = [
        f"% Bootstrap CI table — {run_name}",
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\caption{Bootstrap confidence intervals (95\%, $n=1000$ admission-level "
        r"resamples) for key performance metrics on the held-out test set. "
        r"Values reported as point estimate (95\% CI).}",
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"\textbf{Metric} & \textbf{Value (95\% CI)} \\",
        r"\midrule",
        f"AUROC & {f('auroc')} \\\\",
        f"AUPRC & {f('auprc')} \\\\",
        r"\midrule",
        f"Sensitivity & {f('sensitivity')} \\\\",
        f"Specificity & {f('specificity')} \\\\",
        f"PPV & {f('ppv')} \\\\",
        f"NPV & {f('npv')} \\\\",
        f"FA ratio & {f('fa_ratio')} \\\\",
        r"\midrule",
        f"Detection rate & {f('detection_rate')} \\\\",
        f"Lead time median (h) & {f('lead_median_h', decimals=1)} \\\\",
        f"Detected $\\geq$24h (\\%) & {f('pct_detected_24h', pct=True, decimals=1)} \\\\",
        f"Detected $\\geq$48h (\\%) & {f('pct_detected_48h', pct=True, decimals=1)} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\label{tab:bootstrap_ci}",
        r"\end{table}",
    ]

    with open(out_path, 'w') as f_out:
        f_out.write('\n'.join(lines))
    print(f"  LaTeX table → {out_path}")


# ─────────────────────────────────────────────────────────
#  Batch helpers
# ─────────────────────────────────────────────────────────

GOSH_RUNS = [
    'lstm_focal_attention',
    'gru_focal_attention',
    'transformer_focal',
]

PIC_RUNS = [
    'pic_lstm_bce_attention',
    'pic_gru_bce_attention',
    'pic_transformer_bce_attention',
]


def run_all(runs:       list,
             models_dir: str,
             n_boot:     int,
             min_t:      float):
    for run in runs:
        pred_path = Path(models_dir) / run / 'test_predictions.csv'
        if not pred_path.exists():
            print(f"\nSkipping {run} — test_predictions.csv not found")
            continue

        print(f"\n{'='*55}")
        print(f"  {run}")
        print(f"{'='*55}")

        results = bootstrap_ci(
            str(pred_path), n_boot=n_boot, min_t=min_t
        )

        out_dir = Path(models_dir) / run
        with open(out_dir / 'bootstrap_ci.json', 'w') as f:
            json.dump(results, f, indent=2)

        make_latex_table(
            results, run,
            out_dir / 'bootstrap_ci_table.tex'
        )
        print(f"  Saved → {out_dir}/bootstrap_ci.json")


# ─────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Bootstrap CIs for AKI model evaluation metrics'
    )
    p.add_argument('--predictions', default=None,
                   help='Path to test_predictions.csv for a single run')
    p.add_argument('--out',         default=None,
                   help='Output JSON path (default: same dir as predictions)')
    p.add_argument('--n_boot',      type=int,   default=1000)
    p.add_argument('--alpha',       type=float, default=0.05,
                   help='Significance level (default 0.05 → 95% CI)')
    p.add_argument('--min_threshold', type=float, default=0.05)
    p.add_argument('--seed',        type=int,   default=42)
    # Batch options
    p.add_argument('--all_gosh',    action='store_true',
                   help='Run all GOSH models')
    p.add_argument('--all_pic',     action='store_true',
                   help='Run all PIC models')
    p.add_argument('--models_dir',  default='data/models',
                   help='Models directory (used with --all_gosh / --all_pic)')
    p.add_argument('--pic_models_dir', default='pic_models')
    p.add_argument('--threshold', type=float, default = 0.22)
    return p.parse_args()


def main():
    args = parse_args()

    if args.all_gosh:
        print("Running bootstrap CIs for all GOSH models...")
        run_all(GOSH_RUNS, args.models_dir,
                args.n_boot, args.min_threshold)
        return

    if args.all_pic:
        print("Running bootstrap CIs for all PIC models...")
        run_all(PIC_RUNS, args.pic_models_dir,
                args.n_boot, args.min_threshold)
        return

    if args.predictions is None:
        print("Provide --predictions PATH or use --all_gosh / --all_pic")
        return

    pred_path = Path(args.predictions)
    if not pred_path.exists():
        raise FileNotFoundError(f"Not found: {pred_path}")

    print(f"\nBootstrap CI — {pred_path.parent.name}")
    results = bootstrap_ci(
        str(pred_path),
        n_boot    = args.n_boot,
        alpha     = args.alpha,
        min_t     = args.min_threshold,
        seed      = args.seed,
        threshold = args.threshold,
    )

    out_path = Path(args.out) if args.out else pred_path.parent / 'bootstrap_ci.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  JSON → {out_path}")

    tex_path = out_path.parent / 'bootstrap_ci_table.tex'
    make_latex_table(results, pred_path.parent.name, tex_path)


if __name__ == '__main__':
    main()
