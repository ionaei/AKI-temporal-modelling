"""
monthly_evaluation.py

Silent deployment evaluation: runs the trained LSTM on the full
year of data, segmented into calendar months. Produces per-month
metrics with bootstrap CIs and a longitudinal plot showing
performance stability over time.

Usage
-----
    python monthly_evaluation.py --run transformer_bce --models_dir data/models --windows further_analysis/data/windows_no_dialysis.parquet --cohort further_analysis/data/cohort_no_dialysis.csv --split_index data/processed/split_index_matched.csv --threshold 0.69 --out_dir further_analysis/data/silent_deployment --n_boot 1000

Notes
-----
- Threshold must be the Youden-J value from the validation set.
  Pass it via --threshold. Do NOT recompute from deployment data.
- Bootstrap resamples at admission level within each month.
- Months with fewer than 5 AKI admissions get wider CIs — flagged.
- The deployment windows are processed through load_data with
  fit_scaler=False so the training scaler is applied.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from dataset import load_data, AKISequenceDataset, collate_fn
from models  import build_model


# ─────────────────────────────────────────────────────────
#  Argument parsing
# ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--run',         required=True)
    p.add_argument('--models_dir',  default='data/models')
    p.add_argument('--windows',     required=True,
                   help='Deployment windows CSV (full year)')
    p.add_argument('--cohort',      required=True,
                   help='Deployment cohort CSV (full year)')
    p.add_argument('--split_index', default=None,
                   help='Optional — if windows already have split column '
                        'leave as None')
    p.add_argument('--threshold',   type=float, required=True,
                   help='Youden-J threshold from validation set')
    p.add_argument('--out_dir',     default='data/silent_deployment')
    p.add_argument('--n_boot',      type=int,   default=1000)
    p.add_argument('--min_aki',     type=int,   default=3,
                   help='Minimum AKI admissions per month to include')
    p.add_argument('--seed',        type=int,   default=42)
    return p.parse_args()


# ─────────────────────────────────────────────────────────
#  Attach month labels from cohort
# ─────────────────────────────────────────────────────────

def attach_months(windows: pd.DataFrame,
                   cohort:  pd.DataFrame) -> pd.DataFrame:
    """
    Attach calendar month to each window via the cohort admit date.
    """
    cohort = cohort.copy()
    cohort.columns = cohort.columns.str.lower()

    date_col = None
    for c in ['admit_dt', 'admittime', 'intime', 'admit_date']:
        if c in cohort.columns:
            date_col = c
            break

    if date_col is None:
        # Derive from first window_start per admission
        for c in ['window_start', 'window_end']:
            if c in windows.columns:
                first = (windows.groupby('admission_id')[c]
                         .min().reset_index()
                         .rename(columns={c: 'admit_dt'}))
                cohort = cohort.merge(first, on='admission_id', how='left')
                date_col = 'admit_dt'
                break

    if date_col is None:
        raise ValueError("Cannot find admit date in cohort or windows")

    cohort[date_col] = pd.to_datetime(cohort[date_col])
    cohort['month']  = cohort[date_col].dt.to_period('M')

    adm_month = cohort[['admission_id', 'month']].drop_duplicates('admission_id')
    adm_month['admission_id'] = adm_month['admission_id'].astype(str)
    windows['admission_id']   = windows['admission_id'].astype(str)

    windows = windows.merge(adm_month, on='admission_id', how='left')
    n_missing = windows['month'].isna().sum()
    if n_missing > 0:
        print(f"  WARNING: {n_missing} windows have no month — dropped")
        windows = windows[windows['month'].notna()].copy()

    return windows


# ─────────────────────────────────────────────────────────
#  Inference — run model on all windows, return predictions df
# ─────────────────────────────────────────────────────────

def run_inference(model:     torch.nn.Module,
                   windows:  pd.DataFrame,
                   feat_cols: list,
                   device:   torch.device,
                   batch_size: int = 32) -> pd.DataFrame:
    """Run inference on all windows, return per-step predictions."""
    dataset = AKISequenceDataset(windows, feat_cols)
    loader  = DataLoader(dataset, batch_size=batch_size,
                         shuffle=False, collate_fn=collate_fn)

    model.eval()
    records = []
    with torch.no_grad():
        for batch in loader:
            x       = batch['x'].to(device)
            lengths = batch['lengths']
            out     = model(x, lengths)
            logits  = out[0] if isinstance(out, tuple) else out
            probs   = torch.sigmoid(logits).cpu().numpy()
            y_np    = batch['y'].numpy()

            for b, L in enumerate(lengths):
                L      = L.item()
                adm_id = batch['admission_ids'][b]
                h2o    = batch['hours_to_onset'][b]
                hsa    = batch['hours_since_admit'][b]
                si     = batch['step_indices'][b]

                for t in range(L):
                    records.append({
                        'admission_id':      adm_id,
                        'step_index':        int(si[t]) if t < len(si) else t,
                        'prob':              float(probs[b, t]),
                        'label':             int(y_np[b, t]),
                        'hours_to_onset':    float(h2o[t])
                                             if t < len(h2o) else np.nan,
                        'hours_since_admit': float(hsa[t])
                                             if t < len(hsa) else np.nan,
                        'aki_stage':         int(batch['aki_stages'][b]),
                    })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────
#  Metrics for one month's predictions
# ─────────────────────────────────────────────────────────

def compute_metrics(preds:     pd.DataFrame,
                     threshold: float) -> dict:
    """Compute all metrics for a set of predictions at fixed threshold."""
    all_ids = list(preds['admission_id'].unique())
    aki_ids = set(preds[preds['label'] == 1]['admission_id'].unique())

    if len(aki_ids) == 0:
        return None

    # Window-level
    y    = preds['label'].values
    prob = preds['prob'].values
    try:
        auroc = float(roc_auc_score(y, prob)) \
                if len(np.unique(y)) > 1 else np.nan
        auprc = float(average_precision_score(y, prob)) \
                if len(np.unique(y)) > 1 else np.nan
    except Exception:
        auroc = auprc = np.nan

    # Admission-level
    tp = fp = fn = tn = 0
    leads = []
    for adm_id in all_ids:
        adm    = preds[preds['admission_id'] == adm_id]
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
            if (adm['prob'] >= threshold).any(): fp += 1
            else:                                tn += 1

    eps  = 1e-9
    sens = tp / (tp + fn + eps)
    spec = tn / (tn + fp + eps)
    ppv  = tp / (tp + fp + eps)
    npv  = tn / (tn + fn + eps)
    fa_r = fp / (tp + eps)

    lead_s    = pd.Series(leads).dropna()
    total_aki = len(aki_ids)

    return {
        'n_admissions':     len(all_ids),
        'n_aki':            total_aki,
        'n_nonaki':         len(all_ids) - total_aki,
        'auroc':            auroc,
        'auprc':            auprc,
        'sensitivity':      sens,
        'specificity':      spec,
        'ppv':              ppv,
        'npv':              npv,
        'fa_ratio':         fa_r,
        'detection_rate':   len(lead_s) / total_aki if total_aki > 0 else 0,
        'lead_median_h':    float(lead_s.median())           if len(lead_s) > 0 else np.nan,
        'lead_iqr_25':      float(lead_s.quantile(0.25))     if len(lead_s) > 0 else np.nan,
        'lead_iqr_75':      float(lead_s.quantile(0.75))     if len(lead_s) > 0 else np.nan,
        'pct_detected_24h': float((lead_s >= 24).sum() / total_aki)
                            if total_aki > 0 else 0,
        'pct_detected_48h': float((lead_s >= 48).sum() / total_aki)
                            if total_aki > 0 else 0,
    }


# ─────────────────────────────────────────────────────────
#  Bootstrap CIs for one month
# ─────────────────────────────────────────────────────────

def bootstrap_month(preds:     pd.DataFrame,
                     threshold: float,
                     n_boot:    int,
                     seed:      int = 42) -> dict:
    """Bootstrap admission-level CIs for one month."""
    rng     = np.random.default_rng(seed)
    all_ids = list(preds['admission_id'].unique())
    aki_ids = set(preds[preds['label'] == 1]['admission_id'].unique())

    boot_sens = []; boot_spec  = []
    boot_auro = []; boot_aupr  = []
    boot_lead = []; boot_24h   = []
    boot_ppv  = []; boot_far   = []

    for _ in range(n_boot):
        boot_ids = rng.choice(all_ids, size=len(all_ids), replace=True)
        boot_aki = set(bid for bid in boot_ids if bid in aki_ids)

        if len(boot_aki) == 0:
            continue

        frames = []
        for j, bid in enumerate(boot_ids):
            f = preds[preds['admission_id'] == bid].copy()
            f['admission_id'] = f'{bid}_{j}'
            frames.append(f)
        bd = pd.concat(frames, ignore_index=True)
        bd_aki = set(f'{bid}_{j}' for j, bid in enumerate(boot_ids)
                     if bid in aki_ids)

        m = compute_metrics(bd.assign(
            **{'admission_id': bd['admission_id']}
        ), threshold)
        # Recompute with unique IDs
        m2 = compute_metrics(bd, threshold)
        if m2 is None:
            continue

        boot_sens.append(m2['sensitivity'])
        boot_spec.append(m2['specificity'])
        boot_ppv.append(m2['ppv'])
        boot_far.append(m2['fa_ratio'])
        boot_24h.append(m2['pct_detected_24h'])
        if not np.isnan(m2['lead_median_h']):
            boot_lead.append(m2['lead_median_h'])
        if not np.isnan(m2['auroc']):
            boot_auro.append(m2['auroc'])
        if not np.isnan(m2['auprc']):
            boot_aupr.append(m2['auprc'])

    def ci(vals):
        v = [x for x in vals if not np.isnan(x)]
        if len(v) < 10:
            return (np.nan, np.nan)
        return (float(np.percentile(v, 2.5)),
                float(np.percentile(v, 97.5)))

    return {
        'sensitivity_ci':   ci(boot_sens),
        'specificity_ci':   ci(boot_spec),
        'ppv_ci':           ci(boot_ppv),
        'fa_ratio_ci':      ci(boot_far),
        'auroc_ci':         ci(boot_auro),
        'auprc_ci':         ci(boot_aupr),
        'lead_median_ci':   ci(boot_lead),
        'pct_24h_ci':       ci(boot_24h),
        'n_boot_valid':     len(boot_sens),
    }


# ─────────────────────────────────────────────────────────
#  Longitudinal plots
# ─────────────────────────────────────────────────────────

def plot_monthly_metrics(monthly_df: pd.DataFrame,
                          threshold:  float,
                          save_dir:   Path):
    """
    Four-panel longitudinal plot:
      (a) AUROC and AUPRC over time
      (b) Sensitivity and specificity over time
      (c) Lead time over time
      (d) Detection rate ≥24h over time
    """
    df  = monthly_df.sort_values('month')
    x   = range(len(df))
    xlt = [str(m) for m in df['month']]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f'Silent deployment — monthly performance\n'
        f'(threshold={threshold:.3f}, Youden-J from validation set)',
        fontsize=12
    )

    def plot_metric(ax, col, ci_col, colour, label, ylim=(0,1)):
        y_vals = df[col].values
        ax.plot(x, y_vals, 'o-', color=colour, lw=2, ms=6,
                label=label, zorder=3)
        if ci_col in df.columns:
            lowers = df[ci_col].apply(lambda c: c[0] if isinstance(c,tuple) else np.nan)
            uppers = df[ci_col].apply(lambda c: c[1] if isinstance(c,tuple) else np.nan)
            ax.fill_between(x, lowers, uppers,
                            alpha=0.15, color=colour)
        ax.set_ylim(ylim)

    # (a) AUROC + AUPRC
    ax = axes[0, 0]
    plot_metric(ax, 'auroc', 'auroc_ci', '#2E75B6', 'AUROC')
    plot_metric(ax, 'auprc', 'auprc_ci', '#ED7D31', 'AUPRC')
    ax.axhline(0.5, color='gray', lw=0.8, linestyle='--', label='Chance')
    ax.set_title('(a) Window-level discrimination', fontweight='bold')
    ax.set_ylabel('Score'); ax.legend(fontsize=8); ax.grid(alpha=0.25)

    # (b) Sensitivity + Specificity
    ax = axes[0, 1]
    plot_metric(ax, 'sensitivity', 'sensitivity_ci', '#C0392B', 'Sensitivity')
    plot_metric(ax, 'specificity', 'specificity_ci', '#2E75B6', 'Specificity')
    ax.set_title('(b) Admission-level sensitivity / specificity',
                 fontweight='bold')
    ax.set_ylabel('Score'); ax.legend(fontsize=8); ax.grid(alpha=0.25)

    # (c) Lead time median
    ax = axes[1, 0]
    y_lead = df['lead_median_h'].values
    ax.plot(x, y_lead, 'o-', color='#7030A0', lw=2, ms=6,
            label='Median lead time', zorder=3)
    if 'lead_median_ci' in df.columns:
        lowers = df['lead_median_ci'].apply(
            lambda c: c[0] if isinstance(c,tuple) else np.nan)
        uppers = df['lead_median_ci'].apply(
            lambda c: c[1] if isinstance(c,tuple) else np.nan)
        ax.fill_between(x, lowers, uppers, alpha=0.15, color='#7030A0')
    ax.axhline(24, color='gray', lw=1, linestyle='--', label='24h')
    ax.axhline(48, color='gray', lw=1, linestyle=':',  label='48h')
    ax.set_title('(c) Early warning lead time (h)', fontweight='bold')
    ax.set_ylabel('Hours'); ax.legend(fontsize=8); ax.grid(alpha=0.25)
    ax.set_ylim(bottom=0)

    # (d) % detected ≥24h
    ax = axes[1, 1]
    plot_metric(ax, 'pct_detected_24h', 'pct_24h_ci',
                '#C0392B', '≥24h detected')
    ax.axhline(0.5, color='gray', lw=0.8, linestyle='--',
               label='50% threshold')
    ax.set_title('(d) Detection rate ≥24h before onset',
                 fontweight='bold')
    ax.set_ylabel('Proportion'); ax.legend(fontsize=8); ax.grid(alpha=0.25)

    # Shared x axis formatting
    for ax in axes.flatten():
        ax.set_xticks(list(x))
        ax.set_xticklabels(xlt, rotation=45, ha='right', fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Mark months with low AKI n
        for i, row in enumerate(df.itertuples()):
            if row.n_aki < 5:
                ax.axvspan(i - 0.4, i + 0.4,
                           alpha=0.1, color='orange', zorder=0)

    # Legend for low-n warning
    orange_patch = mpatches.Patch(color='orange', alpha=0.3,
                                   label='n_AKI < 5 (wide CI)')
    fig.legend(handles=[orange_patch], loc='lower center',
               ncol=1, fontsize=8, framealpha=0.8)

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(save_dir / 'monthly_performance.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot → {save_dir}/monthly_performance.png")


def plot_n_per_month(monthly_df: pd.DataFrame, save_dir: Path):
    """Bar chart of admission counts per month."""
    df = monthly_df.sort_values('month')
    x  = range(len(df))

    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.bar(x, df['n_nonaki'], color='#7FB3D3', label='Non-AKI', alpha=0.85)
    ax.bar(x, df['n_aki'], bottom=df['n_nonaki'],
           color='#C0392B', label='AKI', alpha=0.85)
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(m) for m in df['month']],
                       rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Admissions')
    ax.set_title('Monthly admission counts (AKI vs non-AKI)',
                 fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_dir / 'monthly_counts.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot → {save_dir}/monthly_counts.png")


# ─────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    run_dir = Path(args.models_dir) / args.run
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice:    {DEVICE}")
    print(f"Run:       {run_dir}")
    print(f"Threshold: {args.threshold} (Youden-J from val set)")

    # Load config and model
    with open(run_dir / 'config.json') as f:
        cfg = json.load(f)

    # Load deployment data through dataset.py
    # fit_scaler=False → uses training scaler
    # Pass a dummy split_index if none provided
    print("\nLoading deployment data...")

    windows_df = pd.read_parquet(args.windows)
    windows_df.columns = windows_df.columns.str.lower()

    # If no split column, add a dummy 'test' split for all rows
    if 'split' not in windows_df.columns:
        windows_df['split'] = 'test'
        tmp_split = out_dir / '_tmp_split.csv'
        adm_split = (windows_df[['admission_id']]
                     .drop_duplicates()
                     .assign(subject_id=lambda d: d['admission_id'],
                             aki_detected=0,
                             split='test'))
        adm_split.to_csv(tmp_split, index=False)
        split_index_path = str(tmp_split)
    else:
        split_index_path = args.split_index or str(out_dir / '_tmp_split.csv')

    windows, feat_cols, _ = load_data(
        windows_path     = args.windows,
        split_index_path = split_index_path,
        max_seq_len      = cfg.get('max_seq_len', 100),
        forward_fill     = True,
        models_dir       = str(run_dir),
        fit_scaler       = False,
    )

    print(f"  {len(windows):,} windows | "
          f"{windows['admission_id'].nunique():,} admissions")

    # Attach month labels
    cohort = pd.read_csv(args.cohort)
    windows = attach_months(windows, cohort)
    months  = sorted(windows['month'].dropna().unique())
    print(f"  {len(months)} months: {months[0]} → {months[-1]}")

    # Build model
    ckpt  = torch.load(run_dir / 'best_model.pt', map_location=DEVICE)
    model = build_model(
        architecture = cfg['model'],
        input_size   = len(feat_cols),
        hidden_size  = cfg.get('hidden', 128),
        n_layers     = cfg.get('layers', 2),
        dropout      = cfg.get('dropout', 0.3),
        n_heads      = cfg.get('heads', 4),
        attention    = cfg.get('attention', False),
    ).to(DEVICE)
    model.load_state_dict(ckpt['model_state'])
    print(f"  Model loaded (epoch {ckpt['epoch']}, "
          f"val AUPRC {ckpt['val_auprc']:.4f})")

    # Run inference on full year
    print("\nRunning inference on full deployment set...")
    all_preds = run_inference(model, windows, feat_cols, DEVICE)

    # Attach month to predictions
    adm_month = (windows[['admission_id','month']]
                 .drop_duplicates('admission_id'))
    adm_month['admission_id'] = adm_month['admission_id'].astype(str)
    all_preds['admission_id'] = all_preds['admission_id'].astype(str)
    all_preds = all_preds.merge(adm_month, on='admission_id', how='left')
    all_preds.to_csv(out_dir / 'deployment_predictions.csv', index=False)
    print(f"  Predictions saved → {out_dir}/deployment_predictions.csv")

    # Per-month evaluation
    print(f"\nEvaluating per month (threshold={args.threshold:.4f})...")
    monthly_results = []

    for month in months:
        month_preds = all_preds[all_preds['month'] == month].copy()
        n_aki = month_preds[month_preds['label']==1]['admission_id'].nunique()

        if n_aki < args.min_aki:
            print(f"  {month}: skipped (only {n_aki} AKI admissions)")
            continue

        print(f"  {month}: {month_preds['admission_id'].nunique()} admissions, "
              f"{n_aki} AKI...", end=' ')

        metrics = compute_metrics(month_preds, args.threshold)
        if metrics is None:
            print("skipped (no AKI)")
            continue

        #ci = bootstrap_month(month_preds, args.threshold,
            #                  args.n_boot, args.seed)

        row = {'month': month, **metrics}
        #for k, v in ci.items():
        #    row[k] = v

        monthly_results.append(row)
        print(f"sens={metrics['sensitivity']:.3f}  "
              f"spec={metrics['specificity']:.3f}  "
              f"lead={metrics['lead_median_h']:.0f}h")

    monthly_df = pd.DataFrame(monthly_results)

    # Save results
    # Serialise tuples to strings for CSV
    csv_df = monthly_df.copy()
    for col in csv_df.columns:
        if csv_df[col].apply(lambda x: isinstance(x, tuple)).any():
            csv_df[col] = csv_df[col].apply(
                lambda x: f"{x[0]:.3f}–{x[1]:.3f}"
                if isinstance(x, tuple) and not np.isnan(x[0]) else ''
            )
    csv_df['month'] = csv_df['month'].astype(str)
    csv_df.to_csv(out_dir / 'monthly_metrics.csv', index=False)

    # JSON with full CI tuples
    json_results = []
    for row in monthly_results:
        r = dict(row)
        r['month'] = str(r['month'])
        for k, v in r.items():
            if isinstance(v, tuple):
                r[k] = list(v)
        json_results.append(r)
    with open(out_dir / 'monthly_metrics.json', 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"\n  Monthly metrics → {out_dir}/monthly_metrics.csv")

    # Plots
    print("\nGenerating plots...")
    plot_monthly_metrics(monthly_df, args.threshold, out_dir)
    plot_n_per_month(monthly_df, out_dir)

    # Overall year summary
    print(f"\n{'='*55}")
    print("FULL YEAR SUMMARY")
    print(f"{'='*55}")
    print(f"  Months evaluated:   {len(monthly_df)}")
    print(f"  Mean sensitivity:   {monthly_df['sensitivity'].mean():.3f} "
          f"± {monthly_df['sensitivity'].std():.3f}")
    print(f"  Mean specificity:   {monthly_df['specificity'].mean():.3f} "
          f"± {monthly_df['specificity'].std():.3f}")
    print(f"  Mean lead time:     "
          f"{monthly_df['lead_median_h'].mean():.0f}h "
          f"± {monthly_df['lead_median_h'].std():.0f}h")
    print(f"  Mean ≥24h detected: "
          f"{monthly_df['pct_detected_24h'].mean()*100:.1f}%")

    # Save overall summary
    overall = {
        'threshold':        args.threshold,
        'n_months':         len(monthly_df),
        'mean_sensitivity': float(monthly_df['sensitivity'].mean()),
        'std_sensitivity':  float(monthly_df['sensitivity'].std()),
        'mean_specificity': float(monthly_df['specificity'].mean()),
        'std_specificity':  float(monthly_df['specificity'].std()),
        'mean_auroc':       float(monthly_df['auroc'].mean()),
        'mean_auprc':       float(monthly_df['auprc'].mean()),
        'mean_lead_h':      float(monthly_df['lead_median_h'].mean()),
        'mean_pct_24h':     float(monthly_df['pct_detected_24h'].mean()),
    }
    with open(out_dir / 'yearly_summary.json', 'w') as f:
        json.dump(overall, f, indent=2)

    print(f"\n  All outputs → {out_dir}/")


if __name__ == '__main__':
    main()
