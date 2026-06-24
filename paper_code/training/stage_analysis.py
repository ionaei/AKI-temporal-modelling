"""
stage_analysis.py

Reproduces the two stage analysis figures from the paper:
  1. stage_lead_time.png      — lead time histograms per stage (3 panels)
  2. stage_comparison.png     — 4-panel bar chart with bootstrap CIs

Usage
-----
    python stage_analysis.py --predictions data/models/MODEL OF INTEREST/test_predictions.csv --threshold 'UPDATE_THRESHOLD' --out_dir data/models/transformer_bce/plots --n_boot 1000
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import roc_auc_score, average_precision_score


# ─────────────────────────────────────────────────────────
#  Args
# ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--predictions', required=True,
                   help='Path to test_predictions.csv')
    p.add_argument('--threshold',   type=float, required=True,
                   help='Fixed Youden-J threshold from val set')
    p.add_argument('--out_dir',     default='figures')
    p.add_argument('--n_boot',      type=int, default=1000)
    p.add_argument('--seed',        type=int, default=42)
    return p.parse_args()


# ─────────────────────────────────────────────────────────
#  Core metrics for one subset of predictions
# ─────────────────────────────────────────────────────────

def compute_metrics(preds: pd.DataFrame,
                    threshold: float) -> dict:
    """
    Compute all metrics for a set of predictions at fixed threshold.
    preds must have: admission_id, label, prob, hours_to_onset, aki_stage
    """
    all_ids = list(preds['admission_id'].unique())
    aki_ids = set(preds[preds['label'] == 1]['admission_id'].unique())

    if len(aki_ids) == 0:
        return None

    # Window-level
    y, p = preds['label'].values, preds['prob'].values
    try:
        auroc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan
        auprc = float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else np.nan
    except Exception:
        auroc = auprc = np.nan

    # Admission-level
    tp = fp = fn = tn = 0
    leads = []
    for adm_id in all_ids:
        adm    = preds[preds['admission_id'] == adm_id]
        is_aki = adm_id in aki_ids
        if is_aki:
            pre   = adm[adm['hours_to_onset'].notna() & (adm['hours_to_onset'] >= 0)]
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

    eps       = 1e-9
    lead_s    = pd.Series(leads).dropna()
    total_aki = len(aki_ids)

    return {
        'n':               len(all_ids),
        'n_aki':           total_aki,
        'auroc':           auroc,
        'auprc':           auprc,
        'detection_rate':  len(lead_s) / total_aki if total_aki > 0 else 0,
        'lead_median_h':   float(lead_s.median())       if len(lead_s) > 0 else np.nan,
        'lead_iqr_25':     float(lead_s.quantile(0.25)) if len(lead_s) > 0 else np.nan,
        'lead_iqr_75':     float(lead_s.quantile(0.75)) if len(lead_s) > 0 else np.nan,
        'leads':           lead_s.values,   # for histogram
    }


# ─────────────────────────────────────────────────────────
#  Bootstrap CIs for one subset
# ─────────────────────────────────────────────────────────

def bootstrap_metrics(preds:     pd.DataFrame,
                       threshold: float,
                       n_boot:    int,
                       seed:      int = 42) -> dict:
    rng     = np.random.default_rng(seed)
    all_ids = list(preds['admission_id'].unique())
    aki_ids = set(preds[preds['label'] == 1]['admission_id'].unique())

    boot = {k: [] for k in [
        'auroc', 'auprc', 'detection_rate', 'lead_median_h'
    ]}

    for _ in range(n_boot):
        sample_ids = rng.choice(all_ids, size=len(all_ids), replace=True)
        sample_aki = set(sid for sid in sample_ids if sid in aki_ids)
        if len(sample_aki) == 0:
            continue

        frames = []
        for j, sid in enumerate(sample_ids):
            f = preds[preds['admission_id'] == sid].copy()
            f['admission_id'] = f'{sid}__{j}'
            frames.append(f)
        bd = pd.concat(frames, ignore_index=True)

        m = compute_metrics(bd, threshold)
        if m is None:
            continue
        for k in boot:
            val = m.get(k, np.nan)
            if not np.isnan(val):
                boot[k].append(val)

    def ci(vals):
        v = [x for x in vals if not np.isnan(x)]
        if len(v) < 10:
            return (np.nan, np.nan)
        return (float(np.percentile(v, 2.5)),
                float(np.percentile(v, 97.5)))

    return {k: ci(v) for k, v in boot.items()}


# ─────────────────────────────────────────────────────────
#  Figure 1 — Lead time histograms per stage
# ─────────────────────────────────────────────────────────

STAGE_COLOURS = {
    1: '#E8977A',   # orange-salmon (Stage 1)
    2: '#C0614C',   # mid red       (Stage 2)
    3: '#8B2020',   # dark red      (Stage 3)
}

def plot_lead_time_by_stage(preds:     pd.DataFrame,
                             threshold: float,
                             out_path:  Path):
    stages = [1, 2, 3]
    stage_labels = {1: 'Stage 1', 2: 'Stage 2', 3: 'Stage 3'}

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.patch.set_facecolor('black')

    for ax, stage in zip(axes, stages):
        ax.set_facecolor('black')

        # Get AKI admissions for this stage
        stage_aki_ids = set(
            preds[preds['aki_stage'] == stage]['admission_id'].unique()
        )
        stage_preds = preds[preds['admission_id'].isin(stage_aki_ids)]

        m = compute_metrics(stage_preds, threshold)
        if m is None or len(m['leads']) == 0:
            ax.set_title(f'{stage_labels[stage]}\n(no detections)',
                         color='white')
            continue

        leads = m['leads']
        median_h = m['lead_median_h']
        n_aki    = m['n_aki']
        n_det    = len(leads)

        # Histogram
        ax.hist(leads, bins=20, color=STAGE_COLOURS[stage],
                alpha=0.85, edgecolor='none')

        # Reference lines
        ax.axvline(24,       color='#AAAAAA', lw=1.2, linestyle='--',
                   label='24h')
        ax.axvline(48,       color='#888888', lw=1.0, linestyle=':',
                   label='48h')
        ax.axvline(median_h, color='black',   lw=2.0, linestyle='-',
                   label=f'Median: {median_h:.0f}h')

        ax.set_xlabel('Hours before AKI onset', color='white', fontsize=11)
        ax.set_ylabel('Count',                  color='white', fontsize=11)
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        legend = ax.legend(fontsize=9, facecolor='black',
                           labelcolor='white', framealpha=0.8)

        ax.set_title(
            f'{stage_labels[stage]}\n'
            f'Detected: {n_det}/{n_aki} ({n_det/n_aki*100:.0f}%)',
            color='white', fontsize=11, pad=8
        )

    plt.tight_layout(pad=1.5)
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor='black')
    plt.close()
    print(f"  Figure 1 → {out_path}")


# ─────────────────────────────────────────────────────────
#  Figure 2 — 4-panel bar chart with bootstrap CIs
# ─────────────────────────────────────────────────────────

COLOURS = {
    'All AKI': '#2E6E8E',
    'Stage 1': '#E8A96A',
    'Stage 2': '#C8654A',
    'Stage 3': '#8B2B2B',
}

def plot_stage_comparison(stage_results: dict,
                           stage_cis:     dict,
                           out_path:      Path):
    """
    stage_results: {'All AKI': {...}, 'Stage 1': {...}, ...}
    stage_cis:     {'All AKI': {'auroc': (lo, hi), ...}, ...}
    """
    groups = ['All AKI', 'Stage 1', 'Stage 2', 'Stage 3']
    x      = np.arange(len(groups))
    width  = 0.55

    metrics = [
        ('detection_rate', 'Detection rate\n(admission-level)', (0.0, 1.0)),
        ('auroc',          'AUROC\n(window-level)',             (0.4, 1.0)),
        ('auprc',          'AUPRC\n(window-level)',             (0.0, 0.8)),
        ('lead_median_h',  'Median lead time (h)\nbefore AKI onset', None),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5.5))
    fig.suptitle(
        'Model performance by AKI stage — test set '
        '(95% bootstrap confidence intervals)',
        fontsize=12, y=1.01
    )

    for ax, (key, title, ylim) in zip(axes, metrics):
        vals  = [stage_results[g][key] for g in groups]
        lo_ci = [stage_cis[g][key][0]  for g in groups]
        hi_ci = [stage_cis[g][key][1]  for g in groups]
        cols  = [COLOURS[g]            for g in groups]

        # Error bars from CIs
        yerr_lo = [v - lo if not np.isnan(lo) else 0
                   for v, lo in zip(vals, lo_ci)]
        yerr_hi = [hi - v if not np.isnan(hi) else 0
                   for v, hi in zip(vals, hi_ci)]

        bars = ax.bar(x, vals, width=width,
                      color=cols, alpha=0.85, edgecolor='white',
                      linewidth=0.5)

        ax.errorbar(x, vals,
                    yerr=[yerr_lo, yerr_hi],
                    fmt='none', color='#333333', capsize=5,
                    linewidth=1.5, capthick=1.5)

        # Value labels on bars
        for xi, (v, lo, hi) in enumerate(zip(vals, lo_ci, hi_ci)):
            if np.isnan(v): continue
            label = f'{v:.0f}' if key == 'lead_median_h' else f'{v:.2f}'
            ax.text(xi, v + max(yerr_hi) * 0.15,
                    label, ha='center', va='bottom',
                    fontsize=9, fontweight='bold')
            # IQR annotation for lead time
            if key == 'lead_median_h':
                iqr25 = stage_results[g]['lead_iqr_25'] if 'lead_iqr_25' in stage_results[groups[xi]] else np.nan
                iqr75 = stage_results[g]['lead_iqr_75'] if 'lead_iqr_75' in stage_results[groups[xi]] else np.nan
                if not (np.isnan(iqr25) or np.isnan(iqr75)):
                    ax.text(xi, -max(vals)*0.05,
                            f'{iqr25:.0f}|{iqr75:.0f}',
                            ha='center', va='top',
                            fontsize=7, color='#555555')

        # Reference lines for lead time
        if key == 'lead_median_h':
            ax.axhline(24, color='#AAAAAA', lw=1.0, linestyle='--',
                       label='24h', zorder=0)
            ax.axhline(48, color='#888888', lw=1.0, linestyle=':',
                       label='48h', zorder=0)
            ax.legend(fontsize=8, framealpha=0.7)

        ax.set_title(title, fontsize=11, fontweight='bold', pad=6)
        ax.set_xticks(x)
        ax.set_xticklabels(groups, fontsize=9)
        ax.set_ylim(ylim if ylim else (0, max(v for v in vals
                                              if not np.isnan(v)) * 1.35))
        ax.grid(axis='y', alpha=0.25, linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Shared legend
    patches = [mpatches.Patch(color=COLOURS[g], label=g) for g in groups]
    fig.legend(handles=patches, loc='lower center', ncol=4,
               fontsize=10, framealpha=0.8, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout(pad=1.2)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Figure 2 → {out_path}")


# ─────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    preds = pd.read_csv(args.predictions)
    preds.columns = preds.columns.str.lower()

    # Ensure aki_stage is filled for non-AKI admissions (0)
    if 'aki_stage' not in preds.columns:
        raise ValueError("Predictions file must have an 'aki_stage' column. "
                         "Re-run evaluate.py to regenerate predictions.")

    preds['aki_stage'] = preds['aki_stage'].fillna(0).astype(int)

    threshold = args.threshold
    print(f"\nThreshold: {threshold}")
    print(f"Total windows:     {len(preds):,}")
    print(f"Total admissions:  {preds['admission_id'].nunique():,}")
    print(f"AKI admissions:    "
          f"{preds[preds['label']==1]['admission_id'].nunique():,}")
    print(f"\nStage distribution (AKI admissions):")
    aki_adm = preds[preds['label']==1].drop_duplicates('admission_id')
    print(aki_adm['aki_stage'].value_counts().sort_index().to_string())

    # ── Compute metrics and CIs per group ─────────────────
    groups = {
        'All AKI': preds[preds['admission_id'].isin(
            preds[preds['label']==1]['admission_id'].unique()
        )],
        'Stage 1': preds[preds['admission_id'].isin(
            preds[preds['aki_stage']==1]['admission_id'].unique()
        )],
        'Stage 2': preds[preds['admission_id'].isin(
            preds[preds['aki_stage']==2]['admission_id'].unique()
        )],
        'Stage 3': preds[preds['admission_id'].isin(
            preds[preds['aki_stage']==3]['admission_id'].unique()
        )],
    }

    # Include non-AKI admissions for All AKI group
    groups['All AKI'] = preds

    stage_results = {}
    stage_cis     = {}

    for name, grp in groups.items():
        print(f"\n── {name} ──")
        m = compute_metrics(grp, threshold)
        if m is None:
            print(f"  No AKI admissions in this group")
            continue
        print(f"  n_aki={m['n_aki']}  "
              f"det={m['detection_rate']:.3f}  "
              f"AUROC={m['auroc']:.3f}  "
              f"AUPRC={m['auprc']:.3f}  "
              f"lead={m['lead_median_h']:.0f}h")

        print(f"  Bootstrapping ({args.n_boot} iterations)...")
        cis = bootstrap_metrics(grp, threshold, args.n_boot, args.seed)
        for k, (lo, hi) in cis.items():
            print(f"    {k:<20} ({lo:.3f} – {hi:.3f})")

        stage_results[name] = m
        stage_cis[name]     = cis

    # ── Figure 1: lead time histograms ───────────────────
    print("\nGenerating Figure 1 — lead time histograms...")
    plot_lead_time_by_stage(
        preds     = preds,
        threshold = threshold,
        out_path  = out_dir / 'stage_lead_time.png',
    )

    # ── Figure 2: comparison bar chart ───────────────────
    print("Generating Figure 2 — stage comparison chart...")
    plot_stage_comparison(
        stage_results = stage_results,
        stage_cis     = stage_cis,
        out_path      = out_dir / 'stage_comparison.png',
    )

    print("\nDone.")


if __name__ == '__main__':
    main()
