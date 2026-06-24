"""
src/evaluate.py

Load a trained checkpoint and produce all clinical evaluation outputs.

Usage
-----
    python evaluate.py --run lstm_focal_h128_l2_attention --split test

    # Change minimum threshold if model outputs very low probabilities
    python src/evaluate.py --run lstm_focal_h128_l2 --min_threshold 0.1

Outputs (saved to run directory)
---------------------------------
    test_predictions.csv
    evaluation_report.json
    plots/score_distributions.png
    plots/threshold_analysis.png
    plots/risk_trajectory.png
    plots/lead_time.png
    plots/patient_trajectories.png
    plots/discrimination_by_horizon.png
    plots/overprediction.png
"""



import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
)
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import load_data, AKISequenceDataset, collate_fn
from models  import build_model


# ─────────────────────────────────────────────────────────
#  Argument parsing
# ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Evaluate trained AKI model')
    p.add_argument('--run',        default=None,
                   help='Run name (folder under data/models/)')
    p.add_argument('--run_dir',    default=None,
                   help='Full path to run directory')
    p.add_argument('--models_dir', default='data/models')
    p.add_argument('--windows',
                   default='data/processed/windows_matched.csv')
    p.add_argument('--split_index',
                   default='data/processed/split_index_matched.csv')
    p.add_argument('--split', default='test',
                   choices=['val', 'test'])
    p.add_argument('--min_threshold', type=float, default=0.2,
                   help='Minimum threshold to consider in sweep. '
                        'Lower this if model never reaches 0.2.')
    p.add_argument('--min_sensitivity', type=float, default=0.75,
                   help='Minimum sensitivity constraint for threshold '
                        'selection. Threshold maximises specificity '
                        'subject to sensitivity >= this value. '
                        'Falls back to Youden-J if no threshold qualifies.')
    p.add_argument('--n_boot', type=int, default=1000,
                   help='Bootstrap iterations for test set CIs '
                        '(0 = skip bootstrap).')
    return p.parse_args()


# ─────────────────────────────────────────────────────────
#  Inference
# ─────────────────────────────────────────────────────────

def collect_predictions(model, loader, device) -> pd.DataFrame:
    """Run inference — return per-step DataFrame."""
    model.eval()
    records = []
    with torch.no_grad():
        for batch in loader:
            x       = batch['x'].to(device)
            lengths = batch['lengths']
            logits  = model(x, lengths)
            probs   = torch.sigmoid(logits).cpu().numpy()
            y_np    = batch['y'].numpy()

            for b, L in enumerate(lengths):
                L      = L.item()
                adm_id = batch['admission_ids'][b]
                h2o    = batch['hours_to_onset'][b]
                hsa    = batch['hours_since_admit'][b]
                si     = batch['step_indices'][b]
                stage  = batch['aki_stages'][b]

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
                        'aki_stage':         int(stage),
                    })
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────
#  Admission-level helpers
# ─────────────────────────────────────────────────────────

def admission_level_results(preds_df:  pd.DataFrame,
                             aki_ids:  set,
                             all_ids:  list,
                             threshold: float) -> dict:
    """
    Compute admission-level TP/FP/FN/TN.

    TP : AKI admission where model fired BEFORE onset
    FN : AKI admission where model never fired before onset
    FP : non-AKI admission where model fired at any point
    TN : non-AKI admission where model never fired
    """
    tp = fp = fn = tn = 0
    for adm_id in all_ids:
        adm    = preds_df[preds_df['admission_id'] == adm_id]
        is_aki = adm_id in aki_ids
        if is_aki:
            pre   = adm[adm['hours_to_onset'].notna()]
            fired = (pre['prob'] >= threshold).any() if len(pre) > 0 else False
            if fired: tp += 1
            else:      fn += 1
        else:
            fired = (adm['prob'] >= threshold).any()
            if fired: fp += 1
            else:      tn += 1

    sens = tp / (tp + fn + 1e-9)
    spec = tn / (tn + fp + 1e-9)
    ppv  = tp / (tp + fp + 1e-9)
    npv  = tn / (tn + fn + 1e-9)
    fa_r = fp / (tp + 1e-9)
    return {
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'sensitivity': sens, 'specificity': spec,
        'ppv': ppv, 'npv': npv, 'fa_ratio': fa_r,
    }


def find_best_threshold(preds_df:        pd.DataFrame,
                         aki_ids:        set,
                         all_ids:        list,
                         min_threshold:  float = 0.2,
                         min_sensitivity: float = 0.75) -> tuple:
    """
    Sweep thresholds and select the operating point that maximises
    specificity subject to sensitivity >= min_sensitivity.

    Falls back to Youden-J if no threshold meets the sensitivity
    constraint — prevents degenerate high-threshold / zero-detection
    solutions.

    Returns (best_thresh, results_df, best_metrics)
    """
    rows = []
    for thresh in np.arange(min_threshold, 0.95, 0.01):
        m = admission_level_results(preds_df, aki_ids, all_ids, thresh)
        j = m['sensitivity'] + m['specificity'] - 1
        rows.append({
            'threshold':   round(thresh, 2),
            'sensitivity': round(m['sensitivity'], 3),
            'specificity': round(m['specificity'], 3),
            'ppv':         round(m['ppv'], 3),
            'npv':         round(m['npv'], 3),
            'fa_ratio':    round(m['fa_ratio'], 2),
            'youden_j':    round(j, 3),
            'tp': m['tp'], 'fp': m['fp'],
            'fn': m['fn'], 'tn': m['tn'],
        })

    results_df = pd.DataFrame(rows)

    # Select threshold: maximise specificity subject to min sensitivity
    valid = results_df[results_df['sensitivity'] >= min_sensitivity]

    if len(valid) == 0:
        print(f"\n  WARNING: no threshold achieves sensitivity "
              f">= {min_sensitivity} — falling back to Youden-J")
        best_idx    = results_df['youden_j'].idxmax()
        method_used = 'youden_j_fallback'
    else:
        best_idx    = valid['specificity'].idxmax()
        method_used = f'max_specificity|sens>={min_sensitivity}'

    best_thresh = float(results_df.loc[best_idx, 'threshold'])
    best_m      = admission_level_results(
        preds_df, aki_ids, all_ids, best_thresh
    )

    print(f"\n── Threshold sweep (admission-level, min={min_threshold}) ──")
    print(results_df.to_string(index=False))
    print(f"\nBest threshold ({method_used}): {best_thresh:.2f}")
    print(f"  Sensitivity: {best_m['sensitivity']:.3f}  "
          f"(constraint: >= {min_sensitivity})")
    print(f"  Specificity: {best_m['specificity']:.3f}")
    print(f"  Youden J:    {results_df.loc[best_idx,'youden_j']:.3f}")
    print(f"  PPV:         {best_m['ppv']:.3f}")
    print(f"  NPV:         {best_m['npv']:.3f}")
    print(f"  FA ratio:    {best_m['fa_ratio']:.2f}:1")

    return best_thresh, results_df, best_m


def compute_lead_times(preds_df:  pd.DataFrame,
                        aki_ids:  set,
                        threshold: float) -> pd.Series:
    """
    For each AKI admission, find hours before onset of first alert.
    Positive = model alerted N hours before onset.
    """
    leads = []
    for adm_id in aki_ids:
        adm = preds_df[
            (preds_df['admission_id'] == adm_id) &
            (preds_df['hours_to_onset'].notna())
        ].sort_values('step_index')

        alerted = adm[adm['prob'] >= threshold]
        if alerted.empty:
            leads.append(np.nan)
        else:
            leads.append(float(alerted.iloc[0]['hours_to_onset']))

    return pd.Series(leads, name='lead_time_h')


# ─────────────────────────────────────────────────────────
#  Window-level metrics
# ─────────────────────────────────────────────────────────

def window_metrics(preds_df: pd.DataFrame) -> dict:
    y     = preds_df['label'].values
    prob  = preds_df['prob'].values
    auroc = roc_auc_score(y, prob)
    auprc = average_precision_score(y, prob)
    base  = y.mean()
    print(f"\n── Window-level metrics ─────────────────────────────")
    print(f"  AUROC:           {auroc:.3f}")
    print(f"  AUPRC:           {auprc:.3f}")
    print(f"  Random baseline: {base:.3f}")
    print(f"  AUPRC/baseline:  {auprc/base:.1f}×")
    return {'auroc': auroc, 'auprc': auprc, 'baseline': base}


# ─────────────────────────────────────────────────────────
#  Bootstrap CIs (admission-level, fixed threshold)
# ─────────────────────────────────────────────────────────

def bootstrap_ci(preds_df:  pd.DataFrame,
                  aki_ids:  set,
                  all_ids:  list,
                  threshold: float,
                  n_boot:   int = 1000,
                  seed:     int = 42) -> dict:
    """
    Admission-level bootstrap CIs at a FIXED threshold.
    Resamples admissions with replacement (respects clustered
    window structure). Threshold is never recomputed — the
    val-set Youden-J value is used throughout.
    Returns dict of {metric: (lower_95, upper_95)}.
    """
    rng = np.random.default_rng(seed)

    boot = {k: [] for k in [
        'auroc', 'auprc', 'sensitivity', 'specificity',
        'ppv', 'npv', 'fa_ratio', 'detection_rate',
        'lead_median_h', 'pct_detected_24h', 'pct_detected_48h',
    ]}

    for i in range(n_boot):
        # Resample admissions with replacement
        sample_ids = rng.choice(all_ids, size=len(all_ids), replace=True)
        sample_aki = set(sid for sid in sample_ids if sid in aki_ids)
        if len(sample_aki) == 0:
            continue

        # Build boot dataframe with unique admission IDs
        frames = []
        for j, sid in enumerate(sample_ids):
            f = preds_df[preds_df['admission_id'] == sid].copy()
            f['admission_id'] = f'{sid}__{j}'
            frames.append(f)
        bd       = pd.concat(frames, ignore_index=True)
        bd_aki   = set(f'{sid}__{j}'
                       for j, sid in enumerate(sample_ids)
                       if sid in aki_ids)
        bd_ids   = list(bd['admission_id'].unique())

        try:
            # Window-level
            y, p = bd['label'].values, bd['prob'].values
            if len(np.unique(y)) > 1:
                boot['auroc'].append(float(roc_auc_score(y, p)))
                boot['auprc'].append(float(average_precision_score(y, p)))

            # Admission-level — fixed threshold
            m = admission_level_results(bd, bd_aki, bd_ids, threshold)
            boot['sensitivity'].append(m['sensitivity'])
            boot['specificity'].append(m['specificity'])
            boot['ppv'].append(m['ppv'])
            boot['npv'].append(m['npv'])
            boot['fa_ratio'].append(m['fa_ratio'])

            # Lead times
            leads_b = compute_lead_times(bd, bd_aki, threshold).dropna()
            n_aki_b = len(bd_aki)
            boot['detection_rate'].append(
                len(leads_b) / n_aki_b if n_aki_b > 0 else 0)
            if len(leads_b) > 0:
                boot['lead_median_h'].append(float(leads_b.median()))
                boot['pct_detected_24h'].append(
                    float((leads_b >= 24).sum() / n_aki_b))
                boot['pct_detected_48h'].append(
                    float((leads_b >= 48).sum() / n_aki_b))
        except Exception:
            continue

    def ci(vals):
        v = [x for x in vals if not np.isnan(x)]
        if len(v) < 10:
            return (np.nan, np.nan)
        return (float(np.percentile(v, 2.5)),
                float(np.percentile(v, 97.5)))

    return {k: ci(v) for k, v in boot.items()}



def plot_score_distributions(preds_df:  pd.DataFrame,
                              aki_ids:  set,
                              threshold: float,
                              save_path: Path):
    aki_p    = preds_df[ preds_df['admission_id'].isin(aki_ids)]['prob']
    nonaki_p = preds_df[~preds_df['admission_id'].isin(aki_ids)]['prob']

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, clip, title_suffix in zip(
        axes, [None, 0.5], ['all probabilities', 'zoomed (clipped at 0.5)']
    ):
        a = aki_p.clip(upper=clip) if clip else aki_p
        n = nonaki_p.clip(upper=clip) if clip else nonaki_p
        ax.hist(n, bins=50, alpha=0.6, color='steelblue',
                density=True, label=f'Non-AKI  mean={nonaki_p.mean():.3f}')
        ax.hist(a, bins=50, alpha=0.6, color='crimson',
                density=True, label=f'AKI  mean={aki_p.mean():.3f}')
        ax.axvline(threshold, color='black', linestyle='--', lw=1.5,
                   label=f'Threshold ({threshold:.2f})')
        ax.set_xlabel('Predicted probability')
        ax.set_ylabel('Density')
        ax.set_title(f'Score distributions — {title_suffix}\n'
                     f'Separation = {aki_p.mean()-nonaki_p.mean():.3f}')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Score distributions → {save_path.name}")


def plot_threshold_analysis(results_df:  pd.DataFrame,
                             best_thresh: float,
                             save_path:  Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    ax = axes[0]
    ax.plot(results_df['threshold'], results_df['sensitivity'],
            'o-', color='crimson',    lw=2, ms=3, label='Sensitivity')
    ax.plot(results_df['threshold'], results_df['specificity'],
            's-', color='steelblue', lw=2, ms=3, label='Specificity')
    ax.plot(results_df['threshold'], results_df['youden_j'],
            '^-', color='green',     lw=2, ms=3, label="Youden's J")
    ax.axvline(best_thresh, color='black', linestyle='--', lw=1.5,
               label=f'Best ({best_thresh:.2f})')
    ax.set_xlabel('Threshold')
    ax.set_title('Sensitivity / Specificity / Youden J\n(admission-level)')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(results_df['threshold'], results_df['ppv'],
            'o-', color='darkorange', lw=2, ms=3, label='PPV')
    ax.plot(results_df['threshold'],
            results_df['fa_ratio'].clip(upper=20),
            's-', color='gray', lw=2, ms=3,
            label='FA ratio (clipped at 20)')
    ax.axvline(best_thresh, color='black', linestyle='--', lw=1.5,
               label=f'Best ({best_thresh:.2f})')
    ax.set_xlabel('Threshold')
    ax.set_title('PPV and false alert ratio vs threshold')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Threshold analysis → {save_path.name}")


def plot_risk_trajectory(preds_df:   pd.DataFrame,
                          aki_ids:   set,
                          threshold: float,
                          save_path: Path):
    nonaki = preds_df[~preds_df['admission_id'].isin(aki_ids)]
    aki_t  = preds_df[
        preds_df['admission_id'].isin(aki_ids) &
        preds_df['hours_to_onset'].notna()
    ].copy()

    bins   = list(range(-72, 7, 6))
    labels = [f'{b}' for b in bins[:-1]]
    aki_t['bin'] = pd.cut(aki_t['hours_to_onset'],
                          bins=bins, labels=labels, right=True)
    traj = (aki_t.groupby('bin', observed=True)['prob']
            .agg(['mean','sem','count']).reset_index())

    fig, ax = plt.subplots(figsize=(11, 4))
    x = range(len(traj))
    ax.plot(x, traj['mean'], 'o-', color='crimson', lw=2, ms=5,
            label='AKI — mean risk')
    ax.fill_between(x,
        (traj['mean'] - 1.96*traj['sem']).clip(0, 1),
        (traj['mean'] + 1.96*traj['sem']).clip(0, 1),
        alpha=0.2, color='crimson', label='95% CI')
    ax.axhline(nonaki['prob'].mean(), color='steelblue',
               linestyle='--', lw=1.5,
               label=f'Non-AKI mean ({nonaki["prob"].mean():.3f})')
    ax.axhline(threshold, color='gray', linestyle=':', lw=1.5,
               label=f'Threshold ({threshold:.2f})')
    ax.set_xticks(x)
    ax.set_xticklabels(traj['bin'], rotation=45, ha='right')
    ax.set_xlabel('Hours to AKI onset')
    ax.set_ylabel('Mean predicted probability')
    ax.set_title('Predicted AKI risk vs time to onset\n'
                 '(should rise as onset approaches)')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Risk trajectory → {save_path.name}")


def plot_lead_time(lead_times: pd.Series,
                   total_aki:  int,
                   threshold:  float,
                   save_path:  Path):
    clean = lead_times.dropna()
    if len(clean) == 0:
        print("  No detected cases — skipping lead time plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.hist(clean, bins=20, color='crimson', alpha=0.75, edgecolor='white')
    ax.axvline(clean.median(), color='black', lw=2,
               label=f'Median ({clean.median():.0f}h)')
    ax.axvline(24, color='gray',  lw=1.5, linestyle='--', label='24h')
    ax.axvline(48, color='navy',  lw=1.5, linestyle='--', label='48h')
    ax.set_xlabel('Hours before AKI onset (lead time)')
    ax.set_ylabel('AKI admissions')
    ax.set_title(f'Lead time — threshold={threshold:.2f}\n'
                 f'{len(clean)}/{total_aki} AKI cases detected')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    sorted_l   = np.sort(clean)[::-1]
    cumulative = np.arange(1, len(sorted_l)+1) / total_aki * 100
    ax.plot(sorted_l, cumulative, 'o-', color='crimson', ms=3, lw=2)
    for h, ls, col in [(24,'--','gray'), (48,':','navy')]:
        pct = (clean >= h).sum() / total_aki * 100
        ax.axvline(h, color=col, linestyle=ls, lw=1.5,
                   label=f'≥{h}h: {pct:.1f}% of all AKI')
    ax.set_xlabel('Hours before AKI onset')
    ax.set_ylabel('% AKI cases detected')
    ax.set_title('Cumulative detection by lead time')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Lead time → {save_path.name}")


def plot_patient_trajectories(preds_df:  pd.DataFrame,
                               aki_ids:  set,
                               threshold: float,
                               save_path: Path,
                               n_aki:    int = 8,
                               n_nonaki: int = 4):
    nonaki_ids   = [a for a in preds_df['admission_id'].unique()
                    if a not in aki_ids]
    sample_aki   = list(aki_ids)[:n_aki]
    sample_nonaki= nonaki_ids[:n_nonaki]
    all_samples  = sample_aki + sample_nonaki

    ncols = 4
    nrows = (len(all_samples) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(16, nrows * 3.5))
    axes = axes.flatten()

    for i, adm_id in enumerate(all_samples):
        ax     = axes[i]
        adm    = preds_df[preds_df['admission_id']==adm_id]\
                 .sort_values('step_index')
        is_aki = adm_id in aki_ids

        ax.plot(adm['hours_since_admit'], adm['prob'],
                color='crimson' if is_aki else 'steelblue',
                lw=2, marker='o', ms=3)
        ax.axhline(threshold, color='gray', linestyle=':', lw=1,
                   label=f'Threshold ({threshold:.2f})')
        ax.set_ylim(0, 1)

        if is_aki:
            first_pos = adm[adm['label'] == 1]
            if not first_pos.empty:
                onset_h = first_pos.iloc[0]['hours_since_admit']
                ax.axvline(onset_h, color='crimson', linestyle='--',
                           lw=1.5, label='Prediction window opens')
            ax.set_title(f'AKI — {adm_id}', fontsize=8, color='crimson')
        else:
            ax.set_title(f'No AKI — {adm_id}', fontsize=8,
                         color='steelblue')

        ax.set_xlabel('Hours since admission', fontsize=7)
        ax.set_ylabel('AKI risk', fontsize=7)
        ax.legend(fontsize=6); ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)

    for j in range(len(all_samples), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Per-patient risk trajectories', fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Patient trajectories → {save_path.name}")


def plot_discrimination_by_horizon(preds_df:  pd.DataFrame,
                                    aki_ids:  set,
                                    save_path: Path):
    nonaki = preds_df[~preds_df['admission_id'].isin(aki_ids)]
    nonaki_sample = nonaki.sample(min(len(nonaki), 3000), random_state=42)
    aki_p  = preds_df[preds_df['admission_id'].isin(aki_ids)]
    bins   = [(0,6),(6,12),(12,18),(18,24),(24,36),(36,48),(48,72)]
    scores = []

    for lo, hi in bins:
        pos = aki_p[
            (aki_p['hours_to_onset'] >= lo) &
            (aki_p['hours_to_onset'] <  hi)
        ]
        if len(pos) < 5:
            continue
        combined = pd.concat([pos[['prob','label']],
                               nonaki_sample[['prob','label']]])
        try:
            scores.append({
                'horizon': f'{lo}-{hi}h',
                'n':       len(pos),
                'auroc':   roc_auc_score(combined['label'],
                                         combined['prob']),
                'auprc':   average_precision_score(combined['label'],
                                                   combined['prob']),
            })
        except Exception:
            pass

    if not scores:
        print("  Not enough windows for horizon plot")
        return

    df = pd.DataFrame(scores)
    print(f"\n  Discrimination by horizon:")
    print(df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(9, 4))
    x = range(len(df))
    ax.plot(x, df['auroc'], 'o-', lw=2, ms=6, label='AUROC')
    ax.plot(x, df['auprc'], 's-', lw=2, ms=6, label='AUPRC')
    ax.axhline(0.5, color='gray', linestyle='--', lw=1, label='Random')
    ax.set_xticks(x); ax.set_xticklabels(df['horizon'])
    ax.set_xlabel('Hours before AKI onset')
    ax.set_ylabel('Score')
    ax.set_title('Discrimination by prediction horizon\n'
                 '(how early can the model discriminate?)')
    ax.set_ylim(0.3, 1.05)
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Discrimination by horizon → {save_path.name}")


def evaluate_overprediction(preds_df:  pd.DataFrame,
                             aki_ids:  set,
                             threshold: float,
                             save_path: Path) -> dict:
    """
    Analyse false positive behaviour.
    Returns dict of stats for the evaluation report.
    """
    nonaki = preds_df[~preds_df['admission_id'].isin(aki_ids)].copy()
    aki    = preds_df[ preds_df['admission_id'].isin(aki_ids)].copy()

    nonaki_alarm_counts = (
        nonaki.groupby('admission_id')['prob']
        .apply(lambda x: (x >= threshold).sum())
    )

    def max_consecutive(group):
        fired   = (group.sort_values('step_index')['prob'] >= threshold)
        max_run = run = 0
        for v in fired:
            run     = run + 1 if v else 0
            max_run = max(max_run, run)
        return max_run

    nonaki_consecutive = nonaki.groupby('admission_id').apply(max_consecutive)

    nonaki['hour_bin'] = ((nonaki['hours_since_admit'] // 6) * 6).astype(int)
    fp_by_hour = (
        nonaki.groupby('hour_bin')['prob']
        .agg(mean='mean',
             sem=lambda x: x.std() / max(len(x)**0.5, 1))
        .reset_index()
    )

    n_nonaki      = nonaki['admission_id'].nunique()
    never_alarmed = (nonaki_alarm_counts == 0).sum()
    persistent    = (nonaki_consecutive >= 4).sum()
    separation    = aki['prob'].mean() - nonaki['prob'].mean()

    print(f"\n── Overprediction analysis (threshold={threshold:.2f}) ──")
    print(f"  Non-AKI admissions:        {n_nonaki}")
    print(f"  Never alarmed:             "
          f"{never_alarmed} ({never_alarmed/n_nonaki*100:.1f}%)")
    print(f"  Alarmed ≥1 window:         "
          f"{n_nonaki-never_alarmed} "
          f"({(n_nonaki-never_alarmed)/n_nonaki*100:.1f}%)")
    print(f"  Persistent ≥24h alarm:     "
          f"{persistent} ({persistent/n_nonaki*100:.1f}%)")
    print(f"  Mean alarms/non-AKI adm:   "
          f"{nonaki_alarm_counts.mean():.1f}")
    print(f"  Score separation:          {separation:.3f}  "
          f"(>0.10 is meaningful)")
    print(f"  Non-AKI mean prob:         {nonaki['prob'].mean():.3f}")
    print(f"  AKI mean prob:             {aki['prob'].mean():.3f}")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle('Overprediction / False Positive Analysis', fontsize=13)

    # Panel 1: Score distributions
    ax = axes[0, 0]
    ax.hist(nonaki['prob'], bins=50, alpha=0.6, color='steelblue',
            density=True, label=f'Non-AKI ({len(nonaki):,})')
    ax.hist(aki['prob'],    bins=50, alpha=0.6, color='crimson',
            density=True, label=f'AKI ({len(aki):,})')
    ax.axvline(threshold, color='black', linestyle='--', lw=1.5,
               label=f'Threshold ({threshold:.2f})')
    ax.set_xlabel('Predicted probability')
    ax.set_ylabel('Density')
    ax.set_title(f'Score distributions\nSeparation={separation:.3f}')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Panel 2: Alarm count per non-AKI admission
    ax = axes[0, 1]
    nonaki_alarm_counts.hist(bins=20, ax=ax, color='steelblue',
                              alpha=0.8, edgecolor='white')
    ax.axvline(nonaki_alarm_counts.median(), color='black', lw=2,
               label=f'Median ({nonaki_alarm_counts.median():.0f})')
    ax.set_xlabel('Alarm windows per admission')
    ax.set_ylabel('Non-AKI admissions')
    ax.set_title('Alarm count — non-AKI\n(0 = never falsely alarmed)')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Panel 3: Consecutive alarms
    ax = axes[0, 2]
    nonaki_consecutive.hist(bins=15, ax=ax, color='steelblue',
                             alpha=0.8, edgecolor='white')
    ax.axvline(4, color='crimson', linestyle='--', lw=1.5,
               label='24h (4 steps)')
    ax.set_xlabel('Max consecutive alarm steps (×6h)')
    ax.set_ylabel('Non-AKI admissions')
    ax.set_title(f'Persistent false alarms\n'
                 f'≥24h: {persistent}/{n_nonaki}')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Panel 4: Non-AKI risk over admission timeline
    ax = axes[1, 0]
    if len(fp_by_hour) > 0:
        ax.plot(fp_by_hour['hour_bin'], fp_by_hour['mean'],
                'o-', color='steelblue', lw=2, ms=4)
        ax.fill_between(
            fp_by_hour['hour_bin'],
            (fp_by_hour['mean'] - 1.96*fp_by_hour['sem']).clip(0),
            (fp_by_hour['mean'] + 1.96*fp_by_hour['sem']).clip(1),
            alpha=0.2, color='steelblue'
        )
    ax.axhline(threshold, color='gray', linestyle='--', lw=1.5,
               label=f'Threshold ({threshold:.2f})')
    ax.set_xlabel('Hours since admission')
    ax.set_ylabel('Mean predicted risk')
    ax.set_title('Non-AKI risk over admission timeline\n'
                 '(should stay flat and below threshold)')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Panel 5: AKI trajectory vs non-AKI baseline
    ax = axes[1, 1]
    aki_timing = aki[aki['hours_to_onset'].notna()].copy()
    if len(aki_timing) > 0:
        bins_  = list(range(-72, 7, 6))
        lbls   = [f'{b}' for b in bins_[:-1]]
        aki_timing['bin'] = pd.cut(
            aki_timing['hours_to_onset'],
            bins=bins_, labels=lbls, right=True
        )
        traj = (aki_timing.groupby('bin', observed=True)['prob']
                .mean().reset_index())
        ax.plot(range(len(traj)), traj['prob'], 'o-',
                color='crimson', lw=2, ms=4, label='AKI (pre-onset)')
        ax.set_xticks(range(len(traj)))
        ax.set_xticklabels(traj['bin'], rotation=45, ha='right',
                           fontsize=7)
    ax.axhline(nonaki['prob'].mean(), color='steelblue',
               linestyle='--', lw=1.5,
               label=f'Non-AKI mean ({nonaki["prob"].mean():.3f})')
    ax.axhline(threshold, color='gray', linestyle=':', lw=1.5,
               label=f'Threshold ({threshold:.2f})')
    ax.set_ylabel('Mean predicted probability')
    ax.set_title('AKI trajectory vs non-AKI baseline')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Panel 6: Admission-level precision-recall
    ax = axes[1, 2]
    all_ids = list(preds_df['admission_id'].unique())
    precs, recs = [], []
    for t in np.arange(0.05, 0.90, 0.03):
        m = admission_level_results(preds_df, aki_ids, all_ids, t)
        if m['tp'] + m['fp'] > 0 and m['tp'] + m['fn'] > 0:
            precs.append(m['ppv'])
            recs.append(m['sensitivity'])
    ax.plot(recs, precs, 'o-', color='crimson', lw=2, ms=3,
            label='Admission PR curve')
    baseline = len(aki_ids) / len(all_ids)
    ax.axhline(baseline, color='gray', linestyle='--', lw=1,
               label=f'Random ({baseline:.3f})')
    ax.set_xlabel('Detection rate (sensitivity)')
    ax.set_ylabel('Precision (PPV)')
    ax.set_title('Admission-level precision-recall')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Overprediction analysis → {save_path.name}")

    return {
        'nonaki_never_alarmed_pct': float(never_alarmed / n_nonaki),
        'nonaki_persistent_fp_pct': float(persistent / n_nonaki),
        'score_separation':         float(separation),
        'nonaki_mean_prob':         float(nonaki['prob'].mean()),
        'aki_mean_prob':            float(aki['prob'].mean()),
    }


# ─────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Resolve run directory ─────────────────────────────
    if args.run_dir:
        run_dir = Path(args.run_dir)
    elif args.run:
        run_dir = Path(args.models_dir) / args.run
    else:
        raise ValueError("Provide --run NAME or --run_dir PATH")

    if not run_dir.exists():
        raise FileNotFoundError(f"Run not found: {run_dir}")

    print(f"\nEvaluating: {run_dir.name}  |  split: {args.split}")

    with open(run_dir / 'config.json') as f:
        cfg = json.load(f)
    print(f"Model: {cfg['model']}  |  Loss: {cfg['loss']}")

    # ── Load data ─────────────────────────────────────────
    windows, feat_cols, _ = load_data(
        windows_path     = args.windows,
        split_index_path = args.split_index,
        max_seq_len      = cfg.get('max_seq_len', 100),
        forward_fill     = True,
        models_dir       = str(run_dir),
        fit_scaler       = False,
    )

    eval_ds     = AKISequenceDataset(
        windows[windows['split'] == args.split], feat_cols
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=32, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    # ── Load model ────────────────────────────────────────
    ckpt  = torch.load(run_dir / 'best_model.pt', map_location=DEVICE)
    model = build_model(
        architecture = cfg['model'],
        input_size   = len(feat_cols),
        hidden_size  = cfg.get('hidden', 128),
        n_layers     = cfg.get('layers', 2),
        dropout      = cfg.get('dropout', 0.3),
        n_heads      = cfg.get('heads', 4),
        attention = cfg.get('attention', False)
    ).to(DEVICE)
    model.load_state_dict(ckpt['model_state'])
    print(f"Loaded epoch {ckpt['epoch']} "
          f"(val AUPRC={ckpt['val_auprc']:.4f})")

    # ── Inference ─────────────────────────────────────────
    print(f"\nRunning inference...")
    preds_df = collect_predictions(model, eval_loader, DEVICE)
    preds_df.to_csv(
        run_dir / f'{args.split}_predictions.csv', index=False
    )

    aki_ids = set(
        preds_df[preds_df['label'] == 1]['admission_id'].unique()
    )
    all_ids = list(preds_df['admission_id'].unique())
    print(f"  {len(preds_df):,} windows | "
          f"AKI: {len(aki_ids)} | "
          f"Non-AKI: {len(all_ids)-len(aki_ids)}")

    # ── Window metrics ────────────────────────────────────
    wm = window_metrics(preds_df)

    # ── Threshold: compute from val, load for test ────────
    val_thresh_path = run_dir / 'val_threshold.json'

    if args.split == 'val':
        # Compute threshold from val predictions and save
        best_thresh, thresh_df, best_m = find_best_threshold(
            preds_df, aki_ids, all_ids,
            min_threshold   = args.min_threshold,
            min_sensitivity = args.min_sensitivity,
        )
        with open(val_thresh_path, 'w') as f:
            json.dump({
                'threshold':     best_thresh,
                'min_sensitivity': args.min_sensitivity,
            }, f, indent=2)
        print(f"\n  Val threshold saved → {val_thresh_path}")

    else:
        # Load threshold determined on val set
        if not val_thresh_path.exists():
            raise FileNotFoundError(
                f"\nval_threshold.json not found in {run_dir}.\n"
                f"Run --split val first to determine the threshold."
            )
        with open(val_thresh_path) as f:
            best_thresh = float(json.load(f)['threshold'])
        print(f"\n  Using val-set threshold: {best_thresh:.4f}")

        # Run threshold sweep for plots only (not for metric selection)
        _, thresh_df, best_m = find_best_threshold(
            preds_df, aki_ids, all_ids,
            min_threshold   = args.min_threshold,
            min_sensitivity = args.min_sensitivity,
        )
        # Override metrics with the fixed val threshold
        best_m = admission_level_results(
            preds_df, aki_ids, all_ids, best_thresh
        )
        print(f"\n── Metrics at val-set threshold ({best_thresh:.4f}) ──")
        print(f"  Sensitivity: {best_m['sensitivity']:.3f}")
        print(f"  Specificity: {best_m['specificity']:.3f}")
        print(f"  PPV:         {best_m['ppv']:.3f}")
        print(f"  NPV:         {best_m['npv']:.3f}")
        print(f"  FA ratio:    {best_m['fa_ratio']:.2f}:1")

    # ── Lead time at best threshold ───────────────────────
    leads      = compute_lead_times(preds_df, aki_ids, best_thresh)
    lead_clean = leads.dropna()
    total_aki  = len(aki_ids)
    detected   = len(lead_clean)

    print(f"\n── Lead time (threshold={best_thresh:.2f}) ──────────────")
    print(f"  Detected: {detected}/{total_aki} "
          f"({detected/total_aki*100:.1f}%)")
    if len(lead_clean) > 0:
        print(f"  Median:  {lead_clean.median():.0f}h  "
              f"IQR {lead_clean.quantile(0.25):.0f}–"
              f"{lead_clean.quantile(0.75):.0f}h")
        for h in [24, 48, 72]:
            n   = (lead_clean >= h).sum()
            pct = n / total_aki * 100
            print(f"  ≥{h:2d}h:  {n}/{total_aki} ({pct:.1f}%)")

    # ── Lead time sanity check ────────────────────────────
    # If first alert is always at step 0-1, lead time is just
    # "time from admission to onset" not genuine early prediction
    first_alert_step = (
        preds_df[preds_df['admission_id'].isin(aki_ids)]
        .groupby('admission_id')
        .apply(lambda g:
               g[g['prob'] >= best_thresh]['step_index'].min()
               if (g['prob'] >= best_thresh).any() else np.nan)
    )
    median_first_step = first_alert_step.median()
    print(f"\n  First alert step — median={median_first_step:.1f} "
          f"(0=first window of admission)")
    if median_first_step <= 1:
        print(f"  WARNING: model is alerting at admission start — "
              f"lead time reflects time-to-onset not early prediction")
    else:
        print(f"  OK — model fires partway through admission "
              f"(genuine temporal signal)")

    # ── Plots ─────────────────────────────────────────────
    plots_dir = run_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    print(f"\nGenerating plots...")

    plot_score_distributions(
        preds_df, aki_ids, best_thresh,
        plots_dir / 'score_distributions.png'
    )
    plot_threshold_analysis(
        thresh_df, best_thresh,
        plots_dir / 'threshold_analysis.png'
    )
    plot_risk_trajectory(
        preds_df, aki_ids, best_thresh,
        plots_dir / 'risk_trajectory.png'
    )
    plot_lead_time(
        leads, total_aki, best_thresh,
        plots_dir / 'lead_time.png'
    )
    plot_patient_trajectories(
        preds_df, aki_ids, best_thresh,
        plots_dir / 'patient_trajectories.png'
    )
    plot_discrimination_by_horizon(
        preds_df, aki_ids,
        plots_dir / 'discrimination_by_horizon.png'
    )
    overpredict_stats = evaluate_overprediction(
        preds_df, aki_ids, best_thresh,
        plots_dir / 'overprediction.png'
    )

    # ── Save report ───────────────────────────────────────
    best_j = float(
        thresh_df.loc[
            thresh_df['threshold'].round(2) == round(best_thresh, 2),
            'youden_j'
        ].values[0]
    ) if round(best_thresh, 2) in thresh_df['threshold'].round(2).values else None

    report = {
        'run':                      run_dir.name,
        'split':                    args.split,
        'model':                    cfg['model'],
        'loss':                     cfg['loss'],
        # Window-level
        'window_auroc':             round(wm['auroc'], 4),
        'window_auprc':             round(wm['auprc'], 4),
        'auprc_vs_random':          round(wm['auprc'] / wm['baseline'], 2),
        # Threshold
        'best_threshold':           best_thresh,
        'youden_j':                 best_j,
        # Admission-level
        'adm_sensitivity':          round(best_m['sensitivity'], 4),
        'adm_specificity':          round(best_m['specificity'], 4),
        'adm_ppv':                  round(best_m['ppv'], 4),
        'adm_npv':                  round(best_m['npv'], 4),
        'adm_fa_ratio':             round(best_m['fa_ratio'], 2),
        # Lead time
        'detected_pct':             round(detected / total_aki, 4),
        'lead_median_h':            float(lead_clean.median())
                                    if len(lead_clean) > 0 else None,
        'lead_iqr_25':              float(lead_clean.quantile(0.25))
                                    if len(lead_clean) > 0 else None,
        'lead_iqr_75':              float(lead_clean.quantile(0.75))
                                    if len(lead_clean) > 0 else None,
        'pct_detected_24h':         float((lead_clean >= 24).sum() / total_aki)
                                    if len(lead_clean) > 0 else None,
        'pct_detected_48h':         float((lead_clean >= 48).sum() / total_aki)
                                    if len(lead_clean) > 0 else None,
        'first_alert_median_step':  float(median_first_step),
        # Overprediction
        **overpredict_stats,
    }

    report_path = run_dir / f'evaluation_report_{args.split}.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport → {report_path}")

    # ── Bootstrap CIs (test set only) ────────────────────
    if args.split == 'test' and args.n_boot > 0:
        print(f"\n── Bootstrap CIs (n={args.n_boot}, "
              f"fixed threshold={best_thresh:.4f}) ──")
        cis = bootstrap_ci(
            preds_df, aki_ids, all_ids,
            threshold = best_thresh,
            n_boot    = args.n_boot,
        )

        # Print CI summary
        metrics_to_show = [
            ('auroc',            'AUROC'),
            ('auprc',            'AUPRC'),
            ('sensitivity',      'Sensitivity'),
            ('specificity',      'Specificity'),
            ('ppv',              'PPV'),
            ('fa_ratio',         'FA ratio'),
            ('detection_rate',   'Detection rate'),
            ('lead_median_h',    'Lead median (h)'),
            ('pct_detected_24h', '≥24h detected'),
            ('pct_detected_48h', '≥48h detected'),
        ]
        print(f"\n  {'Metric':<22} {'Point':>8}  {'95% CI'}")
        print(f"  {'-'*52}")
        for key, label in metrics_to_show:
            pt  = report.get(f'adm_{key}',
                  report.get(f'window_{key}',
                  report.get(key)))
            if pt is None:
                continue
            lo, hi = cis.get(key, (np.nan, np.nan))
            if np.isnan(lo):
                ci_str = '(insufficient data)'
            else:
                ci_str = f'({lo:.3f} – {hi:.3f})'
            print(f"  {label:<22} {pt:>8.3f}  {ci_str}")

        # Save bootstrap results
        ci_report = {
            'threshold':  best_thresh,
            'n_boot':     args.n_boot,
            'split':      args.split,
        }
        for key, label in metrics_to_show:
            lo, hi = cis.get(key, (np.nan, np.nan))
            ci_report[f'{key}_ci_lower'] = float(lo) if not np.isnan(lo) else None
            ci_report[f'{key}_ci_upper'] = float(hi) if not np.isnan(hi) else None

        ci_path = run_dir / 'bootstrap_ci.json'
        with open(ci_path, 'w') as f:
            json.dump(ci_report, f, indent=2)
        print(f"\n  Bootstrap CIs → {ci_path}")

    # ── Final summary ─────────────────────────────────────
    print(f"\n{'='*52}")
    print(f"SUMMARY — {run_dir.name}")
    print(f"{'='*52}")
    print(f"  AUROC:                {report['window_auroc']:.3f}")
    print(f"  AUPRC:                {report['window_auprc']:.3f}  "
          f"({report['auprc_vs_random']:.1f}× random)")
    print(f"  Best threshold:       {best_thresh:.2f}  "
          f"(Youden J={best_j})")
    print(f"  Sensitivity:          {report['adm_sensitivity']:.3f}")
    print(f"  Specificity:          {report['adm_specificity']:.3f}")
    print(f"  Score separation:     {report['score_separation']:.3f}")
    if report['lead_median_h']:
        print(f"  Lead time median:     {report['lead_median_h']:.0f}h")
        print(f"  Detected ≥24h early:  "
              f"{report['pct_detected_24h']*100:.1f}%")
    print(f"  Non-AKI never alarmed:{report['nonaki_never_alarmed_pct']*100:.1f}%")
    print(f"  Persistent FP ≥24h:   {report['nonaki_persistent_fp_pct']*100:.1f}%")


if __name__ == '__main__':
    main()