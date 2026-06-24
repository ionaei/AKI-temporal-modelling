"""
explain_ig.py

Integrated Gradients explainability for the LSTM + attention AKI model.

Produces four outputs:
  1. ig_feature_importance.csv      — mean |IG| per feature (AKI, non-AKI, diff)
  2. ig_directional_importance.csv  — mean signed IG for AKI admissions
  3. plots/ig_differential.png      — main paper: differential importance
  4. plots/ig_heatmap_{adm_id}.png  — temporal heatmap for selected admissions

Requires: pip install captum

Usage
-----
    python gradient_explain.py --run transformer_bce --models_dir data/models --windows data/processed/windows_matched.csv --split_index data/processed/split_index_matched.csv --split test --n_heatmaps 6

    # PIC
    python explain_ig.py \\
        --run         pic_lstm_bce_attention \\
        --models_dir  pic_models \\
        --windows     pic_processed/windows_matched_pic.csv \\
        --split_index pic_processed/split_index_pic.csv \\
        --split       test
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore')

try:
    from captum.attr import IntegratedGradients, NoiseTunnel
    CAPTUM_AVAILABLE = True
except ImportError:
    print("WARNING: captum not installed. Run: pip install captum")
    CAPTUM_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from dataset import load_data, AKISequenceDataset, collate_fn
from models  import build_model


# ─────────────────────────────────────────────────────────
#  Argument parsing
# ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Integrated Gradients explainability for LSTM+attention'
    )
    p.add_argument('--run',         required=True)
    p.add_argument('--models_dir',  default='data/models')
    p.add_argument('--windows',     default='data/processed/windows_matched.csv')
    p.add_argument('--split_index', default='data/processed/split_index_matched.csv')
    p.add_argument('--split',       default='test',
                   choices=['val','test'])
    p.add_argument('--n_steps',     type=int, default=50,
                   help='IG integration steps — higher = more accurate, slower')
    p.add_argument('--n_heatmaps',  type=int, default=6,
                   help='Number of per-admission heatmaps to generate')
    p.add_argument('--top_n',       type=int, default=25,
                   help='Top N features for importance plots')
    p.add_argument('--batch_size',  type=int, default=1,
                   help='Keep at 1 for IG — attribution per admission')
    p.add_argument('--max_admissions', type=int, default=None,
                   help='Cap number of admissions for speed (None = all)')
    return p.parse_args()


# ─────────────────────────────────────────────────────────
#  Model wrapper for Captum
# ─────────────────────────────────────────────────────────

class LSTMWrapper(nn.Module):
    """
    Captum requires a model that takes only the tensor input.
    We wrap the LSTM to accept (x, lengths) via closure.
    """
    def __init__(self, model, lengths):
        super().__init__()
        self.model   = model
        self.lengths = lengths

    def forward(self, x):
        out = self.model(x, self.lengths)
        logits = out[0] if isinstance(out, tuple) else out
        # Return sigmoid probabilities — shape (B, T)
        return torch.sigmoid(logits)


# ─────────────────────────────────────────────────────────
#  Integrated Gradients per admission
# ─────────────────────────────────────────────────────────

def compute_ig_admission(model:    nn.Module,
                          x:       torch.Tensor,
                          lengths: torch.Tensor,
                          target_step: int,
                          n_steps: int,
                          device:  torch.device) -> np.ndarray:
    """
    Compute IG attributions for one admission at a specific time step.

    Parameters
    ----------
    model       : LSTM model
    x           : (1, T, F) input tensor
    lengths     : (1,) sequence length
    target_step : which time step to attribute (usually last pre-onset step)
    n_steps     : number of IG integration steps

    Returns
    -------
    attributions : (T, F) numpy array — signed attribution per step per feature
    """
    x = x.to(device)

    wrapper = LSTMWrapper(model, lengths.to(device))
    ig      = IntegratedGradients(wrapper)

    # Baseline: all zeros (represents absence of all features)
    baseline = torch.zeros_like(x)

    # Target: the probability at target_step
    # Captum target selects the output index to attribute
    target = int(target_step)

    with torch.no_grad():
        # Check prediction at target step
        out    = model(x, lengths.to(device))
        logits = out[0] if isinstance(out, tuple) else out
        prob   = torch.sigmoid(logits[0, target_step]).item()

    attr, delta = ig.attribute(
        x,
        baselines                = baseline,
        target                   = target,
        n_steps                  = n_steps,
        return_convergence_delta = True,
        internal_batch_size      = 1,
    )
    # attr shape: (1, T, F) → squeeze to (T, F)
    return attr.squeeze(0).detach().cpu().numpy(), prob, float(delta.abs().mean())


# ─────────────────────────────────────────────────────────
#  Main analysis loop
# ─────────────────────────────────────────────────────────

def run_ig_analysis(model:       nn.Module,
                     dataset:    AKISequenceDataset,
                     feat_cols:  list,
                     device:     torch.device,
                     n_steps:    int,
                     max_admissions: int = None) -> dict:
    """
    Run IG attribution for all admissions in dataset.

    For AKI admissions: attribute at the last pre-onset positive window.
    For non-AKI admissions: attribute at the last window.

    Returns dict with per-admission results.
    """
    results = {
        'aki':    [],   # list of (T, F) attribution arrays — AKI admissions
        'nonaki': [],   # non-AKI admissions
        'aki_ids':    [],
        'nonaki_ids': [],
        'convergence_deltas': [],
        'probs': [],
    }

    n_processed = 0
    model.eval()

    for item in dataset:
        if max_admissions and n_processed >= max_admissions:
            break

        x       = item['x'].unsqueeze(0)          # (1, T, F)
        lengths = torch.tensor([item['length']])
        y       = item['y'].numpy()
        adm_id  = item['admission_id']
        h2o     = item['hours_to_onset']

        is_aki = (y == 1).any()

        if is_aki:
            # Find last positive window (just before onset)
            pos_steps = np.where(y == 1)[0]
            # Use the step with minimum hours_to_onset > 0
            valid = [(i, h2o[i]) for i in pos_steps
                     if i < len(h2o) and not np.isnan(h2o[i]) and h2o[i] >= 0]
            if not valid:
                target_step = int(pos_steps[0])
            else:
                target_step = min(valid, key=lambda x: x[1])[0]
        else:
            # Last window of admission
            target_step = item['length'] - 1

        try:
            attr, prob, delta = compute_ig_admission(
                model, x, lengths, target_step, n_steps, device
            )
        except Exception as e:
            print(f"  IG failed for {adm_id}: {e}")
            continue

        if is_aki:
            results['aki'].append(attr)
            results['aki_ids'].append(adm_id)
        else:
            results['nonaki'].append(attr)
            results['nonaki_ids'].append(adm_id)

        results['convergence_deltas'].append(delta)
        results['probs'].append(prob)
        n_processed += 1

        if n_processed % 20 == 0:
            print(f"  Processed {n_processed} admissions...")

    print(f"\n  Total: {n_processed} admissions "
          f"({len(results['aki'])} AKI, {len(results['nonaki'])} non-AKI)")
    print(f"  Mean convergence delta: "
          f"{np.mean(results['convergence_deltas']):.4f} "
          f"(< 0.05 is good)")

    return results


# ─────────────────────────────────────────────────────────
#  Aggregate attributions to feature importance
# ─────────────────────────────────────────────────────────

def aggregate_importance(attr_list: list,
                          feat_cols: list) -> pd.DataFrame:
    """
    Aggregate per-admission (T, F) attribution arrays to
    per-feature importance scores.

    Three metrics per feature:
      mean_abs  : mean |attribution| summed over T — magnitude
      mean_signed: mean attribution summed over T — direction
      pct_positive: % time steps where attribution > 0
    """
    if not attr_list:
        return pd.DataFrame()

    # Sum over time steps for each admission → (F,) per admission
    # Then mean over admissions → (F,)
    abs_sum    = np.mean([np.abs(a).sum(axis=0) for a in attr_list], axis=0)
    signed_sum = np.mean([a.sum(axis=0) for a in attr_list], axis=0)
    pct_pos    = np.mean([
        (a > 0).mean(axis=0) for a in attr_list
    ], axis=0)

    df = pd.DataFrame({
        'feature':      feat_cols,
        'mean_abs':     abs_sum,
        'mean_signed':  signed_sum,
        'pct_positive': pct_pos,
    }).sort_values('mean_abs', ascending=False)

    return df


# ─────────────────────────────────────────────────────────
#  Plots
# ─────────────────────────────────────────────────────────

def clean_label(f):
    return (f.replace('_missing_flag', ' (missing)')
             .replace('_missing',      ' (missing)')
             .replace('_mean', ' mean').replace('_last', ' last')
             .replace('_max', ' max').replace('_min', ' min')
             .replace('_trend', ' trend').replace('_alert', ' alert')
             .replace('_n', ' n').replace('nephrotoxin_', '')
             .replace('drug_class_', '').replace('proc_', '')
             .replace('diag_', 'ICD ').replace('_', ' ')
             .strip().capitalize())


def plot_differential_ig(aki_df:   pd.DataFrame,
                          naki_df:  pd.DataFrame,
                          feat_cols: list,
                          save_path: Path,
                          top_n:    int = 25):
    """
    Differential importance: AKI mean_abs minus non-AKI mean_abs.
    Red = more important for AKI, blue = more important for non-AKI.
    """
    merged = aki_df[['feature','mean_abs']].rename(
        columns={'mean_abs': 'aki_imp'}
    ).merge(
        naki_df[['feature','mean_abs']].rename(
            columns={'mean_abs': 'naki_imp'}
        ),
        on='feature', how='outer'
    ).fillna(0)

    merged['difference'] = merged['aki_imp'] - merged['naki_imp']
    merged = merged.sort_values('difference', ascending=False)

    pos = merged[merged['difference'] > 0].head(top_n)
    neg = merged[merged['difference'] < 0].tail(5)
    df  = pd.concat([pos, neg]).sort_values('difference').reset_index(drop=True)

    bar_colours = ['#C0392B' if v > 0 else '#2E75B6'
                   for v in df['difference']]
    labels = [clean_label(f) for f in df['feature']]
    n_rows = len(df)

    fig, ax = plt.subplots(figsize=(7.5, n_rows * 0.27 + 1.8))
    ax.barh(range(n_rows), df['difference'],
            height=0.6, color=bar_colours, alpha=0.85,
            edgecolor='white', linewidth=0.3)
    ax.axvline(0, color='#333333', linewidth=0.7)
    ax.axhline(4.5, color='#aaaaaa', linewidth=0.6, linestyle='--')

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(labels, fontsize=8)
    ax.tick_params(axis='y', pad=2)
    ax.set_xlabel('Mean |IG| difference  (AKI − non-AKI)', fontsize=9)
    ax.set_title('Integrated Gradients — discriminative feature importance\n'
                 'Positive = higher attribution in AKI admissions',
                 fontsize=10, pad=8)
    ax.grid(axis='x', alpha=0.2, linewidth=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout(pad=0.8)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Differential IG plot → {save_path}")

    return merged


def plot_directional(aki_df: pd.DataFrame,
                      save_path: Path,
                      top_n: int = 20):
    """
    Signed IG for AKI admissions.
    Red = feature value pushes towards AKI prediction.
    Blue = feature value pushes away from AKI prediction.
    """
    df = aki_df.sort_values('mean_signed', ascending=False).head(top_n)
    df = pd.concat([
        aki_df.nlargest(top_n // 2, 'mean_signed'),
        aki_df.nsmallest(5, 'mean_signed'),
    ]).sort_values('mean_signed').reset_index(drop=True)

    bar_colours = ['#C0392B' if v > 0 else '#2E75B6'
                   for v in df['mean_signed']]
    labels = [clean_label(f) for f in df['feature']]

    fig, ax = plt.subplots(figsize=(7.5, len(df) * 0.28 + 1.8))
    ax.barh(range(len(df)), df['mean_signed'],
            height=0.6, color=bar_colours, alpha=0.85,
            edgecolor='white', linewidth=0.3)
    ax.axvline(0, color='#333333', linewidth=0.7)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Mean signed IG attribution (AKI admissions)', fontsize=9)
    ax.set_title('Directional IG — AKI admissions\n'
                 'Positive = high feature value pushes towards AKI prediction',
                 fontsize=10, pad=8)
    ax.grid(axis='x', alpha=0.2, linewidth=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout(pad=0.8)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Directional IG plot → {save_path}")


def plot_temporal_heatmap(attr:      np.ndarray,
                           feat_cols: list,
                           adm_id:    str,
                           hours:     np.ndarray,
                           y:         np.ndarray,
                           save_path: Path,
                           top_n:     int = 20):
    """
    Temporal heatmap for one admission.
    Rows = top features, columns = time steps.
    Colour = signed IG attribution (red=positive, blue=negative).
    """
    # Select top N features by mean |attr| over time for this admission
    mean_abs   = np.abs(attr).mean(axis=0)
    top_idx    = np.argsort(mean_abs)[::-1][:top_n]
    attr_sub   = attr[:, top_idx].T   # (top_n, T)
    labels_sub = [clean_label(feat_cols[i]) for i in top_idx]
    T          = attr.shape[0]

    fig, axes = plt.subplots(
        2, 1, figsize=(max(10, T * 0.35), top_n * 0.28 + 1.5),
        gridspec_kw={'height_ratios': [1, top_n], 'hspace': 0.05}
    )

    # Top panel: AKI label and probability would go here
    # (simple bar showing which steps are positive)
    ax = axes[0]
    colours = ['#C0392B' if yi == 1 else '#DDDDDD' for yi in y]
    ax.bar(range(T), [1]*T, color=colours, width=0.9, alpha=0.8)
    ax.set_xlim(-0.5, T - 0.5)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_ylabel('AKI\nwindow', fontsize=7, rotation=0, ha='right', va='center')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Heatmap
    ax = axes[1]
    vmax = np.abs(attr_sub).max()
    vmax = max(vmax, 1e-6)

    im = ax.imshow(
        attr_sub,
        aspect='auto',
        cmap='RdBu_r',
        vmin=-vmax, vmax=vmax,
        interpolation='nearest',
    )
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(labels_sub, fontsize=7)

    # X axis: hours since admission
    if hours is not None and len(hours) == T:
        tick_every = max(1, T // 10)
        tick_pos   = range(0, T, tick_every)
        ax.set_xticks(list(tick_pos))
        ax.set_xticklabels(
            [f'{hours[i]:.0f}h' for i in tick_pos],
            fontsize=7, rotation=45, ha='right'
        )
    ax.set_xlabel('Hours since admission', fontsize=8)

    plt.colorbar(im, ax=ax, fraction=0.015, pad=0.01,
                 label='IG attribution (red=↑AKI risk)')

    fig.suptitle(f'Temporal IG attribution — admission {adm_id}',
                 fontsize=10, y=1.01)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Heatmap → {save_path.name}")


# ─────────────────────────────────────────────────────────
#  Comparison with attention-weighted importance
# ─────────────────────────────────────────────────────────

def compare_with_attention(ig_df:      pd.DataFrame,
                             attn_path:  Path,
                             save_path:  Path,
                             top_n:      int = 20):
    """
    If attention-weighted importance CSV exists, compare rank
    correlation with IG importance.
    """
    if not attn_path.exists():
        print(f"  Attention importance file not found: {attn_path}")
        return

    attn = pd.read_csv(attn_path)
    if 'feature' not in attn.columns or 'importance' not in attn.columns:
        print(f"  Unexpected columns in attention file: {attn.columns.tolist()}")
        return

    merged = ig_df[['feature','mean_abs']].merge(
        attn[['feature','importance']],
        on='feature', how='inner'
    )
    if len(merged) < 5:
        print(f"  Too few matched features for comparison ({len(merged)})")
        return

    from scipy.stats import spearmanr
    rho, p = spearmanr(merged['mean_abs'], merged['importance'])
    print(f"\n  IG vs attention-weighted importance:")
    print(f"  Spearman ρ = {rho:.3f}  p = {p:.3e}")
    print(f"  (ρ > 0.7 indicates methods broadly agree)")

    # Scatter plot
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(merged['importance'], merged['mean_abs'],
               alpha=0.5, s=20, color='#2E75B6')
    ax.set_xlabel('Attention-weighted importance', fontsize=10)
    ax.set_ylabel('Integrated Gradients |attribution|', fontsize=10)
    ax.set_title(f'IG vs attention-weighted importance\n'
                 f'Spearman ρ={rho:.3f}  p={p:.2e}',
                 fontsize=10)
    ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Label top features
    top   = merged.nlargest(8, 'mean_abs')
    for _, row in top.iterrows():
        ax.annotate(clean_label(row['feature'])[:20],
                    (row['importance'], row['mean_abs']),
                    fontsize=6, alpha=0.8,
                    xytext=(4, 2), textcoords='offset points')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Comparison scatter → {save_path}")


# ─────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    run_dir = Path(args.models_dir) / args.run
    out_dir = run_dir / 'ig_explainability'
    plots   = out_dir / 'plots'
    out_dir.mkdir(exist_ok=True)
    plots.mkdir(exist_ok=True)

    if not CAPTUM_AVAILABLE:
        print("ERROR: pip install captum")
        return

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {DEVICE}")
    print(f"Run:    {run_dir}")

    # Load config
    with open(run_dir / 'config.json') as f:
        cfg = json.load(f)

    # Load data
    print("\nLoading data...")
    windows, feat_cols, _ = load_data(
        windows_path     = args.windows,
        split_index_path = args.split_index,
        max_seq_len      = cfg.get('max_seq_len', 100),
        forward_fill     = True,
        models_dir       = str(run_dir),
        fit_scaler       = False,
    )
    eval_wins = windows[windows['split'] == args.split]
    dataset   = AKISequenceDataset(eval_wins, feat_cols)
    print(f"  {len(dataset)} admissions in {args.split} set")
    print(f"  {len(feat_cols)} features")

    # Load model
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
    model.eval()
    print(f"  Model loaded (epoch {ckpt['epoch']}, "
          f"val AUPRC {ckpt['val_auprc']:.4f})")

    # Run IG
    print(f"\nRunning Integrated Gradients "
          f"(n_steps={args.n_steps})...")
    print(f"  This may take a while — reduce --max_admissions for speed")

    ig_results = run_ig_analysis(
        model, dataset, feat_cols, DEVICE,
        n_steps        = args.n_steps,
        max_admissions = args.max_admissions,
    )

    # Aggregate
    print("\nAggregating attributions...")
    aki_df  = aggregate_importance(ig_results['aki'],    feat_cols)
    naki_df = aggregate_importance(ig_results['nonaki'], feat_cols)

    # Save CSVs
    aki_df.to_csv(out_dir / 'ig_aki_importance.csv',    index=False)
    naki_df.to_csv(out_dir / 'ig_nonaki_importance.csv', index=False)

    # Merge into combined file matching your existing split_importance.csv format
    if len(aki_df) > 0 and len(naki_df) > 0:
        merged = aki_df[['feature','mean_abs','mean_signed']].rename(
            columns={'mean_abs': 'aki_importance',
                     'mean_signed': 'aki_signed'}
        ).merge(
            naki_df[['feature','mean_abs','mean_signed']].rename(
                columns={'mean_abs': 'nonaki_importance',
                         'mean_signed': 'nonaki_signed'}
            ),
            on='feature', how='outer'
        ).fillna(0)
        merged['difference'] = merged['aki_importance'] - merged['nonaki_importance']
        merged['ratio']      = (merged['aki_importance'] /
                                 (merged['nonaki_importance'] + 1e-9))
        merged.to_csv(out_dir / 'ig_split_importance.csv', index=False)
        print(f"  Saved ig_split_importance.csv "
              f"— drop-in replacement for split_importance.csv")

    # Plots
    print("\nGenerating plots...")

    if len(aki_df) > 0 and len(naki_df) > 0:
        diff_df = plot_differential_ig(
            aki_df, naki_df, feat_cols,
            plots / 'ig_differential.png',
            top_n=args.top_n
        )

    if len(aki_df) > 0:
        plot_directional(
            aki_df,
            plots / 'ig_directional_aki.png',
            top_n=args.top_n
        )

    # Temporal heatmaps for selected AKI admissions
    print(f"\nGenerating {args.n_heatmaps} temporal heatmaps...")
    aki_items = [item for item in dataset
                 if item['y'].numpy().any()][:args.n_heatmaps]

    for item in aki_items:
        x       = item['x'].unsqueeze(0).to(DEVICE)
        lengths = torch.tensor([item['length']])
        y       = item['y'].numpy()
        adm_id  = item['admission_id']
        hours   = item['hours_since_admit']

        pos_steps = np.where(y == 1)[0]
        h2o = item['hours_to_onset']
        valid = [(i, h2o[i]) for i in pos_steps
                 if i < len(h2o) and not np.isnan(h2o[i]) and h2o[i] >= 0]
        target_step = min(valid, key=lambda x: x[1])[0] if valid else int(pos_steps[0])

        try:
            attr, prob, _ = compute_ig_admission(
                model, x, lengths, target_step, args.n_steps, DEVICE
            )
            plot_temporal_heatmap(
                attr[:item['length']],
                feat_cols,
                str(adm_id),
                hours[:item['length']],
                y[:item['length']],
                plots / f'heatmap_{adm_id}.png',
                top_n=20,
            )
        except Exception as e:
            print(f"  Heatmap failed for {adm_id}: {e}")

    # Compare with existing attention-weighted importance
    attn_path = run_dir / 'feature_importance_split.csv'
    compare_with_attention(
        aki_df,
        attn_path,
        plots / 'ig_vs_attention_scatter.png',
    )

    # Summary
    print(f"\n{'='*55}")
    print("IG EXPLAINABILITY COMPLETE")
    print(f"{'='*55}")
    print(f"  Output directory: {out_dir}/")
    print(f"\n  Key files:")
    print(f"  ig_split_importance.csv  — use with plot_feature_importance.py")
    print(f"  plots/ig_differential.png — main paper figure")
    print(f"  plots/ig_directional_aki.png — directional analysis")
    print(f"  plots/heatmap_*.png — temporal attribution heatmaps")
    print(f"\n  To regenerate plots from saved CSVs:")
    print(f"  python plot_feature_importance.py \\")
    print(f"      --split {out_dir}/ig_split_importance.csv \\")
    print(f"      --overall {out_dir}/ig_aki_importance.csv")

    if len(aki_df) > 0:
        print(f"\n  Top 10 AKI-discriminative features (IG):")
        if 'difference' in diff_df.columns if 'diff_df' in dir() else False:
            print(diff_df.nlargest(10, 'difference')[
                ['feature','difference']
            ].to_string(index=False))
        else:
            print(aki_df.head(10)[['feature','mean_abs','mean_signed']]
                  .to_string(index=False))


if __name__ == '__main__':
    main()
