"""
pic_inference.py

Zero-shot inference: apply GOSH-trained model to PIC windows.
No retraining — tests genuine cross-site generalisability.

Loads the model exactly as evaluate.py does (build_model + checkpoint),
applies the GOSH val-set threshold without modification, and reports
PIC performance under the same metric framework as the GOSH evaluation.

Usage
-----
    python pic_inference.py --run transformer_bce --models_dir model_weights --windows pic_processed/windows_pic_aligned.csv --out_dir data/zero_shot_pic_results --threshold 0.69
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from dataset import load_data, AKISequenceDataset, collate_fn
from models  import build_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--run',        required=True,
                   help='GOSH run name, e.g. transformer_bce')
    p.add_argument('--models_dir', default='model_weights/transformer_bce')
    p.add_argument('--windows',    required=True,
                   help='PIC windows aligned to GOSH feature_cols.csv')
    p.add_argument('--out_dir',    default='data/pic_results')
    p.add_argument('--threshold', type=float, default=None,
               help='Override the GOSH val threshold directly')
    return p.parse_args()


def collect_predictions(model, loader, device) -> pd.DataFrame:
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


def admission_level_results(preds_df, aki_ids, all_ids, threshold):
    tp = fp = fn = tn = 0
    for adm_id in all_ids:
        adm    = preds_df[preds_df['admission_id'] == adm_id]
        is_aki = adm_id in aki_ids
        if is_aki:
            pre   = adm[adm['hours_to_onset'].notna()]
            fired = (pre['prob'] >= threshold).any() if len(pre) > 0 else False
            if fired: tp += 1
            else:     fn += 1
        else:
            fired = (adm['prob'] >= threshold).any()
            if fired: fp += 1
            else:     tn += 1

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


def compute_lead_times(preds_df, aki_ids, threshold):
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


def main():
    args    = parse_args()
    run_dir = Path(args.models_dir) / args.run
    out     = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plots   = out / 'plots'
    plots.mkdir(exist_ok=True)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {DEVICE}")
    print(f"GOSH run: {run_dir}")

    with open(run_dir / 'config.json') as f:
        cfg = json.load(f)

    if args.threshold is not None:
        best_thresh = args.threshold
        print(f"Using manually specified threshold: {best_thresh:.4f}")
    else:
        val_thresh_path = run_dir / 'val_threshold.json'
        if not val_thresh_path.exists():
            raise FileNotFoundError(
                f"{val_thresh_path} not found. Run evaluate.py --split val "
                f"on the GOSH model first, or pass --threshold directly."
            )
        with open(val_thresh_path) as f:
            best_thresh = float(json.load(f)['threshold'])
        print(f"Using GOSH val-set threshold: {best_thresh:.4f}")

    print("\nLoading PIC windows (aligned to GOSH feature_cols)...")
    pic_windows_raw = pd.read_csv(args.windows)
    #pic_windows_raw.columns = pic_windows_raw.columns.str.lower()

    tmp_split = out / '_pic_split_index.csv'
    adm_split = (pic_windows_raw[['admission_id']]
                 .drop_duplicates()
                 .assign(subject_id=lambda d: d['admission_id'],
                         aki_detected=0,
                         split='test'))
    adm_split.to_csv(tmp_split, index=False)

    windows, feat_cols, _ = load_data(
        windows_path     = args.windows,
        split_index_path = str(tmp_split),
        max_seq_len      = cfg.get('max_seq_len', 100),
        forward_fill     = True,
        models_dir       = str(run_dir),
        fit_scaler       = False,
    )
    print(f"  {len(windows):,} windows | "
          f"{windows['admission_id'].nunique():,} admissions | "
          f"{len(feat_cols)} features (must match GOSH input_size)")

    dataset = AKISequenceDataset(windows, feat_cols)
    loader  = DataLoader(dataset, batch_size=32, shuffle=False,
                         collate_fn=collate_fn)

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
    print(f"\nLoaded GOSH model: epoch {ckpt['epoch']} "
          f"(val AUPRC={ckpt['val_auprc']:.4f})")
    print(f"Architecture: {cfg['model']}  attention={cfg.get('attention', False)}")

    print("\nRunning zero-shot inference on PIC...")
    preds = collect_predictions(model, loader, DEVICE)
    preds.to_csv(out / 'pic_predictions_zeroshot.csv', index=False)
    print(f"  Predictions saved: {len(preds):,} rows")

    aki_ids = set(preds[preds['label'] == 1]['admission_id'].unique())
    all_ids = list(preds['admission_id'].unique())
    print(f"  AKI admissions: {len(aki_ids):,} / {len(all_ids):,}")

    y, p  = preds['label'].values, preds['prob'].values
    base  = y.mean()
    auroc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan
    auprc = average_precision_score(y, p) if len(np.unique(y)) > 1 else np.nan

    print(f"\n{'='*50}")
    print("PIC ZERO-SHOT RESULTS (GOSH threshold applied unmodified)")
    print(f"{'='*50}")
    print(f"  AUROC:  {auroc:.3f}")
    print(f"  AUPRC:  {auprc:.3f}  ({auprc/base:.2f}x random baseline {base:.3f})")
    print(f"  Threshold used (from GOSH val): {best_thresh:.4f}")

    m = admission_level_results(preds, aki_ids, all_ids, best_thresh)
    print(f"\n  Admission-level at GOSH threshold:")
    print(f"    Sensitivity: {m['sensitivity']:.3f}")
    print(f"    Specificity: {m['specificity']:.3f}")
    print(f"    PPV:         {m['ppv']:.3f}")
    print(f"    FA ratio:    {m['fa_ratio']:.2f}:1")

    leads     = compute_lead_times(preds, aki_ids, best_thresh)
    lead_ok   = leads.dropna()
    total_aki = len(aki_ids)
    detected  = len(lead_ok)

    print(f"\n  Lead time:")
    print(f"    Detected: {detected}/{total_aki} "
          f"({detected/total_aki*100:.1f}%)")
    if len(lead_ok) > 0:
        print(f"    Median: {lead_ok.median():.0f}h "
              f"IQR {lead_ok.quantile(0.25):.0f}-{lead_ok.quantile(0.75):.0f}h")
        for h in [24, 48]:
            n = (lead_ok >= h).sum()
            print(f"    >={h}h: {n}/{total_aki} ({n/total_aki*100:.1f}%)")

    aki_mean  = preds[preds['admission_id'].isin(aki_ids)]['prob'].mean()
    naki_mean = preds[~preds['admission_id'].isin(aki_ids)]['prob'].mean()
    sep       = aki_mean - naki_mean
    print(f"\n  Score separation: {sep:.4f} "
          f"(AKI mean {aki_mean:.3f} vs non-AKI {naki_mean:.3f})")

    report = {
        'dataset':          'PIC (zero-shot transfer from GOSH)',
        'gosh_run':         args.run,
        'gosh_threshold':   best_thresh,
        'n_admissions':     len(all_ids),
        'n_aki':            len(aki_ids),
        'window_auroc':     round(float(auroc), 4),
        'window_auprc':     round(float(auprc), 4),
        'auprc_vs_random':  round(float(auprc / base), 2),
        'adm_sensitivity':  round(m['sensitivity'], 4),
        'adm_specificity':  round(m['specificity'], 4),
        'adm_ppv':          round(m['ppv'], 4),
        'adm_fa_ratio':     round(m['fa_ratio'], 2),
        'detection_rate':   round(detected / total_aki, 4),
        'lead_median_h':    float(lead_ok.median()) if len(lead_ok) > 0 else None,
        'pct_detected_24h': float((lead_ok >= 24).sum() / total_aki)
                            if len(lead_ok) > 0 else None,
        'pct_detected_48h': float((lead_ok >= 48).sum() / total_aki)
                            if len(lead_ok) > 0 else None,
        'score_separation': round(float(sep), 4),
    }
    with open(out / 'pic_evaluation_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report -> {out}/pic_evaluation_report.json")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(f'PIC zero-shot transfer ({cfg["model"]} trained on GOSH)',
                fontsize=12)

    ax = axes[0]
    aki_p  = preds[preds['admission_id'].isin(aki_ids)]['prob']
    naki_p = preds[~preds['admission_id'].isin(aki_ids)]['prob']
    ax.hist(naki_p, bins=40, alpha=0.6, color='steelblue',
            density=True, label=f'Non-AKI (n={len(naki_p):,})')
    ax.hist(aki_p,  bins=40, alpha=0.6, color='crimson',
            density=True, label=f'AKI (n={len(aki_p):,})')
    ax.axvline(best_thresh, color='black', linestyle='--', lw=1.5,
               label=f'GOSH threshold ({best_thresh:.2f})')
    ax.set_xlabel('Predicted probability')
    ax.set_ylabel('Density')
    ax.set_title(f'Score distribution\nAUROC={auroc:.3f}  AUPRC={auprc:.3f}')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    if len(lead_ok) > 0:
        ax.hist(lead_ok, bins=20, color='crimson', alpha=0.75, edgecolor='white')
        ax.axvline(lead_ok.median(), color='black', lw=2,
                   label=f'Median {lead_ok.median():.0f}h')
        ax.axvline(24, color='gray', lw=1.5, linestyle='--', label='24h')
        ax.axvline(48, color='black', lw=1.5, linestyle=':', label='48h')
        ax.legend(fontsize=8)
    ax.set_xlabel('Hours before AKI onset (lead time)')
    ax.set_ylabel('Count')
    ax.set_title(f'Lead time\n{detected}/{total_aki} detected '
                f'({detected/total_aki*100:.0f}%)')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots / 'pic_zeroshot_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot -> {plots}/pic_zeroshot_results.png")


if __name__ == '__main__':
    main()