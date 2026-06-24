"""
src/train.py

CLI training script for temporal AKI prediction models.

Usage examples
--------------
# LSTM with focal loss (recommended starting point)
python src/train.py --model lstm --loss focal --lr 0.0005 --epochs 80

# GRU with weighted BCE
python src/train.py --model gru --loss bce_weighted --lr 0.001

# Transformer
python src/train.py --model transformer --loss focal --hidden 128 --heads 4

# Full example with all options
python src/train.py \\
    --model       lstm \\
    --loss        focal \\
    --lr          0.0005 \\
    --hidden      128 \\
    --layers      2 \\
    --dropout     0.3 \\
    --batch       32 \\
    --epochs      80 \\
    --patience    15 \\
    --max_seq_len 100 \\
    --gamma       2.0 \\
    --name        lstm_focal_h128

Each run saves to data/models/<name>/
    best_model.pt     — best checkpoint by val AUPRC
    config.json       — all hyperparameters
    history.csv       — per-epoch metrics
    final_metrics.json— val set metrics at best checkpoint
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

# ── Local imports ─────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import load_data, make_loaders
from models  import build_model
from losses  import build_loss


# ─────────────────────────────────────────────────────────
#  Argument parsing
# ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Train temporal AKI prediction model'
    )

    # Data
    p.add_argument('--windows',     default='data/processed/windows_matched.csv')
    p.add_argument('--split_index', default='data/processed/split_index_matched.csv')
    p.add_argument('--models_dir',  default='data/models')
    p.add_argument('--max_seq_len', type=int, default=100,
                   help='Cap admission sequences at this many steps '
                        '(uses LAST N steps). 100 = 25 days at 6h intervals.')

    # Model
    p.add_argument('--model',   default='lstm',
                   choices=['lstm', 'gru', 'transformer'],
                   help='Model architecture')
    p.add_argument('--attention', action='store_true',
                   help='Use attention variant of LSTM/GRU')
    p.add_argument('--hidden',  type=int,   default=128,
                   help='Hidden layer size')
    p.add_argument('--layers',  type=int,   default=2,
                   help='Number of recurrent/transformer layers')
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--heads',   type=int,   default=4,
                   help='Attention heads (transformer only)')

    # Loss
    p.add_argument('--loss',  default='focal',
                   choices=['focal', 'bce_weighted'],
                   help='Loss function')
    p.add_argument('--gamma', type=float, default=2.0,
                   help='Focal loss gamma (ignored for bce_weighted)')
    p.add_argument('--alpha', type=float, default=None,
                   help='Focal loss alpha. If None, derived from class ratio.')

    # Training
    p.add_argument('--lr',        type=float, default=0.0005)
    p.add_argument('--batch',     type=int,   default=32)
    p.add_argument('--epochs',    type=int,   default=80)
    p.add_argument('--patience',  type=int,   default=15,
                   help='Early stopping patience (epochs without val AUPRC improvement)')
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--seed',      type=int,   default=42)

    # Run name
    p.add_argument('--name', default=None,
                   help='Experiment name. Default: auto-generated from args.')

    return p.parse_args()


# ─────────────────────────────────────────────────────────
#  Training loop
# ─────────────────────────────────────────────────────────

def run_epoch(
    model:      nn.Module,
    loader:     DataLoader,
    criterion:  nn.Module,
    optimizer:  torch.optim.Optimizer,
    device:     torch.device,
    grad_clip:  float,
    train:      bool = True,
) -> tuple[float, float, float]:
    """
    Run one epoch.
    Returns (avg_loss, auroc, auprc) over all non-padded positions.
    """
    model.train() if train else model.eval()
    total_loss  = 0.0
    all_probs   = []
    all_labels  = []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            x       = batch['x'].to(device)
            y       = batch['y'].to(device)
            lengths = batch['lengths']

            logits = model(x, lengths)         # (B, T)
            loss   = criterion(logits, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            total_loss += loss.item()

            # Collect non-padded predictions for metrics
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            y_np  = y.detach().cpu().numpy()
            for b, L in enumerate(lengths):
                all_probs.extend(probs[b, :L].tolist())
                all_labels.extend(y_np[b, :L].tolist())

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)
    avg_loss   = total_loss / len(loader)

    n_unique = len(np.unique(all_labels))
    auroc = roc_auc_score(all_labels, all_probs) if n_unique > 1 else 0.5
    auprc = average_precision_score(all_labels, all_probs) if n_unique > 1 else 0.0

    return avg_loss, auroc, auprc


# ─────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Reproducibility ───────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {DEVICE}")

    # ── Experiment name and output directory ──────────────
    if args.name is None:
        args.name = (f"{args.model}_{args.loss}"
                     f"_h{args.hidden}_l{args.layers}"
                     f"_lr{args.lr}"
                     f"_seq{args.max_seq_len}")
    out_dir = Path(args.models_dir) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run name:   {args.name}")
    print(f"Output dir: {out_dir}")

    # ── Save config ───────────────────────────────────────
    config = vars(args)
    with open(out_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # ── Load data ─────────────────────────────────────────
    windows, feat_cols, scaler = load_data(
        windows_path     = args.windows,
        split_index_path = args.split_index,
        max_seq_len      = args.max_seq_len,
        forward_fill     = True,
        models_dir       = str(out_dir),
        fit_scaler       = True,
    )

    train_loader, val_loader, _ = make_loaders(
        windows, feat_cols, batch_size=args.batch
    )

    # Save feature column list
    pd.Series(feat_cols).to_csv(out_dir / 'feature_cols.csv', index=False)

    # ── Class imbalance ───────────────────────────────────
    train_labels = windows[windows['split'] == 'train']['label'].values
    neg = (train_labels == 0).sum()
    pos = (train_labels == 1).sum()
    pos_weight = neg / pos
    print(f"\n  Class ratio: {neg:,} neg / {pos:,} pos = {pos_weight:.1f}:1")

    # ── Build model ───────────────────────────────────────
    INPUT_SIZE = len(feat_cols)
    model = build_model(
        architecture = args.model,
        input_size   = INPUT_SIZE,
        hidden_size  = args.hidden,
        n_layers     = args.layers,
        dropout      = args.dropout,
        n_heads      = args.heads,
        attention    = args.attention,   
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Model: {args.model.upper()} | "
          f"params: {n_params:,} | "
          f"input: {INPUT_SIZE}")

    # ── Loss ──────────────────────────────────────────────
    criterion = build_loss(
        loss_name   = args.loss,
        pos_weight  = pos_weight,
        focal_alpha = args.alpha,
        focal_gamma = args.gamma,
    )

    # ── Optimiser + scheduler ─────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )
    # Reduce LR when val AUPRC plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    # ── Training loop ─────────────────────────────────────
    print(f"\n{'Epoch':>6}  {'Tr loss':>8}  {'Tr AUPRC':>9}  "
          f"{'Val loss':>8}  {'Val AUPRC':>9}  {'LR':>8}")
    print("-" * 60)

    best_val_auprc = 0.0
    patience_ctr   = 0
    history        = []
    t_start        = time.time()

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_auroc, tr_auprc = run_epoch(
            model, train_loader, criterion, optimizer, DEVICE,
            args.grad_clip, train=True
        )
        vl_loss, vl_auroc, vl_auprc = run_epoch(
            model, val_loader, criterion, optimizer, DEVICE,
            args.grad_clip, train=False
        )

        scheduler.step(vl_auprc)
        current_lr = optimizer.param_groups[0]['lr']

        history.append({
            'epoch':      epoch,
            'train_loss': tr_loss,  'train_auroc': tr_auroc,
            'train_auprc':tr_auprc, 'val_loss':    vl_loss,
            'val_auroc':  vl_auroc, 'val_auprc':   vl_auprc,
            'lr':         current_lr,
        })

        marker = ''
        if vl_auprc > best_val_auprc:
            best_val_auprc = vl_auprc
            patience_ctr   = 0
            torch.save({
                'epoch':         epoch,
                'model_state':   model.state_dict(),
                'val_auprc':     vl_auprc,
                'val_auroc':     vl_auroc,
                'args':          vars(args),
                'feat_cols':     feat_cols,
            }, out_dir / 'best_model.pt')
            marker = ' ← best'
        else:
            patience_ctr += 1

        print(f"{epoch:6d}  {tr_loss:8.4f}  {tr_auprc:9.4f}  "
              f"{vl_loss:8.4f}  {vl_auprc:9.4f}  "
              f"{current_lr:8.6f}{marker}")

        if patience_ctr >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(no improvement for {args.patience} epochs)")
            break

    elapsed = time.time() - t_start
    print(f"\nTraining complete in {elapsed/60:.1f} min")
    print(f"Best val AUPRC: {best_val_auprc:.4f}")

    # ── Save history ──────────────────────────────────────
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(out_dir / 'history.csv', index=False)

    # ── Save final metrics ────────────────────────────────
    best_row = hist_df.loc[hist_df['val_auprc'].idxmax()]
    metrics  = {
        'best_epoch':  int(best_row['epoch']),
        'val_auroc':   round(float(best_row['val_auroc']),  4),
        'val_auprc':   round(float(best_row['val_auprc']),  4),
        'train_auprc': round(float(best_row['train_auprc']), 4),
    }
    with open(out_dir / 'final_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved to {out_dir}/")
    print(f"  best_model.pt | history.csv | "
          f"config.json | final_metrics.json")
    print(f"\nTo evaluate: python src/evaluate.py --run {args.name}")


if __name__ == '__main__':
    main()
