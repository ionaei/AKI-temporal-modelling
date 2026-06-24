"""
src/dataset.py

Data loading, preprocessing, Dataset and DataLoader for temporal AKI prediction.
Shared by train.py and evaluate.py.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

# ── Columns that are never features ──────────────────────
NON_FEATURE_COLS = {
    'admission_id', 'subject_id', 'step_index',
    'window_start', 'window_end', 'hours_since_admit',
    'label', 'aki_stage', 'onset_dt', 'hours_to_onset',
    'exclude_early_onset', 'baseline_creatinine', 'baseline_source',
    'admit_dt', 'discharge_dt', 'dob', 'split', 'rank_from_end', 'proc_dialysis_rrt',
}


def get_feature_cols(windows: pd.DataFrame) -> list[str]:
    return [
        c for c in windows.columns
        if c not in NON_FEATURE_COLS
        and windows[c].dtype in ['float32', 'float64',
                                  'int32', 'int64', int, float]
    ]


def forward_fill_windows(windows: pd.DataFrame,
                          feature_cols: list[str]) -> pd.DataFrame:
    """
    Forward-fill feature values within each admission sequence.
    Only fills non-missing-flag columns.
    The _missing_flag columns are left unchanged — they correctly
    record which 6h steps had no measurement.
    """
    fill_cols = [c for c in feature_cols
                 if not c.endswith('_missing_flag')]
    print("  Applying forward fill within admissions...")
    
    windows = (
        windows
        .sort_values(['admission_id', 'step_index'])
        .reset_index(drop=True)
    )

    windows[fill_cols] = (
        windows.groupby('admission_id')[fill_cols].transform(lambda x: x.ffill()))
    return windows


def load_data(
    windows_path:     str,
    split_index_path: str,
    max_seq_len:      int   = 100,
    forward_fill:     bool  = True,
    models_dir:       str   = 'data/models',
    fit_scaler:       bool  = True,
) -> tuple[pd.DataFrame, list[str], StandardScaler]:
    """
    Load windows CSV, apply preprocessing, return ready-to-use DataFrame.

    Parameters
    ----------
    windows_path     : path to windows_matched.csv
    split_index_path : path to split_index_matched.csv
    max_seq_len      : cap sequence length (take LAST N steps per admission)
    forward_fill     : fill missing lab values from previous step
    models_dir       : where to save/load scaler
    fit_scaler       : True = fit new scaler (training run)
                       False = load existing scaler (evaluation run)

    Returns
    -------
    windows      : preprocessed DataFrame with 'split' column
    feature_cols : list of feature column names (after adding flags)
    scaler       : fitted StandardScaler
    """
    print(f"\nLoading: {windows_path}")
    
    #windows = pd.read_csv(windows_path)
    windows = pd.read_parquet(windows_path)
    windows = windows.sort_values(
        ['admission_id', 'step_index']
    ).reset_index(drop=True)

    print(f"  {len(windows):,} windows | "
          f"{windows['admission_id'].nunique()} admissions | "
          f"{windows['label'].mean()*100:.2f}% positive")

    # ── Sequence length cap ───────────────────────────────
    # Long admissions dominate training — cap at last N steps
    # (most recent data is most predictive)
    seq_lens = windows.groupby('admission_id').size()
    n_capped = (seq_lens > max_seq_len).sum()
    if n_capped > 0:
        print(f"  Capping {n_capped} admissions at {max_seq_len} steps "
              f"(max was {seq_lens.max()})")
        windows['_rank_from_end'] = (
            windows
            .groupby('admission_id')['step_index'].rank(method='first', ascending=False))
        
        windows = windows[windows['_rank_from_end'] <= max_seq_len].copy()
        windows = windows.drop(columns = ['_rank_from_end'])
    windows['step_index'] = windows.groupby('admission_id').cumcount()
    windows = windows.reset_index(drop=True)

    # ── Attach split ──────────────────────────────────────
    split_path = Path(split_index_path)
    if split_path.exists():
        split_idx  = pd.read_csv(split_path)
        subj_split = (split_idx[['subject_id', 'split']]
                      .drop_duplicates(subset='subject_id'))
        windows    = windows.merge(subj_split, on='subject_id', how='left')
        print(f"  Split loaded from {split_path.name}")
    else:
        raise FileNotFoundError(
            f"Split index not found: {split_path}\n"
            "Run the split creation step first."
        )

    # ── Feature columns ───────────────────────────────────
    feature_cols = get_feature_cols(windows)

    # ── Forward fill ──────────────────────────────────────
    if forward_fill:
        windows = forward_fill_windows(windows, feature_cols)

    # ── Missingness indicator flags ───────────────────────
    # Add _missing_flag columns BEFORE scaling
    # These tell the model "no measurement in this window"
    missing_flag_cols = []
    for col in list(feature_cols):  # iterate copy
        if windows[col].isna().any():
            flag = f'{col}_missing_flag'
            if flag not in windows.columns:
                windows[flag] = windows[col].isna().astype(float)
            if flag not in missing_flag_cols:
                missing_flag_cols.append(flag)

    all_feat_cols = feature_cols + missing_flag_cols
    print(f"  {len(feature_cols)} features + "
          f"{len(missing_flag_cols)} missing flags = "
          f"{len(all_feat_cols)} total")

    # ── Fit or load scaler ────────────────────────────────
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    scaler_path = models_path / 'scaler.pkl'

    train_mask = windows['split'] == 'train'
    scaler     = StandardScaler()

    if fit_scaler:
        print("  Fitting scaler on training windows...")
        scaler.fit(windows.loc[train_mask, all_feat_cols].fillna(0))
        joblib.dump(scaler, scaler_path)
        print(f"  Scaler saved → {scaler_path}")
    else:
        if not scaler_path.exists():
            raise FileNotFoundError(
                f"Scaler not found at {scaler_path}. "
                "Run training first (fit_scaler=True)."
            )
        scaler = joblib.load(scaler_path)
        print(f"  Scaler loaded from {scaler_path}")

    windows[all_feat_cols] = scaler.transform(
        windows[all_feat_cols].fillna(0)
    )

    # ── Summary ───────────────────────────────────────────
    print(f"\n  Split summary:")
    print(f"  {'Set':6s}  {'Windows':>8s}  {'Admissions':>10s}  "
          f"{'Patients':>8s}  {'Pos%':>6s}")
    print(f"  {'-'*46}")
    for sp in ['train', 'val', 'test']:
        sub = windows[windows['split'] == sp]
        if len(sub) == 0:
            continue
        print(f"  {sp:6s}  {len(sub):8,}  "
              f"{sub['admission_id'].nunique():10d}  "
              f"{sub['subject_id'].nunique():8d}  "
              f"{sub['label'].mean()*100:6.2f}")

    return windows, all_feat_cols, scaler


# ── PyTorch Dataset ───────────────────────────────────────

class AKISequenceDataset(Dataset):
    """
    One item = one admission = a variable-length sequence of 6h windows.
    Each window is one time step for the LSTM/GRU/Transformer.
    """
    def __init__(self, windows_df: pd.DataFrame, feature_cols: list[str]):
        self.items = []
        for adm_id, grp in windows_df.groupby('admission_id'):
            grp = grp.sort_values('step_index').reset_index(drop=True)

            x = torch.tensor(
                grp[feature_cols].values, dtype=torch.float32
            )
            y = torch.tensor(
                grp['label'].values, dtype=torch.float32
            )
            self.items.append({
                'x':                 x,
                'y':                 y,
                'length':            len(grp),
                'admission_id':      adm_id,
                'hours_to_onset':    grp['hours_to_onset'].values,
                'hours_since_admit': grp['hours_since_admit'].values,
                'aki_stage':         int(grp['aki_stage'].iloc[0])
                                     if 'aki_stage' in grp.columns else 0,
                'step_index':        grp['step_index'].values,
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate_fn(batch: list) -> dict:
    """Pad variable-length sequences to the longest in the batch."""
    batch   = sorted(batch, key=lambda x: x['length'], reverse=True)
    lengths = torch.tensor([b['length'] for b in batch])

    x_padded = pad_sequence(
        [b['x'] for b in batch], batch_first=True, padding_value=0.0
    )
    # Padding positions use -1 so they can be masked in the loss
    y_padded = pad_sequence(
        [b['y'] for b in batch], batch_first=True, padding_value=-1.0
    )
    return {
        'x':                 x_padded,
        'y':                 y_padded,
        'lengths':           lengths,
        'admission_ids':     [b['admission_id']      for b in batch],
        'hours_to_onset':    [b['hours_to_onset']    for b in batch],
        'hours_since_admit': [b['hours_since_admit'] for b in batch],
        'aki_stages':        [b['aki_stage']         for b in batch],
        'step_indices':      [b['step_index']        for b in batch],
    }


def make_loaders(
    windows:      pd.DataFrame,
    feature_cols: list[str],
    batch_size:   int = 32,
    num_workers:  int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train, val, test DataLoaders from preprocessed windows."""
    train_ds = AKISequenceDataset(
        windows[windows['split'] == 'train'], feature_cols
    )
    val_ds   = AKISequenceDataset(
        windows[windows['split'] == 'val'],   feature_cols
    )
    test_ds  = AKISequenceDataset(
        windows[windows['split'] == 'test'],  feature_cols
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers,
    )
    val_loader   = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers,
    )
    test_loader  = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers,
    )

    print(f"\n  DataLoaders: train={len(train_ds)} | "
          f"val={len(val_ds)} | test={len(test_ds)} admissions")
    return train_loader, val_loader, test_loader
