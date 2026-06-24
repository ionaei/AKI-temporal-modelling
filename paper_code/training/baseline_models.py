"""
baseline_models.py

XGBoost and Random Forest baselines for GOSH AKI prediction.

Leakage-prevention design
--------------------------
1. Test patients are isolated at the start and never used in any CV fold.
2. Cross-validation is performed at PATIENT level — no patient's windows
   appear in both train and validation within a single fold.
3. The StandardScaler is fitted INSIDE each CV fold on that fold's
   training windows only.
4. Hyperparameter tuning selects the best config by mean val AUPRC
   across 5 folds.
5. Final models are trained on ALL train+val patients with the best
   hyperparameters, then evaluated ONCE on the held-out test set.

Usage
-----
    python baseline_models.py --windows data/processed/windows_matched.csv --split_index data/processed/split_index_matched.csv --out_dir data/baselines

    # Faster run with smaller grid (for testing)
    python baseline_models.py --fast

Output
------
    data/baselines/
        xgb_cv_results.csv        per-fold per-config CV results
        rf_cv_results.csv
        xgb_best_config.json      best hyperparameters
        rf_best_config.json
        xgb_evaluation_report.json  test set metrics
        rf_evaluation_report.json
        comparison_table.csv        side-by-side with GOSH LSTM results
        plots/
            xgb_score_distributions.png
            rf_score_distributions.png
            xgb_feature_importance.png
            rf_feature_importance.png
            model_comparison.png
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
    confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

warnings.filterwarnings('ignore', category=UserWarning)


# ─────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────

NON_FEATURE_COLS = {
    'admission_id', 'subject_id', 'step_index',
    'window_start', 'window_end', 'hours_since_admit',
    'label', 'aki_stage', 'onset_dt', 'hours_to_onset',
    'exclude_early_onset', 'baseline_creatinine',
    'baseline_source', 'admit_dt', 'discharge_dt',
    'dob', 'split', 'has_aki', 'rank_from_end',
}

N_FOLDS         = 5
RANDOM_SEED     = 42
MIN_SENSITIVITY = 0.75   # threshold selection: max specificity subject to this
MIN_THRESHOLD   = 0.05   # floor on candidate thresholds in the sweep


# ─────────────────────────────────────────────────────────
#  Hyperparameter grids
# ─────────────────────────────────────────────────────────

XGB_GRID = [
    # n_estimators, max_depth, learning_rate, subsample, min_child_weight, scale_pos_weight
    {'n_estimators': 300, 'max_depth': 4, 'learning_rate': 0.05,
     'subsample': 0.8, 'min_child_weight': 5,  'scale_pos_weight': 'auto'},
    {'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.05,
     'subsample': 0.8, 'min_child_weight': 5,  'scale_pos_weight': 'auto'},
    {'n_estimators': 500, 'max_depth': 4, 'learning_rate': 0.02,
     'subsample': 0.7, 'min_child_weight': 10, 'scale_pos_weight': 'auto'},
    {'n_estimators': 500, 'max_depth': 6, 'learning_rate': 0.02,
     'subsample': 0.7, 'min_child_weight': 10, 'scale_pos_weight': 'auto'},
    {'n_estimators': 300, 'max_depth': 3, 'learning_rate': 0.1,
     'subsample': 0.9, 'min_child_weight': 3,  'scale_pos_weight': 'auto'},
]

RF_GRID = [
    # n_estimators, max_depth, min_samples_leaf, max_features, class_weight
    {'n_estimators': 300, 'max_depth': 8,    'min_samples_leaf': 20,
     'max_features': 'sqrt', 'class_weight': 'balanced'},
    {'n_estimators': 300, 'max_depth': 12,   'min_samples_leaf': 20,
     'max_features': 'sqrt', 'class_weight': 'balanced'},
    {'n_estimators': 300, 'max_depth': 8,    'min_samples_leaf': 10,
     'max_features': 'sqrt', 'class_weight': 'balanced'},
    {'n_estimators': 500, 'max_depth': 10,   'min_samples_leaf': 15,
     'max_features': 0.3,    'class_weight': 'balanced'},
    {'n_estimators': 300, 'max_depth': None, 'min_samples_leaf': 50,
     'max_features': 'sqrt', 'class_weight': 'balanced'},
]

# Smaller grid for --fast mode
XGB_GRID_FAST = [XGB_GRID[0], XGB_GRID[2]]
RF_GRID_FAST  = [RF_GRID[0],  RF_GRID[2]]


# ─────────────────────────────────────────────────────────
#  Argument parsing
# ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='XGBoost + RF baselines for GOSH AKI prediction'
    )
    p.add_argument('--windows',
                   default='data/processed/windows_matched.csv')
    p.add_argument('--split_index',
                   default='data/processed/split_index_matched.csv')
    p.add_argument('--out_dir', default='data/baselines')
    p.add_argument('--fast', action='store_true',
                   help='Use small grid for quick testing')
    p.add_argument('--n_jobs', type=int, default=-1,
                   help='Parallel workers (-1 = all cores)')
    return p.parse_args()


# ─────────────────────────────────────────────────────────
#  Data loading
# ─────────────────────────────────────────────────────────

def load_windows(windows_path: str,
                  split_index_path: str) -> tuple:
    """
    Load and preprocess windows.
    Returns (windows_df, feature_cols, patient_splits)

    Leakage note: scaler is NOT fitted here — it is fitted
    inside each CV fold on that fold's training data only.
    """
    print(f"\nLoading {windows_path}")
    windows = pd.read_csv(windows_path)
    windows.columns = windows.columns.str.lower()
    windows = windows.sort_values(
        ['admission_id', 'step_index']
    ).reset_index(drop=True)

    # Load split index
    split_idx = pd.read_csv(split_index_path)
    split_idx.columns = split_idx.columns.str.lower()

    # Normalise split column
    if 'split' not in split_idx.columns:
        raise KeyError(
            f"No 'split' column in split index. "
            f"Columns: {list(split_idx.columns)}"
        )

    # Merge split labels
    if 'admission_id' in split_idx.columns:
        split_idx['admission_id'] = split_idx['admission_id'].astype(str)
        windows['admission_id']   = windows['admission_id'].astype(str)
        if 'split' in windows.columns:
            windows = windows.drop(columns=['split'])
        adm_split = split_idx[['admission_id','split']].drop_duplicates('admission_id')
        windows   = windows.merge(adm_split, on='admission_id', how='inner')
    else:
        split_idx['subject_id'] = split_idx['subject_id'].astype(str)
        windows['subject_id']   = windows['subject_id'].astype(str)
        if 'split' in windows.columns:
            windows = windows.drop(columns=['split'])
        subj_split = split_idx[['subject_id','split']].drop_duplicates('subject_id')
        windows    = windows.merge(subj_split, on='subject_id', how='inner')

    print(f"  {len(windows):,} windows after split merge")
    print(f"  Splits: { {s: (windows['split']==s).sum() for s in ['train','val','test']} }")

    # Feature columns
    feat_cols = [
        c for c in windows.columns
        if c not in NON_FEATURE_COLS
        and windows[c].dtype in [
            'float32', 'float64', 'int32', 'int64', int, float
        ]
    ]
    print(f"  Feature columns: {len(feat_cols)}")

    # Forward fill within admissions
    print("  Forward filling within admissions...")
    fill_cols = [c for c in feat_cols if not c.endswith('_missing_flag')]
    windows = windows.sort_values(['admission_id', 'step_index'])
    windows[fill_cols] = (
        windows.groupby('admission_id')[fill_cols]
        .transform('ffill')
    )
    windows[feat_cols] = windows[feat_cols].fillna(0)

    # Patient-level split map for CV
    if 'subject_id' in windows.columns:
        patient_splits = (
            windows[['subject_id', 'split']]
            .drop_duplicates('subject_id')
            .set_index('subject_id')['split']
            .to_dict()
        )
    else:
        patient_splits = {}

    print(f"\n  Positive rate by split:")
    for s in ['train', 'val', 'test']:
        sub = windows[windows['split'] == s]
        print(f"    {s}: {sub['label'].mean()*100:.2f}%  "
              f"({sub['label'].sum():,} / {len(sub):,})")

    return windows, feat_cols, patient_splits


# ─────────────────────────────────────────────────────────
#  Patient-level CV fold construction
# ─────────────────────────────────────────────────────────

def make_cv_folds(windows: pd.DataFrame,
                   n_folds: int = 5,
                   seed: int = RANDOM_SEED) -> list:
    """
    Construct N stratified folds at PATIENT level.

    Only uses train+val patients — test patients are excluded.
    Each fold returns (train_patient_ids, val_patient_ids).

    Stratified by AKI status so each fold has similar AKI rates.
    """
    rng = np.random.default_rng(seed)

    trainval = windows[windows['split'].isin(['train', 'val'])].copy()

    # Patient-level AKI status (1 if any window in admission is positive)
    if 'subject_id' in trainval.columns:
        pat_col = 'subject_id'
    else:
        pat_col = 'admission_id'

    pat_aki = (
        trainval.groupby(pat_col)['label']
        .max()
        .reset_index()
        .rename(columns={'label': 'has_aki'})
    )

    aki_patients  = pat_aki[pat_aki['has_aki'] == 1][pat_col].tolist()
    naki_patients = pat_aki[pat_aki['has_aki'] == 0][pat_col].tolist()

    rng.shuffle(aki_patients)
    rng.shuffle(naki_patients)

    # Split into n_folds chunks
    aki_folds  = np.array_split(aki_patients,  n_folds)
    naki_folds = np.array_split(naki_patients, n_folds)

    folds = []
    for i in range(n_folds):
        val_pats = (
            list(aki_folds[i]) + list(naki_folds[i])
        )
        train_pats = []
        for j in range(n_folds):
            if j != i:
                train_pats += list(aki_folds[j]) + list(naki_folds[j])
        folds.append((train_pats, val_pats))

    # Verify no overlap
    for i, (tr, va) in enumerate(folds):
        overlap = set(tr) & set(va)
        assert len(overlap) == 0, \
            f"Patient overlap in fold {i}: {len(overlap)} patients"

    print(f"\n  CV folds ({n_folds}-fold, patient-level stratified):")
    pat_col_ref = pat_col
    for i, (tr, va) in enumerate(folds):
        tr_wins = trainval[trainval[pat_col_ref].isin(tr)]
        va_wins = trainval[trainval[pat_col_ref].isin(va)]
        print(f"    Fold {i+1}: "
              f"train {len(tr_wins):,} wins / {len(tr)} pats "
              f"({tr_wins['label'].mean()*100:.1f}% pos)  |  "
              f"val {len(va_wins):,} wins / {len(va)} pats "
              f"({va_wins['label'].mean()*100:.1f}% pos)")

    return folds, pat_col


# ─────────────────────────────────────────────────────────
#  Single fold training + evaluation
# ─────────────────────────────────────────────────────────

def train_eval_fold(windows:    pd.DataFrame,
                     feat_cols:  list,
                     pat_col:    str,
                     train_pats: list,
                     val_pats:   list,
                     model_cls,
                     model_params: dict) -> dict:
    """
    Train a model on one fold and evaluate on validation windows.

    Leakage note:
    - Scaler is fitted on train_wins ONLY
    - Val windows are transformed using the train scaler
    - No val data touches the scaler at any point
    """
    train_wins = windows[windows[pat_col].isin(train_pats)]
    val_wins   = windows[windows[pat_col].isin(val_pats)]

    X_train = train_wins[feat_cols].values.astype(np.float32)
    y_train = train_wins['label'].values.astype(int)
    X_val   = val_wins[feat_cols].values.astype(np.float32)
    y_val   = val_wins['label'].values.astype(int)

    # Fit scaler on training data ONLY
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)   # transform only — no fit

    # Resolve scale_pos_weight
    params = dict(model_params)
    if params.get('scale_pos_weight') == 'auto':
        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()
        params['scale_pos_weight'] = neg / max(pos, 1)

    model = model_cls(**params)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_val)[:, 1]

    n_unique = len(np.unique(y_val))
    auroc = roc_auc_score(y_val, probs) if n_unique > 1 else 0.5
    auprc = average_precision_score(y_val, probs) if n_unique > 1 else 0.0

    return {
        'auroc': auroc,
        'auprc': auprc,
        'n_train': len(y_train),
        'n_val':   len(y_val),
        'pos_train': int(y_train.sum()),
        'pos_val':   int(y_val.sum()),
    }


# ─────────────────────────────────────────────────────────
#  Cross-validated hyperparameter search
# ─────────────────────────────────────────────────────────

def cv_hyperparameter_search(windows:    pd.DataFrame,
                               feat_cols:  list,
                               folds:      list,
                               pat_col:    str,
                               model_name: str,
                               model_cls,
                               param_grid: list,
                               n_jobs:     int = -1) -> tuple:
    """
    Run 5-fold CV for each hyperparameter configuration.
    Select best config by mean val AUPRC across folds.

    Returns (best_params, cv_results_df)
    """
    print(f"\n{'='*55}")
    print(f"  {model_name} — hyperparameter search")
    print(f"  {len(param_grid)} configs × {len(folds)} folds "
          f"= {len(param_grid)*len(folds)} fits")
    print(f"{'='*55}")

    results = []
    for ci, params in enumerate(param_grid):
        fold_aurocs = []
        fold_auprcs = []
        for fi, (train_pats, val_pats) in enumerate(folds):
            fold_res = train_eval_fold(
                windows, feat_cols, pat_col,
                train_pats, val_pats,
                model_cls, params
            )
            fold_aurocs.append(fold_res['auroc'])
            fold_auprcs.append(fold_res['auprc'])

        mean_auroc = float(np.mean(fold_aurocs))
        mean_auprc = float(np.mean(fold_auprcs))
        std_auprc  = float(np.std(fold_auprcs))

        result_row = {
            'config_id':   ci,
            'mean_auroc':  round(mean_auroc, 4),
            'mean_auprc':  round(mean_auprc, 4),
            'std_auprc':   round(std_auprc,  4),
            'fold_auprcs': fold_auprcs,
            **{k: v for k, v in params.items()
               if k != 'scale_pos_weight'},
        }
        results.append(result_row)

        print(f"  Config {ci+1}/{len(param_grid)}: "
              f"AUROC={mean_auroc:.3f}  "
              f"AUPRC={mean_auprc:.3f} ± {std_auprc:.3f}  "
              f"params={params}")

    results_df  = pd.DataFrame(results)
    best_idx    = results_df['mean_auprc'].idxmax()
    best_config = results_df.loc[best_idx]
    best_params = param_grid[int(best_config['config_id'])]

    print(f"\n  Best config (AUPRC={best_config['mean_auprc']:.4f}): "
          f"{best_params}")

    return best_params, results_df


# ─────────────────────────────────────────────────────────
#  Admission-level threshold selection
#  (max specificity subject to sensitivity >= MIN_SENSITIVITY)
# ─────────────────────────────────────────────────────────

def admission_level_sens_spec(df:       pd.DataFrame,
                               aki_ids:  set,
                               all_ids:  list,
                               thresh:   float) -> tuple:
    """
    Compute admission-level sensitivity and specificity at a
    given probability threshold. A true positive requires at
    least one pre-onset window with prob >= thresh.
    """
    tp = fp = fn = tn = 0
    for adm_id in all_ids:
        adm    = df[df['admission_id'] == adm_id]
        is_aki = adm_id in aki_ids
        if is_aki:
            pre   = adm[adm['hours_to_onset'].notna()]
            fired = (pre['prob'] >= thresh).any() if len(pre) > 0 else False
            if fired: tp += 1
            else:     fn += 1
        else:
            fired = (adm['prob'] >= thresh).any()
            if fired: fp += 1
            else:     tn += 1

    sens = tp / (tp + fn + 1e-9)
    spec = tn / (tn + fp + 1e-9)
    return sens, spec


def find_best_threshold_admission_level(
        df:              pd.DataFrame,
        aki_ids:         set,
        all_ids:         list,
        min_sensitivity: float = MIN_SENSITIVITY,
        min_threshold:   float = MIN_THRESHOLD) -> tuple:
    """
    Sweep thresholds and select the one that maximises admission-level
    specificity subject to sensitivity >= min_sensitivity.

    Falls back to Youden-J (sens + spec - 1) if no threshold meets
    the sensitivity constraint.

    Returns (best_threshold, sweep_df)
    """
    rows = []
    for thresh in np.arange(min_threshold, 0.95, 0.01):
        sens, spec = admission_level_sens_spec(df, aki_ids, all_ids, thresh)
        rows.append({
            'threshold':   round(thresh, 2),
            'sensitivity': sens,
            'specificity': spec,
            'youden_j':    sens + spec - 1,
        })

    sweep_df = pd.DataFrame(rows)
    valid    = sweep_df[sweep_df['sensitivity'] >= min_sensitivity]

    if len(valid) == 0:
        print(f"    WARNING: no threshold achieves sensitivity "
              f">= {min_sensitivity} — falling back to Youden-J")
        best_idx = sweep_df['youden_j'].idxmax()
    else:
        best_idx = valid['specificity'].idxmax()

    best_thresh = float(sweep_df.loc[best_idx, 'threshold'])
    print(f"    Threshold (max specificity, sens>={min_sensitivity}): "
          f"{best_thresh:.4f}")
    print(f"      Sensitivity: {sweep_df.loc[best_idx,'sensitivity']:.3f}")
    print(f"      Specificity: {sweep_df.loc[best_idx,'specificity']:.3f}")

    return best_thresh, sweep_df


def train_final_and_evaluate(windows:     pd.DataFrame,
                               feat_cols:  list,
                               pat_col:    str,
                               best_params: dict,
                               model_cls,
                               model_name: str) -> tuple:
    """
    Train final model on ALL train+val patients with best hyperparameters.
    Evaluate ONCE on held-out test set.

    Leakage note:
    - Scaler fitted on train+val windows only
    - Test windows are transformed but never fitted
    - This function is called only once per model
    """
    print(f"\n{'='*55}")
    print(f"  {model_name} — final model (train+val → test)")
    print(f"{'='*55}")

    trainval = windows[windows['split'].isin(['train', 'val'])]
    test     = windows[windows['split'] == 'test']

    X_tv = trainval[feat_cols].values.astype(np.float32)
    y_tv = trainval['label'].values.astype(int)
    X_te = test[feat_cols].values.astype(np.float32)
    y_te = test['label'].values.astype(int)

    # Scaler fitted on train+val ONLY
    scaler = StandardScaler()
    X_tv   = scaler.fit_transform(X_tv)
    X_te   = scaler.transform(X_te)

    params = dict(best_params)
    if params.get('scale_pos_weight') == 'auto':
        neg = (y_tv == 0).sum()
        pos = (y_tv == 1).sum()
        params['scale_pos_weight'] = neg / max(pos, 1)
        print(f"  scale_pos_weight = {params['scale_pos_weight']:.1f}")

    print(f"  Training on {len(X_tv):,} windows "
          f"({y_tv.sum():,} positive)...")
    model = model_cls(**params)
    model.fit(X_tv, y_tv)

    probs    = model.predict_proba(X_te)[:, 1]
    probs_tv = model.predict_proba(X_tv)[:, 1]

    # Window-level metrics
    auroc = roc_auc_score(y_te, probs)
    auprc = average_precision_score(y_te, probs)
    base  = y_te.mean()

    print(f"\n  Window-level (test set):")
    print(f"    AUROC:  {auroc:.4f}")
    print(f"    AUPRC:  {auprc:.4f}  ({auprc/base:.1f}× random)")

    # Threshold: max specificity subject to sensitivity >= MIN_SENSITIVITY
    # (same selection rule used for the deep temporal models — computed
    # at admission level on train+val, never on the held-out test set)
    print(f"  Selecting threshold on train+val (admission-level)...")
    tv_df = trainval.copy()
    tv_df['prob'] = probs_tv
    tv_aki_ids = set(tv_df[tv_df['label'] == 1]['admission_id'].unique())
    tv_all_ids = list(tv_df['admission_id'].unique())

    threshold, thresh_sweep_df = find_best_threshold_admission_level(
        tv_df, tv_aki_ids, tv_all_ids,
        min_sensitivity=MIN_SENSITIVITY,
        min_threshold=MIN_THRESHOLD,
    )

    # Admission-level metrics
    test_df = test.copy()
    test_df['prob'] = probs
    aki_ids = set(
        test_df[test_df['label'] == 1]['admission_id'].unique()
    )
    all_ids = list(test_df['admission_id'].unique())

    tp = fp = fn = tn = 0
    leads = []
    for adm_id in all_ids:
        adm    = test_df[test_df['admission_id'] == adm_id]
        is_aki = adm_id in aki_ids
        if is_aki:
            pre   = adm[adm['hours_to_onset'].notna()]
            fired = (pre['prob'] >= threshold).any() if len(pre) > 0 else False
            if fired:
                tp += 1
                first_alert = pre[pre['prob'] >= threshold].iloc[0]
                leads.append(float(first_alert['hours_to_onset']))
            else:
                fn += 1
                leads.append(np.nan)
        else:
            fired = (adm['prob'] >= threshold).any()
            if fired: fp += 1
            else:     tn += 1

    sens   = tp / (tp + fn + 1e-9)
    spec   = tn / (tn + fp + 1e-9)
    ppv    = tp / (tp + fp + 1e-9)
    fa_r   = fp / (tp + 1e-9)

    lead_s    = pd.Series(leads).dropna()
    total_aki = len(aki_ids)
    detected  = len(lead_s)

    print(f"\n  Admission-level @ threshold={threshold:.3f}:")
    print(f"    TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"    Sensitivity: {sens:.3f}  Specificity: {spec:.3f}")
    print(f"    PPV: {ppv:.3f}  FA ratio: {fa_r:.2f}:1")
    print(f"\n  Lead time:")
    print(f"    Detected: {detected}/{total_aki} "
          f"({detected/total_aki*100:.1f}%)")
    if len(lead_s) > 0:
        print(f"    Median:  {lead_s.median():.0f}h  "
              f"IQR {lead_s.quantile(0.25):.0f}–"
              f"{lead_s.quantile(0.75):.0f}h")
        for h in [24, 48]:
            n   = (lead_s >= h).sum()
            pct = n / total_aki * 100
            print(f"    ≥{h}h: {n}/{total_aki} ({pct:.1f}%)")

    report = {
        'model':             model_name,
        'window_auroc':      round(auroc, 4),
        'window_auprc':      round(auprc, 4),
        'auprc_vs_random':   round(auprc / base, 2),
        'best_threshold':    round(threshold, 4),
        'min_sensitivity_constraint': MIN_SENSITIVITY,
        'adm_sensitivity':   round(sens, 4),
        'adm_specificity':   round(spec, 4),
        'adm_ppv':           round(ppv, 4),
        'adm_fa_ratio':      round(fa_r, 2),
        'detected_pct':      round(detected / total_aki, 4),
        'lead_median_h':     round(float(lead_s.median()), 1)
                             if len(lead_s) > 0 else None,
        'lead_iqr_25':       round(float(lead_s.quantile(0.25)), 1)
                             if len(lead_s) > 0 else None,
        'lead_iqr_75':       round(float(lead_s.quantile(0.75)), 1)
                             if len(lead_s) > 0 else None,
        'pct_detected_24h':  round(float((lead_s>=24).sum()/total_aki), 4)
                             if len(lead_s) > 0 else None,
        'pct_detected_48h':  round(float((lead_s>=48).sum()/total_aki), 4)
                             if len(lead_s) > 0 else None,
        'score_separation':  round(
            float(test_df[test_df['admission_id'].isin(aki_ids)]['prob'].mean()) -
            float(test_df[~test_df['admission_id'].isin(aki_ids)]['prob'].mean()),
            4
        ),
    }

    return model, scaler, test_df, report


# ─────────────────────────────────────────────────────────
#  Feature importance
# ─────────────────────────────────────────────────────────

def plot_feature_importance(model,
                              feat_cols: list,
                              model_name: str,
                              save_path: Path,
                              top_n: int = 25):
    """Plot top N feature importances."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        print(f"  No feature_importances_ for {model_name}")
        return

    imp_df = pd.DataFrame({
        'feature':    feat_cols,
        'importance': importances,
    }).sort_values('importance', ascending=False).head(top_n)

    # Clean up feature names for display
    imp_df['feature_clean'] = (
        imp_df['feature']
        .str.replace('_mean', ' (mean)', regex=False)
        .str.replace('_last', ' (last)', regex=False)
        .str.replace('_missing', ' missing', regex=False)
        .str.replace('_', ' ')
    )

    fig, ax = plt.subplots(figsize=(9, top_n * 0.32 + 1.5))
    colours = ['#C0392B' if 'creatinine' in f or 'urea' in f or 'egfr' in f
               else '#2980B9' for f in imp_df['feature']]
    ax.barh(range(len(imp_df)), imp_df['importance'],
            color=colours, alpha=0.8)
    ax.set_yticks(range(len(imp_df)))
    ax.set_yticklabels(imp_df['feature_clean'], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Feature importance')
    ax.set_title(f'{model_name} — top {top_n} features\n'
                 f'(red = renal markers)')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Feature importance → {save_path.name}")


# ─────────────────────────────────────────────────────────
#  Score distribution plot
# ─────────────────────────────────────────────────────────

def plot_score_distributions(test_df:    pd.DataFrame,
                               aki_ids:   set,
                               threshold: float,
                               model_name: str,
                               save_path: Path):
    aki_p    = test_df[ test_df['admission_id'].isin(aki_ids)]['prob']
    nonaki_p = test_df[~test_df['admission_id'].isin(aki_ids)]['prob']

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(nonaki_p, bins=40, alpha=0.6, color='steelblue',
            density=True, label=f'Non-AKI (n={len(nonaki_p):,})')
    ax.hist(aki_p,    bins=40, alpha=0.6, color='crimson',
            density=True, label=f'AKI (n={len(aki_p):,})')
    ax.axvline(threshold, color='black', linestyle='--', lw=1.5,
               label=f'Threshold ({threshold:.3f})')
    ax.set_xlabel('Predicted probability')
    ax.set_ylabel('Density')
    ax.set_title(f'{model_name} — score distributions\n'
                 f'Separation = {aki_p.mean()-nonaki_p.mean():.3f}')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Score distributions → {save_path.name}")


# ─────────────────────────────────────────────────────────
#  Comparison table
# ─────────────────────────────────────────────────────────

def make_comparison_table(xgb_report: dict,
                           rf_report:  dict,
                           out_dir:    Path):
    """
    Print and save a side-by-side comparison table.
    Includes GOSH LSTM results for reference.
    """
    gosh_lstm = {
        'model':             'LSTM + attention (GOSH)',
        'window_auroc':      0.698,
        'window_auprc':      0.190,
        'auprc_vs_random':   2.2,
        'adm_sensitivity':   0.789,
        'adm_specificity':   0.749,
        'adm_ppv':           0.727,
        'adm_fa_ratio':      2.70,
        'detected_pct':      0.789,
        'lead_median_h':     51.8,
        'pct_detected_24h':  0.583,
        'pct_detected_48h':  0.423,
        'score_separation':  0.038,
    }

    rows = [gosh_lstm, xgb_report, rf_report]
    df   = pd.DataFrame(rows)

    cols_display = [
        'model', 'window_auroc', 'window_auprc', 'auprc_vs_random',
        'adm_sensitivity', 'adm_specificity', 'adm_ppv', 'adm_fa_ratio',
        'lead_median_h', 'pct_detected_24h', 'score_separation',
    ]
    df_disp = df[[c for c in cols_display if c in df.columns]]

    print(f"\n{'='*70}")
    print("MODEL COMPARISON — GOSH test set")
    print(f"{'='*70}")
    print(df_disp.to_string(index=False))

    df.to_csv(out_dir / 'comparison_table.csv', index=False)
    print(f"\n  Comparison table → {out_dir}/comparison_table.csv")


# ─────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)

    xgb_grid = XGB_GRID_FAST if args.fast else XGB_GRID
    rf_grid  = RF_GRID_FAST  if args.fast else RF_GRID

    print(f"\n{'='*55}")
    print("GOSH AKI PREDICTION — BASELINE MODELS")
    print(f"{'='*55}")
    print(f"Grid: {'FAST (2 configs)' if args.fast else f'full ({len(xgb_grid)} XGB, {len(rf_grid)} RF configs)'}")
    print(f"Folds: {N_FOLDS}")
    print(f"Output: {out_dir}")

    # ── Load data ──────────────────────────────────────────
    windows, feat_cols, _ = load_windows(
        args.windows, args.split_index
    )

    # ── Construct CV folds ────────────────────────────────
    # CRITICAL: folds only use train+val patients
    # test patients are isolated here and never touched until
    # final evaluation
    folds, pat_col = make_cv_folds(windows, N_FOLDS, RANDOM_SEED)

    print(f"\n  Test set (held out — never used in CV):")
    test_w = windows[windows['split'] == 'test']
    print(f"    {len(test_w):,} windows | "
          f"{test_w['label'].mean()*100:.2f}% positive")

    # ── XGBoost ───────────────────────────────────────────
    xgb_best_params, xgb_cv_df = cv_hyperparameter_search(
        windows, feat_cols, folds, pat_col,
        'XGBoost',
        lambda **kw: XGBClassifier(
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=RANDOM_SEED,
            n_jobs=args.n_jobs,
            verbosity=0,
            **kw
        ),
        xgb_grid,
        n_jobs=args.n_jobs,
    )

    # Rebuild factory with best params for final training
    xgb_cv_df.to_csv(out_dir / 'xgb_cv_results.csv', index=False)

    xgb_model, xgb_scaler, xgb_test_df, xgb_report = \
        train_final_and_evaluate(
            windows, feat_cols, pat_col,
            xgb_best_params,
            lambda **kw: XGBClassifier(
                eval_metric='logloss',
                use_label_encoder=False,
                random_state=RANDOM_SEED,
                n_jobs=args.n_jobs,
                verbosity=0,
                **kw
            ),
            'XGBoost',
        )

    with open(out_dir / 'xgb_best_config.json', 'w') as f:
        json.dump(xgb_best_params, f, indent=2)
    with open(out_dir / 'xgb_evaluation_report.json', 'w') as f:
        json.dump(xgb_report, f, indent=2)

    aki_ids_xgb = set(
        xgb_test_df[xgb_test_df['label']==1]['admission_id'].unique()
    )
    plot_score_distributions(
        xgb_test_df, aki_ids_xgb,
        xgb_report['best_threshold'],
        'XGBoost', plots_dir / 'xgb_score_distributions.png'
    )
    plot_feature_importance(
        xgb_model, feat_cols, 'XGBoost',
        plots_dir / 'xgb_feature_importance.png'
    )

    # ── Random Forest ─────────────────────────────────────
    rf_best_params, rf_cv_df = cv_hyperparameter_search(
        windows, feat_cols, folds, pat_col,
        'Random Forest',
        lambda **kw: RandomForestClassifier(
            random_state=RANDOM_SEED,
            n_jobs=args.n_jobs,
            **kw
        ),
        rf_grid,
        n_jobs=args.n_jobs,
    )

    rf_cv_df.to_csv(out_dir / 'rf_cv_results.csv', index=False)

    rf_model, rf_scaler, rf_test_df, rf_report = \
        train_final_and_evaluate(
            windows, feat_cols, pat_col,
            rf_best_params,
            lambda **kw: RandomForestClassifier(
                random_state=RANDOM_SEED,
                n_jobs=args.n_jobs,
                **kw
            ),
            'Random Forest',
        )

    with open(out_dir / 'rf_best_config.json', 'w') as f:
        json.dump(rf_best_params, f, indent=2)
    with open(out_dir / 'rf_evaluation_report.json', 'w') as f:
        json.dump(rf_report, f, indent=2)

    aki_ids_rf = set(
        rf_test_df[rf_test_df['label']==1]['admission_id'].unique()
    )
    plot_score_distributions(
        rf_test_df, aki_ids_rf,
        rf_report['best_threshold'],
        'Random Forest', plots_dir / 'rf_score_distributions.png'
    )
    plot_feature_importance(
        rf_model, feat_cols, 'Random Forest',
        plots_dir / 'rf_feature_importance.png'
    )

    # ── Comparison table ──────────────────────────────────
    make_comparison_table(xgb_report, rf_report, out_dir)

    # ── Final summary ─────────────────────────────────────
    print(f"\n{'='*55}")
    print("COMPLETE")
    print(f"{'='*55}")
    print(f"  XGBoost AUROC:  {xgb_report['window_auroc']:.3f}  "
          f"AUPRC: {xgb_report['window_auprc']:.3f}")
    print(f"  RF AUROC:       {rf_report['window_auroc']:.3f}  "
          f"AUPRC: {rf_report['window_auprc']:.3f}")
    print(f"  LSTM (ref):     AUROC 0.698  AUPRC 0.190")
    print(f"\n  All outputs → {out_dir}/")


if __name__ == '__main__':
    main()
