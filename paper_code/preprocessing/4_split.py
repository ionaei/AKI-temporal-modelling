"""
src/split.py

Patient-level stratified split for the AKI prediction windows dataset.

Usage (first run):
    windows = make_patient_splits(windows)
    save_splits(windows)

Usage (subsequent runs, after re-engineering features):
    train, val, test = load_splits(windows)
"""

import pandas as pd
import numpy as np
from pathlib import Path


def make_patient_splits(
    windows:     pd.DataFrame,
    test_frac:   float = 0.15,
    val_frac:    float = 0.15,
    random_seed: int   = 42,
) -> pd.DataFrame:
    """
    Assign a split label to every window, splitting by subject_id so that
    all windows from one patient always land in the same set.

    Stratified: AKI and non-AKI patients are split separately so each set
    has a representative positive rate.
    """
    if "label" not in windows.columns:
        raise ValueError("windows must contain a 'label' column")
    if "subject_id" not in windows.columns:
        raise ValueError("windows must contain a 'subject_id' column")

    # One row per patient: did they EVER develop AKI across all their windows?
    patient_summary = (
        windows.groupby("subject_id")["label"]
        .max()
        .reset_index()
        .rename(columns={"label": "has_aki"})
    )

    aki_patients    = patient_summary[patient_summary["has_aki"] == 1]["subject_id"].to_numpy().copy()
    nonaki_patients = patient_summary[patient_summary["has_aki"] == 0]["subject_id"].to_numpy().copy()

    rng = np.random.default_rng(random_seed)
    rng.shuffle(aki_patients)
    rng.shuffle(nonaki_patients)

    def split_array(arr: np.ndarray, label: str) -> tuple:
        n = len(arr)
        if n == 0:
            return np.array([]), np.array([]), np.array([])

        n_test = max(1, int(n * test_frac))
        n_val  = max(1, int(n * val_frac))

        # BUG FIX: ensure train is never empty
        if n_test + n_val >= n:
            raise ValueError(
                f"Too few {label} patients ({n}) for requested split fractions "
                f"(test={test_frac}, val={val_frac}). "
                f"Reduce fractions or check your dataset."
            )

        # BUG FIX: consistent, readable slice order
        test  = arr[:n_test]
        val   = arr[n_test : n_test + n_val]
        train = arr[n_test + n_val :]
        return train, val, test

    aki_tr,    aki_val,    aki_te    = split_array(aki_patients,    "AKI")
    nonaki_tr, nonaki_val, nonaki_te = split_array(nonaki_patients, "non-AKI")

    split_map: dict = {}
    for pid in aki_tr:     split_map[pid] = "train"
    for pid in aki_val:    split_map[pid] = "val"
    for pid in aki_te:     split_map[pid] = "test"
    for pid in nonaki_tr:  split_map[pid] = "train"
    for pid in nonaki_val: split_map[pid] = "val"
    for pid in nonaki_te:  split_map[pid] = "test"

    windows = windows.copy()
    windows["split"] = windows["subject_id"].map(split_map)

    # Catch any unmapped patients (shouldn't happen, but safe to check)
    unmapped = windows["split"].isna().sum()
    if unmapped > 0:
        raise ValueError(f"{unmapped} windows have no split assignment — check subject_id consistency")

    # Audit
    print("\nSplit summary:")
    print(f"{'set':6s}  {'windows':>8s}  {'patients':>9s}  {'pos rate':>9s}")
    print("-" * 40)
    for sp in ["train", "val", "test"]:
        sub      = windows[windows["split"] == sp]
        pos_rate = sub["label"].mean()
        n_pts    = sub["subject_id"].nunique()
        print(f"{sp:6s}  {len(sub):8d}  {n_pts:9d}  {pos_rate:9.3f}")

    return windows


def save_splits(
    windows: pd.DataFrame,
    out_dir: str = "splits",
) -> None:
    """
    Persist the split assignments to disk.

    Only the index mapping is saved (subject_id + admission_id + window_end → split).
    Full feature matrices are NOT saved here — they are always recomputed from
    windows.parquet and then re-joined with this index in load_splits().
    This avoids stale feature files if you re-run feature engineering.
    """
    if "split" not in windows.columns:
        raise ValueError("Run make_patient_splits() before save_splits()")

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # BUG FIX: save ONLY the index — not the full feature matrix
    split_index = windows[["subject_id", "admission_id", "window_end", "split"]].copy()

    # Normalise window_end to UTC-naive to avoid timezone merge mismatches on reload
    if pd.api.types.is_datetime64tz_dtype(split_index["window_end"]):
        split_index["window_end"] = split_index["window_end"].dt.tz_localize(None)

    out_path = Path(out_dir) / "split_index.parquet"
    split_index.to_parquet(out_path, index=False)
    print(f"Split index saved → {out_path}  ({len(split_index)} rows)")


def load_splits(
    windows: pd.DataFrame,
    out_dir: str = "splits",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Re-apply saved split assignments to a (possibly updated) windows DataFrame.

    Merges on (subject_id, admission_id, window_end) so the mapping is robust
    to row reordering.  Raises if any windows cannot be matched — fail loudly
    rather than silently dropping rows.
    """
    index_path = Path(out_dir) / "split_index.parquet"
    if not index_path.exists():
        raise FileNotFoundError(
            f"No split index found at {index_path}. "
            "Run make_patient_splits() + save_splits() first."
        )

    split_index = pd.read_parquet(index_path)

    # BUG FIX: normalise window_end timezone on both sides before merging
    windows = windows.copy()
    for df in [windows, split_index]:
        if pd.api.types.is_datetime64tz_dtype(df["window_end"]):
            df["window_end"] = df["window_end"].dt.tz_localize(None)

    merge_keys = ["subject_id", "admission_id", "window_end"]
    merged = windows.merge(
        split_index[merge_keys + ["split"]],
        on=merge_keys,
        how="left",
    )

    # BUG FIX: raise on missing assignments rather than silently losing rows
    missing = merged["split"].isna().sum()
    if missing > 0:
        raise ValueError(
            f"{missing} windows could not be matched to a split assignment.\n"
            "This usually means window_end timestamps differ between the saved "
            "index and the recomputed windows (e.g. rounding, timezone change). "
            "Delete the split index and re-run make_patient_splits()."
        )

    train = merged[merged["split"] == "train"].copy()
    val   = merged[merged["split"] == "val"].copy()
    test  = merged[merged["split"] == "test"].copy()

    print(f"Loaded splits — train: {len(train)}, val: {len(val)}, test: {len(test)}")
    return train, val, test

   
