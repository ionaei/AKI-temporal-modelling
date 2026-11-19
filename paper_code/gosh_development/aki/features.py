"""Feature-set selection and preprocessing fitted on the TRAINING split only.

Steps (all fitted on train, applied unchanged to val/test/other sites):
 1. choose columns: by group (for ablations) and optionally restricted to
    the cross-site 'core' set (features available at every listed site);
 2. age-band normalisation for age-dependent physiology (HR, RR, BP,
    creatinine...): value -> (value - band median) / band robust SD;
 3. standardisation (mean/SD) and clipping to +-clip;
 4. missing -> 0 (= training mean after scaling). Missingness is NOT
    re-encoded here; measurement process is only visible through the
    explicit `process` group, so it can be ablated cleanly.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .concepts import NORM_BANDS_DAYS, age_band

log = logging.getLogger(__name__)

ALL_GROUPS = ["static", "time", "creatinine", "lab", "vital", "support", "med", "process"]


def select_features(groups: dict, include: list[str], core: list[str] | None = None,
                    drop: list[str] | None = None) -> list[str]:
    cols = [c for c, g in groups.items() if g in include]
    if core is not None:
        core_set = set(core)
        cols = [c for c in cols if c in core_set]
    if drop:
        cols = [c for c in cols if c not in set(drop)]
    return sorted(cols)


def core_feature_set(availability_files: list[str | Path], min_non_null: float,
                     min_nonzero_binary: float) -> list[str]:
    """Features present (non-trivially) at EVERY listed site."""
    sets = []
    for f in availability_files:
        a = pd.read_csv(f)
        ok = (a["n_unique"] > 1) & (a["non_null"] >= min_non_null)
        # exposure-type features must actually be non-zero often enough; derived
        # medication features (_days72h, _hsince) follow their base flag
        base = a["feature"].str.replace(r"_(days72h|hsince)$", "", regex=True)
        nz = a.set_index("feature")["nonzero"]
        base_nz = base.map(nz).fillna(a["nonzero"])
        binary = a["feature"].str.startswith(("med_", "sup_vent"))
        ok &= ~binary | (base_nz.to_numpy() >= min_nonzero_binary)
        sets.append(set(a.loc[ok, "feature"]))
    core = sorted(set.intersection(*sets)) if sets else []
    log.info("core feature set across %d sites: %d features", len(sets), len(core))
    return core


class Preprocessor:
    def __init__(self, columns: list[str], age_norm_prefixes: list[str], clip: float = 10.0):
        self.columns = list(columns)
        self.age_norm_cols = [c for c in self.columns
                              if any(c.startswith(p) for p in age_norm_prefixes)
                              and not c.endswith("_delta")]
        self.clip = clip

    @staticmethod
    def _band(df):
        age_now = df["age_days"].to_numpy() + df["hours_since_admit"].to_numpy() / 24
        return np.asarray(age_band(age_now, NORM_BANDS_DAYS).astype(str))

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        X = df[self.columns].astype(float)
        band = self._band(df)
        self.band_stats = {}
        for c in self.age_norm_cols:
            stats = {}
            glob_med = X[c].median()
            glob_sd = (X[c].quantile(0.75) - X[c].quantile(0.25)) / 1.349
            for b in np.unique(band):
                x = X.loc[band == b, c].dropna()
                if len(x) >= 50:
                    sd = (x.quantile(0.75) - x.quantile(0.25)) / 1.349
                    stats[b] = (float(x.median()), float(sd if sd > 0 else glob_sd or 1.0))
            stats["_global"] = (float(glob_med) if pd.notna(glob_med) else 0.0,
                                float(glob_sd) if pd.notna(glob_sd) and glob_sd > 0 else 1.0)
            self.band_stats[c] = stats
        X = self._age_norm(X, band)
        self.mean = X.mean().fillna(0)
        self.sd = X.std().replace(0, 1).fillna(1)
        return self

    def _age_norm(self, X, band):
        X = X.copy()
        b = pd.Series(band)
        for c in self.age_norm_cols:
            st = self.band_stats[c]
            g_med, g_sd = st["_global"]
            med = b.map({k: v[0] for k, v in st.items()}).fillna(g_med).to_numpy()
            sd = b.map({k: v[1] for k, v in st.items()}).fillna(g_sd).to_numpy()
            X[c] = (X[c].to_numpy() - med) / sd
        return X

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.columns if c not in df.columns]
        if missing:
            log.warning("%d features absent in this data -> filled with training mean: %s",
                        len(missing), missing[:10])
            df = df.copy()
            for c in missing:
                df[c] = np.nan
        X = df[self.columns].astype(float)
        X = self._age_norm(X, self._band(df))
        X = (X - self.mean) / self.sd
        return X.clip(-self.clip, self.clip).fillna(0).to_numpy(np.float32)

    def save(self, path: Path):
        joblib.dump(self, path)
        Path(path).with_suffix(".columns.json").write_text(json.dumps(self.columns, indent=1))

    @staticmethod
    def load(path: Path) -> "Preprocessor":
        return joblib.load(path)
