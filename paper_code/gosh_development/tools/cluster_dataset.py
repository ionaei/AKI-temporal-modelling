"""Cluster the ICU stays of ONE dataset (run in the environment where that dataset lives).

    python tools/cluster_dataset.py --site gosh   --name GOSH
    python tools/cluster_dataset.py --site pic    --name PIC
    python tools/cluster_dataset.py --site mimic3 --name MIMIC-III

Needs data/<site>/windows/windows.parquet and feature_groups.json (written by `aki.cli windows`).
  1. features = groups --groups (default creatinine lab med static: clinical state and age, not
     monitoring intensity). To avoid counting one variable several times, the derived variants
     named in --drop_suffix (default _min _max _delta _hsince _days72h) are dropped, leaving one
     column per lab (the last value) and one per drug (exposure). Features observed in < 5% of
     stays, not constant, or non-zero in < 2% of stays (rare drugs) are dropped;
  2. one vector per ICU stay = mean over the stay's windows, z-scored WITHIN this dataset,
     clipped to +-5 SD (missing -> 0), PCA(30);
  3. k-means on the PCA scores and t-SNE (2 D) on a random sample of <= --max_stays stays;
  4. k: --k auto (default) takes the k = 2..8 with the highest silhouette score; give a number to
     override. A silhouette below 0.25 means weak structure and a warning is printed.
Each dataset is clustered on its own: cluster numbers and t-SNE axes mean nothing across datasets.
Outputs in results/clusters/<site>/ (export these):
  embedding.csv         site, aki, severe, x, y, cluster   (no identifiers; one row per sampled stay)
  cluster_summary.csv   stays and AKI rate per cluster (counts < 5 suppressed)
  cluster_profiles.csv  mean z-score of every feature in every cluster (what defines each cluster)
  features_used.txt
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parents[1]
MIN_CELL = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    ap.add_argument("--name", help="display name (default: the site name)")
    ap.add_argument("--groups", nargs="+", default=["creatinine", "lab", "med", "static"])
    ap.add_argument("--k", default="auto", help="number of clusters, or 'auto' (best silhouette)")
    ap.add_argument("--drop_suffix", nargs="*",
                    default=["_min", "_max", "_delta", "_hsince", "_days72h"],
                    help="derived variants to drop (pass --drop_suffix with no value to keep all)")
    ap.add_argument("--clip", type=float, default=5.0, help="clip z-scores at +-this many SD")
    ap.add_argument("--min_nonzero", type=float, default=0.02,
                    help="drop features that are non-zero in fewer than this share of stays")
    ap.add_argument("--max_stays", type=int, default=2500)
    ap.add_argument("--perplexity", type=float, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    name = a.name or a.site
    out = Path(a.out) if a.out else ROOT / "results" / "clusters" / a.site
    out.mkdir(parents=True, exist_ok=True)
    wdir = ROOT / "data" / a.site / "windows"

    fg = json.loads((wdir / "feature_groups.json").read_text())
    cols = sorted(f for f, g in fg.items()
                  if g in a.groups and not f.endswith(tuple(a.drop_suffix)))
    w = pd.read_parquet(wdir / "windows.parquet", columns=["stay_id", "aki", "stage_peak"] + cols)
    grp = w.groupby("stay_id")
    X, meta = grp[cols].mean(), grp[["aki", "stage_peak"]].first()
    keep = [f for f in cols if X[f].notna().mean() >= 0.05 and X[f].nunique() > 1
            and (X[f].fillna(0) != 0).mean() >= a.min_nonzero]
    print(f"{name}: {len(X):,} stays ({int(meta['aki'].sum()):,} with AKI); "
          f"{len(keep)} of {len(cols)} features used (groups {a.groups})")
    (out / "features_used.txt").write_text("\n".join(keep))

    rng = np.random.default_rng(a.seed)
    sel = rng.choice(len(X), min(len(X), a.max_stays), replace=False)
    Xs, ms = X[keep].iloc[sel], meta.iloc[sel]
    Z = np.nan_to_num(((Xs - Xs.mean()) / Xs.std().replace(0, 1).fillna(1)).to_numpy(float))
    Z = np.clip(Z, -a.clip, a.clip)                       # rare exposures must not dominate
    Zp = PCA(min(30, len(keep), len(Z) - 1), random_state=a.seed).fit_transform(Z)

    print("silhouette by k (higher = better separated clusters):")
    ss = {}
    for k in range(2, 9):
        lab = KMeans(k, n_init=10, random_state=a.seed).fit_predict(Zp)
        ss[k] = silhouette_score(Zp, lab, sample_size=min(2000, len(Zp)), random_state=a.seed)
    k_use = max(ss, key=ss.get) if str(a.k) == "auto" else int(a.k)
    print("  " + "  ".join(f"k={k}: {v:.2f}" for k, v in ss.items()) + f"   (using k={k_use})")
    if max(ss.values()) < 0.25:
        print("  WARNING: silhouette < 0.25 for every k: weak cluster structure. Treat the groups "
              "as strata along a continuum, not as distinct phenotypes.")
    cluster = KMeans(k_use, n_init=10, random_state=a.seed).fit_predict(Zp)
    xy = TSNE(n_components=2, perplexity=min(a.perplexity, (len(Z) - 1) / 3), init="pca",
              learning_rate="auto", random_state=a.seed).fit_transform(Zp)
    emb = pd.DataFrame({"site": name, "aki": ms["aki"].astype(int).to_numpy(),
                        "severe": (ms["stage_peak"] >= 2).astype(int).to_numpy(),
                        "x": xy[:, 0].round(2), "y": xy[:, 1].round(2), "cluster": cluster})
    emb.to_csv(out / "embedding.csv", index=False)

    rows = []
    for c, g in emb.groupby("cluster"):
        n_aki = int(g["aki"].sum())
        small = 0 < n_aki < MIN_CELL
        rows.append({"cluster": c, "n_stays": len(g), "n_aki": "<5" if small else n_aki,
                     "aki_rate_%": "" if small else round(100 * g["aki"].mean(), 1)})
    summ = pd.DataFrame(rows)
    summ.to_csv(out / "cluster_summary.csv", index=False)
    prof = pd.DataFrame(Z, columns=keep).groupby(cluster).mean().round(2)
    prof.index.name = "cluster"
    prof.to_csv(out / "cluster_profiles.csv")

    pd.set_option("display.width", 200)
    print("\n" + summ.to_string(index=False))
    print("\nwhat defines each cluster (largest mean z-scores):")
    for c, r in prof.iterrows():
        top = r.reindex(r.abs().sort_values(ascending=False).index)[:4]
        print(f"  cluster {c}: " + ", ".join(f"{f} {v:+.1f}" for f, v in top.items()))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()