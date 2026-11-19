"""Run integrated-gradients explainability for EVERY sequence-model run and seed.

    python tools/explain_all.py                     # all runs, all seeds, each on its own site's test set
    python tools/explain_all.py --runs gosh_gru_all gosh_transformer_all --seeds 0 1
    python tools/explain_all.py --split holdout     # explain on the silent-deployment hold-out

Each run is a separate process (one crash or memory error does not stop the
rest) with its own log in logs/. Already-finished explanations are skipped,
so re-running after a crash continues where it stopped (use --redo to force).
Tabular baselines (logreg, hgb) are skipped: integrated gradients needs a
differentiable sequence model.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEQUENCE_MODELS = {"gru", "lstm", "transformer"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", help="run names under runs/ (default: all)")
    ap.add_argument("--seeds", nargs="*", type=int, help="default: all seeds found")
    ap.add_argument("--site", help="data site to explain on (default: the run's training site)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=200, help="windows per class to explain")
    ap.add_argument("--redo", action="store_true", help="re-run even if results exist")
    a = ap.parse_args()

    run_dirs = sorted((ROOT / "runs").glob("*/seed_*/run_config.json"))
    todo, skipped = [], []
    for rc_path in run_dirs:
        sd = rc_path.parent
        rc = json.loads(rc_path.read_text())
        if a.runs and sd.parent.name not in a.runs:
            continue
        if a.seeds is not None and rc.get("seed") not in a.seeds:
            continue
        if rc.get("model") not in SEQUENCE_MODELS:
            skipped.append(f"{sd.parent.name}/{sd.name} ({rc.get('model')}: not a sequence model)")
            continue
        todo.append((sd, a.site or rc["site"]))

    print(f"{len(todo)} run/seed folders to explain; {len(skipped)} skipped")
    for s in skipped:
        print("  skip:", s)
    failed = []
    for i, (sd, site) in enumerate(todo, 1):
        label = f"[{i}/{len(todo)}] {sd.parent.name}/{sd.name} on {site}/{a.split}"
        cmd = [sys.executable, "-m", "aki.cli", "explain", "--run", str(sd), "--site", site,
               "--split", a.split, "--n", str(a.n)]
        if not a.redo:
            cmd.append("--resume")
        t0 = time.time()
        print(label, "...", flush=True)
        r = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                           text=True)
        if r.returncode != 0:
            failed.append(sd)
            print(f"   FAILED (see logs/explain_*). Last error line: "
                  f"{(r.stderr.strip().splitlines() or ['?'])[-1]}")
        else:
            print(f"   ok ({time.time() - t0:.0f}s)")
    print(f"\nfinished: {len(todo) - len(failed)} ok, {len(failed)} failed")
    if failed:
        print("re-run the same command to retry the failures (finished ones are skipped)")
        sys.exit(1)


if __name__ == "__main__":
    main()