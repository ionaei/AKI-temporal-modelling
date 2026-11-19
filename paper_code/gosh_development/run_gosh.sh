#!/usr/bin/env bash
# Full GOSH workflow: data preparation, model training and evaluation on the test set,
# temporal hold-out, feature ablations and explainability.
# Run from gosh_development/ inside the GOSH DRE.
#
# Safe to re-run after a crash: --resume skips everything already finished and
# continues an interrupted model from its last completed epoch.
# To force a stage to run again, delete its output (or run it by hand without --resume).
set -euo pipefail
S=gosh
SEEDS="0 1 2 3 4"
LOSSES="bce focal"          # every sequence model is trained with both losses
MODELS="gru lstm transformer"

# ── 1. data: raw extract -> canonical tables -> KDIGO labels -> 6-h windows -> splits ──
python -m aki.cli extract --resume --site $S
python -m aki.cli label   --resume --site $S
python -m aki.cli windows --resume --site $S
python -m aki.cli split   --resume --site $S      # last 6 months -> 'holdout' (see split_meta.json)

# ── 2. sequence models: all feature groups, both losses ──
for L in $LOSSES; do
  for M in $MODELS; do
    R=${S}_${M}_${L}_all
    python -m aki.cli train     --resume --site $S --model $M --loss $L --name $R --seeds $SEEDS
    python -m aki.cli evaluate  --resume --run runs/$R --site $S --split test
    python -m aki.cli aggregate --resume --run runs/$R
  done
done

# ── 3. non-temporal baselines: logistic regression and gradient boosting ──
for M in logreg hgb; do
  R=${S}_${M}_all
  python -m aki.cli train     --resume --site $S --model $M --name $R --seeds $SEEDS
  python -m aki.cli evaluate  --resume --run runs/$R --site $S --split test
  python -m aki.cli aggregate --resume --run runs/$R
done

# ── 4. temporal hold-out: frozen model, calibration and thresholds ──
for L in $LOSSES; do
  for M in $MODELS; do
    for SEED in $SEEDS; do
      python -m aki.cli deploy --resume --run runs/${S}_${M}_${L}_all/seed_$SEED \
          --site $S --split holdout
    done
  done
done
for M in logreg hgb; do
  for SEED in $SEEDS; do
    python -m aki.cli deploy --resume --run runs/${S}_${M}_all/seed_$SEED --site $S --split holdout
  done
done

# ── 5. feature ablations: every sequence model, both losses ──
#   no_process     all groups except the measurement process
#   process_only   measurement process, static and time features only
#   no_creatinine  all groups except creatinine-derived features
for L in $LOSSES; do
  for M in $MODELS; do
    python -m aki.cli train --resume --site $S --model $M --loss $L --seeds $SEEDS \
        --name ${S}_${M}_${L}_no_process    --groups static time creatinine lab vital support med
    python -m aki.cli train --resume --site $S --model $M --loss $L --seeds $SEEDS \
        --name ${S}_${M}_${L}_process_only  --groups static time process
    python -m aki.cli train --resume --site $S --model $M --loss $L --seeds $SEEDS \
        --name ${S}_${M}_${L}_no_creatinine --groups static time lab vital support med process
    for A in no_process process_only no_creatinine; do
      python -m aki.cli evaluate  --resume --run runs/${S}_${M}_${L}_$A --site $S --split test
      python -m aki.cli aggregate --resume --run runs/${S}_${M}_${L}_$A
    done
  done
done
# logistic regression without measurement-process features (also evaluated on PIC / MIMIC-III)
python -m aki.cli train     --resume --site $S --model logreg --seeds $SEEDS \
    --name ${S}_logreg_no_process --groups static time creatinine lab vital support med
python -m aki.cli evaluate  --resume --run runs/${S}_logreg_no_process --site $S --split test
python -m aki.cli aggregate --resume --run runs/${S}_logreg_no_process

# ── 6. explainability (integrated gradients) for every sequence model, then one report ──
python tools/explain_all.py
python tools/compare_runs.py --site $S
echo "run_gosh.sh finished: open results/report.html"
# Tables and figures of the paper for the selected model: see run_paper_analyses.sh
