#!/bin/bash
# evaluate_all_gosh.sh
#
# Runs val then test evaluation for all three GOSH models.
# Val must run before test — it computes and saves the
# Youden-J threshold that test will load.
#
# Usage
# -----
#   bash evaluate_min_sensitivity.sh
#
# To skip bootstrap (faster):
#   N_BOOT=0 bash evaluate_min_sensitivity.sh

set -e   # exit immediately on any error

WINDOWS="data/processed/windows_matched.csv"
SPLIT="data/processed/split_index_matched.csv"
MODELS_DIR="data/models"
MIN_T=0.2
N_BOOT=0 #${N_BOOT:-1000}    override with N_BOOT=0 bash ... to skip

RUNS=(
    "lstm_focal_h128_l2_attention"
    "gru_focal_h128_l2_attention"
    "lstm_bce_h128_l2_attention"
    "gru_bce_h128_l2_attention"
    "transformer_focal"
    "transformer_bce"
)

echo "=================================================="
echo "GOSH MODEL EVALUATION"
echo "  Models dir: $MODELS_DIR"
echo "  Bootstrap:  $N_BOOT iterations"
echo "=================================================="

for RUN in "${RUNS[@]}"; do
    echo ""
    echo "══════════════════════════════════════════════════"
    echo "  $RUN"
    echo "══════════════════════════════════════════════════"

    # ── Val: compute and save Youden-J threshold ──────────
    echo ""
    echo "  [1/2] Val set — computing threshold..."
    python evaluate.py \
        --run           "$RUN" \
        --models_dir    "$MODELS_DIR" \
        --windows       "$WINDOWS" \
        --split_index   "$SPLIT" \
        --split         val \
        --min_threshold "$MIN_T" \
        --n_boot        0

    # ── Test: load val threshold + bootstrap CIs ──────────
    echo ""
    echo "  [2/2] Test set — evaluating with val threshold + bootstrap..."
    python evaluate.py \
        --run           "$RUN" \
        --models_dir    "$MODELS_DIR" \
        --windows       "$WINDOWS" \
        --split_index   "$SPLIT" \
        --split         test \
        --min_threshold "$MIN_T" \
        --n_boot        "$N_BOOT"

    echo ""
    echo "  ✓ $RUN complete"
    echo "    Reports → $MODELS_DIR/$RUN/evaluation_report_val.json"
    echo "             $MODELS_DIR/$RUN/evaluation_report_test.json"
    echo "             $MODELS_DIR/$RUN/bootstrap_ci.json"
done

echo ""
echo "=================================================="
echo "ALL DONE"
echo "=================================================="
echo ""
echo "Summary of test set thresholds used:"
for RUN in "${RUNS[@]}"; do
    THRESH=$(python -c "
import json
with open('$MODELS_DIR/$RUN/val_threshold.json') as f:
    print(json.load(f)['threshold'])
" 2>/dev/null || echo "N/A")
    echo "  $RUN: tau = $THRESH"
done