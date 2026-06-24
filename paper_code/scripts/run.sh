set -e 
echo "======================================"
echo "Training LSTM and GRU with attention"
echo "======================================"

echo ""

echo ">>>LSTM + FOCAL + Attention"
python train.py \
    --model       lstm \
    --loss        focal \
    --gamma       2.0 \
    --lr          0.0005 \
    --hidden      128 \
    --layers      2 \
    --dropout     0.3 \
    --batch       32 \
    --epochs      80 \
    --patience    15 \
    --max_seq_len 100 \
    --attention \
    --name        lstm_focal_h128_l2_attention

echo ""
echo ">>> GRU + Focal + Attention"
python train.py \
    --model       gru \
    --loss        focal \
    --gamma       2.0 \
    --lr          0.0005 \
    --hidden      128 \
    --layers      2 \
    --dropout     0.3 \
    --batch       32 \
    --epochs      80 \
    --patience    15 \
    --max_seq_len 100 \
    --attention \
    --name        gru_focal_h128_l2_attention

echo ">>> Experiment 3: LSTM + Weighted BCE + Attention"
python train.py \
    --model       lstm \
    --loss        bce_weighted \
    --lr          0.0005 \
    --hidden      128 \
    --layers      2 \
    --dropout     0.3 \
    --batch       32 \
    --epochs      80 \
    --patience    15 \
    --max_seq_len 100 \
    --attention \
    --name        lstm_bce_h128_l2_attention

# ── Experiment 4: GRU + BCE, higher gamma ──────────────
echo ""
echo ">>> Experiment 4: GRU + BCE + Attention"
python train.py \
    --model       gru \
    --loss        bce_weighted \
    --lr          0.0005 \
    --hidden      128 \
    --layers      2 \
    --dropout     0.3 \
    --batch       32 \
    --epochs      80 \
    --patience    15 \
    --max_seq_len 100 \
    --attention \
    --name        gru_bce_h128_l2_attention

# ── Experiment 5: Transformer + Focal ──────────────
echo ""
echo ">>> Experiment 5: Transformer + Focal "
python train.py \
    --model       transformer \
    --loss        focal \
    --lr          0.0005 \
    --hidden      128 \
    --layers      2 \
    --dropout     0.3 \
    --batch       32 \
    --epochs      80 \
    --patience    15 \
    --max_seq_len 100 \
    --attention \
    --name        transformer_focal

# ── Experiment 6: Transformer + BCE ──────────────
echo ""
echo ">>> Experiment 5: Transformer + BCE "
python train.py \
    --model       transformer \
    --loss        bce_weighted \
    --lr          0.0005 \
    --hidden      128 \
    --layers      2 \
    --dropout     0.3 \
    --batch       32 \
    --epochs      80 \
    --patience    15 \
    --max_seq_len 100 \
    --attention \
    --name        transformer_bce


echo ""
echo "============================================"
echo "Evaluating all experiments on test and val set..."
echo "============================================"



WINDOWS="data/processed/windows_matched.csv"
SPLIT="data/processed/split_index_matched.csv"
MODELS_DIR="data/models"
MIN_T=0.2
N_BOOT=${N_BOOT:-1000}   # override with N_BOOT=0 bash ... to skip

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

    # ── Val: compute and save threshold ──────────
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

echo "============"
echo "Traffic Light"
echo "============"

for run in lstm_focal_h128_l2_attention gru_focal_h128_l2_attention lstm_bce_h128_l2_attention gru_bce_h128_l2_attention transformer_focal transformer_bce; do
    echo "=== $run ==="
    python traffic_light_analysis.py --run $run
done



echo "======"
echo "Explainability (IG)"
echo "======="

for run in lstm_focal_h128_l2_attention gru_focal_h128_l2_attention lstm_bce_h128_l2_attention gru_bce_h128_l2_attention transformer_focal transformer_bce; do
    echo ">>> Explaining: ${run}"
    python gradient_explain.py \
        --run ${run} \
        --models_dir data/models \
        --windows data/processed/windows_matched.csv \
        --split_index data/processed/split_index_matched.csv \
        --split test
        --n_heatmaps 6
done

# Comparison table 
echo "========"
echo "Results comparison - all attention models"
python - << 'EOF'
import json
from pathlib import Path

runs = ['lstm_focal_h128_l2_attention', 'gru_focal_h128_l2_attention', 'lstm_bce_h128_l2_attention', 'gru_bce_h128_l2_attention', 'g]

print(f"\n{'Run': ,30} {'AUROC':.6} {'AUPRC':.6} "
    f"{'Sens':.6} {'Spec':.6} {'LeadMed':.8} {'Sep':6}")

print("-"*75)

for run in runs:
    path = Path(f'data/models/{run}/evaluation_report.json')
    if not path.exists():
        print(f"{run:,30} (no report found)")
        continue
    r = json.load(open(path))
    lead = f"{r['lead_median_h']:.0f}h" if r.get('lead_median_h') else "N/A"
    sep = f"{r.get('score_separation', 0):.3f}"
    print(f"{run:,30} "
        f"{r.get('window_auroc',0):6.3f} "
        f"{r.get('window_auprc',0):6.3f} "
        f"{r.get('adm_sensitivity',0):6.3f} "
        f"{r.get('adm_specificity',0):6.3f} "
        f"{lead:.8} "
        f"{sep:.6}")
EOF

echo ""
echo "All done."
echo "models saved to data/models/"
echo "Plots saved to data/models/<run>/plots/"
        
    





    