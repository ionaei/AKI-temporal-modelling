set -e 
echo "======================================"
echo "Bootstrap GRU BCE"
echo "======================================"

echo ""



python bootstrap_ci.py \
        --predictions data/models/gru_bce_h128_l2_attention/test_predictions.csv \
        --out         data/models/gru_bce_h128_l2_attention/bootstrap_ci.json \
        --n_boot      1000 \
        --threshold   0.61 \


echo "======================================"
echo "Bootstrap LSTM BCE"
echo "======================================"

echo ""



python bootstrap_ci.py \
        --predictions data/models/lstm_bce_h128_l2_attention/test_predictions.csv \
        --out         data/models/lstm_bce_h128_l2_attention/bootstrap_ci.json \
        --n_boot      1000 \
        --threshold   0.74 \

echo "======================================"
echo "Bootstrap Transformer BCE"
echo "======================================"

echo ""



python bootstrap_ci.py \
        --predictions data/models/transformer_bce/test_predictions.csv \
        --out         data/models/transformer_bce/bootstrap_ci.json \
        --n_boot      1000 \
        --threshold   0.69 \


done