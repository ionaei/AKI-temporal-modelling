#!/usr/bin/env bash
# Smoke test of the whole pipeline on synthetic data (minutes, CPU only).
set -euo pipefail
python tools/make_synthetic.py
for S in synth_gosh synth_mimic; do
  for st in extract label windows split; do python -m aki.cli $st --site $S; done
done
python -m aki.cli core --sites synth_gosh synth_mimic
python -m aki.cli train --site synth_gosh --model gru --name synth_gru --seeds 0 1
python -m aki.cli evaluate --run runs/synth_gru --site synth_gosh --split test --n_boot 20
python -m aki.cli aggregate --run runs/synth_gru
python -m aki.cli train --site synth_gosh --model transformer --name synth_tf --seeds 0
python -m aki.cli train --site synth_gosh --model gru --name synth_gru_core \
    --core configs/core_features_synth_gosh_synth_mimic.json
python -m aki.cli evaluate --run runs/synth_gru_core --site synth_mimic --split test --n_boot 0 \
    --recalibrate_on_target_val
python -m aki.cli explain --run runs/synth_gru/seed_0 --site synth_gosh --split test --n 30
for st in extract label windows; do python -m aki.cli $st --site synth_gosh_silent; done
python -m aki.cli deploy --run runs/synth_gru/seed_0 --site synth_gosh_silent --split all --n_boot 0
python -m aki.cli deploy --run runs/synth_gru/seed_0 --site synth_gosh --split holdout --n_boot 0
echo "SYNTHETIC SMOKE TEST PASSED"
