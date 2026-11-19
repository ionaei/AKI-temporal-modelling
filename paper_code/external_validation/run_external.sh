#!/usr/bin/env bash
# Zero-shot external validation of GOSH-trained models on PIC (v1.1.0) and the neonatal
# cohort of MIMIC-III (v1.4). Run from external_validation/ where the PhysioNet data live.
#
# Before running:
#   1. set raw_dir in configs/sites/pic.yaml and configs/sites/mimic3.yaml;
#   2. copy the trained GOSH runs (exported from the GOSH DRE) into runs/:
#        runs/gosh_transformer_focal_all/      selected model (Transformer, focal loss)
#        runs/gosh_logreg_all/                 logistic-regression baseline
#        runs/gosh_transformer_focal_no_process/
#        runs/gosh_logreg_no_process/          both without measurement-process features
#      Each seed folder needs model.pt + model_meta.json (or model.joblib), preprocessor.joblib,
#      platt.json, thresholds.json and run_config.json. No GOSH patient data are needed.
set -euo pipefail
SITES="pic mimic3"
RUNS="gosh_transformer_focal_all gosh_logreg_all gosh_transformer_focal_no_process gosh_logreg_no_process"

# ── 1. data: raw tables -> canonical tables -> KDIGO labels -> 6-h windows -> splits ──
# The split (70/15/15 by patient) defines the external validation and test partitions used
# for local recalibration; the 'all' evaluation uses every eligible stay.
for S in $SITES; do
  for st in extract label windows split; do
    python -m aki.cli $st --resume --site $S
  done
done

# ── 2. evaluation with model weights frozen ──
#   --split all                       GOSH calibration and threshold, all stays
#   --split test                      GOSH calibration and threshold, test partition
#   --split test --recalibrate_on_target_val
#                                     calibration and threshold re-derived on the external
#                                     validation partition, applied to the test partition
for S in $SITES; do
  for R in $RUNS; do
    python -m aki.cli evaluate --resume --run runs/$R --site $S --split all  --n_boot 200
    python -m aki.cli evaluate --resume --run runs/$R --site $S --split test --n_boot 200
    python -m aki.cli evaluate --resume --run runs/$R --site $S --split test --n_boot 200 \
        --recalibrate_on_target_val
  done
done

# ── 3. tables ──
# Table 4 and Supplementary Table S6: mean over seeds with seed-averaged bootstrap CIs
python tools/seed_summary.py --runs $RUNS --op orig_spec_at_sens0.75
python tools/compare_runs.py --per_seed
# Paired comparison of the Transformer with logistic regression on the same stays
python tools/paired_compare.py --a gosh_transformer_focal_all --b gosh_logreg_all \
    --evals eval_pic_all eval_mimic3_all eval_pic_test_recal eval_mimic3_test_recal --n_boot 200

# ── 4. dataset shift: feature-level shift of each cohort relative to GOSH ──
for S in $SITES; do
  python tools/shift_diagnostics.py --run runs/gosh_logreg_all/seed_0 --site $S
done

# ── 5. cohort description ──
python tools/table1.py pic:all mimic3:all
python tools/ethnicity_table.py --site mimic3 --file /path/to/mimiciii/1.4/ADMISSIONS.csv.gz \
    --subject_col SUBJECT_ID --eth_col ETHNICITY --columns all
python tools/excluded_profile.py --site pic
python tools/excluded_profile.py --site mimic3
python tools/cluster_dataset.py --site pic    --name PIC
python tools/cluster_dataset.py --site mimic3 --name MIMIC-III
