#!/usr/bin/env bash
# Tables and figures of the paper that are computed at GOSH, for the selected model
# (Transformer, focal loss: runs/gosh_transformer_focal_all). Run after run_gosh.sh,
# from gosh_development/ inside the GOSH DRE. Every tool writes aggregates only to
# results/ (apply the DRE small-number rules before export).
set -euo pipefail
S=gosh
SEL=gosh_transformer_focal_all        # selected model (highest mean validation AUPRC)
LR=gosh_logreg_all                    # logistic-regression baseline
DEMOGRAPHICS=/path/to/gosh/demographics.csv   # raw file with patient id and ethnicity

# Table 1: cohort characteristics (train/val and test columns)
python tools/table1.py $S:train,val $S:test
python tools/ethnicity_table.py --site $S --file $DEMOGRAPHICS \
    --subject_col project_id --eth_col ethnicity_name --columns "train,val" "test"

# Supplementary Table S2: stays excluded for AKI before admission or within 24 h
python tools/excluded_profile.py --site $S --eth_file $DEMOGRAPHICS \
    --subject_col project_id --eth_col ethnicity_name

# Supplementary Figure S2: label validation (XGBoost on pre-onset features)
python tools/label_validation.py --site $S

# Supplementary Figure S3: clustering of ICU stays (t-SNE, k-means)
python tools/cluster_dataset.py --site $S --name GOSH

# Table 2, Supplementary Tables S3 and S4: all runs, mean ± SD over seeds
python tools/compare_runs.py --site $S

# Figure 2: early-warning curves (look-back sensitivity, alert burden, precision)
python tools/alert_curves.py --runs $SEL $LR --evals eval_gosh_test deploy_gosh_holdout

# Table 3: performance by AKI stage
python tools/stage_performance.py --run $SEL --evals eval_gosh_test --n_boot 1000

# Paired comparison of the selected model with logistic regression
python tools/paired_compare.py --a $SEL --b $LR --evals eval_gosh_test deploy_gosh_holdout --n_boot 1000

# Figure 3: traffic-light risk stratification (thresholds chosen on validation)
python tools/traffic_light.py --run runs/$SEL --apply test holdout

# Renal Angina Index: as published, and head to head with the model on unseen stays
python tools/renal_angina.py --site $S --run runs/$SEL --splits test holdout --n_boot 1000

# Figure 4 and Supplementary Table S5: temporal hold-out by month
#   monthly metrics: runs/$SEL/seed_*/deploy_gosh_holdout/monthly.csv (written by run_gosh.sh),
#   collected over seeds in results/deploy_monthly.csv by compare_runs.py
python tools/holdout_demographics.py --site $S
python tools/month_diagnostics.py --run runs/$SEL --site $S --month 2026-02

# Figure 5: integrated-gradients importance difference (AKI minus non-AKI)
#   runs/$SEL/seed_*/explain_gosh_test/ig_by_feature_signed.csv (column importance_diff),
#   collected over seeds in results/explain_differential.csv by compare_runs.py
