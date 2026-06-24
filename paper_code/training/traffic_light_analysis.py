"""
src/traffic_light_analysis.py

Complete traffic light analysis for AKI temporal prediction.

What this does
--------------
1. Loads val + test predictions from a trained run
2. Optimises amber and red thresholds on the VALIDATION set using:
      - Constraints on RED rate (e.g. 10-20% of admissions)
      - Constraints on GREEN rate (e.g. >= 50% of admissions)
      - Maximises Youden's J on GREEN vs RED (amber excluded)
3. Applies optimised thresholds to the TEST set
4. Analyses amber transitions — how amber patients evolve over time
5. Produces all plots and a summary report

Usage (on already-saved model)
-------------------------------
    python UPDATED_traffic_light_analysis.py --run transformer_bce

    # With custom constraints (adjust based on your PICU context)
    python src/traffic_light_analysis.py \\
        --run          MY_RUN_NAME \\
        --red_min      0.10 \\
        --red_max      0.20 \\
        --green_min    0.50 \\
        --resolution   0.05

    # If predictions not yet saved, generate them first:
    python src/evaluate.py --run MY_RUN_NAME --split val
    python src/evaluate.py --run MY_RUN_NAME --split test
    # Then run this script

Outputs (saved to data/models/MY_RUN/plots/)
--------------------------------------------
    threshold_optimisation.png     heatmap of Youden J across threshold sweep
    traffic_light_distribution.png score distributions with optimised thresholds
    traffic_light_population.png   % RED/AMBER/GREEN by AKI status
    traffic_light_trajectories.png per-patient risk curves coloured by state
    amber_transitions.png          how amber patients evolve
    traffic_light_report.json      all numbers in one place
"""

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── Colour scheme ─────────────────────────────────────────
COLOURS = {
    'green': '#2ecc71',
    'amber': '#f39c12',
    'red':   '#e74c3c',
    'grey':  '#bdc3c7',
}

TL_ORDER = ['GREEN', 'AMBER', 'RED']


# ─────────────────────────────────────────────────────────
#  Helper: assign traffic light state
# ─────────────────────────────────────────────────────────

def assign_tl(prob: float, amber_t: float, red_t: float) -> str:
    if prob >= red_t:
        return 'RED'
    elif prob >= amber_t:
        return 'AMBER'
    else:
        return 'GREEN'


def add_tl_col(df: pd.DataFrame, amber_t: float, red_t: float) -> pd.DataFrame:
    df = df.copy()
    df['traffic_light'] = df['prob'].apply(
        assign_tl, amber_t=amber_t, red_t=red_t
    )
    return df


def adm_max_tl(preds_df: pd.DataFrame,
               aki_ids:  set,
               amber_t:  float,
               red_t:    float) -> pd.DataFrame:
    """
    Classify each admission by its maximum predicted probability.
    One row per admission with columns: admission_id, max_prob,
    is_aki, traffic_light.
    """
    adm = (
        preds_df.groupby('admission_id')['prob']
        .max()
        .reset_index()
        .rename(columns={'prob': 'max_prob'})
    )
    adm['is_aki'] = adm['admission_id'].isin(aki_ids)
    adm['traffic_light'] = adm['max_prob'].apply(
        assign_tl, amber_t=amber_t, red_t=red_t
    )
    return adm


# ─────────────────────────────────────────────────────────
#  1. Threshold optimisation
# ─────────────────────────────────────────────────────────

def compute_tl_metrics(preds_df: pd.DataFrame,
                        aki_ids:  set,
                        amber_t:  float,
                        red_t:    float) -> dict:
    """
    Compute traffic light metrics for one (amber, red) threshold pair.

    Classification is based on each admission's MAXIMUM probability
    across all windows — this is what a clinician sees: the worst
    risk score during the admission.

    Youden J is computed on GREEN vs RED only.
    Amber admissions are excluded from J — they are the uncertain zone
    and including them would distort the sensitivity/specificity calculation.
    """
    adm = adm_max_tl(preds_df, aki_ids, amber_t, red_t)
    total   = len(adm)
    n_red   = (adm['traffic_light'] == 'RED').sum()
    n_amber = (adm['traffic_light'] == 'AMBER').sum()
    n_green = (adm['traffic_light'] == 'GREEN').sum()

    # Youden J on definitive cases only (exclude amber)
    definitive = adm[adm['traffic_light'] != 'AMBER']
    if len(definitive) < 2 or definitive['is_aki'].nunique() < 2:
        sensitivity = specificity = youden_j = np.nan
    else:
        tp = ((definitive['is_aki']) & (definitive['traffic_light'] == 'RED')).sum()
        fn = ((definitive['is_aki']) & (definitive['traffic_light'] == 'GREEN')).sum()
        tn = ((~definitive['is_aki']) & (definitive['traffic_light'] == 'GREEN')).sum()
        fp = ((~definitive['is_aki']) & (definitive['traffic_light'] == 'RED')).sum()
        sensitivity = tp / (tp + fn + 1e-9)
        specificity = tn / (tn + fp + 1e-9)
        youden_j    = sensitivity + specificity - 1

    return {
        'amber_threshold': round(amber_t, 4),
        'red_threshold':   round(red_t,   4),
        'n_red':    int(n_red),
        'n_amber':  int(n_amber),
        'n_green':  int(n_green),
        'red_rate':   round(n_red   / total, 4),
        'amber_rate': round(n_amber / total, 4),
        'green_rate': round(n_green / total, 4),
        'sensitivity': round(float(sensitivity), 4) if not np.isnan(sensitivity) else None,
        'specificity': round(float(specificity), 4) if not np.isnan(specificity) else None,
        'youden_j':    round(float(youden_j),    4) if not np.isnan(youden_j)    else None,
    }


def optimise_thresholds(preds_df:       pd.DataFrame,
                         aki_ids:        set,
                         red_rate_min:   float = 0.10,
                         red_rate_max:   float = 0.20,
                         green_rate_min: float = 0.50,
                         resolution:     float = 0.05) -> tuple:
    """
    Sweep all (amber, red) threshold pairs at given resolution.
    Select the pair that maximises Youden's J subject to:
        red_rate_min  <= RED rate   <= red_rate_max
        GREEN rate    >= green_rate_min
        amber_threshold < red_threshold  (always enforced)

    Returns (best_amber, best_red, full_sweep_DataFrame)
    """
    probs     = preds_df['prob'].values
    prob_min  = float(np.percentile(probs, 2))
    prob_max  = float(np.percentile(probs, 98))
    thresholds = np.round(
        np.arange(
            max(prob_min, resolution),
            min(prob_max, 1.0 - resolution) + resolution / 2,
            resolution,
        ), 4
    )

    print(f"\n{'='*55}")
    print("STEP 1 — THRESHOLD OPTIMISATION (validation set)")
    print(f"{'='*55}")
    print(f"  Score range (2nd–98th pct): {prob_min:.3f} – {prob_max:.3f}")
    print(f"  Sweeping {len(thresholds)} thresholds: "
          f"{thresholds[0]:.3f} → {thresholds[-1]:.3f}  "
          f"(step={resolution})")
    print(f"  Constraints:")
    print(f"    RED rate:   {red_rate_min:.0%} – {red_rate_max:.0%}")
    print(f"    GREEN rate: ≥ {green_rate_min:.0%}")
    print(f"  Youden J = sensitivity + specificity − 1")
    print(f"  (computed on GREEN vs RED only — amber excluded)")

    rows = []
    for amber_t, red_t in product(thresholds, thresholds):
        if amber_t >= red_t:
            continue
        rows.append(compute_tl_metrics(preds_df, aki_ids, amber_t, red_t))

    sweep_df = pd.DataFrame(rows)

    # Apply constraints
    valid = sweep_df[
        (sweep_df['red_rate']   >= red_rate_min) &
        (sweep_df['red_rate']   <= red_rate_max) &
        (sweep_df['green_rate'] >= green_rate_min) &
        (sweep_df['youden_j'].notna())
    ]

    if len(valid) == 0:
        print(f"\n  WARNING: no pairs meet all constraints — relaxing.")
        valid = sweep_df[sweep_df['youden_j'].notna()]

    best_idx   = valid['youden_j'].idxmax()
    best_row   = valid.loc[best_idx]
    best_amber = float(best_row['amber_threshold'])
    best_red   = float(best_row['red_threshold'])

    print(f"\n  Valid pairs meeting constraints: {len(valid)}")
    print(f"\n  ✓ Optimal thresholds found:")
    print(f"      Amber threshold : {best_amber:.4f}")
    print(f"      Red threshold   : {best_red:.4f}")
    print(f"      Youden's J      : {best_row['youden_j']:.3f}")
    print(f"      Sensitivity     : {best_row['sensitivity']:.3f}  "
          f"(of definitive GREEN/RED admissions)")
    print(f"      Specificity     : {best_row['specificity']:.3f}")
    print(f"      RED rate        : {best_row['red_rate']*100:.1f}%  "
          f"of all admissions")
    print(f"      AMBER rate      : {best_row['amber_rate']*100:.1f}%")
    print(f"      GREEN rate      : {best_row['green_rate']*100:.1f}%")

    return best_amber, best_red, sweep_df


# ─────────────────────────────────────────────────────────
#  Alternative threshold methods (all use val set only)
# ─────────────────────────────────────────────────────────

def thresholds_aki_percentile(val_preds:  pd.DataFrame,
                               aki_ids:   set,
                               pct_amber: float = 33,
                               pct_red:   float = 67,
                               base_thresh: float = 0.0) -> tuple:
    """
    Option 2 — Percentile-based thresholds from AKI score distribution.
    Amber = pct_amber-th percentile of max AKI scores on val set.
    Red   = pct_red-th   percentile of max AKI scores on val set.
    Guarantees roughly equal thirds of AKI admissions in each category.
    """
    aki_val = val_preds[val_preds['admission_id'].isin(aki_ids)]
    aki_max = aki_val.groupby('admission_id')['prob'].max()

    t_amber = float(np.percentile(aki_max, pct_amber))
    t_red   = float(np.percentile(aki_max, pct_red))

    # Ensure amber > base threshold and red > amber
    t_amber = max(t_amber, base_thresh + 0.005)
    t_red   = max(t_red,   t_amber    + 0.005)

    print(f"\n  [AKI-percentile] Amber={t_amber:.4f}  Red={t_red:.4f}")
    print(f"  (from {pct_amber}th / {pct_red}th pct of val AKI max scores, "
          f"n={len(aki_max)})")
    return t_amber, t_red


def thresholds_capacity_based(val_preds:    pd.DataFrame,
                               aki_ids:     set,
                               red_pct:     float = 10,
                               amber_pct:   float = 30) -> tuple:
    """
    Option 3 — Capacity-based thresholds.
    Red   = top red_pct%   of all admission max scores on val set.
    Amber = top amber_pct% of all admission max scores on val set.
    Clinically motivated: fixes how many admissions are alerted.
    """
    all_max = val_preds.groupby('admission_id')['prob'].max()

    t_red   = float(np.percentile(all_max, 100 - red_pct))
    t_amber = float(np.percentile(all_max, 100 - amber_pct))

    # Ensure amber < red
    if t_amber >= t_red:
        t_amber = t_red - 0.005

    print(f"\n  [Capacity-based] Amber={t_amber:.4f}  Red={t_red:.4f}")
    print(f"  (top {amber_pct}% = Amber+Red, top {red_pct}% = Red, "
          f"n={len(all_max)} val admissions)")
    return t_amber, t_red


def thresholds_aki_coverage(val_preds:    pd.DataFrame,
                             aki_ids:     set,
                             resolution:  float = 0.01,
                             red_min:     float = 0.05,
                             red_max:     float = 0.15) -> tuple:
    """
    Option 1 variant — maximise AKI coverage in Amber+Red
    subject to Red rate constraint.
    Sweeps on val set, selects pair with highest % AKI in Amber+Red
    while keeping Red rate between red_min and red_max.
    """
    all_max  = val_preds.groupby('admission_id')['prob'].max().reset_index()
    all_max.columns = ['admission_id', 'max_prob']
    all_max['is_aki'] = all_max['admission_id'].isin(aki_ids)

    probs      = all_max['max_prob'].values
    prob_min   = float(np.percentile(probs, 2))
    prob_max   = float(np.percentile(probs, 98))
    thresholds = np.round(np.arange(prob_min, prob_max, resolution), 4)

    best_score = -1
    best_amber = best_red = None
    results    = []

    for amber_t in thresholds:
        for red_t in thresholds:
            if amber_t >= red_t:
                continue
            red_rate = (all_max['max_prob'] >= red_t).mean()
            if not (red_min <= red_rate <= red_max):
                continue
            # % of AKI admissions in Amber or Red
            aki_coverage = (
                all_max[all_max['is_aki']]['max_prob'] >= amber_t
            ).mean()
            results.append({
                'amber_t': amber_t, 'red_t': red_t,
                'red_rate': red_rate, 'aki_coverage': aki_coverage,
            })
            if aki_coverage > best_score:
                best_score = aki_coverage
                best_amber = amber_t
                best_red   = red_t

    if best_amber is None:
        print("  [AKI-coverage] WARNING: no valid pair found — using percentile fallback")
        return thresholds_aki_percentile(val_preds, aki_ids)

    print(f"\n  [AKI-coverage] Amber={best_amber:.4f}  Red={best_red:.4f}")
    print(f"  AKI in Amber+Red: {best_score*100:.1f}%  "
          f"Red rate: {(all_max['max_prob']>=best_red).mean()*100:.1f}%")
    return best_amber, best_red


# ─────────────────────────────────────────────────────────
#  Comparison plot — all four methods side by side
# ─────────────────────────────────────────────────────────

def plot_method_comparison(test_preds:   pd.DataFrame,
                            aki_ids_test: set,
                            methods:      dict,
                            save_path:    Path):
    """
    Side-by-side comparison of all threshold methods.
    methods = {'Youden-J': (amber_t, red_t), 'AKI-percentile': ..., ...}
    Two rows: (1) population bar chart  (2) score distribution with thresholds
    """
    n_methods = len(methods)
    fig, axes = plt.subplots(2, n_methods,
                             figsize=(4.5 * n_methods, 9))
    if n_methods == 1:
        axes = axes.reshape(2, 1)

    adm_all = (
        test_preds.groupby('admission_id')['prob']
        .max().reset_index()
        .rename(columns={'prob': 'max_prob'})
    )
    adm_all['is_aki'] = adm_all['admission_id'].isin(aki_ids_test)

    aki_scores    = adm_all[adm_all['is_aki']]['max_prob'].values
    nonaki_scores = adm_all[~adm_all['is_aki']]['max_prob'].values

    for col, (name, (amber_t, red_t)) in enumerate(methods.items()):
        # ── Row 1: population bar chart ───────────────────
        ax = axes[0, col]
        adm = adm_all.copy()
        adm['tl'] = adm['max_prob'].apply(assign_tl,
                                           amber_t=amber_t, red_t=red_t)

        for xi, (is_aki_flag, label) in enumerate(
            [(False, 'Non-AKI'), (True, 'AKI')]
        ):
            sub     = adm[adm['is_aki'] == is_aki_flag]
            bottom  = 0
            for state, col_hex in [
                ('GREEN', COLOURS['green']),
                ('AMBER', COLOURS['amber']),
                ('RED',   COLOURS['red']),
            ]:
                pct = (sub['tl'] == state).mean() * 100
                bar = ax.bar(xi, pct, bottom=bottom,
                             color=col_hex, edgecolor='white',
                             linewidth=0.5, width=0.6)
                if pct > 4:
                    ax.text(xi, bottom + pct / 2,
                            f'{pct:.0f}%',
                            ha='center', va='center',
                            fontsize=9, color='white',
                            fontweight='bold')
                bottom += pct

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Non-AKI', 'AKI'], fontsize=9)
        ax.set_ylim(0, 100)
        ax.set_ylabel('% of admissions', fontsize=8)
        ax.set_title(f'{name}\nτ_a={amber_t:.3f}  τ_r={red_t:.3f}',
                     fontsize=9, fontweight='bold')
        ax.grid(axis='y', alpha=0.25)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Metrics annotation
        m = compute_tl_metrics(test_preds, aki_ids_test, amber_t, red_t)
        ax.set_xlabel(
            f"Sens={m['sensitivity']:.2f}  Spec={m['specificity']:.2f}\n"
            f"RED={m['red_rate']*100:.0f}%  AMBER={m['amber_rate']*100:.0f}%",
            fontsize=7.5
        )

        # ── Row 2: score distributions ────────────────────
        ax2 = axes[1, col]
        bins = np.linspace(
            min(adm_all['max_prob'].min(), amber_t * 0.8),
            max(adm_all['max_prob'].max(), red_t * 1.1),
            30
        )
        ax2.hist(nonaki_scores, bins=bins, alpha=0.55,
                 color='#2E75B6', label='Non-AKI', density=True)
        ax2.hist(aki_scores,    bins=bins, alpha=0.55,
                 color='#C0392B', label='AKI',     density=True)
        ax2.axvline(amber_t, color=COLOURS['amber'], lw=2,
                    linestyle='--', label=f'Amber τ={amber_t:.3f}')
        ax2.axvline(red_t,   color=COLOURS['red'],   lw=2,
                    linestyle='--', label=f'Red τ={red_t:.3f}')
        ax2.set_xlabel('Max predicted probability', fontsize=8)
        ax2.set_ylabel('Density', fontsize=8)
        ax2.legend(fontsize=7, framealpha=0.7)
        ax2.grid(alpha=0.25)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

    # Shared legend for colours
    patches = [
        mpatches.Patch(color=COLOURS['green'], label='GREEN'),
        mpatches.Patch(color=COLOURS['amber'], label='AMBER'),
        mpatches.Patch(color=COLOURS['red'],   label='RED'),
    ]
    fig.legend(handles=patches, loc='lower center', ncol=3,
               fontsize=10, framealpha=0.85,
               bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(
        'Traffic light threshold method comparison — test set\n'
        '(all thresholds optimised on validation set only)',
        fontsize=12, y=1.01
    )
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print(f"  Saved: {save_path.name}")



def apply_and_report(preds_df:  pd.DataFrame,
                      aki_ids:  set,
                      amber_t:  float,
                      red_t:    float,
                      split:    str = 'test') -> dict:
    """Apply optimised thresholds to a split and print results."""
    m = compute_tl_metrics(preds_df, aki_ids, amber_t, red_t)
    adm = adm_max_tl(preds_df, aki_ids, amber_t, red_t)
    
    print(f"\n{'='*55}")
    print(f"STEP 2 — APPLY THRESHOLDS TO {split.upper()} SET")
    print(f"{'='*55}")
    print(f"  Amber threshold: {amber_t:.4f}")
    print(f"  Red threshold:   {red_t:.4f}")
    print(f"\n  Admission classification ({split}):")
    print(f"    GREEN : {m['n_green']:4d} admissions  "
          f"({m['green_rate']*100:.1f}%)  ← low risk, no alert")
    print(f"    AMBER : {m['n_amber']:4d} admissions  "
          f"({m['amber_rate']*100:.1f}%)  ← intermediate, watch closely")
    print(f"    RED   : {m['n_red']:4d} admissions  "
          f"({m['red_rate']*100:.1f}%)  ← high risk, alert")
    print(f"\n  Performance (GREEN vs RED, amber excluded):")
    print(f"    Sensitivity : {m['sensitivity']:.3f}  "
          f"(AKI cases correctly in RED)")
    print(f"    Specificity : {m['specificity']:.3f}  "
          f"(non-AKI cases correctly in GREEN)")
    print(f"    Youden's J  : {m['youden_j']:.3f}")
    print(f"\n  Interpretation:")
    print(f"    Of every 100 admissions: "
          f"~{m['red_rate']*100:.0f} RED, "
          f"~{m['amber_rate']*100:.0f} AMBER, "
          f"~{m['green_rate']*100:.0f} GREEN")
    if m['sensitivity'] and m['specificity']:
        missed_pct = (1 - m['sensitivity']) * 100
        fp_pct     = (1 - m['specificity']) * 100
        print(f"    {missed_pct:.1f}% of AKI cases land in GREEN or AMBER "
              f"(missed by RED threshold)")
        print(f"    {fp_pct:.1f}% of non-AKI cases land in RED "
              f"(false alerts)")
    aki_adm = adm[adm['is_aki']]
    incl_tp = ((aki_adm['traffic_light'] == 'RED') | (aki_adm['traffic_light'] == 'AMBER')).sum()
    incl_sens = incl_tp/len(aki_adm)

    print(f"\n Two ways to interpret sensitivity:")
    print(f" Strict RED only: {m['sensitivity']:.3f}"
          f" definitive high-risk classification")
    print(f" Inclusive RED + AMBERT: {incl_sens:.3f})"
          f"represents AKI cases in the amber watch closely zone")
    print(f" These patients are NOT being ignored = they receive amber monitoring") 
    return m


# ─────────────────────────────────────────────────────────
#  3. Amber transition analysis
# ─────────────────────────────────────────────────────────

def analyse_amber_transitions(preds_df:  pd.DataFrame,
                               aki_ids:  set,
                               amber_t:  float,
                               red_t:    float) -> pd.DataFrame:
    """
    For every episode where an admission enters AMBER state,
    record what happens next: escalates to RED or de-escalates to GREEN.

    An 'episode' is a consecutive run of AMBER windows within one admission.

    Returns DataFrame with one row per amber episode.
    """
    preds_df = add_tl_col(preds_df, amber_t, red_t)
    episodes = []

    for adm_id, grp in preds_df.groupby('admission_id'):
        grp     = grp.sort_values('step_index').reset_index(drop=True)
        is_aki  = adm_id in aki_ids
        states  = grp['traffic_light'].values
        probs   = grp['prob'].values
        hours   = grp['hours_since_admit'].values
        h2o     = grp['hours_to_onset'].values

        i = 0
        while i < len(states):
            if states[i] == 'AMBER':
                # Find end of this amber run
                j = i
                while j < len(states) and states[j] == 'AMBER':
                    j += 1

                # Characterise the episode
                ep_probs    = probs[i:j]
                ep_dur      = float(hours[j-1] - hours[i]) if j > i else 0.0
                mean_prob   = float(np.mean(ep_probs))
                # Trend: positive = rising toward red, negative = falling
                trend       = float(probs[j-1] - probs[i]) if j > i else 0.0

                # What comes next?
                if j < len(states):
                    next_state = states[j]
                    next_h2o   = float(h2o[j]) if not np.isnan(h2o[j]) else np.nan
                else:
                    next_state = 'END_OF_ADMISSION'
                    next_h2o   = np.nan

                episodes.append({
                    'admission_id':          adm_id,
                    'is_aki':                is_aki,
                    'amber_start_h':         float(hours[i]),
                    'amber_end_h':           float(hours[j-1]),
                    'amber_duration_h':      ep_dur,
                    'n_amber_steps':         j - i,
                    'mean_prob_in_amber':    mean_prob,
                    'trend_in_amber':        trend,
                    'next_state':            next_state,
                    'hours_to_onset_at_exit': next_h2o,
                })
                i = j
            else:
                i += 1

    return pd.DataFrame(episodes)


def print_amber_analysis(ep_df: pd.DataFrame):
    """Print detailed amber transition statistics."""
    print(f"\n{'='*55}")
    print("STEP 3 — AMBER TRANSITION ANALYSIS (test set)")
    print(f"{'='*55}")

    if len(ep_df) == 0:
        print("  No amber episodes found — thresholds may be too wide or narrow.")
        return

    aki_ep    = ep_df[ep_df['is_aki']]
    nonaki_ep = ep_df[~ep_df['is_aki']]

    print(f"\n  Total amber episodes:     {len(ep_df)}")
    print(f"    From AKI admissions:    {len(aki_ep)}")
    print(f"    From non-AKI admissions:{len(nonaki_ep)}")
    print(f"\n  What this means:")
    print(f"    Each 'episode' is a consecutive run of windows")
    print(f"    where the patient was in the AMBER zone.")
    print(f"    We track what happens at the END of each episode.")

    # ── AKI episodes ──────────────────────────────────────
    if len(aki_ep) > 0:
        n_to_red = (aki_ep['next_state'] == 'RED').sum()
        n_to_grn = (aki_ep['next_state'] == 'GREEN').sum()
        n_to_end = (aki_ep['next_state'] == 'END_OF_ADMISSION').sum()
        pct_red  = n_to_red / len(aki_ep) * 100
        pct_grn  = n_to_grn / len(aki_ep) * 100

        print(f"\n  AKI amber episodes → next state:")
        print(f"    → RED   : {n_to_red:3d} ({pct_red:.1f}%)  "
              f"← CORRECT escalation (model caught it)")
        print(f"    → GREEN : {n_to_grn:3d} ({pct_grn:.1f}%)  "
              f"← de-escalation before onset (missed opportunity)")
        print(f"    → END   : {n_to_end:3d} ({n_to_end/len(aki_ep)*100:.1f}%)  "
              f"← admission ended while still amber")
        print(f"\n  AKI amber episode characteristics:")
        print(f"    Mean duration          : "
              f"{aki_ep['amber_duration_h'].mean():.1f}h")
        print(f"    Mean prob within amber : "
              f"{aki_ep['mean_prob_in_amber'].mean():.3f}")
        print(f"    Mean trend (+ = rising): "
              f"{aki_ep['trend_in_amber'].mean():.3f}")
        pos_trend = (aki_ep['trend_in_amber'] > 0).mean() * 100
        print(f"    % with rising trend    : {pos_trend:.1f}%  "
              f"(rising = more likely to escalate to RED)")

        # Timing of escalation
        aki_to_red = aki_ep[
            (aki_ep['next_state'] == 'RED') &
            aki_ep['hours_to_onset_at_exit'].notna()
        ]['hours_to_onset_at_exit']
        if len(aki_to_red) > 0:
            print(f"\n  For AKI episodes that escalated to RED:")
            print(f"    Hours to onset at escalation point:")
            print(f"      Median : {aki_to_red.median():.0f}h  "
                  f"IQR {aki_to_red.quantile(0.25):.0f}–"
                  f"{aki_to_red.quantile(0.75):.0f}h")
            within_24 = (aki_to_red <= 24).sum()
            print(f"      Within 24h of onset: "
                  f"{within_24}/{len(aki_to_red)} "
                  f"({within_24/len(aki_to_red)*100:.1f}%)")
            print(f"    Interpretation: RED alert fired "
                  f"median {aki_to_red.median():.0f}h before AKI onset")

    # ── Non-AKI episodes ──────────────────────────────────
    if len(nonaki_ep) > 0:
        n_to_red = (nonaki_ep['next_state'] == 'RED').sum()
        n_to_grn = (nonaki_ep['next_state'] == 'GREEN').sum()
        pct_red  = n_to_red / len(nonaki_ep) * 100
        pct_grn  = n_to_grn / len(nonaki_ep) * 100

        print(f"\n  Non-AKI amber episodes → next state:")
        print(f"    → RED   : {n_to_red:3d} ({pct_red:.1f}%)  "
              f"← false escalation (unnecessary RED alert)")
        print(f"    → GREEN : {n_to_grn:3d} ({pct_grn:.1f}%)  "
              f"← CORRECT de-escalation (self-resolving amber)")
        print(f"    → END   : "
              f"{(nonaki_ep['next_state']=='END_OF_ADMISSION').sum():3d} "
              f"({(nonaki_ep['next_state']=='END_OF_ADMISSION').mean()*100:.1f}%)")
        print(f"\n  Non-AKI amber episode characteristics:")
        print(f"    Mean duration          : "
              f"{nonaki_ep['amber_duration_h'].mean():.1f}h")
        print(f"    Mean prob within amber : "
              f"{nonaki_ep['mean_prob_in_amber'].mean():.3f}")
        neg_trend = (nonaki_ep['trend_in_amber'] <= 0).mean() * 100
        print(f"    % with flat/falling trend: {neg_trend:.1f}%  "
              f"(falling = more likely to return to GREEN)")

    # ── Clinical interpretation ────────────────────────────
    print(f"\n  Clinical interpretation of AMBER state:")
    print(f"    AMBER = 'watch closely' zone — neither safe nor alarming")
    if len(aki_ep) > 0 and len(nonaki_ep) > 0:
        aki_esc  = (aki_ep['next_state']    == 'RED').mean()
        naki_esc = (nonaki_ep['next_state'] == 'RED').mean()
        print(f"    AKI admissions in amber escalate to RED:    "
              f"{aki_esc*100:.1f}%")
        print(f"    Non-AKI admissions in amber escalate to RED:"
              f"{naki_esc*100:.1f}%")
        if aki_esc > naki_esc * 2:
            print(f"    → Amber IS discriminatory: AKI cases escalate "
                  f"{aki_esc/naki_esc:.1f}× more often than non-AKI")
        else:
            print(f"    → Amber has limited discrimination: "
                  f"AKI and non-AKI escalate at similar rates")


# ─────────────────────────────────────────────────────────
#  4. Plots
# ─────────────────────────────────────────────────────────

def plot_threshold_optimisation(sweep_df:       pd.DataFrame,
                                  best_amber:   float,
                                  best_red:     float,
                                  red_rate_min: float,
                                  red_rate_max: float,
                                  save_path:    Path):
    """Heatmap showing Youden J across the full threshold sweep."""
    pivot_j = sweep_df.pivot_table(
        index='amber_threshold', columns='red_threshold',
        values='youden_j', aggfunc='first',
    )
    pivot_r = sweep_df.pivot_table(
        index='amber_threshold', columns='red_threshold',
        values='red_rate', aggfunc='first',
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Traffic Light Threshold Optimisation", fontsize=13)

    # Panel 1: Youden J heatmap
    ax  = axes[0]
    im  = ax.imshow(pivot_j.values, aspect='auto', origin='lower',
                    cmap='RdYlGn', vmin=-0.5, vmax=1.0)
    ax.set_xticks(range(len(pivot_j.columns)))
    ax.set_yticks(range(len(pivot_j.index)))
    ax.set_xticklabels([f'{v:.2f}' for v in pivot_j.columns],
                       rotation=45, ha='right', fontsize=7)
    ax.set_yticklabels([f'{v:.2f}' for v in pivot_j.index], fontsize=7)
    ax.set_xlabel('Red threshold'); ax.set_ylabel('Amber threshold')
    ax.set_title("Youden's J (GREEN vs RED)\nhigher = better")
    plt.colorbar(im, ax=ax)
    # Mark best
    ry = list(pivot_j.index).index(
        min(pivot_j.index, key=lambda x: abs(x - best_amber))
    )
    rx = list(pivot_j.columns).index(
        min(pivot_j.columns, key=lambda x: abs(x - best_red))
    )
    ax.plot(rx, ry, '*', color='black', ms=15,
            label=f'Best ({best_amber:.2f}, {best_red:.2f})')
    ax.legend(fontsize=8)

    # Panel 2: RED rate heatmap
    ax  = axes[1]
    im2 = ax.imshow(pivot_r.values * 100, aspect='auto', origin='lower',
                    cmap='Reds', vmin=0, vmax=50)
    ax.set_xticks(range(len(pivot_r.columns)))
    ax.set_yticks(range(len(pivot_r.index)))
    ax.set_xticklabels([f'{v:.2f}' for v in pivot_r.columns],
                       rotation=45, ha='right', fontsize=7)
    ax.set_yticklabels([f'{v:.2f}' for v in pivot_r.index], fontsize=7)
    ax.set_xlabel('Red threshold'); ax.set_ylabel('Amber threshold')
    ax.set_title(f'RED alert rate (%)\n'
                 f'Constraint: {red_rate_min*100:.0f}–{red_rate_max*100:.0f}%')
    plt.colorbar(im2, ax=ax)
    ax.plot(rx, ry, '*', color='white', ms=15)

    # Panel 3: Valid region scatter
    ax    = axes[2]
    valid = sweep_df[sweep_df['youden_j'].notna()]
    meets = valid[
        (valid['red_rate']   >= red_rate_min) &
        (valid['red_rate']   <= red_rate_max) &
        (valid['green_rate'] >= 0.50)
    ]
    other = valid[~valid.index.isin(meets.index)]

    ax.scatter(other['red_threshold'], other['amber_threshold'],
               c='lightgray', s=15, alpha=0.4, label='Fails constraint')
    if len(meets) > 0:
        sc = ax.scatter(meets['red_threshold'], meets['amber_threshold'],
                        c=meets['youden_j'], cmap='RdYlGn',
                        vmin=-0.5, vmax=1.0, s=50, zorder=5)
        plt.colorbar(sc, ax=ax, label="Youden's J")
    ax.plot(best_red, best_amber, '*', color='black', ms=18, zorder=10,
            label=f'Best ({best_amber:.2f}, {best_red:.2f})')
    ax.set_xlabel('Red threshold'); ax.set_ylabel('Amber threshold')
    ax.set_title('Valid threshold pairs\n(coloured by Youden J)')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path.name}")


def plot_tl_distribution(preds_df:  pd.DataFrame,
                          aki_ids:  set,
                          amber_t:  float,
                          red_t:    float,
                          save_path: Path):
    """Score distributions and admission-level traffic light split."""
    adm = adm_max_tl(preds_df, aki_ids, amber_t, red_t)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(
        f'Traffic Light Distribution  '
        f'(amber≥{amber_t:.3f}, red≥{red_t:.3f})', fontsize=12
    )

    # Panel 1: Score histogram with threshold lines
    ax = axes[0]
    for is_aki, label, col in [
        (False, 'Non-AKI', 'steelblue'),
        (True,  'AKI',     COLOURS['red']),
    ]:
        sub = adm[adm['is_aki'] == is_aki]['max_prob']
        ax.hist(sub, bins=25, alpha=0.55, color=col, density=True,
                label=f'{label} (n={len(sub)})', edgecolor='white')
    ax.axvline(amber_t, color=COLOURS['amber'], lw=2.5, linestyle='--',
               label=f'Amber ({amber_t:.3f})')
    ax.axvline(red_t,   color=COLOURS['red'],   lw=2.5, linestyle='--',
               label=f'Red ({red_t:.3f})')
    ax.axvspan(0,       amber_t, alpha=0.06, color=COLOURS['green'])
    ax.axvspan(amber_t, red_t,   alpha=0.06, color=COLOURS['amber'])
    ax.axvspan(red_t,   1.0,     alpha=0.06, color=COLOURS['red'])
    ax.set_xlabel('Max predicted probability per admission')
    ax.set_ylabel('Density')
    ax.set_title('Score distribution — one score per admission\n'
                 '(max across all windows)')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Panel 2: Stacked bar by AKI status
    ax = axes[1]
    for xi, (is_aki, label) in enumerate([(False,'Non-AKI'),(True,'AKI')]):
        sub    = adm[adm['is_aki'] == is_aki]
        counts = sub['traffic_light'].value_counts()
        total  = len(sub)
        bottom = 0
        for state, col in zip(TL_ORDER, [COLOURS['green'],
                                          COLOURS['amber'],
                                          COLOURS['red']]):
            val = counts.get(state, 0) / total * 100
            ax.bar(label, val, bottom=bottom, color=col,
                   edgecolor='white', linewidth=0.5)
            if val > 4:
                ax.text(xi, bottom + val / 2,
                        f'{val:.1f}%', ha='center', va='center',
                        fontsize=10, fontweight='bold', color='white')
            bottom += val
    patches = [mpatches.Patch(color=COLOURS[s.lower()], label=s)
               for s in TL_ORDER]
    ax.legend(handles=patches, fontsize=9)
    ax.set_ylabel('% of admissions')
    ax.set_ylim(0, 100)
    ax.set_title('Traffic light state per group\n'
                 '(AKI should have more RED, non-AKI more GREEN)')
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path.name}")


def plot_tl_trajectories(preds_df:  pd.DataFrame,
                          aki_ids:  set,
                          amber_t:  float,
                          red_t:    float,
                          save_path: Path,
                          n_aki:    int = 8,
                          n_nonaki: int = 4):
    """Per-patient risk curves coloured by traffic light state."""
    preds_df = add_tl_col(preds_df, amber_t, red_t)
    nonaki_ids   = [a for a in preds_df['admission_id'].unique()
                    if a not in aki_ids]
    all_samples  = list(aki_ids)[:n_aki] + nonaki_ids[:n_nonaki]

    ncols = 4
    nrows = (len(all_samples) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3.8))
    axes = axes.flatten()

    for i, adm_id in enumerate(all_samples):
        ax     = axes[i]
        adm    = preds_df[preds_df['admission_id'] == adm_id]\
                 .sort_values('step_index')
        is_aki = adm_id in aki_ids

        # Background zones
        ax.axhspan(0,       amber_t, alpha=0.07, color=COLOURS['green'])
        ax.axhspan(amber_t, red_t,   alpha=0.07, color=COLOURS['amber'])
        ax.axhspan(red_t,   1.0,     alpha=0.07, color=COLOURS['red'])
        ax.axhline(amber_t, color=COLOURS['amber'], lw=1, linestyle='--',
                   alpha=0.6)
        ax.axhline(red_t,   color=COLOURS['red'],   lw=1, linestyle='--',
                   alpha=0.6)

        hours = adm['hours_since_admit'].values
        probs = adm['prob'].values
        tl    = adm['traffic_light'].values

        for t in range(len(hours) - 1):
            ax.plot(hours[t:t+2], probs[t:t+2],
                    color=COLOURS[tl[t].lower()], lw=2.5,
                    solid_capstyle='round')
        if len(hours) > 0:
            ax.scatter([hours[-1]], [probs[-1]],
                       color=COLOURS[tl[-1].lower()], s=30, zorder=5)

        if is_aki:
            first_pos = adm[adm['label'] == 1]
            if not first_pos.empty:
                ph = first_pos.iloc[0]['hours_since_admit']
                ax.axvline(ph, color='black', lw=1.5, linestyle='--',
                           label='24h window opens')
                ax.axvline(ph + 24, color='darkred', lw=1, linestyle=':',
                           label='~Onset (+24h)')
            ax.set_title(f'AKI — {adm_id}', fontsize=8,
                         color='darkred', fontweight='bold')
        else:
            ax.set_title(f'No AKI — {adm_id}', fontsize=8,
                         color='steelblue')

        ax.set_ylim(0, 1)
        ax.set_xlabel('Hours since admission', fontsize=7)
        ax.set_ylabel('AKI risk', fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2)

        if i == 0:
            patches = [mpatches.Patch(color=COLOURS[s.lower()], label=s)
                       for s in TL_ORDER]
            ax.legend(handles=patches, fontsize=6, loc='upper left')

    for j in range(len(all_samples), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(
        f'Per-patient risk trajectories — traffic light\n'
        f'(amber≥{amber_t:.3f}, red≥{red_t:.3f})',
        fontsize=12, y=1.01
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path.name}")


def plot_amber_transitions(ep_df: pd.DataFrame, save_path: Path):
    """Six-panel amber transition analysis plot."""
    if len(ep_df) == 0:
        print("  No amber episodes to plot.")
        return

    aki_ep    = ep_df[ep_df['is_aki']]
    nonaki_ep = ep_df[~ep_df['is_aki']]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle('Amber Episode Transition Analysis', fontsize=13)

    # Panel 1: Transition destinations
    ax          = axes[0, 0]
    dest_states = ['RED', 'GREEN', 'END_OF_ADMISSION']
    dest_cols   = [COLOURS['red'], COLOURS['green'], COLOURS['grey']]
    for xi, (grp, label) in enumerate(
        [(aki_ep,'AKI'), (nonaki_ep,'Non-AKI')]
    ):
        if len(grp) == 0:
            continue
        bottom = 0
        for dest, col in zip(dest_states, dest_cols):
            val = (grp['next_state'] == dest).mean() * 100
            ax.bar(label, val, bottom=bottom, color=col,
                   edgecolor='white', linewidth=0.5)
            if val > 3:
                ax.text(xi, bottom + val/2, f'{val:.0f}%',
                        ha='center', va='center',
                        fontsize=9, color='white', fontweight='bold')
            bottom += val
    patches = [mpatches.Patch(color=c, label=d)
               for d, c in zip(['→RED','→GREEN','→End'],
                                dest_cols)]
    ax.legend(handles=patches, fontsize=8)
    ax.set_ylabel('% of amber episodes')
    ax.set_ylim(0, 100)
    ax.set_title('What happens after amber?\n'
                 '(AKI should escalate to RED)')
    ax.grid(alpha=0.3, axis='y')

    # Panel 2: Amber duration distribution
    ax = axes[0, 1]
    if len(aki_ep) > 0:
        ax.hist(aki_ep['amber_duration_h'], bins=15, alpha=0.6,
                color=COLOURS['red'], label=f'AKI (n={len(aki_ep)})',
                edgecolor='white')
    if len(nonaki_ep) > 0:
        ax.hist(nonaki_ep['amber_duration_h'], bins=15, alpha=0.6,
                color='steelblue',
                label=f'Non-AKI (n={len(nonaki_ep)})',
                edgecolor='white')
    ax.set_xlabel('Duration of amber episode (hours)')
    ax.set_ylabel('Count')
    ax.set_title('How long do patients stay amber?')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Panel 3: Trend within amber
    ax = axes[0, 2]
    for grp, col, label in [
        (aki_ep,    COLOURS['red'], 'AKI'),
        (nonaki_ep, 'steelblue',    'Non-AKI'),
    ]:
        if len(grp) > 0:
            ax.hist(grp['trend_in_amber'], bins=15, alpha=0.6,
                    color=col, label=label, edgecolor='white')
    ax.axvline(0, color='black', lw=1.5, linestyle='--',
               label='No trend')
    ax.set_xlabel('Prob trend within amber\n(+ve = rising toward red)')
    ax.set_ylabel('Count')
    ax.set_title('Risk trend within amber\n'
                 '(AKI should trend positive)')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Panel 4: Episode count flow
    ax = axes[1, 0]
    for xi, (grp, col, label) in enumerate([
        (aki_ep,    COLOURS['red'], 'AKI'),
        (nonaki_ep, 'steelblue',    'Non-AKI'),
    ]):
        if len(grp) == 0:
            continue
        vals = [
            len(grp),
            (grp['next_state'] == 'RED').sum(),
            (grp['next_state'] == 'GREEN').sum(),
        ]
        cats  = ['Entered\nAMBER', '→ RED', '→ GREEN']
        x_pos = [xi * 0.35 + c for c in range(len(cats))]
        ax.bar(x_pos, vals, width=0.3, color=col, alpha=0.8,
               label=label, edgecolor='white')
        for xp, v in zip(x_pos, vals):
            ax.text(xp, v + 0.3, str(v), ha='center',
                    va='bottom', fontsize=8)
    ax.set_xticks([0.175 + c for c in range(len(cats))])
    ax.set_xticklabels(cats, fontsize=9)
    ax.set_ylabel('Number of amber episodes')
    ax.set_title('Amber episode flow')
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')

    # Panel 5: Timing of amber→RED for AKI cases
    ax = axes[1, 1]
    if len(aki_ep) > 0:
        for dest, col, label in [
            ('RED',   COLOURS['red'],   'Escalated to RED'),
            ('GREEN', COLOURS['green'], 'De-escalated to GREEN'),
        ]:
            sub = aki_ep[
                (aki_ep['next_state'] == dest) &
                aki_ep['hours_to_onset_at_exit'].notna()
            ]['hours_to_onset_at_exit']
            if len(sub) > 0:
                ax.hist(sub, bins=10, alpha=0.6, color=col,
                        label=f'{label} (n={len(sub)})',
                        edgecolor='white')
        ax.axvline(24, color='gray', lw=1.5, linestyle='--',
                   label='24h horizon')
        ax.set_xlabel('Hours to onset at transition point')
        ax.set_ylabel('Count')
        ax.set_title('AKI — when does amber transition occur\n'
                     'relative to onset?')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Panel 6: Mean prob in amber by outcome
    ax    = axes[1, 2]
    items = []
    for grp, aki_label in [(aki_ep,'AKI'), (nonaki_ep,'Non-AKI')]:
        for dest in ['RED', 'GREEN']:
            sub = grp[grp['next_state'] == dest]
            if len(sub) >= 2:
                items.append({
                    'label':  f'{aki_label}\n→{dest}',
                    'mean':   sub['mean_prob_in_amber'].mean(),
                    'std':    sub['mean_prob_in_amber'].std(),
                    'color':  COLOURS['red'] if dest == 'RED' else
                              COLOURS['green'],
                    'alpha':  0.9 if aki_label == 'AKI' else 0.5,
                })
    if items:
        for idx, it in enumerate(items):
            ax.bar(idx, it['mean'], color=it['color'],
                   alpha=it['alpha'], edgecolor='white')
            ax.errorbar(idx, it['mean'], yerr=it['std'],
                        color='black', capsize=4, lw=1.5)
        ax.set_xticks(range(len(items)))
        ax.set_xticklabels([it['label'] for it in items], fontsize=8)
    ax.set_ylabel('Mean probability in amber episode')
    ax.set_ylim(0, 1)
    ax.set_title('How risky is the amber period?\n'
                 '(higher = closer to RED at transition)')
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path.name}")


# ─────────────────────────────────────────────────────────
#  5. Main pipeline
# ─────────────────────────────────────────────────────────

def run_full_analysis(val_preds:     pd.DataFrame,
                       test_preds:   pd.DataFrame,
                       aki_ids_val:  set,
                       aki_ids_test: set,
                       plots_dir:    Path,
                       red_rate_min:   float = 0.10,
                       red_rate_max:   float = 0.20,
                       green_rate_min: float = 0.50,
                       resolution:     float = 0.05,
                       base_thresh:    float = 0.0) -> dict:
    """Run the full traffic light pipeline for all threshold methods."""
    plots_dir.mkdir(exist_ok=True)

    # ── Method 1: Youden-J (original) ────────────────────
    print(f"\n{'='*55}")
    print("METHOD 1 — Youden-J (original)")
    print(f"{'='*55}")
    best_amber_yj, best_red_yj, sweep_df = optimise_thresholds(
        val_preds, aki_ids_val,
        red_rate_min, red_rate_max, green_rate_min, resolution,
    )
    sweep_df.to_csv(plots_dir / 'threshold_sweep_youdenj.csv', index=False)

    # ── Method 2: AKI percentile ──────────────────────────
    print(f"\n{'='*55}")
    print("METHOD 2 — AKI-percentile (33rd/67th of val AKI scores)")
    print(f"{'='*55}")
    best_amber_pct, best_red_pct = thresholds_aki_percentile(
        val_preds, aki_ids_val,
        pct_amber=33, pct_red=67,
        base_thresh=base_thresh,
    )

    # ── Method 3: Capacity-based ──────────────────────────
    print(f"\n{'='*55}")
    print("METHOD 3 — Capacity-based (top 10%=Red, top 30%=Amber+Red)")
    print(f"{'='*55}")
    best_amber_cap, best_red_cap = thresholds_capacity_based(
        val_preds, aki_ids_val,
        red_pct=10, amber_pct=30,
    )

    # ── Method 4: AKI-coverage ────────────────────────────
    print(f"\n{'='*55}")
    print("METHOD 4 — AKI-coverage (maximise AKI in Amber+Red)")
    print(f"{'='*55}")
    best_amber_cov, best_red_cov = thresholds_aki_coverage(
        val_preds, aki_ids_val,
        resolution=0.01,
        red_min=red_rate_min,
        red_max=red_rate_max,
    )

    # ── All methods dict ──────────────────────────────────
    all_methods = {
        'Youden-J':       (best_amber_yj,  best_red_yj),
        'AKI-percentile': (best_amber_pct, best_red_pct),
        'Capacity-based': (best_amber_cap, best_red_cap),
        'AKI-coverage':   (best_amber_cov, best_red_cov),
    }

    print(f"\n{'='*55}")
    print("THRESHOLD SUMMARY (all methods, applied to test set)")
    print(f"{'='*55}")
    print(f"  {'Method':<18} {'Amber':>8} {'Red':>8} "
          f"{'Sens':>7} {'Spec':>7} "
          f"{'RED%':>6} {'AMBER%':>7} {'GREEN%':>7}")
    print(f"  {'-'*70}")
    for name, (at, rt) in all_methods.items():
        m = compute_tl_metrics(test_preds, aki_ids_test, at, rt)
        s  = m['sensitivity'] or 0
        sp = m['specificity'] or 0
        print(f"  {name:<18} {at:>8.4f} {rt:>8.4f} "
              f"{s:>7.3f} {sp:>7.3f} "
              f"{m['red_rate']*100:>6.1f} "
              f"{m['amber_rate']*100:>7.1f} "
              f"{m['green_rate']*100:>7.1f}")

    # ── Apply best Youden-J to test (main result) ─────────
    test_m = apply_and_report(test_preds, aki_ids_test,
                               best_amber_yj, best_red_yj, split='test')

    # ── Amber transition analysis (Youden-J thresholds) ───
    print(f"\n  Running amber transition analysis (Youden-J thresholds)...")
    ep_df = analyse_amber_transitions(
        test_preds, aki_ids_test, best_amber_yj, best_red_yj
    )
    print_amber_analysis(ep_df)

    # ── Plots ─────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("GENERATING PLOTS")
    print(f"{'='*55}")

    # Comparison figure — all 4 methods
    plot_method_comparison(
        test_preds, aki_ids_test, all_methods,
        plots_dir / 'threshold_method_comparison.png'
    )

    # Standard plots for Youden-J (main result)
    plot_threshold_optimisation(
        sweep_df, best_amber_yj, best_red_yj,
        red_rate_min, red_rate_max,
        plots_dir / 'threshold_optimisation.png'
    )
    plot_tl_distribution(
        test_preds, aki_ids_test, best_amber_yj, best_red_yj,
        plots_dir / 'traffic_light_distribution_youdenj.png'
    )
    plot_tl_trajectories(
        test_preds, aki_ids_test, best_amber_yj, best_red_yj,
        plots_dir / 'traffic_light_trajectories_youdenj.png'
    )
    plot_amber_transitions(ep_df, plots_dir / 'amber_transitions_youdenj.png')

    # Individual distribution plots for each alternative method
    # Full set of original plots for every method
    for name, (at, rt) in all_methods.items():
        safe_name = name.lower().replace('-', '_').replace(' ', '_')
        print(f"\n  Generating plots for: {name}")

        plot_tl_distribution(
            test_preds, aki_ids_test, at, rt,
            plots_dir / f'traffic_light_distribution_{safe_name}.png'
        )
        plot_tl_trajectories(
            test_preds, aki_ids_test, at, rt,
            plots_dir / f'traffic_light_trajectories_{safe_name}.png'
        )
        ep = analyse_amber_transitions(
            test_preds, aki_ids_test, at, rt
        )
        plot_amber_transitions(
            ep,
            plots_dir / f'amber_transitions_{safe_name}.png'
        )
    # ── Save all thresholds to JSON ───────────────────────
    all_results = {
        'methods': {
            name: {
                'amber_threshold': float(at),
                'red_threshold':   float(rt),
                **{k: v for k, v in
                   compute_tl_metrics(test_preds, aki_ids_test, at, rt).items()
                   if k not in ('amber_threshold', 'red_threshold')}
            }
            for name, (at, rt) in all_methods.items()
        },
        'recommended': 'Youden-J',
        'amber_threshold':      best_amber_yj,
        'red_threshold':        best_red_yj,
        'optimised_on':         'validation_set',
        'test_sensitivity':     test_m['sensitivity'],
        'test_specificity':     test_m['specificity'],
        'test_youden_j':        test_m['youden_j'],
        'test_red_rate':        test_m['red_rate'],
        'test_amber_rate':      test_m['amber_rate'],
        'test_green_rate':      test_m['green_rate'],
        'amber_episodes_total': int(len(ep_df)),
        'aki_amber_to_red_pct': round(
            float((ep_df[ep_df['is_aki']]['next_state'] == 'RED').mean()), 4
        ) if len(ep_df[ep_df['is_aki']]) > 0 else None,
        'nonaki_amber_to_green_pct': round(
            float((ep_df[~ep_df['is_aki']]['next_state'] == 'GREEN').mean()), 4
        ) if len(ep_df[~ep_df['is_aki']]) > 0 else None,
    }
    return all_results



def main():
    p = argparse.ArgumentParser(
        description='Traffic light threshold optimisation + amber analysis'
    )
    p.add_argument('--run',        required=True,
                   help='Run name (folder under data/models/)')
    p.add_argument('--models_dir', default='data/models')
    p.add_argument('--red_min',    type=float, default=0.10,
                   help='Min fraction of admissions in RED (default 0.10)')
    p.add_argument('--red_max',    type=float, default=0.20,
                   help='Max fraction of admissions in RED (default 0.20)')
    p.add_argument('--green_min',  type=float, default=0.50,
                   help='Min fraction in GREEN (default 0.50)')
    p.add_argument('--resolution', type=float, default=0.05,
                   help='Threshold sweep step size (default 0.05)')
    args = p.parse_args()

    run_dir   = Path(args.models_dir) / args.run
    plots_dir = run_dir / 'plots'

    if not run_dir.exists():
        raise FileNotFoundError(f"Run not found: {run_dir}")

    # Load predictions — must already exist from evaluate.py
    val_path  = run_dir / 'val_predictions.csv'
    test_path = run_dir / 'test_predictions.csv'

    for path in [val_path, test_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Predictions not found: {path}\n"
                f"Run first:\n"
                f"  python evaluate.py --run {args.run} --split val\n"
                f"  python evaluate.py --run {args.run} --split test"
            )

    print(f"\nLoading predictions for: {args.run}")
    val_preds  = pd.read_csv(val_path)
    test_preds = pd.read_csv(test_path)

    aki_ids_val  = set(val_preds[val_preds['label']  == 1]['admission_id'])
    aki_ids_test = set(test_preds[test_preds['label'] == 1]['admission_id'])

    print(f"  Val  set: {val_preds['admission_id'].nunique()} admissions "
          f"({len(aki_ids_val)} AKI)")
    print(f"  Test set: {test_preds['admission_id'].nunique()} admissions "
          f"({len(aki_ids_test)} AKI)")

    results = run_full_analysis(
        val_preds      = val_preds,
        test_preds     = test_preds,
        aki_ids_val    = aki_ids_val,
        aki_ids_test   = aki_ids_test,
        plots_dir      = plots_dir,
        red_rate_min   = args.red_min,
        red_rate_max   = args.red_max,
        green_rate_min = args.green_min,
        resolution     = args.resolution,
        base_thresh    = 0.0,
    )

    out_path = run_dir / 'traffic_light_report.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*55}")
    print("COMPLETE — FINAL SUMMARY (Youden-J thresholds)")
    print(f"{'='*55}")
    print(f"  Run:             {args.run}")
    print(f"  Amber threshold: {results['amber_threshold']:.4f}")
    print(f"  Red threshold:   {results['red_threshold']:.4f}")
    print(f"  Test sensitivity:{results['test_sensitivity']:.3f}")
    print(f"  Test specificity:{results['test_specificity']:.3f}")
    print(f"  Test Youden J:   {results['test_youden_j']:.3f}")
    print(f"  RED rate:        {results['test_red_rate']*100:.1f}%")
    print(f"  AMBER rate:      {results['test_amber_rate']*100:.1f}%")
    print(f"  GREEN rate:      {results['test_green_rate']*100:.1f}%")
    print(f"\n  All 4 method thresholds in: {out_path}")
    print(f"  Comparison plot: {plots_dir}/threshold_method_comparison.png")


if __name__ == '__main__':
    main()

