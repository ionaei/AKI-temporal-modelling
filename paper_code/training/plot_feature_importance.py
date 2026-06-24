"""
plot_feature_importance.py

Generates three figures from attention-weighted feature importance data:

  1. overall_importance.png      — appendix: all features, overall
  2. aki_nonaki_importance.png   — appendix: AKI vs non-AKI side by side
  3. differential_importance.png — MAIN PAPER: top features by AKI minus non-AKI

Usage
-----
    python plot_feature_importance.py \
        --overall  data/models/lstm_focal_h128_l2_attention/attention_feature_importance.csv \
        --split    data/models/lstm_focal_h128_l2_attention/attention_feature_importance_aki_vs_nonaki.csv \
        --out_dir  data/models/lstm_focal_h128_l2_attention/plots/explainability

Expects
-------
    overall: columns [feature, importance, std]
    split:   columns [feature, aki_importance, nonaki_importance,
                       difference, ratio]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ─────────────────────────────────────────────────────────
#  Clinical category colour mapping
# ─────────────────────────────────────────────────────────

# Each tuple: (keywords_to_match, hex_colour, category_label)
# Matched in order — first match wins
CATEGORY_MAP = [
    # Missingness flags — steel blue (distinct from lab values)
    (['_missing', 'missing_flag'],
     '#5B9BD5', 'Missingness indicator'),

    # Blood gas / respiratory
    (['tao2', 'pao2', 'po2', 'pco2', 'spo2', 'fio2', 'sao2',
      'base_excess', 'lactate', 'ph', 'hco3', 'bicarbonate',
      'anion_gap', 'o2'],
     '#B069DB', 'Blood gas / respiratory'),

    # Electrolytes
    (['sodium', 'potassium', 'chloride', 'magnesium', 'phosphate',
      'calcium', 'na_', 'k_', 'cl_'],
     '#4BACC6', 'Electrolytes'),

    # Renal markers
    (['creatinine', 'urea', 'egfr', 'creat'],
     '#E84545', 'Renal markers'),

    # Haematology
    (['haemoglobin', 'hemoglobin', 'platelets', 'wbc', 'neutrophil',
      'lymphocyte', 'hct', 'mchc', 'mcv', 'mch', 'rbc', 'rdw',
      'mpv', 'inr', 'pt_', 'aptt'],
     '#70AD47', 'Haematology / coagulation'),

    # Liver / metabolic
    (['bilirubin', 'alt', 'ast', 'alp', 'albumin', 'protein',
      'glucose', 'cholesterol', 'triglyceride', 'ldh', 'ck_',
      'crp', 'procalcitonin'],
     '#ED7D31', 'Liver / metabolic'),

    # Cardiac
    (['troponin', 'bnp', 'nt_pro'],
     '#A9D18E', 'Cardiac'),

    # Nephrotoxins
    (['nephrotoxin', 'vancomycin', 'gentamicin', 'tobramycin',
      'amikacin', 'colistin', 'amphotericin', 'ciclosporin',
      'tacrolimus', 'ibuprofen', 'contrast', 'aciclovir',
      'methotrexate', 'cisplatin', 'carboplatin'],
     '#C00000', 'Nephrotoxins'),

    # Drug classes
    (['drug_class', 'diuretic', 'vasopressor', 'antibiotic',
      'antifungal', 'immunosuppressant', 'sedation', 'anticoagulant',
      'ppi_h2'],
     '#FF6B6B', 'Medications'),

    # Procedures
    (['proc_', 'cardiac_surgery', 'bypass', 'dialysis', 'rrt',
      'transplant', 'mechanical_vent', 'central_line'],
     '#7030A0', 'Procedures'),

    # Demographics
    (['age_', 'sex_', 'ethnicity_', 'weight', 'height'],
     '#548235', 'Demographics'),

    # Diagnoses / ICD
    (['diag_', 'icd_'],
     '#FFC000', 'Diagnoses (ICD)'),

    # Alert flags
    (['_alert'],
     '#FF9900', 'Clinical alert flags'),

    # Count features
    (['_n', '_count', 'n_unique', 'n_procedures'],
     '#9DC3E6', 'Measurement counts'),

    # Trend features
    (['_trend'],
     '#B4C7E7', 'Trends'),
]

def get_colour(feature: str) -> tuple[str, str]:
    """Return (hex_colour, category_label) for a feature name."""
    f = feature.lower()
    for keywords, colour, label in CATEGORY_MAP:
        if any(kw in f for kw in keywords):
            return colour, label
    return '#888888', 'Other'


def add_colour_legend(ax, colours_used: dict,
                       loc: str = 'lower right',
                       fontsize: int = 8):
    """Add a legend for clinical categories actually present in the plot."""
    patches = [
        mpatches.Patch(color=col, label=lab)
        for lab, col in sorted(colours_used.items())
    ]
    ax.legend(handles=patches, loc=loc, fontsize=fontsize,
              framealpha=0.9, title='Category', title_fontsize=fontsize)


def clean_label(feature: str) -> str:
    """Make feature names human-readable for plot labels."""
    f = (feature
         .replace('_missing_flag', ' (missing)')
         .replace('_missing',      ' (missing)')
         .replace('_mean',   ' mean')
         .replace('_last',   ' last')
         .replace('_min',    ' min')
         .replace('_max',    ' max')
         .replace('_trend',  ' trend')
         .replace('_alert',  ' alert')
         .replace('_count',  ' count')
         .replace('_n',      ' n')
         .replace('nephrotoxin_', '')
         .replace('drug_class_', '')
         .replace('proc_',   '')
         .replace('diag_',   'ICD ')
         .replace('age_',    'age: ')
         .replace('sex_',    'sex: ')
         .replace('ethnicity_', 'ethnicity: ')
         .replace('_', ' ')
         .strip()
         )
    return f.capitalize()


# ─────────────────────────────────────────────────────────
#  Figure 1 — Overall importance (appendix)
# ─────────────────────────────────────────────────────────

def plot_overall(overall: pd.DataFrame,
                  save_path: Path,
                  top_n: int = 30):
    df = overall.sort_values('importance', ascending=False).head(top_n).copy()
    df = df.iloc[::-1]   # reverse so highest is at top of barh

    colours     = []
    cat_present = {}
    for feat in df['feature']:
        col, cat = get_colour(feat)
        colours.append(col)
        cat_present[cat] = col

    labels = [clean_label(f) for f in df['feature']]

    fig, ax = plt.subplots(figsize=(9, top_n * 0.32 + 2))

    bars = ax.barh(range(len(df)), df['importance'],
                   color=colours, alpha=0.85, edgecolor='white',
                   linewidth=0.4)

    if 'std' in df.columns:
        ax.barh(range(len(df)), df['std'],
                left=df['importance'] - df['std'] / 2,
                color='none', edgecolor='#555555',
                linewidth=0.6, alpha=0.5)
        # error bars
        ax.errorbar(df['importance'], range(len(df)),
                    xerr=df['std'], fmt='none',
                    ecolor='#444444', elinewidth=0.8,
                    capsize=2, alpha=0.7)

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Attention-weighted importance', fontsize=10)
    ax.set_title(f'Overall feature importance — top {top_n} features\n'
                 f'(averaged across all admissions)',
                 fontsize=11, pad=10)
    ax.grid(axis='x', alpha=0.25, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    add_colour_legend(ax, cat_present, loc='lower right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Overall importance → {save_path}")


# ─────────────────────────────────────────────────────────
#  Figure 2 — AKI vs non-AKI side by side (appendix)
# ─────────────────────────────────────────────────────────

def plot_aki_nonaki(split: pd.DataFrame,
                     save_path: Path,
                     top_n: int = 25):
    """
    Two horizontal bar charts side by side.
    Left: top features for AKI admissions.
    Right: top features for non-AKI admissions.
    Features are independently ranked for each panel.
    """
    aki_top   = (split.sort_values('aki_importance', ascending=False)
                 .head(top_n).copy())
    naki_top  = (split.sort_values('nonaki_importance', ascending=False)
                 .head(top_n).copy())

    fig, axes = plt.subplots(1, 2, figsize=(16, top_n * 0.33 + 2.5))
    fig.suptitle('Feature importance — AKI vs non-AKI admissions\n'
                 f'(top {top_n} features each, independently ranked)',
                 fontsize=12, y=1.01)

    cat_present = {}
    for ax, df, imp_col, title, colour_col in [
        (axes[0], aki_top,  'aki_importance',
         'AKI admissions',     '#C00000'),
        (axes[1], naki_top, 'nonaki_importance',
         'Non-AKI admissions', '#2E75B6'),
    ]:
        df = df.iloc[::-1]
        colours = []
        for feat in df['feature']:
            col, cat = get_colour(feat)
            colours.append(col)
            cat_present[cat] = col

        labels = [clean_label(f) for f in df['feature']]

        ax.barh(range(len(df)), df[imp_col],
                color=colours, alpha=0.85,
                edgecolor='white', linewidth=0.4)
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel('Attention-weighted importance', fontsize=9)
        ax.set_title(title, fontsize=11, fontweight='bold',
                     color=colour_col, pad=8)
        ax.grid(axis='x', alpha=0.25, linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Shared legend on second panel
    add_colour_legend(axes[1], cat_present, loc='lower right', fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  AKI vs non-AKI → {save_path}")


# ─────────────────────────────────────────────────────────
#  Figure 3 — Differential importance (MAIN PAPER)
# ─────────────────────────────────────────────────────────

def plot_differential(split: pd.DataFrame,
                       save_path: Path,
                       top_n: int = 25):
    """
    Features ranked by (aki_importance - nonaki_importance).
    Positive = more important for AKI prediction.
    Negative = more important for non-AKI prediction.
    Uses a diverging bar chart.
    """
    df = split.copy()
    df = df.sort_values('difference', ascending=False)

    # Take top_n positive and top_n negative
    pos = df[df['difference'] > 0].head(top_n)
    neg = df[df['difference'] < 0].tail(min(10, len(df[df['difference'] < 0])))
    df_plot = pd.concat([pos, neg]).sort_values('difference').reset_index(drop=True)

    colours     = []
    cat_present = {}
    for feat in df_plot['feature']:
        col, cat = get_colour(feat)
        colours.append(col)
        cat_present[cat] = col

    labels = [clean_label(f) for f in df_plot['feature']]

    # Colour bars: positive = warm (AKI-associated), negative = cool
    bar_colours = []
    for val in df_plot['difference']:
        if val > 0:
            bar_colours.append('#C00000')   # red = AKI-driving
        else:
            bar_colours.append('#2E75B6')   # blue = non-AKI-driving

    n_rows = len(df_plot)
    fig, ax = plt.subplots(figsize=(9, n_rows * 0.34 + 3))

    bars = ax.barh(range(n_rows), df_plot['difference'],
                   color=bar_colours, alpha=0.82,
                   edgecolor='white', linewidth=0.4)

    # Category colour strips on left margin
    # for i, feat in enumerate(df_plot['feature']):
    #     col, _ = get_colour(feat)
    #     ax.barh(i, 0.002 * df_plot['difference'].abs().max(),
    #             left=df_plot['difference'].min() * 1.08,
    #             color=col, alpha=1.0, height=0.7)

    ax.axvline(0, color='#333333', linewidth=0.8)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Importance difference (AKI − non-AKI)', fontsize=10)
    ax.set_title(
        f'Discriminative feature importance — top {top_n} features\n'
        'Positive = higher importance in AKI admissions',
        fontsize=11, pad=10
    )
    ax.grid(axis='x', alpha=0.25, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Annotations on bars
    max_val = df_plot['difference'].abs().max()
    for i, (val, feat) in enumerate(
        zip(df_plot['difference'], df_plot['feature'])
    ):
        ha    = 'left'  if val > 0 else 'right'
        xpos  = val + max_val * 0.01 if val > 0 else val - max_val * 0.01
        ax.text(xpos, i, f'{val:.4f}',
                va='center', ha=ha, fontsize=7, color='#333333')

    # Direction labels
    # ax.text(df_plot['difference'].max() * 0.6,
    #         n_rows - 0.5,
    #         '← AKI-associated', fontsize=8,
    #         color='#C00000', ha='center', style='italic')
    # ax.text(df_plot['difference'].min() * 0.6,
    #         n_rows - 0.5,
    #         'non-AKI-associated →', fontsize=8,
    #         color='#2E75B6', ha='center', style='italic')

    # Legend for clinical categories (colour strips)
    #add_colour_legend(ax, cat_present, loc='lower right', fontsize=7)
    plt.xlim([-1.0,0.5])
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Differential importance → {save_path}")


# ─────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--overall',  required=True,
                   help='CSV with columns: feature, importance, std')
    p.add_argument('--split',    required=True,
                   help='CSV with columns: feature, aki_importance, '
                        'nonaki_importance, difference, ratio')
    p.add_argument('--out_dir',  default='figures/explainability')
    p.add_argument('--top_n',    type=int, default=25,
                   help='Number of features per panel')
    p.add_argument('--top_n_overall', type=int, default=30,
                   help='Number of features for overall plot')
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    overall = pd.read_csv(args.overall)
    split   = pd.read_csv(args.split)

    # Validate columns
    assert 'feature'    in overall.columns, "overall needs 'feature' column"
    assert 'importance' in overall.columns, "overall needs 'importance' column"
    for col in ['feature','aki_importance','nonaki_importance','difference']:
        assert col in split.columns, f"split needs '{col}' column"

    print(f"\nLoaded {len(overall):,} features (overall)")
    print(f"Loaded {len(split):,} features (split)")
    print(f"Output → {out_dir}/\n")

    # Figure 1 — overall (appendix)
    plot_overall(
        overall,
        out_dir / 'overall_importance.png',
        top_n=args.top_n_overall,
    )

    # Figure 2 — AKI vs non-AKI (appendix)
    plot_aki_nonaki(
        split,
        out_dir / 'aki_nonaki_importance.png',
        top_n=args.top_n,
    )

    # Figure 3 — differential (main paper)
    plot_differential(
        split,
        out_dir / 'differential_importance.png',
        top_n=args.top_n,
    )

    print("\nDone.")
    print(f"  Main paper figure: {out_dir}/differential_importance.png")
    print(f"  Appendix figures:  {out_dir}/overall_importance.png")
    print(f"                     {out_dir}/aki_nonaki_importance.png")


if __name__ == '__main__':
    main()
