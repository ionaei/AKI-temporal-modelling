"""
generate_clinician_pdfs.py

Generates two PDF booklets for the clinical evaluation study:

  Option 1 — IG Plausibility Booklet
    20 AKI admissions from the test set.
    Shows top IG features at the time of first model alert.
    Clinicians rate clinical plausibility and whether they would act.

  Option 2 — Recognition Time Booklet
    10 AKI admissions from the silent deployment cohort.
    Shows EHR trajectory (labs, blood gases, procedures).
    Clinicians mark when they would first escalate care.

Usage
-----

python clinical_trajectories.py --test_preds data/models/transformer_bce/test_predictions.csv --windows     data/processed/windows_matched.csv --ig_aki data/models/transformer_bce/ig_explainability/ig_aki_importance.csv --ig_diff     data/models/transformer_bce/ig_explainability/ig_split_importance.csv --threshold 0.69 --out_dir clinician_pdfs --study option2 --cohort data/processed/cohort_matched.csv

Outputs
-------
    clinician_pdfs/
        option1_ig_plausibility.pdf
        option2_recognition_time.pdf
        case_list_CONFIDENTIAL.csv   (ground truth — do NOT share with clinicians)
"""

import argparse
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from io import BytesIO

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image, HRFlowable
)

warnings.filterwarnings('ignore')

W, H      = A4
MARGIN    = 1.8 * cm
BODY_W    = W - 2 * MARGIN   # usable width = 17.7cm for A4


# ─────────────────────────────────────────────────────────
#  Styles
# ─────────────────────────────────────────────────────────

def make_styles():
    base = getSampleStyleSheet()
    styles = {
        'title': ParagraphStyle(
            'title', parent=base['Title'],
            fontSize=16, spaceAfter=6, textColor=colors.HexColor('#1A3A5C')
        ),
        'heading': ParagraphStyle(
            'heading', parent=base['Heading2'],
            fontSize=12, spaceAfter=4, textColor=colors.HexColor('#1A3A5C')
        ),
        'subheading': ParagraphStyle(
            'subheading', parent=base['Heading3'],
            fontSize=10, spaceAfter=3, textColor=colors.HexColor('#C0392B')
        ),
        'body': ParagraphStyle(
            'body', parent=base['Normal'],
            fontSize=9, spaceAfter=4, leading=13, alignment=TA_JUSTIFY
        ),
        'small': ParagraphStyle(
            'small', parent=base['Normal'],
            fontSize=8, spaceAfter=3, textColor=colors.HexColor('#555555')
        ),
        'bold': ParagraphStyle(
            'bold', parent=base['Normal'],
            fontSize=9, spaceAfter=4, fontName='Helvetica-Bold'
        ),
        'center': ParagraphStyle(
            'center', parent=base['Normal'],
            fontSize=9, alignment=TA_CENTER
        ),
        'anon': ParagraphStyle(
            'anon', parent=base['Normal'],
            fontSize=8, textColor=colors.HexColor('#C0392B'),
            fontName='Helvetica-BoldOblique', spaceAfter=6
        ),
    }
    return styles


# ─────────────────────────────────────────────────────────
#  Utility: matplotlib figure → ReportLab Image
# ─────────────────────────────────────────────────────────

def fig_to_image(fig, width_cm=16):
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor='white')
    buf.seek(0)
    plt.close(fig)
    return Image(buf, width=width_cm * cm,
                 height=width_cm * cm * 0.55)


# ─────────────────────────────────────────────────────────
#  Core metric helpers
# ─────────────────────────────────────────────────────────

def get_lead_times(preds_df, aki_ids, threshold):
    leads = {}
    for adm_id in aki_ids:
        adm = preds_df[
            (preds_df['admission_id'] == adm_id) &
            preds_df['hours_to_onset'].notna() &
            (preds_df['hours_to_onset'] >= 0)
        ].sort_values('step_index')
        alerted = adm[adm['prob'] >= threshold]
        leads[adm_id] = float(alerted.iloc[0]['hours_to_onset']) \
                        if not alerted.empty else np.nan
    return leads


def clean_label(f):
    return (f.replace('_missing_flag', ' (missing)')
             .replace('_missing', ' (missing)')
             .replace('_mean', ' mean').replace('_last', ' last')
             .replace('_max', ' max').replace('_min', ' min')
             .replace('_trend', ' trend').replace('_alert', ' alert')
             .replace('_n', ' count').replace('poc_', '')
             .replace('proc_', '').replace('diag_', 'ICD ')
             .replace('_', ' ').strip().capitalize())


# ─────────────────────────────────────────────────────────
#  Option 1 helpers
# ─────────────────────────────────────────────────────────

def plot_alert_timepoint_ig(adm_id: str,
                             alert_step: int,
                             ig_heatmap_csv: pd.DataFrame,
                             alert_hour: float,
                             case_num: int,
                             top_n: int = 12):
    """
    Bar chart of IG attributions at EXACTLY the alert time step.
    Uses the per-admission attribution matrix saved by gradient_explain.py
    (one row per time step, one column per feature).

    ig_heatmap_csv must be the wide-format attribution CSV with columns:
        admission_id, step_index, feature_1, feature_2, ...
    """
    adm_df = ig_heatmap_csv[
        (ig_heatmap_csv['admission_id'].astype(str) == str(adm_id)) &
        (ig_heatmap_csv['step_index'] == alert_step)
    ]

    if adm_df.empty:
        return None   # caller will use population fallback

    feat_cols = [c for c in adm_df.columns
                 if c not in {'admission_id', 'step_index'}]
    vals      = adm_df[feat_cols].iloc[0].values.astype(float)
    feat_df   = pd.DataFrame({'feature': feat_cols, 'attribution': vals})
    feat_df['abs'] = feat_df['attribution'].abs()
    feat_df = feat_df.nlargest(top_n, 'abs').sort_values('attribution')

    n   = len(feat_df)
    fig, ax = plt.subplots(figsize=(10, max(4, n * 0.38)))
    cols = ['#C0392B' if v > 0 else '#2E75B6'
            for v in feat_df['attribution']]
    ax.barh(range(n), feat_df['attribution'], color=cols,
            alpha=0.82, edgecolor='white', linewidth=0.4)
    ax.set_yticks(range(n))
    ax.set_yticklabels(
        [clean_label(f) for f in feat_df['feature']], fontsize=8
    )
    ax.axvline(0, color='#333333', lw=0.8)
    ax.set_xlabel('IG attribution at alert time step\n'
                  '(red = pushes towards AKI, blue = suppresses)',
                  fontsize=9)
    ax.set_title(
        f'Case {case_num:02d} — Feature attributions at first alert\n'
        f'(hour {alert_hour:.0f} since admission)',
        fontsize=10
    )
    ax.grid(axis='x', alpha=0.2, lw=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    return fig


def rating_table(styles, case_num):
    """5-point plausibility + action rating table — fits within A4 body width."""
    s   = styles
    # Total = BODY_W = ~477pt. Col widths must sum to <= this.
    C0  = 8.5 * cm   # question column
    C15 = 1.6 * cm   # each of 5 rating columns
    data = [
        [Paragraph('<b>Question</b>', s['bold']),
         Paragraph('<b>1</b>', s['center']),
         Paragraph('<b>2</b>', s['center']),
         Paragraph('<b>3</b>', s['center']),
         Paragraph('<b>4</b>', s['center']),
         Paragraph('<b>5</b>', s['center'])],

        [Paragraph(
            'Clinical plausibility: Do these features make clinical '
            'sense as AKI risk signals in a PICU setting?', s['small']),
         Paragraph('Not\nat all', s['small']),
         Paragraph('', s['small']),
         Paragraph('Neutral', s['small']),
         Paragraph('', s['small']),
         Paragraph('Highly\nplausible', s['small'])],

        [Paragraph(
            'Would you act on this alert? (e.g. nephrology review, '
            'medication change, fluid adjustment, '
            'increased monitoring)', s['small']),
         Paragraph('Definitely\nnot', s['small']),
         Paragraph('', s['small']),
         Paragraph('Uncertain', s['small']),
         Paragraph('', s['small']),
         Paragraph('Definitely\nyes', s['small'])],
    ]

    t = Table(data,
              colWidths=[C0, C15, C15, C15, C15, C15])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1A3A5C')),
        ('TEXTCOLOR',  (0, 0), (-1, 0), colors.white),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1),
         [colors.HexColor('#F4F6F8'), colors.white]),
        ('BOX',       (0, 0), (-1, -1), 0.5, colors.HexColor('#CCCCCC')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CCCCCC')),
        ('VALIGN',    (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTSIZE',  (0, 0), (-1, -1), 8),
        ('TOPPADDING',    (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('WORDWRAP', (0, 0), (-1, -1), True),
    ]))
    return t


# ─────────────────────────────────────────────────────────
#  Option 2 helpers
# ─────────────────────────────────────────────────────────

LAB_PLOT = {
    'creatinine_last':   ('Creatinine (µmol/L)', '#C0392B'),
    'sodium_poc_last':   ('Sodium (mmol/L)',      '#2E75B6'),
    'lactate_last':      ('Lactate (mmol/L)',      '#E67E22'),
    'haematocrit_poc_last': ('Haematocrit (%)',   '#8E44AD'),
    'chloride_last':     ('Chloride (mmol/L)',     '#27AE60'),
    'urea_last':         ('Urea (mmol/L)',          '#16A085'),
}
GAS_PLOT = {
    'tao2_last':         ('TAO2 (mL/dL)', '#E74C3C'),
    'po2_last':          ('PO2 (kPa)',     '#2E75B6'),
    'ph_last':           ('pH',                       '#27AE60'),
    'base_excess_last':  ('Base excess (mmol/L)',      '#8E44AD'),
    'anion_gap_last':    ('Anion gap (mmol/L)',        '#D35400'),
}
PROC_PLOT = [
    ('proc_cardiac_surgery',        'Cardiac surgery',    '#C0392B'),
    ('proc_central_line',           'Central line',       '#E67E22'),
    ('proc_mechanical_vent',        'Mechanical vent',    '#2E75B6'),
    ('proc_neurosurgery',           'Neurosurgery',       '#8E44AD'),
    ('proc_bypass_circuit',         'Bypass circuit',     '#7F8C8D'),
    ('proc_renal_procedure', 'Renal procedure', '#1ABC9C'),
    ('proc_transplant', 'Transplant', '#D35400'),
    ('proc_imaging_contrast', 'Contrast imaging', '#2980B9'),
    ('proc_abdominal_surgery', 'Abdominal surgery', '#F39C12'),
    ('proc_orthopaedic', 'Orthopaedic', '#27AE60'),
    ('proc_bronchoscopy_scope', 'Bronchoscopy', '#6C3483')
    
]

def plot_trajectory(adm_df, case_num, windows_raw):
    """3-panel EHR trajectory plot for one admission."""
    T = adm_df['hours_since_admit'].values

    fig = plt.figure(figsize=(15, 10))
    fig.patch.set_facecolor('white')
    gs = gridspec.GridSpec(3, 2, figure=fig,
                           height_ratios=[2.5, 2.5, 1.6],
                           hspace=0.5, wspace=0.32)

    # Panel 1: Lab values
    ax1 = fig.add_subplot(gs[0, :])
    for col, (label, colour) in LAB_PLOT.items():
        if col not in adm_df.columns:
            continue
        vals = adm_df[col].copy()
        miss = col.replace('_last', '_missing_flag')
        if miss in adm_df.columns:
            vals[adm_df[miss] == 1] = np.nan
        ax1.plot(T, vals, 'o-', color=colour, lw=1.8,
                 ms=3, label=label, alpha=0.85)
    ax1.set_xlabel('Hours since admission', fontsize=9)
    ax1.set_ylabel('Value (clinical units)', fontsize=9)
    ax1.set_title(f'Case {case_num:02d} — Laboratory values',
                  fontsize=10, fontweight='bold')
    ax1.legend(fontsize=7, loc='upper right', framealpha=0.7, ncol=3)
    ax1.grid(alpha=0.2)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Panel 2: Blood gases
    ax2 = fig.add_subplot(gs[1, :])
    for col, (label, colour) in GAS_PLOT.items():
        if col not in adm_df.columns:
            continue
        vals = adm_df[col].copy()
        miss = col.replace('_last', '_missing_flag')
        if miss in adm_df.columns:
            vals[adm_df[miss] == 1] = np.nan
        ax2.plot(T, vals, 's-', color=colour, lw=1.8,
                 ms=3, label=label, alpha=0.85)
    ax2.set_xlabel('Hours since admission', fontsize=9)
    ax2.set_ylabel('Value (clinical units)', fontsize=9)
    ax2.set_title('Blood gas and oxygen parameters',
                  fontsize=10, fontweight='bold')
    ax2.legend(fontsize=7, loc='upper right', framealpha=0.7, ncol=3)
    ax2.grid(alpha=0.2)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Panel 3: Procedures
    ax3 = fig.add_subplot(gs[2, :])
    ax3.set_xlim(T[0] - 1, T[-1] + 1)
    n_rows = len(PROC_PLOT)
    ax3.set_ylim(-0.8, n_rows - 0.2)
    ax3.set_yticks(range(n_rows))
    ax3.set_yticklabels([p[1] for p in PROC_PLOT], fontsize=8)
    ax3.set_xlabel('Hours since admission', fontsize=9)
    ax3.set_title('Procedure and medication exposure',
                  fontsize=10, fontweight='bold')

    for col, label, colour, *_ in [(p[0], p[1], p[2]) for p in PROC_PLOT]:
        if col not in adm_df.columns:
            continue
        y_pos = [p[1] for p in PROC_PLOT].index(label)
        active = adm_df[adm_df[col] > 0]['hours_since_admit'].values
        if len(active) > 0:
            ax3.scatter(active, [y_pos]*len(active),
                       marker='|', s=280, lw=2.5, color=colour, zorder=3)
            if len(active) > 1:
                ax3.barh(y_pos, active[-1]-active[0],
                         left=active[0], height=0.3,
                         color=colour, alpha=0.15, zorder=1)

    ax3.grid(axis='x', alpha=0.2)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    plt.suptitle(
        f'ANONYMISED — FOR RESEARCH USE ONLY — Case {case_num:02d}',
        fontsize=10, color='#C0392B', fontweight='bold', y=1.01
    )
    plt.tight_layout()
    return fig


def recognition_response_table(styles):
    C0 = 11 * cm
    C1 =  6 * cm
    data = [
        [Paragraph('<b>Response</b>', styles['bold']),
         Paragraph('<b>Your answer</b>', styles['bold'])],
        [Paragraph(
            'At what hour since admission would you first escalate '
            'care, request nephrology review, or suspect AKI?',
            styles['small']),
         Paragraph('Hour since admission: __________', styles['small'])],
        [Paragraph(
            'I had no concern about AKI during this admission.',
            styles['small']),
         Paragraph('&#9744; Yes   &#9744; No', styles['small'])],
        [Paragraph(
            'What clinical feature(s) prompted your concern? (free text)',
            styles['small']),
         Paragraph('\n\n\n', styles['small'])],
    ]
    t = Table(data, colWidths=[C0, C1])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1A3A5C')),
        ('TEXTCOLOR',  (0, 0), (-1, 0), colors.white),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1),
         [colors.HexColor('#F4F6F8'), colors.white]),
        ('BOX',       (0, 0), (-1, -1), 0.5, colors.HexColor('#CCCCCC')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CCCCCC')),
        ('VALIGN',    (0, 0), (-1, -1), 'TOP'),
        ('FONTSIZE',  (0, 0), (-1, -1), 8),
        ('TOPPADDING',    (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('ROWHEIGHT',  (0, 3), (-1, 3), 2*cm),
    ]))
    return t


# ─────────────────────────────────────────────────────────
#  Option 1 — IG Plausibility PDF
# ─────────────────────────────────────────────────────────

def build_option1_pdf(cases_o1, preds_df, ig_diff_df,
                       threshold, out_path, styles,
                       ig_attr_csv=None):
    """
    ig_attr_csv: wide-format DataFrame with columns
        admission_id, step_index, <feature_1>, <feature_2>, ...
        This is the per-step attribution matrix from gradient_explain.py.
        If None, falls back to population-level differential.
    """
    doc  = SimpleDocTemplate(
        str(out_path), pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN,
    )
    story = []
    s = styles

    # Cover page
    story += [
        Spacer(1, 2*cm),
        Paragraph('CLINICAL EVALUATION STUDY', s['title']),
        Paragraph('Option 1: Feature Importance Plausibility Rating',
                  s['heading']),
        HRFlowable(width='100%', thickness=1,
                   color=colors.HexColor('#1A3A5C')),
        Spacer(1, 0.5*cm),
        Paragraph(
            'This booklet presents 20 anonymised AKI admissions from a '
            'paediatric intensive care unit. For each case, the model '
            'generated an alert before KDIGO-defined AKI onset. The bar '
            'chart shows which clinical features drove the model\'s '
            'prediction at the exact time of the first alert for that '
            'specific patient.',
            s['body']),
        Spacer(1, 0.3*cm),
        Paragraph(
            'Red bars indicate features pushing the prediction towards '
            'AKI at that time step; blue bars indicate features '
            'suppressing the prediction. Only the alert moment is shown '
            '— not the full admission trajectory.',
            s['body']),
        Spacer(1, 0.3*cm),
        Paragraph(
            'For each case, please rate: (1) whether the highlighted '
            'features make clinical sense as AKI risk signals at this '
            'time point, and (2) whether you would have taken clinical '
            'action based on this pattern.',
            s['body']),
        Spacer(1, 0.3*cm),
        Paragraph(
            'The outcome (whether AKI occurred) and the model\'s '
            'predicted probability are withheld.',
            s['anon']),
        Spacer(1, 0.5*cm),
        Paragraph(
            'Scale: 1 = Not at all / Definitely not  |  '
            '3 = Neutral / Uncertain  |  '
            '5 = Highly plausible / Definitely yes',
            s['small']),
        PageBreak(),
    ]

    for i, case in enumerate(cases_o1):
        adm_id      = str(case['admission_id'])
        alert_h     = case.get('alert_hour', np.nan)
        alert_step  = case.get('alert_step', None)
        lead_h      = case.get('lead_h', np.nan)
        age         = case.get('age', '?')
        sex         = case.get('sex', '?')
        case_num    = i + 1

        story.append(Paragraph('ANONYMISED — FOR RESEARCH USE ONLY', s['anon']))
        story.append(Paragraph(
            f'Case {case_num:02d}   |   Age: {age}   |   Sex: {sex}   |   '
            f'Alert at: '
            f'{"~" + str(int(alert_h)) + " h" if not np.isnan(alert_h) else "?"}  '
            f'since admission',
            s['bold']))
        story.append(Spacer(1, 0.2*cm))

        # ── Per-patient, single-timepoint IG ──────────────
        fig = None
        if ig_attr_csv is not None and alert_step is not None:
            fig = plot_alert_timepoint_ig(
                adm_id, alert_step, ig_attr_csv,
                alert_h if not np.isnan(alert_h) else 0,
                case_num, top_n=12
            )

        # Population fallback if per-patient attribution unavailable
        if fig is None:
            story.append(Paragraph(
                '(Per-patient attribution not available for this case — '
                'showing population-level discriminative features.)',
                s['small']))
            top_feats = ig_diff_df.nlargest(10, 'difference')[
                ['feature', 'difference']
            ].to_dict('records')
            n = len(top_feats)
            f2, ax = plt.subplots(figsize=(10, max(4, n * 0.38)))
            labels = [clean_label(f['feature']) for f in top_feats]
            vals   = [f['difference'] for f in top_feats]
            cols   = ['#C0392B' if v > 0 else '#2E75B6' for v in vals]
            ax.barh(range(n), vals, color=cols, alpha=0.82,
                    edgecolor='white', linewidth=0.4)
            ax.set_yticks(range(n))
            ax.set_yticklabels(labels, fontsize=8)
            ax.axvline(0, color='#333333', lw=0.8)
            ax.set_xlabel('Mean |IG| difference (AKI − non-AKI)', fontsize=9)
            ax.set_title(f'Case {case_num:02d} — Population-level features',
                         fontsize=10)
            ax.grid(axis='x', alpha=0.2); ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            fig = f2

        story.append(fig_to_image(fig, width_cm=14.5))
        story.append(Spacer(1, 0.2*cm))
        story.append(Paragraph(
            f'The model first alerted at approximately '
            f'{"hour " + str(int(alert_h)) if not np.isnan(alert_h) else "an early time point"} '
            f'since admission'
            + (f', providing a lead time of approximately '
               f'{int(lead_h)} hours before KDIGO-defined AKI onset.'
               if not np.isnan(lead_h) else '.'),
            s['small']))
        story.append(Spacer(1, 0.3*cm))

        story.append(Paragraph('Your ratings:', s['bold']))
        story.append(Spacer(1, 0.1*cm))
        story.append(rating_table(s, case_num))
        story.append(Spacer(1, 0.2*cm))

        story.append(Paragraph('Additional comments (optional):', s['bold']))
        freetext = Table(
            [[Paragraph('', s['small'])]],
            colWidths=[BODY_W], rowHeights=[1.4*cm]
        )
        freetext.setStyle(TableStyle([
            ('BOX', (0,0),(-1,-1), 0.75, colors.HexColor('#AAAAAA')),
            ('BACKGROUND', (0,0),(-1,-1), colors.HexColor('#FAFAFA')),
        ]))
        story.append(freetext)

        if i < len(cases_o1) - 1:
            story.append(PageBreak())

    doc.build(story)
    print(f"  Option 1 PDF saved: {out_path}")
    doc  = SimpleDocTemplate(
        str(out_path), pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN,
    )
    story = []
    s = styles

    # Cover page
    story += [
        Spacer(1, 2*cm),
        Paragraph('CLINICAL EVALUATION STUDY', s['title']),
        Paragraph('Option 1: Feature Importance Plausibility Rating',
                  s['heading']),
        HRFlowable(width='100%', thickness=1,
                   color=colors.HexColor('#1A3A5C')),
        Spacer(1, 0.5*cm),
        Paragraph(
            'This booklet presents 20 anonymised AKI admissions from a '
            'paediatric intensive care unit. For each case, the most '
            'influential predictive features at the time of the model\'s '
            'first alert are shown as a temporal attribution heatmap '
            '(rows = features, columns = 6-hour windows). Red cells '
            'indicate features pushing the prediction towards AKI at '
            'that time step; blue cells indicate features suppressing '
            'the prediction.',
            s['body']),
        Spacer(1, 0.3*cm),
        Paragraph(
            'For each case, please rate: (1) whether the highlighted '
            'features make clinical sense as AKI risk signals, and '
            '(2) whether you would have taken clinical action based on '
            'this pattern at the time indicated.',
            s['body']),
        Spacer(1, 0.3*cm),
        Paragraph(
            'The outcome (whether or not AKI subsequently occurred) and '
            'the model\'s predicted probability are withheld. Please '
            'respond based only on the clinical features shown.',
            s['anon']),
        Spacer(1, 0.5*cm),
        Paragraph(
            'Scale: 1 = Not at all / Definitely not  |  '
            '3 = Neutral / Uncertain  |  5 = Highly plausible / Definitely yes',
            s['small']),
        PageBreak(),
    ]

    heatmap_dir = Path(heatmap_dir) if heatmap_dir else None

    for i, case in enumerate(cases_o1):
        adm_id   = str(case['admission_id'])
        alert_h  = case.get('alert_hour', np.nan)
        lead_h   = case.get('lead_h', np.nan)
        age      = case.get('age', '?')
        sex      = case.get('sex', '?')
        case_num = i + 1

        story.append(Paragraph('ANONYMISED — FOR RESEARCH USE ONLY', s['anon']))
        story.append(Paragraph(
            f'Case {case_num:02d}   |   Age: {age}   |   Sex: {sex}   |   '
            f'First alert at: '
            f'{"~" + str(int(alert_h)) + " h" if not np.isnan(alert_h) else "?"}  '
            f'since admission',
            s['bold']))
        story.append(Spacer(1, 0.2*cm))

        # ── Try to load per-admission heatmap (personalised) ──
        heatmap_loaded = False
        if heatmap_dir is not None:
            # gradient_explain.py names files heatmap_{adm_id}.png
            candidates = list(heatmap_dir.glob(f'heatmap_{adm_id}*.png'))
            if candidates:
                img_path = candidates[0]
                story.append(Paragraph(
                    f'Temporal attribution heatmap — features at each '
                    f'6-hour window (this patient only):',
                    s['small']))
                story.append(Spacer(1, 0.1*cm))
                story.append(Image(
                    str(img_path),
                    width=BODY_W,
                    height=BODY_W * 0.55,
                ))
                heatmap_loaded = True

        # ── Fallback: population-level differential bar chart ──
        if not heatmap_loaded:
            top_feats = ig_diff_df.nlargest(10, 'difference')[
                ['feature', 'difference']
            ].to_dict('records')
            fig = plot_ig_features(
                top_feats,
                alert_h if not np.isnan(alert_h) else 0,
                case_num
            )
            story.append(Paragraph(
                'Population-level discriminative features '
                '(per-admission heatmap not available for this case):',
                s['small']))
            story.append(fig_to_image(fig, width_cm=14.5))

        story.append(Spacer(1, 0.3*cm))
        story.append(Paragraph(
            f'The model first alerted at approximately '
            f'{"hour " + str(int(alert_h)) if not np.isnan(alert_h) else "an early time point"} '
            f'since admission, '
            f'{"providing a lead time of ~" + str(int(lead_h)) + " hours before KDIGO-defined AKI onset." if not np.isnan(lead_h) else "."}',
            s['small']))
        story.append(Spacer(1, 0.3*cm))

        # Rating table
        story.append(Paragraph('Your ratings:', s['bold']))
        story.append(Spacer(1, 0.1*cm))
        story.append(rating_table(s, case_num))
        story.append(Spacer(1, 0.25*cm))

        # Free text — use a fixed-height bordered box instead of underscores
        story.append(Paragraph('Additional comments (optional):', s['bold']))
        freetext = Table(
            [[Paragraph('', s['small'])]],
            colWidths=[BODY_W],
            rowHeights=[1.4*cm]
        )
        freetext.setStyle(TableStyle([
            ('BOX', (0,0),(-1,-1), 0.75, colors.HexColor('#AAAAAA')),
            ('BACKGROUND', (0,0),(-1,-1), colors.HexColor('#FAFAFA')),
        ]))
        story.append(freetext)

        if i < len(cases_o1) - 1:
            story.append(PageBreak())

    doc.build(story)
    print(f"  Option 1 PDF saved: {out_path}")


# ─────────────────────────────────────────────────────────
#  Option 2 — Recognition Time PDF
# ─────────────────────────────────────────────────────────

def build_option2_pdf(cases_o2, windows_raw, out_path, styles):
    doc  = SimpleDocTemplate(
        str(out_path), pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN,
    )
    story = []
    s = styles

    # Cover page
    story += [
        Spacer(1, 2*cm),
        Paragraph('CLINICAL EVALUATION STUDY', s['title']),
        Paragraph('Option 2: Clinical Recognition Time Task', s['heading']),
        HRFlowable(width='100%', thickness=1,
                   color=colors.HexColor('#1A3A5C')),
        Spacer(1, 0.5*cm),
        Paragraph(
            'This booklet presents 10 anonymised admissions from a '
            'paediatric intensive care unit. For each case, the full '
            'EHR trajectory is shown (laboratory values, blood gases, '
            'and procedure/medication exposure) from the point of '
            'admission.',
            s['body']),
        Spacer(1, 0.3*cm),
        Paragraph(
            'Please review each trajectory and indicate: at what hour '
            'since admission would you first escalate care, request '
            'nephrology review, or suspect that this patient was at '
            'risk of developing acute kidney injury?',
            s['body']),
        Spacer(1, 0.3*cm),
        Paragraph(
            'The clinical outcome of each admission is withheld. '
            'Please respond based only on the trajectory shown. '
            'There are no correct or incorrect answers — we are '
            'interested in your genuine clinical assessment.',
            s['anon']),
        Spacer(1, 0.3*cm),
        Paragraph(
            'IMPORTANT: Please complete this booklet BEFORE the '
            'Feature Importance Plausibility booklet (Option 1).',
            s['bold']),
        PageBreak(),
    ]

    # One case per page
    for i, case in enumerate(cases_o2):
        adm_id   = case['admission_id']
        age      = case.get('age', '?')
        sex      = case.get('sex', '?')
        case_num = i + 1

        adm_df = (windows_raw[windows_raw['admission_id'] == adm_id]
                  .sort_values('hours_since_admit').copy())

        if len(adm_df) == 0:
            print(f"  WARNING: no windows for {adm_id}")
            continue

        # Trajectory plot
        fig = plot_trajectory(adm_df, case_num, windows_raw)
        story.append(Paragraph('ANONYMISED — FOR RESEARCH USE ONLY', s['anon']))
        story.append(Paragraph(
            f'Case {case_num:02d}   |   Age: {age}   |   Sex: {sex}   |   '
            f'Admission duration shown: {adm_df["hours_since_admit"].max():.0f} h',
            s['bold']))
        story.append(Spacer(1, 0.2*cm))
        story.append(fig_to_image(fig, width_cm=16))
        story.append(Spacer(1, 0.3*cm))

        # Response table
        story.append(Paragraph('Your response:', s['bold']))
        story.append(Spacer(1, 0.1*cm))
        story.append(recognition_response_table(s))

        # Free text box for reasoning
        story.append(Spacer(1, 0.2*cm))
        story.append(Paragraph('Clinical reasoning (free text):', s['bold']))
        freetext = Table(
            [[Paragraph('', s['small'])]],
            colWidths=[BODY_W],
            rowHeights=[2.0*cm]
        )
        freetext.setStyle(TableStyle([
            ('BOX', (0,0),(-1,-1), 0.75, colors.HexColor('#AAAAAA')),
            ('BACKGROUND', (0,0),(-1,-1), colors.HexColor('#FAFAFA')),
        ]))
        story.append(freetext)

        if i < len(cases_o2) - 1:
            story.append(PageBreak())

    doc.build(story)
    print(f"  Option 2 PDF → {out_path}")

def save_option2_images(cases_o2, windows_raw, out_dir: Path):
    """
    Save three PNG files per case — each with subplots for that panel.
        case_01_labs.png       — 6 lab subplots with fixed y-axes + normal range
        case_01_gases.png      — 5 blood gas subplots with fixed y-axes + normal range
        case_01_procedures.png — procedure timeline (unchanged)
    """
    img_dir = out_dir / 'option2_images'
    img_dir.mkdir(parents=True, exist_ok=True)

    LAB_GROUPS = [
        {'col': 'creatinine_last',     'label': 'Creatinine\n(µmol/L)',    'colour': '#C0392B', 'ymin': 0,   'ymax': 500, 'normal': (20, 80)},
        {'col': 'sodium_poc_last',      'label': 'Sodium\n(mmol/L)',        'colour': '#2E75B6', 'ymin': 115, 'ymax': 165, 'normal': (135, 145)},
        {'col': 'lactate_last',         'label': 'Lactate\n(mmol/L)',       'colour': '#E67E22', 'ymin': 0,   'ymax': 15,  'normal': (0.5, 2.0)},
        {'col': 'haematocrit_poc_last', 'label': 'Haematocrit\n(%)',        'colour': '#8E44AD', 'ymin': 10,  'ymax': 60,  'normal': (35, 45)},
        {'col': 'urea_last',            'label': 'Urea\n(mmol/L)',          'colour': '#16A085', 'ymin': 0,   'ymax': 30,  'normal': (2.5, 7.5)},
        {'col': 'chloride_last',        'label': 'Chloride\n(mmol/L)',      'colour': '#27AE60', 'ymin': 90,  'ymax': 120, 'normal': (98, 106)},
    ]

    GAS_GROUPS = [
        {'col': 'ph_last',          'label': 'pH',                   'colour': '#27AE60', 'ymin': 6.9,  'ymax': 7.7,  'normal': (7.35, 7.45)},
        {'col': 'tao2_last',        'label': 'TAO₂ (mL/dL)',        'colour': '#E74C3C', 'ymin': 0,    'ymax': 25,   'normal': None},
        {'col': 'po2_last',         'label': 'PO₂ (kPa)',           'colour': '#2E75B6', 'ymin': 0,    'ymax': 30,   'normal': (10, 13)},
        {'col': 'base_excess_last', 'label': 'Base excess\n(mmol/L)', 'colour': '#8E44AD', 'ymin': -25,  'ymax': 10,   'normal': (-2, 2)},
        {'col': 'anion_gap_last',   'label': 'Anion gap\n(mmol/L)',  'colour': '#D35400', 'ymin': 0,    'ymax': 30,   'normal': (8, 16)},
    ]

    def plot_subpanel(ax, T, adm_df, grp):
        col    = grp['col']
        colour = grp['colour']
        if col not in adm_df.columns:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=8, color='#AAAAAA')
            ax.set_ylabel(grp['label'], fontsize=8)
            ax.set_xlabel('Hours', fontsize=8)
            return
        vals = adm_df[col].copy().astype(float)
        miss = col.replace('_last', '_missing_flag')
        if miss in adm_df.columns:
            vals[adm_df[miss] == 1] = np.nan
        ax.plot(T, vals, 'o-', color=colour, lw=2, ms=5, alpha=0.9)
        if grp.get('normal'):
            lo, hi = grp['normal']
            ax.axhspan(lo, hi, alpha=0.10, color='green', zorder=0)
            ax.axhline(lo, color='green', lw=0.8, linestyle='--', alpha=0.5)
            ax.axhline(hi, color='green', lw=0.8, linestyle='--', alpha=0.5)
            mid = (lo + hi) / 2
            ax.annotate(f'Normal\n{lo}–{hi}',
                        xy=(T[0], mid), fontsize=6,
                        color='green', alpha=0.7, va='center')
        ax.set_ylim(grp['ymin'], grp['ymax'])
        ax.set_ylabel(grp['label'], fontsize=8)
        ax.set_xlabel('Hours since admission', fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    for i, case in enumerate(cases_o2):
        adm_id   = case['admission_id']
        age      = case.get('age', '?')
        sex      = case.get('sex', '?')
        case_num = i + 1
        prefix   = img_dir / f'case_{case_num:02d}'

        adm_df = (windows_raw[windows_raw['admission_id'] == adm_id]
                  .sort_values('hours_since_admit').copy())

        if len(adm_df) == 0:
            print(f"  WARNING: no windows for case {case_num} ({adm_id})")
            continue

        T = adm_df['hours_since_admit'].values
        hdr = (f'Case {case_num:02d}  |  Age: {age}  |  Sex: {sex}  '
               f'|  ANONYMISED — FOR RESEARCH USE ONLY')

        # ── Labs: 2 rows × 3 cols of subplots ────────────
        fig, axes = plt.subplots(2, 3, figsize=(16, 8))
        fig.patch.set_facecolor('white')
        for j, grp in enumerate(LAB_GROUPS):
            ax = axes[j // 3, j % 3]
            plot_subpanel(ax, T, adm_df, grp)
        fig.suptitle(f'{hdr}\nLaboratory values over time',
                     fontsize=10, fontweight='bold', y=1.02, color='#1A3A5C')
        plt.tight_layout()
        fig.savefig(f'{prefix}_labs.png', dpi=180,
                    bbox_inches='tight', facecolor='white')
        plt.close(fig)

        # ── Blood gases: 1 row × 5 cols of subplots ──────
        fig, axes = plt.subplots(1, 5, figsize=(18, 4.5))
        fig.patch.set_facecolor('white')
        for j, grp in enumerate(GAS_GROUPS):
            plot_subpanel(axes[j], T, adm_df, grp)
        fig.suptitle(f'{hdr}\nBlood gas and oxygen parameters',
                     fontsize=10, fontweight='bold', y=1.05, color='#1A3A5C')
        plt.tight_layout()
        fig.savefig(f'{prefix}_gases.png', dpi=180,
                    bbox_inches='tight', facecolor='white')
        plt.close(fig)

        # ── Procedures: unchanged single plot ─────────────
        n_rows = len(PROC_PLOT)
        fig, ax = plt.subplots(figsize=(14, max(4, n_rows * 0.55)))
        fig.patch.set_facecolor('white')
        ax.set_xlim(T[0] - 1, T[-1] + 1)
        ax.set_ylim(-0.8, n_rows - 0.2)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels([p[1] for p in PROC_PLOT], fontsize=9)
        ax.set_xlabel('Hours since admission', fontsize=11)
        ax.set_title(f'{hdr}\nProcedure and medication exposure',
                     fontsize=10, fontweight='bold', color='#1A3A5C')
        for col, label, colour in [(p[0], p[1], p[2]) for p in PROC_PLOT]:
            if col not in adm_df.columns:
                continue
            y_pos  = [p[1] for p in PROC_PLOT].index(label)
            active = adm_df[adm_df[col] > 0]['hours_since_admit'].values
            if len(active) > 0:
                ax.scatter(active, [y_pos]*len(active),
                           marker='|', s=350, lw=3, color=colour, zorder=3)
                if len(active) > 1:
                    ax.barh(y_pos, active[-1]-active[0],
                            left=active[0], height=0.35,
                            color=colour, alpha=0.15, zorder=1)
        ax.grid(axis='x', alpha=0.2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        fig.savefig(f'{prefix}_procedures.png', dpi=180,
                    bbox_inches='tight', facecolor='white')
        plt.close(fig)

        print(f"  Case {case_num:02d} → _labs.png  _gases.png  _procedures.png")

    print(f"\n  All images saved to {img_dir}/")


def has_enough_data(adm_id:        str,
                    windows_df:    pd.DataFrame,
                    min_windows:   int   = 4,
                    min_duration_h: float = 12.0) -> bool:
    """
    Returns True if admission has enough plotted data for a
    meaningful clinical trajectory review.

    Checks that:
      1. At least min_windows 6-hour windows span the admission
      2. Admission duration >= min_duration_h
      3. At least 3 of the plotted lab/gas columns have
         >= 2 real (non-missing-flagged) observations
    """
    adm = windows_df[windows_df['admission_id'].astype(str) == str(adm_id)]
    if len(adm) < min_windows:
        return False
    duration = adm['hours_since_admit'].max() - adm['hours_since_admit'].min()
    if duration < min_duration_h:
        return False

    # Columns shown in the trajectory plots
    PLOTTED_COLS = list(LAB_PLOT.keys()) + list(GAS_PLOT.keys())

    # For each column, check if it has >= 2 real observations
    # (i.e. missing flag is 0 or missing flag column doesn't exist)
    n_rich_cols = 0
    for col in PLOTTED_COLS:
        if col not in adm.columns:
            continue
        miss_col = col.replace('_last', '_missing_flag')
        if miss_col in adm.columns:
            # Count windows where the value was actually measured
            n_real = (adm[miss_col] == 0).sum()
        else:
            # No missing flag — count non-NaN values
            n_real = adm[col].notna().sum()
        if n_real >= 2:
            n_rich_cols += 1

    # Require at least 3 plotted columns to have real data
    return n_rich_cols >= 3

# ─────────────────────────────────────────────────────────
#  Case selection
# ─────────────────────────────────────────────────────────

def select_cases(preds_df, windows, threshold, deploy_preds=None,
                 deploy_wins=None, seed=42, demo_df=None):
    rng = np.random.default_rng(seed)

    aki_ids = set(preds_df[preds_df['label'] == 1]['admission_id'].unique())
    leads   = get_lead_times(preds_df, aki_ids, threshold)
    lead_df = pd.DataFrame.from_dict(
        leads, orient='index', columns=['lead_h']
    ).reset_index().rename(columns={'index': 'admission_id'})
    lead_df['detected'] = lead_df['lead_h'].notna()

    # Get alert hours AND alert step_index for per-patient IG lookup
    alert_info = {}
    for adm_id in aki_ids:
        adm = preds_df[
            (preds_df['admission_id'] == adm_id) &
            preds_df['hours_to_onset'].notna()
        ].sort_values('step_index')
        alerted = adm[adm['prob'] >= threshold]
        if not alerted.empty:
            row = alerted.iloc[0]
            alert_info[adm_id] = {
                'alert_hour': float(
                    row['hours_since_admit']
                    if 'hours_since_admit' in row.index
                    else row['step_index'] * 6
                ),
                'alert_step': int(row['step_index']),
            }

    # Demographic enrichment — prefer cohort_matched (has sex),
    # fall back to windows if cohort not provided
    if demo_df is not None:
        demo = demo_df
    else:
        demo_cols = ['admission_id']
        for col in ['age_years', 'sex']:
            if col in windows.columns:
                demo_cols.append(col)
        demo = windows[demo_cols].drop_duplicates('admission_id') \
               if len(demo_cols) > 1 else None

    def enrich(adm_id):
        row = {
            'admission_id': adm_id,
            'lead_h':       leads.get(adm_id, np.nan),
            'alert_hour':   alert_info.get(adm_id, {}).get('alert_hour', np.nan),
            'alert_step':   alert_info.get(adm_id, {}).get('alert_step', None),
        }
        if demo is not None:
            d = demo[demo['admission_id'] == adm_id]
            if not d.empty:
                if 'age_years' in d.columns:
                    row['age'] = f"{d['age_years'].iloc[0]:.1f} years"
                if 'sex' in d.columns:
                    row['sex'] = str(d['sex'].iloc[0])
        return row

    # ── Option 1: 20 detected AKI cases from test set ────
    det = [a for a in lead_df[lead_df['detected']]['admission_id'].tolist()
           if has_enough_data(a, windows)]
    print(f"  Detected AKI with sufficient data: {len(det)}")    
    rng.shuffle(det)
    early  = [a for a in det if leads.get(a, 0) >= 48][:7]
    medium = [a for a in det
              if 24 <= leads.get(a, 0) < 48
              and a not in early][:7]
    late   = [a for a in det
              if leads.get(a, 0) < 24
              and a not in early + medium][:6]
    o1_ids = early + medium + late
    rng.shuffle(o1_ids)
    cases_o1 = [enrich(a) for a in o1_ids[:20]]

    # ── Option 2: MIXED AKI + non-AKI, all blinded ───────
    # 6 detected AKI cases (NOT overlapping with Option 1)
    # 4 non-AKI cases
    # Clinicians do not know which are AKI — must infer from trajectory
    all_ids     = list(preds_df['admission_id'].unique())
    nonaki_ids = [a for a in all_ids
                  if a not in aki_ids
                  and has_enough_data(a, windows)]
    print(f"  Non-AKI with sufficient data: {len(nonaki_ids)}")

    o2_aki_pool = [a for a in det if a not in o1_ids]
    rng.shuffle(o2_aki_pool)
    rng.shuffle(nonaki_ids)

    o2_aki    = o2_aki_pool[:12]
    o2_nonaki = nonaki_ids[:8]
    o2_all    = o2_aki + o2_nonaki
    rng.shuffle(o2_all)   # shuffle so clinicians cannot infer order = outcome

    def enrich_o2(adm_id):
        row = {
            'admission_id': adm_id,
            'is_aki':       adm_id in aki_ids,   # CONFIDENTIAL — not shown
            'lead_h':       leads.get(adm_id, np.nan),
        }
        if demo is not None:
            d = demo[demo['admission_id'] == adm_id]
            if not d.empty:
                if 'age_years' in d.columns:
                    row['age'] = f"{d['age_years'].iloc[0]:.1f} years"
                if 'sex' in d.columns:
                    row['sex'] = str(d['sex'].iloc[0])
        return row

    if deploy_preds is not None and deploy_wins is not None:
        # Prefer silent deployment data for Option 2
        dep_aki   = set(
            deploy_preds[deploy_preds['label']==1]['admission_id'].unique()
        )
        dep_all   = list(deploy_preds['admission_id'].unique())
        dep_naki = [a for a in dep_all
                if a not in dep_aki
                and has_enough_data(a, deploy_wins)]
        dep_leads = get_lead_times(deploy_preds, dep_aki, threshold)
        dep_det = [a for a, v in dep_leads.items()
               if v is not None and not np.isnan(v) and v >= 24
               and has_enough_data(a, deploy_wins)]
        rng.shuffle(dep_det)
        rng.shuffle(dep_naki)

        dep_demo_cols = ['admission_id']
        for col in ['age_years', 'sex']:
            if col in deploy_wins.columns:
                dep_demo_cols.append(col)
        dep_demo = deploy_wins[dep_demo_cols].drop_duplicates('admission_id') \
                   if len(dep_demo_cols) > 1 else None

        def enrich_dep(adm_id):
            row = {
                'admission_id': adm_id,
                'is_aki':       adm_id in dep_aki,
                'lead_h':       dep_leads.get(adm_id, np.nan),
            }
            if dep_demo is not None:
                d = dep_demo[dep_demo['admission_id'] == adm_id]
                if not d.empty:
                    if 'age_years' in d.columns:
                        row['age'] = f"{d['age_years'].iloc[0]:.1f} years"
                    if 'sex' in d.columns:
                        row['sex'] = str(d['sex'].iloc[0])
            return row

        dep_o2 = dep_det[:6] + dep_naki[:4]
        rng.shuffle(dep_o2)
        cases_o2 = [enrich_dep(a) for a in dep_o2]
    else:
        cases_o2 = [enrich_o2(a) for a in o2_all]

    return cases_o1, cases_o2


# ─────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--test_preds',   required=True)
    p.add_argument('--windows',      required=True)
    p.add_argument('--ig_aki',       required=True,
                   help='ig_aki_importance.csv from gradient_explain.py')
    p.add_argument('--ig_diff',      required=True,
                   help='ig_split_importance.csv from gradient_explain.py')
    p.add_argument('--ig_attr', default=None,
                   help='Wide-format per-admission, per-step attribution CSV. '
                        'Expected columns: admission_id, step_index, '
                        '<feature_1>, <feature_2>, ... '
                        'Generate with gradient_explain.py '
                        '--save_attr_matrix flag. '
                        'Enables single-timepoint personalised IG chart. '
                        'Falls back to population-level IG if absent.')
    p.add_argument('--heatmap_dir', default=None,
                   help='[Deprecated — use --ig_attr instead]')
    p.add_argument('--deploy_preds', default=None,
                   help='deployment_predictions.csv (optional — '
                        'falls back to test set if absent)')
    p.add_argument('--deploy_wins',  default=None,
                   help='deployment windows CSV (optional)')
    p.add_argument('--threshold',    type=float, required=True)
    p.add_argument('--out_dir',      default='clinician_pdfs')
    p.add_argument('--study', choices=['option1', 'option2', 'both'],
                   default='both',
                   help='Which study to generate: option1 (IG plausibility), '
                        'option2 (recognition time), or both.')
    p.add_argument('--cohort', default=None,
                   help='cohort_matched.csv — contains age_years and sex '
                        'per admission. Used for demographic enrichment.')
    p.add_argument('--seed',         type=int, default=42)
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    styles = make_styles()

    print("\nLoading data...")
    preds   = pd.read_csv(args.test_preds)
    windows = pd.read_csv(args.windows)

    # Load demographics from cohort file (has sex column)
    demo_df = None
    if args.cohort and Path(args.cohort).exists():
        demo_df = pd.read_csv(args.cohort)
        demo_df.columns = demo_df.columns.str.lower()
        demo_df['admission_id'] = demo_df['admission_id'].astype(str)
        # Keep only what's needed
        demo_cols = ['admission_id']
        for c in ['age_years', 'sex']:
            if c in demo_df.columns:
                demo_cols.append(c)
        demo_df = demo_df[demo_cols].drop_duplicates('admission_id')
        print(f"  Demographics loaded: {len(demo_df):,} admissions")
    else:
        print("  --cohort not provided — sex will show as ?")

    preds.columns   = preds.columns.str.lower()
    windows.columns = windows.columns.str.lower()

    ig_diff = pd.read_csv(args.ig_diff)
    ig_diff.columns = ig_diff.columns.str.lower()
    if 'difference' not in ig_diff.columns:
        ig_diff['difference'] = (
            ig_diff['aki_importance'] - ig_diff['nonaki_importance']
        )

    deploy_preds = deploy_wins = None
    if args.deploy_preds and Path(args.deploy_preds).exists():
        deploy_preds = pd.read_csv(args.deploy_preds)
        deploy_preds.columns = deploy_preds.columns.str.lower()
    if args.deploy_wins and Path(args.deploy_wins).exists():
        deploy_wins = pd.read_csv(args.deploy_wins)
        deploy_wins.columns = deploy_wins.columns.str.lower()

    print("Selecting cases...")
    cases_o1, cases_o2 = select_cases(
        preds, windows, args.threshold,
        deploy_preds, deploy_wins,
        seed=args.seed,
        demo_df=demo_df,
    )

    print(f"  Option 1 cases: {len(cases_o1)}")
    print(f"  Option 2 cases: {len(cases_o2)}")

    # Save confidential case list
    conf_df = pd.DataFrame(cases_o1 + cases_o2)
    conf_df['study'] = ['option1']*len(cases_o1) + ['option2']*len(cases_o2)
    conf_df.to_csv(out_dir / 'case_list_CONFIDENTIAL.csv', index=False)
    print(f"  Case list → {out_dir}/case_list_CONFIDENTIAL.csv "
          f"(DO NOT share with clinicians)")

    ig_attr_csv = None
    if args.ig_attr and Path(args.ig_attr).exists():
        ig_attr_csv = pd.read_csv(args.ig_attr)
        ig_attr_csv.columns = ig_attr_csv.columns.str.lower()
        print(f"  Loaded per-step attribution matrix: {ig_attr_csv.shape}")
    else:
        print("  --ig_attr not provided — will use population-level fallback")

    if args.study in ('option1', 'both'):
        print("\nBuilding Option 1 PDF (IG plausibility)...")
        build_option1_pdf(
            cases_o1, preds, ig_diff, args.threshold,
            out_dir / 'option1_ig_plausibility.pdf', styles,
            ig_attr_csv=ig_attr_csv,
        )

    if args.study in ('option2', 'both'):
        print("\nSaving Option 2 trajectory images...")
        save_option2_images(cases_o2, windows, out_dir)

    print(f"\nDone. Outputs in {out_dir}/")
    print(f"  Share with clinicians:")
    print(f"    option1_ig_plausibility.pdf")
    print(f"    option2_trajectory_images/")
    print(f"  Keep confidential:")
    print(f"    case_list_CONFIDENTIAL.csv")


if __name__ == '__main__':
    main()