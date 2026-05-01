"""
app.py

AKI Early Warning System — Streamlit Demo
Great Ormond Street Hospital · Paediatric ICU

This demo uses synthetic patient data and a pre-trained LSTM model
to demonstrate how continuous AKI risk prediction would work in a
clinical setting.

Run locally:
    streamlit run app.py

Deploy to Streamlit Cloud:
    Push to GitHub → connect repo at share.streamlit.io
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Local imports ─────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))

from fake_data import generate_demo_cohort, LAB_NORMALS
from model     import AKIPredictor, traffic_light, TL_COLOURS, TL_EMOJI


# ─────────────────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────────────────

st.set_page_config(
    page_title = 'AKI Early Warning · GOSH PICU Demo',
    page_icon  = '🏥',
    layout     = 'wide',
    initial_sidebar_state = 'expanded',
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
  /* Main background */
  .main { background-color: #F7F9FC; }

  /* Traffic light cards */
  .tl-card {
    padding: 14px 18px;
    border-radius: 10px;
    margin-bottom: 10px;
    cursor: pointer;
    border: 2px solid transparent;
    transition: border 0.15s;
  }
  .tl-card:hover { border: 2px solid #333; }
  .tl-red    { background: #FADBD8; }
  .tl-amber  { background: #FDEBD0; }
  .tl-green  { background: #D5F5E3; }

  /* Section headers */
  .section-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #1A252F;
    margin-bottom: 6px;
    border-bottom: 2px solid #E0E0E0;
    padding-bottom: 4px;
  }

  /* Alert banner */
  .alert-banner {
    background: #FADBD8;
    border-left: 5px solid #C0392B;
    padding: 10px 16px;
    border-radius: 6px;
    margin-bottom: 10px;
    font-weight: 600;
    color: #922B21;
  }

  /* Metric boxes */
  .metric-box {
    background: white;
    border-radius: 8px;
    padding: 12px;
    text-align: center;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
  }
  .metric-val {
    font-size: 1.6rem;
    font-weight: 700;
    color: #1A252F;
  }
  .metric-lab {
    font-size: 0.75rem;
    color: #7F8C8D;
    margin-top: 2px;
  }

  footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
#  Session state initialisation
# ─────────────────────────────────────────────────────────

def init_state():
    if 'windows'         not in st.session_state:
        st.session_state.windows    = None
    if 'meta'            not in st.session_state:
        st.session_state.meta       = None
    if 'predictor'       not in st.session_state:
        st.session_state.predictor  = None
    if 'selected_pid'    not in st.session_state:
        st.session_state.selected_pid = None
    if 'current_step'    not in st.session_state:
        st.session_state.current_step = {}
    if 'all_probs'       not in st.session_state:
        st.session_state.all_probs    = {}
    if 'new_obs'         not in st.session_state:
        st.session_state.new_obs      = {}
    if 'amber_t'         not in st.session_state:
        st.session_state.amber_t      = 0.30
    if 'red_t'           not in st.session_state:
        st.session_state.red_t        = 0.60

init_state()


# ─────────────────────────────────────────────────────────
#  Load data and model (cached)
# ─────────────────────────────────────────────────────────

@st.cache_resource(show_spinner='Loading model...')
def load_predictor():
    return AKIPredictor(weights_dir='model_weights')


@st.cache_data(show_spinner='Generating patient cohort...')
def load_cohort():
    return generate_demo_cohort(n_patients=30, seed=42)


# ─────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────

def get_current_prob(pid: str) -> float:
    """Get the latest risk probability for a patient."""
    if pid not in st.session_state.all_probs:
        return 0.0
    probs = st.session_state.all_probs[pid]
    return float(probs[-1]) if len(probs) > 0 else 0.0


def get_patient_windows(pid: str) -> pd.DataFrame:
    w = st.session_state.windows
    return w[w['patient_id'] == pid].sort_values('step_index').reset_index(drop=True)


def get_patient_meta(pid: str) -> dict:
    m = st.session_state.meta
    row = m[m['patient_id'] == pid].iloc[0]
    return row.to_dict()


def run_inference_all():
    """Run inference for all patients up to their current step."""
    predictor = st.session_state.predictor
    meta_df   = st.session_state.meta

    for _, row in meta_df.iterrows():
        pid    = row['patient_id']
        w      = get_patient_windows(pid)
        n_step = st.session_state.current_step.get(pid, len(w) // 2)
        w_sub  = w[w['step_index'] <= n_step].copy()

        # Append any manually entered observations
        if pid in st.session_state.new_obs:
            new_rows = pd.DataFrame(st.session_state.new_obs[pid])
            w_sub    = pd.concat([w_sub, new_rows], ignore_index=True)

        if len(w_sub) == 0:
            continue

        probs, _ = predictor.predict(w_sub)
        st.session_state.all_probs[pid] = probs


def tl_class(status: str) -> str:
    return {'RED':'tl-red','AMBER':'tl-amber','GREEN':'tl-green'}[status]


# ─────────────────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────────────────

with st.sidebar:
    st.image('https://upload.wikimedia.org/wikipedia/en/thumb/b/b9/'
             'Great_Ormond_Street_Hospital_logo.svg/200px-'
             'Great_Ormond_Street_Hospital_logo.svg.png',
             width=140)
    st.markdown('## AKI Early Warning System')
    st.caption('**PICU Demo** · Synthetic data only')
    st.divider()

    # Initialise
    if st.button('▶  Initialise system', type='primary', use_container_width=True):
        with st.spinner('Loading model and generating patients...'):
            st.session_state.predictor = load_predictor()
            windows, meta              = load_cohort()
            st.session_state.windows   = windows
            st.session_state.meta      = meta
            # Set all patients to midpoint of their stay
            for pid in meta['patient_id']:
                w = windows[windows['patient_id'] == pid]
                st.session_state.current_step[pid] = max(0, len(w) // 2)
            run_inference_all()
        st.success('System ready')

    st.divider()

    # Threshold settings
    st.markdown('### ⚙️ Alert thresholds')
    amber_t = st.slider('Amber threshold', 0.10, 0.60,
                         st.session_state.amber_t, 0.01)
    red_t   = st.slider('Red threshold', 0.20, 0.90,
                         st.session_state.red_t,   0.01)
    if amber_t != st.session_state.amber_t or red_t != st.session_state.red_t:
        st.session_state.amber_t = amber_t
        st.session_state.red_t   = red_t

    st.divider()
    st.caption(
        '⚠️ **Demo only.** This application uses entirely synthetic '
        'patient data. No real patient information is used or displayed. '
        'The model was trained on de-identified data from GOSH PICU.'
    )


# ─────────────────────────────────────────────────────────
#  Main app — only shown once initialised
# ─────────────────────────────────────────────────────────

if st.session_state.predictor is None:
    st.title('🏥 AKI Early Warning System')
    st.markdown('### Paediatric ICU · Great Ormond Street Hospital')
    st.info('Click **▶ Initialise system** in the sidebar to load '
            'the model and generate synthetic patient data.')
    st.markdown("""
    #### What this demo shows
    - Continuous AKI risk prediction updated every **6 hours** per patient
    - A **traffic light** risk stratification (Green / Amber / Red)
    - Per-patient **risk trajectories** showing how risk changes over time
    - Ability to **enter new clinical observations** and watch the risk update
    - **Attention-based** explanation of which time periods drove the prediction

    #### Model
    The underlying model is a **bidirectional LSTM with temporal attention**,
    trained on multimodal EHR data (labs, medications, procedures, demographics)
    from GOSH PICU patients. It predicts AKI onset within the **next 24 hours**,
    assessed at **6-hour intervals**.
    """)
    st.stop()


# ─────────────────────────────────────────────────────────
#  Ward overview page
# ─────────────────────────────────────────────────────────

meta_df   = st.session_state.meta
amber_t   = st.session_state.amber_t
red_t     = st.session_state.red_t
predictor = st.session_state.predictor

# Compute current status for all patients
statuses = {}
for pid in meta_df['patient_id']:
    prob          = get_current_prob(pid)
    statuses[pid] = traffic_light(prob, amber_t, red_t)

n_red   = sum(1 for s in statuses.values() if s == 'RED')
n_amber = sum(1 for s in statuses.values() if s == 'AMBER')
n_green = sum(1 for s in statuses.values() if s == 'GREEN')

# ── Header ────────────────────────────────────────────────
col_title, col_time = st.columns([3, 1])
with col_title:
    st.markdown('## 🏥 PICU Ward Overview')
    st.caption('Paediatric ICU · AKI Early Warning System · Synthetic Demo')
with col_time:
    st.markdown(f'<div class="metric-box" style="margin-top:10px">'
                f'<div class="metric-val">06:00</div>'
                f'<div class="metric-lab">Current ward round time</div>'
                f'</div>', unsafe_allow_html=True)

# ── Ward summary metrics ───────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="metric-box">'
                f'<div class="metric-val">{len(meta_df)}</div>'
                f'<div class="metric-lab">Total patients</div>'
                f'</div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-box" style="border-left:4px solid #C0392B">'
                f'<div class="metric-val" style="color:#C0392B">🔴 {n_red}</div>'
                f'<div class="metric-lab">High risk (RED)</div>'
                f'</div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="metric-box" style="border-left:4px solid #F39C12">'
                f'<div class="metric-val" style="color:#F39C12">🟡 {n_amber}</div>'
                f'<div class="metric-lab">Intermediate (AMBER)</div>'
                f'</div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="metric-box" style="border-left:4px solid #27AE60">'
                f'<div class="metric-val" style="color:#27AE60">🟢 {n_green}</div>'
                f'<div class="metric-lab">Low risk (GREEN)</div>'
                f'</div>', unsafe_allow_html=True)

st.divider()

# ── Alert banner for RED patients ──────────────────────────
red_patients = [pid for pid, s in statuses.items() if s == 'RED']
if red_patients:
    names = ', '.join(
        meta_df[meta_df['patient_id']==pid]['name'].values[0]
        for pid in red_patients
    )
    st.markdown(
        f'<div class="alert-banner">⚠️ HIGH RISK ALERT — '
        f'{len(red_patients)} patient(s) require urgent review: {names}</div>',
        unsafe_allow_html=True
    )

# ── Patient list ──────────────────────────────────────────
st.markdown('<div class="section-title">Patient list — click to view details</div>',
            unsafe_allow_html=True)

# Sort: RED first, then AMBER, then GREEN
status_order = {'RED': 0, 'AMBER': 1, 'GREEN': 2}
sorted_pids  = sorted(
    meta_df['patient_id'].tolist(),
    key=lambda p: status_order[statuses[p]]
)

cols_per_row = 3
rows         = [sorted_pids[i:i+cols_per_row]
                for i in range(0, len(sorted_pids), cols_per_row)]

for row_pids in rows:
    cols = st.columns(cols_per_row)
    for col, pid in zip(cols, row_pids):
        with col:
            m      = get_patient_meta(pid)
            prob   = get_current_prob(pid)
            status = statuses[pid]
            step   = st.session_state.current_step.get(pid, 0)
            hours  = step * 6
            emoji  = TL_EMOJI[status]
            css    = tl_class(status)

            if st.button(
                f'{emoji} **{m["name"]}** ({pid})\n\n'
                f'Risk: **{prob:.0%}** · {hours}h in PICU\n\n'
                f'Age {m["age_years"]:.1f}y · {m["sex"].title()} · '
                f'{"Cardiac " if m["cardiac"] else ""}',
                key=f'btn_{pid}',
                use_container_width=True,
                type='secondary',
            ):
                st.session_state.selected_pid = pid


# ─────────────────────────────────────────────────────────
#  Patient detail view
# ─────────────────────────────────────────────────────────

if st.session_state.selected_pid:
    pid    = st.session_state.selected_pid
    meta   = get_patient_meta(pid)
    w_all  = get_patient_windows(pid)
    n_tot  = len(w_all)
    status = statuses[pid]
    prob   = get_current_prob(pid)

    st.divider()
    st.markdown(f'## {TL_EMOJI[status]} Patient: {meta["name"]}')

    # ── Patient info bar ──────────────────────────────────
    ic1, ic2, ic3, ic4, ic5, ic6 = st.columns(6)
    with ic1: st.metric('Patient ID',  pid)
    with ic2: st.metric('NHS Number',  meta['nhs_number'])
    with ic3: st.metric('DOB',         meta['dob'])
    with ic4: st.metric('Age',         f'{meta["age_years"]:.1f} years')
    with ic5: st.metric('Sex',         meta['sex'].title())
    with ic6: st.metric('Cardiac Hx',  '✓ Yes' if meta['cardiac'] else '✗ No')

    st.divider()

    # ── Time controls ──────────────────────────────────────
    st.markdown('<div class="section-title">⏱ Admission timeline control</div>',
                unsafe_allow_html=True)

    tc1, tc2, tc3 = st.columns([2, 1, 1])
    with tc1:
        current_step = st.slider(
            'Current time in admission (step = 6h)',
            min_value  = 1,
            max_value  = n_tot,
            value      = st.session_state.current_step.get(pid, n_tot // 2),
            key        = f'slider_{pid}',
        )
        st.session_state.current_step[pid] = current_step - 1

    with tc2:
        hours_in = (current_step - 1) * 6
        st.markdown(f'<div class="metric-box" style="margin-top:22px">'
                    f'<div class="metric-val">{hours_in}h</div>'
                    f'<div class="metric-lab">Hours in PICU</div>'
                    f'</div>', unsafe_allow_html=True)
    with tc3:
        col = TL_COLOURS[status]
        st.markdown(f'<div class="metric-box" style="margin-top:22px;'
                    f'border-left:5px solid {col}">'
                    f'<div class="metric-val" style="color:{col}">'
                    f'{TL_EMOJI[status]} {prob:.0%}</div>'
                    f'<div class="metric-lab">Current AKI risk</div>'
                    f'</div>', unsafe_allow_html=True)

    # Simulate step-by-step
    if st.button('▶ Step forward 6h', use_container_width=False):
        new_s = min(
            st.session_state.current_step.get(pid, 0) + 1,
            n_tot - 1
        )
        st.session_state.current_step[pid] = new_s
        run_inference_all()
        st.rerun()

    # Run inference for this patient at current step
    w_sub = w_all[w_all['step_index'] <= st.session_state.current_step[pid]].copy()

    # Append new observations if any
    if pid in st.session_state.new_obs and st.session_state.new_obs[pid]:
        new_rows = pd.DataFrame(st.session_state.new_obs[pid])
        w_sub    = pd.concat([w_sub, new_rows], ignore_index=True)

    probs, attn_weights = predictor.predict(w_sub)
    st.session_state.all_probs[pid] = probs

    hours_axis = [i * 6 for i in range(len(probs))]

    st.divider()

    # ── Risk trajectory plot ───────────────────────────────
    st.markdown('<div class="section-title">📈 AKI risk trajectory</div>',
                unsafe_allow_html=True)

    fig = go.Figure()

    # Background shading for zones
    fig.add_hrect(y0=0,        y1=amber_t, fillcolor='#D5F5E3',
                   opacity=0.25, layer='below', line_width=0)
    fig.add_hrect(y0=amber_t,  y1=red_t,   fillcolor='#FDEBD0',
                   opacity=0.25, layer='below', line_width=0)
    fig.add_hrect(y0=red_t,    y1=1.0,     fillcolor='#FADBD8',
                   opacity=0.25, layer='below', line_width=0)

    # Threshold lines
    fig.add_hline(y=amber_t, line_dash='dash',
                   line_color='#F39C12', line_width=1.5,
                   annotation_text=f'Amber {amber_t:.2f}',
                   annotation_position='right')
    fig.add_hline(y=red_t, line_dash='dash',
                   line_color='#C0392B', line_width=1.5,
                   annotation_text=f'Red {red_t:.2f}',
                   annotation_position='right')

    # Risk curve coloured by traffic light
    colours_per_step = [
        TL_COLOURS[traffic_light(p, amber_t, red_t)] for p in probs
    ]
    for i in range(len(probs) - 1):
        fig.add_trace(go.Scatter(
            x=[hours_axis[i], hours_axis[i+1]],
            y=[probs[i],      probs[i+1]],
            mode='lines',
            line=dict(color=colours_per_step[i], width=3),
            showlegend=False,
            hovertemplate=f'Hour {hours_axis[i]}: {probs[i]:.1%}<extra></extra>',
        ))

    # Markers
    fig.add_trace(go.Scatter(
        x=hours_axis, y=probs,
        mode='markers',
        marker=dict(color=colours_per_step, size=8,
                    line=dict(width=1, color='white')),
        showlegend=False,
        hovertemplate='Hour %{x}: %{y:.1%}<extra></extra>',
    ))

    # Current time marker
    current_h = hours_axis[-1]
    fig.add_vline(x=current_h, line_dash='dot',
                   line_color='gray', line_width=1.5,
                   annotation_text='Now', annotation_position='top')

    fig.update_layout(
        yaxis=dict(title='Predicted AKI probability',
                   range=[0, 1], tickformat='.0%'),
        xaxis=dict(title='Hours since PICU admission'),
        height=350,
        margin=dict(l=20, r=80, t=20, b=20),
        plot_bgcolor='#FAFAFA',
        paper_bgcolor='white',
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Attention heatmap ─────────────────────────────────
    if attn_weights is not None:
        st.markdown('<div class="section-title">'
                    '🔍 Temporal attention — which time steps drove the prediction'
                    '</div>', unsafe_allow_html=True)

        fig_attn = go.Figure(go.Bar(
            x    = hours_axis,
            y    = attn_weights,
            marker_color = [
                f'rgba(192,57,43,{min(1, w*8)})'
                for w in attn_weights
            ],
            hovertemplate='Hour %{x}: attention %{y:.3f}<extra></extra>',
        ))
        fig_attn.update_layout(
            xaxis=dict(title='Hours since PICU admission'),
            yaxis=dict(title='Attention weight'),
            height=200,
            margin=dict(l=20, r=20, t=10, b=20),
            plot_bgcolor='#FAFAFA',
            paper_bgcolor='white',
        )
        st.plotly_chart(fig_attn, use_container_width=True)
        st.caption(
            'Higher attention weight = this time period had more influence '
            'on the current risk score. A peak near the end of the trajectory '
            'indicates the model is responding to recent clinical changes.'
        )

    st.divider()

    # ── Current lab values ────────────────────────────────
    st.markdown('<div class="section-title">🧪 Current laboratory values</div>',
                unsafe_allow_html=True)

    current_row = w_sub.iloc[-1]

    lab_display = []
    for lab, (lo, hi, unit) in LAB_NORMALS.items():
        col_last = f'{lab}_last'
        if col_last not in current_row:
            continue
        val = current_row[col_last]
        if pd.isna(val):
            continue
        # Flag as abnormal
        flag = ''
        if val < lo: flag = '⬇ Low'
        if val > hi: flag = '⬆ High'
        lab_display.append({
            'Lab':    lab.replace('_', ' ').title(),
            'Value':  round(float(val), 2),
            'Unit':   unit,
            'Normal': f'{lo} – {hi}',
            'Status': flag if flag else '✓ Normal',
        })

    lab_df = pd.DataFrame(lab_display)
    # Highlight abnormals
    def highlight_flag(val):
        if '⬆' in str(val) or '⬇' in str(val):
            return 'background-color: #FADBD8; color: #922B21; font-weight: bold'
        return ''

    st.dataframe(
        lab_df.style.applymap(highlight_flag, subset=['Status']),
        use_container_width=True,
        height=280,
        hide_index=True,
    )

    # ── New observation entry ─────────────────────────────
    st.divider()
    st.markdown('<div class="section-title">'
                '➕ Add new clinical observation</div>',
                unsafe_allow_html=True)
    st.caption(
        'Enter a new lab result to see how the risk score updates in real time. '
        'This simulates a clinician adding a new creatinine or electrolyte result.'
    )

    nc1, nc2, nc3 = st.columns(3)
    with nc1:
        new_lab = st.selectbox(
            'Laboratory test',
            options=list(LAB_NORMALS.keys()),
            format_func=lambda x: x.replace('_', ' ').title(),
            key=f'new_lab_{pid}',
        )
    with nc2:
        lo, hi, unit = LAB_NORMALS[new_lab]
        new_val = st.number_input(
            f'Value ({unit if unit else "—"})',
            min_value=0.0,
            max_value=float(hi * 3),
            value=float((lo + hi) / 2),
            step=0.1,
            key=f'new_val_{pid}',
        )
    with nc3:
        st.markdown('<div style="height:28px"></div>', unsafe_allow_html=True)
        if st.button('Add observation & update risk',
                      type='primary', key=f'add_obs_{pid}'):
            # Build a new window row
            if pid not in st.session_state.new_obs:
                st.session_state.new_obs[pid] = []
            new_step = (st.session_state.current_step.get(pid, 0) + 1
                        + len(st.session_state.new_obs[pid]))
            new_row  = dict(current_row)  # copy last row
            new_row['step_index']        = new_step
            new_row['hours_since_admit'] = new_step * 6.0
            new_row[f'{new_lab}_last']   = new_val
            new_row[f'{new_lab}_mean']   = new_val
            st.session_state.new_obs[pid].append(new_row)
            st.success(f'Added {new_lab.title()} = {new_val:.1f} {unit}. '
                        f'Risk score updating...')
            st.rerun()

    if pid in st.session_state.new_obs and st.session_state.new_obs[pid]:
        st.caption(
            f'{len(st.session_state.new_obs[pid])} new observation(s) added '
            f'this session. '
        )
        if st.button('Clear added observations', key=f'clear_{pid}'):
            st.session_state.new_obs[pid] = []
            st.rerun()

    st.divider()

    # ── Close detail view ─────────────────────────────────
    if st.button('← Back to ward overview'):
        st.session_state.selected_pid = None
        st.rerun()


# ─────────────────────────────────────────────────────────
#  Footer
# ─────────────────────────────────────────────────────────
st.markdown("""
---
<div style="text-align:center; color:#AAA; font-size:0.75rem; padding:8px">
  AKI Early Warning System · GOSH PICU Research Demo ·
  <b>Synthetic data only — not for clinical use</b> ·
  Model trained under R&amp;D reference 24PE36
</div>
""", unsafe_allow_html=True)