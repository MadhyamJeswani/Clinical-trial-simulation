# ============================================================
# CLINICAL TRIAL AI SIMULATION SYSTEM
# Streamlit Dashboard (Refactored + Debugged)
# ============================================================

import os
import sys
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ============================================================
# PATH SETUP
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"

sys.path.append(str(SRC_DIR))

# ============================================================
# IMPORT CUSTOM MODULES
# ============================================================

try:
    from simulation_engine import run_full_trial
    from ai_agent import run_trial_with_aria
except Exception as e:
    st.error(f"Module Import Error: {e}")
    st.stop()

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Clinical Trial AI Simulation",
    page_icon="🔬",
    layout="wide"
)

# ============================================================
# CONSTANTS
# ============================================================

DISEASE_OPTIONS = [
    "cancer",
    "infectious",
    "cardiovascular",
    "diabetes",
    "mental_health",
    "healthy",
    "general"
]

ORG_OPTIONS = [
    "INDUSTRY",
    "NIH",
    "OTHER",
    "FED",
    "NETWORK"
]

PHASE_COLORS = {
    "Phase 1": "#1A237E",
    "Phase 2": "#2196F3",
    "Phase 3": "#4CAF50"
}

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 2.5rem;
    font-weight: bold;
    color: #1A237E;
}

.metric-box {
    background: #F5F7FA;
    padding: 1rem;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# SAFE HELPERS
# ============================================================

def safe_get(data: Dict, key: str, default=0):
    return data.get(key, default)


def calculate_average(results: List[Dict], key: str):
    values = [safe_get(r, key, 0) for r in results]
    return round(np.mean(values), 2) if values else 0


def decision_emoji(decision: str):
    return "✅ GO" if decision == "GO" else "❌ NO-GO"


# ============================================================
# CHARTS
# ============================================================

def enrollment_chart(results: List[Dict]):

    phases = [r["phase"] for r in results]

    enrolled = [safe_get(r, "total_enrolled") for r in results]
    completed = [safe_get(r, "total_completed") for r in results]
    dropouts = [safe_get(r, "total_dropouts") for r in results]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=phases,
        y=enrolled,
        name="Enrolled"
    ))

    fig.add_trace(go.Bar(
        x=phases,
        y=completed,
        name="Completed"
    ))

    fig.add_trace(go.Bar(
        x=phases,
        y=dropouts,
        name="Dropouts"
    ))

    fig.update_layout(
        barmode="group",
        title="Enrollment Overview",
        height=400
    )

    return fig


def efficacy_chart(results: List[Dict]):

    phases = [r["phase"] for r in results]

    response = [safe_get(r, "response_rate_%") for r in results]
    endpoint = [safe_get(r, "endpoint_rate_%") for r in results]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=phases,
        y=response,
        mode="lines+markers",
        name="Response Rate"
    ))

    fig.add_trace(go.Scatter(
        x=phases,
        y=endpoint,
        mode="lines+markers",
        name="Endpoint Rate"
    ))

    fig.update_layout(
        title="Efficacy Trends",
        height=400
    )

    return fig


def safety_chart(results: List[Dict]):

    phases = [r["phase"] for r in results]
    ae_rate = [safe_get(r, "ae_rate_%") for r in results]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=phases,
        y=ae_rate,
        name="AE Rate"
    ))

    fig.update_layout(
        title="Adverse Event Rate",
        height=400
    )

    return fig


# ============================================================
# DATA TABLES
# ============================================================

def summary_dataframe(results: List[Dict]):

    rows = []

    for r in results:
        rows.append({
            "Phase": safe_get(r, "phase"),
            "Enrolled": safe_get(r, "total_enrolled"),
            "Completed": safe_get(r, "total_completed"),
            "AE Rate": safe_get(r, "ae_rate_%"),
            "Dropout Rate": safe_get(r, "dropout_rate_%"),
            "Response Rate": safe_get(r, "response_rate_%"),
            "P Value": safe_get(r, "p_value"),
            "Decision": safe_get(r, "decision")
        })

    return pd.DataFrame(rows)


# ============================================================
# SIMULATION EXECUTION
# ============================================================

@st.cache_data(show_spinner=False)
def execute_trial(
    drug_name: str,
    disease: str
):

    return run_full_trial(
        drug_name=drug_name,
        disease_category=disease
    )


# ============================================================
# ARIA EXECUTION
# ============================================================

def execute_aria(
    drug_name: str,
    disease: str,
    org_class: str,
    results: List[Dict]
):

    try:

        return run_trial_with_aria(
            drug_name=drug_name,
            simulation_results=results,
            disease_category=disease,
            org_class=org_class
        )

    except Exception as e:

        st.error(f"ARIA Error: {e}")
        return None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("⚙ Simulation Settings")

    drug_name = st.text_input(
        "Drug Name",
        value="OncoCure-X1"
    )

    disease = st.selectbox(
        "Disease Category",
        DISEASE_OPTIONS
    )

    org_class = st.selectbox(
        "Organization Type",
        ORG_OPTIONS
    )

    enable_aria = st.toggle(
        "Enable Dr. ARIA",
        value=True
    )

    compare_mode = st.toggle(
        "Compare Drugs",
        value=False
    )

    if compare_mode:

        st.divider()

        drug_name_2 = st.text_input(
            "Drug 2",
            value="InfectoShield-V2"
        )

        disease_2 = st.selectbox(
            "Disease 2",
            DISEASE_OPTIONS,
            index=1
        )

        org_class_2 = st.selectbox(
            "Org Type 2",
            ORG_OPTIONS,
            index=0
        )

    run_btn = st.button(
        "▶ Run Simulation",
        use_container_width=True
    )

# ============================================================
# HEADER
# ============================================================

st.markdown(
    "<div class='main-title'>🔬 Clinical Trial AI Simulation Dashboard</div>",
    unsafe_allow_html=True
)

st.caption(
    "Capgemini Internship Project 2026"
)

st.divider()

# ============================================================
# MAIN EXECUTION
# ============================================================

if run_btn:

    # --------------------------------------------------------
    # PRIMARY TRIAL
    # --------------------------------------------------------

    with st.spinner("Running Clinical Trial..."):

        results = execute_trial(
            drug_name,
            disease
        )

    aria_results = None

    if enable_aria:

        with st.spinner("Dr. ARIA Analysis..."):

            aria_results = execute_aria(
                drug_name,
                disease,
                org_class,
                results
            )
        # --------------------------------------------------------
        # COMPARISON TRIAL (Drug 2)
        # --------------------------------------------------------
    results2 = None
    aria_results2 = None

    if compare_mode:
        with st.spinner(
                f"Running Trial for {drug_name_2}..."):
            results2 = execute_trial(
                drug_name_2,
                disease_2
            )

        if enable_aria and results2:
            with st.spinner(
                    "Dr. ARIA analyzing Drug 2..."):
                aria_results2 = execute_aria(
                    drug_name_2,
                    disease_2,
                    org_class_2,
                    results2
                )
    # --------------------------------------------------------
    # TABS
    # --------------------------------------------------------

    tab_list = ["Overview", "Safety", "Efficacy", "ARIA"]
    if compare_mode and results2:
        tab_list.append("🔄 Comparison")

    tabs = st.tabs(tab_list)

    # ========================================================
    # OVERVIEW TAB
    # ========================================================

    with tabs[0]:

        st.subheader("Trial Overview")

        cols = st.columns(4)

        cols[0].metric(
            "Patients",
            sum(safe_get(r, "total_enrolled") for r in results)
        )

        cols[1].metric(
            "Avg AE Rate",
            f"{calculate_average(results, 'ae_rate_%')}%"
        )

        cols[2].metric(
            "Avg Response",
            f"{calculate_average(results, 'response_rate_%')}%"
        )

        cols[3].metric(
            "Phases Passed",
            sum(
                1 for r in results
                if safe_get(r, "decision") == "GO"
            )
        )

        st.plotly_chart(
            enrollment_chart(results),
            use_container_width=True
        )

        st.dataframe(
            summary_dataframe(results),
            use_container_width=True
        )

    # ========================================================
    # SAFETY TAB
    # ========================================================

    with tabs[1]:

        st.subheader("Safety Analysis")

        st.plotly_chart(
            safety_chart(results),
            use_container_width=True
        )

        for r in results:

            with st.expander(r["phase"]):

                st.write(
                    f"AE Rate: {safe_get(r, 'ae_rate_%')}%"
                )

                st.write(
                    f"Dropout Rate: {safe_get(r, 'dropout_rate_%')}%"
                )

                st.write(
                    f"Severe AE: {safe_get(r, 'severe_ae')}"
                )

    # ========================================================
    # EFFICACY TAB
    # ========================================================

    with tabs[2]:

        st.subheader("Efficacy Analysis")

        st.plotly_chart(
            efficacy_chart(results),
            use_container_width=True
        )

        efficacy_df = pd.DataFrame([
            {
                "Phase": r["phase"],
                "Response": safe_get(r, "response_rate_%"),
                "Endpoint": safe_get(r, "endpoint_rate_%"),
                "Improvement": safe_get(r, "improvement_%")
            }
            for r in results
        ])

        st.dataframe(
            efficacy_df,
            use_container_width=True
        )

    # ========================================================
    # ARIA TAB
    # ========================================================

    with tabs[3]:

        st.subheader("Dr. ARIA")

        if not enable_aria:

            st.info("ARIA disabled.")

        elif aria_results is None:

            st.error("ARIA unavailable.")

        else:

            for item in aria_results:

                phase = item.get("phase", "Unknown")

                analysis = item.get(
                    "aria_analysis",
                    {}
                )

                with st.expander(phase, expanded=True):

                    st.write(
                        f"Recommendation: "
                        f"{analysis.get('aria_recommendation', 'N/A')}"
                    )

                    st.write(
                        f"Risk Level: "
                        f"{analysis.get('risk_level', 'N/A')}"
                    )

                    findings = analysis.get(
                        "key_findings",
                        []
                    )

                    if findings:

                        st.markdown("### Findings")

                        for f in findings:
                            st.write(f"• {f}")
# ========================================================
# COMPARISON TAB
# ========================================================
if compare_mode and 'results2' in dir() and results2:
    with tabs[4]:
        st.subheader(
            f"Comparison: {drug_name} vs {drug_name_2}"
        )

        # Side by side decisions
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"### 💊 {drug_name}")
            for r in results:
                decision = safe_get(r, 'decision')
                color = (
                    "green" if decision == "GO"
                    else "red"
                )
                st.markdown(
                    f"**{safe_get(r, 'phase')}**: "
                    f":{color}[{decision}]"
                )

        with col2:
            st.markdown(f"### 💊 {drug_name_2}")
            for r in results2:
                decision = safe_get(r, 'decision')
                color = (
                    "green" if decision == "GO"
                    else "red"
                )
                st.markdown(
                    f"**{safe_get(r, 'phase')}**: "
                    f":{color}[{decision}]"
                )

        st.divider()

        # Metrics comparison
        st.subheader("📊 Metrics Comparison")

        comp_data = {
            'Metric': [
                'Avg AE Rate',
                'Avg Dropout Rate',
                'Avg Response Rate',
                'Avg Improvement',
                'Total Patients',
                'Phases Passed'
            ],
            drug_name: [
                f"{calculate_average(results, 'ae_rate_%')}%",
                f"{calculate_average(results, 'dropout_rate_%')}%",
                f"{calculate_average(results, 'response_rate_%')}%",
                f"{calculate_average(results, 'improvement_%')}%",
                sum(safe_get(r, 'total_enrolled', 0) for r in results),
                sum(1 for r in results if safe_get(r, 'decision') == 'GO')
            ],
            drug_name_2: [
                f"{calculate_average(results2, 'ae_rate_%')}%",
                f"{calculate_average(results2, 'dropout_rate_%')}%",
                f"{calculate_average(results2, 'response_rate_%')}%",
                f"{calculate_average(results2, 'improvement_%')}%",
                sum(safe_get(r, 'total_enrolled', 0) for r in results2),
                sum(1 for r in results2 if safe_get(r, 'decision') == 'GO')
            ]
        }

        st.dataframe(
            pd.DataFrame(comp_data),
            use_container_width=True,
            hide_index=True
        )

        # Comparison charts
        st.subheader("📈 Visual Comparison")
        st.write("Loading charts...")
        metrics_to_plot = [
            ('ae_rate_%', 'AE Rate (%)'),
            ('response_rate_%', 'Response Rate (%)'),
            ('dropout_rate_%', 'Dropout Rate (%)')
        ]

        for metric, title in metrics_to_plot:
            try:
                fig = go.Figure()

                # Drug 1 data
                x1 = [r.get('phase', 'Unknown')
                      for r in results]
                y1 = [r.get(metric, 0)
                      for r in results]

                # Drug 2 data
                x2 = [r.get('phase', 'Unknown')
                      for r in results2]
                y2 = [r.get(metric, 0)
                      for r in results2]

                fig.add_trace(go.Bar(
                    name=drug_name,
                    x=x1, y=y1,
                    marker_color='#1A237E'
                ))

                fig.add_trace(go.Bar(
                    name=drug_name_2,
                    x=x2, y=y2,
                    marker_color='#FF6F00'
                ))

                fig.update_layout(
                    title=title,
                    barmode='group',
                    height=350,
                    xaxis_title='Phase',
                    yaxis_title=title,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02
                    ),
                    plot_bgcolor='white',
                    paper_bgcolor='white'
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            except Exception as e:
                st.error(f"Chart error: {e}")

            fig.add_trace(go.Bar(
                name=drug_name_2,
                x=[safe_get(r, 'phase')
                   for r in results2],
                y=[safe_get(r, metric, 0)
                   for r in results2],
                marker_color='#FF6F00'
            ))

            fig.update_layout(
                title=dict(
                    text=title,
                    font=dict(size=16)
                ),
                barmode='group',
                height=350,
                plot_bgcolor='white',
                paper_bgcolor='white',
                xaxis=dict(
                    title='Trial Phase',
                    tickfont=dict(size=13)
                ),
                yaxis=dict(
                    title=title,
                    tickfont=dict(size=13)
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    font=dict(size=12)
                ),
                margin=dict(t=80, b=50, l=60, r=20)
            )
# ============================================================
# WELCOME SCREEN
# ============================================================

else:

    st.info(
        "Configure the simulation settings in the sidebar and click RUN."
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Clinical Trials", "164,733")
    c2.metric("Disease Areas", "6")
    c3.metric("AI Models", "2")
    c4.metric("Agent", "Dr. ARIA v2")