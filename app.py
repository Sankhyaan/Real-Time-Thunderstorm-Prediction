"""
=========================================================
Real-Time Thunderstorm Prediction Dashboard
=========================================================
"""

import pickle
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

from fetch_latest import fetch_latest, fetch_historical
from predict import predict


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Real-Time Thunderstorm Prediction",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ==========================================================
# LOAD MODEL ARTIFACTS
# ==========================================================

with open("final_outputs/feature_importance.pkl", "rb") as f:
    feature_importance = pickle.load(f)


# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown(
"""
<style>
/* Hide top black Streamlit header bar */
header[data-testid="stHeader"] {display:none !important;}
[data-testid="stSidebar"]{display:none;}
[data-testid="collapsedControl"]{display:none;}
footer{visibility:hidden;}
.stApp{background:#F5F3FF;}

/* Remove the huge top gap Streamlit adds */
.stApp > div:first-child{padding-top:0 !important;}
.block-container{padding-top:1rem !important;}

html, body, .stApp{color:#1F2937 !important;}

/* Header */
.header{
    background:linear-gradient(90deg,#4C1D95,#7C3AED,#A78BFA);
    color:white;
    padding:14px 20px;
    border-radius:10px;
    margin-bottom:14px;
    box-shadow:0 3px 10px rgba(109,40,217,.30);
}
.header h1{margin:0;font-size:20px;font-weight:700;}
.header p{margin-top:4px;font-size:12px;opacity:0.88;}

/* Section title */
.section-title{
    font-size:14px;
    font-weight:700;
    color:#5B21B6;
    margin-top:14px;
    margin-bottom:8px;
    letter-spacing:0.3px;
}

/* Tiny SQUARE index card - fixed 80x80 */
.idx-card{
    background:white;
    border-radius:6px;
    border:1px solid #DDD6FE;
    box-shadow:0 1px 3px rgba(109,40,217,.08);
    text-align:center;
    width:80px;
    height:80px;
    display:flex;
    flex-direction:column;
    justify-content:center;
    align-items:center;
    padding:4px;
    flex-shrink:0;
}
.idx-label{color:#7C3AED;font-size:8.5px;font-weight:600;letter-spacing:0.4px;text-transform:uppercase;line-height:1.2;}
.idx-value{font-size:14px;font-weight:700;margin-top:3px;line-height:1;}

/* Indices section outer box - compact centered */
.indices-box{
    display:inline-block;
    background:white;
    border:1px solid #DDD6FE;
    border-radius:10px;
    box-shadow:0 1px 5px rgba(109,40,217,.10);
    padding:12px;
}
.indices-wrapper{
    display:flex;
    justify-content:center;
    width:100%;
}

/* Station info card - fit to content, centered */
.info-row{
    display:flex;
    gap:12px;
    justify-content:center;
    flex-wrap:wrap;
    margin-bottom:4px;
}
.info-card{
    background:white;
    padding:10px 16px;
    border-radius:8px;
    border:1px solid #DDD6FE;
    box-shadow:0 1px 3px rgba(109,40,217,.08);
    display:inline-block;
    min-width:120px;
}
.info-label{color:#7C3AED;font-size:11px;font-weight:600;}
.info-value{color:#1F2937;font-size:17px;font-weight:700;margin-top:2px;white-space:nowrap;}
.info-sub{color:#555;font-size:12px;margin-top:1px;}

/* Fix expander header */
[data-testid="stExpander"] summary{
    background:white !important;
    color:#1F2937 !important;
    border-radius:8px;
}
[data-testid="stExpander"] summary:hover{
    background:#EDE9FE !important;
    color:#1F2937 !important;
}
[data-testid="stExpander"] summary p{
    color:#1F2937 !important;
}

/* Button - lavender gradient, auto width, centered */
.stButton{display:flex;justify-content:center;}
.stButton>button{
    width:auto !important;
    padding:0 36px;
    background:linear-gradient(90deg,#4C1D95,#7C3AED,#A78BFA);color:white;
    border:none;border-radius:8px;height:40px;
    font-size:14px;font-weight:600;
    box-shadow:0 2px 8px rgba(109,40,217,.30);
}
.stButton>button:hover{background:linear-gradient(90deg,#3B0764,#4C1D95,#7C3AED);color:white;}

/* Labels */
label{color:#1F2937 !important;font-weight:600 !important;}
div[role="radiogroup"]{color:#1F2937 !important;}
div[role="radiogroup"] label{color:#1F2937 !important;}
div[role="radiogroup"] p{color:#1F2937 !important;}
[data-baseweb="radio"]{color:#1F2937 !important;}
[data-baseweb="radio"] *{color:#1F2937 !important;}
.stRadio label,.stDateInput label,.stSelectbox label,.stMultiSelect label{color:#1F2937 !important;}

/* Date input - lavender */
[data-baseweb="input"]{
    background:#EDE9FE !important;
    border:1px solid #C4B5FD !important;
    border-radius:8px !important;
}
[data-baseweb="input"] input{
    background:#EDE9FE !important;
    color:#4C1D95 !important;
    font-weight:600 !important;
}
[data-baseweb="input"]:focus-within{
    border-color:#7C3AED !important;
    box-shadow:0 0 0 2px rgba(124,58,237,.15) !important;
}

/* Selectbox - force lavender via testid */
[data-testid="stSelectbox"] > div > div,
[data-testid="stSelectbox"] > div > div > div {
    background-color:#EDE9FE !important;
    border:1px solid #C4B5FD !important;
    border-radius:8px !important;
}
/* Force ALL text inside selectbox to lavender purple */
[data-testid="stSelectbox"] * {
    color:#4C1D95 !important;
    background-color:transparent !important;
}
[data-testid="stSelectbox"] > div > div,
[data-testid="stSelectbox"] > div > div > div,
[data-testid="stSelectbox"] > div > div > div > div {
    background-color:#EDE9FE !important;
}
[data-testid="stSelectbox"] svg { fill:#7C3AED !important; }
/* Hide blinking text cursor inside selectbox */
[data-testid="stSelectbox"] input,
[data-baseweb="select"] input {
    caret-color: transparent !important;
    cursor: default !important;
}

/* Selectbox dropdown list */
[data-baseweb="menu"],
[data-baseweb="popover"] ul {
    background:#F5F3FF !important;
    border:1px solid #DDD6FE !important;
}
[data-baseweb="menu"] li,
[data-baseweb="popover"] [role="option"] {
    color:#1F2937 !important;
    background:#F5F3FF !important;
}
[data-baseweb="menu"] li:hover,
[data-baseweb="popover"] [role="option"]:hover,
[data-baseweb="menu"] li[aria-selected="true"] {
    background:#DDD6FE !important;
    color:#1F2937 !important;
}

/* Radio buttons - remove white highlight background, fix circle colours */
[data-testid="stRadio"] label {
    background:transparent !important;
    border-radius:0 !important;
}
[data-testid="stRadio"] label > div:first-child {
    border-color:#A78BFA !important;
    background:white !important;
}
[data-baseweb="radio"] div[data-checked="true"] > div {
    background:#7C3AED !important;
    border-color:#7C3AED !important;
}

/* Settings container box - lavender border */
[data-testid="stContainer"][data-bordered="true"],
div[data-testid="stVerticalBlockBorderWrapper"] > div {
    background:#F5F3FF !important;
    border:1px solid #DDD6FE !important;
    border-radius:10px !important;
}

/* Dataframe - lavender header and alternating rows */
[data-testid="stDataFrame"] thead tr th {
    background:#EDE9FE !important;
    color:#4C1D95 !important;
    font-weight:700 !important;
    border-bottom:2px solid #C4B5FD !important;
}
[data-testid="stDataFrame"] tbody tr td {
    color:#1F2937 !important;
    border-bottom:1px solid #EDE9FE !important;
}
[data-testid="stDataFrame"] tbody tr:hover td {
    background:#F5F3FF !important;
}
[data-testid="stDataFrame"] table {
    border:1px solid #DDD6FE !important;
    border-radius:8px !important;
    overflow:hidden;
}
/* Download CSV button */
[data-testid="stDownloadButton"] > button {
    background:#EDE9FE !important;
    color:#4C1D95 !important;
    border:1px solid #C4B5FD !important;
    border-radius:8px !important;
    width:100%;
}
[data-testid="stDownloadButton"] > button:hover {
    background:#DDD6FE !important;
}

</style>
""",
unsafe_allow_html=True,
)


# ==========================================================
# HEADER
# ==========================================================

_hl, _hmid, _hr = st.columns([1, 2, 1])
with _hmid:
    st.markdown(
    """
    <div class="header">
    <h1>⚡ Real-Time Thunderstorm Prediction</h1>
    <p>Operational prediction using Upper-Air Soundings, MetPy indices &amp; a trained CatBoost model — Station 43150, Visakhapatnam</p>
    </div>
    """,
    unsafe_allow_html=True,
    )


# ==========================================================
# PREDICTION SETTINGS  (auto-run, no button needed)
# ==========================================================

_l, _mid, _r = st.columns([1, 2, 1])

with _mid:
    settings = st.container(border=True)

    with settings:
        st.markdown('<div class="section-title" style="margin:0 0 8px 0;">Observation Date &amp; Time</div>', unsafe_allow_html=True)

        dc1, dc2 = st.columns(2)

        today = datetime.utcnow().date()

        with dc1:
            sel_date = st.date_input("Date", value=today, label_visibility="collapsed")
        with dc2:
            sel_hour = st.selectbox("UTC Hour", [0, 12],
                                    format_func=lambda x: f"{x:02d}:00 UTC",
                                    label_visibility="collapsed")


# ==========================================================
# RUN MODEL  (auto on load; re-runs when date/hour changes)
# ==========================================================

def _run_key(d, h):
    return f"date={d},hour={h}"

if "last_run_key" not in st.session_state:
    st.session_state["last_run_key"] = None
if "cached_results" not in st.session_state:
    st.session_state["cached_results"] = None

current_key = _run_key(sel_date, sel_hour)
should_run = (
    st.session_state["last_run_key"] is None         # first load
    or st.session_state["last_run_key"] != current_key  # date/hour changed
)

if should_run:
    with st.spinner("Downloading sounding..."):
        if sel_date == today:
            sounding = fetch_latest()
        else:
            observation = datetime(sel_date.year, sel_date.month, sel_date.day, sel_hour)
            sounding = fetch_historical(observation)

        if sounding is None:
            st.error("No sounding is available for the selected date/time.")
            st.stop()

    station      = sounding["station"]
    station_name = sounding["station_name"]
    sounding_time= sounding["sounding_time"]
    profile      = sounding["data"]

    with st.spinner("Calculating meteorological indices..."):
        result = predict(profile)

    st.session_state["cached_results"] = {
        "station": station,
        "station_name": station_name,
        "sounding_time": sounding_time,
        "profile": profile,
        "result": result,
    }
    st.session_state["last_run_key"] = current_key

# Pull from cache
cached = st.session_state["cached_results"]
if cached is None:
    st.stop()

station       = cached["station"]
station_name  = cached["station_name"]
sounding_time = cached["sounding_time"]
profile       = cached["profile"]
result        = cached["result"]

probability     = result["probability"]
raw_probability = result["raw_probability"]
risk            = result["risk"]
indices         = result["indices"]


# ── SINGLE 2-COLUMN LAYOUT: left = Station+Indices, right = Prediction+Chart ──
probability_colours = {
    35: ("#DCFCE7", "#15803D"),
    45: ("#BBF7D0", "#16A34A"),
    55: ("#FEF9C3", "#CA8A04"),
    65: ("#FED7AA", "#EA580C"),
    75: ("#FDBA74", "#F97316"),
    85: ("#FECACA", "#DC2626"),
    95: ("#FCA5A5", "#991B1B"),
}
background, accent = probability_colours.get(probability, ("#E5E7EB", "#7C3AED"))

col_left, col_right = st.columns(2)

# ── LEFT: Station Info ─────────────────────────────────────────────────
with col_left:
    st.markdown('<div class="section-title">Station Information</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="info-row" style="justify-content:flex-start;">
        <div class="info-card"><div class="info-label">Station</div><div class="info-value">{station}</div></div>
        <div class="info-card"><div class="info-label">Location</div><div class="info-value">{station_name}</div></div>
        <div class="info-card"><div class="info-label">Observation Time</div><div class="info-value">{sounding_time.strftime("%d %b %Y")}</div><div class="info-sub">{sounding_time.strftime("%H:00 UTC")}</div></div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="section-title" style="margin-top:18px;">Meteorological Indices</div>', unsafe_allow_html=True)

    items = list(indices.items())
    columns_per_row = 5
    while len(items) % columns_per_row != 0:
        items.append(("", float("nan")))

    rows_html = ""
    for i in range(0, len(items), columns_per_row):
        row_items = items[i:i+columns_per_row]
        row_html = '<div style="display:flex;gap:6px;margin-bottom:6px;">'
        for feature, value in row_items:
            if not feature:
                row_html += '<div style="width:80px;height:80px;flex-shrink:0;"></div>'
                continue
            if pd.isna(value):
                display = "--"
                colour = "#C4B5FD"
            else:
                display = f"{value:.2f}"
                if feature in ["PW","KI","SWEAT","CAPE","CAPE_V","CT","VT","TT"]:
                    colour = "#4C1D95" if value >= 50 else ("#7C3AED" if value >= 30 else "#A78BFA")
                elif feature in ["LI","LI_V","SI"]:
                    colour = "#4C1D95" if value <= -5 else ("#7C3AED" if value <= -2 else "#A78BFA")
                else:
                    colour = "#8B5CF6"
            row_html += f'<div class="idx-card"><div class="idx-label">{feature}</div><div class="idx-value" style="color:{colour};">{display}</div></div>'
        row_html += "</div>"
        rows_html += row_html

    st.markdown(f'<div class="indices-box" style="display:inline-block;">{rows_html}</div>', unsafe_allow_html=True)

# ── RIGHT: Thunderstorm Prediction (compact) + Feature Importance ──────
with col_right:
    st.markdown('<div class="section-title">Thunderstorm Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
<div style="background:white;border:1px solid #DDD6FE;padding:10px 16px;border-radius:12px;box-shadow:0 1px 6px rgba(109,40,217,.09);display:flex;align-items:center;gap:16px;">
  <div style="flex:1;text-align:center;">
    <div style="color:#5B21B6;font-size:10px;font-weight:600;letter-spacing:0.5px;text-transform:uppercase;margin-bottom:2px;">Operational Probability</div>
    <div style="color:{accent};font-size:28px;font-weight:800;line-height:1;">{probability}%</div>
    <div style="color:#374151;font-size:11px;font-weight:700;margin:3px 0 8px 0;">{risk}</div>
    <div style="background:#EDE9FE;height:6px;border-radius:6px;overflow:hidden;">
      <div style="width:{probability}%;background:{accent};height:6px;border-radius:6px;"></div>
    </div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title" style="margin-top:18px;">Feature Importance</div>', unsafe_allow_html=True)

    importance_df = pd.DataFrame(
        list(feature_importance.items()),
        columns=["Feature", "Importance"]
    )
    importance_df = (
        importance_df
        .sort_values("Importance", ascending=True)
        .reset_index(drop=True)
    )

    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        text="Importance",
    )
    fig.update_traces(
        marker_color="#7C3AED",
        texttemplate="%{text:.2f}",
        textposition="outside",
        textfont=dict(size=11, color="#1F2937"),
        hovertemplate="<b>%{y}</b><br>Importance=%{x:.2f}<extra></extra>"
    )
    fig.update_layout(
        height=500,
        margin=dict(l=10, r=60, t=10, b=40),
        xaxis_title="Importance",
        yaxis_title="",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=12, color="#1F2937"),
        xaxis=dict(
            gridcolor="#EDE9FE",
            linecolor="#6B7280",
            tickfont=dict(color="#1F2937", size=11),
            title_font=dict(color="#1F2937", size=12)
        ),
        yaxis=dict(
            linecolor="#6B7280",
            tickfont=dict(color="#1F2937", size=11)
        )
    )
    st.plotly_chart(fig, use_container_width=True)


# ==========================================================
# RAW SOUNDING DATA
# ==========================================================

st.markdown(
"""
<div class="section-title">Upper-Air Sounding Data</div>
""",
unsafe_allow_html=True
)

with st.expander("Show Sounding Data", expanded=False):

    sounding_df = profile.copy()
    numeric_columns = sounding_df.select_dtypes(include="number").columns
    sounding_df[numeric_columns] = sounding_df[numeric_columns].round(2)

    st.dataframe(sounding_df, use_container_width=True, hide_index=True, height=500)

    csv = sounding_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="upper_air_sounding.csv",
        mime="text/csv",
        use_container_width=True
    )


# ==========================================================
# PROBABILITY SCALE
# ==========================================================

st.markdown('<div class="section-title">Operational Probability Scale</div>', unsafe_allow_html=True)

probability_table = pd.DataFrame({
    "Operational Probability": ["35%","45%","55%","65%","75%","85%","95%"],
    "Risk Category": ["Very Low","Low","Moderately Low","Moderate","Moderately High","High","Very High"]
})
st.dataframe(probability_table, use_container_width=True, hide_index=True)


# ==========================================================
# MODEL INFORMATION
# ==========================================================

st.markdown('<div class="section-title">Model Information</div>', unsafe_allow_html=True)

info1, info2, info3 = st.columns(3)

with info1:
    st.markdown("""
    <div style="background:#EDE9FE;border:1px solid #DDD6FE;border-radius:8px;padding:14px 16px;">
    <div style="color:#5B21B6;font-size:12px;font-weight:700;margin-bottom:4px;">Model</div>
    <div style="color:#4C1D95;font-size:13px;">CatBoost Classifier</div>
    </div>""", unsafe_allow_html=True)

with info2:
    st.markdown("""
    <div style="background:#EDE9FE;border:1px solid #DDD6FE;border-radius:8px;padding:14px 16px;">
    <div style="color:#5B21B6;font-size:12px;font-weight:700;margin-bottom:4px;">Meteorological Features</div>
    <div style="color:#4C1D95;font-size:13px;">Upper-Air Stability Indices</div>
    </div>""", unsafe_allow_html=True)

with info3:
    st.markdown("""
    <div style="background:#EDE9FE;border:1px solid #DDD6FE;border-radius:8px;padding:14px 16px;">
    <div style="color:#5B21B6;font-size:12px;font-weight:700;margin-bottom:4px;">Prediction Type</div>
    <div style="color:#4C1D95;font-size:13px;">Operational Thunderstorm Forecast</div>
    </div>""", unsafe_allow_html=True)


# ==========================================================
# NOTES
# ==========================================================

st.markdown('<div class="section-title">Notes</div>', unsafe_allow_html=True)

st.markdown("""
- Operational probabilities (35–95%) are mapped from the CatBoost model probability using calibrated operational thresholds.
- The prediction is generated using the latest available upper-air sounding and derived meteorological indices.
- Higher operational probabilities indicate a greater likelihood of thunderstorm occurrence.
- The prediction should be interpreted together with current synoptic observations and radar imagery.
""")


# ==========================================================
# FOOTER
# ==========================================================

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<hr style="margin-top:30px;margin-bottom:15px;">', unsafe_allow_html=True)

left, right = st.columns([3, 1])

with left:
    st.caption("Real-Time Thunderstorm Prediction Dashboard")
    st.caption("CatBoost Machine Learning • MetPy • Upper-Air Soundings")

with right:
    st.caption(f"Generated: {datetime.now().strftime('%d %b %Y %H:%M')}")