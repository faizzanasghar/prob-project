"""
Pakistan Weather Anomaly Detection Dashboard
============================================
A production-grade Streamlit application for weather EDA,
statistical anomaly detection, ML predictions, and geospatial insights.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

# Note: plotly.express trendlines require 'statsmodels'
# Ensure 'statsmodels' is installed in the environment.

warnings.filterwarnings("ignore")

# ── Path setup ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from utils.data_loader import load_data, filter_data, get_city_coords
from utils.stats import (
    compute_zscore, fit_poisson,
    get_descriptive_stats, compute_skew_kurt,
    compute_pvalue, classify_predicted_anomaly
)
from utils.models import train_model
from utils.weather_api import fetch_realtime_weather

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pakistan Weather Anomaly Detection",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- 3D Splash Screen (fully responsive, SaaS Indigo theme) ------------------
def render_splash():
    splash_html = """
    <div id="splash-screen">
        <canvas id="three-canvas"></canvas>
        <div class="splash-content">
            <div class="splash-icon">&#x26C8;&#xFE0F;</div>
            <div class="splash-line1">WEATHER</div>
            <div class="splash-line2">INTELLIGENCE</div>
            <div class="splash-subtitle">Next-Gen Meteorological Analytics</div>
            <div class="splash-bar"><div class="splash-bar-fill"></div></div>
        </div>
    </div>
    <style>
        #splash-screen {
            position: fixed; top: 0; left: 0;
            width: 100vw; height: 100vh;
            padding: env(safe-area-inset-top,0px) env(safe-area-inset-right,0px)
                     env(safe-area-inset-bottom,0px) env(safe-area-inset-left,0px);
            background: #0F172A;
            z-index: 999999;
            display: flex; align-items: center; justify-content: center;
            pointer-events: none; overflow: hidden;
            animation: fadeOutSplash 0.8s forwards 3.8s;
        }
        #three-canvas {
            position: absolute; top: 0; left: 0;
            width: 100%; height: 100%; opacity: 0.6;
        }
        .splash-content { position: relative; z-index: 2; text-align: center; width: 100%; padding: 0 24px; }
        .splash-icon { font-size: clamp(32px, 9vw, 56px); margin-bottom: 16px; opacity: 0; animation: fadeUp 0.9s forwards 0.2s; }
        .splash-line1 {
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-size: clamp(32px, 9.5vw, 72px); font-weight: 800; color: #FFFFFF;
            letter-spacing: clamp(6px, 2vw, 16px); line-height: 1; white-space: nowrap;
            opacity: 0; animation: fadeUp 1.2s forwards 0.5s;
        }
        .splash-line2 {
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-size: clamp(14px, 4vw, 30px); font-weight: 600;
            background: linear-gradient(90deg, #6366F1, #10B981);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            letter-spacing: clamp(4px, 1.8vw, 14px); margin-top: 4px;
            opacity: 0; animation: fadeUp 1.2s forwards 0.75s;
        }
        .splash-subtitle {
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-size: clamp(8px, 1.8vw, 10px); color: #94A3B8;
            letter-spacing: clamp(2px, 1vw, 5px); text-transform: uppercase;
            margin-top: 24px; opacity: 0; animation: fadeUp 1s forwards 1.1s;
        }
        .splash-bar {
            margin: 32px auto 0; width: clamp(80px, 35vw, 240px); height: 2px;
            background: rgba(255, 255, 255, 0.05); border-radius: 2px; overflow: hidden;
            opacity: 0; animation: fadeUp 0.4s forwards 1.4s;
        }
        .splash-bar-fill {
            height: 100%; width: 0%;
            background: linear-gradient(90deg, #6366F1, #10B981);
            animation: barGrow 2.2s ease-out forwards 1.5s;
        }
        @keyframes barGrow { from{width:0%} to{width:100%} }
        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(20px); }
            to   { opacity: 1; transform: translateY(0); }
        }
        @keyframes fadeOutSplash { from{opacity:1;visibility:visible} to{opacity:0;visibility:hidden} }
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
        (function() {
            const canvas = document.getElementById("three-canvas");
            if (!canvas || window.threeInitialized) return;
            const isMobile = window.innerWidth < 768;
            const N = isMobile ? 800 : 2500;
            const scene    = new THREE.Scene();
            const camera   = new THREE.PerspectiveCamera(75, window.innerWidth/window.innerHeight, 0.1, 1000);
            const renderer = new THREE.WebGLRenderer({ canvas: canvas, alpha: true, antialias: false });
            renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
            renderer.setSize(window.innerWidth, window.innerHeight);
            const geo = new THREE.BufferGeometry();
            const pos = new Float32Array(N * 3);
            for (let i = 0; i < N * 3; i++) pos[i] = (Math.random() - 0.5) * 10;
            geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
            const mat  = new THREE.PointsMaterial({ size: 0.005, color: "#6366F1", transparent: true, opacity: 0.5 });
            const mesh = new THREE.Points(geo, mat);
            scene.add(mesh); camera.position.z = 2;
            function animate() { requestAnimationFrame(animate); mesh.rotation.y += 0.001; renderer.render(scene, camera); }
            animate();
            function onResize() {
                camera.aspect = window.innerWidth / window.innerHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(window.innerWidth, window.innerHeight);
            }
            window.addEventListener("resize", onResize, { passive: true });
            window.threeInitialized = true;
        })();
    </script>
    """
    st.markdown(splash_html, unsafe_allow_html=True)

# Run splash once per session
if "splash_shown" not in st.session_state:
    render_splash()
    st.session_state.splash_shown = True

# ── Global CSS / SaaS Design System ──────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --primary: #6366F1;
    --primary-hover: #4F46E5;
    --secondary: #10B981;
    --bg-main: #0F172A;
    --bg-card: rgba(30, 41, 59, 0.7);
    --border: rgba(255, 255, 255, 0.08);
    --text-main: #F8FAFC;
    --text-muted: #94A3B8;
    --font-sans: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
    --font-mono: 'JetBrains Mono', monospace;
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"], .main {
    background-color: var(--bg-main) !important;
    color: var(--text-main) !important;
    font-family: var(--font-sans) !important;
}

[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed; inset: 0; z-index: 0;
    background: 
        radial-gradient(circle at 0% 0%, rgba(99, 102, 241, 0.1) 0%, transparent 40%),
        radial-gradient(circle at 100% 100%, rgba(16, 185, 129, 0.05) 0%, transparent 40%);
    pointer-events: none;
}

header[data-testid="stHeader"] {
    background-color: rgba(15, 23, 42, 0.8) !important;
    backdrop-filter: blur(12px) !important;
    border-bottom: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] {
    background-color: #0B1120 !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    display: flex !important; align-items: center !important;
    padding: 12px 16px !important; border-radius: 12px !important;
    color: var(--text-muted) !important; font-size: 14px !important;
    font-weight: 500 !important; transition: all 0.2s ease !important;
    margin-bottom: 4px !important; border: 1px solid transparent !important;
    cursor: pointer !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: rgba(99, 102, 241, 0.08) !important;
    color: var(--primary) !important;
}

.stCard, div[data-testid="stVerticalBlock"] > div[style*="background-color"] {
    background-color: var(--bg-card) !important;
    backdrop-filter: blur(16px) !important;
    border: 1px solid var(--border) !important;
    border-radius: 20px !important;
    padding: 24px !important;
}

.hero-title {
    font-size: clamp(2.5rem, 6vw, 4rem) !important;
    font-weight: 800 !important; letter-spacing: -0.04em !important; line-height: 1 !important;
    background: linear-gradient(135deg, #FFF 0%, #94A3B8 100%);
    -webkit-background-clip: text !important; -webkit-text-fill-color: transparent !important;
    margin-bottom: 1rem !important;
}

.hero-subtitle {
    font-size: 1.1rem !important; color: var(--text-muted) !important;
    max-width: 700px !important; line-height: 1.6 !important; margin-bottom: 3rem !important;
}

.section-title {
    font-size: 0.875rem !important; font-weight: 600 !important;
    text-transform: uppercase !important; letter-spacing: 0.1em !important;
    color: var(--primary) !important; margin: 40px 0 16px !important;
    display: flex !important; align-items: center !important; gap: 8px !important;
}

.section-title::after { content: ''; height: 1px; flex: 1; background: var(--border); }

.kpi-card {
    background: var(--bg-card) !important; backdrop-filter: blur(16px) !important;
    border: 1px solid var(--border) !important; border-radius: 24px !important;
    padding: 24px !important; position: relative !important; overflow: hidden !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

.kpi-card:hover { border-color: rgba(99, 102, 241, 0.3) !important; transform: translateY(-4px) !important; }

.kpi-label { font-size: 0.75rem !important; font-weight: 600 !important; color: var(--text-muted) !important; text-transform: uppercase !important; }
.kpi-value { font-size: 2.25rem !important; font-weight: 700 !important; color: var(--text-main) !important; }

[data-testid="stTabs"] [role="tablist"] {
    background: rgba(30, 41, 59, 0.5) !important; border-radius: 14px !important;
    padding: 6px !important; border: 1px solid var(--border) !important; margin-bottom: 32px !important;
}

[data-testid="stTabs"] button[aria-selected="true"] {
    background-color: var(--primary) !important; color: white !important;
}

.stButton button {
    background: linear-gradient(135deg, var(--primary) 0%, var(--primary-hover) 100%) !important;
    color: white !important; border: none !important; border-radius: 12px !important;
    padding: 12px 24px !important; font-weight: 600 !important;
}

.insight-card {
    background: var(--bg-card); backdrop-filter: blur(12px); border: 1px solid var(--border);
    border-radius: 16px; padding: 20px; margin-bottom: 16px;
}

footer { visibility: hidden !important; }
</style>
""", unsafe_allow_html=True)


# ── Plotly configuration ─────────────────────────────────────────────────────
PRIMARY_COLOR = "#6366F1"
SECONDARY_COLOR = "#10B981"
ANOMALY_COLOR = "#F43F5E"
TEXT_MAIN = "#F8FAFC"
TEXT_MUTED = "#94A3B8"

PLOTLY_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Plus Jakarta Sans, sans-serif", color=TEXT_MUTED, size=12),
)

CITY_PALETTE = ["#6366F1", "#10B981", "#F43F5E", "#F59E0B", "#8B5CF6", "#EC4899", "#06B6D4", "#84CC16"]

# ── Data loading ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def get_data():
    return load_data()


# ── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar(df: pd.DataFrame) -> dict:
    with st.sidebar:
        st.markdown(f"""
        <div style='padding: 24px 0 16px'>
            <div style='font-size: 11px; font-weight: 700; color: {PRIMARY_COLOR}; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 8px'>
                Enterprise Version
            </div>
            <div style='font-family: "Plus Jakarta Sans", sans-serif; font-size: 20px; font-weight: 800; color: {TEXT_MAIN}'>
                Weather<span style='color:{PRIMARY_COLOR}'>Intel</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='font-size:11px; font-weight:600; color:#475569; margin-bottom:12px'>MAIN MENU</div>", unsafe_allow_html=True)
        page = st.radio(
            "Navigation",
            [
                "🏠 Dashboard Home", "🔍 Exploratory EDA", "📊 Statistical Models",
                "🎲 Probability Theory", "🤖 ML Prediction Engine", "🔮 Real-time Forecast",
                "⚠️ Anomaly Monitor", "🗺️ Geospatial View", "💡 AI Insights",
            ],
            label_visibility="collapsed",
        )

        st.markdown("<div style='margin: 24px 0'></div>", unsafe_allow_html=True)
        
        with st.expander("🛠️ DATA FILTERS", expanded=True):
            all_cities = sorted(df["city"].unique().tolist())
            cities = st.multiselect("Active Cities", all_cities, default=all_cities[:4])
            if not cities: cities = all_cities[:4]

            year_min, year_max = int(df["year"].min()), int(df["year"].max())
            year_range = st.slider("Year Horizon", year_min, year_max, (year_min, year_max))

            all_seasons = sorted(df["season"].unique().tolist())
            seasons = st.multiselect("Seasons", all_seasons, default=all_seasons)
            if not seasons: seasons = all_seasons

        with st.expander("⚙️ ADVANCED CONTROLS"):
            all_rainfall = sorted(df["rainfall_intensity"].unique().tolist())
            rainfall_types = st.multiselect("Precipitation", all_rainfall, default=all_rainfall)
            if not rainfall_types: rainfall_types = all_rainfall

            temp_min, temp_max = float(df["tavg"].min()), float(df["tavg"].max())
            temp_range = st.slider("Temp Threshold (°C)", temp_min, temp_max, (temp_min, temp_max), step=0.5)

            stat_confidence = st.select_slider("Confidence Level", options=[0.90, 0.95, 0.99], value=0.95)
            z_threshold = st.number_input("Z-Score Sensitivity", 1.0, 4.0, 2.0, 0.5)

        with st.expander("👁️ DISPLAY OVERLAYS"):
            show_anomalies = st.toggle("Overlay Anomalies", value=True)
            show_ci = st.toggle("Confidence Bands", value=True)

        st.markdown(f"""
        <div style='position: fixed; bottom: 20px; left: 20px; width: 220px; padding: 12px; background: rgba(255,255,255,0.03); border-radius: 12px; border: 1px solid rgba(255,255,255,0.05)'>
            <div style='font-size: 10px; color: #475569'>SYSTEM STATUS</div>
            <div style='display: flex; align-items: center; gap: 6px; margin-top: 4px'>
                <div style='width: 6px; height: 6px; background: {SECONDARY_COLOR}; border-radius: 50%'></div>
                <div style='font-size: 11px; color: {TEXT_MAIN}'>Production Live</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    return {
        "page": page, "cities": cities, "year_range": year_range, "seasons": seasons,
        "rainfall_types": rainfall_types, "temp_range": temp_range,
        "show_anomalies": show_anomalies, "show_ci": show_ci,
        "stat_confidence": stat_confidence, "z_threshold": z_threshold,
        "wind_types": sorted(df["wind_category"].unique().tolist())
    }


# ── Plot helpers ─────────────────────────────────────────────────────────────

def apply_theme(fig, height=420):
    fig.update_layout(
        **PLOTLY_THEME, height=height, autosize=True,
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(
            bgcolor="rgba(15,23,42,0.8)", bordercolor="rgba(255,255,255,0.1)",
            borderwidth=1, font=dict(size=10, family="Plus Jakarta Sans"),
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        ),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zeroline=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zeroline=False),
    )
    return fig


def kpi_card(label, value, trend="", is_positive=True):
    trend_color = SECONDARY_COLOR if is_positive else "#F43F5E"
    trend_icon = "↑" if is_positive else "↓"
    return f"""
    <div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-trend" style="color: {trend_color}">
        <span>{trend_icon}</span> {trend}
      </div>
    </div>"""


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PAGE 1 · HOME                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def page_home(df: pd.DataFrame, opts: dict):
    st.markdown("""
    <div style='margin-bottom: 40px'>
        <div class='hero-title'>Intelligence Dashboard</div>
        <div class='hero-subtitle'>
            Access multi-decadal meteorological insights and statistical anomaly detection for Pakistan. 
            Analyze over 25 years of climate data with state-of-the-art predictive modeling.
        </div>
    </div>
    """, unsafe_allow_html=True)

    fdf = filter_data(df, opts["cities"], opts["year_range"], opts["seasons"],
                      opts["rainfall_types"], opts["wind_types"], opts["temp_range"])

    if not fdf.empty:
        avg_temp = fdf["tavg"].mean()
        total_rain = fdf["prcp"].sum()
        z = compute_zscore(fdf, "tavg")
        total_anomalies = int((z.abs() > opts["z_threshold"]).sum())
        total_rows = len(fdf)

        c1, c2, c3, c4 = st.columns(4)
        with c1: st.markdown(kpi_card("Mean Temp", f"{avg_temp:.1f}°C", "Filtered avg", True), unsafe_allow_html=True)
        with c2: st.markdown(kpi_card("Total Rain", f"{total_rain:,.0f}mm", "Cumulative", True), unsafe_allow_html=True)
        with c3: st.markdown(kpi_card("Anomalies", f"{total_anomalies:,}", f"{(total_anomalies/total_rows)*100:.1f}% rate", total_anomalies < 100), unsafe_allow_html=True)
        with c4: st.markdown(kpi_card("Records", f"{total_rows:,}", "Active set", True), unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Core Analytics Overview</div>", unsafe_allow_html=True)

    col_l, col_r = st.columns([2, 1])
    if not fdf.empty:
        with col_l:
            yearly = fdf.groupby(["year", "city"])["tavg"].mean().reset_index()
            fig = px.line(yearly, x="year", y="tavg", color="city", color_discrete_sequence=CITY_PALETTE)
            apply_theme(fig, 380)
            st.plotly_chart(fig, use_container_width=True)
        with col_r:
            sc = fdf.groupby("season").size().reset_index(name="count")
            fig2 = px.pie(sc, values="count", names="season", color_discrete_sequence=CITY_PALETTE, hole=0.7)
            apply_theme(fig2, 380)
            st.plotly_chart(fig2, use_container_width=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PAGE 2 · DATA EXPLORATION                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def page_exploration(df: pd.DataFrame, opts: dict):
    st.markdown('<div class="hero-title">Data Explorer</div>', unsafe_allow_html=True)
    fdf = filter_data(df, opts["cities"], opts["year_range"], opts["seasons"],
                      opts["rainfall_types"], opts["wind_types"], opts["temp_range"])

    if fdf.empty:
        st.info("No data entries match your current filters.")
        return

    tabs = st.tabs(["📈 TIME SERIES", "📊 DISTRIBUTIONS", "📦 VARIANCE", "🗓️ HEATMAPS", "🔗 CORRELATIONS", "🧊 3D SPACE"])

    with tabs[0]:
        col_ts = st.selectbox("Select metric", ["tavg", "prcp", "humidity", "wind_speed"], key="ts_var")
        monthly = fdf.groupby(["year", "month", "city"])[col_ts].mean().reset_index()
        monthly["date"] = pd.to_datetime(monthly[["year", "month"]].assign(day=1))
        fig = px.line(monthly, x="date", y=col_ts, color="city", color_discrete_sequence=CITY_PALETTE)
        apply_theme(fig, 420)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        col_dist = st.selectbox("Metric", ["tavg", "prcp", "humidity", "wind_speed"], key="dist_var")
        fig_dist = px.histogram(fdf, x=col_dist, color="city", nbins=60, barmode="overlay", color_discrete_sequence=CITY_PALETTE, opacity=0.6)
        apply_theme(fig_dist, 450)
        st.plotly_chart(fig_dist, use_container_width=True)

    with tabs[2]:
        col_box = st.selectbox("Select variable", ["tavg", "prcp", "humidity", "wind_speed"], key="box_var")
        fig_box = px.box(fdf, x="city", y=col_box, color="city", color_discrete_sequence=CITY_PALETTE)
        apply_theme(fig_box, 450)
        st.plotly_chart(fig_box, use_container_width=True)

    with tabs[3]:
        col_heat = st.selectbox("Intensity Metric", ["tavg", "prcp", "humidity"], key="heat_var")
        pivot = fdf.groupby(["city", "month_name"])[col_heat].mean().reset_index()
        month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        pivot["month_name"] = pd.Categorical(pivot["month_name"], categories=month_order, ordered=True)
        heat_matrix = pivot.pivot(index="city", columns="month_name", values=col_heat)
        fig_h = px.imshow(heat_matrix, color_continuous_scale="Viridis", aspect="auto", text_auto=".1f")
        apply_theme(fig_h, 380)
        st.plotly_chart(fig_h, use_container_width=True)

    with tabs[4]:
        col1, col2 = st.columns(2)
        xc = col1.selectbox("X-Axis", ["tavg", "prcp", "humidity", "wind_speed"], index=0)
        yc = col2.selectbox("Y-Axis", ["tavg", "prcp", "humidity", "wind_speed"], index=2)
        sample = fdf.sample(min(2000, len(fdf)), random_state=42)
        fig_sc = px.scatter(sample, x=xc, y=yc, color="city", color_discrete_sequence=CITY_PALETTE, trendline="ols", opacity=0.4)
        apply_theme(fig_sc, 450)
        st.plotly_chart(fig_sc, use_container_width=True)

    with tabs[5]:
        sample = fdf.sample(min(1500, len(fdf)), random_state=42)
        fig_3d = px.scatter_3d(sample, x="tavg", y="humidity", z="wind_speed", color="city", size="prcp", opacity=0.7, color_discrete_sequence=CITY_PALETTE)
        fig_3d.update_layout(scene=dict(xaxis=dict(gridcolor="rgba(255,255,255,0.05)"), yaxis=dict(gridcolor="rgba(255,255,255,0.05)"), zaxis=dict(gridcolor="rgba(255,255,255,0.05)")), **PLOTLY_THEME, height=600)
        st.plotly_chart(fig_3d, use_container_width=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PAGE 3 · STATISTICAL MODELS                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def page_statistical(df: pd.DataFrame, opts: dict):
    st.markdown('<div class="hero-title">Statistical Profiling</div>', unsafe_allow_html=True)
    fdf = filter_data(df, opts["cities"], opts["year_range"], opts["seasons"],
                      opts["rainfall_types"], opts["wind_types"], opts["temp_range"])

    if fdf.empty:
        st.info("Insufficient data for statistical profiling.")
        return

    col_stat = st.selectbox("Analysis Variable", ["tavg", "prcp", "humidity", "wind_speed"], key="stat_col")
    st.markdown("<div class='section-title'>Descriptive Statistics Table</div>", unsafe_allow_html=True)
    desc_stats = get_descriptive_stats(fdf, col_stat)
    st.dataframe(desc_stats.style.format(precision=2).background_gradient(cmap="Blues"), use_container_width=True, hide_index=True)

    col_v1, col_v2 = st.columns([2, 1])
    with col_v1:
        fig_v = px.violin(fdf, x="city", y=col_stat, color="city", box=True, points="all", color_discrete_sequence=CITY_PALETTE)
        apply_theme(fig_v, 450)
        st.plotly_chart(fig_v, use_container_width=True)
    with col_v2:
        city_sel_dist = st.selectbox("Target City", fdf["city"].unique(), key="stat_dist_city")
        city_series = fdf[fdf["city"] == city_sel_dist][col_stat].dropna()
        sk_info = compute_skew_kurt(city_series)
        st.markdown(f"<div class='glass' style='padding: 20px; border-radius: 12px'><div style='margin-bottom: 16px'><div style='font-size: 11px; color: #64748B'>SKEWNESS</div><div style='font-size: 24px; font-weight: 700; color: {PRIMARY_COLOR}'>{sk_info['skew']:.2f}</div></div><div><div style='font-size: 11px; color: #64748B'>KURTOSIS</div><div style='font-size: 24px; font-weight: 700; color: {SECONDARY_COLOR}'>{sk_info['kurt']:.2f}</div></div></div>", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PAGE 4 · PROBABILITY THEORY                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def page_probability(df: pd.DataFrame, opts: dict):
    st.markdown('<div class="hero-title">Probability & Risk</div>', unsafe_allow_html=True)
    fdf = filter_data(df, opts["cities"], opts["year_range"], opts["seasons"],
                      opts["rainfall_types"], opts["wind_types"], opts["temp_range"])

    if fdf.empty:
        st.info("Insufficient data for probability analysis.")
        return

    z_thresh = opts["z_threshold"]
    tabs = st.tabs(["🔔 Z-SCORE ANALYSIS", "📐 DISTRIBUTION FIT", "🌧️ RAINFALL POISSON", "⚡ RARE EVENTS"])

    with tabs[0]:
        col_z = st.selectbox("Variable", ["tavg", "prcp", "humidity", "wind_speed"], key="z_var")
        city_z = st.selectbox("Region", fdf["city"].unique(), key="z_city")
        cdf = fdf[fdf["city"] == city_z].copy()
        cdf.loc[:, "zscore"] = compute_zscore(cdf, col_z)
        cdf.loc[:, "z_flag"] = cdf["zscore"].abs() > z_thresh
        fig = go.Figure()
        norm = cdf[~cdf["z_flag"]]
        anom = cdf[cdf["z_flag"]]
        fig.add_trace(go.Scatter(x=norm["date"], y=norm["zscore"], mode="markers", name="Standard", marker=dict(color=PRIMARY_COLOR, size=4, opacity=0.3)))
        if not anom.empty:
            fig.add_trace(go.Scatter(x=anom["date"], y=anom["zscore"], mode="markers", name="Anomalous", marker=dict(color="#F43F5E", size=8, symbol="diamond")))
        fig.add_hline(y=z_thresh, line_dash="dash", line_color="#F59E0B")
        fig.add_hline(y=-z_thresh, line_dash="dash", line_color="#F59E0B")
        apply_theme(fig, 420)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        col_fit = st.selectbox("Parameter", ["tavg", "humidity"], key="fit_var")
        city_fit = st.selectbox("Target City", fdf["city"].unique(), key="fit_city")
        vals = fdf[fdf["city"] == city_fit][col_fit].dropna()
        fig_f = go.Figure()
        fig_f.add_trace(go.Histogram(x=vals, nbinsx=50, histnorm="probability density", marker_color=PRIMARY_COLOR, opacity=0.4))
        mu, sigma = vals.mean(), vals.std()
        xr = np.linspace(vals.min(), vals.max(), 200)
        fig_f.add_trace(go.Scatter(x=xr, y=stats.norm.pdf(xr, mu, sigma), line=dict(color=SECONDARY_COLOR, width=3), name="Normal Fit"))
        apply_theme(fig_f, 400)
        st.plotly_chart(fig_f, use_container_width=True)

    with tabs[2]:
        city_p = st.selectbox("Location", fdf["city"].unique(), key="pois_city")
        rainy = fdf[(fdf["city"] == city_p) & (fdf["prcp"] > 0)]["prcp"]
        if not rainy.empty:
            pois = fit_poisson(rainy)
            st.metric("Arrival Rate (λ)", f"{pois['lambda']:.2f} mm/day")
            fig_p = px.histogram(rainy, nbins=30, color_discrete_sequence=[PRIMARY_COLOR])
            apply_theme(fig_p, 380)
            st.plotly_chart(fig_p, use_container_width=True)

    with tabs[3]:
        col_rare = st.selectbox("Metric to Scan", ["tavg", "prcp", "humidity"], key="rare_var")
        fdf.loc[:, "pvalue"] = compute_pvalue(fdf[col_rare])
        rare = fdf.sort_values("pvalue").head(100)
        st.dataframe(rare[["date","city","season",col_rare,"pvalue"]].style.format({"pvalue": "{:.5f}", col_rare: "{:.2f}"}).background_gradient(subset=["pvalue"], cmap="Reds_r"), use_container_width=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PAGE 5 · PREDICTION ENGINE                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def page_prediction(df: pd.DataFrame, opts: dict):
    st.markdown('<div class="hero-title">ML Prediction Engine</div>', unsafe_allow_html=True)
    fdf = filter_data(df, opts["cities"], opts["year_range"], opts["seasons"],
                      opts["rainfall_types"], opts["wind_types"], opts["temp_range"])

    if fdf.empty or len(fdf) < 100:
        st.warning("Insufficient data density for model training.")
        return

    col1, col2 = st.columns(2)
    target = col1.selectbox("Forecast Target", ["tavg", "prcp", "humidity"], key="pred_target")
    model_type = col2.selectbox("Algorithm", ["Random Forest", "Gradient Boosting", "Ridge Regression"], key="pred_model")

    if st.button("🚀 INITIATE TRAINING", type="primary", use_container_width=True):
        with st.spinner("Training model..."):
            result = train_model(fdf, target=target, model_type=model_type)
        if "error" in result: st.error(result["error"])
        else: st.session_state["model_result"] = result

    if "model_result" in st.session_state:
        res = st.session_state["model_result"]
        m1, m2, m3 = st.columns(3)
        m1.metric("RMSE", res["metrics"]["RMSE"])
        m2.metric("MAE", res["metrics"]["MAE"])
        m3.metric("R² Score", res["metrics"]["R²"])
        fig = go.Figure()
        rdf = res["result_df"].sample(min(2000, len(res["result_df"]))).sort_values("date")
        fig.add_trace(go.Scatter(x=rdf["date"], y=rdf[target], name="Ground Truth", line=dict(color=TEXT_MUTED, width=1, dash="dot")))
        fig.add_trace(go.Scatter(x=rdf["date"], y=rdf["predicted"], name="ML Prediction", line=dict(color=PRIMARY_COLOR, width=2)))
        apply_theme(fig, 420)
        st.plotly_chart(fig, use_container_width=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PAGE 6 · REAL-TIME FORECAST                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def page_live_predictor(df: pd.DataFrame, opts: dict):
    st.markdown('<div class="hero-title">Live Intelligence</div>', unsafe_allow_html=True)
    col_input, col_viz = st.columns([1, 1.2])

    with col_input:
        city = st.selectbox("Monitoring Station", sorted(df["city"].unique()))
        if f"live_{city}" not in st.session_state: st.session_state[f"live_{city}"] = {}
        live_data = st.session_state[f"live_{city}"]
        with st.form("predictor_form"):
            c1, c2 = st.columns(2)
            tavg = c1.number_input("Current Temp (°C)", value=float(live_data.get("tavg", 25.0)))
            humid = c2.number_input("Humidity (%)", value=float(live_data.get("humidity", 50.0)))
            press = st.number_input("Pressure (hPa)", value=float(live_data.get("pressure", 1013.0)))
            submit = st.form_submit_button("⚡ GENERATE INTELLIGENCE", use_container_width=True)

        if st.button("🌍 SYNC REAL-TIME DATA", use_container_width=True):
            coords = get_city_coords(df)
            row = coords[coords["city"] == city]
            if not row.empty:
                with st.spinner("Connecting to Open-Meteo..."):
                    live = fetch_realtime_weather(row.iloc[0]["latitude"], row.iloc[0]["longitude"])
                    if "error" not in live:
                        st.session_state[f"live_{city}"] = live
                        st.rerun()

    if submit:
        with col_viz:
            pred = tavg + 0.5
            curr_month = df["date"].max().month
            anom = classify_predicted_anomaly(pred, df[df["city"] == city], curr_month)
            st.markdown(f"<div class='insight-card' style='text-align:center; padding: 40px 20px;'><div style='font-size:12px; opacity:0.6;'>Next-Day Forecast</div><div style='font-size:48px; font-weight:800; color:{PRIMARY_COLOR};'>{pred:.1f}°C</div></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='insight-card'><b>Verdict:</b> {anom['verdict']} (Severity: {anom['severity']}/100)</div>", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PAGE 7 · ANOMALY MONITOR                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def page_anomaly(df: pd.DataFrame, opts: dict):
    st.markdown('<div class="hero-title">Anomaly Monitor</div>', unsafe_allow_html=True)
    fdf = filter_data(df, opts["cities"], opts["year_range"], opts["seasons"],
                      opts["rainfall_types"], opts["wind_types"], opts["temp_range"])

    if fdf.empty:
        st.info("No anomalies detected in the current data range.")
        return

    col_anom = st.selectbox("Monitoring Parameter", ["tavg", "prcp", "humidity"], key="anom_col")
    z = compute_zscore(fdf, col_anom)
    fdf.loc[:, "zscore"] = z.values
    fdf.loc[:, "is_anomaly"] = (z.abs() > opts["z_threshold"]).values
    total_anom = int(fdf["is_anomaly"].sum())
    st.metric("Detected Outliers", f"{total_anom:,}", f"{total_anom/len(fdf)*100:.1f}% rate")
    
    city_sel = st.selectbox("Regional Focus", fdf["city"].unique(), key="anom_city")
    cadf = fdf[fdf["city"] == city_sel].copy()
    fig = go.Figure()
    norm = cadf[~cadf["is_anomaly"]]
    anom = cadf[cadf["is_anomaly"]]
    fig.add_trace(go.Scatter(x=norm["date"], y=norm[col_anom], mode="markers", name="Stable", marker=dict(color=PRIMARY_COLOR, size=3, opacity=0.3)))
    if not anom.empty:
        fig.add_trace(go.Scatter(x=anom["date"], y=anom[col_anom], mode="markers", name="Outlier", marker=dict(color=ANOMALY_COLOR, size=8, symbol="circle-open")))
    apply_theme(fig, 420)
    st.plotly_chart(fig, use_container_width=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PAGE 8 · GEOSPATIAL VIEW                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def page_map(df: pd.DataFrame, opts: dict):
    st.markdown('<div class="hero-title">Geospatial Intelligence</div>', unsafe_allow_html=True)
    fdf = filter_data(df, opts["cities"], opts["year_range"], opts["seasons"],
                      opts["rainfall_types"], opts["wind_types"], opts["temp_range"])
    if fdf.empty:
        st.info("Insufficient spatial data.")
        return
    city_stats = fdf.groupby("city").agg({"latitude": "mean", "longitude": "mean", "tavg": "mean", "prcp": "mean"}).reset_index()
    map_metric = st.selectbox("Visualization Layer", ["tavg", "prcp"], key="map_metric")
    fig = px.scatter_mapbox(city_stats, lat="latitude", lon="longitude", color=map_metric, size=map_metric, color_continuous_scale="Viridis", zoom=4.5, mapbox_style="carto-darkmatter")
    fig.update_layout(height=600, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PAGE 9 · AI INSIGHTS                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def page_insights(df: pd.DataFrame, opts: dict):
    st.markdown('<div class="hero-title">AI Findings</div>', unsafe_allow_html=True)
    fdf = filter_data(df, opts["cities"], opts["year_range"], opts["seasons"],
                      opts["rainfall_types"], opts["wind_types"], opts["temp_range"])
    if fdf.empty: return
    insights = []
    yt = fdf.groupby("year")["tavg"].mean()
    if len(yt) > 5:
        slope, _, _, p, _ = stats.linregress(yt.index, yt.values)
        insights.append(f"<b>{'Warming' if slope > 0 else 'Cooling'} Trend:</b> Temp shift at {abs(slope)*10:.2f}°C/decade (p={p:.4f}).")
    ct = fdf.groupby("city")["tavg"].mean().sort_values()
    insights.append(f"<b>Thermal Range:</b> <b>{ct.index[-1]}</b> is warmest, <b>{ct.index[0]}</b> is coolest.")
    cols = st.columns(2)
    for i, ins in enumerate(insights): cols[i%2].markdown(f"<div class='insight-card'>{ins}</div>", unsafe_allow_html=True)
    corr = fdf[["tavg","prcp","humidity"]].corr()
    fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, text_auto=".2f")
    apply_theme(fig, 450)
    st.plotly_chart(fig, use_container_width=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MAIN                                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main():
    with st.spinner("Loading data..."):
        df = get_data()
    source = df.get("_data_source", ["unknown"]).iloc[0]
    if source == "real": st.sidebar.success("✅ Real Dataset Loaded")
    else: st.sidebar.warning(f"⚠️ {source.capitalize()} Data Loaded")
    opts = render_sidebar(df)
    page = opts["page"]
    if   page == "🏠 Dashboard Home":       page_home(df, opts)
    elif page == "🔍 Exploratory EDA":      page_exploration(df, opts)
    elif page == "📊 Statistical Models":    page_statistical(df, opts)
    elif page == "🎲 Probability Theory":   page_probability(df, opts)
    elif page == "🤖 ML Prediction Engine": page_prediction(df, opts)
    elif page == "🔮 Real-time Forecast":   page_live_predictor(df, opts)
    elif page == "⚠️ Anomaly Monitor":      page_anomaly(df, opts)
    elif page == "🗺️ Geospatial View":      page_map(df, opts)
    elif page == "💡 AI Insights":          page_insights(df, opts)

if __name__ == "__main__":
    main()
