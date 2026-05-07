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
from plotly.subplots import make_subplots
import streamlit as st
from scipy import stats
from streamlit_lottie import st_lottie
import json
import requests

warnings.filterwarnings("ignore")

# ── Path setup ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from utils.data_loader import load_data, filter_data, get_city_coords
from utils.stats import (
    compute_ci, flag_ci_anomalies, compute_zscore, flag_zscore_anomalies,
    compute_pvalue, flag_pvalue_anomalies, normality_test, fit_poisson,
    combined_anomaly_flag, monthly_stats, anomaly_frequency,
    get_descriptive_stats, compute_skew_kurt, get_ci_table,
    classify_predicted_anomaly
)
from utils.models import (
    train_model, get_most_extreme_year, 
    train_next_day_model, predict_next_day
)
from utils.weather_api import fetch_realtime_weather
import streamlit.components.v1 as components

# ── Helpers ──────────────────────────────────────────────────────────────────

def load_lottieurl(url: str):
    """Safely load a Lottie animation from a URL."""
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pakistan Weather Anomaly Detection",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 3D Splash Screen ────────────────────────────────────────────────────────
def render_splash():
    splash_html = """
    <div id="splash-screen">
        <canvas id="three-canvas"></canvas>
        <div class="splash-content">
            <div class="splash-title">WEATHER<span>INTELLIGENCE</span></div>
            <div class="splash-subtitle">Advanced Meteorological Computing</div>
        </div>
    </div>
    <style>
        #splash-screen {
            position: fixed;
            top: 0; left: 0;
            width: 100vw; height: 100vh;
            background-color: #05070A;
            z-index: 999999;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            pointer-events: none;
            /* Pure CSS Fallback: Fade out and hide after 4 seconds */
            animation: fadeOutSplash 1s forwards 3.5s;
        }
        #three-canvas {
            position: absolute;
            top: 0; left: 0;
            width: 100%; height: 100%;
            opacity: 0.6;
        }
        .splash-content { z-index: 2; text-align: center; }
        .splash-title {
            font-family: 'Segoe UI', sans-serif;
            font-size: 42px;
            font-weight: 800;
            color: #FFFFFF;
            letter-spacing: 8px;
            margin-bottom: 10px;
            opacity: 0;
            animation: fadeInText 1.5s forwards 0.5s;
        }
        .splash-title span { color: #58A6FF; }
        .splash-subtitle {
            font-family: 'Segoe UI', sans-serif;
            font-size: 11px;
            color: #8B949E;
            letter-spacing: 4px;
            text-transform: uppercase;
            opacity: 0;
            animation: fadeInText 1.5s forwards 1.2s;
        }
        @keyframes fadeInText {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes fadeOutSplash {
            from { opacity: 1; visibility: visible; }
            to { opacity: 0; visibility: hidden; }
        }
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
        // Only run Three.js if it's the first time
        if (!window.threeInitialized) {
            const canvas = document.getElementById('three-canvas');
            if (canvas) {
                const scene = new THREE.Scene();
                const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
                const renderer = new THREE.WebGLRenderer({ canvas: canvas, alpha: true, antialias: true });
                renderer.setSize(window.innerWidth, window.innerHeight);
                
                const particlesGeometry = new THREE.BufferGeometry();
                const posArray = new Float32Array(3000 * 3);
                for(let i=0; i < 3000 * 3; i++) { posArray[i] = (Math.random() - 0.5) * 10; }
                particlesGeometry.setAttribute('position', new THREE.BufferAttribute(posArray, 3));
                
                const material = new THREE.PointsMaterial({ size: 0.005, color: '#58A6FF', transparent: true, opacity: 0.5 });
                const particlesMesh = new THREE.Points(particlesGeometry, material);
                scene.add(particlesMesh);
                camera.position.z = 2;
                
                function animate() {
                    requestAnimationFrame(animate);
                    particlesMesh.rotation.y += 0.001;
                    renderer.render(scene, camera);
                }
                animate();
                window.threeInitialized = true;
            }
        }
    </script>
    """
    st.markdown(splash_html, unsafe_allow_html=True)

# Run splash once per session
if "splash_shown" not in st.session_state:
    render_splash()
    st.session_state.splash_shown = True

# ── Global CSS / Dark Theme ──────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

/* ══ RESET & BASE ══ */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stMain"], .main {
    background: #03080f !important;
    color: #d8e8f8 !important;
    font-family: 'Syne', sans-serif !important;
}
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    z-index: 0;
    background:
        radial-gradient(ellipse 80% 50% at 10% 0%, rgba(56,189,248,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 90% 100%, rgba(129,140,248,0.06) 0%, transparent 60%),
        radial-gradient(ellipse 50% 60% at 50% 50%, rgba(6,182,212,0.03) 0%, transparent 70%);
    pointer-events: none;
}

/* ══ SIDEBAR ══ */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #040c17 0%, #060e1c 100%) !important;
    border-right: 1px solid rgba(56,189,248,0.12) !important;
    box-shadow: 4px 0 30px rgba(0,0,0,0.6) !important;
}
[data-testid="stSidebar"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #06b6d4, #38bdf8);
    background-size: 200% 100%;
    animation: shimmerBar 3s ease infinite;
}
@keyframes shimmerBar {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    display: flex !important;
    align-items: center !important;
    padding: 9px 14px !important;
    border-radius: 8px !important;
    color: rgba(180,210,240,0.65) !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    letter-spacing: 0.01em !important;
    transition: all 0.2s ease !important;
    margin-bottom: 2px !important;
    border: 1px solid transparent !important;
    cursor: pointer !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: rgba(56,189,248,0.07) !important;
    color: #38bdf8 !important;
    border-color: rgba(56,189,248,0.18) !important;
}
[data-testid="stSidebar"] .stSlider [role="slider"] {
    background: #38bdf8 !important;
    border: 2px solid #03080f !important;
    box-shadow: 0 0 8px rgba(56,189,248,0.5) !important;
}

/* ══ PAGE TITLES ══ */
.page-title {
    font-family: 'Syne', sans-serif !important;
    font-size: 2.4rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em !important;
    background: linear-gradient(135deg, #e0f2fe 0%, #38bdf8 35%, #818cf8 65%, #c7d2fe 100%) !important;
    background-size: 200% 200% !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    animation: titleFlow 5s ease infinite !important;
    margin-bottom: 4px !important;
    line-height: 1.15 !important;
}
@keyframes titleFlow {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.page-subtitle {
    font-size: 12px !important;
    color: rgba(148,180,220,0.55) !important;
    font-family: 'DM Mono', monospace !important;
    letter-spacing: 0.04em !important;
    margin-bottom: 28px !important;
}

/* ══ SECTION TITLES ══ */
.section-title {
    font-family: 'Syne', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    color: #e0f2fe !important;
    border-left: 3px solid #38bdf8 !important;
    padding-left: 12px !important;
    margin: 32px 0 16px !important;
    letter-spacing: 0.01em !important;
}

/* ══ KPI CARDS ══ */
.kpi-card {
    background: rgba(255,255,255,0.03) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-top: 1px solid rgba(56,189,248,0.15) !important;
    border-radius: 16px !important;
    padding: 22px 24px !important;
    margin-bottom: 8px !important;
    position: relative !important;
    overflow: hidden !important;
    transition: transform 0.25s cubic-bezier(0.34,1.56,0.64,1),
                box-shadow 0.25s ease,
                border-color 0.25s ease !important;
}
.kpi-card:hover {
    transform: translateY(-5px) !important;
    box-shadow: 0 20px 40px rgba(0,0,0,0.5), 0 0 20px rgba(56,189,248,0.07) !important;
    border-color: rgba(56,189,248,0.25) !important;
}
.kpi-label {
    font-family: 'DM Mono', monospace !important;
    font-size: 10px !important;
    font-weight: 500 !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    color: rgba(148,180,220,0.5) !important;
    margin-bottom: 10px !important;
}
.kpi-value {
    font-family: 'Syne', sans-serif !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    color: #38bdf8 !important;
    line-height: 1.1 !important;
    letter-spacing: -0.02em !important;
}
.kpi-sub {
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
    color: rgba(100,140,180,0.45) !important;
    margin-top: 6px !important;
}

/* ══ NATIVE ST.METRIC ══ */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.03) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-top: 1px solid rgba(56,189,248,0.2) !important;
    border-radius: 14px !important;
    padding: 18px 20px !important;
    transition: all 0.25s ease !important;
}
[data-testid="stMetric"]:hover {
    border-color: rgba(56,189,248,0.3) !important;
    box-shadow: 0 8px 24px rgba(0,0,0,0.3) !important;
    transform: translateY(-2px) !important;
}
[data-testid="stMetricLabel"] {
    color: rgba(148,180,220,0.6) !important;
    font-size: 11px !important;
    font-family: 'DM Mono', monospace !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
    color: #38bdf8 !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    font-family: 'Syne', sans-serif !important;
}

/* ══ TABS ══ */
[data-testid="stTabs"] [role="tablist"] {
    background: rgba(255,255,255,0.02) !important;
    border-radius: 10px !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    padding: 4px !important;
    gap: 2px !important;
}
[data-testid="stTabs"] button {
    font-family: 'Syne', sans-serif !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    color: rgba(148,180,220,0.55) !important;
    border-radius: 7px !important;
    border: none !important;
    padding: 7px 16px !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.02em !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #38bdf8 !important;
    background: rgba(56,189,248,0.12) !important;
    border-bottom: none !important;
}

/* ══ DATAFRAMES ══ */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(56,189,248,0.12) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* ══ BADGES ══ */
.anomaly-badge {
    display: inline-flex !important;
    align-items: center !important;
    background: rgba(248,81,73,0.1) !important;
    color: #ff6b6b !important;
    border: 1px solid rgba(248,81,73,0.3) !important;
    border-radius: 6px !important;
    padding: 4px 12px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 12px !important;
    animation: anomalyPulse 2s ease infinite !important;
}
@keyframes anomalyPulse {
    0%,100% { box-shadow: 0 0 0 0 rgba(248,81,73,0); }
    50%      { box-shadow: 0 0 8px 2px rgba(248,81,73,0.12); }
}
.normal-badge {
    display: inline-flex !important;
    align-items: center !important;
    background: rgba(52,211,153,0.1) !important;
    color: #34d399 !important;
    border: 1px solid rgba(52,211,153,0.3) !important;
    border-radius: 6px !important;
    padding: 4px 12px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 12px !important;
}

/* ══ INSIGHT CARDS ══ */
.insight-card {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-left: 3px solid #38bdf8 !important;
    border-radius: 0 12px 12px 0 !important;
    padding: 16px 20px !important;
    margin-bottom: 10px !important;
    font-size: 13.5px !important;
    color: rgba(200,225,245,0.85) !important;
    line-height: 1.65 !important;
}

/* ══ PLOTLY CONTAINERS ══ */
[data-testid="stPlotlyChart"] {
    border-radius: 14px !important;
    overflow: hidden !important;
    border: 1px solid rgba(255,255,255,0.05) !important;
    transition: box-shadow 0.3s ease !important;
}
[data-testid="stPlotlyChart"]:hover {
    box-shadow: 0 0 30px rgba(56,189,248,0.05), 0 8px 30px rgba(0,0,0,0.4) !important;
}

/* ══ SCROLLBAR ══ */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #03080f; }
::-webkit-scrollbar-thumb { background: rgba(56,189,248,0.2); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: rgba(56,189,248,0.35); }
footer { visibility: hidden !important; }

/* ══════════════════════════════════════════════════
   RESPONSIVE CSS — ALL SCREEN SIZES
   ══════════════════════════════════════════════════ */

/* ── Base overflow fixes ── */
*, *::before, *::after { box-sizing: border-box; }
.main .block-container {
    padding-left: 1rem !important;
    padding-right: 1rem !important;
    max-width: 100% !important;
}
img, video, iframe, canvas { max-width: 100% !important; }
[data-testid="stDataFrame"]  { max-width: 100% !important; overflow-x: auto !important; }
[data-testid="stPlotlyChart"] { max-width: 100% !important; }
[data-testid="stTabs"] [role="tablist"] { flex-wrap: wrap !important; }

/* ── Small desktop / large tablet  ≤ 1024px ── */
@media (max-width: 1024px) {
    .page-title                       { font-size: 2rem !important; }
    .kpi-value                        { font-size: 1.7rem !important; }
    [data-testid="stMetricValue"]     { font-size: 1.5rem !important; }
    [data-testid="stTabs"] button     { font-size: 11px !important; padding: 6px 12px !important; }
    .kpi-card                         { padding: 18px 20px !important; }
}

/* ── Tablet  ≤ 768px ── */
@media (max-width: 768px) {
    .page-title {
        font-size: 1.6rem !important;
        letter-spacing: -0.02em !important;
    }
    .page-subtitle {
        font-size: 10px !important;
        letter-spacing: 0.02em !important;
        margin-bottom: 16px !important;
    }
    .section-title {
        font-size: 0.9rem !important;
        margin: 20px 0 12px !important;
    }
    .kpi-card {
        padding: 14px 16px !important;
        border-radius: 12px !important;
        margin-bottom: 6px !important;
    }
    .kpi-value                        { font-size: 1.5rem !important; }
    .kpi-label                        { font-size: 9px !important; letter-spacing: 0.12em !important; }
    .kpi-sub                          { font-size: 10px !important; }
    [data-testid="stMetric"]          { padding: 14px 16px !important; }
    [data-testid="stMetricValue"]     { font-size: 1.4rem !important; }
    [data-testid="stMetricLabel"]     { font-size: 10px !important; }
    [data-testid="stTabs"] [role="tablist"] {
        padding: 3px !important;
        border-radius: 8px !important;
    }
    [data-testid="stTabs"] button {
        font-size: 10px !important;
        padding: 6px 10px !important;
        border-radius: 6px !important;
    }
    .insight-card                     { font-size: 12px !important; padding: 12px 16px !important; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        padding: 12px 14px !important;
        font-size: 14px !important;
        min-height: 44px !important;
    }
    .main .block-container {
        padding-left: 0.75rem !important;
        padding-right: 0.75rem !important;
        padding-top: 1rem !important;
    }
}

/* ── Mobile  ≤ 480px ── */
@media (max-width: 480px) {
    .page-title {
        font-size: 1.25rem !important;
        letter-spacing: -0.01em !important;
        line-height: 1.2 !important;
    }
    .page-subtitle {
        font-size: 9px !important;
        word-break: break-word !important;
    }
    .section-title {
        font-size: 0.82rem !important;
        margin: 14px 0 10px !important;
    }
    .kpi-card                         { padding: 12px 14px !important; border-radius: 10px !important; }
    .kpi-value                        { font-size: 1.25rem !important; }
    .kpi-label                        { font-size: 8px !important; letter-spacing: 0.08em !important; }
    .kpi-sub                          { font-size: 9px !important; }
    [data-testid="stMetric"]          { padding: 12px 14px !important; }
    [data-testid="stMetricValue"]     { font-size: 1.2rem !important; }
    [data-testid="stMetricLabel"]     { font-size: 9px !important; }
    [data-testid="stTabs"] button     { font-size: 9px !important; padding: 5px 8px !important; }
    .anomaly-badge, .normal-badge     { font-size: 10px !important; padding: 3px 8px !important; }
    .insight-card                     { font-size: 11px !important; padding: 10px 12px !important; line-height: 1.6 !important; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        padding: 13px 14px !important;
        font-size: 15px !important;
        min-height: 48px !important;
    }
    .main .block-container            { padding-left: 0.5rem !important; padding-right: 0.5rem !important; }
    /* iOS prevents auto-zoom on inputs (font-size must be ≥16px) */
    input, select, textarea           { font-size: 16px !important; }
    [data-testid="stNumberInput"] input,
    [data-testid="stTextInput"] input { font-size: 16px !important; }
}

/* ── Columns stack on small phones  ≤ 640px ── */
@media (max-width: 640px) {
    [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
        gap: 0.5rem !important;
    }
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        min-width: 0 !important;
        flex: 1 1 100% !important;
    }
    [data-testid="stSidebar"] {
        min-width: 260px !important;
        max-width: 85vw !important;
    }
}

/* ── Touch devices — disable hover lift animations ── */
@media (hover: none) and (pointer: coarse) {
    .kpi-card:hover                   { transform: none !important; box-shadow: none !important; }
    [data-testid="stMetric"]:hover    { transform: none !important; box-shadow: none !important; }
    [data-testid="stPlotlyChart"]:hover { box-shadow: none !important; }
}
</style>
""", unsafe_allow_html=True)

# ── UPGRADED Plotly theme ────────────────────────────────────────────────────
PLOTLY_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(3,8,15,0)",      # transparent — card background shows through
    plot_bgcolor="rgba(8,14,24,0.6)",
    font=dict(
        family="'Syne', 'DM Mono', system-ui, sans-serif",
        color="#a8c8e8",
        size=12,
    ),
)

ANOMALY_COLOR = "#f87171"   # soft red
NORMAL_COLOR  = "#34d399"   # emerald
PRIMARY_COLOR = "#38bdf8"   # sky blue
ACCENT_COLOR  = "#a78bfa"   # violet

CITY_PALETTE = [
    "#38bdf8", "#34d399", "#f87171", "#fbbf24",
    "#a78bfa", "#fb7185", "#22d3ee", "#4ade80",
]

# ── Data loading ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def get_data():
    return load_data()


# ── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar(df: pd.DataFrame) -> dict:
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding: 20px 0 12px'>
          <div style='font-size:32px; margin-bottom:6px'>⛈️</div>
          <div style='font-family:Syne,sans-serif; font-size:17px; font-weight:800;
                      background:linear-gradient(135deg,#38bdf8,#818cf8);
                      -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                      letter-spacing:0.05em'>
            WX · INTEL
          </div>
          <div style='font-family:"DM Mono",monospace; font-size:10px;
                      color:rgba(100,140,180,0.5); letter-spacing:0.15em; margin-top:4px'>
            PAKISTAN · 2000–2024
          </div>
        </div>
        <hr style='border-color:rgba(56,189,248,0.1); margin:8px 0 16px'>
        """, unsafe_allow_html=True)

        # Navigation
        page = st.radio(
            "Navigation",
            [
                "🏠 Home",
                "🔍 Data Exploration",
                "📊 Statistical Analysis",
                "🎲 Probability Analysis",
                "🤖 Prediction Engine",
                "🔮 Live Predictor",
                "⚠️ Anomaly Dashboard",
                "🗺️ Map Visualization",
                "💡 Insights",
            ],
            label_visibility="collapsed",
        )

        st.markdown("<hr style='border-color:#21262D; margin:12px 0'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:11px;font-weight:700;color:#6E7681;letter-spacing:1px;text-transform:uppercase;margin-bottom:8px'>Filters</div>", unsafe_allow_html=True)

        all_cities = sorted(df["city"].unique().tolist())
        cities = st.multiselect("Cities", all_cities, default=all_cities[:4])
        if not cities:
            cities = all_cities[:4]

        year_min, year_max = int(df["year"].min()), int(df["year"].max())
        year_range = st.slider("Year Range", year_min, year_max, (year_min, year_max))

        all_seasons = sorted(df["season"].unique().tolist())
        seasons = st.multiselect("Seasons", all_seasons, default=all_seasons)
        if not seasons:
            seasons = all_seasons

        all_rainfall = sorted(df["rainfall_intensity"].unique().tolist())
        rainfall_types = st.multiselect("Rainfall Intensity", all_rainfall, default=all_rainfall)
        if not rainfall_types:
            rainfall_types = all_rainfall

        all_wind = sorted(df["wind_category"].unique().tolist())
        wind_types = st.multiselect("Wind Category", all_wind, default=all_wind)
        if not wind_types:
            wind_types = all_wind

        temp_min = float(df["tavg"].min())
        temp_max = float(df["tavg"].max())
        temp_range = st.slider("Temperature Range (°C)", temp_min, temp_max, (temp_min, temp_max), step=0.5)

        st.markdown("<hr style='border-color:#21262D; margin:12px 0'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:11px;font-weight:700;color:#6E7681;letter-spacing:1px;text-transform:uppercase;margin-bottom:8px'>Display Options</div>", unsafe_allow_html=True)
        show_anomalies  = st.toggle("Show Anomalies",   value=True)
        show_ci         = st.toggle("Show Confidence Intervals", value=True)
        show_predictions = st.toggle("Show Predictions", value=True)

        st.markdown("<hr style='border-color:#21262D; margin:12px 0'>", unsafe_allow_html=True)
        stat_confidence = st.selectbox("CI Confidence Level", [0.90, 0.95, 0.99], index=1)
        z_threshold = st.number_input("Z-Score Threshold", min_value=1.0, max_value=4.0, value=2.0, step=0.5)

    return {
        "page": page,
        "cities": cities,
        "year_range": year_range,
        "seasons": seasons,
        "rainfall_types": rainfall_types,
        "wind_types": wind_types,
        "temp_range": temp_range,
        "show_anomalies": show_anomalies,
        "show_ci": show_ci,
        "show_predictions": show_predictions,
        "stat_confidence": stat_confidence,
        "z_threshold": z_threshold,
    }


# ── Plot helpers ─────────────────────────────────────────────────────────────

def apply_theme(fig, height=420):
    fig.update_layout(
        **PLOTLY_THEME,
        height=height,
        autosize=True,
        margin=dict(l=36, r=16, t=44, b=36),
        legend=dict(
            bgcolor="rgba(3,8,15,0.7)",
            bordercolor="rgba(56,189,248,0.15)",
            borderwidth=1,
            font=dict(size=11),
        ),
        xaxis=dict(
            gridcolor="rgba(56,189,248,0.06)",
            linecolor="rgba(56,189,248,0.1)",
            tickfont=dict(size=11, family="DM Mono, monospace"),
        ),
        yaxis=dict(
            gridcolor="rgba(56,189,248,0.06)",
            linecolor="rgba(56,189,248,0.1)",
            tickfont=dict(size=11, family="DM Mono, monospace"),
        ),
        hoverlabel=dict(
            bgcolor="rgba(3,8,15,0.9)",
            bordercolor="rgba(56,189,248,0.3)",
            font=dict(size=12, family="DM Mono, monospace", color="#d8e8f8"),
        ),
    )
    return fig



def kpi_card(label, value, sub=""):
    return f"""
    <div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-sub">{sub}</div>
    </div>"""


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PAGE 1 · HOME                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def page_home(df: pd.DataFrame, opts: dict):
    col_t, col_l = st.columns([3, 1])
    col_t, col_l = st.columns([3, 1])
    with col_t:
        st.markdown('<div class="page-title">🌦️ Pakistan Weather Anomaly Detection</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-subtitle">AI-POWERED ANALYSIS · 25 YEARS OF METEOROLOGICAL DATA · 2000–2024</div>', unsafe_allow_html=True)
    with col_l:
        lottie_weather = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_kljhtubl.json")
        if lottie_weather:
            st_lottie(lottie_weather, height=120, key="weather_lottie")

    fdf = filter_data(df, opts["cities"], opts["year_range"], opts["seasons"],
                      opts["rainfall_types"], opts["wind_types"], opts["temp_range"])

    # ── KPI Row ──
    avg_temp = fdf["tavg"].mean() if not fdf.empty else 0
    total_rain = fdf["prcp"].sum() if not fdf.empty else 0
    total_rows = len(fdf)

    # Quick anomaly estimate (|z| > 2)
    if not fdf.empty and "tavg" in fdf.columns:
        z = compute_zscore(fdf, "tavg")
        total_anomalies = int((z.abs() > 2).sum())
    else:
        total_anomalies = 0

    extreme_year = get_most_extreme_year(fdf) if not fdf.empty else "N/A"

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi_card("Avg Temperature", f"{avg_temp:.1f} °C", f"Based on {total_rows:,} records"), unsafe_allow_html=True)
    c2.markdown(kpi_card("Total Rainfall", f"{total_rain:,.0f} mm", f"Across all selected cities"), unsafe_allow_html=True)
    c3.markdown(kpi_card("Detected Anomalies", f"{total_anomalies:,}", f"{100*total_anomalies/max(total_rows,1):.1f}% of records"), unsafe_allow_html=True)
    c4.markdown(kpi_card("Most Extreme Year", str(extreme_year), "By anomaly frequency"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Two-panel overview ──
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown('<div class="section-title">Annual Mean Temperature Trends</div>', unsafe_allow_html=True)
        if not fdf.empty:
            yearly = fdf.groupby(["year", "city"])["tavg"].mean().reset_index()
            fig = px.line(yearly, x="year", y="tavg", color="city",
                          color_discrete_sequence=CITY_PALETTE,
                          labels={"tavg": "Avg Temp (°C)", "year": "Year", "city": "City"})
            fig.update_traces(line_width=2)
            apply_theme(fig, 360)
            st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-title">Records by Season</div>', unsafe_allow_html=True)
        if not fdf.empty:
            sc = fdf.groupby("season").size().reset_index(name="count")
            fig2 = px.pie(sc, values="count", names="season",
                          color_discrete_sequence=[PRIMARY_COLOR, NORMAL_COLOR, ACCENT_COLOR, ANOMALY_COLOR],
                          hole=0.55)
            fig2.update_traces(textinfo="percent+label", textfont_size=13)
            apply_theme(fig2, 360)
            st.plotly_chart(fig2, use_container_width=True)

    # ── Bottom bar charts ──
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-title">Monthly Rainfall Distribution</div>', unsafe_allow_html=True)
        if not fdf.empty:
            mr = fdf.groupby("month_name")["prcp"].mean().reset_index()
            month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
            mr["month_name"] = pd.Categorical(mr["month_name"], categories=month_order, ordered=True)
            mr = mr.sort_values("month_name")
            fig3 = px.bar(mr, x="month_name", y="prcp",
                          color="prcp", color_continuous_scale="Blues",
                          labels={"prcp": "Avg Precip (mm)", "month_name": "Month"})
            apply_theme(fig3, 300)
            fig3.update_coloraxes(showscale=False)
            st.plotly_chart(fig3, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-title">Wind Category Frequency</div>', unsafe_allow_html=True)
        if not fdf.empty:
            wc = fdf["wind_category"].value_counts().reset_index()
            wc.columns = ["category", "count"]
            fig4 = px.bar(wc, x="category", y="count",
                          color="count", color_continuous_scale="Teal",
                          labels={"count": "Days", "category": "Wind Category"})
            apply_theme(fig4, 300)
            fig4.update_coloraxes(showscale=False)
            st.plotly_chart(fig4, use_container_width=True)

    # ── Data preview ──
    with st.expander("📄 Raw Data Preview", expanded=False):
        st.dataframe(fdf.head(200).style.format(precision=2), use_container_width=True, height=300)

    st.markdown(f"""
    <div style='margin-top:24px; padding:16px; background:#1C2333; border-radius:8px; border:1px solid #30363D;'>
      <b style='color:#58A6FF'>📊 Dataset Summary</b><br>
      <span style='color:#8B949E; font-size:13px'>
        {total_rows:,} records · {fdf['city'].nunique()} cities · 
        Years: {int(fdf['year'].min()) if not fdf.empty else 'N/A'} – {int(fdf['year'].max()) if not fdf.empty else 'N/A'} ·
        Columns: {len(fdf.columns)}
      </span>
    </div>
    """, unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PAGE 2 · DATA EXPLORATION                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def page_exploration(df: pd.DataFrame, opts: dict):
    st.markdown('<div class="page-title">🔍 Data Exploration</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Comprehensive exploratory analysis of temperature, precipitation, humidity and more</div>', unsafe_allow_html=True)

    fdf = filter_data(df, opts["cities"], opts["year_range"], opts["seasons"],
                      opts["rainfall_types"], opts["wind_types"], opts["temp_range"])

    if fdf.empty:
        st.warning("No data matches the selected filters.")
        return

    tabs = st.tabs(["📈 Time Series", "📊 Distributions", "📦 Boxplots", "🗓️ Heatmap", "🔗 Scatter Plots", "🧊 3D Relationship"])

    # ── Tab 1: Time Series ──
    with tabs[0]:
        col_ts = st.selectbox("Select variable", ["tavg", "tmin", "tmax", "prcp", "humidity", "wind_speed"], key="ts_var")

        monthly = fdf.groupby(["year", "month", "city"])[col_ts].mean().reset_index()
        monthly["date"] = pd.to_datetime(monthly[["year", "month"]].assign(day=1))

        fig = go.Figure()
        for i, city in enumerate(fdf["city"].unique()):
            sub = monthly[monthly["city"] == city]
            fig.add_trace(go.Scatter(
                x=sub["date"], y=sub[col_ts], name=city,
                line=dict(width=1.8, color=CITY_PALETTE[i % len(CITY_PALETTE)]),
                mode="lines",
            ))

        fig.update_layout(
            title=f"Monthly Average {col_ts.upper()} by City",
            xaxis_title="Date", yaxis_title=col_ts,
            **PLOTLY_THEME, height=420, hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Annual trend
        annual = fdf.groupby(["year", "city"])[col_ts].mean().reset_index()
        fig2 = px.line(annual, x="year", y=col_ts, color="city",
                       color_discrete_sequence=CITY_PALETTE,
                       title=f"Annual Average {col_ts.upper()}")
        fig2.update_traces(mode="lines+markers", marker_size=5)
        apply_theme(fig2)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Tab 2: Distributions ──
    with tabs[1]:
        col_dist = st.selectbox("Select variable", ["tavg", "prcp", "humidity", "pressure", "wind_speed"], key="dist_var")
        ncols = 2
        cities_list = list(fdf["city"].unique())

        fig = make_subplots(rows=(len(cities_list)+1)//ncols, cols=ncols,
                            subplot_titles=[f"{c}" for c in cities_list])
        for i, city in enumerate(cities_list):
            r, c = divmod(i, ncols)
            vals = fdf[fdf["city"]==city][col_dist].dropna()
            fig.add_trace(
                go.Histogram(x=vals, name=city, nbinsx=40,
                             marker_color=CITY_PALETTE[i % len(CITY_PALETTE)],
                             showlegend=False),
                row=r+1, col=c+1,
            )
        fig.update_layout(title=f"Distribution of {col_dist.upper()} per City",
                          **PLOTLY_THEME, height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Overall histogram
        fig_all = px.histogram(fdf, x=col_dist, color="city",
                               nbins=60, barmode="overlay",
                               color_discrete_sequence=CITY_PALETTE,
                               title=f"Overall {col_dist.upper()} Distribution",
                               opacity=0.7)
        apply_theme(fig_all)
        st.plotly_chart(fig_all, use_container_width=True)

    # ── Tab 3: Boxplots ──
    with tabs[2]:
        col_box = st.selectbox("Select variable", ["tavg", "tmin", "tmax", "prcp", "humidity", "wind_speed"], key="box_var")
        group_by = st.radio("Group by", ["City", "Season", "Month"], horizontal=True, key="box_grp")
        grp_map = {"City": "city", "Season": "season", "Month": "month_name"}

        fig = px.box(fdf, x=grp_map[group_by], y=col_box, color="city" if group_by != "City" else None,
                     color_discrete_sequence=CITY_PALETTE,
                     title=f"{col_box.upper()} Boxplot by {group_by}",
                     points="outliers")
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 4: Heatmap ──
    with tabs[3]:
        col_heat = st.selectbox("Select variable", ["tavg", "prcp", "humidity", "sunshine_hours"], key="heat_var")

        pivot = fdf.groupby(["city", "month_name"])[col_heat].mean().reset_index()
        month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        pivot["month_name"] = pd.Categorical(pivot["month_name"], categories=month_order, ordered=True)
        heat_matrix = pivot.pivot(index="city", columns="month_name", values=col_heat)

        fig = px.imshow(
            heat_matrix,
            color_continuous_scale="RdYlBu_r" if col_heat == "tavg" else "Blues",
            title=f"Average {col_heat.upper()} — City × Month Heatmap",
            aspect="auto", text_auto=".1f",
        )
        fig.update_layout(**PLOTLY_THEME, height=380)
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 5: Scatter ──
    with tabs[4]:
        sc_pairs = [
            ("tavg", "humidity", "Temperature vs Humidity"),
            ("prcp", "cloud_cover", "Rainfall vs Cloud Cover"),
            ("wind_speed", "pressure", "Wind Speed vs Pressure"),
            ("sunshine_hours", "tavg", "Sunshine Hours vs Temperature"),
        ]
        choice = st.selectbox("Scatter pair", [p[2] for p in sc_pairs], key="sc_pair")
        xc, yc, title = next(p for p in sc_pairs if p[2] == choice)

        sample = fdf.sample(min(3000, len(fdf)), random_state=42)
        fig = px.scatter(sample, x=xc, y=yc, color="city",
                         color_discrete_sequence=CITY_PALETTE,
                         trendline="ols", trendline_scope="overall",
                         opacity=0.6, title=title,
                         labels={xc: xc, yc: yc})
        apply_theme(fig, 450)
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 6: 3D Relationship ──
    with tabs[5]:
        st.markdown('<div class="section-title">3D Atmospheric Space</div>', unsafe_allow_html=True)
        sample = fdf.sample(min(2000, len(fdf)), random_state=42)
        fig_3d = px.scatter_3d(sample, x="tavg", y="humidity", z="wind_speed",
                               color="city", size="prcp", opacity=0.8,
                               color_discrete_sequence=CITY_PALETTE,
                               title="3D Correlation: Temp vs Humidity vs Wind Speed",
                               labels={"tavg": "Temp (°C)", "humidity": "Humidity (%)", "wind_speed": "Wind (m/s)"})
        fig_3d.update_layout(
            scene=dict(
                xaxis=dict(backgroundcolor="#0E1117", gridcolor="#30363D", showbackground=True),
                yaxis=dict(backgroundcolor="#0E1117", gridcolor="#30363D", showbackground=True),
                zaxis=dict(backgroundcolor="#0E1117", gridcolor="#30363D", showbackground=True),
                bgcolor="#0E1117"
            ),
            **PLOTLY_THEME, height=600
        )
        st.plotly_chart(fig_3d, use_container_width=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PAGE 3 · STATISTICAL ANALYSIS                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def page_statistical(df: pd.DataFrame, opts: dict):
    st.markdown('<div class="page-title">📊 Statistical Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Deep statistical profiling: Descriptive stats, Confidence Intervals, Skewness, and Regional Variance</div>', unsafe_allow_html=True)

    fdf = filter_data(df, opts["cities"], opts["year_range"], opts["seasons"],
                      opts["rainfall_types"], opts["wind_types"], opts["temp_range"])

    if fdf.empty:
        st.warning("No data matches the selected filters.")
        return

    col_stat = st.selectbox("Variable to analyse", ["tavg", "tmin", "tmax", "prcp", "humidity", "wind_speed"], key="stat_col")
    ci_conf = opts["stat_confidence"]

    # ── Table 1: Descriptive Statistics ──
    st.markdown('<div class="section-title">Table 1: Descriptive Statistics by City</div>', unsafe_allow_html=True)
    desc_stats = get_descriptive_stats(fdf, col_stat)
    st.dataframe(desc_stats.style.format(precision=2), use_container_width=True, hide_index=True)

    # ── Table 2: 95% Confidence Intervals ──
    st.markdown(f'<div class="section-title">Table 2: {int(ci_conf*100)}% Confidence Intervals for {col_stat.upper()}</div>', unsafe_allow_html=True)
    ci_table = get_ci_table(fdf, col_stat, ci_conf)
    st.dataframe(ci_table.style.format(precision=3), use_container_width=True, hide_index=True)

    # ── Violin Plot: Regional Variance ──
    st.markdown('<div class="section-title">Variance and Regional Variation (Violin Plot)</div>', unsafe_allow_html=True)
    fig_v = px.violin(fdf, x="city", y=col_stat, color="city", box=True, points="all",
                      color_discrete_sequence=CITY_PALETTE,
                      title=f"Regional Distribution and Variance of {col_stat.upper()}")
    apply_theme(fig_v, 450)
    st.plotly_chart(fig_v, use_container_width=True)

    # ── Distribution with KDE & Skewness ──
    st.markdown('<div class="section-title">Probability Density and Skewness</div>', unsafe_allow_html=True)
    city_sel_dist = st.selectbox("Select city for Density Analysis", fdf["city"].unique(), key="stat_dist_city")
    city_series = fdf[fdf["city"] == city_sel_dist][col_stat].dropna()
    
    sk_info = compute_skew_kurt(city_series)
    
    col_l, col_r = st.columns([3, 1])
    with col_l:
        # Histogram + KDE
        import plotly.figure_factory as ff
        fig_kde = ff.create_distplot([city_series.values], [city_sel_dist], 
                                     show_hist=True, show_rug=False,
                                     colors=[PRIMARY_COLOR])
        fig_kde.update_layout(title=f"{city_sel_dist}: {col_stat.upper()} Density & KDE", **PLOTLY_THEME, height=400)
        st.plotly_chart(fig_kde, use_container_width=True)
    
    with col_r:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.metric("Skewness", sk_info["skew"])
        st.metric("Kurtosis", sk_info["kurt"])
        badge = '<span class="normal-badge">Normal Dist</span>' if abs(sk_info["skew"]) < 0.5 else '<span class="anomaly-badge">Skewed Dist</span>'
        st.markdown(badge, unsafe_allow_html=True)

    # ── CI Overlay on Monthly Averages ──
    st.markdown('<div class="section-title">Monthly Mean ± CI Overlay</div>', unsafe_allow_html=True)
    city_sel = st.selectbox("Select city for CI Trend", fdf["city"].unique(), key="ci_city")
    city_df = fdf[fdf["city"] == city_sel].copy()

    monthly = city_df.groupby("month")[col_stat].agg(["mean","std","count"]).reset_index()
    monthly["se"] = monthly["std"] / np.sqrt(monthly["count"].clip(1))
    from scipy.stats import t as t_dist
    t_val = t_dist.ppf((1 + ci_conf) / 2, df=monthly["count"].clip(1) - 1)
    monthly["ci_upper"] = monthly["mean"] + t_val * monthly["se"]
    monthly["ci_lower"] = monthly["mean"] - t_val * monthly["se"]

    fig = go.Figure()
    if opts["show_ci"]:
        fig.add_trace(go.Scatter(
            x=list(monthly["month"]) + list(monthly["month"])[::-1],
            y=list(monthly["ci_upper"]) + list(monthly["ci_lower"])[::-1],
            fill="toself", fillcolor="rgba(88,166,255,0.12)",
            line=dict(color="rgba(0,0,0,0)"),
            name=f"{int(ci_conf*100)}% CI", showlegend=True,
        ))

    fig.add_trace(go.Scatter(
        x=monthly["month"], y=monthly["mean"],
        mode="lines+markers", name="Monthly Mean",
        line=dict(color=PRIMARY_COLOR, width=2.5),
        marker=dict(size=7),
    ))
    
    fig.update_layout(
        title=f"{city_sel}: {col_stat.upper()} Monthly Mean ± {int(ci_conf*100)}% CI",
        xaxis=dict(title="Month", tickvals=list(range(1,13)),
                   ticktext=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]),
        yaxis_title=col_stat, **PLOTLY_THEME, height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PAGE 4 · PROBABILITY ANALYSIS                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def page_probability(df: pd.DataFrame, opts: dict):
    st.markdown('<div class="page-title">🎲 Probability Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Z-scores, p-values, distribution fitting, and statistically rare event detection</div>', unsafe_allow_html=True)

    fdf = filter_data(df, opts["cities"], opts["year_range"], opts["seasons"],
                      opts["rainfall_types"], opts["wind_types"], opts["temp_range"])

    if fdf.empty:
        st.warning("No data matches the selected filters.")
        return

    z_thresh = opts["z_threshold"]
    tabs = st.tabs(["🔔 Z-Scores", "📐 Distribution Fit", "🌧️ Poisson (Rain)", "⚡ Rare Events"])

    # ── Tab 1: Z-Scores ──
    with tabs[0]:
        col_z = st.selectbox("Variable", ["tavg", "prcp", "humidity", "wind_speed"], key="z_var")
        city_z = st.selectbox("City", fdf["city"].unique(), key="z_city")

        cdf = fdf[fdf["city"] == city_z].copy()
        cdf["zscore"] = compute_zscore(cdf, col_z)
        cdf["pvalue"] = compute_pvalue(cdf[col_z])
        cdf["z_flag"] = cdf["zscore"].abs() > z_thresh

        # Z-score time series
        fig = go.Figure()
        normal_pts = cdf[~cdf["z_flag"]]
        anom_pts   = cdf[cdf["z_flag"]]

        fig.add_trace(go.Scatter(
            x=normal_pts["date"], y=normal_pts["zscore"],
            mode="markers", name="Normal",
            marker=dict(color=NORMAL_COLOR, size=4, opacity=0.5),
        ))
        if opts["show_anomalies"] and not anom_pts.empty:
            fig.add_trace(go.Scatter(
                x=anom_pts["date"], y=anom_pts["zscore"],
                mode="markers", name=f"Anomaly (|z|>{z_thresh})",
                marker=dict(color=ANOMALY_COLOR, size=7, symbol="diamond"),
            ))
        fig.add_hline(y=z_thresh, line_dash="dash", line_color="#FFD700", annotation_text=f"+{z_thresh}σ")
        fig.add_hline(y=-z_thresh, line_dash="dash", line_color="#FFD700", annotation_text=f"-{z_thresh}σ")
        fig.add_hline(y=0, line_color="#30363D", line_width=1)

        fig.update_layout(title=f"{city_z}: {col_z.upper()} Z-Scores over Time",
                          xaxis_title="Date", yaxis_title="Z-Score",
                          **PLOTLY_THEME, height=420)
        st.plotly_chart(fig, use_container_width=True)

        # Z-score distribution
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(x=cdf["zscore"].dropna(), nbinsx=60,
                                    marker_color=PRIMARY_COLOR, opacity=0.7, name="Z-Score Dist"))
        x_range = np.linspace(-4, 4, 200)
        fig2.add_trace(go.Scatter(x=x_range, y=stats.norm.pdf(x_range) * len(cdf) * 0.15,
                                  mode="lines", name="Normal PDF",
                                  line=dict(color=ACCENT_COLOR, width=2.5, dash="dot")))
        fig2.add_vline(x=z_thresh, line_dash="dash", line_color=ANOMALY_COLOR)
        fig2.add_vline(x=-z_thresh, line_dash="dash", line_color=ANOMALY_COLOR)
        fig2.update_layout(title="Z-Score Distribution", **PLOTLY_THEME, height=320)
        st.plotly_chart(fig2, use_container_width=True)

        # Stats
        n_z = int(cdf["z_flag"].sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Z-Score Anomalies", n_z, f"{100*n_z/max(len(cdf),1):.1f}%")
        c2.metric("Max |Z-Score|", f"{cdf['zscore'].abs().max():.2f}")
        c3.metric("Records Analysed", len(cdf))

    # ── Tab 2: Distribution Fit ──
    with tabs[1]:
        col_fit = st.selectbox("Variable", ["tavg", "tmax", "tmin", "humidity", "pressure"], key="fit_var")
        city_fit = st.selectbox("City", fdf["city"].unique(), key="fit_city")

        vals = fdf[fdf["city"] == city_fit][col_fit].dropna()

        # Normality test
        norm_result = normality_test(vals)

        col_a, col_b = st.columns([3, 1])
        with col_a:
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=vals, nbinsx=50, histnorm="probability density",
                                       marker_color=PRIMARY_COLOR, opacity=0.65, name="Observed"))
            # Fitted normal
            mu, sigma = vals.mean(), vals.std()
            x_range = np.linspace(vals.min(), vals.max(), 300)
            fig.add_trace(go.Scatter(x=x_range, y=stats.norm.pdf(x_range, mu, sigma),
                                     mode="lines", name="Normal Fit",
                                     line=dict(color=NORMAL_COLOR, width=2.5)))
            # Fitted t
            nu, loc_t, scale_t = stats.t.fit(vals)
            fig.add_trace(go.Scatter(x=x_range, y=stats.t.pdf(x_range, nu, loc_t, scale_t),
                                     mode="lines", name="Student-t Fit",
                                     line=dict(color=ACCENT_COLOR, width=2.5, dash="dot")))
            fig.update_layout(title=f"{city_fit}: {col_fit.upper()} Distribution Fit",
                              **PLOTLY_THEME, height=380)
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.markdown('<div class="section-title">Normality Test</div>', unsafe_allow_html=True)
            if norm_result["p_value"] is not None:
                st.metric(norm_result["test"], f"p = {norm_result['p_value']:.4f}")
                badge = '<span class="normal-badge">✓ Normal</span>' if norm_result["is_normal"] else '<span class="anomaly-badge">✗ Non-Normal</span>'
                st.markdown(badge, unsafe_allow_html=True)
                st.markdown(f"<div style='margin-top:12px;font-size:13px;color:#8B949E'>μ = {mu:.2f}<br>σ = {sigma:.2f}</div>", unsafe_allow_html=True)

        # QQ Plot
        st.markdown('<div class="section-title">Q-Q Plot</div>', unsafe_allow_html=True)
        theoretical_q = stats.norm.ppf(np.linspace(0.01, 0.99, len(vals)))
        observed_q = np.sort(vals.values)
        fig_qq = go.Figure()
        fig_qq.add_trace(go.Scatter(x=theoretical_q, y=observed_q, mode="markers",
                                    marker=dict(color=PRIMARY_COLOR, size=4, opacity=0.5),
                                    name="Q-Q Points"))
        q_line = np.linspace(theoretical_q.min(), theoretical_q.max(), 2)
        fig_qq.add_trace(go.Scatter(x=q_line, y=mu + sigma * q_line,
                                    mode="lines", name="Reference Line",
                                    line=dict(color=ANOMALY_COLOR, width=2, dash="dash")))
        fig_qq.update_layout(title="Normal Q-Q Plot",
                             xaxis_title="Theoretical Quantiles",
                             yaxis_title="Sample Quantiles",
                             **PLOTLY_THEME, height=360)
        st.plotly_chart(fig_qq, use_container_width=True)

    # ── Tab 3: Poisson (Rainfall) ──
    with tabs[2]:
        city_p = st.selectbox("City", fdf["city"].unique(), key="pois_city")
        rainy = fdf[(fdf["city"] == city_p) & (fdf["prcp"] > 0)]["prcp"]

        pois_result = fit_poisson(rainy)
        lam = pois_result["lambda"]

        from scipy.stats import poisson as sp_poisson
        k_vals = np.arange(0, int(rainy.max()) + 1)
        pmf_vals = sp_poisson.pmf(k_vals, lam)

        fig = go.Figure()
        fig.add_trace(go.Histogram(x=rainy.round(), histnorm="probability",
                                   name="Observed", marker_color=PRIMARY_COLOR, opacity=0.65,
                                   nbinsx=40))
        fig.add_trace(go.Scatter(x=k_vals, y=pmf_vals, mode="lines+markers",
                                 name=f"Poisson(λ={lam:.2f})",
                                 line=dict(color=ANOMALY_COLOR, width=2.5),
                                 marker=dict(size=5)))
        fig.update_layout(title=f"{city_p}: Rainfall Poisson Distribution Fit",
                          xaxis_title="Daily Rainfall (mm)", yaxis_title="Probability",
                          **PLOTLY_THEME, height=400)
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Poisson λ", f"{lam:.3f} mm")
        c2.metric("KS Statistic", f"{pois_result['ks_stat']:.4f}")
        c3.metric("KS p-value", f"{pois_result['ks_p']:.4f}")

    # ── Tab 4: Extreme Precipitation ──
    with tabs[3]:
        st.markdown('<div class="section-title">Extreme Precipitation Skewness</div>', unsafe_allow_html=True)
        city_ext = st.selectbox("Select City for Extremes", fdf["city"].unique(), key="ext_city")
        rain_data = fdf[(fdf["city"] == city_ext) & (fdf["prcp"] > 0)]["prcp"]
        
        if not rain_data.empty:
            ext_skew = compute_skew_kurt(rain_data)
            
            col1, col2 = st.columns([3, 1])
            with col1:
                fig_ext = px.histogram(rain_data, nbins=50, title=f"{city_ext}: Rainy Day Distribution (Tail Analysis)",
                                      color_discrete_sequence=[ANOMALY_COLOR], opacity=0.7)
                apply_theme(fig_ext, 360)
                st.plotly_chart(fig_ext, use_container_width=True)
            
            with col2:
                st.markdown("<br><br>", unsafe_allow_html=True)
                st.metric("Rain Skewness", ext_skew["skew"])
                st.info("High positive skewness (>1) indicates a long tail of extreme rainfall events.")

        # ── Rare Events Table ──
        st.markdown('<div class="section-title">Statistically Rare Events (p < α)</div>', unsafe_allow_html=True)
        col_rare = st.selectbox("Variable", ["tavg", "prcp", "humidity", "wind_speed"], key="rare_var")
        alpha = st.slider("Significance Level (α)", 0.01, 0.10, 0.05, 0.01, key="alpha_sl")

        p_flags = flag_pvalue_anomalies(fdf[col_rare], alpha=alpha)
        fdf["p_flag"] = p_flags
        fdf["pvalue"] = compute_pvalue(fdf[col_rare])

        rare_events = fdf[fdf["p_flag"]].sort_values("pvalue").head(200)

        fig = px.scatter(rare_events.sample(min(500, len(rare_events)), random_state=42),
                         x="date", y=col_rare, color="city",
                         color_discrete_sequence=CITY_PALETTE,
                         size="pvalue",
                         title=f"Rare Events Timeline",
                         labels={col_rare: col_rare, "date": "Date"})
        apply_theme(fig, 400)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("View Rare Events Table", expanded=False):
            st.dataframe(rare_events[["date","city","season",col_rare,"pvalue"]].head(100)
                         .style.format({"pvalue": "{:.6f}", col_rare: "{:.2f}"}),
                         use_container_width=True, hide_index=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PAGE 5 · PREDICTION ENGINE                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def page_prediction(df: pd.DataFrame, opts: dict):
    st.markdown('<div class="page-title">🤖 Prediction Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Machine learning regression models for temperature and rainfall forecasting</div>', unsafe_allow_html=True)

    fdf = filter_data(df, opts["cities"], opts["year_range"], opts["seasons"],
                      opts["rainfall_types"], opts["wind_types"], opts["temp_range"])

    if fdf.empty or len(fdf) < 100:
        st.warning("Insufficient data for model training. Please expand your filters.")
        return

    col1, col2 = st.columns(2)
    target = col1.selectbox("Prediction Target", ["tavg", "prcp", "tmax", "humidity"], key="pred_target")
    model_type = col2.selectbox("Model", ["Random Forest", "Gradient Boosting", "Linear Regression", "Ridge Regression"], key="pred_model")

    if st.button("🚀 Train Model", type="primary"):
        with st.spinner(f"Training {model_type}..."):
            result = train_model(fdf, target=target, model_type=model_type)

        if "error" in result:
            st.error(result["error"])
            return

        st.session_state["model_result"] = result

    if "model_result" not in st.session_state:
        st.info("👆 Configure and train a model to see predictions.")
        return

    result = st.session_state["model_result"]
    rdf = result["result_df"]
    metrics = result["metrics"]

    # ── Metrics ──
    st.markdown('<div class="section-title">Model Performance</div>', unsafe_allow_html=True)
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("RMSE", metrics["RMSE"])
    mc2.metric("MAE", metrics["MAE"])
    mc3.metric("R²", metrics["R²"])
    mc4.metric("Train / Test", f"{result['n_train']} / {result['n_test']}")

    # ── Actual vs Predicted ──
    if opts["show_predictions"]:
        st.markdown('<div class="section-title">Actual vs Predicted</div>', unsafe_allow_html=True)
        sample = rdf.sample(min(2000, len(rdf)), random_state=42).sort_values("date")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sample["date"], y=sample[target],
                                 mode="lines", name="Actual",
                                 line=dict(color=NORMAL_COLOR, width=1.5)))
        fig.add_trace(go.Scatter(x=sample["date"], y=sample["predicted"],
                                 mode="lines", name="Predicted",
                                 line=dict(color=PRIMARY_COLOR, width=1.5, dash="dot")))
        fig.update_layout(title=f"Actual vs Predicted — {target.upper()}",
                          xaxis_title="Date", yaxis_title=target,
                          **PLOTLY_THEME, height=420, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        # Scatter actual vs predicted
        fig2 = px.scatter(rdf.sample(min(2000, len(rdf)), random_state=42),
                          x=target, y="predicted", color="city",
                          color_discrete_sequence=CITY_PALETTE,
                          trendline="ols", opacity=0.5,
                          title="Predicted vs Actual (Scatter)")
        # Add perfect prediction line
        mn, mx = rdf[target].min(), rdf[target].max()
        fig2.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx],
                                  mode="lines", name="Perfect Fit",
                                  line=dict(color=ANOMALY_COLOR, dash="dash", width=2)))
        apply_theme(fig2, 420)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Residuals ──
    st.markdown('<div class="section-title">Residual Analysis</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        # Residual distribution
        fig3 = go.Figure()
        fig3.add_trace(go.Histogram(x=rdf["residual"], nbinsx=60,
                                    marker_color=PRIMARY_COLOR, opacity=0.7, name="Residuals"))
        x_r = np.linspace(rdf["residual"].min(), rdf["residual"].max(), 200)
        mu_r, sig_r = rdf["residual"].mean(), rdf["residual"].std()
        fig3.add_trace(go.Scatter(x=x_r, y=stats.norm.pdf(x_r, mu_r, sig_r) * len(rdf) * (x_r[1]-x_r[0]) * 10,
                                  mode="lines", name="Normal Fit",
                                  line=dict(color=NORMAL_COLOR, width=2, dash="dot")))
        fig3.update_layout(title="Residual Distribution", **PLOTLY_THEME, height=320)
        st.plotly_chart(fig3, use_container_width=True)

    with col_b:
        # Residuals over time
        sample_r = rdf.sample(min(2000, len(rdf)), random_state=42).sort_values("date")
        high_res = sample_r[sample_r["residual_anomaly"]]
        low_res  = sample_r[~sample_r["residual_anomaly"]]

        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=low_res["date"], y=low_res["residual"],
                                  mode="markers", name="Normal Residual",
                                  marker=dict(color=NORMAL_COLOR, size=4, opacity=0.4)))
        if opts["show_anomalies"] and not high_res.empty:
            fig4.add_trace(go.Scatter(x=high_res["date"], y=high_res["residual"],
                                      mode="markers", name="High Residual",
                                      marker=dict(color=ANOMALY_COLOR, size=7, symbol="x")))
        fig4.add_hline(y=result["residual_threshold"], line_dash="dash", line_color=ANOMALY_COLOR)
        fig4.add_hline(y=-result["residual_threshold"], line_dash="dash", line_color=ANOMALY_COLOR)
        fig4.update_layout(title="Residuals over Time", **PLOTLY_THEME, height=320)
        st.plotly_chart(fig4, use_container_width=True)

    # ── Feature Importance ──
    if result.get("feature_importance"):
        st.markdown('<div class="section-title">Feature Importance</div>', unsafe_allow_html=True)
        fi = pd.DataFrame(list(result["feature_importance"].items()), columns=["Feature","Importance"])
        fi = fi.sort_values("Importance", ascending=True)
        fig5 = px.bar(fi, x="Importance", y="Feature", orientation="h",
                      color="Importance", color_continuous_scale="Blues",
                      title="Feature Importance")
        apply_theme(fig5, 350)
        fig5.update_coloraxes(showscale=False)
        st.plotly_chart(fig5, use_container_width=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PAGE 6 · ANOMALY DASHBOARD                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def page_anomaly(df: pd.DataFrame, opts: dict):
    st.markdown('<div class="page-title">⚠️ Anomaly Detection Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Combined CI + Z-Score + Residual anomaly detection with interactive visualisations</div>', unsafe_allow_html=True)

    fdf = filter_data(df, opts["cities"], opts["year_range"], opts["seasons"],
                      opts["rainfall_types"], opts["wind_types"], opts["temp_range"])

    if fdf.empty:
        st.warning("No data matches the selected filters.")
        return

    col_anom = st.selectbox("Variable", ["tavg", "prcp", "humidity", "wind_speed"], key="anom_col")
    ci_conf  = opts["stat_confidence"]
    z_thresh = opts["z_threshold"]

    # Compute all flags
    with st.spinner("Computing anomalies..."):
        ci_flag = flag_ci_anomalies(fdf, col_anom, ci_conf)
        z_flag  = flag_zscore_anomalies(fdf, col_anom, z_thresh)

        # Use residuals if model has been trained
        res_col, res_thresh = None, None
        if "model_result" in st.session_state:
            mr = st.session_state["model_result"]
            if mr["target"] == col_anom and "residual" in mr["result_df"].columns:
                rdf_merge = mr["result_df"][["residual","residual_anomaly"]]
                fdf = fdf.join(rdf_merge, how="left")
                res_col    = "residual"
                res_thresh = mr["residual_threshold"]

        fdf["ci_anomaly"] = ci_flag.values
        fdf["z_anomaly"]  = z_flag.values
        fdf["zscore"]     = compute_zscore(fdf, col_anom).values

        if res_col and res_col in fdf.columns:
            fdf["is_anomaly"] = fdf["ci_anomaly"] | fdf["z_anomaly"] | fdf["residual_anomaly"]
        else:
            fdf["is_anomaly"] = fdf["ci_anomaly"] | fdf["z_anomaly"]

    total_anom = int(fdf["is_anomaly"].sum())
    pct_anom   = 100 * total_anom / max(len(fdf), 1)

    # ── KPI ──
    kc1, kc2, kc3, kc4 = st.columns(4)
    kc1.metric("Total Anomalies", f"{total_anom:,}", f"{pct_anom:.1f}%")
    kc2.metric("CI Anomalies",    f"{int(fdf['ci_anomaly'].sum()):,}")
    kc3.metric("Z-Score Anomalies", f"{int(fdf['z_anomaly'].sum()):,}")
    res_count = int(fdf["residual_anomaly"].sum()) if "residual_anomaly" in fdf.columns else 0
    kc4.metric("Residual Anomalies", f"{res_count:,}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Time series with anomaly markers ──
    st.markdown('<div class="section-title">Anomaly Time Series</div>', unsafe_allow_html=True)

    city_anom = st.selectbox("City", fdf["city"].unique(), key="anom_city")
    cadf = fdf[fdf["city"] == city_anom].copy()

    normal = cadf[~cadf["is_anomaly"]]
    anom   = cadf[cadf["is_anomaly"]]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=normal["date"], y=normal[col_anom],
                             mode="markers", name="Normal",
                             marker=dict(color=NORMAL_COLOR, size=3, opacity=0.5)))
    if opts["show_anomalies"] and not anom.empty:
        fig.add_trace(go.Scatter(x=anom["date"], y=anom[col_anom],
                                 mode="markers", name="🚨 Anomaly",
                                 marker=dict(color=ANOMALY_COLOR, size=8, symbol="circle-open",
                                             line=dict(width=2))))
    fig.update_layout(title=f"{city_anom}: {col_anom.upper()} — Anomaly Markers",
                      xaxis_title="Date", yaxis_title=col_anom,
                      **PLOTLY_THEME, height=420)
    st.plotly_chart(fig, use_container_width=True)

    # ── Flag breakdown stacked bar ──
    st.markdown('<div class="section-title">Anomaly Flag Breakdown by City</div>', unsafe_allow_html=True)
    breakdown = fdf.groupby("city").agg(
        CI=("ci_anomaly", "sum"),
        ZScore=("z_anomaly", "sum"),
    ).reset_index()

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=breakdown["city"], y=breakdown["CI"],
                          name="CI Anomalies", marker_color=ACCENT_COLOR))
    fig2.add_trace(go.Bar(x=breakdown["city"], y=breakdown["ZScore"],
                          name="Z-Score Anomalies", marker_color=ANOMALY_COLOR))
    fig2.update_layout(barmode="stack", title="Anomaly Count by City and Method",
                       **PLOTLY_THEME, height=360)
    st.plotly_chart(fig2, use_container_width=True)

    # ── Heatmap ──
    st.markdown('<div class="section-title">Anomaly Frequency Heatmap (City × Month)</div>', unsafe_allow_html=True)
    anom_heat = anomaly_frequency(fdf, "is_anomaly")
    if not anom_heat.empty:
        fig3 = px.imshow(anom_heat, color_continuous_scale="Reds", aspect="auto",
                         text_auto=True, title="Anomaly Count: City × Month")
        fig3.update_layout(**PLOTLY_THEME, height=350)
        st.plotly_chart(fig3, use_container_width=True)

    # ── Annual trend ──
    st.markdown('<div class="section-title">Anomaly Count per Year</div>', unsafe_allow_html=True)
    annual_anom = fdf.groupby("year")["is_anomaly"].sum().reset_index()
    fig4 = px.bar(annual_anom, x="year", y="is_anomaly",
                  color="is_anomaly", color_continuous_scale="Reds",
                  title="Total Anomalies per Year",
                  labels={"is_anomaly": "Anomaly Count", "year": "Year"})
    apply_theme(fig4, 340)
    fig4.update_coloraxes(showscale=False)
    st.plotly_chart(fig4, use_container_width=True)

    # ── Table ──
    st.markdown('<div class="section-title">Anomaly Records Table</div>', unsafe_allow_html=True)
    display_cols = ["date","city","season",col_anom,"zscore","ci_anomaly","z_anomaly","is_anomaly"]
    display_cols = [c for c in display_cols if c in fdf.columns]
    anom_table = fdf[fdf["is_anomaly"]][display_cols].sort_values("date", ascending=False)

    st.dataframe(
        anom_table.head(500).style.format({"zscore": "{:.2f}", col_anom: "{:.2f}"}),
        use_container_width=True, height=350, hide_index=True,
    )
    st.caption(f"Showing top 500 of {len(anom_table):,} anomaly records")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PAGE 7 · MAP VISUALIZATION                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def page_map(df: pd.DataFrame, opts: dict):
    st.markdown('<div class="page-title">🗺️ Map Visualization</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Geospatial view of weather anomalies and climate patterns across Pakistan</div>', unsafe_allow_html=True)

    fdf = filter_data(df, opts["cities"], opts["year_range"], opts["seasons"],
                      opts["rainfall_types"], opts["wind_types"], opts["temp_range"])

    if fdf.empty:
        st.warning("No data.")
        return

    # Compute city-level anomaly stats
    z = compute_zscore(fdf, "tavg")
    fdf["zscore"] = z.values
    fdf["z_flag"] = (z.abs() > opts["z_threshold"]).values

    city_stats = fdf.groupby("city").agg(
        latitude=("latitude", "mean"),
        longitude=("longitude", "mean"),
        avg_temp=("tavg", "mean"),
        avg_prcp=("prcp", "mean"),
        anomaly_count=("z_flag", "sum"),
        total_records=("z_flag", "count"),
        max_temp=("tmax", "max"),
        min_temp=("tmin", "min"),
    ).reset_index()
    city_stats["anomaly_pct"] = 100 * city_stats["anomaly_count"] / city_stats["total_records"].clip(1)

    map_metric = st.selectbox("Colour by", ["anomaly_pct", "avg_temp", "avg_prcp", "max_temp"], key="map_metric")
    metric_labels = {
        "anomaly_pct": "Anomaly %",
        "avg_temp": "Avg Temp (°C)",
        "avg_prcp": "Avg Precip (mm)",
        "max_temp": "Max Temp (°C)",
    }

    # Bubble map
    fig = px.scatter_mapbox(
        city_stats,
        lat="latitude", lon="longitude",
        size="anomaly_count",
        color=map_metric,
        color_continuous_scale="Reds" if "anomaly" in map_metric else "RdYlBu_r",
        hover_name="city",
        hover_data={
            "avg_temp": ":.1f",
            "avg_prcp": ":.1f",
            "anomaly_count": True,
            "anomaly_pct": ":.1f",
            "latitude": False,
            "longitude": False,
        },
        size_max=40,
        mapbox_style="carto-darkmatter",
        zoom=4.8,
        center={"lat": 30.0, "lon": 70.0},
        title=f"Pakistan Cities — {metric_labels[map_metric]}",
        labels={map_metric: metric_labels[map_metric]},
    )
    fig.update_layout(
        paper_bgcolor="#0E1117",
        font_color="#C9D1D9",
        height=560,
        margin=dict(l=0, r=0, t=48, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    # City comparison table
    st.markdown('<div class="section-title">City-Level Summary</div>', unsafe_allow_html=True)
    display = city_stats.copy()
    display.columns = ["City","Lat","Lon","Avg Temp °C","Avg Precip mm",
                        "Anomaly Count","Total Records","Max Temp °C","Min Temp °C","Anomaly %"]
    st.dataframe(display.drop(columns=["Lat","Lon"])
                 .sort_values("Anomaly %", ascending=False)
                 .style.format(precision=2)
                 .background_gradient(subset=["Anomaly %"], cmap="Reds"),
                 use_container_width=True, hide_index=True)

    # Choropleth-style bar
    st.markdown('<div class="section-title">City Anomaly Intensity</div>', unsafe_allow_html=True)
    fig2 = px.bar(
        city_stats.sort_values("anomaly_pct", ascending=True),
        x="anomaly_pct", y="city", orientation="h",
        color="anomaly_pct", color_continuous_scale="Reds",
        title="Anomaly Rate by City (%)",
        labels={"anomaly_pct": "Anomaly %", "city": "City"},
    )
    apply_theme(fig2, 360)
    fig2.update_coloraxes(showscale=False)
    st.plotly_chart(fig2, use_container_width=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PAGE 8 · INSIGHTS                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def page_insights(df: pd.DataFrame, opts: dict):
    st.markdown('<div class="page-title">💡 Automated Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">AI-generated observations from trend analysis, city comparisons, and seasonal patterns</div>', unsafe_allow_html=True)

    fdf = filter_data(df, opts["cities"], opts["year_range"], opts["seasons"],
                      opts["rainfall_types"], opts["wind_types"], opts["temp_range"])

    if fdf.empty:
        st.warning("No data.")
        return

    insights = []

    # ── Temperature trend ──
    yearly_temp = fdf.groupby("year")["tavg"].mean()
    if len(yearly_temp) > 3:
        slope, intercept, r, p, se = stats.linregress(yearly_temp.index, yearly_temp.values)
        trend_dir = "warming" if slope > 0 else "cooling"
        sig = "statistically significant" if p < 0.05 else "not statistically significant"
        insights.append(
            f"📈 <b>Temperature Trend:</b> Average temperature has been <b>{trend_dir}</b> at "
            f"<b>{abs(slope)*10:.2f}°C per decade</b> — this trend is <b>{sig}</b> (p={p:.4f})."
        )

    # ── Hottest / coldest city ──
    city_temp = fdf.groupby("city")["tavg"].mean().sort_values()
    if len(city_temp) >= 2:
        insights.append(
            f"🌡️ <b>City Temperature Range:</b> <b>{city_temp.index[-1]}</b> is the warmest city "
            f"(avg {city_temp.iloc[-1]:.1f}°C), while <b>{city_temp.index[0]}</b> is the coolest "
            f"(avg {city_temp.iloc[0]:.1f}°C)."
        )

    # ── Wettest city ──
    city_rain = fdf.groupby("city")["prcp"].mean().sort_values(ascending=False)
    if len(city_rain) >= 1:
        insights.append(
            f"🌧️ <b>Rainfall:</b> <b>{city_rain.index[0]}</b> receives the most precipitation "
            f"(avg {city_rain.iloc[0]:.2f} mm/day). "
            f"<b>{city_rain.index[-1]}</b> is the driest ({city_rain.iloc[-1]:.2f} mm/day)."
        )

    # ── Season analysis ──
    season_temp = fdf.groupby("season")["tavg"].mean().sort_values(ascending=False)
    season_rain = fdf.groupby("season")["prcp"].mean().sort_values(ascending=False)
    if len(season_temp) >= 2:
        insights.append(
            f"🍂 <b>Seasonal Patterns:</b> <b>{season_temp.index[0]}</b> is the hottest season "
            f"({season_temp.iloc[0]:.1f}°C avg), while <b>{season_rain.index[0]}</b> brings the "
            f"most rainfall ({season_rain.iloc[0]:.2f} mm/day)."
        )

    # ── Extreme days ──
    if "is_hot_day" in fdf.columns:
        hot_count  = int(fdf["is_hot_day"].sum())
        hot_city   = fdf.groupby("city")["is_hot_day"].sum().idxmax()
        insights.append(
            f"☀️ <b>Extreme Heat:</b> <b>{hot_count:,}</b> extreme heat days (tmax ≥ 40°C) recorded. "
            f"<b>{hot_city}</b> experiences the most extreme heat events."
        )

    # ── Humidity ──
    humid_season = fdf.groupby("season")["humidity"].mean().idxmax()
    humid_val    = fdf.groupby("season")["humidity"].mean().max()
    insights.append(
        f"💧 <b>Humidity:</b> <b>{humid_season}</b> is the most humid season "
        f"(avg {humid_val:.1f}%). This correlates with monsoon activity."
    )

    # ── Wind ──
    windy_city = fdf.groupby("city")["wind_speed"].mean().idxmax()
    calm_city  = fdf.groupby("city")["wind_speed"].mean().idxmin()
    insights.append(
        f"🌬️ <b>Wind Patterns:</b> <b>{windy_city}</b> is the windiest city on average, "
        f"while <b>{calm_city}</b> experiences the calmest conditions."
    )

    # ── Anomaly year ──
    z = compute_zscore(fdf, "tavg")
    fdf["z_flag"] = (z.abs() > opts["z_threshold"]).values
    extreme_year = int(fdf.groupby("year")["z_flag"].sum().idxmax())
    extreme_count = int(fdf.groupby("year")["z_flag"].sum().max())
    insights.append(
        f"⚡ <b>Most Anomalous Year:</b> <b>{extreme_year}</b> had the highest number of temperature "
        f"anomalies ({extreme_count:,} days with |z| > {opts['z_threshold']})."
    )

    # ── Cloud ──
    if "cloud_cover" in fdf.columns:
        cc_month = fdf.groupby("month_name")["cloud_cover"].mean().idxmax()
        insights.append(
            f"☁️ <b>Cloud Cover:</b> <b>{cc_month}</b> has the highest average cloud cover, "
            f"often associated with the monsoon or winter western disturbances."
        )

    # Render cards
    st.markdown('<div class="section-title">Key Findings</div>', unsafe_allow_html=True)
    for insight in insights:
        st.markdown(f'<div class="insight-card">{insight}</div>', unsafe_allow_html=True)

    # ── Correlation matrix ──
    st.markdown('<div class="section-title">Feature Correlation Matrix</div>', unsafe_allow_html=True)
    num_cols = [c for c in ["tavg","tmin","tmax","prcp","humidity","pressure",
                             "wind_speed","cloud_cover","sunshine_hours","temp_range"]
                if c in fdf.columns]
    corr = fdf[num_cols].corr()
    fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    text_auto=".2f", aspect="auto",
                    title="Pearson Correlation Matrix")
    fig.update_layout(**PLOTLY_THEME, height=480)
    st.plotly_chart(fig, use_container_width=True)

    # ── Seasonal radar ──
    st.markdown('<div class="section-title">Seasonal Climate Profile</div>', unsafe_allow_html=True)
    radar_vars = [c for c in ["tavg","prcp","humidity","wind_speed","cloud_cover"] if c in fdf.columns]
    season_profile = fdf.groupby("season")[radar_vars].mean().reset_index()

    # Normalise to [0,1]
    sp_norm = season_profile.copy()
    for v in radar_vars:
        mn, mx = sp_norm[v].min(), sp_norm[v].max()
        sp_norm[v] = (sp_norm[v] - mn) / max(mx - mn, 1e-6)

    fig2 = go.Figure()
    for i, row in sp_norm.iterrows():
        vals_r = [row[v] for v in radar_vars] + [row[radar_vars[0]]]
        labels = radar_vars + [radar_vars[0]]
        fig2.add_trace(go.Scatterpolar(
            r=vals_r, theta=labels, fill="toself",
            name=row["season"],
            line=dict(color=CITY_PALETTE[i % len(CITY_PALETTE)]),
        ))
    fig2.update_layout(
        polar=dict(bgcolor="#1C2333",
                   radialaxis=dict(visible=True, range=[0, 1], gridcolor="#30363D"),
                   angularaxis=dict(gridcolor="#30363D")),
        title="Normalised Seasonal Climate Profile",
        **PLOTLY_THEME, height=420,
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── Year-over-year comparison ──
    st.markdown('<div class="section-title">Year-over-Year Temperature Anomaly</div>', unsafe_allow_html=True)
    baseline = fdf.groupby("month")["tavg"].mean()
    fdf["temp_anomaly_yoy"] = fdf.apply(lambda r: r["tavg"] - baseline.get(r["month"], r["tavg"]), axis=1)
    yoy = fdf.groupby("year")["temp_anomaly_yoy"].mean().reset_index()

    fig3 = go.Figure()
    colors = [ANOMALY_COLOR if v > 0 else PRIMARY_COLOR for v in yoy["temp_anomaly_yoy"]]
    fig3.add_trace(go.Bar(x=yoy["year"], y=yoy["temp_anomaly_yoy"],
                          marker_color=colors, name="Temp Anomaly vs Baseline"))
    fig3.add_hline(y=0, line_color="#30363D", line_width=1.5)
    fig3.update_layout(title="Annual Mean Temperature Anomaly vs Monthly Baseline",
                       xaxis_title="Year", yaxis_title="Temperature Anomaly (°C)",
                       **PLOTLY_THEME, height=360)
    st.plotly_chart(fig3, use_container_width=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PAGE 9 · LIVE PREDICTOR                                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def page_live_predictor(df: pd.DataFrame, opts: dict):
    st.markdown('<div class="page-title">🔮 Live Next-Day Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Interactive forecasting: Input current conditions to predict tomorrow\'s temperature</div>', unsafe_allow_html=True)

    col_input, col_viz = st.columns([1, 1.2])

    with col_input:
        st.markdown('<div class="section-title">Current Conditions</div>', unsafe_allow_html=True)
        
        with st.container():
            st.markdown("""
            <style>
            .stForm {
                background: rgba(255,255,255,0.03) !important;
                backdrop-filter: blur(10px);
                border-radius: 15px;
                padding: 20px;
                border: 1px solid rgba(56,189,248,0.1) !important;
            }
            </style>
            """, unsafe_allow_html=True)
            
            with st.form("predictor_form"):
                city = st.selectbox("Select City", sorted(df["city"].unique()))
                
                # Fetch button outside the form usually, but we can do a trick or just use it as part of the page
                # For simplicity, we'll put an 'Auto-fill' checkbox or handle it via session state
                
                c1, c2 = st.columns(2)
                
                # Use session state for default values
                if f"live_{city}" in st.session_state:
                    live_data = st.session_state[f"live_{city}"]
                else:
                    live_data = {}

                tavg = c1.number_input("Today's Avg Temp (°C)", value=float(live_data.get("tavg", 25.0)), step=0.5)
                humid = c2.number_input("Humidity (%)", value=float(live_data.get("humidity", 50.0)), min_value=0.0, max_value=100.0)
                
                c3, c4 = st.columns(2)
                press = c3.number_input("Pressure (hPa)", value=float(live_data.get("pressure", 1010.0)), step=1.0)
                wind = c4.number_input("Wind Speed (m/s)", value=float(live_data.get("wind_speed", 5.0)), step=0.5)
                
                c5, c6 = st.columns(2)
                cloud = c5.slider("Cloud Cover (%)", 0, 100, int(live_data.get("cloud_cover", 20)))
                sun = c6.slider("Sunshine Hours", 0.0, 14.0, 8.0)
                
                model_choice = st.selectbox("Forecast Model", ["Random Forest", "Gradient Boosting", "Ridge Regression"])
                
                submit = st.form_submit_button("✨ Generate Forecast", use_container_width=True)

        # Separate button for live fetching
        if st.button("🌍 Fetch Real-Time Data for " + city, use_container_width=True):
            coords = get_city_coords(df)
            row = coords[coords["city"] == city]
            if not row.empty:
                lat, lon = row.iloc[0]["latitude"], row.iloc[0]["longitude"]
                with st.spinner(f"Connecting to Open-Meteo for {city}..."):
                    live = fetch_realtime_weather(lat, lon)
                    if "error" in live:
                        st.error(live["error"])
                    else:
                        st.session_state[f"live_{city}"] = live
                        st.success(f"Real-time data fetched for {city}!")
                        st.rerun()

    if submit:
        # Prepare input dict
        from datetime import datetime
        today = datetime.now()
        input_data = {
            "month": today.month,
            "day_of_year": today.timetuple().tm_yday,
            "tavg": tavg,
            "humidity": humid,
            "pressure": press,
            "wind_speed": wind,
            "cloud_cover": cloud,
            "sunshine_hours": sun
        }
        
        with st.spinner("AI is analyzing patterns..."):
            # Filter data for the specific city to make the model more localized if possible
            city_df = df[df["city"] == city]
            if len(city_df) < 50:
                city_df = df # Fallback to all data if city data is sparse
                
            model_res = train_next_day_model(city_df, target="tavg", model_type=model_choice)
            
            if "error" in model_res:
                st.error(f"Prediction Error: {model_res['error']}")
            else:
                pred_res = predict_next_day(model_res, input_data)
                
                with col_viz:
                    st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)
                    
                    # Big metric display
                    st.markdown(f"""
                    <div style='background:rgba(56,189,248,0.05); border:1px solid #38bdf8; border-radius:20px; padding:30px; text-align:center; margin-bottom:20px'>
                        <div style='font-size:14px; color:#8B949E; text-transform:uppercase; letter-spacing:2px'>Predicted Temperature Tomorrow</div>
                        <div style='font-size:64px; font-weight:800; color:#38bdf8; margin:10px 0'>{pred_res['prediction']}°C</div>
                        <div style='font-size:16px; color:#34d399'>Expected Range: {pred_res['ci_lower']}°C – {pred_res['ci_upper']}°C</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Model confidence plot
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = model_res["metrics"]["R²"] * 100,
                        title = {'text': "Model Confidence (R² %)"},
                        gauge = {
                            'axis': {'range': [0, 100], 'tickcolor': "#38bdf8"},
                            'bar': {'color': "#38bdf8"},
                            'bgcolor': "rgba(0,0,0,0)",
                            'borderwidth': 2,
                            'bordercolor': "rgba(56,189,248,0.2)",
                            'steps': [
                                {'range': [0, 50], 'color': 'rgba(248,113,113,0.1)'},
                                {'range': [50, 80], 'color': 'rgba(251,191,36,0.1)'},
                                {'range': [80, 100], 'color': 'rgba(52,211,153,0.1)'}
                            ],
                        }
                    ))
                    apply_theme(fig, 300)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.info(f"💡 Based on {model_res['n_train'] + model_res['n_test']} historical records for {city if len(df[df['city']==city]) >= 50 else 'all cities'}.")

                    # ─── ANOMALY CLASSIFICATION ───────────────────────────────────────────
                    from datetime import datetime
                    curr_month = datetime.now().month
                    anomaly_result = classify_predicted_anomaly(
                        pred_res["prediction"], city_df, curr_month
                    )

                    verdict = anomaly_result["verdict"]
                    severity = anomaly_result["severity"]

                    # Color map
                    verdict_config = {
                        "Normal":           {"color": "#34d399", "bg": "rgba(52,211,153,0.08)", "border": "#34d399", "icon": "✅"},
                        "Marginal Anomaly": {"color": "#fbbf24", "bg": "rgba(251,191,36,0.08)",  "border": "#fbbf24", "icon": "⚠️"},
                        "Warm Anomaly":     {"color": "#fb923c", "bg": "rgba(251,146,60,0.08)",  "border": "#fb923c", "icon": "🔥"},
                        "Cold Anomaly":     {"color": "#60a5fa", "bg": "rgba(96,165,250,0.08)",  "border": "#60a5fa", "icon": "💧"},
                        "Extreme Heat":     {"color": "#ef4444", "bg": "rgba(239,68,68,0.12)",   "border": "#ef4444", "icon": "🔴"},
                        "Extreme Cold":     {"color": "#a78bfa", "bg": "rgba(167,139,250,0.12)", "border": "#a78bfa", "icon": "🟣"},
                    }
                    cfg = verdict_config.get(verdict, verdict_config["Normal"])

                    st.markdown(f"""
                    <div style='background:{cfg['bg']}; border:2px solid {cfg['border']};
                                border-radius:16px; padding:24px; margin:16px 0;'>
                        <div style='display:flex; align-items:center; gap:12px; margin-bottom:12px;'>
                            <span style='font-size:32px'>{cfg['icon']}</span>
                            <div>
                                <div style='font-size:11px; color:#8B949E; text-transform:uppercase; letter-spacing:2px;'>Anomaly Verdict</div>
                                <div style='font-size:24px; font-weight:800; color:{cfg['color']};'>{verdict}</div>
                            </div>
                            <div style='margin-left:auto; text-align:right;'>
                                <div style='font-size:11px; color:#8B949E;'>Severity Score</div>
                                <div style='font-size:28px; font-weight:800; color:{cfg['color']};'>{severity}<span style='font-size:14px'>/100</span></div>
                            </div>
                        </div>
                        <div style='display:grid; grid-template-columns:1fr 1fr 1fr; gap:12px; margin-bottom:16px;'>
                            <div style='background:rgba(255,255,255,0.03); border-radius:10px; padding:12px; text-align:center;'>
                                <div style='font-size:11px; color:#8B949E;'>Z-Score</div>
                                <div style='font-size:20px; font-weight:700; color:{cfg['color']};'>{anomaly_result['zscore']:+.2f}σ</div>
                            </div>
                            <div style='background:rgba(255,255,255,0.03); border-radius:10px; padding:12px; text-align:center;'>
                                <div style='font-size:11px; color:#8B949E;'>p-value</div>
                                <div style='font-size:20px; font-weight:700; color:{cfg['color']};'>{anomaly_result['pvalue']:.4f}</div>
                            </div>
                            <div style='background:rgba(255,255,255,0.03); border-radius:10px; padding:12px; text-align:center;'>
                                <div style='font-size:11px; color:#8B949E;'>Historical Mean</div>
                                <div style='font-size:20px; font-weight:700; color:#C9D1D9;'>{anomaly_result['hist_mean']}°C</div>
                            </div>
                        </div>
                        <div style='background:rgba(255,255,255,0.02); border-radius:10px; padding:14px;
                                    font-size:13px; color:#8B949E; line-height:1.6;'>
                            {anomaly_result['explanation']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Historical distribution with predicted value overlay
                    month_hist = city_df[city_df["month"] == curr_month]["tavg"].dropna()
                    if len(month_hist) > 0:
                        st.markdown('<div class="section-title">Predicted vs Historical Distribution</div>', unsafe_allow_html=True)
                        fig_hist = go.Figure()
                        fig_hist.add_trace(go.Histogram(
                            x=month_hist, nbinsx=30, name="Historical",
                            marker_color=PRIMARY_COLOR, opacity=0.6, histnorm="probability density"
                        ))
                        # Normal fit
                        import numpy as np
                        x_range = np.linspace(month_hist.min(), month_hist.max(), 200)
                        mu, sigma = month_hist.mean(), month_hist.std()
                        from scipy import stats as sp_stats
                        fig_hist.add_trace(go.Scatter(
                            x=x_range, y=sp_stats.norm.pdf(x_range, mu, sigma),
                            mode="lines", name="Normal Fit",
                            line=dict(color="#38bdf8", width=2, dash="dot")
                        ))
                        # CI band
                        fig_hist.add_vrect(
                            x0=anomaly_result["ci_lower"], x1=anomaly_result["ci_upper"],
                            fillcolor="rgba(52,211,153,0.08)", line_width=0,
                            annotation_text="95% CI", annotation_position="top left"
                        )
                        # Predicted value
                        fig_hist.add_vline(
                            x=pred_res["prediction"], line_color=cfg["color"],
                            line_width=3, line_dash="solid",
                            annotation_text=f"Predicted: {pred_res['prediction']}°C",
                            annotation_font_color=cfg["color"]
                        )
                        # Historical mean
                        fig_hist.add_vline(
                            x=anomaly_result["hist_mean"], line_color="#8B949E",
                            line_width=1.5, line_dash="dash",
                            annotation_text=f"Hist. Mean: {anomaly_result['hist_mean']}°C",
                            annotation_font_color="#8B949E"
                        )
                        import calendar
                        month_name = calendar.month_abbr[curr_month]
                        fig_hist.update_layout(
                            title=f"{city}: {month_name} Temperature Distribution vs Prediction",
                            xaxis_title="Temperature (°C)", yaxis_title="Density",
                            **PLOTLY_THEME, height=320
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)

                    # Feature Importance
                    if model_res.get("feature_importance"):
                        st.markdown('<div class="section-title">Feature Importance</div>', unsafe_allow_html=True)
                        imp_df = pd.DataFrame({
                            "Feature": list(model_res["feature_importance"].keys()),
                            "Importance": list(model_res["feature_importance"].values())
                        }).sort_values("Importance", ascending=False)
                        fig_imp = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                                         color="Importance", color_continuous_scale="Blues")
                        apply_theme(fig_imp, 300)
                        fig_imp.update_coloraxes(showscale=False)
                        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        with col_viz:
            st.markdown("<br><br><br>", unsafe_allow_html=True)
            lottie_ai = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_7wwmupbm.json")
            if lottie_ai:
                st_lottie(lottie_ai, height=300, key="ai_lottie")
            st.markdown("<div style='text-align:center; color:#8B949E'>Fill in the current weather data and click 'Generate Forecast' to see AI predictions.</div>", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MAIN                                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main():
    with st.spinner("Loading Pakistan weather data..."):
        df = get_data()
    
    # Data source indicator
    source = df.get("_data_source", ["unknown"]).iloc[0]
    if source == "real":
        st.sidebar.success("✅ Real Dataset Loaded")
    else:
        st.sidebar.warning("⚠️ Using Synthetic Data (CSV not found)")

    opts = render_sidebar(df)
    page = opts["page"]

    if   page == "🏠 Home":                 page_home(df, opts)
    elif page == "🔍 Data Exploration":     page_exploration(df, opts)
    elif page == "📊 Statistical Analysis": page_statistical(df, opts)
    elif page == "🎲 Probability Analysis": page_probability(df, opts)
    elif page == "🤖 Prediction Engine":    page_prediction(df, opts)
    elif page == "🔮 Live Predictor":       page_live_predictor(df, opts)
    elif page == "⚠️ Anomaly Dashboard":    page_anomaly(df, opts)
    elif page == "🗺️ Map Visualization":    page_map(df, opts)
    elif page == "💡 Insights":             page_insights(df, opts)

    # Footer
    st.markdown("""
    <div style='text-align:center; padding:32px 0 8px; color:#3D444D; font-size:12px;'>
        🌦️ Pakistan Weather Anomaly Detection Dashboard &nbsp;·&nbsp;
        Built with Streamlit, Plotly & scikit-learn &nbsp;·&nbsp;
        Data: 2000–2024
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
