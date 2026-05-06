# 🌦️ Pakistan Weather Anomaly Detection Dashboard

> A production-grade meteorological intelligence platform — 25 years of real weather data (2000–2024), deep statistical analysis, ML-powered forecasting, and real-time anomaly detection across Pakistan's major cities.

[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?logo=streamlit)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikit-learn)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📸 Overview

This dashboard provides an end-to-end weather intelligence system for **6 major cities in Pakistan**, offering:

- **Exploratory Data Analysis (EDA)** across 31,779 real historical records
- **Statistical Anomaly Detection** using Confidence Intervals, Z-scores, and p-values
- **ML-powered next-day temperature forecasting** with anomaly classification
- **Real-time weather data** via the Open-Meteo API (no API key required)
- **Interactive probability analysis** including distribution fitting and Poisson modeling

---

## 🏙️ Dataset — Cities Covered

| City | Records | Avg Temp | Date Range |
|------|---------|----------|------------|
| **Islamabad** | 9,132 | 21.1°C | 2000–2024 |
| **Karachi** | 5,844 | 26.1°C | 2000–2024 |
| **Peshawar** | 5,114 | 22.1°C | 2000–2024 |
| **Gilgit** | 4,383 | 9.1°C | 2000–2024 |
| **Quetta** | 4,018 | 17.7°C | 2000–2024 |
| **Lahore** | 3,288 | 23.7°C | 2000–2024 |

**Total: 31,779 records · 26 features · Zero missing values**

---

## 🧭 Application Pages

| Page | Description |
|------|-------------|
| 🏠 **Home** | KPI overview, annual temperature trends, rainfall & wind summaries |
| 🔍 **Data Exploration** | Time series, histograms, boxplots, monthly heatmaps, 3D scatter plots |
| 📊 **Statistical Analysis** | Table 1 (Descriptive Stats), Table 2 (95% CI), Violin plots, KDE with Skewness/Kurtosis |
| 🎲 **Probability Analysis** | Z-scores, p-values, Normality tests, Distribution fitting, Poisson rainfall model, Extreme precipitation skewness |
| 🤖 **Prediction Engine** | Random Forest / Gradient Boosting / Ridge Regression with RMSE, MAE, R² metrics |
| 🔮 **Live Predictor** | Input today's conditions → get tomorrow's temperature + **anomaly verdict** |
| ⚠️ **Anomaly Dashboard** | Combined CI + Z-score + Residual anomaly detection with severity scoring |
| 🗺️ **Map Visualization** | Bubble map coloured by anomaly intensity, city-level comparison |
| 💡 **Insights** | Auto-generated statistical narratives, seasonal radar, correlation matrix |

---

## 🔮 Live Predictor — Anomaly Intelligence

The flagship feature: input current weather conditions (or fetch them live) and get:

1. **Predicted temperature for tomorrow** (±95% Confidence Interval)
2. **Anomaly Verdict** — color-coded classification:
   - ✅ Normal — within historical range
   - ⚠️ Marginal Anomaly — outside CI (p < 0.05)
   - 🔥 Warm Anomaly — z ≥ +2σ
   - 💧 Cold Anomaly — z ≤ −2σ
   - 🔴 Extreme Heat — z ≥ +3σ
   - 🟣 Extreme Cold — z ≤ −3σ
3. **Z-Score, p-value, Historical CI** — full statistical breakdown
4. **Distribution chart** — prediction overlaid on the historical density curve
5. **Feature Importance** — which inputs drove the prediction

### 🌍 Real-Time Data
Click **"Fetch Real-Time Data"** for any city to auto-populate the form with live conditions from the [Open-Meteo API](https://open-meteo.com/) (free, no API key needed).

---

## 📐 Anomaly Detection Methodology

```
Combined Anomaly = CI_anomaly  OR  |Z-score| > threshold  OR  high_residual
```

| Method | Logic |
|--------|-------|
| **Confidence Interval** | Value outside 95% CI for that city × month group |
| **Z-score** | \|z\| > 2.0 (configurable in sidebar, default 2.0σ) |
| **Residual** | \|residual\| > 2 × RMSE from the trained ML model |
| **Severity Score** | 0–100: `min(100, |z| / 3.0 × 100)` |

Statistical tests used: **Shapiro-Wilk** (normality), **Kolmogorov-Smirnov** (distribution fit), **Student's t-interval** (CI), **Poisson PMF** (rainfall).

---

## 🤖 Machine Learning Architecture

### Next-Day Temperature Forecasting
- **Models**: Random Forest, Gradient Boosting (Huber loss), Ridge Regression
- **Feature Engineering**:
  - Cyclical temporal encoding: `month_sin/cos`, `day_sin/cos`
  - Today's temperature as primary anchor (`tavg`)
  - Humidity, pressure, wind speed, cloud cover, sunshine hours
- **Physical Consistency Constraint**: Prediction clipped to ±15% of today's baseline (unless extreme pressure drop detected)
- **Pipeline**: `StandardScaler` fit only on training data — no leakage

---

## 📁 Project Structure

```
weather_dashboard/
├── app.py                          # Main Streamlit app (9 pages, 1900+ lines)
├── requirements.txt                # Pinned dependencies
├── pakistan_weather_2000_2024.csv  # Real dataset (31,779 records)
├── Statistical_Analysis.ipynb      # Jupyter notebook (methodology reference)
└── utils/
    ├── data_loader.py              # CSV loading, preprocessing, filtering
    ├── stats.py                    # CI, Z-score, p-values, skewness, anomaly classifier
    ├── models.py                   # ML training, next-day forecasting, physical constraints
    └── weather_api.py              # Open-Meteo real-time data fetcher
```

---

## ⚙️ Installation & Setup

```bash
# 1. Clone the repository
git clone https://github.com/faizzanasghar/prob-project.git
cd prob-project/weather_dashboard

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

> The app loads the real dataset automatically if `pakistan_weather_2000_2024.csv` is present.
> Without it, realistic synthetic data is generated as a fallback.

---

## 📦 Tech Stack

| Layer | Technology |
|-------|-----------|
| **UI** | Streamlit, Plotly (interactive charts), Plotly Figure Factory (KDE) |
| **Data** | pandas, NumPy |
| **Statistics** | SciPy (Shapiro-Wilk, KS test, t-interval, Poisson) |
| **ML** | scikit-learn (RF, GBM, Ridge), StandardScaler Pipeline |
| **API** | Open-Meteo (free, no key) via `requests` |
| **Visualization** | Plotly Express, Plotly Graph Objects, Violin plots, Bubble maps |

---

## 👨‍💻 Authors

**Faizan asghar** — [GitHub](https://github.com/faizzanasghar)

---

*Built for university Probability & Statistics project — Pakistan weather data 2000–2024*
