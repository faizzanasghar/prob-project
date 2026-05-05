# 🌦️ Pakistan Weather Anomaly Detection Dashboard

A production-grade Streamlit application for 25 years of Pakistan weather data analysis.

## Features

| Page | Description |
|------|-------------|
| 🏠 Home | KPI cards, trend overview, rainfall/wind summaries |
| 🔍 Data Exploration | Time series, histograms, boxplots, heatmaps, scatter plots |
| 📊 Statistical Analysis | Mean, Std Dev, 95% CI, CI-based anomaly flagging |
| 🎲 Probability Analysis | Z-scores, p-values, normality tests, Poisson fit |
| 🤖 Prediction Engine | Random Forest / Linear Regression with RMSE, MAE, R² |
| ⚠️ Anomaly Dashboard | Combined CI + Z-score + Residual anomaly detection |
| 🗺️ Map Visualization | Bubble map coloured by anomaly intensity |
| 💡 Insights | Auto-generated trends, city comparisons, seasonal patterns |

## Installation

```bash
# 1. Clone / copy this directory
cd weather_dashboard

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place your CSV in the same directory
cp path/to/pakistan_weather_2000_2024.csv .

# 4. Run the app
streamlit run app.py
```

The app also works **without the CSV** — it will auto-generate
realistic synthetic data matching the schema so you can explore the UI.

## Project Structure

```
weather_dashboard/
├── app.py                    # Main Streamlit app (multi-page)
├── requirements.txt
├── pakistan_weather_2000_2024.csv   # ← place your data here
├── .streamlit/
│   └── config.toml           # Dark theme config
└── utils/
    ├── __init__.py
    ├── data_loader.py         # Data loading, preprocessing, filtering
    ├── stats.py               # CI, Z-score, p-values, anomaly flagging
    └── models.py              # Regression model training & evaluation
```

## Anomaly Detection Logic

```
Anomaly = CI_anomaly  OR  |Z-score| > threshold  OR  high_residual
```

- **CI anomaly**: value outside 95% confidence interval for that city × month group
- **Z-score anomaly**: |z| > 2.0 (configurable in sidebar)
- **Residual anomaly**: |residual| > 2 × RMSE from the trained ML model

## Tech Stack

- **Streamlit** — UI framework
- **Plotly** — interactive visualisations
- **pandas / numpy** — data manipulation
- **scipy.stats** — statistical tests (Shapiro-Wilk, KS, normality)
- **scikit-learn** — Random Forest, Gradient Boosting, Linear Regression
