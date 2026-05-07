import pandas as pd
import numpy as np
import os
import streamlit as st

DATA_PATH = "pakistan_weather_2000_2024.csv"

def load_data(path: str = None) -> pd.DataFrame:
    """Load and preprocess the Pakistan weather dataset."""
    if path is None:
        # Try to find the CSV relative to this file
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(base_dir, DATA_PATH)
    
    if not os.path.exists(path):
        # Generate synthetic data matching the schema for demo purposes
        df = _generate_synthetic_data()
        df["_data_source"] = "synthetic"
    else:
        df = pd.read_csv(path, parse_dates=["date"])
        df["_data_source"] = "real"
        # Validate the loaded data
        try:
            _validate_data(df)
        except Exception as e:
            st.error(f"Data Validation Error: {e}")
            # Fallback to synthetic if real data is corrupted
            df = _generate_synthetic_data()
            df["_data_source"] = "synthetic (fallback)"

        # Map real CSV column names to internal names used by the app
        rename_map = {
            "prcp(Precipitation)": "prcp",
            "wspd(Wind Speed)": "wind_speed",
        }
        df = df.rename(columns=rename_map)
        
        # Normalize categorical columns to match sidebar filter options
        if "rainfall_intensity" in df.columns:
            df["rainfall_intensity"] = df["rainfall_intensity"].str.capitalize()
        if "wind_category" in df.columns:
            wcat_map = {"calm": "Calm", "breezy": "Breeze", "windy": "Moderate", "strong": "Strong", "storm": "Storm"}
            df["wind_category"] = df["wind_category"].str.lower().map(wcat_map).fillna("Moderate")

    df = _preprocess(df)
    return df


def _validate_data(df: pd.DataFrame) -> None:
    """Validate the structure and content of the weather data."""
    required_columns = ['date', 'city', 'latitude', 'longitude']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if df.empty:
        raise ValueError("Dataset is empty")

    # Check date column
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        raise ValueError("Date column must be datetime type")


def _preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and enrich the dataframe."""
    # Ensure date column is datetime
    if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df = df.dropna(subset=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Derived time columns
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["month_name"] = df["date"].dt.strftime("%b")
    df["day_of_year"] = df["date"].dt.dayofyear

    # Ensure numeric types
    numeric_cols = [
        "tavg", "tmin", "tmax", "prcp", "humidity", "pressure",
        "wind_speed", "wind_deg", "wind_gust", "cloud_cover",
        "sunshine_hours", "temp_range", "latitude", "longitude",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill synthetic fields if missing
    if "temp_range" not in df.columns and "tmax" in df.columns and "tmin" in df.columns:
        df["temp_range"] = df["tmax"] - df["tmin"]

    if "is_hot_day" not in df.columns and "tmax" in df.columns:
        df["is_hot_day"] = (df["tmax"] >= 40).astype(int)

    if "is_cold_day" not in df.columns and "tmin" in df.columns:
        df["is_cold_day"] = (df["tmin"] <= 5).astype(int)

    if "season" not in df.columns:
        df["season"] = df["month"].map({
            12: "Winter", 1: "Winter", 2: "Winter",
            3: "Spring", 4: "Spring", 5: "Spring",
            6: "Summer", 7: "Summer", 8: "Summer",
            9: "Autumn", 10: "Autumn", 11: "Autumn",
        })

    if "rainfall_intensity" not in df.columns:
        if "prcp" in df.columns:
            conditions = [
                df["prcp"] == 0,
                df["prcp"] < 5,
                df["prcp"] < 15,
                df["prcp"] < 30,
            ]
            choices = ["None", "Light", "Moderate", "Heavy"]
            df["rainfall_intensity"] = np.select(conditions, choices, default="Extreme")

    if "wind_category" not in df.columns and "wind_speed" in df.columns:
        conditions = [
            df["wind_speed"] < 5,
            df["wind_speed"] < 15,
            df["wind_speed"] < 25,
            df["wind_speed"] < 40,
        ]
        choices = ["Calm", "Breeze", "Moderate", "Strong"]
        df["wind_category"] = np.select(conditions, choices, default="Storm")

    return df


def _generate_synthetic_data() -> pd.DataFrame:
    """Generate realistic synthetic Pakistan weather data using vectorized operations."""
    np.random.seed(42)
    cities_dict = {
        "Karachi":    (24.86, 67.01),
        "Lahore":     (31.55, 74.35),
        "Islamabad":  (33.72, 73.04),
        "Peshawar":   (34.01, 71.57),
        "Quetta":     (30.19, 67.01),
        "Multan":     (30.19, 71.47),
        "Faisalabad": (31.42, 73.08),
        "Hyderabad":  (25.39, 68.38),
    }

    dates = pd.date_range("2000-01-01", "2024-12-31", freq="D")
    num_dates = len(dates)
    num_cities = len(cities_dict)
    
    city_names = list(cities_dict.keys())
    city_indices = np.repeat(np.arange(num_cities), num_dates)
    all_dates = np.tile(dates, num_cities)
    all_dates_series = pd.Series(all_dates)
    
    lats = np.array([cities_dict[c][0] for c in city_names])[city_indices]
    lons = np.array([cities_dict[c][1] for c in city_names])[city_indices]
    
    doys = all_dates_series.dt.dayofyear
    months = all_dates_series.dt.month
    years = all_dates_series.dt.year
    
    base_temp = np.where(lats > 30, 22, 28)
    seasonal = 15 * np.sin(2 * np.pi * (doys - 90) / 365)
    warming = (years - 2000) * 0.04
    noise = np.random.normal(0, 3, size=num_dates * num_cities)
    
    tavg = base_temp + seasonal + warming + noise
    tmin = tavg - np.random.uniform(3, 8, size=num_dates * num_cities)
    tmax = tavg + np.random.uniform(3, 8, size=num_dates * num_cities)
    
    prcp = np.zeros(num_dates * num_cities)
    monsoon_mask = np.isin(months, [7, 8, 9]) & (lats > 25)
    winter_mask = np.isin(months, [12, 1, 2]) & (lats > 30)
    other_mask = ~(monsoon_mask | winter_mask)
    
    prcp[monsoon_mask] = np.random.exponential(8, size=monsoon_mask.sum())
    prcp[winter_mask] = np.random.exponential(3, size=winter_mask.sum())
    
    rare_events = (np.random.random(size=other_mask.sum()) < 0.2)
    prcp[other_mask] = np.where(rare_events, np.random.exponential(0.5, size=other_mask.sum()), 0)
    
    prcp = np.maximum(0, prcp)
    humidity = np.clip(40 + prcp * 2 + np.random.normal(0, 10, size=num_dates * num_cities), 20, 100)
    pressure = np.random.normal(1013, 8, size=num_dates * num_cities)
    wind_speed = np.random.gamma(2, 3, size=num_dates * num_cities)
    wind_gust = wind_speed + np.random.exponential(3, size=num_dates * num_cities)
    
    df = pd.DataFrame({
        "date": all_dates,
        "city": [city_names[i] for i in city_indices],
        "latitude": lats + np.random.normal(0, 0.01, size=num_dates * num_cities),
        "longitude": lons + np.random.normal(0, 0.01, size=num_dates * num_cities),
        "tavg": np.round(tavg, 1),
        "tmin": np.round(tmin, 1),
        "tmax": np.round(tmax, 1),
        "prcp": np.round(prcp, 1),
        "humidity": np.round(humidity, 1),
        "pressure": np.round(pressure, 1),
        "wind_speed": np.round(wind_speed, 1),
        "wind_deg": np.random.randint(0, 360, size=num_dates * num_cities),
        "wind_gust": np.round(wind_gust, 1),
        "cloud_cover": np.round(np.random.uniform(0, 100, size=num_dates * num_cities), 1),
        "sunshine_hours": np.round(np.random.uniform(0, 12, size=num_dates * num_cities), 1),
    })
    
    return df


def get_city_coords(df: pd.DataFrame) -> pd.DataFrame:
    """Return city-level coordinate summary."""
    return (
        df.groupby("city")
        .agg(latitude=("latitude", "mean"), longitude=("longitude", "mean"))
        .reset_index()
    )


def filter_data(
    df: pd.DataFrame,
    cities: list,
    year_range: tuple,
    seasons: list,
    rainfall_intensities: list,
    wind_categories: list,
    temp_range: tuple,
) -> pd.DataFrame:
    """Apply all sidebar filters."""
    mask = (
        df["city"].isin(cities)
        & df["year"].between(*year_range)
        & df["season"].isin(seasons)
        & df["rainfall_intensity"].isin(rainfall_intensities)
        & df["wind_category"].isin(wind_categories)
        & df["tavg"].between(*temp_range)
    )
    return df[mask].copy()
