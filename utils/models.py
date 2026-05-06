import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings("ignore")

# ─── Feature Engineering ─────────────────────────────────────────────────────

# Features for predicting the SAME day (Anomaly Detection)
SAME_DAY_TEMP_FEATURES = [
    "month", "day_of_year", "humidity", "pressure",
    "cloud_cover", "wind_speed", "sunshine_hours",
]

# Features for predicting the NEXT day (Forecasting)
NEXT_DAY_TEMP_FEATURES = [
    "tavg", "humidity", "pressure", "cloud_cover", 
    "wind_speed", "sunshine_hours", "month_sin", "month_cos", 
    "day_sin", "day_cos"
]

RAIN_FEATURES = [
    "month", "day_of_year", "humidity", "cloud_cover",
    "wind_speed", "pressure", "tavg",
]


def cyclical_encode(df: pd.DataFrame) -> pd.DataFrame:
    """Encode month and day_of_year using sine/cosine transformations."""
    df = df.copy()
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    
    # Approx 365.25 days
    df["day_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    return df


def build_features(df: pd.DataFrame, feature_cols: list, target_col: str) -> tuple:
    """Return (X, y) arrays dropping NaNs and performing cyclical encoding."""
    # Ensure cyclical features are present if requested
    if any(c in feature_cols for c in ["month_sin", "day_sin"]):
        df = cyclical_encode(df)

    cols = feature_cols + [target_col]
    available = [c for c in cols if c in df.columns]
    sub = df[available].dropna()

    usable_features = [c for c in feature_cols if c in sub.columns]

    X = sub[usable_features].values
    y = sub[target_col].values
    idx = sub.index
    return X, y, idx, usable_features


# ─── Model Training ──────────────────────────────────────────────────────────

MODEL_MAP = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=150, 
        learning_rate=0.05, 
        max_depth=4, 
        loss='huber', 
        random_state=42
    ),
}


def train_model(
    df: pd.DataFrame,
    target: str = "tavg",
    model_type: str = "Random Forest",
    test_size: float = 0.2,
    is_next_day: bool = False
) -> dict:
    """
    Train a regression model to predict `target`.
    Returns a result dict with model, metrics, predictions, residuals.
    """
    if is_next_day:
        feature_cols = NEXT_DAY_TEMP_FEATURES if "tavg" in target else RAIN_FEATURES
    else:
        feature_cols = SAME_DAY_TEMP_FEATURES if target == "tavg" else RAIN_FEATURES

    X, y, idx, used_features = build_features(df, feature_cols, target)

    if len(X) < 50:
        return {"error": "Insufficient data (need ≥ 50 rows after dropping NaNs)"}

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, idx, test_size=test_size, random_state=42
    )

    model = MODEL_MAP.get(model_type, RandomForestRegressor(n_estimators=100, random_state=42))
    scaler = StandardScaler()

    # Scale inputs for linear models
    if "Regression" in model_type:
        X_train_fit = scaler.fit_transform(X_train)
        X_test_fit = scaler.transform(X_test)
        X_all_fit = scaler.transform(X)
    else:
        X_train_fit = X_train
        X_test_fit = X_test
        X_all_fit = X

    model.fit(X_train_fit, y_train)

    y_pred_test = model.predict(X_test_fit)
    y_pred_all = model.predict(X_all_fit)

    residuals = y - y_pred_all
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
    mae = float(mean_absolute_error(y_test, y_pred_test))
    r2 = float(r2_score(y_test, y_pred_test))

    # Feature importance
    importance = None
    if hasattr(model, "feature_importances_"):
        importance = dict(zip(used_features, model.feature_importances_))
    elif hasattr(model, "coef_"):
        importance = dict(zip(used_features, np.abs(model.coef_)))

    # Residual threshold for anomaly flagging (2 × RMSE)
    residual_threshold = 2 * rmse

    # Build result dataframe
    result_df = df.loc[idx].copy()
    result_df["predicted"] = y_pred_all
    result_df["residual"] = residuals
    result_df["residual_anomaly"] = result_df["residual"].abs() > residual_threshold

    return {
        "model": model,
        "model_type": model_type,
        "target": target,
        "features": used_features,
        "metrics": {"RMSE": round(rmse, 3), "MAE": round(mae, 3), "R²": round(r2, 4)},
        "result_df": result_df,
        "residual_threshold": round(residual_threshold, 3),
        "feature_importance": importance,
        "y_test": y_test,
        "y_pred_test": y_pred_test,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "scaler": scaler if "Regression" in model_type else None
    }


def train_next_day_model(df: pd.DataFrame, target: str = "tavg", model_type: str = "Random Forest") -> dict:
    """
    Train a model to predict tomorrow's target based on today's features.
    """
    ldf = df.sort_values(["city", "date"]).copy()
    # Create target for next day: shift(-1) within each city
    ldf["target_next_day"] = ldf.groupby("city")[target].shift(-1)
    
    # Drop the last record for each city since it has no 'next day' target
    ldf = ldf.dropna(subset=["target_next_day"])
    
    # Use NEXT_DAY_TEMP_FEATURES
    feature_cols = NEXT_DAY_TEMP_FEATURES if target == "tavg" else RAIN_FEATURES
    
    # Train using the standard pipeline but with the shifted target
    result = train_model(ldf, target="target_next_day", model_type=model_type, is_next_day=True)
    
    if "error" not in result:
        result["original_target"] = target
        # result["features"] should already be used_features from train_model
        
    return result


def predict_next_day(model_dict: dict, input_data: dict) -> dict:
    """
    Predict next day value based on a trained model dictionary and manual input data.
    Enforces physical consistency constraints.
    """
    if "error" in model_dict:
        return {"error": "Invalid model"}
        
    model = model_dict["model"]
    features = model_dict["features"]
    scaler = model_dict.get("scaler")
    
    # Encode input if month/day are provided
    if "month" in input_data:
        m = input_data["month"]
        input_data["month_sin"] = np.sin(2 * np.pi * m / 12)
        input_data["month_cos"] = np.cos(2 * np.pi * m / 12)
    if "day_of_year" in input_data:
        d = input_data["day_of_year"]
        input_data["day_sin"] = np.sin(2 * np.pi * d / 365.25)
        input_data["day_cos"] = np.cos(2 * np.pi * d / 365.25)

    # Prepare input vector
    input_row = []
    for f in features:
        if f not in input_data:
            # Fallback for missing cyclical features if not already in input_data
            if f == "month_sin": input_data[f] = np.sin(2 * np.pi * input_data.get("month", 1) / 12)
            elif f == "month_cos": input_data[f] = np.cos(2 * np.pi * input_data.get("month", 1) / 12)
            elif f == "day_sin": input_data[f] = np.sin(2 * np.pi * input_data.get("day_of_year", 1) / 365.25)
            elif f == "day_cos": input_data[f] = np.cos(2 * np.pi * input_data.get("day_of_year", 1) / 365.25)
            
        val = input_data.get(f, 0.0)
        input_row.append(val)
        
    X_input = np.array([input_row])
    
    if scaler:
        X_input = scaler.transform(X_input)
        
    prediction = float(model.predict(X_input)[0])
    
    # ─── Physical Consistency Constraint ──────────────────────────────────────
    # Avoid a 17-degree crash unless extreme pressure drops occur.
    # We clip the prediction to not deviate more than 15% from today's Tavg
    # unless pressure is very low (indicating a major storm).
    today_tavg = input_data.get("tavg", prediction)
    pressure = input_data.get("pressure", 1013)
    
    if pressure > 1000: # Normal range
        max_diff = max(abs(today_tavg) * 0.15, 3.0) # At least 3 degrees allowed
        prediction = np.clip(prediction, today_tavg - max_diff, today_tavg + max_diff)
    
    # Simple CI estimate based on RMSE
    rmse = model_dict["metrics"]["RMSE"]
    ci_lower = prediction - 1.96 * rmse
    ci_upper = prediction + 1.96 * rmse
    
    return {
        "prediction": round(prediction, 2),
        "ci_lower": round(ci_lower, 2),
        "ci_upper": round(ci_upper, 2),
        "rmse": rmse
    }


# ─── Utility ─────────────────────────────────────────────────────────────────

def get_most_extreme_year(df: pd.DataFrame) -> int:
    """Return year with most anomalies (combined hot+cold days + extreme rain)."""
    if "is_anomaly" in df.columns:
        return int(df.groupby("year")["is_anomaly"].sum().idxmax())
    if "is_hot_day" in df.columns and "is_cold_day" in df.columns:
        extremes = df.groupby("year")[["is_hot_day", "is_cold_day"]].sum().sum(axis=1)
        return int(extremes.idxmax())
    return int(df["year"].max())
