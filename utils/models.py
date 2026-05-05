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

TEMP_FEATURES = [
    "month", "day_of_year", "humidity", "pressure",
    "cloud_cover", "wind_speed", "sunshine_hours",
]

RAIN_FEATURES = [
    "month", "day_of_year", "humidity", "cloud_cover",
    "wind_speed", "pressure", "tavg",
]


def build_features(df: pd.DataFrame, feature_cols: list, target_col: str) -> tuple:
    """Return (X, y) arrays dropping NaNs."""
    cols = feature_cols + [target_col]
    available = [c for c in cols if c in df.columns]
    sub = df[available].dropna()

    missing_features = [c for c in feature_cols if c not in df.columns]
    usable_features = [c for c in feature_cols if c in df.columns]

    X = sub[usable_features].values
    y = sub[target_col].values
    idx = sub.index
    return X, y, idx, usable_features


# ─── Model Training ──────────────────────────────────────────────────────────

MODEL_MAP = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
}


def train_model(
    df: pd.DataFrame,
    target: str = "tavg",
    model_type: str = "Random Forest",
    test_size: float = 0.2,
) -> dict:
    """
    Train a regression model to predict `target`.
    Returns a result dict with model, metrics, predictions, residuals.
    """
    feature_cols = TEMP_FEATURES if target == "tavg" else RAIN_FEATURES

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
