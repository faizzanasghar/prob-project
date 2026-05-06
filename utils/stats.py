import pandas as pd
import numpy as np
from scipy import stats


# ─── Confidence Interval ────────────────────────────────────────────────────

def compute_ci(series: pd.Series, confidence: float = 0.95) -> dict:
    """Compute mean, std, and confidence interval for a series."""
    n = len(series.dropna())
    if n < 2:
        return {}
    mean = series.mean()
    se = stats.sem(series.dropna())
    ci = stats.t.interval(confidence, df=n - 1, loc=mean, scale=se)
    return {
        "mean": mean,
        "std": series.std(),
        "ci_lower": ci[0],
        "ci_upper": ci[1],
        "n": n,
    }


def flag_ci_anomalies(df: pd.DataFrame, col: str, confidence: float = 0.95) -> pd.Series:
    """Return boolean Series: True = outside CI per city+month group."""
    anomaly = pd.Series(False, index=df.index)

    for (city, month), group in df.groupby(["city", "month"]):
        ci = compute_ci(group[col])
        if not ci or "ci_lower" not in ci or "ci_upper" not in ci:
            continue
        idx = group.index
        outside = (group[col] < ci["ci_lower"]) | (group[col] > ci["ci_upper"])
        anomaly.loc[idx] = outside

    return anomaly


# ─── Z-Score ─────────────────────────────────────────────────────────────────

def compute_zscore(df: pd.DataFrame, col: str) -> pd.Series:
    """Z-score normalised per city."""
    z = pd.Series(np.nan, index=df.index)
    for city, group in df.groupby("city"):
        s = group[col].dropna()
        if s.std() == 0:
            continue
        z_vals = (group[col] - s.mean()) / s.std()
        z.loc[group.index] = z_vals
    return z


def flag_zscore_anomalies(df: pd.DataFrame, col: str, threshold: float = 2.0) -> pd.Series:
    """Return boolean Series: True = |z| > threshold."""
    z = compute_zscore(df, col)
    return z.abs() > threshold


# ─── p-values ────────────────────────────────────────────────────────────────

def compute_pvalue(series: pd.Series) -> pd.Series:
    """Two-tailed p-value under normal distribution for each point."""
    clean = series.dropna()
    mean = clean.mean()
    std = clean.std()
    if std == 0:
        return pd.Series(1.0, index=series.index)
    z = (series - mean) / std
    p_vals = 2 * (1 - stats.norm.cdf(np.abs(z)))
    return pd.Series(p_vals, index=series.index)


def flag_pvalue_anomalies(series: pd.Series, alpha: float = 0.05) -> pd.Series:
    """Return boolean Series: True = p < alpha."""
    return compute_pvalue(series) < alpha


# ─── Normality Test ──────────────────────────────────────────────────────────

def normality_test(series: pd.Series) -> dict:
    """Shapiro-Wilk or D'Agostino K² test (auto-selects by sample size)."""
    clean = series.dropna()
    n = len(clean)
    if n < 8:
        return {"test": "insufficient_data", "statistic": None, "p_value": None, "is_normal": None}

    if n <= 5000:
        stat, p = stats.shapiro(clean.sample(min(n, 5000), random_state=42))
        test_name = "Shapiro-Wilk"
    else:
        stat, p = stats.normaltest(clean)
        test_name = "D'Agostino K²"

    return {
        "test": test_name,
        "statistic": round(float(stat), 4),
        "p_value": round(float(p), 6),
        "is_normal": p > 0.05,
    }


# ─── Distribution Fitting ────────────────────────────────────────────────────

def fit_poisson(series: pd.Series) -> dict:
    """Fit a Poisson distribution to non-negative integer-rounded data."""
    clean = series.dropna().clip(lower=0)
    lam = clean.mean()
    # KS test against fitted Poisson (discretised)
    rounded = clean.round().astype(int)
    # Compare empirical CDF to Poisson CDF
    from scipy.stats import poisson as sp_poisson
    ks_stat, ks_p = stats.kstest(rounded, sp_poisson(lam).cdf)
    return {"lambda": round(lam, 3), "ks_stat": round(ks_stat, 4), "ks_p": round(ks_p, 6)}


# ─── Combined Anomaly Score ──────────────────────────────────────────────────

def combined_anomaly_flag(
    df: pd.DataFrame,
    col: str,
    ci_confidence: float = 0.95,
    z_threshold: float = 2.0,
    residual_col: str = None,
    residual_threshold: float = None,
) -> pd.Series:
    """
    Anomaly = CI_anomaly OR z_anomaly OR (residual_anomaly if provided).
    Returns boolean Series.
    """
    ci_flag = flag_ci_anomalies(df, col, confidence=ci_confidence)
    z_flag = flag_zscore_anomalies(df, col, threshold=z_threshold)

    combined = ci_flag | z_flag

    if residual_col is not None and residual_col in df.columns and residual_threshold is not None:
        res_flag = df[residual_col].abs() > residual_threshold
        combined = combined | res_flag

    return combined


# ─── Monthly Statistics ──────────────────────────────────────────────────────

def monthly_stats(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Return monthly mean ± 2σ bounds."""
    grp = df.groupby(["city", "month", "month_name"])[col].agg(
        mean="mean", std="std", count="count"
    ).reset_index()
    grp["upper_ci"] = grp["mean"] + 1.96 * grp["std"] / np.sqrt(grp["count"])
    grp["lower_ci"] = grp["mean"] - 1.96 * grp["std"] / np.sqrt(grp["count"])
    grp["upper_2std"] = grp["mean"] + 2 * grp["std"]
    grp["lower_2std"] = grp["mean"] - 2 * grp["std"]
    return grp


# ─── Descriptive Statistics ──────────────────────────────────────────────────

def get_descriptive_stats(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Return full descriptive stats (Table 1 style) grouped by city."""
    stats_df = df.groupby("city")[col].describe().reset_index()
    return stats_df


def compute_skew_kurt(series: pd.Series) -> dict:
    """Compute skewness and kurtosis."""
    clean = series.dropna()
    if len(clean) < 3:
        return {"skew": 0.0, "kurt": 0.0}
    return {
        "skew": round(float(stats.skew(clean)), 3),
        "kurt": round(float(stats.kurtosis(clean)), 3),
    }


def get_ci_table(df: pd.DataFrame, col: str, confidence: float = 0.95) -> pd.DataFrame:
    """Return CI table (Table 2 style) with Mean and Standard Error."""
    rows = []
    for city in df["city"].unique():
        s = df[df["city"] == city][col].dropna()
        n = len(s)
        if n < 2: continue
        mean = s.mean()
        se = stats.sem(s)
        ci = stats.t.interval(confidence, df=n-1, loc=mean, scale=se)
        rows.append({
            "City": city,
            "Mean": round(mean, 3),
            "Std Error": round(se, 3),
            f"CI Lower ({int(confidence*100)}%)": round(ci[0], 3),
            f"CI Upper ({int(confidence*100)}%)": round(ci[1], 3),
        })
    return pd.DataFrame(rows)


# ─── Anomaly Frequency ───────────────────────────────────────────────────────

def anomaly_frequency(df: pd.DataFrame, anomaly_col: str = "is_anomaly") -> pd.DataFrame:
    """Pivot table: city × month anomaly counts."""
    if anomaly_col not in df.columns:
        return pd.DataFrame()
    pivot = df.pivot_table(
        index="city",
        columns="month_name",
        values=anomaly_col,
        aggfunc="sum",
        fill_value=0,
    )
    month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    cols = [m for m in month_order if m in pivot.columns]
    return pivot[cols]


# ─── Predicted Anomaly Classifier ────────────────────────────────────────────

def classify_predicted_anomaly(
    predicted_temp: float,
    city_df: pd.DataFrame,
    month: int,
    confidence: float = 0.95,
) -> dict:
    """
    Classify whether a predicted temperature is anomalous relative to the
    historical distribution for that city and month.

    Returns a dict with:
        verdict       : 'Normal' | 'Warm Anomaly' | 'Cold Anomaly' | 'Extreme Heat' | 'Extreme Cold'
        severity      : 0-100 severity score
        zscore        : z-score of prediction vs historical mean
        pvalue        : two-tailed p-value
        ci_lower/upper: historical 95% CI bounds
        hist_mean     : historical monthly mean
        hist_std      : historical monthly std
        explanation   : human-readable explanation string
    """
    month_data = city_df[city_df["month"] == month]["tavg"].dropna()

    if len(month_data) < 10:
        return {"verdict": "Insufficient data", "severity": 0,
                "explanation": "Not enough historical data for this month."}

    hist_mean = float(month_data.mean())
    hist_std  = float(month_data.std())
    n = len(month_data)

    if hist_std == 0:
        return {"verdict": "Normal", "severity": 0, "explanation": "No variance in historical data."}

    zscore = (predicted_temp - hist_mean) / hist_std
    pvalue = float(2 * (1 - stats.norm.cdf(abs(zscore))))

    se = stats.sem(month_data)
    ci = stats.t.interval(confidence, df=n - 1, loc=hist_mean, scale=se)

    severity = min(100, round(abs(zscore) / 3.0 * 100))

    is_outside_ci = predicted_temp < ci[0] or predicted_temp > ci[1]
    is_significant = pvalue < (1 - confidence)

    if abs(zscore) >= 3.0:
        verdict = "Extreme Heat" if zscore > 0 else "Extreme Cold"
    elif abs(zscore) >= 2.0:
        verdict = "Warm Anomaly" if zscore > 0 else "Cold Anomaly"
    elif is_outside_ci and is_significant:
        verdict = "Marginal Anomaly"
    else:
        verdict = "Normal"

    direction = "above" if zscore > 0 else "below"
    explanation = (
        f"Predicted {predicted_temp:.1f}°C is {abs(zscore):.2f}σ {direction} the historical "
        f"average for this month ({hist_mean:.1f}°C ± {hist_std:.1f}°C). "
        f"Historical 95% CI: [{ci[0]:.1f}°C, {ci[1]:.1f}°C]. "
        f"p-value = {pvalue:.4f} — "
        f"{'Statistically significant anomaly.' if is_significant else 'Within normal range.'}"
    )

    return {
        "verdict": verdict,
        "severity": severity,
        "zscore": round(zscore, 3),
        "pvalue": round(pvalue, 4),
        "ci_lower": round(ci[0], 2),
        "ci_upper": round(ci[1], 2),
        "hist_mean": round(hist_mean, 2),
        "hist_std": round(hist_std, 2),
        "n_samples": n,
        "explanation": explanation,
        "is_anomaly": is_outside_ci and is_significant,
    }
