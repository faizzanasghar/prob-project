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
