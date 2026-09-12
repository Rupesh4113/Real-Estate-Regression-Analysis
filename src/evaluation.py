"""
Econometric Evaluation & Diagnostics Module.
Computes MAE, MSE, RMSE, R2, Multicollinearity VIF diagnostics,
residual distributions, and segment error breakdowns.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute standard regression performance metrics dynamically.

    Returns:
    --------
    dict with keys: 'mae', 'mse', 'rmse', 'r2'
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    mse = float(mean_squared_error(y_true_arr, y_pred_arr))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true_arr, y_pred_arr))

    return {
        "mae": round(mae, 3),
        "mse": round(mse, 3),
        "rmse": round(rmse, 3),
        "r2": round(r2, 4),
    }


def calculate_vif(X: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Variance Inflation Factor (VIF) for all numerical features.

    Parameters:
    -----------
    X : pd.DataFrame
        DataFrame of numerical predictors (without target).

    Returns:
    --------
    pd.DataFrame with 'Feature' and 'VIF' columns, sorted descending by VIF.
    """
    # Select only numeric columns and drop constants / nulls
    X_num = X.select_dtypes(include=[np.number]).dropna()
    if X_num.empty:
        return pd.DataFrame(columns=["Feature", "VIF"])

    vif_data = []
    # Add constant column for proper VIF calculation
    X_with_const = X_num.copy()
    X_with_const["_const"] = 1.0

    for i, col in enumerate(X_num.columns):
        try:
            val = variance_inflation_factor(X_with_const.values, i)
            # Clip if infinite or extreme
            vif_data.append({"Feature": col, "VIF": round(float(val), 2)})
        except Exception:
            vif_data.append({"Feature": col, "VIF": np.nan})

    vif_df = pd.DataFrame(vif_data).sort_values(by="VIF", ascending=False).reset_index(drop=True)
    return vif_df


def evaluate_price_segments(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """
    Evaluate prediction error across price segments:
    - Budget (< 25th percentile)
    - Mid-market (25th - 50th percentile)
    - Premium (50th - 75th percentile)
    - Luxury (> 75th percentile)
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)

    q25 = np.percentile(y_t, 25)
    q50 = np.percentile(y_t, 50)
    q75 = np.percentile(y_t, 75)

    segments = []
    for val in y_t:
        if val <= q25:
            segments.append("Budget")
        elif val <= q50:
            segments.append("Mid-Market")
        elif val <= q75:
            segments.append("Premium")
        else:
            segments.append("Luxury")

    df_eval = pd.DataFrame({
        "actual": y_t,
        "predicted": y_p,
        "segment": segments,
        "abs_error": np.abs(y_t - y_p),
        "sq_error": (y_t - y_p) ** 2,
    })

    segment_order = ["Budget", "Mid-Market", "Premium", "Luxury"]
    summary = []

    for seg in segment_order:
        sub = df_eval[df_eval["segment"] == seg]
        if len(sub) > 0:
            mae = float(sub["abs_error"].mean())
            rmse = float(np.sqrt(sub["sq_error"].mean()))
            r2 = float(r2_score(sub["actual"], sub["predicted"])) if len(sub) > 1 else 0.0
            price_range = f"${sub['actual'].min():.1f}k - ${sub['actual'].max():.1f}k"
            summary.append({
                "Segment": seg,
                "Properties": len(sub),
                "Price Range": price_range,
                "MAE ($10k)": round(mae, 2),
                "RMSE ($10k)": round(rmse, 2),
                "Segment R²": round(r2, 3),
            })

    return pd.DataFrame(summary)


def compute_residual_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Calculate statistical diagnostic tests on residuals.
    Includes Skewness, Kurtosis, and normality metrics.
    """
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    res_skew = float(stats.skew(residuals))
    res_kurt = float(stats.kurtosis(residuals))

    # Jarque-Bera normality test
    jb_stat, jb_pvalue = stats.jarque_bera(residuals)

    return {
        "mean_residual": round(float(np.mean(residuals)), 4),
        "std_residual": round(float(np.std(residuals)), 3),
        "skewness": round(res_skew, 3),
        "kurtosis": round(res_kurt, 3),
        "jb_statistic": round(float(jb_stat), 2),
        "jb_pvalue": float(jb_pvalue),
    }
