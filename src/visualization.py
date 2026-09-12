"""
Interactive Visualization Module for Real Estate Econometrics.
Generates publication-quality Plotly figures for EDA, multicollinearity,
residual diagnostics, coefficient comparisons, and sensitivity curves.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats


# Professional color tokens
PRIMARY_COLOR = "#2563EB"
SECONDARY_COLOR = "#0D9488"
ACCENT_COLOR = "#F59E0B"
DANGER_COLOR = "#EF4444"
GRID_COLOR = "#E2E8F0"
CARD_BG = "#FFFFFF"


def plot_price_distribution(df: pd.DataFrame, price_col: str = "property_price") -> go.Figure:
    """Plot distribution of property prices with mean and median indicators."""
    fig = px.histogram(
        df,
        x=price_col,
        nbins=40,
        marginal="box",
        title="Distribution of Property Price ($10k units)",
        labels={price_col: "Property Price ($10,000s)"},
        color_discrete_sequence=[PRIMARY_COLOR],
        opacity=0.85,
    )
    mean_val = df[price_col].mean()
    median_val = df[price_col].median()

    fig.add_vline(x=mean_val, line_dash="dash", line_color="#DC2626", annotation_text=f"Mean: ${mean_val:.1f}k")
    fig.add_vline(x=median_val, line_dash="dot", line_color="#059669", annotation_text=f"Median: ${median_val:.1f}k")

    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, features: Optional[List[str]] = None) -> go.Figure:
    """Generate interactive correlation heatmap for numerical predictors."""
    if features is None:
        num_df = df.select_dtypes(include=[np.number])
    else:
        num_df = df[features].select_dtypes(include=[np.number])

    corr_matrix = num_df.corr().round(2)

    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1.0,
        zmax=1.0,
        title="Predictor Correlation Matrix (Multicollinearity Diagnostic)",
    )
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def plot_scatter_with_trend(
    df: pd.DataFrame,
    x_col: str,
    y_col: str = "property_price",
    color_col: Optional[str] = "property_condition",
) -> go.Figure:
    """Create scatter plot with trendline."""
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col if color_col in df.columns else None,
        trendline="ols",
        trendline_color_override="#1E293B",
        title=f"{y_col.replace('_', ' ').title()} vs {x_col.replace('_', ' ').title()}",
        labels={x_col: x_col.replace('_', ' ').title(), y_col: "Price ($10k)"},
        opacity=0.6,
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def plot_coefficients_comparison(df_coef: pd.DataFrame) -> go.Figure:
    """
    Side-by-side grouped bar chart comparing standardized coefficients
    for OLS, Lasso, and Ridge.
    """
    fig = go.Figure()

    # OLS bars
    fig.add_trace(go.Bar(
        x=df_coef["Feature"],
        y=df_coef["OLS"],
        name="OLS (Unregularized)",
        marker_color="#94A3B8",
    ))

    # Lasso bars
    fig.add_trace(go.Bar(
        x=df_coef["Feature"],
        y=df_coef["LASSO"],
        name="Lasso (L1 Regularized)",
        marker_color="#EF4444",
    ))

    # Ridge bars
    fig.add_trace(go.Bar(
        x=df_coef["Feature"],
        y=df_coef["RIDGE"],
        name="Ridge (L2 Regularized)",
        marker_color="#2563EB",
    ))

    fig.update_layout(
        barmode="group",
        title="Standardized Coefficients Comparison (OLS vs Lasso vs Ridge)",
        xaxis_title="Features",
        yaxis_title="Standardized Beta Weight",
        template="plotly_white",
        xaxis_tickangle=-45,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=70, b=120),
    )
    return fig


def plot_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Ridge") -> go.Figure:
    """Plot Actual vs Predicted values with ideal 45-degree y=x line."""
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)

    min_val = min(float(y_t.min()), float(y_p.min()))
    max_val = max(float(y_t.max()), float(y_p.max()))

    fig = go.Figure()

    # Scatter points
    fig.add_trace(go.Scatter(
        x=y_t,
        y=y_p,
        mode="markers",
        name="Predictions",
        marker=dict(color=PRIMARY_COLOR, opacity=0.5, size=6),
    ))

    # Reference y=x line
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode="lines",
        name="Ideal Fit (y = x)",
        line=dict(color="#DC2626", dash="dash", width=2),
    ))

    fig.update_layout(
        title=f"Actual vs Predicted Property Price ({model_name})",
        xaxis_title="Actual Price ($10k)",
        yaxis_title="Predicted Price ($10k)",
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def plot_residual_diagnostics(y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    """Plot Residuals vs Predicted values to assess homoscedasticity."""
    y_p = np.asarray(y_pred)
    res = np.asarray(y_true) - y_p

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=y_p,
        y=res,
        mode="markers",
        name="Residuals",
        marker=dict(color="#64748B", opacity=0.5, size=6),
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="#DC2626", line_width=2)

    fig.update_layout(
        title="Residuals vs Fitted Values (Homoscedasticity Check)",
        xaxis_title="Fitted Price ($10k)",
        yaxis_title="Residual (Actual - Predicted)",
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def plot_residual_distribution(y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    """Plot histogram and KDE of model residuals."""
    res = np.asarray(y_true) - np.asarray(y_pred)

    fig = px.histogram(
        x=res,
        nbins=40,
        marginal="rug",
        title="Residual Error Distribution",
        labels={"x": "Residual Error ($10k)"},
        color_discrete_sequence=[SECONDARY_COLOR],
        opacity=0.8,
    )
    fig.add_vline(x=0, line_dash="dash", line_color="#DC2626")
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def plot_qq_plot(y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    """Normal Q-Q Plot of residuals."""
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    res_std = (residuals - np.mean(residuals)) / np.std(residuals)

    osm, osr = stats.probplot(res_std, dist="norm")
    theo_q = osm[0]
    sample_q = osm[1]
    slope = osr[0]
    intercept = osr[1]

    line_x = np.array([theo_q.min(), theo_q.max()])
    line_y = slope * line_x + intercept

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=theo_q,
        y=sample_q,
        mode="markers",
        name="Sample Quantiles",
        marker=dict(color=PRIMARY_COLOR, size=6, opacity=0.7),
    ))
    fig.add_trace(go.Scatter(
        x=line_x,
        y=line_y,
        mode="lines",
        name="Theoretical Normal Line",
        line=dict(color="#DC2626", dash="dash", width=2),
    ))
    fig.update_layout(
        title="Normal Q-Q Plot of Residuals",
        xaxis_title="Theoretical Quantiles (Normal Distribution)",
        yaxis_title="Standardized Residuals",
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def plot_error_by_segment(df_segments: pd.DataFrame) -> go.Figure:
    """Bar chart comparing RMSE and MAE across market price tiers."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_segments["Segment"],
        y=df_segments["RMSE ($10k)"],
        name="RMSE ($10k)",
        marker_color="#DC2626",
    ))
    fig.add_trace(go.Bar(
        x=df_segments["Segment"],
        y=df_segments["MAE ($10k)"],
        name="MAE ($10k)",
        marker_color="#F59E0B",
    ))
    fig.update_layout(
        barmode="group",
        title="Prediction Error by Price Segment (Tier Consistency)",
        xaxis_title="Market Segment",
        yaxis_title="Error ($10,000s)",
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def plot_what_if(df_curve: pd.DataFrame, feature_name: str) -> go.Figure:
    """Sensitivity curve for What-If scenario analysis."""
    fig = px.line(
        df_curve,
        x=feature_name,
        y="predicted_price_usd",
        title=f"Sensitivity Analysis: {feature_name.replace('_', ' ').title()} vs Valuation",
        labels={
            feature_name: feature_name.replace('_', ' ').title(),
            "predicted_price_usd": "Estimated Property Value ($)",
        },
    )
    fig.update_traces(line=dict(color=PRIMARY_COLOR, width=3))
    fig.update_layout(
        template="plotly_white",
        yaxis_tickformat="$,.0f",
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig
