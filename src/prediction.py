"""
Inference, Valuation, Sensitivity What-If, and Local Explainability Engine.
Translates input features into property valuations with confidence intervals,
local linear attributions, and sensitivity curves.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def predict_property_value(
    pipeline: Pipeline,
    input_dict: Dict[str, Any],
    rmse: float = 4.82,
    confidence_level: float = 0.95,
) -> Dict[str, Any]:
    """
    Predict property valuation from raw user input dictionary.

    Returns:
    --------
    dict containing:
      - 'predicted_price_10k': float ($10k units)
      - 'predicted_price_usd': float (Total USD)
      - 'lower_ci_usd': float
      - 'upper_ci_usd': float
      - 'uncertainty_usd': float (half-width margin of error)
    """
    input_df = pd.DataFrame([input_dict])

    # Predict in target units ($10k)
    y_pred_10k = float(pipeline.predict(input_df)[0])

    # 95% confidence/prediction margin based on model test RMSE
    z_score = 1.96 if confidence_level == 0.95 else 1.645
    margin_10k = z_score * rmse

    lower_10k = max(0.0, y_pred_10k - margin_10k)
    upper_10k = y_pred_10k + margin_10k

    return {
        "predicted_price_10k": round(y_pred_10k, 2),
        "predicted_price_usd": round(y_pred_10k * 10000.0, 2),
        "lower_ci_10k": round(lower_10k, 2),
        "upper_ci_10k": round(upper_10k, 2),
        "lower_ci_usd": round(lower_10k * 10000.0, 2),
        "upper_ci_usd": round(upper_10k * 10000.0, 2),
        "margin_usd": round(margin_10k * 10000.0, 2),
    }


def explain_prediction(
    pipeline: Pipeline,
    input_dict: Dict[str, Any],
    top_n: int = 8,
) -> pd.DataFrame:
    """
    Decompose a linear model prediction into baseline + individual feature contributions:
        Contribution_j = beta_j * ((x_trans_j - mean_j) / std_j)
    Expressed in non-causal language and converted to both $10k and USD impact.
    """
    fe = pipeline.named_steps["feature_engineer"]
    scaler = pipeline.named_steps["scaler"]
    regressor = pipeline.named_steps["regressor"]

    # Transform single input
    df_raw = pd.DataFrame([input_dict])
    df_engineered = fe.transform(df_raw)
    feature_names = fe.get_feature_names_out()

    x_vals = df_engineered.values[0]
    means = scaler.mean_
    scales = scaler.scale_
    coefs = regressor.coef_

    # Calculate standardized deviations from mean and their linear valuation impact
    contributions = []
    for name, x, mu, s, beta in zip(feature_names, x_vals, means, scales, coefs):
        z_score = (x - mu) / (s if s != 0 else 1.0)
        impact_10k = beta * z_score
        impact_usd = impact_10k * 10000.0

        direction = "Positive (Higher)" if impact_usd >= 0 else "Negative (Discount)"
        sign_str = "+" if impact_usd >= 0 else "-"

        # Human-readable feature name
        clean_name = (
            name.replace("_", " ")
            .replace("sqft", "sq ft")
            .replace("log ", "Log of ")
            .title()
        )

        contributions.append({
            "Feature": clean_name,
            "Raw Value": round(float(x), 2),
            "Impact ($10k)": round(float(impact_10k), 2),
            "Valuation Impact ($)": round(float(impact_usd), 2),
            "Display Impact": f"{sign_str}${abs(impact_usd):,.0f}",
            "Direction": direction,
            "AbsImpact": abs(impact_usd),
        })

    df_contrib = pd.DataFrame(contributions)
    # Sort by absolute impact
    df_contrib = (
        df_contrib.sort_values(by="AbsImpact", ascending=False)
        .head(top_n)
        .drop(columns=["AbsImpact"])
        .reset_index(drop=True)
    )

    return df_contrib


def generate_what_if_curve(
    pipeline: Pipeline,
    base_input: Dict[str, Any],
    feature_to_vary: str,
    min_val: float,
    max_val: float,
    num_steps: int = 50,
) -> pd.DataFrame:
    """
    Generate sensitivity data varying a single predictor across a continuous grid
    while holding all other property attributes constant.
    """
    grid_vals = np.linspace(min_val, max_val, num_steps)
    rows = []

    for v in grid_vals:
        modified_input = base_input.copy()
        modified_input[feature_to_vary] = v
        rows.append(modified_input)

    df_grid = pd.DataFrame(rows)
    preds_10k = pipeline.predict(df_grid)
    preds_usd = preds_10k * 10000.0

    return pd.DataFrame({
        feature_to_vary: grid_vals,
        "predicted_price_10k": preds_10k,
        "predicted_price_usd": preds_usd,
    })
