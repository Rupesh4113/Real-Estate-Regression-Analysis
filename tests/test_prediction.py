"""
Unit Tests for Valuation Engine, Sensitivity Analysis, and Explainability.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_generator import generate_real_estate_dataset
from src.preprocessing import DataPreprocessor, prepare_train_test_split
from src.models import build_pipeline
from src.prediction import (
    predict_property_value,
    explain_prediction,
    generate_what_if_curve,
)


@pytest.fixture(scope="module")
def fitted_pipeline():
    """Module-level fixture providing fitted Ridge pipeline."""
    df = generate_real_estate_dataset(n_samples=300, random_state=42)
    X_tr, _, y_tr, _ = prepare_train_test_split(df, test_size=0.20, random_state=42)

    preprocessor = DataPreprocessor()
    X_proc, _ = preprocessor.fit_transform(X_tr)

    pipe = build_pipeline("ridge")
    pipe.fit(X_proc, y_tr)
    return pipe


@pytest.fixture
def sample_property():
    """Sample residential property specification."""
    return {
        "living_area_sqft": 2400.0,
        "bedrooms": 4,
        "bathrooms": 2.5,
        "property_age": 12.0,
        "lot_size": 8500.0,
        "parking_spaces": 2,
        "property_condition": "Good",
        "neighborhood_quality": 8.0,
        "distance_to_transit": 1.5,
        "distance_to_school": 1.2,
        "crime_index": 25.0,
        "accessibility_score": 8.5,
        "local_income": 95.0,
        "unemployment_rate": 4.5,
        "inflation_rate": 3.2,
    }


def test_predict_property_value(fitted_pipeline, sample_property):
    """Test prediction formatting, confidence intervals, and USD conversion."""
    result = predict_property_value(fitted_pipeline, sample_property, rmse=4.82)

    assert "predicted_price_10k" in result
    assert "predicted_price_usd" in result
    assert "lower_ci_usd" in result
    assert "upper_ci_usd" in result
    assert result["lower_ci_usd"] < result["predicted_price_usd"] < result["upper_ci_usd"]
    assert result["predicted_price_usd"] > 200000.0


def test_explain_prediction(fitted_pipeline, sample_property):
    """Test linear feature attribution breakdown."""
    df_explain = explain_prediction(fitted_pipeline, sample_property, top_n=6)

    assert len(df_explain) == 6
    assert "Feature" in df_explain.columns
    assert "Valuation Impact ($)" in df_explain.columns
    assert "Direction" in df_explain.columns
    assert "Display Impact" in df_explain.columns

    # High living area or neighborhood should be among influential features
    features_list = df_explain["Feature"].str.lower().tolist()
    assert any("living" in f or "neighborhood" in f or "area" in f for f in features_list)


def test_what_if_sensitivity_curve(fitted_pipeline, sample_property):
    """Test What-If continuous grid generation."""
    df_curve = generate_what_if_curve(
        fitted_pipeline,
        sample_property,
        feature_to_vary="living_area_sqft",
        min_val=1000.0,
        max_val=4500.0,
        num_steps=20,
    )

    assert len(df_curve) == 20
    assert "living_area_sqft" in df_curve.columns
    assert "predicted_price_usd" in df_curve.columns
    # Check monotonic increase with living area
    assert df_curve["predicted_price_usd"].iloc[-1] > df_curve["predicted_price_usd"].iloc[0]
