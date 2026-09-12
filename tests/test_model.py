"""
Unit Tests for Model Building, Regularization, and Cross-Validation.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_generator import generate_real_estate_dataset
from src.preprocessing import DataPreprocessor, prepare_train_test_split
from src.models import build_pipeline, run_5fold_cv, run_loocv_fast, extract_coefficients
from src.evaluation import calculate_metrics, calculate_vif


@pytest.fixture(scope="module")
def prepared_data():
    """Module-level fixture providing preprocessed train and test data."""
    df = generate_real_estate_dataset(n_samples=300, random_state=42)
    X_tr, X_te, y_tr, y_te = prepare_train_test_split(df, test_size=0.20, random_state=42)

    preprocessor = DataPreprocessor()
    X_train_proc, _ = preprocessor.fit_transform(X_tr)
    X_test_proc, _ = preprocessor.transform(X_te)

    return X_train_proc, X_test_proc, y_tr, y_te


def test_model_pipeline_training(prepared_data):
    """Test that OLS, Lasso, and Ridge pipelines fit and predict valid scores."""
    X_train, X_test, y_train, y_test = prepared_data

    for model_name in ["ols", "ridge", "lasso"]:
        pipeline = build_pipeline(model_name)
        pipeline.fit(X_train, y_train)

        preds = pipeline.predict(X_test)
        metrics = calculate_metrics(y_test.values, preds)

        assert metrics["r2"] > 0.60, f"{model_name} R2 should be reasonable"
        assert metrics["rmse"] > 0.0, f"{model_name} RMSE should be positive"
        assert len(preds) == len(y_test)


def test_5fold_cross_validation(prepared_data):
    """Test 5-Fold cross-validation outputs consistent metrics."""
    X_train, _, y_train, _ = prepared_data
    pipe = build_pipeline("ridge")
    pipe.fit(X_train, y_train)

    cv_results = run_5fold_cv(pipe, X_train, y_train)

    assert len(cv_results["fold_details"]) == 5
    assert cv_results["mean_rmse"] > 0
    assert cv_results["mean_r2"] > 0.50


def test_analytical_loocv(prepared_data):
    """Test fast analytical LOOCV runs and produces expected metric format."""
    X_train, _, y_train, _ = prepared_data
    pipe = build_pipeline("ridge")
    pipe.fit(X_train, y_train)

    loocv_metrics = run_loocv_fast(pipe, X_train, y_train)

    assert "rmse" in loocv_metrics
    assert "mae" in loocv_metrics
    assert "r2" in loocv_metrics
    assert loocv_metrics["rmse"] > 0


def test_vif_calculation(prepared_data):
    """Test VIF calculation on numerical predictors."""
    X_train, _, _, _ = prepared_data
    vif_df = calculate_vif(X_train)

    assert "Feature" in vif_df.columns
    assert "VIF" in vif_df.columns
    assert len(vif_df) > 5
    # living_area_sqft and bedrooms should be high VIF
    top_features = vif_df.head(5)["Feature"].tolist()
    assert any(f in top_features for f in ["living_area_sqft", "bedrooms"])
