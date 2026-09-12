"""
Unit Tests for Preprocessing and Feature Engineering Pipelines.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_generator import generate_real_estate_dataset
from src.preprocessing import (
    DataPreprocessor,
    prepare_train_test_split,
    transform_target,
    inverse_transform_target,
    get_missing_value_summary,
)
from src.feature_engineering import RealEstateFeatureEngineer


@pytest.fixture
def sample_data():
    """Generate small synthetic dataset fixture."""
    return generate_real_estate_dataset(n_samples=200, random_state=42)


def test_missing_value_imputation(sample_data):
    """Test that missing values are identified and fully imputed."""
    missing_before = get_missing_value_summary(sample_data)
    assert len(missing_before) > 0, "Expected missing values in raw dataset"

    preprocessor = DataPreprocessor()
    transformed_df, stats = preprocessor.fit_transform(sample_data.drop(columns=["property_price"]))

    assert transformed_df.isnull().sum().sum() == 0, "All missing values should be imputed"
    assert stats["imputed_cells"] > 0, "Imputation count should be greater than zero"


def test_outlier_capping(sample_data):
    """Test that extreme outliers are capped within IQR bounds."""
    preprocessor = DataPreprocessor(outlier_iqr_multiplier=2.5)
    transformed_df, stats = preprocessor.fit_transform(sample_data.drop(columns=["property_price"]))

    # Check bounds respected
    for col, (lower, upper) in preprocessor.outlier_bounds_.items():
        if col in transformed_df.columns:
            assert transformed_df[col].min() >= lower - 1e-5
            assert transformed_df[col].max() <= upper + 1e-5


def test_train_test_split_shapes(sample_data):
    """Test train/test partition proportions without data leakage."""
    X_train, X_test, y_train, y_test = prepare_train_test_split(
        sample_data, target_col="property_price", test_size=0.20, random_state=42
    )

    assert len(X_train) == 160
    assert len(X_test) == 40
    assert len(y_train) == 160
    assert len(y_test) == 40
    assert "property_price" not in X_train.columns


def test_target_transformation_invertibility():
    """Test that log1p target transformation accurately inverts with expm1."""
    original = np.array([15.5, 45.0, 72.8, 120.4, 350.0])
    transformed = transform_target(original)
    inverted = inverse_transform_target(transformed)

    np.testing.assert_allclose(original, inverted, rtol=1e-6)


def test_feature_engineering_transformers(sample_data):
    """Test generation of interaction, polynomial, and log features."""
    fe = RealEstateFeatureEngineer(
        apply_log_transforms=True,
        apply_interactions=True,
        apply_polynomials=True,
    )
    X = sample_data.drop(columns=["property_price"]).fillna(1.0)
    transformed = fe.fit_transform(X)

    output_features = fe.get_feature_names_out()

    # Check that engineered features exist in output
    assert "log_living_area_sqft" in output_features
    assert "living_x_neighborhood" in output_features
    assert "property_age_squared" in output_features
    assert "property_condition_score" in output_features
    assert transformed.shape[1] == len(output_features)
    assert not transformed.isnull().values.any()
