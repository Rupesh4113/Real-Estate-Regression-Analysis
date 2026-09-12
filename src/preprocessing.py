"""
Data Preprocessing Pipeline for Real Estate Regression Analysis.
Handles missing value imputation, statistical outlier capping (Winsorization),
target log transformations, and train-test partitioning without data leakage.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


NUMERICAL_PREDICTORS = [
    "living_area_sqft",
    "bedrooms",
    "bathrooms",
    "property_age",
    "lot_size",
    "parking_spaces",
    "neighborhood_quality",
    "distance_to_transit",
    "distance_to_school",
    "crime_index",
    "accessibility_score",
    "local_income",
    "unemployment_rate",
    "inflation_rate",
]

CATEGORICAL_PREDICTORS = [
    "property_condition",
]

TARGET_COLUMN = "property_price"


def get_missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a detailed summary of missing values per column.
    """
    total = df.isnull().sum()
    percent = (df.isnull().sum() / len(df)) * 100.0
    summary = pd.DataFrame({
        "Missing Count": total,
        "Percentage (%)": percent.round(2),
        "Data Type": df.dtypes.astype(str),
    })
    return summary[summary["Missing Count"] > 0]


class DataPreprocessor:
    """
    Robust leak-free preprocessor tracking fitted statistics from training data.
    """

    def __init__(
        self,
        num_cols: Optional[List[str]] = None,
        cat_cols: Optional[List[str]] = None,
        outlier_iqr_multiplier: float = 3.0,
    ):
        self.num_cols = num_cols or NUMERICAL_PREDICTORS
        self.cat_cols = cat_cols or CATEGORICAL_PREDICTORS
        self.outlier_iqr_multiplier = outlier_iqr_multiplier
        self.numerical_medians_: Dict[str, float] = {}
        self.categorical_modes_: Dict[str, str] = {}
        self.outlier_bounds_: Dict[str, Tuple[float, float]] = {}
        self.is_fitted: bool = False

    def fit(self, X: pd.DataFrame) -> "DataPreprocessor":
        """Fit imputation medians/modes and outlier bounds on training partition."""
        X_df = X.copy()

        # 1. Fit numerical medians & outlier bounds
        for col in self.num_cols:
            if col in X_df.columns:
                median_val = float(X_df[col].median())
                self.numerical_medians_[col] = median_val

                # Compute conservative outlier bounds (IQR * multiplier) to avoid truncating luxury homes
                q25 = float(X_df[col].quantile(0.25))
                q75 = float(X_df[col].quantile(0.75))
                iqr = q75 - q25
                lower_bound = max(0.0, q25 - self.outlier_iqr_multiplier * iqr)
                upper_bound = q75 + self.outlier_iqr_multiplier * iqr
                self.outlier_bounds_[col] = (lower_bound, upper_bound)

        # 2. Fit categorical modes
        for col in self.cat_cols:
            if col in X_df.columns:
                mode_series = X_df[col].mode()
                self.categorical_modes_[col] = mode_series.iloc[0] if not mode_series.empty else "Average"

        self.is_fitted = True
        return self

    def transform(
        self,
        X: pd.DataFrame,
        cap_outliers: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Apply missing value imputation and optional outlier capping.
        Returns transformed DataFrame and statistics on imputations/cappings.
        """
        if not self.is_fitted:
            raise ValueError("DataPreprocessor must be fitted before transforming.")

        X_out = X.copy()
        stats = {
            "imputed_cells": int(X_out.isnull().sum().sum()),
            "capped_cells": 0,
        }

        # Impute numericals
        for col, median_val in self.numerical_medians_.items():
            if col in X_out.columns:
                X_out[col] = X_out[col].fillna(median_val)

        # Impute categoricals
        for col, mode_val in self.categorical_modes_.items():
            if col in X_out.columns:
                X_out[col] = X_out[col].fillna(mode_val)

        # Capping outliers
        if cap_outliers:
            for col, (lower, upper) in self.outlier_bounds_.items():
                if col in X_out.columns:
                    mask = (X_out[col] < lower) | (X_out[col] > upper)
                    stats["capped_cells"] += int(mask.sum())
                    X_out[col] = X_out[col].clip(lower=lower, upper=upper)

        return X_out, stats

    def fit_transform(
        self,
        X: pd.DataFrame,
        cap_outliers: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        return self.fit(X).transform(X, cap_outliers=cap_outliers)


def prepare_train_test_split(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split dataset into 80/20 train and test sets deterministically.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def transform_target(y: pd.Series) -> pd.Series:
    """Log1p transform target variable."""
    return np.log1p(y)


def inverse_transform_target(y_log: np.ndarray) -> np.ndarray:
    """Invert log1p transform back to natural price units ($10k)."""
    return np.expm1(y_log)
