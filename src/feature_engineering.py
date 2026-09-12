"""
Feature Engineering Transformers for Real Estate Econometric Regression.
Generates log transforms, domain interaction terms, and depreciation polynomial terms
with full scikit-learn Pipeline compatibility and feature name tracking.
"""

from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class RealEstateFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer that applies:
    1. Log-transformations to skewed predictors (living_area_sqft, lot_size)
    2. Domain interaction features:
       - living_area_sqft * neighborhood_quality
       - living_area_sqft * (10 - distance_to_school)
       - local_income * neighborhood_quality
    3. Non-linear age depreciation polynomial:
       - property_age^2
    4. Ordinal encoding of property_condition
    """

    CONDITION_MAP = {
        "Fair": 1.0,
        "Average": 2.0,
        "Good": 3.0,
        "Excellent": 4.0,
    }

    def __init__(
        self,
        apply_log_transforms: bool = True,
        apply_interactions: bool = True,
        apply_polynomials: bool = True,
    ):
        self.apply_log_transforms = apply_log_transforms
        self.apply_interactions = apply_interactions
        self.apply_polynomials = apply_polynomials
        self.feature_names_out_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "RealEstateFeatureEngineer":
        """Validate input and prepare feature names."""
        # Convert to DataFrame if array
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self._generate_feature_names(X)
        return self

    def _generate_feature_names(self, X: pd.DataFrame) -> None:
        names = []
        base_cols = [c for c in X.columns]

        for col in base_cols:
            if col == "property_condition":
                names.append("property_condition_score")
            else:
                names.append(col)

        if self.apply_log_transforms:
            if "living_area_sqft" in base_cols:
                names.append("log_living_area_sqft")
            if "lot_size" in base_cols:
                names.append("log_lot_size")

        if self.apply_interactions:
            if "living_area_sqft" in base_cols and "neighborhood_quality" in base_cols:
                names.append("living_x_neighborhood")
            if "living_area_sqft" in base_cols and "distance_to_school" in base_cols:
                names.append("living_x_school_proximity")
            if "local_income" in base_cols and "neighborhood_quality" in base_cols:
                names.append("income_x_neighborhood")

        if self.apply_polynomials:
            if "property_age" in base_cols:
                names.append("property_age_squared")

        self.feature_names_out_ = names

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering transformations."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        df = X.copy()

        # Map condition to ordinal score
        if "property_condition" in df.columns:
            df["property_condition_score"] = (
                df["property_condition"]
                .map(self.CONDITION_MAP)
                .fillna(2.0)
                .astype(float)
            )
            df = df.drop(columns=["property_condition"])

        # Log transformations
        if self.apply_log_transforms:
            if "living_area_sqft" in df.columns:
                df["log_living_area_sqft"] = np.log1p(df["living_area_sqft"])
            if "lot_size" in df.columns:
                df["log_lot_size"] = np.log1p(df["lot_size"])

        # Interaction features
        if self.apply_interactions:
            if "living_area_sqft" in df.columns and "neighborhood_quality" in df.columns:
                df["living_x_neighborhood"] = (
                    df["living_area_sqft"] * (df["neighborhood_quality"] / 10.0)
                )
            if "living_area_sqft" in df.columns and "distance_to_school" in df.columns:
                # Proximity score = 1 / (distance + 0.5)
                df["living_x_school_proximity"] = (
                    df["living_area_sqft"] / (df["distance_to_school"] + 0.5)
                )
            if "local_income" in df.columns and "neighborhood_quality" in df.columns:
                df["income_x_neighborhood"] = (
                    (df["local_income"] / 100.0) * df["neighborhood_quality"]
                )

        # Polynomial features
        if self.apply_polynomials:
            if "property_age" in df.columns:
                df["property_age_squared"] = df["property_age"] ** 2

        # Reindex to ensure consistent column ordering matching feature_names_out_
        if self.feature_names_out_:
            existing = [c for c in self.feature_names_out_ if c in df.columns]
            df = df[existing]

        return df

    def get_feature_names_out(self, input_features=None) -> List[str]:
        """Return the engineered feature column names."""
        return self.feature_names_out_
