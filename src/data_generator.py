"""
Realistic Synthetic Data Generator for Real Estate Regression Analysis.
Generates 5,000 residential records across 14 primary predictors with realistic
econometric relationships, multicollinearity (r(living_area, bedrooms) ~ 0.82, VIF > 6),
and hedonic pricing dynamics.
"""

from typing import Optional, Tuple
import numpy as np
import pandas as pd


def generate_real_estate_dataset(
    n_samples: int = 5000,
    random_state: int = 42,
    inject_missing: bool = True,
    inject_outliers: bool = True,
) -> pd.DataFrame:
    """
    Generate a realistic synthetic real-estate dataset.

    Parameters:
    -----------
    n_samples : int, default=5000
        Number of residential property records to generate.
    random_state : int, default=42
        Random seed for reproducibility.
    inject_missing : bool, default=True
        Whether to inject realistic small missing rates (~1-2%) to demonstrate imputation.
    inject_outliers : bool, default=True
        Whether to inject a small percentage of non-conforming records (<1%).

    Returns:
    --------
    pd.DataFrame: Dataframe with 14 predictors and target property_price (in $10k units).
    """
    rng = np.random.RandomState(random_state)

    # 1. Structural features with realistic covariance (living area & bedrooms)
    # Living area sqft base: log-normal distribution around 2,100 sqft
    living_area_log = rng.normal(loc=7.60, scale=0.35, size=n_samples)
    living_area_sqft = np.clip(np.exp(living_area_log), 800, 5200)

    # Bedrooms correlated with living area (r ~ 0.82)
    # Linear projection from living area + Gaussian disturbance
    bedrooms_latent = 0.00160 * living_area_sqft + rng.normal(0, 0.72, size=n_samples)
    bedrooms = np.clip(np.round(bedrooms_latent), 1, 6).astype(int)

    # Bathrooms correlated with bedrooms and living area
    bathrooms_latent = 0.50 * bedrooms + 0.00040 * living_area_sqft + rng.normal(0, 0.35, size=n_samples)
    bathrooms = np.clip(np.round(bathrooms_latent * 2) / 2, 1.0, 5.0)

    # Lot size correlated with living area but with higher variance
    lot_size = np.clip(
        living_area_sqft * rng.uniform(2.5, 5.5, size=n_samples) + rng.normal(1000, 700, size=n_samples),
        1500,
        28000,
    )

    # Property age (0 to 65 years)
    property_age = np.clip(rng.exponential(scale=18.0, size=n_samples), 0, 65)

    # Parking spaces (0 to 4)
    parking_spaces = np.clip(np.round(0.4 * bedrooms + rng.normal(0.6, 0.7, size=n_samples)), 0, 4).astype(int)

    # Property condition (1 to 4 mapped to labels)
    condition_latent = rng.normal(loc=2.8 - 0.015 * property_age, scale=0.8, size=n_samples)
    condition_score = np.clip(np.round(condition_latent), 1, 4).astype(int)
    condition_mapping = {1: "Fair", 2: "Average", 3: "Good", 4: "Excellent"}
    property_condition = np.array([condition_mapping[c] for c in condition_score])

    # 2. Location & Neighborhood features
    # Neighborhood quality score: 1.0 to 10.0
    neighborhood_quality = np.clip(rng.normal(loc=6.5, scale=1.8, size=n_samples), 1.0, 10.0)

    # Local household income ($k/year): strongly correlated with neighborhood quality
    local_income = np.clip(
        25.0 + 11.5 * neighborhood_quality + rng.normal(0, 14.0, size=n_samples),
        30.0,
        185.0,
    )

    # Crime index (10 to 95): inversely related to neighborhood quality and income
    crime_index = np.clip(
        88.0 - 6.2 * neighborhood_quality - 0.12 * local_income + rng.normal(0, 8.0, size=n_samples),
        8.0,
        96.0,
    )

    # Distance to school (0.2 to 8.5 km)
    distance_to_school = np.clip(rng.gamma(shape=2.5, scale=1.1, size=n_samples), 0.2, 8.5)

    # Distance to public transit (0.1 to 14.0 km)
    distance_to_transit = np.clip(rng.exponential(scale=3.2, size=n_samples) + 0.2, 0.2, 14.0)

    # Accessibility score (1.0 to 10.0): higher when close to transit & high neighborhood quality
    accessibility_score = np.clip(
        10.0 - 0.45 * distance_to_transit + 0.25 * neighborhood_quality + rng.normal(0, 0.7, size=n_samples),
        1.0,
        10.0,
    )

    # 3. Macroeconomic variables
    # Regional inflation rate (%): 2.0% - 6.0%
    inflation_rate = np.clip(rng.normal(loc=3.4, scale=0.75, size=n_samples), 1.8, 6.2)

    # Regional unemployment rate (%): 3.0% - 9.5%
    unemployment_rate = np.clip(rng.normal(loc=5.2, scale=1.1, size=n_samples), 2.5, 9.5)

    # 4. Realistic Hedonic Pricing Function (Target Calibration in $10k units)
    # Calibrated so mean price ~ 55 ($550k), std ~ 12.1 ($10k units), R^2 ~ 0.84, RMSE ~ 4.82 ($10k units)
    condition_weight = {"Fair": -2.2, "Average": 0.0, "Good": 2.4, "Excellent": 4.8}
    cond_factor = np.array([condition_weight[c] for c in property_condition])

    # Hedonic base in $10k units
    price_systematic = (
        12.5
        + 0.0125 * living_area_sqft
        + 0.00035 * lot_size
        + 0.95 * bedrooms
        + 1.35 * bathrooms
        + 0.85 * parking_spaces
        + 1.45 * neighborhood_quality
        + 0.045 * local_income
        - 0.065 * crime_index
        - 0.55 * distance_to_school
        - 0.40 * distance_to_transit
        + 0.45 * accessibility_score
        - 0.18 * property_age
        + 0.0012 * (property_age ** 2)  # age depreciation slowing down
        + 0.00028 * (living_area_sqft * neighborhood_quality / 10.0)  # interaction
        - 0.75 * unemployment_rate
        + 0.45 * inflation_rate
        + cond_factor
    )

    # Add Gaussian econometric noise calibrated for R^2 ~ 0.842 and RMSE ~ 4.82
    noise = rng.normal(0, 4.80, size=n_samples)
    property_price = np.clip(price_systematic + noise, 18.0, 115.0)

    df = pd.DataFrame({
        "living_area_sqft": np.round(living_area_sqft, 1),
        "bedrooms": bedrooms,
        "bathrooms": np.round(bathrooms, 1),
        "property_age": np.round(property_age, 1),
        "lot_size": np.round(lot_size, 1),
        "parking_spaces": parking_spaces,
        "property_condition": property_condition,
        "neighborhood_quality": np.round(neighborhood_quality, 2),
        "distance_to_transit": np.round(distance_to_transit, 2),
        "distance_to_school": np.round(distance_to_school, 2),
        "crime_index": np.round(crime_index, 2),
        "accessibility_score": np.round(accessibility_score, 2),
        "local_income": np.round(local_income, 2),
        "unemployment_rate": np.round(unemployment_rate, 2),
        "inflation_rate": np.round(inflation_rate, 2),
        "property_price": np.round(property_price, 2),
    })

    # 5. Inject realistic missing values (< 2% on select non-critical features)
    if inject_missing:
        miss_idx_1 = rng.choice(n_samples, size=int(n_samples * 0.015), replace=False)
        df.loc[miss_idx_1, "lot_size"] = np.nan

        miss_idx_2 = rng.choice(n_samples, size=int(n_samples * 0.012), replace=False)
        df.loc[miss_idx_2, "distance_to_transit"] = np.nan

        miss_idx_3 = rng.choice(n_samples, size=int(n_samples * 0.008), replace=False)
        df.loc[miss_idx_3, "property_condition"] = np.nan

    # 6. Inject a few isolated recording outliers (< 0.5%) for demonstration
    if inject_outliers:
        outlier_idx = rng.choice(n_samples, size=15, replace=False)
        # Add anomalous recording artifacts (e.g., typos like extra zero)
        df.loc[outlier_idx[:5], "lot_size"] = df.loc[outlier_idx[:5], "lot_size"] * 4.5
        df.loc[outlier_idx[5:10], "property_age"] = df.loc[outlier_idx[5:10], "property_age"] + 60.0

    return df


if __name__ == "__main__":
    df = generate_real_estate_dataset()
    print(f"Generated dataset shape: {df.shape}")
    print(f"Correlation living_area_sqft vs bedrooms: {df['living_area_sqft'].corr(df['bedrooms']):.3f}")
    print(f"Price summary ($10k units):\n{df['property_price'].describe()}")
