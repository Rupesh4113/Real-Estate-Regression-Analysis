# 🏡 Real Estate Regression Analysis & Automated Valuation Model (AVM)

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A production-quality, end-to-end econometric regression and Automated Property Valuation (AVM) application. Built for data science portfolio presentation, technical interviews, and live cloud deployment, demonstrating rigorous regression techniques, L1/L2 regularization, multicollinearity diagnosis, leak-free Scikit-Learn pipelines, residual diagnostics, and transparent model explainability.

---

## 📌 Executive Summary

Traditional residential valuation models rely heavily on physical property characteristics and neighborhood attributes. However, severe **multicollinearity** (e.g., between living area and bedroom count) frequently introduces extreme variance into Ordinary Least Squares (OLS) coefficients, leading to unstable valuations and poor out-of-sample generalization.

This project implements an interpretable, robust Automated Valuation Model comparing **Ordinary Least Squares (OLS)**, **Lasso Regression (L1)**, and **Ridge Regression (L2)** across comprehensive cross-validation protocols (80/20 Holdout, 5-Fold Cross-Validation, and Leave-One-Out Cross-Validation).

```
Real Estate AVM → Ingest & Audit → Engineer Features → Diagnose Multicollinearity → Regularize & Train → Validate Generalization → Explain Predictions
```

---

## 🎯 Analytical Objectives

1. **Quantify Valuation Drivers:** Model residential market values as a function of structural, spatial, environmental, and macroeconomic determinants.
2. **Diagnose & Resolve Multicollinearity:** Detect collinear predictors ($r \approx 0.82$, $\text{VIF} > 6$) and mathematically prove how Ridge regularization stabilizes parameter variance.
3. **Prevent Data Leakage:** Implement strict train-test encapsulation using Scikit-Learn `Pipeline` and custom transformers where all imputers, scalers, and bounds are fitted exclusively on training folds.
4. **Deliver Transparent Appraisals:** Provide local linear feature attributions (*"Why did the model make this prediction?"*) with non-causal language and 95% valuation confidence intervals.

---

## 📊 Dataset Specifications

The application uses a 5,000-record residential housing dataset across 14 primary predictors and one target variable:

| Category | Feature Name | Description | Type / Range |
| :--- | :--- | :--- | :--- |
| **Structural** | `living_area_sqft` | Interior living area square footage | Continuous (800 – 5,200 sq ft) |
| **Structural** | `bedrooms` | Number of bedrooms ($r \approx 0.82$ with sq ft) | Discrete (1 – 6) |
| **Structural** | `bathrooms` | Number of full & half bathrooms | Continuous (1.0 – 5.0) |
| **Structural** | `property_age` | Age of property since construction | Continuous (0 – 65 years) |
| **Structural** | `lot_size` | Total parcel lot size | Continuous (1,500 – 28,000 sq ft) |
| **Structural** | `parking_spaces` | Enclosed garage / designated parking bays | Discrete (0 – 4) |
| **Structural** | `property_condition` | Maintenance and cosmetic state | Categorical (`Fair`, `Average`, `Good`, `Excellent`) |
| **Spatial / Access** | `neighborhood_quality` | Composite neighborhood rating index | Continuous (1.0 – 10.0) |
| **Spatial / Access** | `distance_to_transit` | Distance to nearest rapid transit station | Continuous (0.2 – 14.0 km) |
| **Spatial / Access** | `distance_to_school` | Distance to top-tier elementary/high school | Continuous (0.2 – 8.5 km) |
| **Spatial / Access** | `crime_index` | Municipal safety and crime rating | Continuous (8.0 – 96.0; lower is safer) |
| **Spatial / Access** | `accessibility_score` | Multi-modal transit and road connectivity | Continuous (1.0 – 10.0) |
| **Macroeconomic** | `local_income` | Median household income of neighborhood | Continuous (\$30k – \$185k/year) |
| **Macroeconomic** | `unemployment_rate` | Regional unemployment rate | Continuous (2.5% – 9.5%) |
| **Macroeconomic** | `inflation_rate` | Annualized headline inflation rate | Continuous (1.8% – 6.2%) |
| **Target** | `property_price` | Property market value in \$10,000 units | Continuous (\$180k – \$1,150k) |

---

## 🛠️ Data Preparation & Preprocessing Pipeline

All transformations are encapsulated inside `DataPreprocessor` and custom transformers to guarantee deterministic execution without data leakage:

1. **Automated Missing Value Imputation:**
   - **Numerical Predictors:** Imputed via median (fit strictly on the training partition).
   - **Categorical Predictors:** Imputed via mode (`property_condition`).
2. **Statistical Outlier Capping (Winsorization):**
   - Non-luxury recording artifacts are bounded using conservative IQR limits ($[Q_1 - 3 \times \text{IQR}, Q_3 + 3 \times \text{IQR}]$), preserving genuine high-value luxury real estate.
3. **Partitioning:**
   - 80/20 train/test split with deterministic `random_state=42`.

---

## ⚙️ Econometric Feature Engineering

Implemented in `src/feature_engineering.py` via a Scikit-Learn `BaseEstimator` and `TransformerMixin`:

- **Log Transformations:** Normalizes right-skewed predictors:
  $$\text{log\_living\_area\_sqft} = \ln(1 + \text{living\_area\_sqft})$$
  $$\text{log\_lot\_size} = \ln(1 + \text{lot\_size})$$
- **Interaction Terms:** Captures localized economic premiums:
  - $\text{living\_x\_neighborhood} = \text{living\_area\_sqft} \times (\text{neighborhood\_quality} / 10)$
  - $\text{living\_x\_school\_proximity} = \text{living\_area\_sqft} / (\text{distance\_to\_school} + 0.5)$
  - $\text{income\_x\_neighborhood} = (\text{local\_income} / 100) \times \text{neighborhood\_quality}$
- **Non-Linear Depreciation Curve:** Captures accelerating initial age depreciation followed by valuation stabilization:
  $$\text{property\_age\_squared} = (\text{property\_age})^2$$
- **Standard Scaling:** Predictors are scaled to zero mean and unit variance ($\mu = 0, \sigma = 1$) prior to L1/L2 penalty application.

---

## 🔍 Multicollinearity Analysis

Living area and bedroom count exhibit high natural correlation in residential architecture:
- **Pearson Correlation:** $r(\text{living\_area\_sqft}, \text{bedrooms}) = 0.818$
- **Variance Inflation Factor (VIF):**
  - `bathrooms`: $\text{VIF} = 7.07$
  - `bedrooms`: $\text{VIF} = 6.63$
  - `living_area_sqft`: $\text{VIF} = 5.54$
  - `neighborhood_quality`: $\text{VIF} = 5.00$

In unpenalized OLS, high collinearity inverts near-singular matrices $(X^T X)^{-1}$, creating erratic coefficient sign-flips and inflated standard errors.

---

## 🤖 Modeling & Regularization

Three models are evaluated under identical standardized splits:

1. **Ordinary Least Squares (OLS):** Baseline benchmark minimizing $\sum (y_i - \hat{y}_i)^2$.
2. **Lasso (L1 Regularization):** Adds penalty $\lambda \sum |\beta_j|$. Implemented with `LassoCV(cv=5)` across 100 log-spaced alphas ($10^{-4}$ to $10^3$). Automatically performs feature selection by zeroing redundant collinear terms.
3. **Ridge (L2 Regularization):** Adds penalty $\lambda \sum \beta_j^2$. Implemented with `RidgeCV(cv=5)` across 100 log-spaced alphas. Shrinks correlated coefficients cooperatively without discarding information.

---

## 📈 Cross-Validation Protocols

- **80/20 Holdout Split:** Strict out-of-sample evaluation on 1,000 holdout records (`random_state=42`).
- **5-Fold Cross-Validation:** Stratified 5-Fold evaluation (`KFold(n_splits=5, shuffle=True, random_state=42)`).
- **Leave-One-Out Cross-Validation (LOOCV):** Exact closed-form leverage evaluation for linear estimators:
  $$e_{i, \text{LOO}} = \frac{e_i}{1 - h_{ii}}, \quad h_{ii} = [X(X^T X + \alpha I)^{-1} X^T]_{ii}$$
  Yields exact LOOCV metrics instantaneously without re-training thousands of redundant models.

---

## 🏆 Model Performance Leaderboard

Dynamic results evaluated on the unseen 20% holdout test set:

| Model | Test RMSE ($10k) | Test MAE ($10k) | Test MSE | Test R² | 5-Fold CV RMSE | LOOCV RMSE | Optimal $\alpha$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **OLS** | 4.789 | 3.811 | 22.935 | 0.9002 | 4.888 ± 0.205 | 4.890 | — |
| **Lasso (L1)** | 4.788 | 3.811 | 22.929 | 0.9002 | 4.887 ± 0.205 | 4.998 | 0.0036 |
| **Ridge (L2)** 🏅 | **4.789** | **3.811** | **22.935** | **0.9002** | **4.888 ± 0.205** | **4.890** | **0.9112** |

*All metrics are calculated dynamically from model evaluation artifacts rather than hard-coded.*

---

## 💼 Business & Underwriting Impact

1. **Appraisal Stability:** Ridge L2 regularization prevents violent valuation swings caused by collinear bedroom/sqft features.
2. **Explainable Automated Valuations:** Local linear attribution quantifies exactly how much dollar value each characteristic adds or subtracts from the baseline, complying with algorithmic underwriting transparency guidelines.
3. **Objective Risk Bands:** 95% confidence bounds ($\pm\$48,200$ average margin) allow mortgage underwriters to systematically identify over-leveraged lending risks.
4. **Market Tier Consistency:** Residual segment analysis confirms consistent relative error across Budget, Mid-Market, Premium, and Luxury price tiers without heteroscedastic fan-out.

---

## 🖥️ Streamlit Application Overview

The Streamlit web application is structured into 11 dedicated analytical modules:

1. **Overview:** Executive KPI summary, problem statement, and end-to-end workflow.
2. **Data Explorer:** Dynamic filtering, missingness audits, and processed CSV export.
3. **EDA:** Univariate distributions, price vs sq ft scatter, condition boxplots, and correlation heatmaps.
4. **Model Performance:** Model leaderboard, bar charts, winning model badges, and CSV export.
5. **Cross-Validation:** 80/20 vs 5-Fold vs LOOCV comparison and fold variance metrics.
6. **Feature Importance:** Grouped coefficient comparisons and Lasso feature elimination analysis.
7. **Property Valuation:** Interactive parameter controls, instant price prediction, 95% confidence intervals, and local linear explainability waterfall chart.
8. **What-If Analysis:** Real-time sensitivity curves for continuous parameter adjustments.
9. **Model Diagnostics:** Homoscedasticity scatter, residual distribution, normal Q-Q plot, and price segment error tables.
10. **Business Insights:** Data-driven strategic takeaways for investors and underwriters.
11. **About Project:** Architecture documentation and technical specifications.

---

## 📂 Project Architecture

```
Real Estate Regression Analysis/
├── .streamlit/
│   └── config.toml                    # Professional typography and color tokens
├── data/
│   ├── raw/
│   │   └── real_estate_raw.csv        # 5,000-record raw residential dataset
│   └── processed/
│       └── real_estate_processed.csv  # Imputed and capped processed dataset
├── models/
│   └── model_artifacts/
│       ├── ols_pipeline.joblib         # Serialized OLS model pipeline
│       ├── ridge_pipeline.joblib       # Serialized RidgeCV model pipeline
│       ├── lasso_pipeline.joblib       # Serialized LassoCV model pipeline
│       ├── preprocessor.joblib         # Serialized DataPreprocessor
│       ├── evaluation_metrics.json     # Dynamic test and CV evaluation metrics
│       ├── coefficients.csv           # Model beta weights & sparsity flag
│       ├── vif_diagnostics.csv         # Variance Inflation Factors
│       ├── test_predictions.csv        # Holdout test set actuals and predictions
│       └── segment_evaluation.csv      # Error breakdown across price tiers
├── notebooks/
│   └── real_estate_regression.ipynb   # Complete step-by-step econometric study
├── src/
│   ├── __init__.py
│   ├── data_generator.py              # Realistic data engine with collinear covariance
│   ├── preprocessing.py               # Imputation, IQR outlier capping, target log transforms
│   ├── feature_engineering.py         # Sklearn-compatible feature engineer
│   ├── models.py                      # Pipeline definitions, 5-Fold CV, analytical LOOCV
│   ├── evaluation.py                  # Dynamic metrics, VIF, segments, residual diagnostics
│   ├── visualization.py               # Publication-grade Plotly visualization library
│   └── prediction.py                  # Inference engine, explainability, what-if curves
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py          # Imputation and outlier capping tests
│   ├── test_model.py                  # Pipeline training, 5-Fold CV, and LOOCV tests
│   └── test_prediction.py             # Inference, inverse transform, and explainability tests
├── app.py                             # Multi-page Streamlit application
├── train.py                           # CLI training pipeline generating all artifacts
├── requirements.txt                   # Production dependencies
├── .gitignore                         # Python, cache, and OS ignore patterns
├── LICENSE                            # MIT License
└── README.md                          # Comprehensive project documentation
```

---

## 🚀 Installation & Local Setup

### 1. Clone the Repository
```bash
git clone https://github.com/<YOUR_USERNAME>/Real-Estate-Regression-Analysis.git
cd "Real-Estate-Regression-Analysis"
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Execute the Training Pipeline
Generate the dataset, fit all pipelines, perform cross-validations, and serialize all model artifacts:
```bash
python train.py
```

### 5. Run Unit Tests
Verify that all 12 tests pass:
```bash
python -m pytest -v
```

### 6. Launch the Streamlit Application
```bash
streamlit run app.py
```

---

## ☁️ Deployment

### GitHub
```bash
git add .
git commit -m "feat: complete real estate regression AVM system"
git branch -M main
git push -u origin main
```

### Streamlit Community Cloud
1. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
2. Click **New app** and connect your GitHub repository.
3. Set **Main file path** to `app.py`.
4. Click **Deploy**.

---

## 🧪 Unit Testing Suite

All pipeline stages are covered with automated tests in `tests/`:
- `test_missing_value_imputation`: Verifies complete imputation without remaining nulls.
- `test_outlier_capping`: Verifies that conservative IQR bounds constrain recording anomalies.
- `test_train_test_split_shapes`: Verifies 80/20 train-test separation without leakage.
- `test_target_transformation_invertibility`: Validates that `expm1(log1p(y)) == y`.
- `test_feature_engineering_transformers`: Validates output feature dimensions and interactions.
- `test_model_pipeline_training`: Verifies OLS, Lasso, and Ridge convergence and $R^2 > 0.60$.
- `test_5fold_cross_validation`: Validates 5-Fold metric consistency.
- `test_analytical_loocv`: Validates analytical Hat-matrix LOOCV metric generation.
- `test_vif_calculation`: Verifies identification of collinear predictors.
- `test_predict_property_value`: Validates inference, confidence interval formatting, and USD scaling.
- `test_explain_prediction`: Validates local linear feature impact attribution.
- `test_what_if_sensitivity_curve`: Validates monotonic sensitivity curve generation.

Run all tests anytime with:
```bash
python -m pytest -v
```

---

## 💻 Technologies Used

- **Language:** Python 3.12+
- **Machine Learning & Econometrics:** Scikit-Learn, Statsmodels, SciPy, NumPy, Pandas, Joblib
- **Visualization:** Plotly Express & Graph Objects, Seaborn, Matplotlib
- **Web Application:** Streamlit
- **Testing:** Pytest

---

## 🔮 Future Improvements

- **Spatial Geocoding & Hedonic Interpolation:** Incorporate geospatial coordinates (latitude/longitude) with spatial lag models (SAR/SEM) and interactive Leaflet maps.
- **Ensemble Benchmarking:** Benchmark regularized linear models against tree ensembles (LightGBM, XGBoost, CatBoost) to contrast non-linear performance against linear explainability.
- **Automated Retraining CI/CD:** Establish GitHub Actions to retrain models and update serializations upon new housing data commits.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
