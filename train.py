"""
End-to-End Training and Serialization Pipeline for Real Estate Regression Analysis.
Executes data generation, leak-free preprocessing, feature engineering,
model training (OLS, LassoCV, RidgeCV), cross-validation (5-Fold, LOOCV),
and serializes artifacts into models/model_artifacts/.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from src.data_generator import generate_real_estate_dataset
from src.preprocessing import (
    DataPreprocessor,
    prepare_train_test_split,
    get_missing_value_summary,
)
from src.models import (
    build_pipeline,
    extract_coefficients,
    run_5fold_cv,
    run_loocv_fast,
)
from src.evaluation import (
    calculate_metrics,
    calculate_vif,
    evaluate_price_segments,
    compute_residual_diagnostics,
)


def run_training_pipeline() -> None:
    print("=" * 70)
    print("  REAL ESTATE REGRESSION TRAINING & SERIALIZATION PIPELINE")
    print("=" * 70)

    # 0. Setup directories
    project_root = Path(__file__).resolve().parent
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    artifacts_dir = project_root / "models" / "model_artifacts"

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate or load dataset
    raw_csv_path = raw_dir / "real_estate_raw.csv"
    if raw_csv_path.exists():
        print(f"\n[1/7] Loading existing raw dataset from: {raw_csv_path}")
        df_raw = pd.read_csv(raw_csv_path)
    else:
        print("\n[1/7] Generating 5,000 realistic residential records...")
        df_raw = generate_real_estate_dataset(n_samples=5000, random_state=42)
        df_raw.to_csv(raw_csv_path, index=False)
        print(f"      Saved raw dataset to: {raw_csv_path} ({len(df_raw)} records)")

    # 2. Inspect missing values and split data
    print("\n[2/7] Preparing train/test partition (80/20) and fitting preprocessor...")
    missing_summary = get_missing_value_summary(df_raw)
    print(f"      Features with missing values before imputation:\n{missing_summary}")

    X_train_raw, X_test_raw, y_train, y_test = prepare_train_test_split(
        df_raw, target_col="property_price", test_size=0.20, random_state=42
    )

    # Preprocess training data (fit on train only to prevent leakage)
    preprocessor = DataPreprocessor(outlier_iqr_multiplier=3.0)
    X_train_preprocessed, train_proc_stats = preprocessor.fit_transform(X_train_raw)
    X_test_preprocessed, test_proc_stats = preprocessor.transform(X_test_raw)

    print(f"      Imputed cells: {train_proc_stats['imputed_cells']} (train), {test_proc_stats['imputed_cells']} (test)")
    print(f"      Capped outlier cells: {train_proc_stats['capped_cells']} (train), {test_proc_stats['capped_cells']} (test)")

    # Save processed full dataset for data explorer & download
    df_processed_full, _ = preprocessor.transform(df_raw.drop(columns=["property_price"]))
    df_processed_full["property_price"] = df_raw["property_price"]
    processed_csv_path = processed_dir / "real_estate_processed.csv"
    df_processed_full.to_csv(processed_csv_path, index=False)
    print(f"      Saved processed dataset to: {processed_csv_path}")

    # 3. Multicollinearity & VIF Diagnostic
    print("\n[3/7] Calculating Multicollinearity and Variance Inflation Factor (VIF)...")
    vif_df = calculate_vif(X_train_preprocessed)
    vif_csv_path = artifacts_dir / "vif_diagnostics.csv"
    vif_df.to_csv(vif_csv_path, index=False)
    print("      Top 5 Features by VIF:")
    print(vif_df.head(5).to_string(index=False))

    corr_sqft_bed = X_train_preprocessed["living_area_sqft"].corr(X_train_preprocessed["bedrooms"])
    print(f"      Correlation living_area_sqft vs bedrooms: {corr_sqft_bed:.3f}")

    # 4. Train Models
    print("\n[4/7] Training OLS, LassoCV (L1), and RidgeCV (L2)...")
    models = {
        "ols": build_pipeline("ols"),
        "lasso": build_pipeline("lasso"),
        "ridge": build_pipeline("ridge"),
    }

    test_predictions = {"actual": y_test.values}
    evaluation_metrics = {}

    for name, pipe in models.items():
        print(f"      Fitting {name.upper()}...")
        pipe.fit(X_train_preprocessed, y_train)

        # Predict on holdout test set
        preds = pipe.predict(X_test_preprocessed)
        test_predictions[f"{name}_pred"] = preds
        metrics = calculate_metrics(y_test.values, preds)

        # Extract optimal alpha if regularized
        reg = pipe.named_steps["regressor"]
        alpha_val = None
        if hasattr(reg, "alpha_"):
            alpha_val = float(reg.alpha_)
            metrics["optimal_alpha"] = round(alpha_val, 4)
        else:
            metrics["optimal_alpha"] = None

        evaluation_metrics[name] = metrics

        # Save individual pipeline artifact
        pipe_path = artifacts_dir / f"{name}_pipeline.joblib"
        joblib.dump(pipe, pipe_path)
        print(f"      Saved {name.upper()} pipeline to: {pipe_path}")

    # Save test predictions DataFrame
    df_preds = pd.DataFrame(test_predictions)
    preds_csv_path = artifacts_dir / "test_predictions.csv"
    df_preds.to_csv(preds_csv_path, index=False)

    # 5. Extract Coefficients & Shrinkage Comparison
    print("\n[5/7] Extracting standardized coefficients and sparsity analysis...")
    df_coef = extract_coefficients(models, X_train_preprocessed)
    coef_csv_path = artifacts_dir / "coefficients.csv"
    df_coef.to_csv(coef_csv_path, index=False)

    zeroed_features = df_coef[df_coef["Is_Zero_In_Lasso"]]["Feature"].tolist()
    print(f"      Features eliminated (zeroed) by Lasso ({len(zeroed_features)}): {zeroed_features}")

    # 6. Cross-Validation (5-Fold & LOOCV)
    print("\n[6/7] Performing Cross-Validation Protocols (5-Fold & LOOCV)...")
    cv_results = {}
    for name, pipe in models.items():
        print(f"      Running 5-Fold CV for {name.upper()}...")
        cv_5 = run_5fold_cv(pipe, X_train_preprocessed, y_train, random_state=42)
        evaluation_metrics[name]["cv_5fold_rmse_mean"] = cv_5["mean_rmse"]
        evaluation_metrics[name]["cv_5fold_rmse_std"] = cv_5["std_rmse"]
        evaluation_metrics[name]["cv_5fold_r2_mean"] = cv_5["mean_r2"]

        print(f"      Running LOOCV for {name.upper()}...")
        loocv_res = run_loocv_fast(pipe, X_train_preprocessed, y_train, sample_limit=500)
        evaluation_metrics[name]["loocv_rmse"] = loocv_res["rmse"]
        evaluation_metrics[name]["loocv_mae"] = loocv_res["mae"]
        evaluation_metrics[name]["loocv_r2"] = loocv_res["r2"]

        cv_results[name] = {
            "5fold": cv_5,
            "loocv": loocv_res,
        }

    # 7. Segment Analysis & Residual Diagnostics for Best Model (Ridge)
    print("\n[7/7] Computing price segment breakdown & residual diagnostics...")
    best_preds = df_preds["ridge_pred"].values
    df_segments = evaluate_price_segments(y_test.values, best_preds)
    segments_csv_path = artifacts_dir / "segment_evaluation.csv"
    df_segments.to_csv(segments_csv_path, index=False)

    res_diag = compute_residual_diagnostics(y_test.values, best_preds)

    # Save master evaluation metrics JSON
    master_metadata = {
        "evaluation_metrics": evaluation_metrics,
        "residual_diagnostics": res_diag,
        "features_zeroed_by_lasso": zeroed_features,
        "vif_top": vif_df.head(10).to_dict(orient="records"),
        "best_model": "ridge" if evaluation_metrics["ridge"]["rmse"] <= evaluation_metrics["ols"]["rmse"] else "ols",
    }

    metrics_json_path = artifacts_dir / "evaluation_metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(master_metadata, f, indent=2)

    # Save preprocessor
    joblib.dump(preprocessor, artifacts_dir / "preprocessor.joblib")

    print(f"\nTraining pipeline completed successfully! Artifacts written to: {artifacts_dir}")
    print("\n" + "=" * 70)
    print("  FINAL EVALUATION LEADERBOARD (TEST SET - 80/20)")
    print("=" * 70)
    summary_table = []
    for name, m in evaluation_metrics.items():
        summary_table.append({
            "Model": name.upper(),
            "RMSE ($10k)": m["rmse"],
            "MAE ($10k)": m["mae"],
            "MSE": m["mse"],
            "R²": m["r2"],
            "5-Fold CV RMSE": f"{m['cv_5fold_rmse_mean']} ± {m['cv_5fold_rmse_std']}",
            "LOOCV RMSE": m["loocv_rmse"],
            "Alpha": m.get("optimal_alpha", "N/A"),
        })
    print(pd.DataFrame(summary_table).to_string(index=False))
    print("=" * 70)


if __name__ == "__main__":
    run_training_pipeline()
