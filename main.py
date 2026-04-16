"""
================================================================================
RANDOM FOREST MODEL FOR SHEAR DEFORMATION PREDICTION OF
RHA-PTLD BLENDED CONCRETE BEAMS
================================================================================
Project : Modelling the Shear Deformation of Rice Husk Ash and Palm Tree
          Liquid Distillate Blended Concrete Beams
Author  : Adejumo, Adeyemi Ayodeji
================================================================================

METHODOLOGY (Section 3.3.4 of Thesis)
---------------------------------------
The Random Forest (RF) algorithm, an ensemble learning method constructed
from multiple decision trees, is selected for its ability to capture complex,
nonlinear relationships and its robustness against overfitting when trained
on experimental datasets.

Input Features (X):
  1.  rha_pct          — RHA as % cement replacement (0, 5, 10, 15, 20, 25, 30)
  2.  ptld_pct         — PTLD as % water replacement  (0, 1, 2 ... 10)
  3.  fcu_28           — Compressive strength at 28 days (MPa)
  4.  ft_28            — Split tensile strength at 28 days (MPa)
  5.  E_28             — Modulus of elasticity at 28 days (GPa)
  6.  curing_days      — Test age (7 or 28 days)
  7.  a_d_ratio        — Shear span-to-effective depth ratio (a/d)
  8.  applied_load_kN  — Applied point load P (kN)
  9.  span_mm          — Beam span L (mm)
  10. depth_mm         — Beam total depth h (mm)
  11. width_mm         — Beam width b (mm)

Target (y):
  shear_deflection_mm  — Shear component of mid-span deflection (mm)
                         isolated from total LVDT deflection using:
                         w_shear = w_total (LVDT) - w_bending (TBT)

Validation Metrics (Section 3.3.4):
  - Mean Absolute Error  (MAE)
  - Root Mean Square Error (RMSE)
  - Coefficient of Determination (R²)

Comparison (Section 3.3.2):
  RF predictions are compared against:
    (a) Physical LVDT deflection measurements
    (b) Analytical Timoshenko model predictions

NOTE ON DATA
------------
This script is written to receive REAL experimental data from the
laboratory. A synthetic dataset is generated as a placeholder so the
pipeline can be tested end-to-end before lab work is complete.
The section marked "REPLACE WITH REAL DATA" will be replaced with CSV loader.
================================================================================
"""

import os
import json
import warnings
import numpy as np # type: ignore
import pandas as pd # type: ignore
import joblib # type: ignore
from typing import Dict, Any, Tuple

from sklearn.ensemble import RandomForestRegressor # type: ignore
from sklearn.model_selection import train_test_split, RandomizedSearchCV # type: ignore

warnings.filterwarnings("ignore")
np.random.seed(42)

from config import (OUTPUT_DIR, FEATURE_COLS, TARGET_COL, RF_PARAM_DIST, 
                    BEAM_B, BEAM_H, BEAM_L, KAPPA, POISSON_V)
from data import generate_synthetic_dataset, engineer_features
from plots import (plot_predicted_vs_actual, plot_residuals, plot_feature_importance, 
                   plot_shear_contribution, plot_three_way_comparison)
from evaluation import compute_metrics, run_cross_validation

def main() -> Tuple[RandomForestRegressor, pd.DataFrame, Dict[str, Any]]:
    """
    Main orchestration pipeline for generating synthetic target values,
    training a RandomForestRegressor, producing metric tests, and creating comparisons
    through data analytics and visualization.
    
    Returns
    -------
    Tuple[RandomForestRegressor, pd.DataFrame, Dict[str, Any]]: The trained model, populated dataframe properties, and isolated metrics dict context
    """
    print("=" * 70)
    print(" RANDOM FOREST — SHEAR DEFORMATION PREDICTION")
    print(" RHA-PTLD Blended Concrete Beams")
    print("=" * 70)

    # ── 1. Load data ─────────────────────────────────────────────────────
    print("\n[1] Loading dataset...")
    df: pd.DataFrame = generate_synthetic_dataset()

    # ── 2. Feature engineering ───────────────────────────────────────────
    print("\n[2] Engineering features...")
    df = engineer_features(df)
    print(f"  Feature columns : {FEATURE_COLS}")

    X: np.ndarray = df[FEATURE_COLS].values
    y: np.ndarray = df[TARGET_COL].values

    print(f"  X shape : {X.shape}")
    print(f"  y range : {y.min():.5f} – {y.max():.5f} mm")

    # ── 3. Train / test split (80 / 20) ──────────────────────────────────
    print("\n[3] Splitting dataset (80% train / 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    _, df_test = train_test_split(df, test_size=0.20, random_state=42)

    print(f"  Train : {X_train.shape[0]} samples")
    print(f"  Test  : {X_test.shape[0]}  samples")

    # ── 4. Hyperparameter tuning ──────────────────────────────────────────
    print("\n[4] Hyperparameter tuning (RandomizedSearchCV, 5-fold CV)...")
    base_rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    search  = RandomizedSearchCV(
        base_rf,
        param_distributions=RF_PARAM_DIST,
        n_iter=60,
        cv=5,
        scoring="neg_mean_squared_error",
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)
    best_params: Dict[str, Any] = search.best_params_
    print(f"  Best parameters : {best_params}")

    # ── 5. Final model training ───────────────────────────────────────────
    print("\n[5] Training final Random Forest model on full training set...")
    model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    if hasattr(model, "oob_score_"):
        print(f"  OOB R² score : {model.oob_score_:.4f}")

    # ── 6. Predictions ────────────────────────────────────────────────────
    y_pred_train: np.ndarray = model.predict(X_train)
    y_pred_test: np.ndarray  = model.predict(X_test)

    tbt_all: np.ndarray  = df["tbt_prediction_mm"].values - df["w_bending_mm"].values
    tbt_test: np.ndarray = (df_test["tbt_prediction_mm"].values - df_test["w_bending_mm"].values)

    # ── 7. Evaluation metrics ─────────────────────────────────────────────
    print("\n[6] Evaluation Metrics:")
    metrics_rf_train: Dict[str, Any] = compute_metrics(y_train,      y_pred_train, "RF — Train")
    metrics_rf_test: Dict[str, Any]  = compute_metrics(y_test,       y_pred_test,  "RF — Test")
    metrics_tbt_all: Dict[str, Any]  = compute_metrics(y,            tbt_all,      "TBT — All")

    all_metrics = [metrics_rf_train, metrics_rf_test, metrics_tbt_all]
    pd.DataFrame(all_metrics).to_csv(
        os.path.join(OUTPUT_DIR, "evaluation_metrics.csv"), index=False
    )

    # ── 8. Cross-validation ───────────────────────────────────────────────
    print("\n[7] Cross-validation on full dataset...")
    r2_cv, rmse_cv = run_cross_validation(model, X, y, k=5)

    cv_results: Dict[str, Any] = {
        "k_folds"        : 5,
        "R2_mean"        : round(float(r2_cv.mean()),   4),
        "R2_std"         : round(float(r2_cv.std()),    4),
        "RMSE_mean_mm"   : round(float(rmse_cv.mean()), 5),
        "RMSE_std_mm"    : round(float(rmse_cv.std()),  5),
    }
    with open(os.path.join(OUTPUT_DIR, "cv_results.json"), "w") as f:
        json.dump(cv_results, f, indent=2)

    # ── 9. Plots ──────────────────────────────────────────────────────────
    print("\n[8] Generating plots...")
    plot_predicted_vs_actual(
        y_train, y_pred_train,
        y_test,  y_pred_test,
        tbt_all, metrics_rf_test, metrics_tbt_all
    )
    plot_residuals(y_test, y_pred_test)
    plot_feature_importance(model, X_test, y_test, FEATURE_COLS)
    plot_shear_contribution(df)
    plot_three_way_comparison(df_test, y_pred_test)

    # ── 10. Save model & metadata ─────────────────────────────────────────
    print("\n[9] Saving model and metadata...")
    model_path: str = os.path.join(OUTPUT_DIR, "rf_shear_model.joblib")
    joblib.dump(model, model_path)
    print(f"  Model saved → {model_path}")

    metadata: Dict[str, Any] = {
        "model"          : "RandomForestRegressor",
        "n_estimators"   : model.n_estimators,
        "best_params"    : best_params,
        "feature_cols"   : FEATURE_COLS,
        "target_col"     : TARGET_COL,
        "train_samples"  : int(X_train.shape[0]),
        "test_samples"   : int(X_test.shape[0]),
        "metrics_test"   : {k: round(v, 5) if isinstance(v, float) else v for k, v in metrics_rf_test.items()
                            if k != "label"},
        "cv_results"     : cv_results,
        "beam_dims_mm"   : f"{BEAM_B}×{BEAM_H}×{BEAM_L}",
        "kappa_cowper"   : KAPPA,
        "poisson_ratio"  : POISSON_V,
    }
    with open(os.path.join(OUTPUT_DIR, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved → {os.path.join(OUTPUT_DIR, 'model_metadata.json')}")

    # ── 11. Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(" SUMMARY")
    print("=" * 70)
    print(f"  Random Forest  — Test R² : {metrics_rf_test['R2']:.4f} | "
          f"MAE : {metrics_rf_test['MAE']:.4f} mm | "
          f"RMSE : {metrics_rf_test['RMSE']:.4f} mm")
    print(f"  TBT Analytical — All  R² : {metrics_tbt_all['R2']:.4f} | "
          f"MAE : {metrics_tbt_all['MAE']:.4f} mm | "
          f"RMSE : {metrics_tbt_all['RMSE']:.4f} mm")
    print(f"\n  All outputs saved to: ./{OUTPUT_DIR}/")
    print("=" * 70)

    return model, df, metrics_rf_test

def predict_new(model: RandomForestRegressor, rha_pct: float, ptld_pct: float, fcu_MPa: float, ft_MPa: float, E_GPa: float,
                curing_days: int = 28, applied_load_kN: float = 15.0,
                span_mm: float = 600.0, depth_mm: float = 250.0, width_mm: float = 150.0) -> float:
    """
    Predict shear deflection for a single new data point.

    Parameters
    ----------
    model          : RandomForestRegressor - trained RandomForestRegressor (loaded via joblib.load)
    rha_pct        : float - RHA replacement of cement (%)
    ptld_pct       : float - PTLD replacement of water (%)
    fcu_MPa        : float - compressive strength (MPa)
    ft_MPa         : float - split tensile strength (MPa)
    E_GPa          : float - modulus of elasticity (GPa)
    curing_days    : int - 7 or 28
    applied_load_kN: float - point load P (kN)
    span_mm        : float - beam span (mm)
    depth_mm       : float - beam depth (mm)
    width_mm       : float - beam width (mm)

    Returns
    -------
    float : w_shear_mm - predicted shear deflection (mm)
    """
    a_d: float = (span_mm / 3.0) / (depth_mm - 25.0 - 6.0)
    G: float   = E_GPa * 1000.0 / (2.0 * (1.0 + POISSON_V))
    kGA: float = KAPPA * G * (width_mm * depth_mm)
    I: float   = (width_mm * depth_mm ** 3) / 12.0
    EI: float  = E_GPa * 1000.0 * I

    row: pd.DataFrame = pd.DataFrame([{
        "rha_pct"         : rha_pct,
        "ptld_pct"        : ptld_pct,
        "curing_days"     : curing_days,
        "fcu_MPa"         : fcu_MPa,
        "ft_MPa"          : ft_MPa,
        "E_GPa"           : E_GPa,
        "a_d_ratio"       : a_d,
        "applied_load_kN" : applied_load_kN,
        "span_mm"         : span_mm,
        "depth_mm"        : depth_mm,
        "width_mm"        : width_mm,
        "G_MPa"           : G,
        "kGA"             : kGA,
        "EI_over_kGA"     : EI / kGA,
        "fcu_ft"          : fcu_MPa / (ft_MPa + 1e-9),
    }])
    w_shear_mm: float = float(model.predict(row[FEATURE_COLS].values)[0])
    return w_shear_mm

if __name__ == "__main__":
    trained_model, dataset, test_metrics = main()