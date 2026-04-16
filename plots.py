import matplotlib # type: ignore
matplotlib.use("Agg")
import matplotlib.pyplot as plt # type: ignore
import os
import numpy as np # type: ignore
import pandas as pd # type: ignore
from sklearn.inspection import permutation_importance # type: ignore
from typing import Dict, Any, List
from matplotlib.axes import Axes # type: ignore

from config import OUTPUT_DIR

def _add_metrics_box(ax: Axes, m: Dict[str, Any]) -> None:
    """
    Helper function to append metric string inside a matplotlib plot.
    """
    txt: str = (f"MAE  = {m['MAE']:.4f} mm\n"
           f"RMSE = {m['RMSE']:.4f} mm\n"
           f"R²   = {m['R2']:.4f}\n"
           f"MAPE = {m['MAPE']:.2f} %")
    ax.text(0.04, 0.96, txt, transform=ax.transAxes,
            fontsize=8.5, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                      alpha=0.85, edgecolor="gray"))

def plot_predicted_vs_actual(y_true_train: np.ndarray, y_pred_train: np.ndarray,
                              y_true_test: np.ndarray,  y_pred_test: np.ndarray,
                              tbt_pred: np.ndarray,     metrics_rf: Dict[str, Any], metrics_tbt: Dict[str, Any]) -> None:
    """
    Creates a scatter plot comparison between Random Forest models
    and Timoshenko Beam Theory.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(
        "Predicted vs Measured Shear Deflection\n"
        "RHA–PTLD Blended Concrete Beams (Timoshenko + Random Forest)",
        fontsize=12, fontweight="bold", y=1.01
    )

    ax = axes[0]
    lims: List[float] = [
        float(min(y_true_train.min(), y_true_test.min()) * 0.9),
        float(max(y_true_train.max(), y_true_test.max()) * 1.1),
    ]
    ax.plot(lims, lims, "k--", lw=1.2, label="Perfect fit (1:1)")
    ax.scatter(y_true_train, y_pred_train, c="#2a7ae2", alpha=0.6,
               s=50, edgecolors="white", lw=0.5, label="Train set", zorder=3)
    ax.scatter(y_true_test,  y_pred_test,  c="#e87e2a", alpha=0.8,
               s=60, edgecolors="white", lw=0.5,
               label=f"Test set  (R²={metrics_rf['R2']:.4f})", zorder=4)
    ax.set_xlabel("Measured w_shear (mm) — LVDT",   fontsize=11)
    ax.set_ylabel("Predicted w_shear (mm) — RF", fontsize=11)
    ax.set_title("Random Forest Model", fontsize=11, fontweight="bold")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    _add_metrics_box(ax, metrics_rf)

    ax = axes[1]
    y_true_all: np.ndarray = np.concatenate([y_true_train, y_true_test])
    ax.plot(lims, lims, "k--", lw=1.2, label="Perfect fit (1:1)")
    ax.scatter(y_true_all, tbt_pred, c="#2aae5e", alpha=0.7,
               s=55, edgecolors="white", lw=0.5,
               label=f"TBT analytical  (R²={metrics_tbt['R2']:.4f})", zorder=3)
    ax.set_xlabel("Measured w_shear (mm) — LVDT",    fontsize=11)
    ax.set_ylabel("Predicted w_shear (mm) — TBT", fontsize=11)
    ax.set_title("Timoshenko Beam Theory (Analytical)", fontsize=11, fontweight="bold")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    _add_metrics_box(ax, metrics_tbt)

    plt.tight_layout()
    path: str = os.path.join(OUTPUT_DIR, "predicted_vs_actual.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")

def plot_residuals(y_true_test: np.ndarray, y_pred_test: np.ndarray) -> None:
    """
    Plots the residual analysis of RF model showing spread distribution over tests.
    """
    residuals: np.ndarray = y_true_test - y_pred_test
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Residual Analysis — Random Forest (Test Set)",
                 fontsize=12, fontweight="bold")

    axes[0].axhline(0, color="k", lw=1.2, ls="--")
    axes[0].scatter(y_pred_test, residuals, c="#e87e2a",
                    alpha=0.75, s=55, edgecolors="white", lw=0.5)
    axes[0].set_xlabel("Predicted w_shear (mm)", fontsize=11)
    axes[0].set_ylabel("Residual (Measured − Predicted, mm)", fontsize=11)
    axes[0].set_title("Residuals vs Fitted", fontsize=11, fontweight="bold")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(residuals, bins=12, color="#2a7ae2", edgecolor="white",
                 alpha=0.8, linewidth=0.8)
    axes[1].axvline(0, color="k", lw=1.2, ls="--")
    axes[1].set_xlabel("Residual (mm)", fontsize=11)
    axes[1].set_ylabel("Frequency",     fontsize=11)
    axes[1].set_title("Residual Distribution", fontsize=11, fontweight="bold")
    axes[1].grid(True, alpha=0.3)
    mu: float = float(residuals.mean())
    sd: float = float(residuals.std())
    axes[1].text(0.97, 0.95,
                 f"μ = {mu:.4f}\nσ = {sd:.4f}",
                 transform=axes[1].transAxes, fontsize=9,
                 ha="right", va="top",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                           alpha=0.85, edgecolor="gray"))

    plt.tight_layout()
    path: str = os.path.join(OUTPUT_DIR, "residual_analysis.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")

def plot_feature_importance(model: Any, X_test: np.ndarray, y_test: np.ndarray, feature_names: List[str]) -> None:
    """
    Renders feature importance charts for the trained sklearn model.
    """
    mdi_imp: pd.Series = pd.Series(
        model.feature_importances_, index=feature_names
    ).sort_values(ascending=True)

    perm = permutation_importance(
        model, X_test, y_test, n_repeats=30, random_state=42, n_jobs=-1
    )
    perm_imp: pd.Series = pd.Series(
        perm.importances_mean, index=feature_names
    ).sort_values(ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Feature Importance — Random Forest Shear Deformation Model",
                 fontsize=12, fontweight="bold")

    colors_mdi: List[str]  = ["#2a7ae2" if v > mdi_imp.median() else "#aac8f0"
                   for v in mdi_imp]
    colors_perm: List[str] = ["#e87e2a" if v > perm_imp.median() else "#f5c89a"
                   for v in perm_imp]

    axes[0].barh(mdi_imp.index, mdi_imp.values, color=colors_mdi,
                 edgecolor="white", linewidth=0.5)
    axes[0].set_xlabel("Mean Decrease in Impurity (MDI)", fontsize=11)
    axes[0].set_title("MDI Importance", fontsize=11, fontweight="bold")
    axes[0].grid(True, axis="x", alpha=0.3)

    axes[1].barh(perm_imp.index, perm_imp.values, color=colors_perm,
                 edgecolor="white", linewidth=0.5,
                 xerr=perm.importances_std, error_kw={"ecolor": "gray",
                                                       "capsize": 3})
    axes[1].set_xlabel("Mean Decrease in R² (Permutation)", fontsize=11)
    axes[1].set_title("Permutation Importance (±1 SD)", fontsize=11,
                      fontweight="bold")
    axes[1].grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    path: str = os.path.join(OUTPUT_DIR, "feature_importance.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")

def plot_shear_contribution(df: pd.DataFrame) -> None:
    """
    Contributes to heat map demonstrating percentages of structural loads across features.
    % shear contribution = (w_shear / w_total) × 100
    """
    df_copy: pd.DataFrame = df.copy()
    df_copy["shear_pct"] = (df_copy["w_shear_mm"] / df_copy["w_total_lvdt_mm"]) * 100.0

    pivot: pd.DataFrame = df_copy.groupby(["rha_pct", "ptld_pct"])["shear_pct"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    sc = ax.scatter(
        pivot["rha_pct"], pivot["ptld_pct"],
        c=pivot["shear_pct"], cmap="RdYlGn_r",
        s=180, edgecolors="white", linewidths=0.8, zorder=3
    )
    cbar = plt.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Mean Shear Contribution (%)", fontsize=10)

    for _, row in pivot.iterrows():
        ax.annotate(f"{row['shear_pct']:.1f}%",
                    (row["rha_pct"], row["ptld_pct"]),
                    fontsize=8.5, ha="center", va="bottom",
                    xytext=(0, 8), textcoords="offset points")

    ax.set_xlabel("RHA Replacement (%)", fontsize=11)
    ax.set_ylabel("PTLD Replacement (%)", fontsize=11)
    ax.set_title(
        "Shear Deformation Contribution to Total Deflection\n"
        "% shear = (w_shear / w_total) × 100  [Section 3.3.3]",
        fontsize=11, fontweight="bold"
    )
    ax.grid(True, alpha=0.3)

    path: str = os.path.join(OUTPUT_DIR, "shear_contribution.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")

def plot_three_way_comparison(df_test: pd.DataFrame, y_pred_rf: np.ndarray) -> None:
    """
    Outputs multi-bar comparison showing measured LVDT, vs Random Forest predictions, vs TBT Analytical.
    """
    fig, ax = plt.subplots(figsize=(11, 5))
    x: np.ndarray = np.arange(len(df_test))
    width: float = 0.28

    ax.bar(x - width, df_test["w_shear_mm"].values,
           width, label="LVDT Measured", color="#444", alpha=0.85,
           edgecolor="white")
    ax.bar(x,         df_test["tbt_prediction_mm"].values - df_test["w_bending_mm"].values,
           width, label="TBT Analytical", color="#2a7ae2", alpha=0.85,
           edgecolor="white")
    ax.bar(x + width, y_pred_rf,
           width, label="Random Forest", color="#e87e2a", alpha=0.85,
           edgecolor="white")

    ax.set_xlabel("Test Sample Index", fontsize=11)
    ax.set_ylabel("Shear Deflection w_shear (mm)", fontsize=11)
    ax.set_title(
        "Three-Way Comparison: LVDT Measurement vs TBT vs Random Forest\n"
        "(Test Set — Section 3.3.2)",
        fontsize=11, fontweight="bold"
    )
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path: str = os.path.join(OUTPUT_DIR, "three_way_comparison.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")
