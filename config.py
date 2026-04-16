"""
================================================================================
RANDOM FOREST MODEL FOR SHEAR DEFORMATION PREDICTION OF
RHA-PTLD BLENDED CONCRETE BEAMS - CONFIGURATIONS
================================================================================
Project : Modelling the Shear Deformation of Rice Husk Ash and Palm Tree
          Liquid Distillate Blended Concrete Beams using Timoshenko Beam
          Theory and Machine Learning
Author  : Adejumo, Adeyemi Ayodeji  (24/405CIEC/092)
Degree  : M.Eng Structural Engineering, University of Abuja
Supervisor : Engr. Dr. K. J. Taku
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
================================================================================
"""

import os
from typing import List, Dict, Any

OUTPUT_DIR: str = "rf_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# BEAM & MATERIAL CONSTANTS (from experimental program, Section 3.1.5)
# Beam dimensions: 150 × 250 × 600 mm
# ─────────────────────────────────────────────────────────────────────────────
BEAM_B: float      = 150.0       # mm  — width
BEAM_H: float      = 150.0       # mm  — total depth
BEAM_L: float      = 450.0       # mm  — span
EFF_D: float       = BEAM_H      # effective depth mm.
KAPPA: float       = 0.8455      # Cowper shear correction factor for ν = 0.20
POISSON_V: float   = 0.20        # Poisson's ratio for standard RHA concrete (Section 3.2.1)
A_SECTION: float   = BEAM_B * BEAM_H # mm²  gross cross-section area

# For four-point bending setup (Figure 3.1/3.2 of thesis):
# Two point loads placed symmetrically; shear span a = L/3
SHEAR_SPAN: float  = BEAM_L / 3.0   # mm

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE COLUMNS (input to the model)
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_COLS: List[str] = [
    "rha_pct",
    "ptld_pct",
    "curing_days",
    "fcu_MPa",
    "ft_MPa",
    "E_GPa",
    "a_d_ratio",
    "applied_load_kN",
    "span_mm",
    "depth_mm",
    "width_mm",
    "G_MPa",
    "kGA",
    "EI_over_kGA",
    "fcu_ft",
]

TARGET_COL: str = "w_shear_mm"

# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETER SEARCH SPACE
# ─────────────────────────────────────────────────────────────────────────────
RF_PARAM_DIST: Dict[str, Any] = {
    "n_estimators"      : [100, 200, 300, 400, 500],
    "max_depth"         : [None, 5, 10, 15, 20],
    "min_samples_split" : [2, 4, 6, 8],
    "min_samples_leaf"  : [1, 2, 3, 4],
    "max_features"      : ["sqrt", "log2", 0.5, 0.75],
    "bootstrap"         : [True],
    "oob_score"         : [True],
}
