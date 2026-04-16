# Modelling Shear Deformation of RHA-PTLD Blended Concrete Beams

**Author:** Adejumo, Adeyemi Ayodeji



---

## Project Overview
This repository contains the implementation of a **Random Forest (RF)** machine learning model designed to predict the shear deformation component of mid-span deflection in Rice Husk Ash (RHA) and Palm Tree Liquid Distillate (PTLD) blended concrete beams. 

The research leverages **Timoshenko Beam Theory (TBT)** to isolate shear deformation from experimental total deflection measurements and utilizes ensemble learning to capture the complex, non-linear relationships between mix proportions, curing age, beam geometry, and mechanical properties.

## Methodology
The prediction pipeline follows the steps outlined in Section 3.3.4 of the thesis:
1.  **Analytical Baseline:** Using Cowper's shear correction factors and isotropic elasticity to define Timoshenko parameters.
2.  **Dataset Preparation:** Processing experimental data (RHA/PTLD percentages, compressive/tensile strengths, span-depth ratios).
3.  **Feature Engineering:** Deriving advanced mechanical features like shear rigidity ($kGA$), flexural rigidity ($EI$), and brittleness indices.
4.  **Model Optimization:** Hyperparameter tuning of the Random Forest algorithm using `RandomizedSearchCV` with 5-fold cross-validation.
5.  **Multi-Way Validation:** Comparing RF outcomes against both laboratory LVDT measurements and analytical Timoshenko models.

## Repository Structure
The project is modularized for maintainability and academic clarity:

- `main.py`: The primary orchestration script for model training and evaluation.
- `config.py`: Centralized configuration for beam constants, material properties, and ML parameters.
- `analytical.py`: Implementation of Timoshenko Beam Theory equations and empirical material models.
- `data.py`: Modules for synthetic data generation and mechanical feature engineering.
- `evaluation.py`: Performance measurement suite (MAE, RMSE, R², MAPE) and cross-validation runners.
- `plots.py`: Visualization engine for residual analysis, feature importance, and model comparisons.

## Input Features
The model utilizes 11 primary inputs:
- `rha_pct`: RHA as % cement replacement (0-30%)
- `ptld_pct`: PTLD as % water replacement (0-10%)
- `fcu_28`: Compressive strength at 28 days (MPa)
- `ft_28`: Split tensile strength at 28 days (MPa)
- `E_28`: Modulus of elasticity at 28 days (GPa)
- `curing_days`: Test age (7 or 28 days)
- `a_d_ratio`: Shear span-to-effective depth ratio
- `applied_load_kN`: Applied point load P (kN)
- `span_mm`, `depth_mm`, `width_mm`: Beam geometry

## Getting Started

### Prerequisites
- Python 3.8+
- Required libraries: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `joblib`

### Installation
```bash
git clone <repository-url>
cd masters
pip install -r requirements.txt
```

### Running the Model
To execute the full training, evaluation, and plotting pipeline:
```bash
python3 main.py
```

Outputs will be generated in the `rf_outputs/` directory, including:
- Trained model (`rf_shear_model.joblib`)
- Performance metrics (`evaluation_metrics.csv`)
- Visualization plots (`predicted_vs_actual.png`, `feature_importance.png`, etc.)

## Citation
If using this work for structural engineering research, please refer to the primary thesis:
> Adejumo, A. A. (2026). *Modelling the Shear Deformation of Rice Husk Ash and Palm Tree Liquid Distillate Blended Concrete Beams using Timoshenko Beam Theory and Machine Learning*. M.Eng Thesis, University of Abuja.
