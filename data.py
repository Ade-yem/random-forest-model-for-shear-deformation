import numpy as np # type: ignore
import pandas as pd # type: ignore
import os
from typing import List, Dict, Any

from config import POISSON_V, KAPPA, BEAM_B, BEAM_H, BEAM_L, SHEAR_SPAN, EFF_D, OUTPUT_DIR
from analytical import predict_fcu, predict_ft, predict_E, cowper_kappa, timoshenko_total_deflection

def generate_synthetic_dataset() -> pd.DataFrame:
    """
    Generate a synthetic experimental dataset using the Scheffe 10 mixes,
    two curing ages (7 and 28 days), and a range of applied loads.
    This mirrors the experimental program in Section 3.1.5.
    Total rows: 10 mixes × 2 ages × 5 load levels = 100 observations.
    
    Returns
    -------
    pd.DataFrame: Synthetically generated dataset representing laboratory measurements
    """
    scheffe_mixes: List[Dict[str, Any]] = [
        {"mix": "M1",  "rha":  0, "ptld":  0},
        {"mix": "M2",  "rha": 30, "ptld":  0},
        {"mix": "M3",  "rha":  0, "ptld": 10},
        {"mix": "M4",  "rha": 15, "ptld":  5},
        {"mix": "M5",  "rha": 15, "ptld":  0},
        {"mix": "M6",  "rha":  0, "ptld":  5},
        {"mix": "M7",  "rha": 10, "ptld":  3},
        {"mix": "M8",  "rha": 20, "ptld":  7},
        {"mix": "M9",  "rha": 30, "ptld": 10},
        {"mix": "M10", "rha": 15, "ptld":  5},
    ]

    curing_ages: List[int]    = [7, 28]
    load_levels_kN: List[float] = [5.0, 10.0, 15.0, 20.0, 25.0]

    records: List[Dict[str, Any]] = []
    for m in scheffe_mixes:
        rha: float = float(m["rha"])
        ptld: float = float(m["ptld"])
        fcu_28: float = predict_fcu(rha, ptld, base_fcu=25.0)
        ft_28: float  = predict_ft(fcu_28)
        E_28: float   = predict_E(fcu_28)

        for age in curing_ages:
            # Mechanical properties at 7-day are ~65% of 28-day (typical gain)
            age_factor: float = 0.65 if age == 7 else 1.0
            fcu_age: float = fcu_28 * age_factor
            ft_age: float  = ft_28  * age_factor
            E_age: float   = E_28   * age_factor

            for P in load_levels_kN:
                a_d: float = SHEAR_SPAN / EFF_D

                # Timoshenko analytical prediction
                nu: float = POISSON_V
                kappa: float = cowper_kappa(nu)
                w_total, w_bend, w_shear = timoshenko_total_deflection(
                    P, E_age, nu, BEAM_B, BEAM_H, BEAM_L, kappa
                )

                # Simulate LVDT measurement: Timoshenko + small random error
                noise: float = float(np.random.normal(0.0, 0.015 * w_total))
                w_lvdt: float = w_total + noise

                records.append({
                    "mix_id"          : m["mix"],
                    "rha_pct"         : rha,
                    "ptld_pct"        : ptld,
                    "curing_days"     : age,
                    "fcu_MPa"         : round(fcu_age,  3),
                    "ft_MPa"          : round(ft_age,   3),
                    "E_GPa"           : round(E_age,    3),
                    "a_d_ratio"       : round(a_d,      4),
                    "applied_load_kN" : P,
                    "span_mm"         : BEAM_L,
                    "depth_mm"        : BEAM_H,
                    "width_mm"        : BEAM_B,
                    "w_total_lvdt_mm" : round(w_lvdt,   5),
                    "w_bending_mm"    : round(w_bend,   5),
                    "w_shear_mm"      : round(w_shear,  5),   # TARGET
                    "tbt_prediction_mm": round(w_total, 5),
                })

    df: pd.DataFrame = pd.DataFrame(records)
    df.to_csv(os.path.join(OUTPUT_DIR, "synthetic_dataset.csv"), index=False)
    print(f"  Dataset shape : {df.shape}")
    print(f"  Mixes         : {df['mix_id'].nunique()}")
    print(f"  Curing ages   : {sorted(df['curing_days'].unique())} days")
    print(f"  Load range    : {df['applied_load_kN'].min()}–{df['applied_load_kN'].max()} kN")
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive additional features from raw inputs that encode structural
    mechanics knowledge (Section 3.2 of thesis).

    Derived features:
      G_MPa    — Shear modulus from isotropic relation G = E/[2(1+ν)]
      kGA      — Timoshenko shear rigidity (Section 3.2.2)
      EI       — Flexural rigidity (Section 3.2.3)
      EI_over_kGA — Ratio governing shear-to-bending deformation split
      fcu_ft   — Compressive-to-tensile strength ratio (brittleness index)
      
    Parameters
    ----------
    df: pd.DataFrame - Raw generated dataframe
    
    Returns
    -------
    pd.DataFrame: Dataframe containing engineered machine learning features
    """
    df = df.copy()
    df["G_MPa"]       = df["E_GPa"] * 1000.0 / (2.0 * (1.0 + POISSON_V))
    df["kGA"]         = KAPPA * df["G_MPa"] * (df["width_mm"] * df["depth_mm"])
    df["I_mm4"]       = (df["width_mm"] * df["depth_mm"] ** 3) / 12.0
    df["EI"]          = df["E_GPa"] * 1000.0 * df["I_mm4"]
    df["EI_over_kGA"] = df["EI"] / df["kGA"]
    df["fcu_ft"]      = df["fcu_MPa"] / df["ft_MPa"]
    return df
