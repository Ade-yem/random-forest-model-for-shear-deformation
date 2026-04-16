"""
TIMOSHENKO ANALYTICAL FUNCTIONS  (Section 3.2.3 of Thesis)
These implement Equation 20 of the thesis, used for:
  (a) generating synthetic target values before real data is available
  (b) the comparison baseline in Section 3.3.2

MECHANICAL PROPERTY MODELS
Literature-calibrated empirical relations for RHA-PTLD concrete.
"""

import numpy as np # type: ignore
from typing import Optional, Tuple
from config import POISSON_V

def shear_modulus(E_GPa: float, nu: float = POISSON_V) -> float:
    """
    Derive shear modulus G from elastic modulus E and Poisson's ratio ν.
    Isotropic elasticity relationship (Section 3.2.1, Equation in thesis):
        G = E / [2(1 + ν)]
        
    Parameters
    ----------
    E_GPa : float — Modulus of elasticity in GPa
    nu    : float — Poisson's ratio
    
    Returns
    -------
    G_MPa : float — Shear modulus in MPa
    """
    E_MPa: float = E_GPa * 1000.0
    G_MPa: float = E_MPa / (2.0 * (1.0 + nu))
    return G_MPa

def cowper_kappa(nu: float) -> float:
    """
    Cowper (1966) shear correction factor for a rectangular section.
    κ = 10(1 + ν) / (12 + 11ν)   — Section 2.4.3 / Equation in Section 3.2.2
    
    Parameters
    ----------
    nu : float — Poisson's ratio
    
    Returns
    -------
    float — shear correction factor
    """
    return 10.0 * (1.0 + nu) / (12.0 + 11.0 * nu)

def second_moment_of_area(b: float, h: float) -> float:
    """
    Second moment of area for rectangular section.
    I = b·h³ / 12
    
    Parameters
    ----------
    b : float — width of the section
    h : float — height of the section
    
    Returns
    -------
    float — second moment of area
    """
    return (b * h ** 3) / 12.0

def timoshenko_shear_deflection(
    P_kN: float, E_GPa: float, nu: float, b: float, h: float, L: float, kappa: Optional[float] = None
) -> float:
    """
    Shear component of mid-span deflection for a simply supported beam under
    two symmetric point loads (four-point bending, load at L/3 and 2L/3).

    From Equation 19 (Thesis Section 3.2.3):
        w_s(x) = (1 / κGA) · [P·x + nL·x/2 - n·x²/2]

    For symmetric four-point loading without self-weight (n ≈ 0 at mid-span):
        w_s(L/2) = P / (κ·G·A) · (L/2 - a/2)
    where a = shear span = L/3.

    Parameters
    ----------
    P_kN  : float - applied point load (kN)
    E_GPa : float - modulus of elasticity (GPa)
    nu    : float - Poisson's ratio
    b     : float - beam width (mm)
    h     : float - beam total depth (mm)
    L     : float - beam span (mm)
    kappa : float - shear correction factor (default: Cowper formula)

    Returns
    -------
    float - shear deflection at mid-span (mm)
    """
    if kappa is None:
        kappa = cowper_kappa(nu)

    P_N: float   = P_kN * 1000.0                  # kN → N
    G_MPa: float = shear_modulus(E_GPa, nu)       # MPa = N/mm²
    A_mm2: float = b * h                          # mm²

    shear_rigidity: float = kappa * G_MPa * A_mm2  # N (kGA in N)

    # Reaction at each support for two equal point loads P at L/3 and 2L/3:
    # R = P (by symmetry; total load = 2P split equally)
    # Shear force in region 0 < x < L/3: V = P
    # w_s at mid-span (from integrating Section 3.2.3 Eq.17–19):
    # w_s = V_at_midspan / (kGA) · (L/2 - a)  simplified for constant shear zone
    # More precisely for this loading:
    # w_s(L/2) = [P * (L/3)] / (kGA)
    a: float = L / 3.0
    w_shear_mm: float = (P_N * a) / shear_rigidity

    return w_shear_mm

def timoshenko_total_deflection(
    P_kN: float, E_GPa: float, nu: float, b: float, h: float, L: float, kappa: Optional[float] = None
) -> Tuple[float, float, float]:
    """
    Total mid-span deflection under symmetric four-point bending.
    Timoshenko Equation 20 (thesis Section 3.2.3):
        w(x) = w_b(x) + w_s(x)

    Bending component at mid-span for two-point load at L/3, 2L/3:
        w_b(L/2) = 23·P·L³ / (1296·EI)

    Parameters
    ----------
    P_kN  : float - applied point load (kN)
    E_GPa : float - modulus of elasticity (GPa)
    nu    : float - Poisson's ratio
    b     : float - beam width (mm)
    h     : float - beam total depth (mm)
    L     : float - beam span (mm)
    kappa : float - shear correction factor (default: Cowper formula)
    
    Returns
    -------
    Tuple[float, float, float] - w_total_mm, w_bending_mm, w_shear_mm in mm
    """
    if kappa is None:
        kappa = cowper_kappa(nu)

    P_N: float    = P_kN * 1000.0
    E_MPa: float  = E_GPa * 1000.0
    I_mm4: float  = second_moment_of_area(b, h)
    EI: float     = E_MPa * I_mm4                 # N·mm²

    # Bending deflection (standard formula for 4-pt loading at third points)
    w_bending_mm: float  = (23.0 * P_N * (L ** 3)) / (1296.0 * EI)
    w_shear_mm: float    = timoshenko_shear_deflection(P_kN, E_GPa, nu, b, h, L, kappa)
    w_total_mm: float    = w_bending_mm + w_shear_mm

    return w_total_mm, w_bending_mm, w_shear_mm

def predict_fcu(rha: float, ptld: float, base_fcu: float = 25.0) -> float:
    """
    Empirical compressive strength model for RHA-PTLD concrete.
    - RHA optimal range 5–15%: pozzolanic gain peaks ~+15%
    - PTLD optimal 5–10%: plasticising gain ~+10%
    - Beyond optimum both decline (Section 2.1.3 / 2.2.3)
    
    Parameters
    ----------
    rha: float - RHA replacement %
    ptld: float - PTLD replacement %
    base_fcu: float - Base compressive strength in MPa
    
    Returns
    -------
    float - Compressive strength in MPa
    """
    rha_effect: float  = 1.0 + 0.15 * np.exp(-0.5 * ((rha - 10.0) / 8.0) ** 2)
    ptld_effect: float = 1.0 + 0.10 * np.exp(-0.5 * ((ptld - 6.0)  / 3.0) ** 2)
    return base_fcu * rha_effect * ptld_effect

def predict_ft(fcu: float) -> float:
    """
    Split tensile strength from compressive strength.
    ft ≈ 0.3 · fcu^(2/3)  (ACI 363 approximation)
    
    Parameters
    ----------
    fcu: float - Compressive strength in MPa
    
    Returns
    -------
    float - Split tensile strength in MPa
    """
    return 0.3 * (fcu ** (2.0 / 3.0))

def predict_E(fcu: float) -> float:
    """
    Static modulus of elasticity from compressive strength.
    E ≈ 9.1 · fcu^(1/3)  GPa  (Pauw-type, suitable for blended concrete)
    
    Parameters
    ----------
    fcu: float - Compressive strength in MPa
    
    Returns
    -------
    float - Static modulus of elasticity in GPa
    """
    return 9.1 * (fcu ** (1.0 / 3.0))
