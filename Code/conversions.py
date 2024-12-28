import numpy as np
import sympy as sp

def mL_to_micrometers_cubed(mL):
    """
    Convert milliliters to micrometers cubed.
    inputs:
        mL (float): volume in milliliters
    """
    mL_sym = sp.symbols('mL')
    conversion_expr = mL_sym * 1e12
    result = conversion_expr.subs(mL_sym, mL)

    print(f"Conversion: {result} micrometers^3")

    return np.float32(result)

def inverse_mL_to_micrometers_cubed(mL_inv):
    """
    Convert inverse milliliters to inverse micrometers cubed.
    inputs:
        mL_inv (float): volume in inverse milliliters
    """
    mL_inv_sym = sp.symbols('mL_inv')
    conversion_expr = mL_inv_sym * 1e-12
    result = conversion_expr.subs(mL_inv_sym, mL_inv)

    print(f"Conversion: {result} micrometers^-3")

    return np.float32(result)

def convert_D_to_Ps(D, K, d):
    """
    Convert diffusion coefficient to permeability.
    inputs:
        D (float): diffusion coefficient in micrometers squared per second
        K (float): partition coefficient
        d (float): thickness of the membrane in micrometers
    """
    return D * K / d

def convert_Ps_to_D(Ps, K, d):
    """
    Convert permeability to diffusion coefficient.
    inputs:
        Ps (float): permeability in micrometers per second
        K (float): partition coefficient
        d (float): thickness of the membrane in micrometers
    """
    return Ps * d / K