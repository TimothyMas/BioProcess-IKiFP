# bpcomp/assay.py
from __future__ import annotations

def beer_lambert_rate_mM_min(mabs_per_min: float, eps_mM_inv_cm: float, path_cm: float) -> float:
    return (mabs_per_min / 1000.0) / (eps_mM_inv_cm * path_cm)

def u_per_ml_stock(rate_mM_min: float, V_rxn_mL: float, V_samp_mL: float, predilution: float) -> float:
    if V_samp_mL == 0: return float("nan")
    return rate_mM_min * (V_rxn_mL / V_samp_mL) * max(1.0, float(predilution))
