# bpcomp/env_models.py
from __future__ import annotations
from typing import Mapping, Union, Tuple, Dict
import numpy as np

def temp_penalty(T: float, T_opt: float, width: float = 12.0) -> float:
    return float(np.exp(-((T - T_opt)**2) / (2*(width/2.355)**2)))

def pH_penalty(pH: float, pH_opt: float, width: float = 1.6) -> float:
    return float(np.exp(-((pH - pH_opt)**2) / (2*(width/2.355)**2)))

def O2_penalty(pref: str, O2_pct: float) -> float:
    pref = (pref or "").lower()
    if pref == "aerobic":
        return float(np.exp(-((O2_pct-21.0)**2)/(2*(12/2.355)**2)))
    if pref == "anaerobic":
        return float(np.exp(-((O2_pct-0.5)**2)/(2*(3/2.355)**2)))
    return float(np.exp(-((O2_pct-10.0)**2)/(2*(18/2.355)**2)))

def medium_quality_score(carbon_gL: float, N_mM: float, salts_ok: bool, trace_ok: bool, vitamins_ok: bool) -> float:
    score = 0.0
    score += min(carbon_gL/10.0, 1.0)*0.35
    score += min(N_mM/20.0, 1.0)*0.25
    score += (0.1 if salts_ok else 0.0) + (0.15 if trace_ok else 0.0) + (0.15 if vitamins_ok else 0.0)
    return float(min(max(score,0.0),1.0))

def adjust_params_by_environment(
    mu_ref: float,
    K_ref: float,
    lag_ref_h: float,
    env: Mapping[str, Union[float, str]],
) -> Tuple[float, float, float, Dict[str, float]]:

    T = float(env["T"]); pH = float(env["pH"]); O2 = float(env["O2"])
    T_opt = float(env["T_opt"]); pH_opt = float(env["pH_opt"]); O2_pref = str(env["O2_pref"])
    medium_score = float(env["medium_score"])

    tp = temp_penalty(T, T_opt)
    pp = pH_penalty(pH, pH_opt)
    op = O2_penalty(O2_pref, O2)
    mp = medium_score

    fitness = tp * pp * (0.6*op + 0.4) * (0.6*mp + 0.4)
    mu = mu_ref * (0.2 + 0.8*fitness)
    K  = K_ref  * (0.3 + 0.7*(0.5*op + 0.5*mp))
    lag = max(0.05, lag_ref_h * (1.4 - 0.9*fitness))
    return float(mu), float(K), float(lag), dict(tp=tp, pp=pp, op=op, mp=mp, fitness=fitness)

# --- extra, lightweight modifiers for anaerobic workflows ---

def orp_penalty(orp_mv: float, o2_tol_pct: float = 0.5) -> float:
    """
    Penalize growth when redox is too oxidizing for strict anaerobes.
    Simple sigmoid around −200 mV; more negative is better (→1.0).
    """
    s = 1.0 / (1.0 + np.exp(0.02*(orp_mv + 200.0)))  # center ~ -200 mV
    return float(min(1.05, max(0.5, s*1.1)))

def acceptor_factor(acceptor: str) -> float:
    """
    Coarse multiplicative tweak for μ depending on terminal acceptor.
    """
    a = (acceptor or "").lower()
    if "nitrate" in a:  return 1.00
    if "nitrite" in a:  return 0.85
    if "mn" in a:       return 0.95
    return 0.90  # none/unknown

# --- Auto-estimators: used when refs are unknown -------------------------------

def _baseline_mu_for_pref(o2_pref: str) -> float:
    p = (o2_pref or "").lower()
    if p == "aerobic": return 0.9
    if p == "anaerobic": return 0.35
    return 0.55  # microaerophilic / unknown

def estimate_refs_from_env(
    o2_pref: str,
    env: Mapping[str, Union[float, str]],
) -> Tuple[float, float, float]:
    """
    Estimate (mu_ref, K_ref, lag_ref_h) when the user doesn't know them.
    Coarse, conservative heuristics tuned for typical lab cultures.
    """
    T = float(env.get("T", 30.0)); pH = float(env.get("pH", 7.2))
    T_opt = float(env.get("T_opt", 30.0)); pH_opt = float(env.get("pH_opt", 7.2))
    O2 = float(env.get("O2", 0.5)); medium_score = float(env.get("medium_score", 0.6))

    tp = temp_penalty(T, T_opt)
    pp = pH_penalty(pH, pH_opt)
    op = O2_penalty(str(o2_pref), O2)

    # Reference μ scaled by environmental fitness (keep conservative floor/ceil)
    mu0 = _baseline_mu_for_pref(o2_pref)
    fitness = tp * pp * (0.6*op + 0.4) * (0.6*medium_score + 0.4)
    mu_ref = float(max(0.05, min(1.5, mu0 * (0.6 + 0.8*fitness))))

    # K_ref heuristic: base by trophic type & medium quality
    K_base = 7.0 if (o2_pref or "").lower() == "aerobic" else 4.5
    K_ref = float(max(0.5, min(12.0, K_base * (0.3 + 0.9*medium_score))))

    # Lag_ref heuristic: longer if fitness is poor
    lag_ref_h = float(max(0.1, min(6.0, 0.4 + 2.8*(1.0 - fitness))))

    return mu_ref, K_ref, lag_ref_h
