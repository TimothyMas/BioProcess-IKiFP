# bpcomp/growth.py
from __future__ import annotations
from typing import Tuple, List
import numpy as np
from numpy.typing import ArrayLike, NDArray

def simulate_growth(time_h: ArrayLike, y0: float, model: str,
                    mu: float, K: float, lag_h: float, death_rate: float=0.0) -> NDArray[np.float64]:
    t = np.asarray(time_h, dtype=float)
    y = np.zeros_like(t, dtype=float)
    s = 1/(1+np.exp(-(t - lag_h)*3.5))
    if model.lower().startswith("exp"):
        lnC = np.log(max(y0,1e-6)); g = np.exp(lnC + mu*(t - lag_h)); y = (1-s)*y0 + s*g
    elif model.lower().startswith("logistic"):
        A = (K/max(y0,1e-9)) - 1.0; g = K/(1 + A*np.exp(-mu*(t - lag_h))); y = (1-s)*y0 + s*g
    else:
        A = 5.0; g = K*np.exp(-A*np.exp(-mu*(t - lag_h))); y = (1-s)*y0 + s*g

    if death_rate and death_rate>0:
        try:
            t_stat = float(lag_h + max(0.0, np.log(max(K,1e-6)/max(y0,1e-6))) / max(mu,1e-6))
        except Exception:
            t_stat = float(lag_h + 4.0)
        decay = np.exp(-death_rate*np.maximum(0.0, t - t_stat))
        y = y * decay
    return np.maximum(y, 1e-6)

def od_to_cfu(od, scale: float=8e8):
    return np.asarray(od, dtype=float) * float(scale)

def _linreg(x: ArrayLike, y: ArrayLike) -> Tuple[float, float]:
    x_arr = np.asarray(x, dtype=float); y_arr = np.asarray(y, dtype=float)
    A = np.vstack([x_arr, np.ones_like(x_arr)]).T
    a, b = np.linalg.lstsq(A, y_arr, rcond=None)[0]
    return float(a), float(b)

def rolling_mu_exponential(t: ArrayLike, y: ArrayLike, wpts: int=3):
    t_arr = np.asarray(t, dtype=float); y_arr = np.asarray(y, dtype=float)
    m = np.isfinite(t_arr) & np.isfinite(y_arr) & (y_arr > 0)
    t_arr, y_arr = t_arr[m], y_arr[m]
    if t_arr.size < wpts:
        return np.array([], dtype=float), np.array([], dtype=float)
    tc: list[float] = []; mu: list[float] = []
    for i in range(len(t_arr) - wpts + 1):
        ti = t_arr[i:i+wpts]; yi = y_arr[i:i+wpts]
        a, _ = _linreg(ti, np.log(yi))
        tc.append(float(np.mean(ti))); mu.append(float(a))
    return np.asarray(tc, dtype=float), np.asarray(mu, dtype=float)

def predict_time_to_target_exp(t_now: float, od_now: float, od_target: float, mu: float):
    import numpy as np
    if not (np.isfinite(mu) and mu>0 and np.isfinite(od_now) and od_now>0 and np.isfinite(od_target) and od_target>od_now):
        return (float("nan"), float("nan"))
    delta_h = np.log(od_target/od_now)/mu
    return (t_now + delta_h, delta_h)
