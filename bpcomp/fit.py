# bpcomp/fit.py
from __future__ import annotations
from typing import Tuple
import numpy as np
from numpy.typing import ArrayLike
import math

def linfit(x: ArrayLike, y: ArrayLike, through_zero: bool=False):
    x_arr = np.asarray(x, dtype=float); y_arr = np.asarray(y, dtype=float)
    m = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr, y_arr = x_arr[m], y_arr[m]
    if x_arr.size < 2: raise ValueError("Need at least two points.")
    if through_zero:
        a = float(np.sum(x_arr*y_arr) / np.sum(x_arr*x_arr)); b = 0.0
    else:
        A = np.vstack([x_arr, np.ones_like(x_arr)]).T
        a, b = np.linalg.lstsq(A, y_arr, rcond=None)[0]
    y_hat = a*x_arr + b; resid = y_arr - y_hat
    ss_res = float(np.sum(resid**2)); ss_tot = float(np.sum((y_arr - np.mean(y_arr))**2))
    r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else np.nan
    rmse = math.sqrt(ss_res / x_arr.size)
    return dict(a=float(a), b=float(b), y_hat=y_hat, resid=resid, R2=r2, RMSE=rmse, x=x_arr, y=y_arr)

def quadfit(x: ArrayLike, y: ArrayLike):
    x_arr = np.asarray(x, dtype=float); y_arr = np.asarray(y, dtype=float)
    m = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr, y_arr = x_arr[m], y_arr[m]
    if x_arr.size < 3: raise ValueError("Need at least three points.")
    a, b, c = np.polyfit(x_arr, y_arr, 2)
    y_hat = a*x_arr**2 + b*x_arr + c; resid = y_arr - y_hat
    ss_res = float(np.sum(resid**2)); ss_tot = float(np.sum((y_arr - np.mean(y_arr))**2))
    r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else np.nan
    rmse = math.sqrt(ss_res / x_arr.size)
    return dict(a=float(a), b=float(b), c=float(c), y_hat=y_hat, resid=resid, R2=r2, RMSE=rmse, x=x_arr, y=y_arr)

def invert_linear(yv: float, a: float, b: float) -> float:
    if a == 0: return float("nan")
    return (yv - b) / a

def invert_quadratic(yv: float, a: float, b: float, c: float, xr: Tuple[float,float]) -> float:
    A, B, C = a, b, c - yv
    if abs(A) < 1e-12:
        return (yv - c) / (b if b != 0 else float("nan"))
    disc = B*B - 4*A*C
    if -1e-12 < disc < 0: disc = 0.0
    if disc < 0: return float("nan")
    r1 = (-B + math.sqrt(disc)) / (2*A); r2 = (-B - math.sqrt(disc)) / (2*A)
    lo, hi = xr
    for r in (r1, r2):
        if lo - 0.1*(hi-lo) <= r <= hi + 0.1*(hi-lo):
            return float(r)
    return float(r1 if r1>0 else r2)
