# bpcomp/utils.py
from __future__ import annotations
import math
import numpy as np

def safe_div(a: float, b: float) -> float:
    return float("nan") if (b is None or not np.isfinite(b) or b == 0) else a / b

def ci95_from_sd(sd: float, n: int) -> float:
    return 1.96 * (sd / math.sqrt(n)) if (n and n > 1 and np.isfinite(sd)) else float("nan")
