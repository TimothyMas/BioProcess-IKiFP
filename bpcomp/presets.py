# bpcomp/presets.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict

@dataclass
class OrganismPreset:
    name: str
    T_opt: float
    pH_opt: float
    O2_pref: str
    mu_ref: float
    K_ref: float
    lag_ref_h: float
    notes: str

ORGANISMS: Dict[str, OrganismPreset] = {
    "E. coli K-12 (aerobic)": OrganismPreset(
        name="E. coli K-12 (aerobic)",
        T_opt=37.0, pH_opt=7.0, O2_pref="aerobic",
        mu_ref=0.9, K_ref=7.0, lag_ref_h=0.5,
        notes="Fast grower; rich media (e.g., LB/TB) raise K; minimal media lower K and μ."
    ),
    "Aromatoleum evansii (anaerobic)": OrganismPreset(
        name="Aromatoleum evansii (anaerobic)",
        T_opt=30.0, pH_opt=7.2, O2_pref="anaerobic",
        mu_ref=0.35, K_ref=4.0, lag_ref_h=2.0,
        notes="Strictly anaerobic workflows; electron acceptors (e.g., nitrate) can modulate μ."
    ),
}
