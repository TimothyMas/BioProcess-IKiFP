# bpcomp/models.py
from __future__ import annotations
from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator, ConfigDict

O2Pref = Literal["aerobic", "anaerobic", "microaerophilic"]

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

class CustomCultureModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = "Custom culture"
    O2_pref: O2Pref = "aerobic"

    T_opt: float = Field(default=30.0, ge=0.0, le=80.0)
    pH_opt: float = Field(default=7.0, ge=3.0, le=10.0)

    mu_ref_auto: bool = True
    K_ref_auto: bool = True
    lag_ref_auto: bool = True

    mu_ref: Optional[float] = Field(default=None, ge=0.01, le=3.0)
    K_ref: Optional[float] = Field(default=None, ge=0.2, le=15.0)
    lag_ref_h: Optional[float] = Field(default=None, ge=0.0, le=24.0)

    cfu_per_od: float = Field(default=8e8, ge=1e6, le=1e11)

    orp_mV: Optional[float] = None
    electron_acceptor: str = ""
    notes: str = ""

    @field_validator("T_opt", "pH_opt", "mu_ref", "K_ref", "lag_ref_h", "cfu_per_od", mode="before")
    @classmethod
    def coerce_float(cls, v):
        if v is None or v == "":
            return None
        try:
            return float(v)
        except Exception:
            return None

    @field_validator("pH_opt")
    @classmethod
    def clamp_pH(cls, v: float) -> float:
        return _clamp(v, 3.0, 10.0)

    @field_validator("cfu_per_od")
    @classmethod
    def clamp_cfu(cls, v: float) -> float:
        return _clamp(v, 1e6, 1e11)
