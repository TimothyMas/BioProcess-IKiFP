# bpcomp/custom.py
from __future__ import annotations
import json
from typing import List, Tuple
import streamlit as st
from pydantic import ValidationError
from jsonschema import validate as js_validate, Draft202012Validator
from bpcomp.models import CustomCultureModel
from bpcomp.presets import OrganismPreset

# ---------- Session-backed catalog ----------
def _ensure_state():
    st.session_state.setdefault("_custom_catalog", [CustomCultureModel().model_dump()])
    st.session_state.setdefault("_custom_selected", 0)
    st.session_state.setdefault("_custom_json", json.dumps(st.session_state["_custom_catalog"]))

def get_catalog() -> List[dict]:
    _ensure_state()
    # Ensure entries are valid, coerce on-the-fly
    cleaned = []
    for rec in st.session_state["_custom_catalog"]:
        try:
            m = CustomCultureModel(**rec)
        except ValidationError:
            m = CustomCultureModel()  # fallback safe default
        cleaned.append(m.model_dump())
    st.session_state["_custom_catalog"] = cleaned
    return cleaned

def get_selected_idx() -> int:
    _ensure_state()
    return int(st.session_state["_custom_selected"])

def set_selected(idx: int) -> None:
    _ensure_state()
    cat = get_catalog()
    j = max(0, min(idx, len(cat)-1))
    st.session_state["_custom_selected"] = j

def add_new_culture() -> int:
    _ensure_state()
    cat = get_catalog()
    cat.append(CustomCultureModel().model_dump())
    st.session_state["_custom_catalog"] = cat
    st.session_state["_custom_selected"] = len(cat)-1
    st.session_state["_custom_json"] = json.dumps(cat)
    return st.session_state["_custom_selected"]

def duplicate_culture(idx: int) -> int:
    _ensure_state()
    cat = get_catalog()
    i = max(0, min(idx, len(cat)-1))
    dup = dict(cat[i])
    dup["name"] = f"{dup.get('name','Custom')} (copy)"
    cat.append(dup)
    st.session_state["_custom_catalog"] = cat
    st.session_state["_custom_selected"] = len(cat)-1
    st.session_state["_custom_json"] = json.dumps(cat)
    return st.session_state["_custom_selected"]

def delete_culture(idx: int) -> None:
    _ensure_state()
    cat = get_catalog()
    if len(cat) <= 1:
        return
    i = max(0, min(idx, len(cat)-1))
    del cat[i]
    st.session_state["_custom_catalog"] = cat
    st.session_state["_custom_selected"] = max(0, min(i, len(cat)-1))
    st.session_state["_custom_json"] = json.dumps(cat)

def update_selected(rec: dict) -> None:
    _ensure_state()
    cat = get_catalog()
    i = get_selected_idx()
    # Validate/coerce with Pydantic
    try:
        m = CustomCultureModel(**rec)
    except ValidationError:
        m = CustomCultureModel()
    cat[i] = m.model_dump()
    st.session_state["_custom_catalog"] = cat
    st.session_state["_custom_json"] = json.dumps(cat)

# ---------- Serialization / schema ----------
def export_catalog_json() -> bytes:
    _ensure_state()
    cat = get_catalog()
    return json.dumps(cat, indent=2).encode("utf-8")

def import_catalog_json(data: bytes) -> None:
    _ensure_state()
    try:
        raw = json.loads(data.decode("utf-8"))
        if not isinstance(raw, list):
            raise ValueError("JSON must be a list of cultures.")
        # validate via Pydantic coercion
        cleaned = []
        for rec in raw:
            m = CustomCultureModel(**rec)
            cleaned.append(m.model_dump())
        st.session_state["_custom_catalog"] = cleaned
        st.session_state["_custom_selected"] = 0
        st.session_state["_custom_json"] = json.dumps(cleaned)
    except Exception as e:
        st.error(f"Import failed: {e}")

def schema_json() -> str:
    """Return a JSON Schema for CustomCultureModel."""
    try:
        # Pydantic v2 exports JSON schema
        from pydantic.json_schema import models_json_schema  # lazy import
        sch = CustomCultureModel.model_json_schema()
        return json.dumps(sch, indent=2)
    except Exception:
        # Fallback minimal schema
        sch = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "CustomCultureModel",
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "O2_pref": {"enum": ["aerobic","anaerobic","microaerophilic"]},
                "T_opt": {"type": "number"},
                "pH_opt": {"type": "number"},
                "mu_ref_auto": {"type": "boolean"},
                "K_ref_auto": {"type": "boolean"},
                "lag_ref_auto": {"type": "boolean"},
                "mu_ref": {"type": ["number","null"]},
                "K_ref": {"type": ["number","null"]},
                "lag_ref_h": {"type": ["number","null"]},
                "cfu_per_od": {"type": "number"},
                "orp_mV": {"type": ["number","null"]},
                "electron_acceptor": {"type": "string"},
                "notes": {"type": "string"},
            },
            "required": ["name","O2_pref"]
        }
        return json.dumps(sch, indent=2)

# ---------- Defaults, helpers, and conversion ----------
def default_by_o2(pref: str) -> Tuple[float,float,float,float]:
    p = (pref or "aerobic").lower()
    # rough heuristics
    if p == "anaerobic":
        return (0.35, 4.0, 2.0, 4e8)  # mu, K, lag, cfu/OD
    if p == "microaerophilic":
        return (0.45, 5.0, 1.0, 5e8)
    return (0.9, 7.0, 0.5, 8e8)

def as_float(v, default: float) -> float:
    try:
        if v is None or v == "":
            return float(default)
        return float(v)
    except Exception:
        return float(default)

def record_to_preset(rec: dict) -> OrganismPreset:
    # resolve autos
    mu_d, K_d, lag_d, cfu_d = default_by_o2(rec.get("O2_pref","aerobic"))
    mu = rec.get("mu_ref", None) if not rec.get("mu_ref_auto", True) else mu_d
    K  = rec.get("K_ref", None) if not rec.get("K_ref_auto", True) else K_d
    lag= rec.get("lag_ref_h", None) if not rec.get("lag_ref_auto", True) else lag_d
    mu = as_float(mu, mu_d)
    K = as_float(K, K_d)
    lag = as_float(lag, lag_d)

    name = str(rec.get("name","Custom"))
    T_opt = as_float(rec.get("T_opt"), 30.0)
    pH_opt = as_float(rec.get("pH_opt"), 7.0)
    O2_pref = str(rec.get("O2_pref","aerobic"))
    notes = str(rec.get("notes",""))

    # Attach CFU heuristic for downstream tabs
    preset = OrganismPreset(
        name=name, T_opt=T_opt, pH_opt=pH_opt, O2_pref=O2_pref,
        mu_ref=mu, K_ref=K, lag_ref_h=lag, notes=notes
    )
    # monkey-attach a hint attribute
    setattr(preset, "cfu_per_od", as_float(rec.get("cfu_per_od"), cfu_d))
    return preset

def add_from_preset(p: OrganismPreset) -> int:
    cat = get_catalog()
    rec = {
        "name": f"{p.name} (copy)",
        "O2_pref": p.O2_pref,
        "T_opt": float(p.T_opt),
        "pH_opt": float(p.pH_opt),
        "mu_ref_auto": False, "mu_ref": float(p.mu_ref),
        "K_ref_auto": False,  "K_ref": float(p.K_ref),
        "lag_ref_auto": False, "lag_ref_h": float(p.lag_ref_h),
        "cfu_per_od": 8e8 if "E. coli" in p.name else 4e8,
        "notes": p.notes or "Copied from preset.",
        "orp_mV": None,
        "electron_acceptor": "",
    }
    cat.append(rec)
    st.session_state["_custom_catalog"] = cat
    st.session_state["_custom_selected"] = len(cat)-1
    st.session_state["_custom_json"] = json.dumps(cat)
    return st.session_state["_custom_selected"]
