# bpcomp/state.py
from __future__ import annotations
import streamlit as st
from typing import Any, List

# Per-culture namespacing of Streamlit session_state keys
BASE_KEYS = [
    "plan", "growth_df", "assay_calc", "bradford_fit", "protein_table", "purif_full", "kpis", "sim_df",
    "peak_mean", "peak_sd", "peak_cv", "peak_ci", "OD_start",
    "organism", "env", "constants"
]

def ns_key(culture: str, key: str) -> str:
    culture = culture or "Culture 1"
    return f"{culture}::{key}"

def get(culture: str, key: str, default: Any = None) -> Any:
    return st.session_state.get(ns_key(culture, key), default)

def set(culture: str, key: str, value: Any) -> None:
    st.session_state[ns_key(culture, key)] = value

def ensure_cultures() -> List[str]:
    if "cultures" not in st.session_state or not isinstance(st.session_state["cultures"], list) or not st.session_state["cultures"]:
        st.session_state["cultures"] = ["Culture 1"]
    return st.session_state["cultures"]

def add_culture(name: str) -> None:
    ensure_cultures()
    if name and name not in st.session_state["cultures"]:
        st.session_state["cultures"].append(name)

def delete_culture(name: str) -> None:
    ensure_cultures()
    if name in st.session_state["cultures"] and len(st.session_state["cultures"]) > 1:
        pref = f"{name}::"
        for k in list(st.session_state.keys()):
            if isinstance(k, str) and k.startswith(pref):
                del st.session_state[k]
        st.session_state["cultures"] = [c for c in st.session_state["cultures"] if c != name]
        if st.session_state.get("current_culture") == name:
            st.session_state["current_culture"] = st.session_state["cultures"][0]

def rename_culture(old: str, new: str) -> None:
    ensure_cultures()
    if not new or new in st.session_state["cultures"]:
        return
    for k in list(st.session_state.keys()):
        if isinstance(k, str) and k.startswith(f"{old}::"):
            suffix = k.split("::", 1)[1]
            st.session_state[f"{new}::{suffix}"] = st.session_state[k]
            del st.session_state[k]
    st.session_state["cultures"] = [new if c == old else c for c in st.session_state["cultures"]]

def clear_keys(culture: str | None = None):
    """Clear stored state. If culture is None, clear all cultures' keys (but keep culture list)."""
    if culture is None:
        for c in list(ensure_cultures()):
            clear_keys(c)
        return
    for k in BASE_KEYS:
        nk = ns_key(culture, k)
        if nk in st.session_state:
            del st.session_state[nk]
