# bpcomp/utils.py
from __future__ import annotations
import math
from typing import Optional, Any, Dict
import numpy as np
import pandas as pd
import streamlit as st

def safe_div(a: float, b: float) -> float:
    return float("nan") if (b is None or not np.isfinite(b) or b == 0) else a / b

def ci95_from_sd(sd: float, n: int) -> float:
    return 1.96 * (sd / math.sqrt(n)) if (n and n > 1 and np.isfinite(sd)) else float("nan")

def as_float(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)

# ---------- NEW: generic editable DataFrame with add/remove rows + add columns ----------
def editable_table(
    df_in: pd.DataFrame,
    key: str,
    *,
    allow_add_cols: bool = True,
    column_defaults: Optional[Dict[str, Any]] = None,
    column_config: Optional[Dict[str, Any]] = None,
    helptext: Optional[str] = None,
) -> pd.DataFrame:
    """
    Renders a table with:
      - add/remove rows (num_rows='dynamic')
      - optional “Add column” UI (applies to a working copy, not original)
      - optional per-column config and defaults for new rows

    Returns the edited DataFrame.
    """
    df = df_in.copy()
    if helptext:
        st.caption(helptext)

    # --- Add column UI
    if allow_add_cols:
        with st.expander("➕ Add column", expanded=False):
            nc1, nc2 = st.columns([2, 1])
            new_col = nc1.text_input("New column name", key=f"{key}::newcol_name", placeholder="e.g., note, replicate_id")
            default_val = nc2.text_input("Default value", key=f"{key}::newcol_default", placeholder="(optional)")
            cadd = st.button("Add column", key=f"{key}::btn_addcol")
            if cadd:
                if new_col and new_col not in df.columns:
                    # Try to infer numeric default, fall back to string
                    dv: Any
                    try:
                        dv = float(default_val)
                    except Exception:
                        dv = default_val if default_val != "" else np.nan
                    df[new_col] = dv
                else:
                    st.info("Provide a unique column name.")

    # --- Build defaults for new rows
    if column_defaults:
        # Streamlit will fill with NaN; we post-fill after edit to keep UX clean
        pass

    edited = st.data_editor(
        df,
        key=f"{key}::editor",
        hide_index=True,
        num_rows="dynamic",
        use_container_width=True,
        column_config=column_config or {},
    )

    # Apply defaults for any new rows (where columns are missing/NaN)
    if column_defaults:
        for col, dv in column_defaults.items():
            if col in edited.columns:
                edited[col] = edited[col].fillna(dv)

    return edited
