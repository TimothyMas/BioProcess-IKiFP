# tabs/t07_purification.py
from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from bpcomp.utils import safe_div, editable_table

ROLE_OPTIONS = ["Product","Load","Waste","Other"]

def render(container, culture_name: str):
    with container:
        st.subheader("Purification summary & KPIs")
        known = set()
        if "assay_calc" in st.session_state and not st.session_state["assay_calc"].empty:
            known.update(st.session_state["assay_calc"]["Sample"].dropna().astype(str).tolist())
        if "protein_table" in st.session_state and not st.session_state["protein_table"].empty:
            known.update(st.session_state["protein_table"]["Sample"].dropna().astype(str).tolist())
        base = sorted(list(known)) or ["Peak_pool","Endpoint","Supernatant"]

        default_vols = pd.DataFrame({"Fraction": base, "Vol_mL": [8.0, 6.0, 65.0][:len(base)] + ([10.0] * max(0, len(base)-3))})
        if "Role" not in default_vols.columns:
            def _role(n: str) -> str:
                s=str(n)
                return "Load" if ("Supernatant" in s or "Load" in s) else ("Product" if ("Peak" in s or "Endpoint" in s or "Eluate" in s) else ("Waste" if ("Waste" in s or "Flow" in s) else "Other"))
            default_vols["Role"] = [ _role(n) for n in default_vols["Fraction"] ]
        vol_df = st.data_editor(default_vols, num_rows="dynamic", width="stretch", hide_index=True,
                                column_config={"Vol_mL": st.column_config.NumberColumn(min_value=0.01, step=0.01),
                                               "Role": st.column_config.SelectboxColumn(options=ROLE_OPTIONS)})

        calc = st.session_state.get("assay_calc"); prot = st.session_state.get("protein_table")
        if calc is None or calc.empty:
            st.info("Fill activity data first in tab ⑤.")
            return
        if prot is None or prot.empty:
            st.info("Compute protein in tab ⑥ first.")
            return

        act = calc.rename(columns={"Sample":"Fraction"})
        pro = prot.rename(columns={"Sample":"Fraction"})
        merged = vol_df.merge(act[["Fraction","U_per_mL_stock"]], on="Fraction", how="left").merge(pro[["Fraction","Protein_mg_mL"]], on="Fraction", how="left")
        peak_ci = st.session_state.get("peak_ci", np.nan)
        merged["Total_U"] = merged["U_per_mL_stock"] * merged["Vol_mL"]
        merged["Total_Protein_mg"] = merged["Protein_mg_mL"] * merged["Vol_mL"]
        merged["SA_U_per_mg"] = merged.apply(lambda r: safe_div(float(r["U_per_mL_stock"]), float(r["Protein_mg_mL"])), axis=1)
        merged["U_CI95_pm"] = np.where(merged["Fraction"].astype(str).str.contains("Peak", na=False), peak_ci, np.nan)

        st.markdown("**Purification table (fully editable; add rows/cols)**")
        merged_edit = editable_table(
            merged,
            key=f"{culture_name}::purif_full",
            allow_add_cols=True,
            column_config={
                "Role": st.column_config.SelectboxColumn(options=ROLE_OPTIONS),
            },
            helptext="You can add missing fractions, rename, change roles, or add notes/IDs."
        )

        # Recompute KPIs from the possibly-edited table
        load_units = merged_edit.loc[merged_edit["Role"]=="Load", "Total_U"].sum(min_count=1)
        rec_units  = merged_edit.loc[merged_edit["Role"]=="Product", "Total_U"].sum(min_count=1)
        yield_pct = safe_div(rec_units*100.0, load_units)
        sa_prod = merged_edit.loc[merged_edit["Role"]=="Product", "SA_U_per_mg"].replace([np.inf,-np.inf], np.nan).dropna()
        sa_load = merged_edit.loc[merged_edit["Role"]=="Load", "SA_U_per_mg"].replace([np.inf,-np.inf], np.nan).dropna()
        purif_factor = safe_div(float(sa_prod.mean()) if len(sa_prod) else np.nan, float(sa_load.mean()) if len(sa_load) else np.nan)

        g1, g2 = st.columns(2)
        g1.plotly_chart(go.Figure(go.Indicator(mode="number", value=0 if not np.isfinite(yield_pct) else yield_pct, title={"text":"Yield (%)"}, number={"valueformat":".2f"})), use_container_width=True)
        g2.plotly_chart(go.Figure(go.Indicator(mode="number", value=0 if not np.isfinite(purif_factor) else purif_factor, title={"text":"Purification factor (×)"}, number={"valueformat":".2f"})), use_container_width=True)

        st.session_state["purif_full"] = merged_edit
        st.session_state["kpis"] = dict(yield_pct=yield_pct, purif_factor=purif_factor,
                                        peak_mean=st.session_state.get("peak_mean", np.nan),
                                        peak_sd=st.session_state.get("peak_sd", np.nan),
                                        peak_cv=st.session_state.get("peak_cv", np.nan),
                                        peak_ci=peak_ci)
