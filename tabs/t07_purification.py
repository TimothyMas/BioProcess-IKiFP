# tabs/t07_purification.py
from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from bpcomp.utils import safe_div
from bpcomp.state import get as get_ns, set as set_ns

def render(container, culture: str):
    with container:
        st.subheader("Purification summary & KPIs")
        known = set()
        calc = get_ns(culture, "assay_calc", pd.DataFrame())
        prot = get_ns(culture, "protein_table", pd.DataFrame())
        if not calc.empty:
            known.update(calc["Sample"].dropna().astype(str).tolist())
        if not prot.empty:
            known.update(prot["Sample"].dropna().astype(str).tolist())
        base = sorted(list(known)) or ["Peak_pool","Endpoint","Supernatant"]

        default_vols = pd.DataFrame({"Fraction": base, "Vol_mL": [8.0, 6.0, 65.0][:len(base)] + ([10.0] * max(0, len(base)-3))})
        if "Role" not in default_vols.columns:
            def _role(n: str) -> str:
                s=str(n)
                return "Load" if ("Supernatant" in s or "Load" in s) else ("Product" if ("Peak" in s or "Endpoint" in s or "Eluate" in s) else ("Waste" if ("Waste" in s or "Flow" in s) else "Other"))
            default_vols["Role"] = [ _role(n) for n in default_vols["Fraction"] ]
        vol_df = st.data_editor(default_vols, num_rows="dynamic", width="stretch", hide_index=True,
                                column_config={"Vol_mL": st.column_config.NumberColumn(min_value=0.01, step=0.01),
                                               "Role": st.column_config.SelectboxColumn(options=["Product","Load","Waste","Other"])})

        if calc is None or calc.empty:
            st.info("Fill activity data first in tab ⑤.")
            return
        if prot is None or prot.empty:
            st.info("Compute protein in tab ⑥ first.")
            return

        act = calc.rename(columns={"Sample":"Fraction"})
        pro = prot.rename(columns={"Sample":"Fraction"})
        merged = vol_df.merge(act[["Fraction","U_per_mL_stock"]], on="Fraction", how="left").merge(pro[["Fraction","Protein_mg_mL"]], on="Fraction", how="left")
        peak_ci = get_ns(culture, "peak_ci", np.nan)
        merged["Total_U"] = merged["U_per_mL_stock"] * merged["Vol_mL"]
        merged["Total_Protein_mg"] = merged["Protein_mg_mL"] * merged["Vol_mL"]
        merged["SA_U_per_mg"] = merged.apply(lambda r: safe_div(float(r["U_per_mL_stock"]), float(r["Protein_mg_mL"])), axis=1)
        merged["U_CI95_pm"] = np.where(merged["Fraction"].astype(str).str.contains("Peak", na=False), peak_ci, np.nan)
        st.dataframe(merged.style.format(precision=4), use_container_width=True)

        load_units = merged.loc[merged["Role"]=="Load", "Total_U"].sum(min_count=1)
        rec_units  = merged.loc[merged["Role"]=="Product", "Total_U"].sum(min_count=1)
        yield_pct = safe_div(rec_units*100.0, load_units)
        sa_prod = merged.loc[merged["Role"]=="Product", "SA_U_per_mg"].replace([np.inf,-np.inf], np.nan).dropna()
        sa_load = merged.loc[merged["Role"]=="Load", "SA_U_per_mg"].replace([np.inf,-np.inf], np.nan).dropna()
        purif_factor = safe_div(float(sa_prod.mean()) if len(sa_prod) else np.nan, float(sa_load.mean()) if len(sa_load) else np.nan)

        g1, g2 = st.columns(2)
        g1.plotly_chart(go.Figure(go.Indicator(mode="number", value=0 if not np.isfinite(yield_pct) else yield_pct, title={"text":"Yield (%)"}, number={"valueformat":".2f"})), use_container_width=True)
        g2.plotly_chart(go.Figure(go.Indicator(mode="number", value=0 if not np.isfinite(purif_factor) else purif_factor, title={"text":"Purification factor (×)"}, number={"valueformat":".2f"})), use_container_width=True)

        set_ns(culture, "purif_full", merged)
        set_ns(culture, "kpis", dict(yield_pct=yield_pct, purif_factor=purif_factor,
                                     peak_mean=get_ns(culture, "peak_mean", np.nan),
                                     peak_sd=get_ns(culture, "peak_sd", np.nan),
                                     peak_cv=get_ns(culture, "peak_cv", np.nan),
                                     peak_ci=peak_ci))
