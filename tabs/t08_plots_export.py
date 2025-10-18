# tabs/t08_plots_export.py
from __future__ import annotations
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from bpcomp.state import get as get_ns

def render(container, culture: str):
    with container:
        st.subheader("Plots & Export")
        purif_full = get_ns(culture, "purif_full", pd.DataFrame()); kpis = get_ns(culture, "kpis", {})
        if purif_full.empty:
            st.info("Complete previous tabs to enable plots and export.")
            return

        oc1, oc2, oc3, oc4 = st.columns(4)
        chart_type = oc1.selectbox("Chart type", ["Bar", "Line"])
        order = oc2.selectbox("Order rows by", ["Original", "Specific activity (desc)", "Total units (desc)"])
        y_scale = oc3.selectbox("Y-scale", ["Linear", "Log"])
        show_grid = oc4.toggle("Gridlines", value=True)
        oc5, oc6 = st.columns(2)
        markers = oc5.toggle("Show markers (line)", value=True)
        opacity = oc6.slider("Opacity", 0.2, 1.0, 0.9, 0.05)
        roles_to_plot = st.multiselect("Include roles", ["Product","Load","Waste","Other"], default=["Product","Load"])

        plot_df = purif_full[purif_full["Role"].isin(roles_to_plot)].copy()
        if order == "Specific activity (desc)":
            plot_df = plot_df.sort_values("SA_U_per_mg", ascending=False, na_position="last")
        elif order == "Total units (desc)":
            plot_df = plot_df.sort_values("Total_U", ascending=False, na_position="last")

        if chart_type == "Bar":
            fig_sa = go.Figure(); fig_sa.add_bar(x=plot_df["Fraction"], y=plot_df["SA_U_per_mg"], name="Specific activity (U/mg)", opacity=opacity)
        else:
            fig_sa = go.Figure(); fig_sa.add_trace(go.Scatter(x=plot_df["Fraction"], y=plot_df["SA_U_per_mg"], mode="lines+markers" if markers else "lines", name="Specific activity (U/mg)", opacity=opacity))
        fig_sa.update_layout(title="Specific activity by fraction", yaxis_title="U/mg", xaxis_title="Fraction", template="plotly_white")
        if y_scale == "Log": fig_sa.update_yaxes(type="log", dtick=1)
        fig_sa.update_xaxes(showgrid=show_grid); fig_sa.update_yaxes(showgrid=show_grid)
        st.plotly_chart(fig_sa, use_container_width=True)

        if chart_type == "Bar":
            fig_units = px.bar(plot_df, x="Fraction", y="Total_U", title="Total Units by fraction", labels={"Total_U":"Units (U)"}, opacity=opacity, template="plotly_white")
        else:
            fig_units = go.Figure(); fig_units.add_trace(go.Scatter(x=plot_df["Fraction"], y=plot_df["Total_U"], mode="lines+markers" if markers else "lines", name="Total Units (U)", opacity=opacity))
            fig_units.update_layout(title="Total Units by fraction", xaxis_title="Fraction", yaxis_title="Units (U)", template="plotly_white")
        if y_scale == "Log": fig_units.update_yaxes(type="log", dtick=1)
        fig_units.update_xaxes(showgrid=show_grid); fig_units.update_yaxes(showgrid=show_grid)
        st.plotly_chart(fig_units, use_container_width=True)

        bio = io.BytesIO()
        with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
            (get_ns(culture, "sim_df", pd.DataFrame())).to_excel(writer, sheet_name="Simulated_Growth", index=False)
            (get_ns(culture, "growth_df", pd.DataFrame())).to_excel(writer, sheet_name="Growth_OD600", index=False)
            (get_ns(culture, "assay_calc", pd.DataFrame())).to_excel(writer, sheet_name="Assay", index=False)
            (get_ns(culture, "protein_table", pd.DataFrame())).to_excel(writer, sheet_name="Protein", index=False)
            plot_df.to_excel(writer, sheet_name="Purification", index=False)
            pd.DataFrame({
                "Metric":[ "Yield (%)", "Purification factor (×)", "Peak mean (U/mL)","Peak SD","Peak CV (%)","Peak 95% CI ± (U/mL)" ],
                "Value":[ kpis.get("yield_pct", np.nan), kpis.get("purif_factor", np.nan),
                          kpis.get("peak_mean", np.nan), kpis.get("peak_sd", np.nan), kpis.get("peak_cv", np.nan), kpis.get("peak_ci", np.nan) ]
            }).to_excel(writer, sheet_name="KPIs", index=False)
        st.download_button("⬇️ Download run bundle (Excel)", data=bio.getvalue(), file_name=f"bioprocess_run_bundle_{culture.replace(' ','_')}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
