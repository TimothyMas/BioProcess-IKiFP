# tabs/t05_activity.py
from __future__ import annotations
import pandas as pd
import numpy as np
import streamlit as st
from bpcomp.assay import beer_lambert_rate_mM_min, u_per_ml_stock
from bpcomp.utils import ci95_from_sd
from bpcomp.state import set as set_ns

def render(container, culture: str, epsilon: float, path_cm: float, V_rxn_mL: float, V_samp_mL: float):
    with container:
        st.subheader("Activity assay (slopes in mABS/min)")
        st.markdown("Enter **background** and **after** slopes. **PreDilution** = dilution of the stock **before** pipetting.")
        default_act = pd.DataFrame({
            "Sample": ["Peak_1","Peak_2","Peak_3","Endpoint","Supernatant"],
            "Background_mABS_min": [0.56, 0.62, 1.31, 0.66, 10.46],
            "ValueAfter_mABS_min": [237.47, 246.41, 227.87, 50.69, 125.31],
            "PreDilution": [5,5,5,1,15]
        })
        up = st.file_uploader("Optional: upload activity CSV/XLSX", type=["csv","xlsx"], key="up_act")
        if up is not None:
            try:
                act_df = pd.read_csv(up) if up.name.endswith(".csv") else pd.read_excel(up)
            except Exception as e:
                st.error(f"Could not read file: {e}"); act_df = default_act.copy()
        else:
            act_df = default_act.copy()

        act_df = st.data_editor(act_df, num_rows="dynamic", width="stretch", hide_index=True,
                                column_config={
                                    "Background_mABS_min": st.column_config.NumberColumn(format="%.5f"),
                                    "ValueAfter_mABS_min": st.column_config.NumberColumn(format="%.5f"),
                                    "PreDilution": st.column_config.NumberColumn(min_value=1, step=1),
                                })
        calc = act_df.copy()
        calc["Final_mABS_min"] = calc["ValueAfter_mABS_min"] - calc["Background_mABS_min"]
        calc["Rate_mM_min"] = calc["Final_mABS_min"].apply(lambda s: beer_lambert_rate_mM_min(float(s), float(epsilon), float(path_cm)))
        calc["U_per_mL_stock"] = calc.apply(lambda r: u_per_ml_stock(float(r["Rate_mM_min"]), float(V_rxn_mL), float(V_samp_mL), max(1, float(r["PreDilution"]))), axis=1)
        st.markdown("**Calculated activities (per mL of stock):**")
        st.dataframe(calc.style.format(precision=5), use_container_width=True)
        set_ns(culture, "assay_calc", calc)

        peak_vals = (calc.loc[calc["Sample"].astype(str).str.contains("Peak", na=False), "U_per_mL_stock"]
                        .dropna().to_numpy(dtype=float))
        peak_mean = float(np.mean(peak_vals)) if peak_vals.size else float("nan")
        peak_sd   = float(np.std(peak_vals, ddof=1)) if peak_vals.size > 1 else float("nan")
        peak_cv   = (peak_sd/peak_mean*100.0) if (peak_vals.size>1 and peak_mean>0) else float("nan")
        peak_ci   = ci95_from_sd(peak_sd, len(peak_vals)) if peak_vals.size > 1 else float("nan")

        set_ns(culture, "peak_mean", peak_mean)
        set_ns(culture, "peak_sd", peak_sd)
        set_ns(culture, "peak_cv", peak_cv)
        set_ns(culture, "peak_ci", peak_ci)

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<div class="metric-card"><b>Peak mean</b><div class="tight">{(0 if np.isnan(peak_mean) else peak_mean):.3f} U/mL</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><b>SD</b><div class="tight">{(0 if np.isnan(peak_sd) else peak_sd):.3f} U/mL</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card"><b>CV%</b><div class="tight">{(0 if np.isnan(peak_cv) else peak_cv):.2f}%</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="metric-card"><b>95% CI ±</b><div class="tight">{(0 if np.isnan(peak_ci) else peak_ci):.3f} U/mL</div></div>', unsafe_allow_html=True)
