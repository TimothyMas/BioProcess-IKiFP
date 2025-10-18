# tabs/t02_simulation.py
from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
from bpcomp.growth import simulate_growth, od_to_cfu, rolling_mu_exponential
from bpcomp.plots import growth_plot, cfu_plot
from bpcomp.state import get as get_ns, set as set_ns

def _estimate_from_table(growth_df: pd.DataFrame):
    g = growth_df[growth_df["valid"]].copy()
    if g.empty: return None
    rep = st.selectbox("Replicate to use", sorted(g["replicate"].astype(str).unique().tolist()))
    gg = g[g["replicate"].astype(str) == str(rep)].sort_values("time_h")

    y0 = float(gg.iloc[0].get("od600_smooth", gg.iloc[0]["od600"]))
    tc, mu_series = rolling_mu_exponential(
        gg["time_h"].to_numpy(float),
        gg.get("od600_smooth", gg["od600"]).to_numpy(float),
        wpts=3
    )
    mu_est = float(np.nanmedian(mu_series[:max(3, len(mu_series)//2)])) if mu_series.size else np.nan
    K_est = float(np.nanmax(gg.get("od600_smooth", gg["od600"]))) * 1.1
    try:
        lag_est = float(gg.loc[gg.get("od600_smooth", gg["od600"]) > y0 * 1.2, "time_h"].min())
    except Exception:
        lag_est = 0.5
    return dict(y0=y0, mu=mu_est, K=K_est, lag=max(0.0, lag_est))

def render(container, culture: str, mu_eff: float, K_eff: float, lag_eff: float, org, epsilon: float, path_cm: float):
    with container:
        st.subheader("Growth simulation across phases")
        s1, s2, s3, s4 = st.columns(4)
        sim_model = s1.selectbox("Model", ["Logistic (default)", "Gompertz", "Exponential"], index=0)
        sim_hours = s2.number_input("Total time (h)", value=18.0, min_value=1.0, step=1.0)
        dt = s3.number_input("Sampling interval (h)", value=0.25, min_value=0.05, step=0.05, format="%.2f")
        death_rate = s4.number_input("Death rate after stationary (1/h)", value=0.00, min_value=0.0, step=0.01)

        mode = st.radio("Parameter source", ["Manual sliders", "Use monitoring table"], horizontal=True)

        if mode == "Manual sliders":
            mu_sim = st.slider("Max μ (ln/h)", 0.05, 2.0, float(mu_eff), 0.01)
            K_sim  = st.slider("K (OD)", 0.5, 12.0, float(K_eff), 0.1)
            lag_sim = st.slider("Lag phase (h)", 0.0, 8.0, float(lag_eff), 0.1)
            y0 = st.number_input("Initial OD600", value=float(get_ns(culture, "OD_start", 0.05)),
                                 min_value=0.001, step=0.001, format="%.3f")
        else:
            growth_df = get_ns(culture, "growth_df", pd.DataFrame())
            if growth_df is None or growth_df.empty:
                st.info("No monitoring table found yet. Switch to 'Growth monitoring' tab and add/edit data.")
                return
            est = _estimate_from_table(growth_df)
            if est is None:
                st.info("No valid rows in the monitoring table.")
                return
            c1, c2, c3, c4 = st.columns(4)
            y0   = c1.number_input("Initial OD600 (from table)", value=float(est["y0"]), min_value=1e-6, step=0.001, format="%.3f")
            mu_sim = c2.number_input("μ (ln/h, estimated)", value=float(est["mu"]) if np.isfinite(est["mu"]) else float(mu_eff), min_value=0.01, step=0.01)
            K_sim  = c3.number_input("K (OD, estimated)", value=float(est["K"]) if np.isfinite(est["K"]) else float(K_eff), min_value=0.2, step=0.1)
            lag_sim = c4.number_input("Lag (h, estimated)", value=float(est["lag"]), min_value=0.0, step=0.1)

        t = np.arange(0.0, sim_hours + 1e-9, dt)
        model_key = "logistic" if sim_model.startswith("Logistic") else ("gompertz" if sim_model.startswith("Gompertz") else "exp")
        y = simulate_growth(t, y0, model_key, mu_sim, K_sim, lag_sim, death_rate=death_rate)

        # organism settings for CFU scaling
        org_name = getattr(org, "name", None) or (org.get("name") if isinstance(org, dict) else "")
        cfu_scale = (org.get("cfu_per_od", 8e8) if isinstance(org, dict) else (8e8 if "E. coli" in str(org_name) else 4e8))
        cfu = od_to_cfu(y, scale=float(cfu_scale))

        st.plotly_chart(growth_plot(t, y), use_container_width=True)
        st.plotly_chart(cfu_plot(t, cfu), use_container_width=True)

        sim_df = pd.DataFrame({"time_h": t, "OD600": y, "log10_OD600": np.log10(y), "CFU_per_mL": cfu})
        st.dataframe(sim_df.head(20), use_container_width=True)
        set_ns(culture, "sim_df", sim_df)
