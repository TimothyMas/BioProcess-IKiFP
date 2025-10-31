# tabs/t02_simulation.py
from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
from bpcomp.growth import simulate_growth, od_to_cfu, rolling_mu_exponential
from bpcomp.plots import growth_plot, cfu_plot

def _estimate_from_table(growth_df: pd.DataFrame):
    # Use valid rows only
    g = growth_df[growth_df["valid"]].copy()
    if g.empty: return None
    # Pick the last replicate edited/first by default
    rep = st.selectbox("Replicate to use", sorted(g["replicate"].astype(str).unique().tolist()), help="Which replicate to estimate parameters from.")
    gg = g[g["replicate"].astype(str)==str(rep)].sort_values("time_h")

    # y0 from first valid value
    y0 = float(gg.iloc[0].get("od600_smooth", gg.iloc[0]["od600"]))
    # μ from local slopes (median of early growth)
    tc, mu_series = rolling_mu_exponential(
        gg["time_h"].to_numpy(float),
        gg.get("od600_smooth", gg["od600"]).to_numpy(float),
        wpts=3
    )
    mu_est = float(np.nanmedian(mu_series[:max(3, len(mu_series)//2)])) if mu_series.size else np.nan
    # K from observed max * 1.1
    K_est = float(np.nanmax(gg.get("od600_smooth", gg["od600"]))) * 1.1
    # lag ~ time to exceed 1.2×y0 (rough)
    try:
        lag_est = float(gg.loc[gg.get("od600_smooth", gg["od600"]) > y0*1.2, "time_h"].min())
    except Exception:
        lag_est = 0.5
    return dict(y0=y0, mu=mu_est, K=K_est, lag=max(0.0, lag_est))


def render(container, mu_eff: float, K_eff: float, lag_eff: float, org, **kwargs):
    with container:
        st.subheader("Growth simulation across phases")
        s1, s2, s3, s4 = st.columns(4)
        sim_model = s1.selectbox("Model", ["Logistic (default)", "Gompertz", "Exponential"], index=0, help="Choose the growth law used to generate the curve.")
        sim_hours = s2.number_input("Total time (h)", value=18.0, min_value=1.0, step=1.0, help="Simulation horizon.")
        dt = s3.number_input("Sampling interval (h)", value=0.25, min_value=0.05, step=0.05, format="%.2f", help="Temporal resolution for the simulated series.")
        death_rate = s4.number_input("Death rate after stationary (1/h)", value=0.00, min_value=0.0, step=0.01, help="Applies exponential decay after stationary onset.")

        mode = st.radio("Parameter source", ["Manual sliders", "Use monitoring table"], horizontal=True, help="Use environment-adjusted defaults or estimate from your monitoring table.")

        if mode == "Manual sliders":
            mu_sim = st.slider("Max μ (ln/h)", 0.05, 2.0, float(mu_eff), 0.01, help="Maximal specific growth rate.")
            K_sim  = st.slider("K (OD)", 0.5, 12.0, float(K_eff), 0.1, help="Carrying capacity (asymptotic OD).")
            lag_sim = st.slider("Lag phase (h)", 0.0, 8.0, float(lag_eff), 0.1, help="Delay before exponential onset.")
            y0 = st.number_input("Initial OD600", value=float(st.session_state.get("OD_start", 0.05)),
                                 min_value=0.001, step=0.001, format="%.3f",
                                 help="Starting optical density at time zero.")
        else:
            growth_df = st.session_state.get("growth_df", pd.DataFrame())
            if growth_df is None or growth_df.empty:
                st.info("No monitoring table found yet. Switch to 'Growth monitoring' tab and add/edit data.")
                return
            est = _estimate_from_table(growth_df)
            if est is None:
                st.info("No valid rows in the monitoring table.")
                return
            c1, c2, c3, c4 = st.columns(4)
            y0   = c1.number_input("Initial OD600 (from table)", value=float(est["y0"]), min_value=1e-6, step=0.001, format="%.3f", help="Taken from the first valid point.")
            mu_sim = c2.number_input("μ (ln/h, estimated)", value=float(est["mu"]) if np.isfinite(est["mu"]) else float(mu_eff), min_value=0.01, step=0.01, help="Median early ln(OD) slope.")
            K_sim  = c3.number_input("K (OD, estimated)", value=float(est["K"]) if np.isfinite(est["K"]) else float(K_eff), min_value=0.2, step=0.1, help="~1.1×observed max OD.")
            lag_sim = c4.number_input("Lag (h, estimated)", value=float(est["lag"]), min_value=0.0, step=0.1, help="Approx. time to exceed 1.2×initial OD.")

        # Simulate
        t = np.arange(0.0, sim_hours+1e-9, dt)
        model_key = "logistic" if sim_model.startswith("Logistic") else ("gompertz" if sim_model.startswith("Gompertz") else "exp")
        y = simulate_growth(t, y0, model_key, mu_sim, K_sim, lag_sim, death_rate=death_rate)

        # CFU scale from organism hint if available, else heuristic
        cfu_scale = getattr(org, "cfu_per_od", None)
        if cfu_scale is None:
            cfu_scale = 8e8 if "E. coli" in getattr(org, "name", "") else 4e8
        cfu = od_to_cfu(y, scale=float(cfu_scale))

        # Plot controls
        use_edited_for_plots = st.toggle("Use edited table for plots", value=False, help="If ON, plots read from the editable table below instead of live model output (freeze & annotate).")

        # Build table
        sim_df = pd.DataFrame({"time_h":t, "OD600":y, "log10_OD600":np.log10(np.maximum(y, 1e-12)), "CFU_per_mL":cfu})

        # Editable table UI
        st.markdown("**Simulated series**")
        col_add, col_name = st.columns([1,3])
        add_col_clicked = col_add.button("➕ Add blank column", help="Append a new NaN column for notes/calcs.")
        new_col_name = col_name.text_input("New column name", value="", placeholder="e.g., Notes, pO₂_meas, feed_rate...", help="Will be added to the table above.")
        if add_col_clicked and new_col_name.strip():
            if new_col_name not in sim_df.columns:
                sim_df[new_col_name.strip()] = np.nan

        editable = st.toggle("Editable results table", value=True, help="Turn ON to add rows/columns and annotate the table.")
        if editable:
            edited = st.data_editor(
                sim_df,
                num_rows="dynamic",  # ← allows adding rows
                use_container_width=True,
                hide_index=True,
            )
            # Optionally drive plots from edited table
            if use_edited_for_plots:
                # keep only the required columns if present
                t_src = pd.to_numeric(edited.get("time_h", pd.Series(dtype=float)), errors="coerce").to_numpy()
                y_src = pd.to_numeric(edited.get("OD600", pd.Series(dtype=float)), errors="coerce").to_numpy()
                good = np.isfinite(t_src) & np.isfinite(y_src)
                if good.any():
                    t_plot = t_src[good]
                    y_plot = y_src[good]
                else:
                    t_plot, y_plot = t, y
            else:
                t_plot, y_plot = t, y

            st.session_state["sim_df"] = edited.copy()
        else:
            st.dataframe(sim_df, use_container_width=True)
            t_plot, y_plot = t, y
            st.session_state["sim_df"] = sim_df.copy()

        # Plots (from chosen source)
        st.plotly_chart(growth_plot(t_plot, y_plot), use_container_width=True)
        cfu_plot_vals = od_to_cfu(y_plot, scale=float(cfu_scale))
        st.plotly_chart(cfu_plot(t_plot, cfu_plot_vals), use_container_width=True)
