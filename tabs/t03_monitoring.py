# tabs/t03_monitoring.py
from __future__ import annotations
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from bpcomp.growth import rolling_mu_exponential
from bpcomp.state import get as get_ns, set as set_ns

def render(container, culture: str):
    with container:
        st.subheader("Growth monitoring (OD/CFU)")
        st.markdown("Upload OD600/CFU data or try the demo dataset.")
        up = st.file_uploader("Upload CSV/XLSX", type=["csv","xlsx"])
        if up is not None:
            try:
                raw = pd.read_csv(up) if up.name.endswith(".csv") else pd.read_excel(up)
            except Exception as e:
                st.error(f"Could not read file: {e}"); raw = pd.DataFrame()
        else:
            raw = pd.DataFrame({"time_h":[0,1,2,3,4,5], "od600":[0.05,0.09,0.16,0.3,0.58,1.05], "replicate":["A"]*6})

        if raw.empty:
            st.stop()

        cols = list(raw.columns)
        c1, c2, c3 = st.columns(3)
        time_col = c1.selectbox("Time (h)", cols, index=cols.index("time_h") if "time_h" in cols else 0)
        od_col   = c2.selectbox("OD600", cols, index=cols.index("od600") if "od600" in cols else min(1,len(cols)-1))
        rep_col  = c3.selectbox("Replicate", ["(none)"]+cols, index=(cols.index("replicate")+1 if "replicate" in cols else 0))

        df = pd.DataFrame({
            "time_h": pd.to_numeric(raw[time_col], errors="coerce"),
            "od600": pd.to_numeric(raw[od_col], errors="coerce"),
            "replicate": (raw[rep_col] if rep_col!="(none)" else "-")
        }).sort_values(["replicate","time_h"]).reset_index(drop=True)

        with st.expander("Data cleaning & preprocessing"):
            cA, cB, cC = st.columns(3)
            od_min = cA.number_input("OD min (keep ≥)", value=0.005, min_value=0.0, step=0.001)
            od_max = cB.number_input("OD max (keep ≤)", value=6.0, min_value=0.1, step=0.1)
            smooth_win = int(cC.number_input("Smoothing window (points, 1 = off)", value=1, min_value=1, step=1))
        df["valid"] = df["od600"].between(od_min, od_max)
        if smooth_win>1:
            out = []
            for _, g in df.groupby("replicate", dropna=False):
                yv = g["od600"].to_numpy(dtype=float)
                if len(yv)>=smooth_win:
                    ys = pd.Series(yv).rolling(smooth_win, center=True, min_periods=1).mean().to_numpy(dtype=float)
                else:
                    ys = yv
                g = g.copy(); g["od600_smooth"] = ys; out.append(g)
            df = pd.concat(out, ignore_index=True)
        else:
            df["od600_smooth"] = df["od600"]

        st.markdown("**Clean data (editable)**")
        df_edit = st.data_editor(df, hide_index=True, num_rows="dynamic", use_container_width=True)
        dplot = df_edit[df_edit["valid"]].copy()

        fig = px.line(dplot, x="time_h", y="od600_smooth", color="replicate", markers=True, title="Growth curves")
        fig.update_layout(template="plotly_white", height=420, yaxis_title="OD600", xaxis_title="Time (h)")
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Local μ(t) estimator (ln OD slope)"):
            wloc = st.number_input("Window (points)", value=3, min_value=3, step=1)
            fig_mu = go.Figure()
            for rep, g in dplot.groupby("replicate", dropna=False):
                tc, mul = rolling_mu_exponential(
                    g["time_h"].to_numpy(dtype=float),
                    g["od600_smooth"].to_numpy(dtype=float),
                    int(wloc)
                )
                if tc.size:
                    fig_mu.add_trace(go.Scatter(x=tc, y=mul, mode="lines+markers", name=str(rep)))
            fig_mu.update_layout(template="plotly_white", height=360, title="μ(t) from ln(OD) slope")
            fig_mu.update_xaxes(title="Time (h)"); fig_mu.update_yaxes(title="μ (ln/h)")
            st.plotly_chart(fig_mu, use_container_width=True)

        set_ns(culture, "growth_df", df_edit)
