# tabs/t10_dashboard.py
from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from bpcomp.custom import get_catalog, record_to_preset
from bpcomp.env_models import adjust_params_by_environment
from bpcomp.growth import simulate_growth

@st.cache_data(show_spinner=False)
def _sim_spark(mu: float, K: float, lag: float, y0: float) -> pd.DataFrame:
    t = np.linspace(0, 12, 121)
    y = simulate_growth(t, y0, "logistic", mu, K, lag, death_rate=0.0)
    return pd.DataFrame({"t": t, "y": y})

def render(container, env: dict, **kwargs):
    with container:
        st.subheader("Multi-culture dashboard")
        st.caption("Quick glance across your custom cultures: effective params in current environment, simple time-to-target and a sparkline.")

        cat = get_catalog()
        if not cat:
            st.info("No custom cultures yet. Go to the sidebar → Custom library to add some.")
            return

        c1, c2, c3 = st.columns(3)
        y0 = c1.number_input("Assumed start OD600", value=0.05, min_value=0.001, step=0.001, format="%.3f")
        target = c2.number_input("Target OD600", value=2.0, min_value=0.1, step=0.1)
        rows = c3.slider("Cards per row", 2, 4, 3)

        # ensure each card's chart has a unique key
        card_id = 0

        # grid
        chunks = [cat[i:i+rows] for i in range(0, len(cat), rows)]
        for chunk in chunks:
            cols = st.columns(len(chunk))
            for col, rec in zip(cols, chunk):
                org = record_to_preset(rec)
                env_local = dict(env, T_opt=float(org.T_opt), pH_opt=float(org.pH_opt), O2_pref=str(org.O2_pref))
                mu, K, lag, comps = adjust_params_by_environment(org.mu_ref, org.K_ref, org.lag_ref_h, env_local)

                # time-to-target (exponential rough)
                ttt = np.nan
                if mu > 0 and target > y0:
                    ttt = float(np.log(target / y0) / mu)

                # sparkline
                df = _sim_spark(mu, K, lag, y0)
                fig = go.Figure(go.Scatter(x=df["t"], y=df["y"], mode="lines", line=dict(width=2)))
                fig.update_layout(margin=dict(l=8, r=8, t=6, b=6), height=120, template="plotly_white",
                                  xaxis=dict(visible=False), yaxis=dict(visible=False))

                with col:
                    st.markdown(
                        f"""
<div class="metric-card">
<b>{org.name}</b><br><span class="muted">{org.O2_pref} • T_opt {org.T_opt:.1f}°C • pH_opt {org.pH_opt:.2f}</span>
<hr style="margin:6px 0"/>
μ<sub>eff</sub>=<b>{mu:.2f}</b> ln·h⁻¹ • K=<b>{K:.1f}</b> • lag=<b>{lag:.2f}</b> h<br>
<span class="muted">T:{comps['tp']:.2f} pH:{comps['pp']:.2f} O₂:{comps['op']:.2f} Medium:{comps['mp']:.2f}</span><br>
Time→{target:g}: <b>{(0 if np.isnan(ttt) else ttt):.2f} h</b>
</div>
                        """,
                        unsafe_allow_html=True
                    )
                    # Unique key per chart to avoid StreamlitDuplicateElementId
                    st.plotly_chart(fig, use_container_width=True, key=f"spark_{card_id}_{org.name}")
                    card_id += 1
