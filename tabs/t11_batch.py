# tabs/t11_batch.py
from __future__ import annotations
import io
import numpy as np
import pandas as pd
import streamlit as st
from bpcomp.custom import get_catalog, record_to_preset
from bpcomp.env_models import adjust_params_by_environment
from bpcomp.growth import simulate_growth

@st.cache_data(show_spinner=False)
def _simulate(mu: float, K: float, lag: float, y0: float, hours: float, dt: float, model_key: str, death: float):
    t = np.arange(0.0, hours + 1e-12, dt)
    y = simulate_growth(t, y0, model_key, mu, K, lag, death_rate=death)
    return pd.DataFrame({"time_h": t, "OD600": y, "log10_OD600": np.log10(np.maximum(y, 1e-12))})

def render(container, env: dict, **kwargs):
    with container:
        st.subheader("Batch simulation (multi-culture)")
        cat = get_catalog()
        if not cat:
            st.info("No custom cultures yet. Go to the sidebar → Custom library to add some.")
            return
        names = [c.get("name","Custom") for c in cat]
        choose = st.multiselect("Cultures to simulate", names, default=names[:min(5, len(names))])

        c1, c2, c3, c4 = st.columns(4)
        model = c1.selectbox("Model", ["Logistic", "Gompertz", "Exponential"], index=0)
        hours = c2.number_input("Total time (h)", 6.0, 168.0, 18.0, 1.0)
        dt = c3.number_input("Δt (h)", 0.05, 2.0, 0.25, 0.05, format="%.2f")
        death = c4.number_input("Death rate (1/h)", 0.0, 1.0, 0.0, 0.01)

        y0 = st.number_input("Start OD600", 0.001, 2.0, 0.05, 0.001, format="%.3f")

        if not choose:
            st.stop()

        model_key = "logistic" if model == "Logistic" else ("gompertz" if model == "Gompertz" else "exp")

        out_sheets = {}
        for nm in choose:
            rec = cat[names.index(nm)]
            org = record_to_preset(rec)
            env_local = dict(env, T_opt=float(org.T_opt), pH_opt=float(org.pH_opt), O2_pref=str(org.O2_pref))
            mu, K, lag, _ = adjust_params_by_environment(org.mu_ref, org.K_ref, org.lag_ref_h, env_local)
            df = _simulate(mu, K, lag, y0, hours, dt, model_key, death)
            out_sheets[nm] = df

        # Show a preview of the first selection
        first = choose[0]
        st.dataframe(out_sheets[first].head(20), use_container_width=True)

        # Export
        bio = io.BytesIO()
        with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
            for nm, df in out_sheets.items():
                safe = "".join([c for c in nm if c.isalnum() or c in (" ","_","-")]).strip()[:31] or "Sheet"
                df.to_excel(writer, sheet_name=safe, index=False)
        st.download_button(
            "⬇️ Download batch simulations (Excel)",
            data=bio.getvalue(),
            file_name="batch_simulations.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
