# tabs/t01_planning.py
from __future__ import annotations
import math
import streamlit as st
from bpcomp.state import get as get_ns, set as set_ns

def render(container, culture: str, mu_eff: float, K_eff: float):
    with container:
        st.subheader("Culture planning & inoculum calculator")
        c1, c2, c3 = st.columns(3)
        V_total = c1.number_input("Culture final volume (mL)", value=1000.0, min_value=1.0, step=10.0)
        OD_start = c2.number_input("Target starting OD600", value=float(get_ns(culture, "OD_start", 0.05)), min_value=0.001, step=0.01, format="%.3f")
        OD_stock = c3.number_input("Inoculum OD600 (stock)", value=5.0, min_value=0.01, step=0.1)

        V_inoc_mL = (OD_start * V_total) / OD_stock if (OD_stock>0) else float("nan")
        V_media_mL = V_total - V_inoc_mL if V_inoc_mL==V_inoc_mL else float("nan")
        st.markdown(f"- **Inoculum volume**: **{V_inoc_mL:.2f} mL**  \n- **Medium to add**: **{V_media_mL:.2f} mL**")

        st.markdown("**Harvest target & model**")
        c1, c2, c3, c4 = st.columns(4)
        OD_harvest = c1.number_input("Desired harvest OD600", value=2.0, min_value=0.1, step=0.1)
        model_choice = c2.selectbox("Model", ["Exponential", "Logistic", "Gompertz"])
        mu_mode = c3.selectbox("μ estimation", ["Use effective μ (from environment)", "Manual μ"])
        mu_manual = c4.number_input("μ (ln/h) if manual", value=0.50, min_value=0.01, step=0.01)
        mu_use = mu_eff if mu_mode.startswith("Use") else float(mu_manual)

        if model_choice == "Exponential":
            t_to_harvest = math.log(OD_harvest/OD_start)/mu_use if (OD_harvest>OD_start and mu_use>0) else float("nan")
        elif model_choice == "Logistic":
            K_guess = st.number_input("Carrying capacity K (OD)", value=float(K_eff), step=0.1, min_value=0.2)
            if OD_harvest < K_guess and mu_use>0:
                lnA = math.log((K_guess/OD_start) - 1.0); z = math.log(OD_harvest/(K_guess - OD_harvest))
                t_to_harvest = (z - lnA)/mu_use
            else:
                t_to_harvest = float("nan")
        else:
            K_guess = st.number_input("Asymptotic K (OD)", value=float(K_eff), step=0.1, min_value=0.2)
            A_guess = st.number_input("Shape A (lag control)", value=5.0, min_value=0.1, step=0.1)
            val = -math.log(OD_harvest/K_guess)
            t_to_harvest = -(1.0/mu_use)*math.log(val/max(1e-9, A_guess)) if val>0 else float("nan")

        st.markdown(f"**Estimated time to harvest** (using {model_choice}): **{t_to_harvest:.2f} h**")
        set_ns(culture, "plan", dict(OD_harvest=OD_harvest, mu_guess=mu_use, K_guess=float(locals().get("K_guess", K_eff))))
        set_ns(culture, "OD_start", OD_start)
