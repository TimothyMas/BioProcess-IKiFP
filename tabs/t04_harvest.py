# tabs/t04_harvest.py
from __future__ import annotations
import pandas as pd
import numpy as np
import streamlit as st
from bpcomp.growth import predict_time_to_target_exp
from bpcomp.state import get as get_ns

def render(container, culture: str):
    with container:
        st.subheader("Harvest predictor")
        plan = get_ns(culture, "plan", {})
        growth_df = get_ns(culture, "growth_df", pd.DataFrame())
        c1, c2 = st.columns(2)
        OD_target   = c1.number_input("Target harvest OD600", value=float(plan.get("OD_harvest", 2.0)), min_value=0.1, step=0.1)
        mu_for_pred = c2.number_input("μ for prediction (ln/h)", value=float(plan.get("mu_guess", 0.5)), min_value=0.01, step=0.01)

        preds = []
        if not growth_df.empty:
            last_rows = growth_df[growth_df["valid"]].sort_values(["replicate","time_h"]).groupby("replicate", dropna=False).tail(1)
            for _, r in last_rows.iterrows():
                t_now = float(r["time_h"])
                od_now = float(r.get("od600_smooth", r["od600"]))
                t_tgt, d_h = predict_time_to_target_exp(t_now, od_now, OD_target, mu_for_pred)
                preds.append({"replicate": r["replicate"], "t_now_h": t_now, "OD_now": od_now, "t_target_h": t_tgt, "delta_h": d_h})
        pred_df = pd.DataFrame(preds)
        st.dataframe(pred_df.style.format(precision=4), use_container_width=True)
