# tabs/t06_bradford.py
from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from bpcomp.fit import linfit, quadfit, invert_linear, invert_quadratic
from bpcomp.state import get as get_ns, set as set_ns

def render(container, culture: str):
    with container:
        st.subheader("Bradford calibration → unknowns")
        default_std = pd.DataFrame({"Conc_mg_mL":[1.0,0.75,0.5,0.25,0.1],
                                    "A595":[1.012,0.894,0.723,0.545,0.400],
                                    "Use":[True, True, True, True, True]})
        std_df = st.data_editor(default_std, num_rows="dynamic", hide_index=True, use_container_width=True)
        fit_mode = st.radio("Model", ["Linear", "Linear (through 0)", "Quadratic"], horizontal=True)
        use_df = std_df[std_df["Use"].fillna(True)]
        x = use_df["Conc_mg_mL"].to_numpy(dtype=float)
        y = use_df["A595"].to_numpy(dtype=float)

        try:
            if fit_mode == "Linear":
                fit_res = linfit(x, y, through_zero=False)
                eq = f"y = {fit_res['a']:.6f}·x + {fit_res['b']:.6f}"
                a_l, b_l = float(fit_res['a']), float(fit_res['b'])
                inv = (lambda Y, a=a_l, b=b_l: invert_linear(Y, a, b))
                set_ns(culture, "bradford_fit", dict(mode="Linear", eq=eq, params={"a": a_l, "b": b_l}))
            elif fit_mode == "Linear (through 0)":
                fit_res = linfit(x, y, through_zero=True)
                eq = f"y = {fit_res['a']:.6f}·x"
                a0 = float(fit_res['a'])
                inv = (lambda Y, a=a0: invert_linear(Y, a, 0.0))
                set_ns(culture, "bradford_fit", dict(mode="Linear0", eq=eq, params={"a": a0, "b": 0.0}))
            else:
                fit_res = quadfit(x, y)
                eq = f"y = {fit_res['a']:.6f}·x² + {fit_res['b']:.6f}·x + {fit_res['c']:.6f}"
                aq, bq, cq = (float(fit_res['a']), float(fit_res['b']), float(fit_res['c']))
                xr = (float(np.min(fit_res["x"])), float(np.max(fit_res["x"])))
                inv = (lambda Y, a=aq, b=bq, c=cq, xr=xr: invert_quadratic(Y, a, b, c, xr))
                set_ns(culture, "bradford_fit", dict(mode="Quadratic", eq=eq, params={"a": aq, "b": bq, "c": cq, "xr": xr}))

            st.markdown(f"**Calibration:** {eq} • **R²={fit_res['R2']:.5f}**, **RMSE={fit_res['RMSE']:.5f}**")
            _x = np.asarray(fit_res["x"], dtype=float)
            x_plot = np.linspace(float(np.min(_x)), float(np.max(_x)), 200)
            if fit_mode.startswith("Linear"):
                y_plot = (fit_res["a"]*x_plot + (0.0 if "through 0" in fit_mode else fit_res["b"]))
            else:
                y_plot = fit_res["a"]*x_plot**2 + fit_res["b"]*x_plot + fit_res["c"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fit_res["x"], y=fit_res["y"], mode="markers", name="Standards"))
            fig.add_trace(go.Scatter(x=x_plot, y=y_plot, mode="lines", name=f"{fit_mode} fit"))
            fig.update_layout(template="plotly_white", title="Bradford calibration", xaxis_title="Protein (mg/mL)", yaxis_title="A595")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("**Unknowns (apply calibration)**")
            default_unknowns = pd.DataFrame({"Sample":["Peak_pool","Endpoint","Supernatant"],
                                             "A595":[0.617667, np.nan, 0.933667],
                                             "Dilution":[5, 1, 15]})
            unk_df = st.data_editor(default_unknowns, num_rows="dynamic", hide_index=True, use_container_width=True)

            p = get_ns(culture, "bradford_fit", {}).get("params", {}); mode = get_ns(culture, "bradford_fit", {}).get("mode")
            if mode == "Linear":
                inv_f = (lambda Y, a=p["a"], b=p["b"]: invert_linear(Y, a, b))
            elif mode == "Linear0":
                inv_f = (lambda Y, a=p["a"]: invert_linear(Y, a, 0.0))
            else:
                inv_f = (lambda Y, a=p["a"], b=p["b"], c=p["c"], xr=p["xr"]: invert_quadratic(Y, a, b, c, xr))

            def conc_from_A(A: float, Dil: float) -> float:
                if not np.isfinite(A): return np.nan
                x_diluted = inv_f(float(A))
                return x_diluted * float(Dil) if np.isfinite(x_diluted) else np.nan

            out = unk_df.copy()
            out["Protein_mg_mL"] = [conc_from_A(float(a) if pd.notna(a) else np.nan, float(d)) for a, d in zip(out["A595"], out["Dilution"])]
            st.dataframe(out.style.format(precision=6), use_container_width=True)
            set_ns(culture, "protein_table", out)
        except Exception as e:
            st.error(f"Calibration failed: {e}")
            set_ns(culture, "bradford_fit", None)
