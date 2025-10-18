# app.py
from __future__ import annotations
import streamlit as st
from typing import Mapping, Union, Optional, cast

from bpcomp.config import set_page, inject_css
from bpcomp.presets import ORGANISMS
from bpcomp.env_models import (
    medium_quality_score,
    adjust_params_by_environment,
    acceptor_factor,
    orp_penalty,
    estimate_refs_from_env,
)
from bpcomp.state import (
    clear_keys, ensure_cultures, add_culture, delete_culture, rename_culture,
    get as get_ns, set as set_ns,
)

from tabs import (
    t01_planning, t02_simulation, t03_monitoring, t04_harvest,
    t05_activity, t06_bradford, t07_purification, t08_plots_export, t09_settings
)

# ---- robust numeric coercion
def as_float(x: Optional[Union[float, int, str]], default: Union[float, int, str]) -> float:
    try:
        return float(x) if x is not None else float(default)
    except Exception:
        try:
            return float(default)
        except Exception:
            return 0.0

set_page()
inject_css()

st.title("🧫 Bioprocess Companion")
st.caption("Multi-culture ready • Use the selector below to switch between cultures")

# ── Sidebar: culture manager, organism, environment
with st.sidebar:
    st.header("⚙️ Setup & Constants")

    # --- Culture manager ---
    st.markdown("**Cultures**")
    cultures = ensure_cultures()
    if "current_culture" not in st.session_state:
        st.session_state["current_culture"] = cultures[0]
    cur = st.selectbox("Active culture", cultures,
                       index=(cultures.index(st.session_state["current_culture"])
                              if st.session_state["current_culture"] in cultures else 0))
    st.session_state["current_culture"] = cur

    cadd, cdel = st.columns([2, 1])
    new_name = cadd.text_input("New culture name", value="", placeholder="e.g., Culture 2")
    if cadd.button("➕ Add"):
        name_to_add = new_name or f"Culture {len(cultures) + 1}"
        add_culture(name_to_add)
        st.session_state["current_culture"] = name_to_add
        st.rerun()
    if cdel.button("🗑️ Delete", help="Keeps at least one culture"):
        delete_culture(cur)
        st.rerun()

    r1, r2 = st.columns([2, 1])
    rename_to = r1.text_input("Rename", value=cur)
    if r2.button("✏️ Apply") and rename_to and rename_to != cur:
        rename_culture(cur, rename_to)
        st.session_state["current_culture"] = rename_to
        st.rerun()

    st.divider()

    # --- Organism source (fixed: no double-click) ---
    st.caption("Organism source")

    org_mode_key = f"{cur}::org_mode"
    if org_mode_key not in st.session_state:
        st.session_state[org_mode_key] = "Preset"
    org_mode = st.radio(
        "Source",
        ["Preset", "Custom"],
        horizontal=True,
        index=["Preset", "Custom"].index(st.session_state[org_mode_key]),
        key=org_mode_key,
    )

    if org_mode == "Preset":
        preset_key = f"{cur}::preset_name"
        preset_names = list(ORGANISMS.keys())
        if preset_key not in st.session_state:
            st.session_state[preset_key] = preset_names[0]
        try:
            preset_index = preset_names.index(st.session_state[preset_key])
        except ValueError:
            preset_index = 0
        org_name = st.selectbox(
            "Organism preset",
            preset_names,
            index=preset_index,
            key=preset_key,
        )

        p = ORGANISMS[org_name]
        org = dict(
            mode="Preset",
            name=p.name,
            T_opt=float(p.T_opt),
            pH_opt=float(p.pH_opt),
            O2_pref=str(p.O2_pref),
            mu_ref=float(p.mu_ref),
            K_ref=float(p.K_ref),
            lag_ref_h=float(p.lag_ref_h),
            notes=str(p.notes),
            auto_refs=False,
            cfu_per_od=4e8 if "Aromatoleum" in p.name else 8e8,
            o2_tol_pct=0.5 if p.O2_pref == "anaerobic" else 5.0,
        )
        set_ns(cur, "organism", org)

    else:
        prior = get_ns(cur, "organism", {})
        st.text_input(
            "Name",
            value=str(prior.get("name", "Custom organism")),
            key=f"{cur}::org_custom_name",
        )
        cA, cB, cC = st.columns(3)
        O2_pref = cA.selectbox(
            "O₂ preference",
            ["aerobic", "anaerobic", "microaerophilic"],
            index=["aerobic","anaerobic","microaerophilic"].index(str(prior.get("O2_pref","anaerobic"))),
            key=f"{cur}::org_custom_o2pref",
        )
        auto_refs_key = f"{cur}::org_custom_auto_refs"
        if auto_refs_key not in st.session_state:
            st.session_state[auto_refs_key] = bool(prior.get("auto_refs", True))
        auto_refs = st.toggle(
            "Auto-estimate μ_ref / K_ref / lag_ref",
            key=auto_refs_key,
        )
        T_opt  = cB.number_input(
            "T_opt (°C)",
            value=as_float(prior.get("T_opt"), 30.0),
            step=0.5,
            key=f"{cur}::org_custom_Topt",
        )
        pH_opt = cC.number_input(
            "pH_opt",
            value=as_float(prior.get("pH_opt"), 7.3),
            step=0.1, format="%.2f",
            key=f"{cur}::org_custom_pHopt",
        )
        c1, c2, c3 = st.columns(3)
        mu_ref = c1.number_input(
            "μ_ref (ln/h)",
            value=as_float(prior.get("mu_ref"), 0.35),
            min_value=0.01, step=0.01,
            disabled=st.session_state[auto_refs_key],
            key=f"{cur}::org_custom_mu",
        )
        K_ref  = c2.number_input(
            "K_ref (OD)",
            value=as_float(prior.get("K_ref"), 4.0),
            min_value=0.2, step=0.1,
            disabled=st.session_state[auto_refs_key],
            key=f"{cur}::org_custom_K",
        )
        lag_h  = c3.number_input(
            "Lag_ref (h)",
            value=as_float(prior.get("lag_ref_h"), 2.0),
            min_value=0.01, step=0.05,
            disabled=st.session_state[auto_refs_key],
            key=f"{cur}::org_custom_lag",
        )
        c4, c5 = st.columns(2)
        cfu_per_od = c4.number_input(
            "CFU per OD600 (1/mL)",
            value=as_float(prior.get("cfu_per_od"), 4e8),
            step=1e7, format="%.0f",
            key=f"{cur}::org_custom_cfu_scale",
        )
        o2_tol_pct = c5.number_input(
            "O₂ tolerance threshold (%)",
            value=as_float(prior.get("o2_tol_pct"), 0.5),
            min_value=0.0, step=0.1,
            key=f"{cur}::org_custom_o2tol",
        )
        notes = st.text_input(
            "Notes",
            value=str(prior.get("notes", "User-defined organism parameters")),
            key=f"{cur}::org_custom_notes",
        )

        org = dict(
            mode="Custom",
            name=st.session_state[f"{cur}::org_custom_name"],
            O2_pref=st.session_state[f"{cur}::org_custom_o2pref"],
            T_opt=float(st.session_state[f"{cur}::org_custom_Topt"]),
            pH_opt=float(st.session_state[f"{cur}::org_custom_pHopt"]),
            mu_ref=float(st.session_state[f"{cur}::org_custom_mu"]),
            K_ref=float(st.session_state[f"{cur}::org_custom_K"]),
            lag_ref_h=float(st.session_state[f"{cur}::org_custom_lag"]),
            cfu_per_od=float(st.session_state[f"{cur}::org_custom_cfu_scale"]),
            o2_tol_pct=float(st.session_state[f"{cur}::org_custom_o2tol"]),
            notes=st.session_state[f"{cur}::org_custom_notes"],
            auto_refs=bool(st.session_state[auto_refs_key]),
        )
        set_ns(cur, "organism", org)

    # --- Respiration & Gas (fixed: toggles/selects keep state without double-click) ---
    st.caption("Respiration & Gas")
    env_prior = get_ns(cur, "env", {})

    auto_key = f"{cur}::auto_anaerobe"
    if auto_key not in st.session_state:
        st.session_state[auto_key] = bool(env_prior.get("auto_anaerobe", True))
    auto_anaerobe = st.toggle("Auto anaerobe settings (acceptor & ORP)", key=auto_key)

    acceptor_key = f"{cur}::acceptor"
    if acceptor_key not in st.session_state:
        st.session_state[acceptor_key] = str(env_prior.get("acceptor", "auto"))
    orp_key = f"{cur}::orp_mv"
    if orp_key not in st.session_state:
        st.session_state[orp_key] = float(as_float(env_prior.get("orp_mv"), -250.0))

    r1, r2 = st.columns(2)
    acceptor = r1.selectbox(
        "Electron acceptor",
        ["auto", "nitrate (NO₃⁻)","nitrite (NO₂⁻)","Mn(IV) (MnO₂)","none"],
        index=["auto", "nitrate (NO₃⁻)","nitrite (NO₂⁻)","Mn(IV) (MnO₂)","none"].index(st.session_state[acceptor_key]),
        key=acceptor_key,
        disabled=st.session_state[auto_key],
    )
    orp_mv = r2.number_input(
        "ORP setpoint (mV)",
        value=float(st.session_state[orp_key]),
        step=10.0,
        help="Keep ≤ −200 mV for strict anaerobes; more negative often safer.",
        key=orp_key,
        disabled=st.session_state[auto_key],
    )

    if st.session_state[auto_key]:
        # Derive automatic targets but also keep user's manual selections stored
        acceptor = "nitrate (NO₃⁻)" if (org["O2_pref"] == "anaerobic") else "none"
        orp_mv = -250.0 if (org["O2_pref"] == "anaerobic") else -100.0

    g1, g2 = st.columns(2)
    co2_key = f"{cur}::co2_pct"
    if co2_key not in st.session_state:
        st.session_state[co2_key] = float(as_float(env_prior.get("co2_pct"), 10.0))
    overp_key = f"{cur}::overpress"
    if overp_key not in st.session_state:
        st.session_state[overp_key] = float(as_float(env_prior.get("overpress"), 0.0))
    co2_pct = g1.slider("CO₂ in headspace (%)", 0.0, 30.0, float(st.session_state[co2_key]), 1.0, key=co2_key)
    overpress = g2.number_input("Overpressure (mbar)", value=float(st.session_state[overp_key]), step=10.0, key=overp_key)

    redox_key = f"{cur}::redox_ind"
    if redox_key not in st.session_state:
        st.session_state[redox_key] = bool(env_prior.get("redox_ind", True))
    redox_ind = st.toggle("Use resazurin indicator", key=redox_key)

    # --- Environment ---
    st.caption("Environment")
    c1, c2 = st.columns(2)
    T_key = f"{cur}::T"
    if T_key not in st.session_state:
        st.session_state[T_key] = float(as_float(env_prior.get("T"), org["T_opt"]))
    pH_key = f"{cur}::pH"
    if pH_key not in st.session_state:
        st.session_state[pH_key] = float(as_float(env_prior.get("pH"), org["pH_opt"]))
    O2_key = f"{cur}::O2"
    if O2_key not in st.session_state:
        st.session_state[O2_key] = float(as_float(env_prior.get("O2"), 0.5 if org["O2_pref"]=="anaerobic" else 21.0))

    T = c1.number_input("Temperature (°C)", value=float(st.session_state[T_key]), step=0.5, key=T_key)
    pH = c2.number_input("pH", value=float(st.session_state[pH_key]), step=0.1, format="%.2f", key=pH_key)
    O2 = st.slider("Dissolved/Headspace O₂ (%)", 0.0, 21.0, float(st.session_state[O2_key]), 0.1, key=O2_key)

    # --- Medium planner ---
    st.caption("Medium planner (smart)")
    m1, m2 = st.columns(2)
    carb_key = f"{cur}::carbon_gL"
    if carb_key not in st.session_state:
        st.session_state[carb_key] = float(as_float(env_prior.get("carbon_gL"), 10.0))
    n_key = f"{cur}::N_mM"
    if n_key not in st.session_state:
        st.session_state[n_key] = float(as_float(env_prior.get("N_mM"), 15.0))
    carbon_gL = m1.slider("Carbon source (g/L)", 0.0, 30.0, float(st.session_state[carb_key]), 0.5, key=carb_key)
    N_mM     = m2.slider("Nitrogen (mM)", 0.0, 50.0, float(st.session_state[n_key]), 1.0, key=n_key)

    b1, b2, b3 = st.columns(3)
    buffer_key = f"{cur}::buffer_sys"
    if buffer_key not in st.session_state:
        st.session_state[buffer_key] = str(env_prior.get("buffer_sys", "bicarbonate"))
    reducer_key = f"{cur}::reducer"
    if reducer_key not in st.session_state:
        st.session_state[reducer_key] = str(env_prior.get("reducer", "ascorbate"))
    salts_key = f"{cur}::salts_ok"
    if salts_key not in st.session_state:
        st.session_state[salts_key] = bool(env_prior.get("salts_ok", True))

    buffer_sys = b1.selectbox("Buffer", ["phosphate","bicarbonate"],
                              index=["phosphate","bicarbonate"].index(st.session_state[buffer_key]),
                              key=buffer_key)
    reducer = b2.selectbox("Reducing agent", ["ascorbate","cysteine","DTT","none"],
                           index=["ascorbate","cysteine","DTT","none"].index(st.session_state[reducer_key]),
                           key=reducer_key)
    salts_ok  = b3.checkbox("Trace salts/vitamins OK", value=bool(st.session_state[salts_key]), key=salts_key)

    cn_target = (carbon_gL/12.0) / max(1e-6, (N_mM/1000.0))
    st.caption(f"C:N helper (mol ratio, rough): **{cn_target:.1f}**")

    # --- Lab constants (activity) ---
    st.caption("Lab constants (activity)")
    const_prior = get_ns(cur, "constants", {})
    c3, c4 = st.columns(2)
    eps_key = f"{cur}::epsilon"
    if eps_key not in st.session_state:
        st.session_state[eps_key] = float(as_float(const_prior.get("epsilon"), 3.40))
    path_key = f"{cur}::path_cm"
    if path_key not in st.session_state:
        st.session_state[path_key] = float(as_float(const_prior.get("path_cm"), 1.00))
    c3.number_input("ε (mM⁻¹·cm⁻¹)", value=float(st.session_state[eps_key]), min_value=0.0001, step=0.01, key=eps_key)
    c4.number_input("Path length (cm)", value=float(st.session_state[path_key]), min_value=0.01, step=0.01, key=path_key)
    c5, c6 = st.columns(2)
    vrx_key = f"{cur}::V_rxn_mL"
    if vrx_key not in st.session_state:
        st.session_state[vrx_key] = float(as_float(const_prior.get("V_rxn_mL"), 1.000))
    vs_key = f"{cur}::V_samp_mL"
    if vs_key not in st.session_state:
        st.session_state[vs_key] = float(as_float(const_prior.get("V_samp_mL"), 0.010))
    c5.number_input("V_rxn (mL)", value=float(st.session_state[vrx_key]), min_value=0.01, step=0.01, key=vrx_key)
    c6.number_input("V_samp (mL)", value=float(st.session_state[vs_key]), min_value=0.001, step=0.001, key=vs_key)

    set_ns(cur, "constants", dict(
        epsilon=float(st.session_state[eps_key]),
        path_cm=float(st.session_state[path_key]),
        V_rxn_mL=float(st.session_state[vrx_key]),
        V_samp_mL=float(st.session_state[vs_key]),
    ))

    # -- persist env & compute score/effective params
    medium_score = medium_quality_score(
        float(st.session_state[carb_key]),
        float(st.session_state[n_key]),
        bool(st.session_state[salts_key]),
        True, True
    )

    # derive final acceptor/ORP if auto
    final_acceptor = acceptor
    final_orp = float(orp_mv)
    if st.session_state[auto_key]:
        final_acceptor = "nitrate (NO₃⁻)" if (org["O2_pref"] == "anaerobic") else "none"
        final_orp = -250.0 if (org["O2_pref"] == "anaerobic") else -100.0

    env = dict(
        T=float(st.session_state[T_key]),
        pH=float(st.session_state[pH_key]),
        O2=float(st.session_state[O2_key]),
        acceptor=final_acceptor,
        orp_mv=final_orp,
        auto_anaerobe=bool(st.session_state[auto_key]),
        co2_pct=float(st.session_state[co2_key]),
        overpress=float(st.session_state[overp_key]),
        redox_ind=bool(st.session_state[redox_key]),
        carbon_gL=float(st.session_state[carb_key]),
        N_mM=float(st.session_state[n_key]),
        buffer_sys=str(st.session_state[buffer_key]),
        reducer=str(st.session_state[reducer_key]),
        medium_score=medium_score,
        T_opt=org["T_opt"], pH_opt=org["pH_opt"], O2_pref=org["O2_pref"]
    )
    set_ns(cur, "env", env)

    # typed env for model
    typed_env: Mapping[str, Union[float, str]] = cast(
        Mapping[str, Union[float, str]],
        {
            "T": float(env["T"]),
            "pH": float(env["pH"]),
            "O2": float(env["O2"]),
            "T_opt": float(org["T_opt"]),
            "pH_opt": float(org["pH_opt"]),
            "O2_pref": str(org["O2_pref"]),
            "medium_score": float(env["medium_score"]),
        },
    )

    # Choose references: manual or auto
    use_auto_refs = bool(org.get("auto_refs", False))
    missing_ref = any([
        not isinstance(org.get("mu_ref"), (int,float)) or float(org.get("mu_ref", 0)) <= 0,
        not isinstance(org.get("K_ref"), (int,float)) or float(org.get("K_ref", 0)) <= 0,
        not isinstance(org.get("lag_ref_h"), (int,float)) or float(org.get("lag_ref_h", 0)) <= 0,
    ])
    if use_auto_refs or missing_ref:
        mu_ref_use, K_ref_use, lag_ref_use = estimate_refs_from_env(str(org["O2_pref"]), typed_env)
    else:
        mu_ref_use  = float(org["mu_ref"])
        K_ref_use   = float(org["K_ref"])
        lag_ref_use = float(org["lag_ref_h"])

    mu_eff, K_eff, lag_eff, comps = adjust_params_by_environment(
        float(mu_ref_use),
        float(K_ref_use),
        float(lag_ref_use),
        typed_env,
    )
    # ORP / acceptor modifiers
    mu_eff *= acceptor_factor(final_acceptor)
    mu_eff *= orp_penalty(float(final_orp), as_float(org.get("o2_tol_pct"), 0.5))

    st.markdown(
        f"""
<div class="helpbox">
<b>{org['name']}</b><br>{org['notes']}<br>
<hr style="margin:6px 0"/>
<b>Effective params</b>: μ<sub>max</sub> ≈ <b>{mu_eff:.2f}</b> ln·h⁻¹ • K ≈ <b>{K_eff:.1f}</b> OD • lag ≈ <b>{lag_eff:.2f}</b> h
<br><span class="muted">Fitness - T:{comps['tp']:.2f} pH:{comps['pp']:.2f} O₂:{comps['op']:.2f} Medium:{comps['mp']:.2f} • ORP:{final_orp:.0f} mV • Acceptor:{final_acceptor}</span>
</div>
""",
        unsafe_allow_html=True
    )

tabs = st.tabs([
    "① Culture planning",
    "② Growth simulation",
    "③ Growth monitoring (OD/CFU)",
    "④ Harvest predictor",
    "⑤ Activity assay",
    "⑥ Bradford & Protein",
    "⑦ Purification & KPIs",
    "⑧ Plots & Export",
    "⑨ Settings & Session",
])

culture = st.session_state.get("current_culture", "Culture 1")
consts = get_ns(culture, "constants", {})
epsilon = as_float(consts.get("epsilon"), 3.40)
path_cm = as_float(consts.get("path_cm"), 1.00)
V_rxn_mL = as_float(consts.get("V_rxn_mL"), 1.000)
V_samp_mL = as_float(consts.get("V_samp_mL"), 0.010)
org_for_tabs = get_ns(culture, "organism", org)

t01_planning.render(tabs[0], culture, mu_eff, K_eff)
t02_simulation.render(tabs[1], culture, mu_eff, K_eff, lag_eff, org_for_tabs, float(epsilon), float(path_cm))
t03_monitoring.render(tabs[2], culture)
t04_harvest.render(tabs[3], culture)
t05_activity.render(tabs[4], culture, float(epsilon), float(path_cm), float(V_rxn_mL), float(V_samp_mL))
t06_bradford.render(tabs[5], culture)
t07_purification.render(tabs[6], culture)
t08_plots_export.render(tabs[7], culture)
t09_settings.render(tabs[8], culture, clear_keys, env, org_for_tabs["name"])
