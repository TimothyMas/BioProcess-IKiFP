# app.py
from __future__ import annotations
import inspect
from typing import Optional, Any
import streamlit as st
import pandas as pd
import numpy as np

from bpcomp.config import set_page, inject_css
from bpcomp.presets import ORGANISMS, OrganismPreset
from bpcomp.env_models import medium_quality_score, adjust_params_by_environment
from bpcomp.state import clear_keys

# Custom helpers
from bpcomp.custom import (
    get_catalog, get_selected_idx, set_selected,
    add_new_culture, duplicate_culture, delete_culture,
    update_selected, export_catalog_json, import_catalog_json,
    record_to_preset, default_by_o2, as_float, add_from_preset,
)

# Tabs (existing)
from tabs import (
    t01_planning, t02_simulation, t03_monitoring, t04_harvest,
    t05_activity, t06_bradford, t07_purification, t08_plots_export, t09_settings
)

# Optional electrochem tab
_t10_ec: Optional[Any] = None
try:
    from tabs import t10_electrochem as _t10_ec
except Exception:
    _t10_ec = None

# New tabs
from tabs import t10_dashboard, t11_batch


# ─────────────────────────────
# Helpers
# ─────────────────────────────

def _safe_render(tab_module: Any, container, **kwargs) -> None:
    """Safely call a tab module's render(container, **kwargs) with filtered args."""
    if tab_module is None:
        return
    render = getattr(tab_module, "render", None)
    if render is None:
        return
    sig = inspect.signature(render)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    render(container, **filtered)


def _seed_defaults() -> None:
    st.session_state.setdefault("org_source", "Preset")
    st.session_state.setdefault("org_name", next(iter(ORGANISMS.keys())))
    st.session_state.setdefault("_last_org_key", ("Preset", st.session_state["org_name"]))

    org0 = ORGANISMS[st.session_state["org_name"]]
    st.session_state.setdefault("T", float(org0.T_opt))
    st.session_state.setdefault("pH", float(org0.pH_opt))
    st.session_state.setdefault("O2", 21.0 if org0.O2_pref.lower() == "aerobic" else 0.5)

    st.session_state.setdefault("carbon_gL", 10.0)
    st.session_state.setdefault("N_mM", 15.0)
    st.session_state.setdefault("salts_ok", True)
    st.session_state.setdefault("trace_ok", True)
    st.session_state.setdefault("vitamins_ok", True)

    st.session_state.setdefault("epsilon", 3.40)
    st.session_state.setdefault("path_cm", 1.00)
    st.session_state.setdefault("V_rxn_mL", 1.000)
    st.session_state.setdefault("V_samp_mL", 0.010)

    st.session_state.setdefault("compact", False)


def _apply_pending_source_switch() -> None:
    """If a button requested switching organism source (e.g., Preset → Custom),
    perform the switch BEFORE widgets are created in this run, then clear flags.
    """
    pending = st.session_state.get("_pending_source", None)
    if not pending:
        return

    # Set the radio value safely (widget not created yet in this run)
    st.session_state["org_source"] = pending

    if pending == "Custom library":
        idx = st.session_state.get("_pending_custom_select_idx", None)
        if idx is not None:
            try:
                set_selected(int(idx))
            except Exception:
                pass

    # Cleanup flags
    for k in ("_pending_source", "_pending_custom_select_idx"):
        if k in st.session_state:
            del st.session_state[k]


def _load_demo_run() -> None:
    # Growth / sim demo
    t = np.arange(0, 18.01, 2.0)
    od = np.array([0.0494, 0.0656, 0.0985, 0.1467, 0.2171, 0.3178, 0.4586, 0.6487, 0.8936, 1.1908])
    growth_df = pd.DataFrame({"time_h": t[:len(od)], "od600": od, "replicate": ["A"] * len(od)})
    growth_df["valid"] = True
    growth_df["od600_smooth"] = growth_df["od600"]
    st.session_state["growth_df"] = growth_df

    st.session_state["sim_df"] = pd.DataFrame({
        "time_h": t[:len(od)], "OD600": od, "log10_OD600": np.log10(od), "CFU_per_mL": od * 8e8
    })

    # Assay demo
    eps = float(st.session_state.get("epsilon", 3.40))
    path = float(st.session_state.get("path_cm", 1.00))
    Vrxn = float(st.session_state.get("V_rxn_mL", 1.000))
    Vsamp = float(st.session_state.get("V_samp_mL", 0.010))

    act_raw = pd.DataFrame({
        "Sample": ["Peak_1", "Peak_2", "Peak_3", "Endpoint", "Supernatant"],
        "Background_mABS_min": [0.56, 0.62, 1.31, 0.66, 10.46],
        "ValueAfter_mABS_min": [237.47, 246.41, 227.87, 50.69, 125.31],
        "PreDilution": [5, 5, 5, 1, 15]
    })
    calc = act_raw.copy()
    calc["Final_mABS_min"] = calc["ValueAfter_mABS_min"] - calc["Background_mABS_min"]
    calc["Rate_mM_min"] = (calc["Final_mABS_min"] / 1000.0) / max(eps * path, 1e-12)
    calc["U_per_mL_stock"] = calc["Rate_mM_min"] * (Vrxn / max(Vsamp, 1e-12)) * calc["PreDilution"].clip(lower=1)
    st.session_state["assay_calc"] = calc

    # Bradford demo
    prot = pd.DataFrame({"Sample": ["Peak_pool", "Endpoint", "Supernatant"],
                         "A595": [0.617667, 0.210000, 0.933667], "Dilution": [5, 1, 15]})
    prot["Protein_mg_mL"] = [2.1, 0.35, 3.0]
    st.session_state["protein_table"] = prot

    # Reset KPIs/purif (will be recomputed)
    st.session_state["purif_full"] = pd.DataFrame()
    st.session_state["kpis"] = {}


# ─────────────────────────────
# App bootstrap
# ─────────────────────────────

set_page()
inject_css()
_seed_defaults()
_apply_pending_source_switch()

st.title("🧫 Bioprocess Companion")
st.caption("Simulate & plan cultures • Monitor OD/CFU • Predict harvest • Medium planner • Activity & purification • Plots • Exports")

qa1, qa2 = st.columns([1, 1])
with qa1:
    if st.button("📦 Load demo run", type="secondary",
                 help="Prefill all tabs with a consistent demo dataset you can edit."):
        _load_demo_run()
        st.toast("Demo data loaded.", icon="✅")
with qa2:
    st.toggle("Compact layout", key="compact", help="Smaller paddings & controls; nice for small screens.")
    if st.session_state.get("compact"):
        st.markdown("""
        <style>.block-container{padding-top:.6rem}.stButton>button,.stDownloadButton>button{padding:.3rem .55rem}
        label,.stRadio>div[role=radiogroup]{font-size:.95rem}</style>
        """, unsafe_allow_html=True)

# ─────────────────────────────
# Sidebar — Setup & Constants
# ─────────────────────────────
with st.sidebar:
    st.header("⚙️ Setup & Constants")

    src = st.radio(
        "Organism source",
        ["Preset", "Custom library"],
        horizontal=True,
        key="org_source",
        help="Use a built-in preset or your own saved cultures (add/duplicate/import/export)."
    )

    if src == "Preset":
        # Choose preset
        if "org_name" not in st.session_state:
            st.session_state["org_name"] = next(iter(ORGANISMS.keys()))
        org_name = st.selectbox(
            "Preset organism",
            list(ORGANISMS.keys()),
            index=list(ORGANISMS.keys()).index(st.session_state["org_name"]),
            key="org_name",
            help="Choose a built-in organism profile."
        )
        org = ORGANISMS[org_name]

        # Actions
        c_reset, c_copy = st.columns(2)
        if c_reset.button("↺ Reset to preset", help="Reset environment sliders (T, pH, O₂) to this preset's defaults."):
            st.session_state["T"] = float(org.T_opt)
            st.session_state["pH"] = float(org.pH_opt)
            st.session_state["O2"] = 21.0 if org.O2_pref.lower() == "aerobic" else 0.5
            st.toast("Environment reset.", icon="🔄")

        if c_copy.button("➡️ Copy to custom", help="Clone this preset into your Custom library to tweak it."):
            new_idx = add_from_preset(org)
            # Queue the switch and rerun so radio builds with the updated value safely
            st.session_state["_pending_source"] = "Custom library"
            st.session_state["_pending_custom_select_idx"] = int(new_idx)
            st.rerun()

        # Keep env tied when changing presets
        if st.session_state.get("_last_org_key") != ("Preset", org_name):
            st.session_state["T"] = float(org.T_opt)
            st.session_state["pH"] = float(org.pH_opt)
            st.session_state["O2"] = 21.0 if org.O2_pref.lower() == "aerobic" else 0.5
            st.session_state["_last_org_key"] = ("Preset", org_name)

    else:
        # Custom library UI
        catalog = get_catalog()
        names = [r.get("name", f"Custom {i+1}") for i, r in enumerate(catalog)]
        sel_idx = int(get_selected_idx())

        top = st.columns([5, 1, 1, 1, 1])
        with top[0]:
            selected_label = st.selectbox(
                "Custom culture",
                names,
                index=sel_idx,
                key="custom_sel_box",
                help="Pick an entry from your custom library."
            )
            new_idx = names.index(selected_label)
        if new_idx != sel_idx:
            set_selected(int(new_idx))
            sel_idx = int(get_selected_idx())

        if top[1].button("➕", help="Add a blank culture.", key="cust_add", type="secondary", use_container_width=True):
            set_selected(add_new_culture())
            sel_idx = int(get_selected_idx())

        if top[2].button("📄", help="Duplicate selected culture.", key="cust_dup", type="secondary", use_container_width=True):
            set_selected(duplicate_culture(int(sel_idx)))
            sel_idx = int(get_selected_idx())

        if top[3].button("🗑️", help="Delete selected culture (keeps at least one).", key="cust_del", type="secondary", use_container_width=True):
            delete_culture(int(sel_idx))
            sel_idx = int(get_selected_idx())

        with top[4]:
            pop = st.popover("⋯", use_container_width=True)
            with pop:
                st.caption("Import / Export")
                up = st.file_uploader("Import JSON", type=["json"])
                if up is not None:
                    import_catalog_json(up.getvalue())
                    sel_idx = int(get_selected_idx())
                st.download_button(
                    "Export library (JSON)",
                    data=export_catalog_json(),
                    file_name="custom_cultures.json",
                    use_container_width=True
                )

        # Form for selected record
        catalog = get_catalog()
        if not catalog:
            set_selected(add_new_culture())
            catalog = get_catalog()
            sel_idx = int(get_selected_idx())

        rec = dict(catalog[min(max(0, sel_idx), len(catalog) - 1)])

        c1, c2 = st.columns([2, 1], gap="small")
        rec["name"] = c1.text_input("Name", value=str(rec.get("name", "Custom culture")), key="cust_name",
                                    help="A friendly label for this culture.")
        rec["O2_pref"] = c2.selectbox(
            "O₂ preference",
            ["aerobic", "anaerobic", "microaerophilic"],
            index=["aerobic", "anaerobic", "microaerophilic"].index(str(rec.get("O2_pref", "aerobic"))),
            key="cust_o2",
            help="Guides O₂ penalty and default autos."
        )

        c3, c4 = st.columns(2, gap="small")
        rec["T_opt"] = c3.number_input(
            "T_opt (°C)",
            value=float(rec.get("T_opt", 30.0)),
            step=0.5,
            key="cust_T",
            help="Temperature of maximal fitness."
        )
        rec["pH_opt"] = c4.number_input(
            "pH_opt",
            value=float(rec.get("pH_opt", 7.0)),
            step=0.1, format="%.2f",
            key="cust_pH",
            help="pH of maximal fitness."
        )

        with st.expander("Growth references", expanded=False):
            a1, a2, a3 = st.columns(3, gap="small")
            rec["mu_ref_auto"] = a1.checkbox(
                "Auto μ_ref",
                value=bool(rec.get("mu_ref_auto", True)),
                key="cust_mu_auto",
                help="If ON, μ_ref is guessed from O₂ preference; OFF lets you set it."
            )
            rec["K_ref_auto"] = a2.checkbox(
                "Auto K_ref",
                value=bool(rec.get("K_ref_auto", True)),
                key="cust_K_auto",
                help="If ON, K_ref (OD) is guessed; OFF lets you set it."
            )
            rec["lag_ref_auto"] = a3.checkbox(
                "Auto lag_ref",
                value=bool(rec.get("lag_ref_auto", True)),
                key="cust_lag_auto",
                help="If ON, lag_ref (h) is guessed; OFF lets you set it."
            )

            d_mu, d_K, d_lag, d_cfu = default_by_o2(rec.get("O2_pref", "aerobic"))
            b1, b2, b3 = st.columns(3, gap="small")

            mu_val = d_mu if rec["mu_ref_auto"] else float(rec.get("mu_ref", d_mu))
            K_val = d_K if rec["K_ref_auto"] else float(rec.get("K_ref", d_K))
            lag_val = d_lag if rec["lag_ref_auto"] else float(rec.get("lag_ref_h", d_lag))

            if rec["mu_ref_auto"]:
                b1.markdown(f"μ_ref (auto): **{mu_val:.2f} ln/h**")
                rec["mu_ref"] = None
            else:
                rec["mu_ref"] = b1.number_input("μ_ref (ln/h)", value=mu_val, min_value=0.01, step=0.01, key="cust_mu")

            if rec["K_ref_auto"]:
                b2.markdown(f"K_ref (auto): **{K_val:.1f} OD**")
                rec["K_ref"] = None
            else:
                rec["K_ref"] = b2.number_input("K_ref (OD)", value=K_val, min_value=0.2, step=0.1, key="cust_K")

            if rec["lag_ref_auto"]:
                b3.markdown(f"Lag_ref (auto): **{lag_val:.2f} h**")
                rec["lag_ref_h"] = None
            else:
                rec["lag_ref_h"] = b3.number_input("Lag_ref (h)", value=lag_val, min_value=0.0, step=0.1, key="cust_lag")

        with st.expander("Extras (optional)", expanded=False):
            e1, e2, e3 = st.columns(3, gap="small")
            _, _, _, d_cfu = default_by_o2(rec.get("O2_pref", "aerobic"))
            rec["cfu_per_od"] = e1.number_input(
                "CFU per OD600 (heuristic)",
                value=float(rec.get("cfu_per_od", d_cfu)),
                step=1e8, format="%.0f", key="cust_cfu",
                help="Used for OD→CFU conversion in growth plots."
            )
            rec["orp_mV"] = e2.text_input("ORP setpoint (mV)", value=str(rec.get("orp_mV") or ""), key="cust_orp",
                                          help="Optional process control target.")
            rec["electron_acceptor"] = e3.text_input("Electron acceptor", value=str(rec.get("electron_acceptor") or ""), key="cust_ea",
                                                     help="e.g., nitrate, fumarate (optional).")
            rec["notes"] = st.text_area("Notes", value=str(rec.get("notes") or ""), key="cust_notes")

        update_selected(rec)
        org = record_to_preset(rec)

        if st.session_state.get("_last_org_key") != ("Custom", org.name):
            st.session_state["T"] = float(org.T_opt)
            st.session_state["pH"] = float(org.pH_opt)
            st.session_state["O2"] = 21.0 if org.O2_pref.lower() == "aerobic" else 0.5
            st.session_state["_last_org_key"] = ("Custom", org.name)

    # Environment
    with st.expander("Environment", expanded=True):
        c1, c2 = st.columns(2, gap="small")
        T = c1.number_input("Temperature (°C)", value=float(st.session_state.get("T", 30.0)), step=0.5, key="T")
        pH = c2.number_input("pH", value=float(st.session_state.get("pH", 7.0)), step=0.1, format="%.2f", key="pH")
        O2 = st.slider("Dissolved/Headspace O₂ (%)", 0.0, 21.0, float(st.session_state.get("O2", 21.0)), 0.1, key="O2")

    # Medium
    with st.expander("Medium planner", expanded=False):
        m1, m2 = st.columns(2, gap="small")
        carbon_gL = m1.slider("Carbon source (g/L)", 0.0, 30.0, float(st.session_state.get("carbon_gL", 10.0)), 0.5, key="carbon_gL")
        N_mM = m2.slider("Nitrogen (mM)", 0.0, 50.0, float(st.session_state.get("N_mM", 15.0)), 1.0, key="N_mM")
        salts_ok = st.checkbox("Buffer & salts OK (phosphate, Mg²⁺, etc.)", value=bool(st.session_state.get("salts_ok", True)), key="salts_ok")
        trace_ok = st.checkbox("Trace metals OK (Fe, Mo/W, etc.)", value=bool(st.session_state.get("trace_ok", True)), key="trace_ok")
        vitamins_ok = st.checkbox("Vitamins/biotin OK", value=bool(st.session_state.get("vitamins_ok", True)), key="vitamins_ok")
    medium_score = medium_quality_score(carbon_gL, N_mM, salts_ok, trace_ok, vitamins_ok)

    # Activity constants
    with st.expander("Lab constants (activity)", expanded=False):
        c3, c4 = st.columns(2, gap="small")
        epsilon = c3.number_input("ε (mM⁻¹·cm⁻¹)", value=float(st.session_state.get("epsilon", 3.40)),
                                  min_value=0.0001, step=0.01, key="epsilon")
        path_cm = c4.number_input("Path length (cm)", value=float(st.session_state.get("path_cm", 1.00)),
                                  min_value=0.01, step=0.01, key="path_cm")
        c5, c6 = st.columns(2, gap="small")
        V_rxn_mL = c5.number_input("V_rxn (mL)", value=float(st.session_state.get("V_rxn_mL", 1.000)),
                                   min_value=0.01, step=0.01, key="V_rxn_mL")
        V_samp_mL = c6.number_input("V_samp (mL)", value=float(st.session_state.get("V_samp_mL", 0.010)),
                                    min_value=0.001, step=0.001, key="V_samp_mL")

# Resolve active organism
if st.session_state.get("org_source") == "Preset":
    org = ORGANISMS[st.session_state["org_name"]]
else:
    current_catalog = get_catalog()
    idx = int(get_selected_idx())
    org = record_to_preset(current_catalog[min(max(0, idx), len(current_catalog) - 1)])

# Effective params given current environment
env = dict(
    T=float(st.session_state["T"]),
    pH=float(st.session_state["pH"]),
    O2=float(st.session_state["O2"]),
    medium_score=float(medium_score),
    T_opt=float(org.T_opt),
    pH_opt=float(org.pH_opt),
    O2_pref=str(org.O2_pref),
)
mu_eff, K_eff, lag_eff, comps = adjust_params_by_environment(org.mu_ref, org.K_ref, org.lag_ref_h, env)

st.markdown(f"""
<div class="helpbox">
<b>{org.name}</b><br>{org.notes}<br>
<hr style="margin:6px 0"/>
<b>Effective params</b>: μ<sub>max</sub> ≈ <b>{mu_eff:.2f}</b> ln·h⁻¹ • K ≈ <b>{K_eff:.1f}</b> OD • lag ≈ <b>{lag_eff:.2f}</b> h
<br><span class="muted">Fitness components - T:{comps['tp']:.2f} pH:{comps['pp']:.2f} O₂:{comps['op']:.2f} Medium:{comps['mp']:.2f}</span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────
# Tabs
# ─────────────────────────────
labels = [
    "① Culture planning",
    "② Growth simulation",
    "③ Growth monitoring (OD/CFU)",
    "④ Harvest predictor",
    "⑤ Activity assay",
    "⑥ Bradford & Protein",
    "⑦ Purification & KPIs",
    "⑧ Plots & Export",
    "⑨ Settings & Session",
    # "⑩ Dashboard",
    # "⑪ Batch simulation",
]
if _t10_ec is not None:
    labels.append("⑫ Electrochemistry")

tabs = st.tabs(labels)

# shared kwargs into tabs
common = dict(
    mu_eff=mu_eff, K_eff=K_eff, lag_eff=lag_eff, org=org,
    epsilon=st.session_state["epsilon"], path_cm=st.session_state["path_cm"],
    V_rxn_mL=st.session_state["V_rxn_mL"], V_samp_mL=st.session_state["V_samp_mL"],
    env=env, clear_keys_fn=clear_keys, clear_keys=clear_keys,
    organism_name=org.name, culture_name=org.name, culture=org,
)

_safe_render(t01_planning, tabs[0], **common)
_safe_render(t02_simulation, tabs[1], **common)
_safe_render(t03_monitoring, tabs[2], **common)
_safe_render(t04_harvest, tabs[3], **common)
_safe_render(t05_activity, tabs[4], **common)
_safe_render(t06_bradford, tabs[5], **common)
_safe_render(t07_purification, tabs[6], **common)
_safe_render(t08_plots_export, tabs[7], **common)
_safe_render(t09_settings, tabs[8], **common)
# _safe_render(t10_dashboard, tabs[9], **common)
# _safe_render(t11_batch, tabs[10], **common)
if _t10_ec is not None:
    _safe_render(_t10_ec, tabs[-1], **common)
