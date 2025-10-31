# tabs/t10_electrochem.py
from __future__ import annotations

import io, os, math, colorsys, hashlib
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
import matplotlib.colors as mcolors
import streamlit as st

APP_VERSION = "2.8.0"

# ---------- local CSS ----------
def _inject_local_css() -> None:
    st.markdown(
        """
        <style>
        .cv-card { background:#fafbfd;border:1px solid #e6e9f2;border-radius:12px;padding:12px; }
        .cv-help { color:#6b7280;font-size:0.92rem;margin-top:-6px;margin-bottom:8px; }
        .stTabs [data-baseweb="tab-list"] { gap: 0.3rem; }
        .stTabs [data-baseweb="tab"] { padding: 8px 10px; border-radius: 10px; }
        .cv-muted { color:#6b7280; }
        </style>
        """,
        unsafe_allow_html=True,
    )

# ---------- utilities ----------
@st.cache_data(show_spinner=False)
def load_table(file_bytes: bytes) -> np.ndarray:
    try:
        df = pd.read_csv(io.BytesIO(file_bytes), header=None, delim_whitespace=True, engine="python")
    except Exception:
        df = pd.read_csv(io.BytesIO(file_bytes), header=None)
    df = df.dropna(axis=1, how="all")
    return df.to_numpy(dtype=float)

def build_axes(n_rows: int, Ep_mV: float, Ek_mV: float, reverse_mode: str) -> Tuple[np.ndarray, np.ndarray]:
    E_fwd = np.linspace(Ep_mV, Ek_mV, n_rows)
    E_rev = np.linspace(Ek_mV, Ep_mV, n_rows) if reverse_mode.startswith("Ek") else np.linspace(Ep_mV, Ek_mV, n_rows)
    return E_fwd, E_rev

def moving_average(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1: return x.copy()
    if k % 2 == 0: k += 1
    pad = k // 2
    xpad = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(xpad, np.ones(k)/k, mode="valid")

def poly_baseline(E: np.ndarray, I: np.ndarray, regions: Sequence[Tuple[float,float]], order: int = 1) -> np.ndarray:
    mask = np.zeros_like(E, dtype=bool)
    for emin, emax in regions:
        if emin > emax: emin, emax = emax, emin
        mask |= (E >= emin) & (E <= emax)
    if not np.any(mask):
        coeffs = np.polyfit([E[0], E[-1]], [I[0], I[-1]], 1)
        return np.polyval(coeffs, E)
    coeffs = np.polyfit(E[mask], I[mask], order)
    return np.polyval(coeffs, E)

def integrate_charge_uC(E_mV: np.ndarray, I_uA: np.ndarray, scan_rate_mV_s: float) -> float:
    if E_mV.size < 2: return float("nan")
    area = float(np.trapz(np.asarray(I_uA, dtype=float), np.asarray(E_mV, dtype=float)))
    denom = float(scan_rate_mV_s) if scan_rate_mV_s else 1.0
    return float(area / denom)

def ir_correct_E(E_mV: np.ndarray, I_uA: np.ndarray, R_ohm: float) -> np.ndarray:
    return np.asarray(E_mV, dtype=float) - (np.asarray(I_uA, dtype=float) * float(R_ohm)) / 1000.0

def normalize_current(I_uA: np.ndarray, area_cm2: Optional[float], scan_rate_mV_s: float, mode: str) -> Tuple[np.ndarray,str]:
    I_uA = np.asarray(I_uA, dtype=float)
    if mode == "Current (µA)":
        return I_uA.copy(), "I (µA)"
    if mode == "Current density (µA/cm²)":
        if not area_cm2 or area_cm2 <= 0: return I_uA.copy(), "I (µA)"
        return I_uA/float(area_cm2), "J (µA/cm²)"
    v_Vs = (float(scan_rate_mV_s) if scan_rate_mV_s else 1.0)/1000.0
    return I_uA / v_Vs, "C (µF)"

ColorLike = Union[str, Tuple[float,float,float], Tuple[float,float,float,float]]

def lighten_color(color: ColorLike, amount: float = 0.35) -> Tuple[float,float,float]:
    try:
        r,g,b = mcolors.to_rgb(color)
    except Exception:
        r,g,b = (0.0,0.0,0.0)
    h,l,s = colorsys.rgb_to_hls(r,g,b)
    l = 1 - amount*(1-l)
    r2,g2,b2 = colorsys.hls_to_rgb(h,l,s)
    return (float(r2), float(g2), float(b2))

# ---------- analysis helpers ----------
def _peak_metrics(E: np.ndarray, I: np.ndarray) -> Tuple[Optional[Tuple[float,float]], Optional[Tuple[float,float]]]:
    """Return (anodic_peak(E,I), cathodic_peak(E,I)) as (E_mV, I_uA) tuples or None."""
    n = len(E)
    if n < 10: return None, None
    lo = int(0.03*n); hi = int(0.97*n)
    Iseg = I[lo:hi]; Eseg = E[lo:hi]
    if Iseg.size == 0: return None, None
    ia = int(np.argmax(Iseg)); ic = int(np.argmin(Iseg))
    anod = (float(Eseg[ia]), float(Iseg[ia])) if Iseg[ia] != 0 else None
    cath = (float(Eseg[ic]), float(Iseg[ic])) if Iseg[ic] != 0 else None
    return anod, cath

def _loop_area_uC(Ef: np.ndarray, If: np.ndarray, Er: np.ndarray, Ir: np.ndarray, v_mV_s: float) -> float:
    """Area between forward and reverse branches, normalized by scan rate -> microcoulombs."""
    Ef_s, If_s = np.array(Ef), np.array(If)
    order = np.argsort(Ef_s)
    Ef_s, If_s = Ef_s[order], If_s[order]
    Er_s, Ir_s = np.array(Er), np.array(Ir)
    Emin, Emax = max(Ef_s.min(), Er_s.min()), min(Ef_s.max(), Er_s.max())
    mask = (Ef_s >= Emin) & (Ef_s <= Emax)
    if np.sum(mask) < 3: return float("nan")
    Ir_interp = np.interp(Ef_s[mask], Er_s, Ir_s)
    diff = np.abs(If_s[mask] - Ir_interp)
    area = float(np.trapz(diff, Ef_s[mask]))
    denom = float(v_mV_s) if v_mV_s else 1.0
    return float(area / denom)

def _reversibility_call(delta_Ep_mV: float) -> str:
    if not (delta_Ep_mV == delta_Ep_mV):  # NaN
        return "insufficient peaks for reversibility estimate"
    if delta_Ep_mV <= 40:
        return "faster than Nernstian (likely noise/overlap)"
    if 40 < delta_Ep_mV <= 80:
        return "near-reversible (n≈1 at 25 °C)"
    if 80 < delta_Ep_mV <= 150:
        return "quasi-reversible (kinetic limitations likely)"
    return "irreversible/strongly sluggish electron transfer"

def _midzone_drift(I: np.ndarray) -> float:
    n = len(I)
    if n < 10: return 0.0
    a = np.median(I[:max(3, n//10)])
    b = np.median(I[-max(3, n//10):])
    return float(b - a)

def _make_report_md(title: str, items: List[str]) -> str:
    lines = [f"# {title}", ""]
    for s in items:
        lines.append(f"- {s}")
    return "\n".join(lines) + "\n"

# ---------- main render ----------
def render(container):
    _inject_local_css()
    with container:
        st.subheader("Electrochemistry — Cyclic Voltammetry (CV) Plotter Pro")
        st.caption("Upload a table with 2×N columns (forward, reverse per cycle).")

        col_ctrl, col_plot = st.columns([1.05, 1.55], gap="large")

        # --- dataset library state ---
        if "cv_files" not in st.session_state:
            # Each entry: {"id": int, "name": str, "alias": str, "data": np.ndarray, "sig": str}
            st.session_state.cv_files = []
            st.session_state.cv_next_id = 1
        if "cv_active_id" not in st.session_state:
            st.session_state.cv_active_id = None
        if "cv_seen_keys" not in st.session_state:
            st.session_state.cv_seen_keys = set()

        def _rerun():
            fn = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
            if callable(fn): fn()

        def _signature(bytes_blob: bytes, name: str) -> str:
            h = hashlib.md5(bytes_blob).hexdigest()
            return f"{h}:{name}:{len(bytes_blob)}"

        def _add_files_to_library(uploaded_files: Optional[List[Any]]):
            for f in uploaded_files or []:
                b = f.getvalue()
                sig = _signature(b, f.name)
                if sig in st.session_state.cv_seen_keys:
                    continue
                try:
                    arr = load_table(b)
                except Exception:
                    continue
                stem = os.path.splitext(f.name)[0]
                st.session_state.cv_files.append({
                    "id": st.session_state.cv_next_id,
                    "name": f.name,
                    "alias": stem,
                    "data": arr,
                    "sig": sig,
                })
                st.session_state.cv_next_id += 1
                st.session_state.cv_seen_keys.add(sig)

        def _delete_file(file_id: int):
            keep = []
            removed_sig = None
            for x in st.session_state.cv_files:
                if x["id"] == file_id:
                    removed_sig = x.get("sig")
                else:
                    keep.append(x)
            st.session_state.cv_files = keep
            if removed_sig:
                st.session_state.cv_seen_keys.discard(removed_sig)

        def _active_ids() -> List[int]:
            return [x["id"] for x in st.session_state.cv_files]

        # ---- controls
        with col_ctrl:
            # ---- Data (multi-file library)
            with st.container(border=True):
                st.markdown("#### Data")

                up_list = st.file_uploader(
                    label="Upload TXT/CSV (columns: fwd, rev, fwd, rev, ...)",
                    type=["txt", "csv", "dat"],
                    accept_multiple_files=True,
                    key="cv_up_multi",
                )
                st.markdown(
                    '<div class="cv-help">Each adjacent column pair should form one CV cycle. '
                    'You can add multiple files; they persist below until deleted. Re-uploads are deduplicated.</div>',
                    unsafe_allow_html=True,
                )

                if up_list:
                    _add_files_to_library(up_list)

                if st.session_state.cv_files:
                    st.markdown("**Loaded files**")
                    ids = [e["id"] for e in st.session_state.cv_files]
                    names_by_id = {e["id"]: e["name"] for e in st.session_state.cv_files}
                    alias_by_id = {e["id"]: e["alias"] for e in st.session_state.cv_files}
                    if st.session_state.cv_active_id not in ids:
                        st.session_state.cv_active_id = ids[0]

                    selected_id = st.radio(
                        "Active dataset",
                        ids,
                        index=ids.index(st.session_state.cv_active_id),
                        format_func=lambda _id: f"{alias_by_id[_id]}  ·  {names_by_id[_id]}",
                        horizontal=False,
                        key="cv_active_radio",
                    )
                    st.session_state.cv_active_id = selected_id

                    hdr = st.columns([0.38, 0.22, 0.20, 0.20])
                    hdr[0].write("**Filename**"); hdr[1].write("**Alias**"); hdr[2].write("**Shape**"); hdr[3].write("**Actions**")

                    to_delete: Optional[int] = None
                    for entry in st.session_state.cv_files:
                        c1, c2, c3, c4 = st.columns([0.38, 0.22, 0.20, 0.20])
                        c1.write(entry["name"])
                        raw_alias = c2.text_input(
                            "",
                            value=entry["alias"],
                            key=f"cv_alias_{entry['id']}",
                            label_visibility="collapsed",
                        )
                        sanitized = (raw_alias or "").strip()
                        if sanitized and sanitized != entry["alias"]:
                            entry["alias"] = sanitized
                        c3.write(f"{entry['data'].shape}")
                        if c4.button("Delete", key=f"cv_del_{entry['id']}"):
                            to_delete = entry["id"]

                    if to_delete is not None:
                        _delete_file(to_delete)
                        remaining = _active_ids()
                        st.session_state.cv_active_id = remaining[0] if remaining else None
                        _rerun()
                else:
                    st.info("No files loaded yet.")

                # Numeric inputs
                cEp, cEk = st.columns(2)
                Ep_mV = cEp.number_input("Ep (mV)", value=-400.0, step=10.0, format="%.1f")
                Ek_mV = cEk.number_input("Ek (mV)", value=500.0, step=10.0, format="%.1f")
                cScan, cOrder = st.columns(2)
                scan_rate = cScan.number_input("Scan rate v (mV/s)", value=25.0, step=5.0, format="%.1f")
                reverse_mode = cOrder.selectbox("Reverse column ordering", ["Ek→Ep (descending)", "Ep→Ek (ascending)"], index=1)

            with st.container(border=True):
                st.markdown("#### What to show")
                show_forward = st.checkbox("Show forward", value=True)
                show_reverse = st.checkbox("Show reverse", value=True)
                show_concat  = st.checkbox("Show concatenated loop", value=False)

            with st.container(border=True):
                st.markdown("#### Labels & legend")
                legend_scheme = st.radio("Legend entries", ["Per-branch","Per-cycle (combined)","Per-group"], index=1)
                inline_labels = st.checkbox("Inline labels on curves", value=False)
                cPos, cSize = st.columns(2)
                inline_pos  = cPos.selectbox("Inline label position", ["Right vertex","Left vertex"], index=0)
                inline_font = cSize.slider("Inline label size", 6, 14, 8, 1)
                show_legend_all = st.checkbox("Show legend on all plots", value=True)
                legend_place = st.selectbox("Legend placement", ["Best","Outside Right","None"], index=0)

            with st.container(border=True):
                st.markdown("#### Style & axes")
                palette = st.selectbox("Palette", ["Lab (Tab10)","Set2","B/W + highlight"], index=0)
                grid_on = st.checkbox("Grid", value=True)
                zero_line = st.checkbox("Zero current line", value=True)
                fw_lw = st.slider("Line width (forward)", 0.5, 4.0, 2.0, 0.1)
                rv_lw = st.slider("Line width (reverse)", 0.5, 4.0, 2.0, 0.1)
                rv_style = st.selectbox("Reverse line style", ["dashed","dotted","solid"], index=0)
                manual_ylim = st.checkbox("Manual Y limits", value=False)
                cY1, cY2 = st.columns(2)
                ymin = cY1.number_input("Ymin", value=-35.0, step=1.0, format="%.1f")
                ymax = cY2.number_input("Ymax", value=25.0, step=1.0, format="%.1f")

            with st.container(border=True):
                st.markdown("#### Processing")
                Rs = st.number_input("Series resistance Rs (Ω)", value=0.0, step=10.0, format="%.1f")
                area_cm2 = st.number_input("Electrode area (cm², optional)", value=0.0, step=0.01, format="%.2f")
                norm_mode = st.selectbox("Y-normalization", ["Current (µA)","Current density (µA/cm²)","Capacitance est. C (µF)"], index=0)
                smooth_k = st.slider("Smoothing window (points; odd)", 1, 101, 1, 2)

            with st.container(border=True):
                st.markdown("#### Baseline (Δ)")
                use_baseline = st.checkbox("Enable Δ subtraction", value=False)
                baseline_mode = st.radio("Baseline source", ["Cycle index","Average of first N cycles","Polynomial from regions"])
                cB1, cB2 = st.columns(2)
                baseline_cycle = cB1.number_input("Baseline cycle (1-based)", min_value=1, value=1, step=1)
                baseline_avgN  = cB2.number_input("Average first N cycles", min_value=1, value=1, step=1)
                left_region = st.text_input("Baseline regions [mV] (comma-separated ranges a:b)", value="-400:-300, 350:500")
                poly_order = st.slider("Polynomial order", 1, 3, 1)

            # ---- Compare mode controls
            with st.container(border=True):
                st.markdown("#### Compare")
                cmp_enable = st.checkbox("Enable compare mode", value=False, key="cv_cmp_enable")
                cmp_ids: List[int] = []
                cmp_layout = "Overlay (single axes)"
                color_by_dataset = True
                if cmp_enable and st.session_state.cv_files:
                    all_ids = [e["id"] for e in st.session_state.cv_files]
                    alias_by_id = {e["id"]: e["alias"] for e in st.session_state.cv_files}
                    default_ids = [st.session_state.cv_active_id] if st.session_state.cv_active_id else all_ids[:1]
                    cmp_ids = st.multiselect("Datasets to include (max 4)", all_ids, default=default_ids, format_func=lambda _id: alias_by_id[_id])
                    if len(cmp_ids) > 4:
                        cmp_ids = cmp_ids[:4]
                    cmp_layout = st.radio("Layout", ["Overlay (single axes)", "Grid (one per dataset)"], index=0, horizontal=True)
                    color_by_dataset = st.checkbox("Color by dataset (override per-cycle colors)", value=True)

            with st.container(border=True):
                st.markdown("#### Reference & event")
                ref_E_obs = st.number_input("Marker observed E (mV)", value=float("nan"), step=5.0, format="%.1f")
                ref_E_true = st.number_input("Marker true E vs ref (mV)", value=float("nan"), step=5.0, format="%.1f")
                show_vertex = st.checkbox("Show vertex shading", value=True)
                vertex_pct = st.slider("Vertex band (% of range)", 0, 10, 3)

        # ---- require an active dataset
        if st.session_state.cv_active_id is None:
            with col_plot:
                st.info("Add files above and choose one as the active dataset to begin.")
            return

        # ---- helpers
        def calibration_offset(obs: float, true: float) -> float:
            if math.isnan(obs) or math.isnan(true): return 0.0
            return true - obs
        dE_cal = calibration_offset(ref_E_obs, ref_E_true)

        def compute_mats_for(data_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str, int, int]:
            if data_raw.ndim != 2 or (data_raw.shape[1] % 2 != 0):
                return np.zeros((0,0)), np.zeros((0,0)), "I (µA)", 0, 0
            n_rows, n_cols = data_raw.shape
            n_cycles = n_cols // 2
            E_fwd, E_rev = build_axes(n_rows, Ep_mV, Ek_mV, reverse_mode)
            data_proc = data_raw.copy()
            if smooth_k > 1:
                for j in range(n_cols):
                    data_proc[:, j] = moving_average(data_proc[:, j], smooth_k)
            F = np.empty_like(data_proc, dtype=float)
            E_out = np.empty_like(data_proc, dtype=float)
            yname = "I (µA)"
            for i in range(n_cycles):
                If = data_proc[:, 2*i]; Ir = data_proc[:, 2*i+1]
                Ef = E_fwd.copy(); Er = E_rev.copy()
                if Rs != 0.0:
                    Ef = ir_correct_E(Ef, If, Rs); Er = ir_correct_E(Er, Ir, Rs)
                Ef = Ef + dE_cal; Er = Er + dE_cal
                IfN, yname = normalize_current(If, area_cm2 if area_cm2>0 else None, scan_rate, norm_mode)
                IrN, _     = normalize_current(Ir, area_cm2 if area_cm2>0 else None, scan_rate, norm_mode)
                F[:, 2*i] = IfN; F[:, 2*i+1] = IrN
                E_out[:, 2*i] = Ef; E_out[:, 2*i+1] = Er
            return E_out, F, yname, n_rows, n_cycles

        # ---- active dataset mats
        _active = next(x for x in st.session_state.cv_files if x["id"] == st.session_state.cv_active_id)
        active_alias = _active["alias"]
        E_mat, I_mat, yname_loc, n_rows, n_cycles = compute_mats_for(_active["data"])
        if n_cycles == 0:
            col_plot.error(f"Expected an even number of columns (pairs of fwd/rev). Found shape {_active['data'].shape}."); return

        # ---- cycle visibility UI (compact)
        with col_ctrl:
            with st.container(border=True):
                st.markdown("#### Cycle visibility")

                mode = st.radio(
                    "Show",
                    ["All", "First N", "Range", "Odd", "Even", "Custom"],
                    horizontal=True,
                    key="cv_vis_mode",
                    index=0
                )

                N_default = st.session_state.get("cv_vis_firstN", min(6, n_cycles))
                A_default = st.session_state.get("cv_vis_range_a", 1)
                B_default = st.session_state.get("cv_vis_range_b", min(6, n_cycles))
                custom_default = st.session_state.get("cv_visible_custom", list(range(1, min(6, n_cycles)+1)))

                if mode == "First N":
                    N = st.number_input("N", 1, n_cycles, value=int(N_default), step=1)
                    st.session_state.cv_vis_firstN = int(N)
                elif mode == "Range":
                    c1, c2 = st.columns(2)
                    A = c1.number_input("From", 1, n_cycles, value=int(A_default), step=1)
                    B = c2.number_input("To",   1, n_cycles, value=int(B_default), step=1)
                    st.session_state.cv_vis_range_a = int(A)
                    st.session_state.cv_vis_range_b = int(B)
                elif mode == "Custom":
                    st.session_state.cv_visible_custom = st.multiselect(
                        "Pick cycles",
                        options=list(range(1, n_cycles+1)),
                        default=custom_default,
                    )

            # Label manager & group editors (kept)
            with st.container(border=True):
                st.markdown("#### Label manager")
                cLM1, cLM2 = st.columns(2)
                with cLM1:
                    st.caption("Quick rename")
                    base = st.text_input("Base name", value="Cycle", key="cv_base")
                    start = st.number_input("Start index", value=1, step=1, key="cv_start")
                    # Apply to current visible cycles
                    def _active_cycles_for_labels() -> List[int]:
                        return []  # not used directly; legacy kept minimal
                    tgt = st.multiselect(
                        "Apply to cycles",
                        options=list(range(1, n_cycles+1)),
                        default=list(range(1, n_cycles+1)),
                        key="cv_apply_cycles"
                    )
                    if st.button("Apply pattern", key="cv_apply_btn"):
                        if "cv_legend_df" not in st.session_state or st.session_state.cv_legend_df.shape[0] != n_cycles:
                            st.session_state.cv_legend_df = pd.DataFrame([
                                {"Cycle": i+1, "Legend fwd": f"C{i+1} fwd", "Legend rev": f"C{i+1} rev"} for i in range(n_cycles)
                            ])
                        for k, cyc in enumerate(sorted(tgt)):
                            namep = f"{base} {start+k}"
                            st.session_state.cv_legend_df.loc[st.session_state.cv_legend_df["Cycle"]==cyc, "Legend fwd"] = namep + " fwd"
                            st.session_state.cv_legend_df.loc[st.session_state.cv_legend_df["Cycle"]==cyc, "Legend rev"] = namep + " rev"
                with cLM2:
                    st.caption("Groups (for Per-group legend)")
                    if "cv_groups_df" not in st.session_state:
                        st.session_state.cv_groups_df = pd.DataFrame([
                            {"Group":"Blank","Cycles":"1-3","Legend":"Blank"},
                            {"Group":"4-HNE","Cycles":"4","Legend":"4-HNE + enzyme"}
                        ])
                    st.session_state.cv_groups_df = st.data_editor(
                        st.session_state.cv_groups_df, num_rows="dynamic", use_container_width=True, key="cv_groups_editor"
                    )

                st.markdown("Custom legend names editor")
                if "cv_legend_df" not in st.session_state or st.session_state.cv_legend_df.shape[0] != n_cycles:
                    st.session_state.cv_legend_df = pd.DataFrame([
                        {"Cycle": i+1, "Legend fwd": f"C{i+1} fwd", "Legend rev": f"C{i+1} rev"} for i in range(n_cycles)
                    ])
                st.session_state.cv_legend_df = st.data_editor(
                    st.session_state.cv_legend_df, num_rows="dynamic", use_container_width=True, key="cv_legend_editor"
                )

                with st.expander("Advanced: per-cycle color overrides (hex or named)"):
                    if "cv_color_df" not in st.session_state or st.session_state.cv_color_df.shape[0] != n_cycles:
                        st.session_state.cv_color_df = pd.DataFrame([{"Cycle": i+1, "Color": ""} for i in range(n_cycles)])
                    st.session_state.cv_color_df = st.data_editor(
                        st.session_state.cv_color_df, num_rows="dynamic", use_container_width=True, key="cv_color_editor"
                    )

        # ---- visibility resolver
        def active_cycles() -> List[int]:
            m = st.session_state.get("cv_vis_mode", "All")
            if m == "All":
                return list(range(1, n_cycles+1))
            if m == "First N":
                N = int(st.session_state.get("cv_vis_firstN", n_cycles))
                return list(range(1, min(max(N,1), n_cycles)+1))
            if m == "Range":
                a = int(st.session_state.get("cv_vis_range_a", 1))
                b = int(st.session_state.get("cv_vis_range_b", n_cycles))
                lo, hi = min(a,b), max(a,b)
                lo = max(1, lo); hi = min(n_cycles, hi)
                return list(range(lo, hi+1))
            if m == "Odd":
                return [i for i in range(1, n_cycles+1) if i % 2 == 1]
            if m == "Even":
                return [i for i in range(1, n_cycles+1) if i % 2 == 0]
            # Custom
            return [c for c in st.session_state.get("cv_visible_custom", []) if 1 <= c <= n_cycles]

        # ---- legend/color helpers
        color_override: Dict[int, Optional[str]] = {}
        if "cv_color_df" in st.session_state:
            for _, r in st.session_state.cv_color_df.iterrows():
                cyc = int(r.get("Cycle", 0) or 0)
                col_raw = r.get("Color", "")
                col = (str(col_raw) if col_raw is not None else "").strip()
                color_override[cyc] = col or None

        def legend_text(cycle_idx: int, branch: str) -> str:
            row = st.session_state.cv_legend_df[st.session_state.cv_legend_df["Cycle"]==cycle_idx]
            if row.empty: return f"C{cycle_idx} " + ("fwd" if branch=="fwd" else "rev")
            return str(row.iloc[0]["Legend fwd" if branch=="fwd" else "Legend rev"])

        def build_group_map() -> List[Tuple[str, Set[int], str]]:
            def parse(txt: str) -> Set[int]:
                out=set()
                for part in (txt or "").split(","):
                    p=part.strip()
                    if not p: continue
                    if "-" in p:
                        a,b=p.split("-"); a=int(a); b=int(b)
                        for x in range(min(a,b), max(a,b)+1): out.add(x)
                    else: out.add(int(p))
                return out
            groups=[]
            for _, r in st.session_state.cv_groups_df.iterrows():
                try:
                    groups.append((str(r["Group"]), parse(str(r["Cycles"])), str(r["Legend"])))
                except Exception:
                    pass
            return groups

        def cycle_color(cycle_idx: int, branch: str):
            override = color_override.get(cycle_idx)
            if override:
                base = override
            else:
                if palette == "B/W + highlight":
                    hl = set(int(x) for x in st.session_state.cv_meta_df.loc[st.session_state.cv_meta_df.get("Highlight", False)==True, "Cycle"]) if "cv_meta_df" in st.session_state else set()
                    base = (0,0,0) if (cycle_idx in hl) else (0.33,0.33,0.33)
                elif palette == "Set2":
                    base = plt.get_cmap("Set2")((cycle_idx-1) % 8)
                else:
                    base = plt.get_cmap("tab10")((cycle_idx-1) % 10)
            if branch == "rev" and palette != "B/W + highlight":
                return lighten_color(base, 0.45)
            return base

        def plot_range(ax):
            base_min = min(Ep_mV, Ek_mV) + (0.0 if math.isnan(ref_E_obs) or math.isnan(ref_E_true) else (ref_E_true-ref_E_obs))
            base_max = max(Ep_mV, Ek_mV) + (0.0 if math.isnan(ref_E_obs) or math.isnan(ref_E_true) else (ref_E_true-ref_E_obs))
            ax.set_xlim(base_min, base_max)
            if manual_ylim: ax.set_ylim(ymin, ymax)

        def draw_vertex_shading(ax):
            if not show_vertex: return
            base_min = min(Ep_mV, Ek_mV)
            base_max = max(Ep_mV, Ek_mV)
            rng = base_max - base_min
            if vertex_pct > 0:
                band = vertex_pct/100.0 * rng
                ax.axvspan(base_min, base_min+band, color="0.92", alpha=0.5, zorder=0)
                ax.axvspan(base_max-band, base_max, color="0.92", alpha=0.5, zorder=0)

        def place_legend(ax):
            if legend_place == "None": return
            ls_map = {"dashed":"--","dotted":":","solid":"-"}
            if legend_scheme == "Per-branch":
                handles, labels_list = ax.get_legend_handles_labels()
                seen=set(); new_h=[]; new_l=[]
                for h,l in zip(handles, labels_list):
                    if not l or l in seen: continue
                    seen.add(l); new_h.append(h); new_l.append(l)
                if legend_place == "Outside Right":
                    ax.legend(new_h, new_l, bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=8, frameon=False)
                else:
                    ax.legend(new_h, new_l, ncol=2, fontsize=8, loc="best", frameon=False)
            else:
                proxies=[]; labels_out=[]
                if legend_scheme == "Per-cycle (combined)":
                    for c in active_cycles():
                        col_f = cycle_color(c,"fwd"); col_r = cycle_color(c,"rev")
                        h1 = Line2D([], [], color=col_f, linestyle="-", lw=fw_lw)
                        h2 = Line2D([], [], color=col_r, linestyle={"dashed":"--","dotted":":","solid":"-"}.get(rv_style,"--"), lw=rv_lw)
                        name = legend_text(c,"fwd").replace(" fwd","").replace(" rev","")
                        proxies.append((h1,h2)); labels_out.append(name)
                else:
                    for gname, gset, glabel in build_group_map():
                        inter = [c for c in active_cycles() if c in gset]
                        if not inter: continue
                        c = inter[0]
                        col_f = cycle_color(c,"fwd"); col_r = cycle_color(c,"rev")
                        h1 = Line2D([], [], color=col_f, linestyle="-", lw=fw_lw)
                        h2 = Line2D([], [], color=col_r, linestyle={"dashed":"--","dotted":":","solid":"-"}.get(rv_style,"--"), lw=rv_lw)
                        proxies.append((h1,h2)); labels_out.append(glabel or gname)
                if proxies:
                    if legend_place == "Outside Right":
                        ax.legend(proxies, labels_out, handler_map={tuple: HandlerTuple(None)}, bbox_to_anchor=(1.02,1.0), loc="upper left", fontsize=8, frameon=False)
                    else:
                        ax.legend(proxies, labels_out, handler_map={tuple: HandlerTuple(None)}, ncol=2, fontsize=8, loc="best", frameon=False)

        # ---- plots
        with col_plot:
            # ----- COMPARE VIEW
            if cmp_enable and cmp_ids and len(cmp_ids) >= 2:
                alias_by_id = {e["id"]: e["alias"] for e in st.session_state.cv_files}
                data_by_id  = {e["id"]: e["data"] for e in st.session_state.cv_files}
                ds_cmap = plt.get_cmap("tab10")
                ls_map = {"dashed": "--", "dotted": ":", "solid": "-"}

                if cmp_layout.startswith("Overlay"):
                    st.subheader("Compare — Overlay")
                    figC, axC = plt.subplots(figsize=(10, 6.5), dpi=150)
                    axC.set_facecolor("white")
                    draw_vertex_shading(axC)

                    for k, dsid in enumerate(cmp_ids):
                        alias = alias_by_id[dsid]
                        base_col = ds_cmap(k % 10)
                        Em, Im, _, _, nc = compute_mats_for(data_by_id[dsid])
                        if Em.size == 0: continue
                        vis = [c for c in active_cycles() if 1 <= c <= nc]

                        def ds_col(branch: str):
                            return base_col if branch == "fwd" else lighten_color(base_col, 0.45)

                        for i in vis:
                            if show_forward:
                                axC.plot(
                                    Em[:, 2*(i-1)], Im[:, 2*(i-1)],
                                    lw=fw_lw,
                                    color=(ds_col("fwd") if color_by_dataset else cycle_color(i,"fwd")),
                                    linestyle="-",
                                )
                            if show_reverse:
                                axC.plot(
                                    Em[:, 2*(i-1)+1], Im[:, 2*(i-1)+1],
                                    lw=rv_lw,
                                    color=(ds_col("rev") if color_by_dataset else cycle_color(i,"rev")),
                                    linestyle=ls_map.get(rv_style, "--"),
                                )
                            if show_concat and show_forward and show_reverse:
                                Ecat = np.concatenate([Em[:, 2*(i-1)], Em[1:, 2*(i-1)+1]])
                                Icat = np.concatenate([Im[:, 2*(i-1)], Im[1:, 2*(i-1)+1]])
                                axC.plot(
                                    Ecat, Icat, lw=fw_lw*0.9, alpha=0.8,
                                    color=(ds_col("fwd") if color_by_dataset else cycle_color(i,"fwd")),
                                    linestyle="-",
                                )

                    axC.set_xlabel("E (mV)"); axC.set_ylabel(yname_loc); axC.set_title("Overlay comparison")
                    if grid_on: axC.grid(True, alpha=0.25)
                    if zero_line: axC.axhline(0, color="0.5", lw=0.7, alpha=0.6, zorder=0)
                    plot_range(axC)

                    # dataset-level legend
                    proxies, labels = [], []
                    for k, dsid in enumerate(cmp_ids):
                        alias = alias_by_id[dsid]
                        base_col = ds_cmap(k % 10)
                        fwd = Line2D([], [], color=(base_col if color_by_dataset else "0.2"), linestyle="-", lw=fw_lw)
                        rev = Line2D([], [], color=(lighten_color(base_col, 0.45) if color_by_dataset else "0.5"),
                                     linestyle=ls_map.get(rv_style, "--"), lw=rv_lw)
                        proxies.append((fwd, rev)); labels.append(alias)

                    if legend_place != "None":
                        if legend_place == "Outside Right":
                            axC.legend(proxies, labels, handler_map={tuple: HandlerTuple(None)},
                                       bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=9, frameon=False)
                        else:
                            axC.legend(proxies, labels, handler_map={tuple: HandlerTuple(None)},
                                       loc="best", fontsize=9, frameon=False)

                    st.pyplot(figC, clear_figure=True)

                else:
                    st.subheader("Compare — Grid")
                    n = len(cmp_ids)
                    ncols = 2 if n >= 2 else 1
                    nrows = int(np.ceil(n / ncols))
                    figG, axesG = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.0*ncols, 3.2*nrows), dpi=150, squeeze=False)
                    for ax, dsid in zip(axesG.ravel(), cmp_ids):
                        alias = alias_by_id[dsid]
                        Em, Im, _, _, nc = compute_mats_for(data_by_id[dsid])
                        if Em.size == 0:
                            ax.axis("off"); continue
                        vis = [c for c in active_cycles() if 1 <= c <= nc]
                        for i in vis:
                            if show_forward:
                                ax.plot(Em[:, 2*(i-1)], Im[:, 2*(i-1)], lw=fw_lw, color=cycle_color(i,"fwd"), linestyle="-")
                            if show_reverse:
                                ax.plot(Em[:, 2*(i-1)+1], Im[:, 2*(i-1)+1], lw=rv_lw, color=cycle_color(i,"rev"),
                                        linestyle={"dashed":"--","dotted":":","solid":"-"}.get(rv_style,"--"))
                        ax.set_title(alias, fontsize=10)
                        ax.grid(grid_on, alpha=0.25)
                        if zero_line: ax.axhline(0, color="0.5", lw=0.7, alpha=0.6, zorder=0)
                        plot_range(ax)
                    for ax in axesG.ravel()[len(cmp_ids):]:
                        ax.axis('off')
                    figG.tight_layout()
                    st.pyplot(figG, clear_figure=True)

            # ----- SINGLE DATASET VIEW
            st.markdown(f"#### Legend preview — **Active file:** `{active_alias}`")
            prev_fig, prev_ax = plt.subplots(figsize=(4.2,1.8), dpi=150)
            prev_ax.axis("off")
            ls_map = {"dashed":"--","dotted":":","solid":"-"}
            if active_cycles():
                c = active_cycles()[0]
                if legend_scheme == "Per-branch":
                    h1 = Line2D([], [], color=cycle_color(c,"fwd"), linestyle="-", lw=fw_lw, label=legend_text(c,"fwd"))
                    h2 = Line2D([], [], color=cycle_color(c,"rev"), linestyle=ls_map.get(rv_style,"--"), lw=rv_lw, label=legend_text(c,"rev"))
                    prev_ax.legend(handles=[h1,h2], loc="center", ncol=2, frameon=False)
                else:
                    proxies=[]; labels_out=[]
                    if legend_scheme == "Per-cycle (combined)":
                        for cyc in active_cycles()[:2]:
                            h1 = Line2D([], [], color=cycle_color(cyc,"fwd"), linestyle="-", lw=fw_lw)
                            h2 = Line2D([], [], color=cycle_color(cyc,"rev"), linestyle=ls_map.get(rv_style,"--"), lw=rv_lw)
                            proxies.append((h1,h2)); labels_out.append(legend_text(cyc,"fwd").replace(" fwd",""))
                    else:
                        for gname, gset, glabel in [*build_group_map()][:2]:
                            inter = [cyc for cyc in active_cycles() if cyc in gset]
                            if not inter: continue
                            cyc = inter[0]
                            h1 = Line2D([], [], color=cycle_color(cyc,"fwd"), linestyle="-", lw=fw_lw)
                            h2 = Line2D([], [], color=cycle_color(cyc,"rev"), linestyle=ls_map.get(rv_style,"--"), lw=rv_lw)
                            proxies.append((h1,h2)); labels_out.append(glabel or gname)
                    if proxies:
                        prev_ax.legend(proxies, labels_out, handler_map={tuple: HandlerTuple(None)}, loc="center", ncol=1, frameon=False)
            st.pyplot(prev_fig, clear_figure=True)

            st.markdown("---")
            st.subheader("Clear raw plot")
            def plot_raw_clean():
                fig, ax = plt.subplots(figsize=(9,6), dpi=150)
                draw_vertex_shading(ax)
                for i in active_cycles():
                    if show_forward:
                        ax.plot(E_mat[:, 2*(i-1)], I_mat[:, 2*(i-1)], lw=fw_lw, color=cycle_color(i,"fwd"),
                                linestyle="-", label=legend_text(i,"fwd") if legend_scheme=="Per-branch" else None)
                    if show_reverse:
                        ax.plot(E_mat[:, 2*(i-1)+1], I_mat[:, 2*(i-1)+1], lw=rv_lw, color=cycle_color(i,"rev"),
                                linestyle=ls_map.get(rv_style,"--"), label=legend_text(i,"rev") if legend_scheme=="Per-branch" else None)
                    if show_concat and show_forward and show_reverse:
                        Ecat = np.concatenate([E_mat[:, 2*(i-1)], E_mat[1:, 2*(i-1)+1]])
                        Icat = np.concatenate([I_mat[:, 2*(i-1)], I_mat[1:, 2*(i-1)+1]])
                        ax.plot(Ecat, Icat, lw=fw_lw*0.9, alpha=0.7, color=cycle_color(i,"fwd"), linestyle="-")
                plot_range(ax); ax.set_xlabel("E (mV)"); ax.set_ylabel(yname_loc); ax.set_title(f"Raw | {n_cycles} cycles, v={scan_rate:.1f} mV/s")
                if grid_on: ax.grid(True, alpha=0.25)
                if zero_line: ax.axhline(0, color="0.5", lw=0.7, alpha=0.6, zorder=0)
                if show_legend_all: place_legend(ax)
                return fig

            fig_clean = plot_raw_clean()
            st.pyplot(fig_clean, clear_figure=True)

            st.subheader("Additional plots")
            tabs = st.tabs(["Δ (baseline)", "dI/dE", "Forward-only", "Reverse-only", "Small multiples"])

            with tabs[0]:
                I_delta: Optional[np.ndarray] = None
                if use_baseline:
                    if baseline_mode == "Cycle index":
                        idx = max(1, int(baseline_cycle)) - 1; idx = min(idx, n_cycles-1)
                        base = I_mat[:, 2*idx:2*idx+2]
                        I_delta = np.zeros_like(I_mat)
                        for i in range(n_cycles):
                            I_delta[:, 2*i]   = I_mat[:, 2*i]   - base[:, 0]
                            I_delta[:, 2*i+1] = I_mat[:, 2*i+1] - base[:, 1]
                    elif baseline_mode == "Average of first N cycles":
                        N = max(1, min(int(baseline_avgN), n_cycles))
                        base = np.mean(I_mat[:, :2*N], axis=1).reshape(-1,1)
                        I_delta = I_mat - np.repeat(base, I_mat.shape[1], axis=1)
                    else:
                        def parse_regions(txt: str):
                            out=[]
                            for chunk in txt.split(","):
                                c=chunk.strip()
                                if not c: continue
                                a,b=c.split(":")
                                out.append((float(a)+dE_cal, float(b)+dE_cal))
                            return out
                        regions = parse_regions(left_region)
                        I_delta = np.zeros_like(I_mat)
                        for j in range(0, I_mat.shape[1], 2):
                            Ef = E_mat[:, j]; If = I_mat[:, j]
                            Er = E_mat[:, j+1]; Ir = I_mat[:, j+1]
                            bf = poly_baseline(Ef, If, regions, poly_order)
                            br = poly_baseline(Er, Ir, regions, poly_order)
                            I_delta[:, j]   = If - bf
                            I_delta[:, j+1] = Ir - br

                if use_baseline and (I_delta is not None):
                    fig2, ax2 = plt.subplots(figsize=(9,6), dpi=150)
                    for i in active_cycles():
                        if show_forward:
                            ax2.plot(E_mat[:, 2*(i-1)], I_delta[:, 2*(i-1)], lw=fw_lw, color=cycle_color(i,"fwd"))
                        if show_reverse:
                            ax2.plot(E_mat[:, 2*(i-1)+1], I_delta[:, 2*(i-1)+1], lw=rv_lw, color=cycle_color(i,"rev"),
                                     linestyle={"dashed":"--","dotted":":","solid":"-"}.get(rv_style,"--"))
                    plot_range(ax2); ax2.set_xlabel("E (mV)"); ax2.set_ylabel(yname_loc); ax2.set_title("Δ vs baseline")
                    if grid_on: ax2.grid(True, alpha=0.25)
                    if zero_line: ax2.axhline(0, color="0.5", lw=0.7, alpha=0.6, zorder=0)
                    if show_legend_all: place_legend(ax2)
                    st.pyplot(fig2, clear_figure=True)
                else:
                    st.info("Enable Δ subtraction to view this plot.")

            with tabs[1]:
                d = np.zeros_like(I_mat)
                for j in range(I_mat.shape[1]): d[:, j] = np.gradient(I_mat[:, j], E_mat[:, j])
                fig3, ax3 = plt.subplots(figsize=(9,6), dpi=150)
                for i in active_cycles():
                    if show_forward:
                        ax3.plot(E_mat[:, 2*(i-1)], d[:, 2*(i-1)], lw=fw_lw, color=cycle_color(i,"fwd"))
                    if show_reverse:
                        ax3.plot(E_mat[:, 2*(i-1)+1], d[:, 2*(i-1)+1], lw=rv_lw, color=cycle_color(i,"rev"),
                                 linestyle={"dashed":"--","dotted":":","solid":"-"}.get(rv_style,"--"))
                plot_range(ax3); ax3.set_xlabel("E (mV)"); ax3.set_ylabel("dI/dE"); ax3.set_title("Derivative dI/dE (from raw)")
                if grid_on: ax3.grid(True, alpha=0.25)
                if zero_line: ax3.axhline(0, color="0.5", lw=0.7, alpha=0.6, zorder=0)
                if show_legend_all: place_legend(ax3)
                st.pyplot(fig3, clear_figure=True)

            with tabs[2]:
                fig4, ax4 = plt.subplots(figsize=(9,6), dpi=150)
                for i in active_cycles():
                    ax4.plot(E_mat[:, 2*(i-1)], I_mat[:, 2*(i-1)], lw=fw_lw, color=cycle_color(i,"fwd"))
                plot_range(ax4); ax4.set_xlabel("E (mV)"); ax4.set_ylabel(yname_loc); ax4.set_title("Forward-only (raw)")
                if grid_on: ax4.grid(True, alpha=0.25)
                if zero_line: ax4.axhline(0, color="0.5", lw=0.7, alpha=0.6, zorder=0)
                if show_legend_all: place_legend(ax4)
                st.pyplot(fig4, clear_figure=True)

            with tabs[3]:
                fig5, ax5 = plt.subplots(figsize=(9,6), dpi=150)
                for i in active_cycles():
                    ax5.plot(E_mat[:, 2*(i-1)+1], I_mat[:, 2*(i-1)+1], lw=rv_lw, color=cycle_color(i,"rev"),
                             linestyle={"dashed":"--","dotted":":","solid":"-"}.get(rv_style,"--"))
                plot_range(ax5); ax5.set_xlabel("E (mV)"); ax5.set_ylabel(yname_loc); ax5.set_title("Reverse-only (raw)")
                if grid_on: ax5.grid(True, alpha=0.25)
                if zero_line: ax5.axhline(0, color="0.5", lw=0.7, alpha=0.6, zorder=0)
                if show_legend_all: place_legend(ax5)
                st.pyplot(fig5, clear_figure=True)

            with tabs[4]:
                n = len(active_cycles())
                ncols = 4 if n >= 8 else 3 if n >= 6 else 2
                nrows = int(np.ceil(n / ncols)) if n > 0 else 1
                fig6, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3.2*ncols, 2.6*nrows), dpi=150, squeeze=False)
                cyc_list = active_cycles()
                for ax, cyc in zip(axes.ravel(), cyc_list):
                    if show_forward:
                        ax.plot(E_mat[:, 2*(cyc-1)], I_mat[:, 2*(cyc-1)], lw=fw_lw, color=cycle_color(cyc,"fwd"))
                    if show_reverse:
                        ax.plot(E_mat[:, 2*(cyc-1)+1], I_mat[:, 2*(cyc-1)+1], lw=rv_lw, color=cycle_color(cyc,"rev"),
                                linestyle={"dashed":"--","dotted":":","solid":"-"}.get(rv_style,"--"))
                    ax.grid(grid_on, alpha=0.2)
                    if zero_line: ax.axhline(0, color="0.7", lw=0.6)
                    ax.set_title(legend_text(cyc,"fwd").replace(" fwd","").replace(" rev",""), fontsize=9); ax.tick_params(labelsize=8)
                    plot_range(ax)
                for ax in axes.ravel()[len(cyc_list):]: ax.axis('off')
                fig6.tight_layout(); st.pyplot(fig6, clear_figure=True)

            # ---- AUTO-ANALYSIS (table UI)
            st.markdown("---")
            st.subheader("Auto-analysis & report")

            if st.button("Analyze current view"):
                rows: List[Dict[str, Any]] = []

                def analyze_one(alias: str, Em: np.ndarray, Im: np.ndarray, nc: int):
                    vis = [c for c in active_cycles() if 1 <= c <= nc]
                    for i in vis:
                        Ef = Em[:, 2*(i-1)]; If = Im[:, 2*(i-1)]
                        Er = Em[:, 2*(i-1)+1]; Ir = Im[:, 2*(i-1)+1]
                        anod_f, _ = _peak_metrics(Ef, If)
                        _, cath_r = _peak_metrics(Er, Ir)
                        Ep_a = anod_f[0] if anod_f else float("nan")
                        Ep_c = cath_r[0] if cath_r else float("nan")
                        dEp  = (Ep_a - Ep_c) if (Ep_a == Ep_a and Ep_c == Ep_c) else float("nan")
                        area_uC = _loop_area_uC(Ef, If, Er, Ir, scan_rate)
                        drift = _midzone_drift(If)
                        call  = _reversibility_call(abs(dEp) if dEp == dEp else float("nan"))
                        rows.append({
                            "Dataset": alias,
                            "Cycle": i,
                            "Ep(a) [mV]": round(Ep_a, 1) if Ep_a == Ep_a else None,
                            "Ep(c) [mV]": round(Ep_c, 1) if Ep_c == Ep_c else None,
                            "ΔEp [mV]": round(abs(dEp), 1) if dEp == dEp else None,
                            "Loop area [µC]": round(area_uC, 2) if area_uC == area_uC else None,
                            "Drift(mid) [µA]": round(drift, 2),
                            "Reversibility": call,
                        })

                if cmp_enable and cmp_ids and len(cmp_ids) >= 2:
                    alias_by_id = {e["id"]: e["alias"] for e in st.session_state.cv_files}
                    data_by_id  = {e["id"]: e["data"] for e in st.session_state.cv_files}
                    for dsid in cmp_ids:
                        Em, Im, _, _, nc = compute_mats_for(data_by_id[dsid])
                        analyze_one(alias_by_id[dsid], Em, Im, nc)
                else:
                    analyze_one(active_alias, E_mat, I_mat, n_cycles)

                df_report = pd.DataFrame(rows)
                st.dataframe(df_report, use_container_width=True, hide_index=True)

                # exports
                csv_bytes = df_report.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download analysis (CSV)", data=csv_bytes,
                                   file_name="cv_auto_analysis.csv", mime="text/csv")

                md = _make_report_md("CV Auto-analysis", [
                    f"{r['Dataset']} · C{r['Cycle']}: Ep(a)={r['Ep(a) [mV]']} mV, Ep(c)={r['Ep(c) [mV]']} mV, "
                    f"ΔEp={r['ΔEp [mV]']} mV, loop={r['Loop area [µC]']} µC, drift={r['Drift(mid) [µA]']}, {r['Reversibility']}"
                    for _, r in df_report.iterrows()
                ])
                st.download_button("⬇️ Download summary (Markdown)", data=md.encode("utf-8"),
                                   file_name="cv_auto_report.md", mime="text/markdown")

            # ---- export image
            st.markdown("---"); st.subheader("Exports")
            buf = io.BytesIO()
            fig_export = plot_raw_clean(); fig_export.savefig(buf, format="png", dpi=300, bbox_inches="tight"); plt.close(fig_export)
            st.download_button("⬇️ Download raw_clean.png", data=buf.getvalue(), file_name="cv_raw_clean.png", mime="image/png")

            with st.expander("Diagnostics"):
                st.write(f"App v{APP_VERSION} | Active: {active_alias} | Data shape: {_active['data'].shape} | Cycles: {n_cycles}")
                st.write(f"Reverse mapping: {reverse_mode} | Compare enabled: {cmp_enable} ({len(cmp_ids) if cmp_enable else 0} datasets)")
                if not (math.isnan(ref_E_true) or math.isnan(ref_E_obs)):
                    st.write(f"Calibration ΔE = {ref_E_true - ref_E_obs:+.1f} mV")
                else:
                    st.write("Calibration ΔE = +0.0 mV")
