# tabs/t10_electrochem.py
from __future__ import annotations
import os, io, uuid, shutil, zipfile, glob, re
from typing import List, Optional, Any, Dict, Tuple
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# --- Safe local import of user's cv.py and chrono.py (search several common places) ---
def _import_local_module(modname: str):
    import importlib.util, sys
    candidates: List[str] = []
    candidates.append(os.path.join(os.path.abspath("."), f"{modname}.py"))
    try:
        app_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        candidates.append(os.path.join(app_dir, f"{modname}.py"))
    except Exception:
        pass
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, ".."))
    candidates.append(os.path.join(repo_root, f"{modname}.py"))
    candidates.append(os.path.join(repo_root, "bpcomp", f"{modname}.py"))

    for cand in candidates:
        if os.path.isfile(cand):
            spec = importlib.util.spec_from_file_location(modname, cand)
            if spec and spec.loader:
                m = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(m)  # type: ignore[attr-defined]
                return m
    return None

_cv = _import_local_module("cv")
_chrono = _import_local_module("chrono")

# ---------------- helpers ----------------
def _session_tmp_dir() -> str:
    sid = st.session_state.get("_echem_session_id")
    if not sid:
        sid = uuid.uuid4().hex[:8]
        st.session_state["_echem_session_id"] = sid
    root = os.path.join(os.path.abspath("."), ".echem_tmp", sid)
    os.makedirs(root, exist_ok=True)
    return root

def _to_bytes(obj: Any) -> Optional[bytes]:
    if obj is None:
        return None
    if isinstance(obj, (bytes, bytearray)):
        return bytes(obj)
    if isinstance(obj, memoryview):
        return obj.tobytes()
    tb = getattr(obj, "tobytes", None)
    if callable(tb):
        try:
            v = tb()
            if isinstance(v, (bytes, bytearray)):
                return bytes(v)
            if isinstance(v, memoryview):
                return v.tobytes()
        except Exception:
            return None
    return None

def _save_uploads(files: List[Any], subdir: str, suffix: Optional[str]=None) -> List[str]:
    base = os.path.join(_session_tmp_dir(), subdir + (f"_{suffix}" if suffix else ""))
    os.makedirs(base, exist_ok=True)
    paths: List[str] = []
    for f in files:
        if f is None:
            continue
        name = os.path.basename(getattr(f, "name", "upload.bin")).replace("..", ".")
        dst = os.path.join(base, name)

        data_bytes: Optional[bytes] = None
        for meth in ("getvalue", "read", "getbuffer"):
            fn = getattr(f, meth, None)
            if callable(fn):
                try:
                    v = fn()
                    if isinstance(v, (bytes, bytearray)):
                        data_bytes = bytes(v)
                        break
                    if isinstance(v, memoryview):
                        data_bytes = v.tobytes()
                        break
                    b = _to_bytes(v)
                    if b is not None:
                        data_bytes = b
                        break
                except Exception:
                    continue
        if data_bytes is None:
            continue
        with open(dst, "wb") as fh:
            fh.write(data_bytes)
        paths.append(dst)
    return paths

def _zip_dir(path: str) -> bytes:
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(path):
            for fn in files:
                full = os.path.join(root, fn)
                arc = os.path.relpath(full, start=path)
                z.write(full, arc)
    return bio.getvalue()

def _find_plots(start_dirs: List[str]) -> List[str]:
    out: List[str] = []
    for d in start_dirs:
        pdir = os.path.join(os.path.dirname(d) if os.path.isfile(d) else d, "plots")
        out.extend(sorted(glob.glob(os.path.join(pdir, "*.png"))))
    return sorted(set(out))

# ---------- Built-in CV fallback (no cv.py required) ----------
def _detect_channels(cols: List[str]) -> List[int]:
    ch: set[int] = set()
    patE = re.compile(r"^E(\d+)_mV$", re.IGNORECASE)
    patI = re.compile(r"^I(\d+)_uA$", re.IGNORECASE)
    for c in cols:
        m = patE.match(c) or patI.match(c)
        if m:
            try:
                ch.add(int(m.group(1)))
            except Exception:
                pass
    return sorted(ch)

def _read_cv_file(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception:
        try:
            df = pd.read_csv(path, sep=None, engine="python")
        except Exception:
            df = pd.read_csv(path, delim_whitespace=True)
    return df

def _builtin_cv(
    files: List[str],
    area: float,
    area_unit: str,
    overlay: bool,
    channels: Optional[List[int]],
    smooth_window: int,
    decimate_every: Optional[int],
    rev_eps_mV: float,
    rev_min_run: int,
    rev_smooth_window: int,
    plot_color: str,
    line_style: str,
    marker: str,
) -> Tuple[bool, List[str]]:
    if not files:
        return False, []
    out_dir = os.path.join(os.path.dirname(files[0]), "plots")
    os.makedirs(out_dir, exist_ok=True)

    # gather channels present across files
    union_channels: set[int] = set()
    dfs: Dict[str, pd.DataFrame] = {}
    for fp in files:
        df = _read_cv_file(fp)
        dfs[fp] = df
        union_channels.update(_detect_channels(df.columns.tolist()))
    all_channels = sorted(union_channels)
    if not all_channels:
        for fp, df in dfs.items():
            if {"Voltage", "Current"} <= set(map(str, df.columns)):
                all_channels = [1]
                break
    if not all_channels:
        return False, []

    chans = channels if (channels and len(channels)>0) else all_channels

    saved_imgs: List[str] = []
    # overlay: one figure per channel, multiple files; otherwise per (file,channel)
    if overlay:
        for k in chans:
            fig, ax = plt.subplots(figsize=(6, 4))
            ylab = "I (A)"  # <-- initialize to prevent 'possibly unbound'
            plotted_any = False
            for fp, df in dfs.items():
                # pick columns
                if {"Voltage", "Current"} <= set(map(str, df.columns)):
                    E = pd.to_numeric(df["Voltage"], errors="coerce")
                    I = pd.to_numeric(df["Current"], errors="coerce")
                else:
                    Ek = f"E{k}_mV"; Ik = f"I{k}_uA"
                    if Ek not in df.columns or Ik not in df.columns:
                        continue
                    E = pd.to_numeric(df[Ek], errors="coerce")/1000.0
                    I = pd.to_numeric(df[Ik], errors="coerce")/1e6
                mask = E.notna() & I.notna()
                E, I = E[mask], I[mask]
                if E.empty:
                    continue
                if smooth_window and smooth_window >= 3 and smooth_window % 2 == 1:
                    I = I.rolling(smooth_window, center=True, min_periods=1).mean()
                if decimate_every and decimate_every >= 2:
                    E = E.iloc[::decimate_every]
                    I = I.iloc[::decimate_every]
                if area > 0:
                    Aeff = area / (1.0 if area_unit=="m^2" else 1e4)  # cm^2 → m^2
                    I = I / Aeff
                    ylab = "J (A/m²)"
                ax.plot(
                    E, I,
                    linestyle=line_style,
                    marker=(None if marker.strip()=="" else marker),
                    label=os.path.basename(fp),
                    color=(None if plot_color.strip()=="" else plot_color),
                )
                plotted_any = True
            if not plotted_any:
                plt.close(fig)
                continue
            ax.set_title(f"CV channel {k} (overlay)")
            ax.set_xlabel("E (V vs. ref)")
            ax.set_ylabel(ylab)
            ax.legend(loc="best")
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            out = os.path.join(out_dir, f"cv_ch{k}_overlay.png")
            fig.savefig(out, dpi=150)
            plt.close(fig)
            saved_imgs.append(out)
    else:
        for fp, df in dfs.items():
            for k in chans:
                fig, ax = plt.subplots(figsize=(6, 4))
                # default y label
                ylab = "I (A)"
                # choose columns
                if {"Voltage", "Current"} <= set(map(str, df.columns)):
                    E = pd.to_numeric(df["Voltage"], errors="coerce")
                    I = pd.to_numeric(df["Current"], errors="coerce")
                else:
                    Ek = f"E{k}_mV"; Ik = f"I{k}_uA"
                    if Ek not in df.columns or Ik not in df.columns:
                        plt.close(fig)
                        continue
                    E = pd.to_numeric(df[Ek], errors="coerce")/1000.0
                    I = pd.to_numeric(df[Ik], errors="coerce")/1e6
                mask = E.notna() & I.notna()
                E, I = E[mask], I[mask]
                if E.empty:
                    plt.close(fig)
                    continue
                if smooth_window and smooth_window >= 3 and smooth_window % 2 == 1:
                    I = I.rolling(smooth_window, center=True, min_periods=1).mean()
                if decimate_every and decimate_every >= 2:
                    E = E.iloc[::decimate_every]
                    I = I.iloc[::decimate_every]
                if area > 0:
                    Aeff = area / (1.0 if area_unit=="m^2" else 1e4)
                    I = I / Aeff
                    ylab = "J (A/m²)"
                ax.plot(
                    E, I,
                    linestyle=line_style,
                    marker=(None if marker.strip()=="" else marker),
                    color=(None if plot_color.strip()=="" else plot_color),
                )
                ax.set_title(f"{os.path.basename(fp)} • ch{k}")
                ax.set_xlabel("E (V vs. ref)")
                ax.set_ylabel(ylab)
                ax.grid(True, alpha=0.25)
                fig.tight_layout()
                out = os.path.join(out_dir, f"{os.path.splitext(os.path.basename(fp))[0]}_ch{k}.png")
                fig.savefig(out, dpi=150)
                plt.close(fig)
                saved_imgs.append(out)

    return True, saved_imgs

# ---------------- UI ----------------
def render(container, culture_name: str):
    with container:
        st.subheader("⑩ Electrochemistry (CV / Chrono)")
        st.caption("Upload CV CSVs (wide format: E{k}_mV & I{k}_uA, or legacy Voltage/Current) or chrono *.txt. If `cv.py` is missing, a built-in CV plotter is used and PNGs are still generated.")

        # --- Global inputs shared by CV & Chrono ---
        c1, c2, c3 = st.columns(3)
        area = c1.number_input("Electrode area", value=1.00, min_value=1e-9, step=0.01, format="%.4f")
        area_unit = c2.selectbox("Area unit", ["cm^2", "m^2"], index=0)
        clear_btn = c3.button("🧹 Clear temp echem session", type="secondary")

        if clear_btn:
            tmp = _session_tmp_dir()
            try:
                shutil.rmtree(tmp, ignore_errors=True)
            except Exception:
                pass
            st.session_state.pop("_echem_session_id", None)
            st.success("Electrochemistry temp workspace cleared.")

        st.markdown("---")

        # ============ CV SECTION ============
        st.markdown("### Cyclic Voltammetry")
        cv_cols = st.columns([1,1,1,1])
        overlay = cv_cols[0].checkbox("Overlay across files", value=True)
        channels_mode = cv_cols[1].radio("Channels", ["All", "Selected"], horizontal=True)
        choose_ch = st.multiselect("Select channel numbers (if 'Selected')", options=[1,2,3,4,5,6,7,8], default=[1])

        st.markdown("**Advanced (optional)**")
        a1, a2, a3, a4 = st.columns(4)
        smooth_window = a1.number_input("Smoothing window (odd, 0=off)", value=0, min_value=0, step=1)
        decimate_every = a2.number_input("Downsample: keep every N points (blank=off → put 0)", value=0, min_value=0, step=1)
        rev_eps_mV = a3.number_input("Reversal hysteresis (mV)", value=2.0, min_value=0.0, step=0.5, format="%.2f")
        rev_min_run = a4.number_input("Min run length (points)", value=20, min_value=1, step=1)
        b1 = st.number_input("Potential smoothing (odd)", value=5, min_value=1, step=2)

        st.markdown("**Style (optional)**")
        s1, s2, s3 = st.columns(3)
        plot_color = s1.text_input("Matplotlib color (empty = cycle)", value="")
        line_style = s2.selectbox("Line", ["-", "--", "-.", ":", " "], index=0)
        marker = s3.selectbox("Marker", ["", "o", ".", "_", "*", "+", "x", "s", "^", "D"], index=1)

        cv_files = st.file_uploader("Upload CV CSV files", type=["csv", "txt"], accept_multiple_files=True, key="cv_upload")
        go_cv = st.button("Run CV")

        cv_plot_dirs: List[str] = []
        if go_cv:
            if not cv_files:
                st.error("Please upload at least one CV file.")
            else:
                saved = _save_uploads(cv_files, "cv")
                if _cv is not None:
                    file_arg = ",".join(saved)
                    ch_arg: Optional[List[int]] = None if channels_mode == "All" else ([int(x) for x in choose_ch] or [])
                    smooth_norm = int(smooth_window)
                    if smooth_norm % 2 == 0 and smooth_norm != 0:
                        smooth_norm += 1
                    decimate_norm = int(decimate_every) if int(decimate_every) >= 2 else None
                    rev_smooth = int(b1)
                    if rev_smooth % 2 == 0:
                        rev_smooth += 1
                    try:
                        ok = _cv.main(
                            file_arg,
                            float(area),
                            area_unit,
                            1 if overlay else 0,
                            channels=ch_arg,
                            smooth_window=smooth_norm,
                            decimate_every=decimate_norm,
                            rev_eps_mV=float(rev_eps_mV),
                            rev_min_run=int(rev_min_run),
                            rev_smooth_window=rev_smooth,
                            plot_color=plot_color,
                            line_style=line_style,
                            marker=marker,
                        )
                    except Exception as e:
                        ok = False
                        st.exception(e)
                    if ok:
                        cv_plot_dirs = saved
                        st.success("CV plots generated (external cv.py).")
                    else:
                        st.error("cv.py failed; try the built-in parser by removing/renaming cv.py.")
                else:
                    ch_arg = None if channels_mode == "All" else ([int(x) for x in choose_ch] or [])
                    ok, imgs = _builtin_cv(
                        saved, float(area), area_unit, bool(overlay), ch_arg,
                        int(smooth_window), (int(decimate_every) if int(decimate_every)>=2 else None),
                        float(rev_eps_mV), int(rev_min_run), int(b1),
                        plot_color, line_style, marker
                    )
                    if ok:
                        cv_plot_dirs = saved
                        st.success("CV plots generated (built-in).")
                    else:
                        st.error("Could not parse CV files. Expect columns like E1_mV, I1_uA (or Voltage, Current).")

        # Show CV images if any were produced
        if cv_plot_dirs:
            imgs = _find_plots(cv_plot_dirs)
            if imgs:
                st.markdown("**Generated CV plots**")
                for p in imgs:
                    st.image(p, caption=os.path.basename(p), use_column_width=True)
                z_root = os.path.dirname(os.path.dirname(imgs[0])) if imgs else ""
                if z_root and os.path.isdir(z_root):
                    z = _zip_dir(z_root)
                    st.download_button("⬇️ Download CV plots (zip)", data=z, file_name="cv_plots.zip", mime="application/zip")

        st.markdown("---")

        # ============ CHRONO SECTION ============
        st.markdown("### Chronoamperometry")
        st.caption("Upload a set of *.txt files (one or more). The utility reads the folder and emits I–T and J–T per channel.")
        chrono_files = st.file_uploader("Upload chrono *.txt files", type=["txt"], accept_multiple_files=True, key="chrono_upload")
        go_chr = st.button("Run Chrono", disabled=(_chrono is None), help=(None if _chrono is not None else "chrono.py not found (no built-in for chrono)"))

        chr_plot_dir = ""
        if go_chr:
            if not chrono_files:
                st.error("Please upload at least one *.txt file for chrono.")
            elif _chrono is None:
                st.error("Could not import `chrono.py`. Place it in the project root.")
            else:
                saved = _save_uploads(chrono_files, "chrono", suffix="anode")
                anode_dir = os.path.dirname(saved[0]) if saved else ""
                try:
                    ok = _chrono.main(
                        anode_dir,
                        "",
                        float(area),
                        area_unit,
                        anode_plot_color="blue",
                        anode_line_style="-",
                        anode_marker="o",
                        cathode_plot_color="red",
                        cathode_line_style="-",
                        cathode_marker="o",
                    )
                except Exception as e:
                    ok = False
                    st.exception(e)
                if ok:
                    chr_plot_dir = anode_dir
                    st.success("Chrono plots generated.")
                else:
                    st.error("Chrono plotting failed.")

        if chr_plot_dir:
            imgs = _find_plots([chr_plot_dir])
            if imgs:
                st.markdown("**Generated Chrono plots**")
                for p in imgs:
                    st.image(p, caption=os.path.basename(p), use_column_width=True)
                plot_dir = os.path.join(chr_plot_dir, "plots")
                if os.path.isdir(plot_dir):
                    z = _zip_dir(plot_dir)
                    st.download_button("⬇️ Download Chrono plots (zip)", data=z, file_name="chrono_plots.zip", mime="application/zip")
