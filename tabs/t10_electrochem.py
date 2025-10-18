# tabs/t10_electrochem.py
from __future__ import annotations
import os, io, uuid, shutil, zipfile, glob
from typing import List, Optional, Any, cast
import streamlit as st

# --- Safe local import of user's cv.py and chrono.py ---
def _import_local_module(modname: str):
    try:
        return __import__(modname)
    except Exception:
        import importlib.util
        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.abspath(os.path.join(here, ".."))
        cand = os.path.join(root, f"{modname}.py")
        if not os.path.isfile(cand):
            return None
        spec = importlib.util.spec_from_file_location(modname, cand)
        if spec and spec.loader:
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)  # type: ignore[attr-defined]
            return m
        return None

_cv = _import_local_module("cv")
_chrono = _import_local_module("chrono")

# --- helpers ---
def _session_tmp_dir() -> str:
    sid = st.session_state.get("_echem_session_id")
    if not sid:
        sid = uuid.uuid4().hex[:8]
        st.session_state["_echem_session_id"] = sid
    root = os.path.join(os.path.abspath("."), ".echem_tmp", sid)
    os.makedirs(root, exist_ok=True)
    return root

def _to_bytes(obj: Any) -> Optional[bytes]:
    """
    Convert common buffer-like objects to bytes in a type-safe way.
    We deliberately avoid a generic 'bytes(obj)' fallback to satisfy Pylance.
    """
    if obj is None:
        return None
    if isinstance(obj, (bytes, bytearray)):
        return bytes(obj)
    if isinstance(obj, memoryview):
        return obj.tobytes()

    # Objects that expose .tobytes()
    tb = getattr(obj, "tobytes", None)
    if callable(tb):
        try:
            res: Any = tb()
            if isinstance(res, (bytes, bytearray)):
                return bytes(res)
            if isinstance(res, memoryview):
                return res.tobytes()
        except Exception:
            return None

    # Nothing recognized
    return None

def _save_uploads(files: List[Any], subdir: str, suffix: Optional[str]=None) -> List[str]:
    """
    Save uploaded files to a session temp folder.
    Treat 'files' as Any with .name and bytes-returning methods.
    """
    base = os.path.join(_session_tmp_dir(), subdir + (f"_{suffix}" if suffix else ""))
    os.makedirs(base, exist_ok=True)
    paths: List[str] = []
    for f in files:
        if f is None:
            continue
        name = os.path.basename(getattr(f, "name", "upload.bin")).replace("..", ".")
        dst = os.path.join(base, name)

        # Try getvalue() (bytes), then read() (bytes), then getbuffer() (memoryview)
        data_bytes: Optional[bytes] = None

        getvalue = getattr(f, "getvalue", None)
        if callable(getvalue):
            try:
                v: Any = getvalue()
                data_bytes = _to_bytes(v) if not isinstance(v, (bytes, bytearray)) else bytes(cast(bytes, v))
            except Exception:
                data_bytes = None

        if data_bytes is None:
            read = getattr(f, "read", None)
            if callable(read):
                try:
                    v2: Any = read()
                    data_bytes = _to_bytes(v2) if not isinstance(v2, (bytes, bytearray)) else bytes(cast(bytes, v2))
                except Exception:
                    data_bytes = None

        if data_bytes is None:
            getbuffer = getattr(f, "getbuffer", None)
            if callable(getbuffer):
                try:
                    v3: Any = getbuffer()  # often returns memoryview
                    if isinstance(v3, memoryview):
                        data_bytes = v3.tobytes()
                    else:
                        data_bytes = _to_bytes(v3)
                except Exception:
                    data_bytes = None

        if data_bytes is None:
            # Skip files we couldn't read safely
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

def render(container, culture_name: str):
    with container:
        st.subheader("⑩ Electrochemistry (CV / Chrono)")
        st.caption("Upload CV CSVs (multi-channel `E{k}_mV, I{k}_uA`) or legacy V/I CSVs; optionally upload chrono *.txt sets. The app writes to a temp folder, calls your `cv.py`/`chrono.py`, and previews any generated PNGs.")

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
        marker = s3.selectbox("Marker", [" ", "o", ".", "_", "*", "+", "x", "square", "triangle", "diamond"], index=1)

        cv_files = st.file_uploader("Upload CV CSV files", type=["csv", "txt"], accept_multiple_files=True, key="cv_upload")
        go_cv = st.button("Run CV", disabled=(_cv is None), help=("cv.py not found" if _cv is None else None))

        cv_plot_dirs: List[str] = []
        if go_cv:
            if not cv_files:
                st.error("Please upload at least one CV file.")
            elif _cv is None:
                st.error("Could not import `cv.py`. Place it in the project root.")
            else:
                saved = _save_uploads(cv_files, "cv")
                file_arg = ",".join(saved)
                ch_arg: Optional[List[int]]
                if channels_mode == "All":
                    ch_arg = None
                else:
                    ch_arg = [int(x) for x in choose_ch] or []

                # Normalize optional params
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
                    st.success("CV plots generated.")
                else:
                    st.error("CV plotting failed (see error logs if generated).")

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
        go_chr = st.button("Run Chrono", disabled=(_chrono is None), help=("chrono.py not found" if _chrono is None else None))

        chr_plot_dir = ""
        if go_chr:
            if not chrono_files:
                st.error("Please upload at least one *.txt file for chrono.")
            elif _chrono is None:
                st.error("Could not import `chrono.py`. Place it in the project root.")
            else:
                saved = _save_uploads(chrono_files, "chrono", suffix="anode")
                # Use the saved folder as the anode path (chrono expects a directory)
                anode_dir = os.path.dirname(saved[0]) if saved else ""
                try:
                    ok = _chrono.main(
                        anode_dir,
                        "",                  # cathode dir not used in your version
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
