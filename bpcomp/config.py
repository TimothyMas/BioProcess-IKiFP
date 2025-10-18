# bpcomp/config.py
import streamlit as st

def set_page():
    st.set_page_config(
        page_title="🧫 Bioprocess Companion - Culture → Enzyme → Analytics",
        page_icon="🧫", layout="wide"
    )

def inject_css():
    st.markdown(r"""
    <style>
    :root { color-scheme: light only; }
    html, body, .stApp { background: #ffffff; }
    .metric-card {border:1px solid #e5e7eb; border-radius:14px; padding:14px; background:#fafafa;}
    .section {margin-top:0.75rem;}
    .muted {color:#6b7280; font-size:0.92rem;}
    .tight {margin-top:-8px;}
    .stDownloadButton > button, .stButton > button {border-radius:10px;}
    [data-testid="stDataFrame"] div[role="columnheader"] { background:#f7f7f8; }
    hr { border: 0; height: 1px; background: #eee; }
    code { background: #f3f4f6; padding: 2px 6px; border-radius: 6px; }
    .helpbox {background:#f8fafc;border:1px solid #e5e7eb;border-radius:10px;padding:12px;font-size:0.92rem;}
    </style>
    """, unsafe_allow_html=True)
