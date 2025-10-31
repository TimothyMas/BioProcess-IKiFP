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

    /* Cards & helpers */
    .card {border:1px solid #e5e7eb; border-radius:12px; padding:12px; background:#fafafa;}
    .muted {color:#6b7280; font-size:0.92rem;}
    .tight {margin-top:-8px;}
    .helpbox {background:#f8fafc;border:1px solid #e5e7eb;border-radius:10px;padding:12px;font-size:0.92rem;}

    /* Sidebar spacing */
    section[data-testid="stSidebar"] .block-container {padding-top: 0.6rem;}
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3 {margin-bottom: 0.3rem;}

    /* Buttons & inputs */
    .stDownloadButton > button, .stButton > button {border-radius:10px; padding: 0.35rem 0.6rem;}
    .icon-btn > button {padding:0.25rem 0.45rem; border-radius:10px;}
    .sx-compact .stNumberInput input, 
    .sx-compact .stTextInput input {height: 2.1rem;}
    .sx-compact .stTextArea textarea {min-height: 4.5rem;}

    /* DataFrames */
    [data-testid="stDataFrame"] div[role="columnheader"] { background:#f7f7f8; }

    /* Popover body tighter */
    div[data-testid="stPopoverBody"] .block-container {padding: 0.5rem 0.6rem;}
    </style>
    """, unsafe_allow_html=True)

