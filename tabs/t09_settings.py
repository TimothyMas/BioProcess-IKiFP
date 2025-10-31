# tabs/t09_settings.py
from __future__ import annotations
import io, json
import pandas as pd
import streamlit as st
from bpcomp.custom import export_catalog_json, import_catalog_json, schema_json

def render(container, clear_keys_fn, env, organism_name: str):
    with container:
        st.subheader("Settings & Session")
        if st.button("♻️ Reset session (clear cached state)", type="secondary",
                     help="Clears Streamlit session_state keys and re-runs the app."):
            clear_keys_fn()
            st.rerun()

        # Autosave restore
        st.markdown("#### Custom library autosave")
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "⬇️ Export custom library (JSON)",
                data=export_catalog_json(),
                file_name="custom_cultures.json"
            )
        with col2:
            up = st.file_uploader("Import custom library JSON", type=["json"])
            if up is not None:
                import_catalog_json(up.getvalue())
                st.success("Library imported.")

        st.markdown("#### JSON Schema")
        st.download_button(
            "Download JSON Schema",
            data=schema_json().encode("utf-8"),
            file_name="custom_cultures.schema.json",
            mime="application/json"
        )

        # Session bundle (unchanged + help)
        sess = dict(
            setup=dict(organism=organism_name, environment=env),
            plan = st.session_state.get("plan", {}),
            growth_df = st.session_state.get("growth_df", pd.DataFrame()).to_dict(orient="list"),
            assay_calc = st.session_state.get("assay_calc", pd.DataFrame()).to_dict(orient="list"),
            bradford_fit = st.session_state.get("bradford_fit", None),
            protein_table = st.session_state.get("protein_table", pd.DataFrame()).to_dict(orient="list"),
            purif_full = st.session_state.get("purif_full", pd.DataFrame()).to_dict(orient="list"),
            kpis = st.session_state.get("kpis", {}),
            sim_df = st.session_state.get("sim_df", pd.DataFrame()).to_dict(orient="list"),
        )
        js = json.dumps(sess, indent=2, default=str).encode("utf-8")
        st.download_button("💾 Download session JSON", data=js,
                           file_name="bioprocess_session.json", mime="application/json",
                           help="Snapshot everything important for this run.")
        
        # Templates (unchanged)
        st.markdown("#### Input templates")
        growth_template = pd.DataFrame({"time_h":[0.0,1.0,2.0], "od600":[0.05,0.08,0.14], "replicate":["A","A","A"]})
        assay_template = pd.DataFrame({"Sample":["Peak_1","Peak_2","Peak_3","Endpoint","Supernatant"],
                                       "Background_mABS_min":[0.0]*5, "ValueAfter_mABS_min":[0.0]*5, "PreDilution":[5,5,5,1,15]})
        std_template = pd.DataFrame({"Conc_mg_mL":[1.0,0.75,0.5,0.25,0.1], "A595":[1.012,0.894,0.723,0.545,0.400], "Use":[True]*5})
        unk_template = pd.DataFrame({"Sample":["Peak_pool","Endpoint","Supernatant"], "A595":[0.0,0.0,0.0], "Dilution":[5,1,15]})
        bio = io.BytesIO()
        with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
            growth_template.to_excel(writer, sheet_name="TEMPLATE_Growth_OD600", index=False)
            assay_template.to_excel(writer, sheet_name="TEMPLATE_Assay", index=False)
            std_template.to_excel(writer, sheet_name="TEMPLATE_Bradford", index=False)
            unk_template.to_excel(writer, sheet_name="TEMPLATE_Unknowns", index=False)
        st.download_button("⬇️ Download input templates (Excel)", data=bio.getvalue(),
                           file_name="bioprocess_input_templates.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
