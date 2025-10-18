# tabs/t09_settings.py
from __future__ import annotations
import io, json
import pandas as pd
import streamlit as st
from bpcomp.state import get as get_ns

def render(container, culture: str, clear_keys_fn, env, organism_name: str):
    with container:
        st.subheader("Settings & Session")
        if st.button("♻️ Reset this culture", type="secondary", help="Clears data only for the active culture"):
            clear_keys_fn(culture)
            st.rerun()

        sess = dict(
            setup=dict(
                culture=culture,
                organism=get_ns(culture, "organism", {}),
                env=get_ns(culture, "env", {}),
                constants=get_ns(culture, "constants", {}),
            ),
            plan = get_ns(culture, "plan", {}),
            growth_df = get_ns(culture, "growth_df", pd.DataFrame()).to_dict(orient="list"),
            assay_calc = get_ns(culture, "assay_calc", pd.DataFrame()).to_dict(orient="list"),
            bradford_fit = get_ns(culture, "bradford_fit", None),
            protein_table = get_ns(culture, "protein_table", pd.DataFrame()).to_dict(orient="list"),
            purif_full = get_ns(culture, "purif_full", pd.DataFrame()).to_dict(orient="list"),
            kpis = get_ns(culture, "kpis", {}),
            sim_df = get_ns(culture, "sim_df", pd.DataFrame()).to_dict(orient="list"),
        )
        js = json.dumps(sess, indent=2, default=str).encode("utf-8")
        st.download_button("💾 Download session JSON", data=js, file_name=f"bioprocess_session_{culture.replace(' ','_')}.json", mime="application/json")

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
        st.download_button("⬇️ Download input templates (Excel)", data=bio.getvalue(), file_name="bioprocess_input_templates.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
