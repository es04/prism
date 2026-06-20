import streamlit as st
import pandas as pd
import numpy as np
import io

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))


REQUIRED_SCHEMAS = {
    "studentInfo": {
        "required": ["id_student","final_result"],
        "optional": ["code_module","code_presentation","gender","region",
                     "highest_education","imd_band","age_band",
                     "num_of_prev_attempts","studied_credits","disability"],
        "description": "Core student demographics and final academic outcome.",
    },
    "studentAssessment": {
        "required": ["id_assessment","id_student","score"],
        "optional": ["date_submitted","is_banked"],
        "description": "Assessment submission records with scores.",
    },
    "studentVle": {
        "required": ["id_student","id_site","date","sum_click"],
        "optional": ["code_module","code_presentation"],
        "description": "Virtual Learning Environment interaction logs.",
    },
    "studentRegistration": {
        "required": ["id_student","code_module","code_presentation","date_registration"],
        "optional": ["date_unregistration"],
        "description": "Course registration and withdrawal dates.",
    },
    "assessments": {
        "required": ["id_assessment","assessment_type"],
        "optional": ["code_module","code_presentation","date","weight"],
        "description": "Assessment metadata (type, weight, schedule).",
    },
    "courses": {
        "required": ["code_module","code_presentation"],
        "optional": ["module_presentation_length"],
        "description": "Module and presentation metadata.",
    },
    "vle": {
        "required": ["id_site","activity_type"],
        "optional": ["code_module","code_presentation","week_from","week_to"],
        "description": "VLE resource/activity type definitions.",
    },
}

STATUS_ICONS = {True: "✅", False: "❌", None: "⚠️"}


def _validate_df(name: str, df: pd.DataFrame) -> dict:
    schema   = REQUIRED_SCHEMAS[name]
    cols     = set(df.columns.tolist())
    required = set(schema["required"])
    missing  = required - cols
    extra    = cols - required - set(schema["optional"])
    return {
        "ok":      len(missing) == 0,
        "missing": list(missing),
        "extra":   list(extra),
        "rows":    len(df),
        "cols":    len(df.columns),
    }


def render():
    theme.inject_theme_css()
    st.markdown("""
    <div class="page-header">
        <h2>📤 Upload & Configure Data</h2>
        <p>Upload OULAD CSV files, inspect column mappings, and validate schemas before running the pipeline.</p>
    </div>
    """, unsafe_allow_html=True)

    if "uploaded_dfs" not in st.session_state:
        st.session_state["uploaded_dfs"] = {}

    # Mode selector 
    mode = st.radio(
        "Data Source",
        ["📁  Use built-in OULAD dataset (data/ folder)",
         "📤  Upload my own CSV files"],
        horizontal=True,
    )

    if "Use built-in" in mode:
        st.markdown("""
        <div class="success-box">
            ✅ Using the bundled OULAD dataset from the <code>data/</code> directory.
            All 7 CSV files are pre-loaded. Click <b>Run Pipeline</b> to train models.
        </div>
        """, unsafe_allow_html=True)
        st.session_state["uploaded_dfs"] = {}  # clear uploads, fall back to data/

        if st.button("🚀 Run Pipeline with Built-in Data", type="primary", use_container_width=True):
            st.session_state["prism_ready"] = False
            from pages.state_manager import get_or_init_state
import theme
            if get_or_init_state(force_retrain=False):
                st.success("✅ Pipeline complete! Navigate to any page to explore results.")
                st.rerun()
        return

    # Upload mode 
    st.markdown("""
    <div class="info-box">
        Upload all 7 OULAD CSV files. Each file is validated against its expected schema.
        Only <b>studentInfo</b> is mandatory — all others enhance prediction quality.
    </div>
    """, unsafe_allow_html=True)

    upload_cols = st.columns(2)
    file_names  = list(REQUIRED_SCHEMAS.keys())
    uploaded    = {}

    for i, fname in enumerate(file_names):
        col = upload_cols[i % 2]
        schema = REQUIRED_SCHEMAS[fname]
        with col:
            with st.expander(
                f"{'🔴' if fname == 'studentInfo' else '🔵'} **{fname}.csv**"
                + (" *(required)*" if fname == 'studentInfo' else ""),
                expanded=(fname == "studentInfo")
            ):
                st.caption(schema["description"])
                st.markdown(
                    f"**Required columns:** `{'`, `'.join(schema['required'])}`  \n"
                    f"**Optional columns:** `{'`, `'.join(schema['optional'])}`"
                )

                f = st.file_uploader(
                    f"Upload {fname}.csv",
                    type=["csv"],
                    key=f"upload_{fname}",
                    label_visibility="collapsed",
                )

                if f is not None:
                    try:
                        df = pd.read_csv(f)
                        val = _validate_df(fname, df)
                        uploaded[fname] = df

                        # Show detected columns
                        st.markdown(f"**Detected {len(df.columns)} columns, {len(df):,} rows**")

                        # Colour-coded column chips
                        chip_html = ""
                        req = set(schema["required"])
                        opt = set(schema["optional"])
                        for c in df.columns:
                            if c in req:
                                color = "#22c55e"
                            elif c in opt:
                                color = "#3b82f6"
                            else:
                                color = "#94a3b8"
                            chip_html += (
                                f'<span style="background:{color}22;color:{color};'
                                f'border:1px solid {color}44;border-radius:999px;'
                                f'padding:1px 8px;font-size:.75rem;margin:2px;'
                                f'display:inline-block;">{c}</span>'
                            )
                        st.markdown(chip_html, unsafe_allow_html=True)
                        st.caption("🟢 required  🔵 optional  ⚫ extra")

                        if val["ok"]:
                            st.success(f"✅ Schema valid")
                        else:
                            st.error(f"❌ Missing: {', '.join(val['missing'])}")

                        # Preview
                        with st.expander("Preview (5 rows)"):
                            st.dataframe(df.head(5), use_container_width=True)

                    except Exception as e:
                        st.error(f"Error reading file: {e}")
                else:
                    # Check if already in session
                    if fname in st.session_state.get("uploaded_dfs", {}):
                        df = st.session_state["uploaded_dfs"][fname]
                        uploaded[fname] = df
                        st.info(f"↩️ Using previously uploaded ({len(df):,} rows)")

    # Update session state 
    if uploaded:
        st.session_state["uploaded_dfs"].update(uploaded)

    # Supabase cloud sync 
    st.markdown("<hr class='section'>", unsafe_allow_html=True)
    st.subheader("☁️ Supabase Cloud Storage")

    from utils.supabase_client import is_connected
    from utils.db_sync import log_upload, fetch_upload_history, clear_all_synced_data

    if not is_connected():
        st.markdown("""
        <div class="warn-box">
            ⚠️ Supabase isn't configured yet. Add your project URL and anon key to
            <code>.streamlit/secrets.toml</code> (see <code>.streamlit/secrets.toml.example</code>)
            to enable persistent cloud storage of uploaded CSVs, student records, and predictions.
        </div>
        """, unsafe_allow_html=True)
    else:
        current_uploads = st.session_state.get("uploaded_dfs", {})
        if current_uploads:
            st.caption("Save the CSV files you've uploaded above to Supabase Storage for persistence across sessions.")
            if st.button("☁️ Save Uploaded CSVs to Supabase", use_container_width=True):
                with st.spinner("Uploading files to Supabase…"):
                    outcomes = []
                    for fname, df in current_uploads.items():
                        val = _validate_df(fname, df)
                        csv_bytes = df.to_csv(index=False).encode("utf-8")
                        ok, msg = log_upload(
                            dataset_name=fname,
                            file_bytes=csv_bytes,
                            file_name=f"{fname}.csv",
                            row_count=val["rows"],
                            col_count=val["cols"],
                            schema_valid=val["ok"],
                        )
                        outcomes.append((fname, ok, msg))
                for fname, ok, msg in outcomes:
                    if ok:
                        st.success(f"✅ {fname}.csv → Supabase")
                    else:
                        st.error(f"❌ {fname}.csv: {msg}")
        else:
            st.caption("Upload CSV files above, then come back here to save them to Supabase.")

        with st.expander("📜 Upload History (from Supabase)"):
            hist_df = fetch_upload_history()
            if hist_df.empty:
                st.caption("No uploads recorded in Supabase yet.")
            else:
                st.dataframe(hist_df, use_container_width=True, hide_index=True)

        with st.expander("🗑️ Danger zone — clear synced cloud data"):
            st.caption(
                "Deletes every row from the `students` and `predictions` tables in Supabase "
                "(does not touch uploaded CSVs in Storage). Useful housekeeping if you're on "
                "the free tier and re-run the pipeline often."
            )
            if st.button("Clear synced students & predictions", type="secondary"):
                ok, msg = clear_all_synced_data()
                if ok:
                    st.success(f"✅ {msg}")
                else:
                    st.error(f"❌ {msg}")

    # Validation summary and column mapping viewer
    st.markdown("<hr class='section'>", unsafe_allow_html=True)
    st.subheader("📋 Upload Status Summary")

    import os
    DATA_DIR = os.path.join(pathlib.Path(__file__).parent.parent, "data")

    rows = []
    all_ok = True
    for fname, schema in REQUIRED_SCHEMAS.items():
        in_upload  = fname in st.session_state.get("uploaded_dfs", {})
        in_default = os.path.exists(os.path.join(DATA_DIR, f"{fname}.csv"))
        source = "Uploaded" if in_upload else ("Built-in" if in_default else "Missing")

        if in_upload:
            df  = st.session_state["uploaded_dfs"][fname]
            val = _validate_df(fname, df)
            ok  = val["ok"]
            info = f"{len(df):,} rows · {len(df.columns)} cols"
        elif in_default:
            df   = pd.read_csv(os.path.join(DATA_DIR, f"{fname}.csv"))
            val  = _validate_df(fname, df)
            ok   = val["ok"]
            info = f"{len(df):,} rows · {len(df.columns)} cols (built-in)"
        else:
            ok   = (fname != "studentInfo")   # only studentInfo is truly required
            info = "Not available"

        if fname == "studentInfo" and not ok:
            all_ok = False

        rows.append({
            "Dataset":   fname,
            "Source":    source,
            "Status":    "✅ Valid" if ok else ("❌ Invalid" if source != "Missing" else "⚠️ Missing"),
            "Details":   info,
            "Required":  "Yes" if fname == "studentInfo" else "No",
        })

    status_df = pd.DataFrame(rows)
    st.dataframe(status_df, use_container_width=True, hide_index=True)

    # Column mapping viewer     
    st.markdown("<hr class='section'>", unsafe_allow_html=True)
    st.subheader("🗺️ Interactive Column Mapping Inspector")

    sel_ds = st.selectbox("Select dataset to inspect", list(REQUIRED_SCHEMAS.keys()))
    schema = REQUIRED_SCHEMAS[sel_ds]

    src_df = None
    if sel_ds in st.session_state.get("uploaded_dfs", {}):
        src_df = st.session_state["uploaded_dfs"][sel_ds]
    elif os.path.exists(os.path.join(DATA_DIR, f"{sel_ds}.csv")):
        src_df = pd.read_csv(os.path.join(DATA_DIR, f"{sel_ds}.csv"))

    if src_df is not None:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Column Details**")
            col_info = []
            req = set(schema["required"])
            opt = set(schema["optional"])
            for col in src_df.columns:
                dtype    = str(src_df[col].dtype)
                null_pct = f"{src_df[col].isnull().mean()*100:.1f}%"
                role     = "Required" if col in req else ("Optional" if col in opt else "Extra")
                uniq     = src_df[col].nunique()
                col_info.append({"Column": col, "Role": role, "DType": dtype,
                                 "Nulls": null_pct, "Unique Values": uniq})
            st.dataframe(pd.DataFrame(col_info), use_container_width=True, hide_index=True)

        with c2:
            st.markdown("**Sample Data (10 rows)**")
            st.dataframe(src_df.sample(min(10, len(src_df)), random_state=1),
                         use_container_width=True)

        # Null heatmap
        if src_df.isnull().any().any():
            st.markdown("**Missing Value Heatmap**")
            import plotly.express as px
            miss = src_df.isnull().astype(int)
            fig  = px.imshow(miss.head(200).T, color_continuous_scale=["#dbeafe","#ef4444"],
                             labels={"color":"Missing"},
                             aspect="auto", height=250)
            fig.update_layout(margin=dict(t=5,b=5,l=5,r=5), coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        # Value distribution for each column
        st.markdown("**Column Value Distributions**")
        sel_col = st.selectbox("Select column", src_df.columns.tolist())
        import plotly.express as px
        col_data = src_df[sel_col].dropna()
        if col_data.dtype == object or col_data.nunique() < 20:
            vc = col_data.value_counts().reset_index()
            vc.columns = [sel_col, "Count"]
            fig2 = px.bar(vc, x=sel_col, y="Count",
                          color_discrete_sequence=["#2563eb"], height=280)
        else:
            fig2 = px.histogram(col_data, nbins=30, height=280,
                                color_discrete_sequence=["#2563eb"])
        fig2.update_layout(margin=dict(t=5,b=20,l=5,r=5))
        st.plotly_chart(fig2, use_container_width=True)

    # Run pipeline button 
    st.markdown("<hr class='section'>", unsafe_allow_html=True)

    can_run = "studentInfo" in st.session_state.get("uploaded_dfs", {}) or \
              os.path.exists(os.path.join(DATA_DIR, "studentInfo.csv"))

    if can_run:
        if st.button("🚀 Run Full Pipeline with Uploaded Data", type="primary",
                     use_container_width=True):
            st.session_state["prism_ready"] = False
            from pages.state_manager import get_or_init_state
import theme
            if get_or_init_state(force_retrain=True):
                st.success("✅ Pipeline complete! Navigate to any page to explore results.")
                st.rerun()
    else:
        st.error("❌ studentInfo.csv is required to run the pipeline.")