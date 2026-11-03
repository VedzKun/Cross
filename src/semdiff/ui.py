"""
Streamlit UI for semdiff.

Run with:
    streamlit run src/semdiff/ui.py

Features:
  - Upload two documents (PDF/DOCX/TXT) or paste text
  - Select classification mode
  - View side-by-side diff with color-coded severity
  - Download HTML report and JSON export
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

try:
    import streamlit as st
except ImportError:
    print("Streamlit not installed. Run: pip install streamlit")
    sys.exit(1)

# -------------------------------------------------------------------------
# Page config
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="SemDiff — Regulatory Document Diff",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Settings")
    input_mode = st.radio("Input mode", ["Upload files", "Paste text"], index=0)
    classification_mode = st.selectbox(
        "Classification mode",
        ["rule_only", "combined", "model_only"],
        index=0,
        help=(
            "rule_only: fast, no ML model needed\n"
            "combined: rule flags + DistilBERT (requires fine-tuned checkpoint)\n"
            "model_only: DistilBERT only"
        ),
    )
    checkpoint_path = st.text_input(
        "Classifier checkpoint (optional)",
        value="models/classifier",
        help="Path to fine-tuned DistilBERT checkpoint directory",
    )
    sim_threshold = st.slider("Similarity threshold", 0.3, 0.9, 0.50, 0.05)
    enable_cache = st.checkbox("Enable embedding cache", value=True)

st.title("📄 Cross-Lingual Semantic Diff")
st.markdown(
    "Compare regulatory documents across **English, Hindi, and Tamil** "
    "and classify each change by severity: `cosmetic` / `substantive` / `critical`."
)

# -------------------------------------------------------------------------
# Input area
# -------------------------------------------------------------------------
if input_mode == "Upload files":
    col1, col2 = st.columns(2)
    with col1:
        file_a = st.file_uploader("Version A (older)", type=["pdf", "docx", "txt"])
        label_a = st.text_input("Label A", value="Version A")
    with col2:
        file_b = st.file_uploader("Version B (newer)", type=["pdf", "docx", "txt"])
        label_b = st.text_input("Label B", value="Version B")
    text_a = text_b = None
else:
    col1, col2 = st.columns(2)
    with col1:
        label_a = st.text_input("Label A", value="Version A")
        text_a = st.text_area("Paste Version A text", height=300, placeholder="Paste or type document text here...")
    with col2:
        label_b = st.text_input("Label B", value="Version B")
        text_b = st.text_area("Paste Version B text", height=300, placeholder="Paste or type document text here...")
    file_a = file_b = None

run_btn = st.button("🔍 Compare", type="primary", use_container_width=True)

# -------------------------------------------------------------------------
# Run comparison
# -------------------------------------------------------------------------
if run_btn:
    can_run = False
    if input_mode == "Upload files" and file_a and file_b:
        can_run = True
    elif input_mode == "Paste text" and text_a and text_b:
        can_run = True

    if not can_run:
        st.warning("Please provide both inputs before comparing.")
    else:
        with st.spinner("Running semantic alignment..."):
            try:
                from src.semdiff.ingestion.readers import DocumentReader
                from src.semdiff.embedding.labse import LaBSEEmbedder
                from src.semdiff.alignment.aligner import SemanticAligner
                from src.semdiff.alignment.chain import AmendmentChain
                from src.semdiff.classifier.pipeline import ClassificationPipeline
                from src.semdiff.reporting.html_report import generate_html_report
                from src.semdiff.reporting.json_export import export_diff_to_json
                from src.semdiff.config import default_settings

                default_settings.ensure_directories()
                t0 = time.time()
                reader = DocumentReader()
                embedder = LaBSEEmbedder(
                    enable_cache=enable_cache,
                    cache_db_path=default_settings.embedding_cache_db,
                )

                # Save uploads to temp files
                if input_mode == "Upload files":
                    with tempfile.NamedTemporaryFile(
                        suffix=Path(file_a.name).suffix, delete=False
                    ) as ta:
                        ta.write(file_a.getbuffer())
                        path_a = Path(ta.name)
                    with tempfile.NamedTemporaryFile(
                        suffix=Path(file_b.name).suffix, delete=False
                    ) as tb:
                        tb.write(file_b.getbuffer())
                        path_b = Path(tb.name)
                    doc_a = reader.read(path_a, doc_id=file_a.name, version="v1")
                    doc_b = reader.read(path_b, doc_id=file_b.name, version="v2")
                    path_a.unlink(missing_ok=True)
                    path_b.unlink(missing_ok=True)
                else:
                    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w", encoding="utf-8") as ta:
                        ta.write(text_a)
                        path_a = Path(ta.name)
                    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w", encoding="utf-8") as tb:
                        tb.write(text_b)
                        path_b = Path(tb.name)
                    doc_a = reader.read(path_a, version="v1")
                    doc_b = reader.read(path_b, version="v2")
                    path_a.unlink(missing_ok=True)
                    path_b.unlink(missing_ok=True)

                aligner = SemanticAligner(sim_threshold=sim_threshold)
                chain = AmendmentChain(embedder=embedder, aligner=aligner)
                chain.add_version(doc_a)
                chain.add_version(doc_b)
                diff = chain.get_diffs()[0]
                timelines = chain.get_timelines()

                ckpt = Path(checkpoint_path) if checkpoint_path else None
                clf = ClassificationPipeline(
                    mode=classification_mode,
                    classifier_checkpoint=ckpt if (ckpt and ckpt.exists()) else None,
                )
                diff = clf.classify_diff_result(diff)

                elapsed = time.time() - t0

                # -------------------------
                # Display results
                # -------------------------
                sev = diff.stats.get("severity_counts", {})
                atype = diff.stats.get("alignment_type_counts", {})

                c1, c2, c3, c4, c5, c6 = st.columns(6)
                c1.metric("Total", len(diff.aligned_pairs))
                c2.metric("🟡 Cosmetic", sev.get("cosmetic", 0))
                c3.metric("🟠 Substantive", sev.get("substantive", 0))
                c4.metric("🔴 Critical", sev.get("critical", 0))
                c5.metric("➕ Inserts", atype.get("insert", 0))
                c6.metric("✖ Deletes", atype.get("delete", 0))
                st.caption(f"⏱ Completed in {elapsed:.1f}s")

                st.divider()

                # Severity filter
                sev_filter = st.multiselect(
                    "Filter by severity",
                    ["unchanged", "cosmetic", "substantive", "critical"],
                    default=["cosmetic", "substantive", "critical"],
                )

                # Pair display
                COLORS = {
                    "unchanged": "#f0f9ff",
                    "cosmetic": "#fefce8",
                    "substantive": "#fff7ed",
                    "critical": "#fef2f2",
                }
                BORDERS = {
                    "unchanged": "#bae6fd",
                    "cosmetic": "#fde68a",
                    "substantive": "#fdba74",
                    "critical": "#fca5a5",
                }

                shown = 0
                for pair in diff.aligned_pairs:
                    if pair.severity.value not in sev_filter:
                        continue
                    shown += 1
                    if shown > 200:
                        st.info("Showing first 200 pairs. Download HTML for full report.")
                        break

                    bg = COLORS.get(pair.severity.value, "#ffffff")
                    bdr = BORDERS.get(pair.severity.value, "#e2e8f0")
                    with st.container():
                        st.markdown(
                            f'<div style="background:{bg};border-left:4px solid {bdr};'
                            f'border-radius:8px;padding:12px;margin-bottom:8px;">',
                            unsafe_allow_html=True,
                        )
                        col_a, col_b, col_meta = st.columns([5, 5, 2])
                        with col_a:
                            if pair.text_a:
                                st.markdown(f"**{label_a}**")
                                st.markdown(pair.text_a)
                            else:
                                st.markdown("*[not present]*")
                        with col_b:
                            if pair.text_b:
                                st.markdown(f"**{label_b}**")
                                st.markdown(pair.text_b)
                            else:
                                st.markdown("*[not present]*")
                        with col_meta:
                            st.markdown(
                                f"`{pair.severity.value}`  \n"
                                f"`{pair.alignment_type.value}`  \n"
                                f"sim: `{pair.similarity_score:.3f}`  \n"
                                f"conf: `{pair.confidence:.2f}`"
                            )
                            if pair.rule_flags:
                                for f in pair.rule_flags:
                                    st.markdown(f"🚩 `{f}`")
                        st.markdown("</div>", unsafe_allow_html=True)

                # Downloads
                st.divider()
                html_report = generate_html_report(diff, label_a, label_b, timelines)
                json_export = export_diff_to_json(diff, timelines)

                dl_col1, dl_col2 = st.columns(2)
                with dl_col1:
                    st.download_button(
                        "⬇️ Download HTML Report",
                        data=html_report,
                        file_name="semdiff_report.html",
                        mime="text/html",
                        use_container_width=True,
                    )
                with dl_col2:
                    st.download_button(
                        "⬇️ Download JSON Export",
                        data=json_export,
                        file_name="semdiff_result.json",
                        mime="application/json",
                        use_container_width=True,
                    )

            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                st.code(traceback.format_exc())

