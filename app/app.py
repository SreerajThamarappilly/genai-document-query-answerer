# app/app.py
"""
Streamlit application.

Allows users to upload a PDF and ask questions.  
It uses the global QueryEngine stored in session state and can show an embedding cluster plot.
"""

from __future__ import annotations
import os
import tempfile
import streamlit as st

import sys
from pathlib import Path
# Add the src folder to sys.path if it's not already there
SRC_PATH = Path(__file__).resolve().parents[1] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

from query_engine import QueryEngine
from utils.visualize import plot_embeddings

# Initialize engine once per session
if 'engine' not in st.session_state:
    st.session_state.engine = QueryEngine()
engine: QueryEngine = st.session_state.engine

def main():
    st.title("📄 GenAI Document QA")
    st.write("Upload a PDF and ask questions.  The system uses a RAG pipeline to answer from the document.")

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        try:
            engine.load_document(tmp_path)
            st.success(f"Loaded and indexed '{uploaded_file.name}' successfully.")
        except Exception as exc:
            st.error(f"Failed to load document: {exc}")

    query = st.text_input("Ask a question about the document:")
    if st.button("Answer"):
        with st.spinner("Retrieving answer…"):
            answer = engine.answer_query(query)
            st.markdown(f"**Answer:** {answer}")

    if st.checkbox("Show embedding clusters (PCA)"):
        if not engine.document_loaded or engine.vector_store.index is None:
            st.warning("No document loaded or no embeddings available.")
        else:
            # reconstruct embeddings from FAISS index
            index = engine.vector_store.index
            n_vecs = index.ntotal  # type: ignore[annotation-unchecked]
            embeddings = index.reconstruct_n(0, n_vecs)  # type: ignore[annotation-unchecked]
            labels = [f"p{sec.page}_{sec.section_type}" for sec in engine.vector_store.sections]
            fig = plot_embeddings(embeddings, labels=labels)
            st.pyplot(fig)

if __name__ == "__main__":
    main()
