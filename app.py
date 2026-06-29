import streamlit as st
import os
import logging
from dotenv import load_dotenv

load_dotenv()

from config import UPLOAD_DIR, LOG_DIR, LOG_FILE
from ingestion.loader import save_uploaded_file, load_documents, chunk_documents
from retrieval.vector_store import build_vector_index, load_vector_index
from pipeline.rag_pipeline import get_query_engine, query_knowledge_base

# ── Logging Setup ──────────────────────────────────────────────────────
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),          # also print to terminal
    ],
)
logger = logging.getLogger(__name__)
logger.info("=== Enterprise RAG app starting ===")

# ── Streamlit Config ───────────────────────────────────────────────────
st.set_page_config(page_title="Enterprise RAG - Knowledge Base", layout="wide")

st.title("Enterprise RAG - Knowledge Base")

if "index" not in st.session_state:
    st.session_state.index = load_vector_index()

if "query_engine" not in st.session_state:
    st.session_state.query_engine = None

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = set()

with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF and CSV files", 
        type=["pdf", "csv"], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        # Only re-index if new files are uploaded (avoid re-processing on every rerun)
        new_file_names = {f.name for f in uploaded_files}
        if new_file_names != st.session_state.indexed_files:
            with st.spinner("Processing files..."):
                file_paths = []
                for file in uploaded_files:
                    file_path = save_uploaded_file(file)
                    file_paths.append(file_path)
                
                documents = load_documents(file_paths)
                nodes = chunk_documents(documents)
                index = build_vector_index(nodes)
                
                st.session_state.index = index
                st.session_state.query_engine = None  # Reset so it rebuilds with new index
                st.session_state.indexed_files = new_file_names
                logger.info(f"Indexed {len(nodes)} chunks from {len(uploaded_files)} files")
                st.success(f"✅ Indexed {len(nodes)} chunks from {len(uploaded_files)} files")
        else:
            st.info("📄 Files already indexed")

if st.session_state.index is not None:
    st.sidebar.success("📚 Knowledge base ready")

query = st.text_input("Ask a question based on your documents:")

if query:
    if st.session_state.index is None:
        st.warning("⚠️ Please upload files first")
    else:
        # Build query engine once and cache in session state
        if st.session_state.query_engine is None:
            try:
                st.session_state.query_engine = get_query_engine(st.session_state.index)
            except ValueError as e:
                st.error(f"⚠️ {str(e)}")
                logger.error(f"Query engine creation failed: {e}")
                st.stop()

        with st.spinner("🔍 Searching knowledge base..."):
            result = query_knowledge_base(st.session_state.query_engine, query)

        # Display the answer
        st.markdown(result["answer"])

        # Show which LLM was used
        st.caption(f"🤖 Answered by: {result['llm_used']}")

        # Display sources in an expander
        if result["sources"]:
            with st.expander("📄 View Sources"):
                for i, source in enumerate(result["sources"]):
                    st.markdown(f"**{source['filename']}**")
                    st.caption(source["text"])
                    if source["score"]:
                        st.caption(f"Relevance score: {source['score']:.4f}")
                    if i < len(result["sources"]) - 1:
                        st.divider()
