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

# ── Streamlit Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Enterprise RAG - Knowledge Base",
    page_icon="🧠",
    layout="wide",
)

# ── Session State Initialization ───────────────────────────────────────
if "index" not in st.session_state:
    st.session_state.index = load_vector_index()

if "query_engine" not in st.session_state:
    st.session_state.query_engine = None

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = set()

if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "llm_provider" not in st.session_state:
    st.session_state.llm_provider = "ollama"

# ── Sidebar ────────────────────────────────────────────────────────────
with st.sidebar:
    # -- App branding --
    st.markdown("# 🧠 Enterprise RAG")
    st.markdown("**Multi-Source Knowledge Base**")
    st.divider()

    # -- Upload section --
    st.markdown("### 📁 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF and CSV files",
        type=["pdf", "csv"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        # Only re-index when the set of uploaded file names changes
        new_file_names = {f.name for f in uploaded_files}
        if new_file_names != st.session_state.indexed_files:
            with st.spinner("Processing files…"):
                file_paths = []
                for file in uploaded_files:
                    file_path = save_uploaded_file(file)
                    file_paths.append(file_path)

                documents = load_documents(file_paths)
                nodes = chunk_documents(documents)
                index = build_vector_index(nodes)

                st.session_state.index = index
                st.session_state.query_engine = None  # force rebuild
                st.session_state.indexed_files = new_file_names
                st.session_state.total_chunks = len(nodes)
                logger.info(
                    f"Indexed {len(nodes)} chunks from {len(uploaded_files)} files"
                )

            # Show indexing results
            st.success(f"✅ {len(uploaded_files)} files indexed")
            st.caption(f"📦 {len(nodes)} total chunks")
        else:
            st.success(f"✅ {len(st.session_state.indexed_files)} files indexed")
            st.caption(f"📦 {st.session_state.total_chunks} total chunks")

    st.divider()

    # -- Settings section --
    st.markdown("### ⚙️ Settings")

    provider_options = ["Ollama (Local)", "Gemini (API)"]
    # Map session_state value to selectbox index
    current_idx = 0 if st.session_state.llm_provider == "ollama" else 1

    selected_provider = st.selectbox(
        "LLM Provider",
        options=provider_options,
        index=current_idx,
        label_visibility="collapsed",
    )

    # Derive canonical provider key from selectbox label
    new_provider = "ollama" if selected_provider == "Ollama (Local)" else "gemini"

    if new_provider != st.session_state.llm_provider:
        st.session_state.llm_provider = new_provider
        st.session_state.query_engine = None  # force rebuild with new LLM
        logger.info(f"LLM provider switched to '{new_provider}'")

    # LLM status indicator
    if st.session_state.llm_provider == "ollama":
        st.markdown("🟢 Ollama running locally")
    else:
        st.markdown("🔵 Gemini API")

# ── Main Area ──────────────────────────────────────────────────────────
# Header row with title + clear history button
header_left, header_right = st.columns([6, 1])
with header_left:
    st.markdown("## Ask Your Knowledge Base")
    st.caption("Upload documents in the sidebar, then ask anything.")
with header_right:
    if st.session_state.chat_history:
        if st.button("🗑️ Clear History"):
            st.session_state.chat_history = []
            st.rerun()

# ── No index loaded → prompt user ──────────────────────────────────────
if st.session_state.index is None:
    st.info("👈 Upload PDF or CSV files from the sidebar to get started")
    st.stop()

# ── Query interface ────────────────────────────────────────────────────
input_col, btn_col = st.columns([5, 1])
with input_col:
    query = st.text_input(
        "Ask a question about your documents…",
        label_visibility="collapsed",
        placeholder="Ask a question about your documents…",
    )
with btn_col:
    ask_clicked = st.button("Ask", type="primary", use_container_width=True)

# Process query
if ask_clicked and query:
    # Build query engine if needed
    if st.session_state.query_engine is None:
        try:
            st.session_state.query_engine = get_query_engine(
                st.session_state.index,
                provider=st.session_state.llm_provider,
            )
        except ValueError as e:
            st.error(f"⚠️ {str(e)}")
            logger.error(f"Query engine creation failed: {e}")
            st.stop()

    with st.spinner("🔍 Searching knowledge base…"):
        result = query_knowledge_base(st.session_state.query_engine, query)

    # Append to chat history
    st.session_state.chat_history.append({
        "question": query,
        "answer": result["answer"],
        "sources": result["sources"],
        "llm_used": result["llm_used"],
    })
    st.rerun()  # rerun so the new entry renders immediately in history

# ── Chat History ───────────────────────────────────────────────────────
if st.session_state.chat_history:
    st.divider()
    st.markdown("### 💬 Query History")

    # Display newest first
    for entry in reversed(st.session_state.chat_history):
        st.info(f"🙋 **You:** {entry['question']}")
        st.success(f"🤖 **Answer:** {entry['answer']}")
        st.caption(f"Answered by: {entry['llm_used']}")

        if entry["sources"]:
            with st.expander(f"📄 Sources ({len(entry['sources'])})"):
                for i, source in enumerate(entry["sources"]):
                    st.markdown(f"**{source['filename']}**")
                    st.caption(source["text"])
                    if source.get("score"):
                        st.caption(f"Relevance: {source['score']:.4f}")
                    if i < len(entry["sources"]) - 1:
                        st.divider()
        st.markdown("---")
