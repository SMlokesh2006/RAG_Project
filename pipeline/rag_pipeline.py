import os
import logging
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from dotenv import load_dotenv

from config import (
    LLM_PROVIDER,
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
    GEMINI_MODEL,
    EMBED_MODEL,
    SIMILARITY_TOP_K,
)

logger = logging.getLogger(__name__)

# Track which LLM is actually in use (set by get_llm)
_active_llm_name: str = "none"


def get_llm():
    """
    Return a LlamaIndex-compatible LLM based on LLM_PROVIDER in config.

    - If "ollama": tries to connect to a local Ollama instance.
      On connection failure, automatically falls back to Gemini.
    - If "gemini": reads GEMINI_API_KEY from .env and returns the Gemini LLM.

    Raises:
        ValueError: If Gemini is selected/fallen-back-to but no API key is set.
    """
    global _active_llm_name
    load_dotenv()

    if LLM_PROVIDER == "ollama":
        try:
            from llama_index.llms.ollama import Ollama

            llm = Ollama(
                model=OLLAMA_MODEL,
                base_url=OLLAMA_BASE_URL,
                request_timeout=30.0,
            )
            # Verify connectivity with a lightweight call
            llm.complete("hi")
            _active_llm_name = f"ollama/{OLLAMA_MODEL}"
            logger.info(f"Initialized Ollama LLM: {_active_llm_name}")
            return llm
        except Exception as e:
            logger.warning(f"Ollama unavailable ({e}), falling back to Gemini")
            print("⚠️ Ollama unavailable, falling back to Gemini")

    # Gemini path (either explicit or fallback)
    from llama_index.llms.gemini import Gemini

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_key_here":
        logger.error("GEMINI_API_KEY is not set or still has the placeholder value.")
        raise ValueError(
            "Gemini API key is not configured. "
            "Please set a valid GEMINI_API_KEY in the .env file."
        )

    llm = Gemini(
        model_name=f"models/{GEMINI_MODEL}",
        api_key=api_key,
    )
    _active_llm_name = f"gemini/{GEMINI_MODEL}"
    logger.info(f"Initialized Gemini LLM: {_active_llm_name}")
    return llm


def get_query_engine(index):
    """
    Create a query engine from a LlamaIndex VectorStoreIndex.

    Sets up the LLM (via get_llm) and embedding model, applies them to
    the global LlamaIndex Settings, and returns a configured query engine.

    Args:
        index: A LlamaIndex VectorStoreIndex object.

    Returns:
        A configured query engine ready to answer questions.
    """
    llm = get_llm()
    Settings.llm = llm
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
    logger.info(f"Embedding model set to '{EMBED_MODEL}'")

    query_engine = index.as_query_engine(
        similarity_top_k=SIMILARITY_TOP_K,
        response_mode="compact",
    )
    return query_engine


def query_knowledge_base(query_engine, question: str) -> dict:
    """
    Query the knowledge base and return a structured response.

    Args:
        query_engine: A LlamaIndex query engine instance.
        question: The user's question string.

    Returns:
        A dict with:
            - "answer": The LLM-generated answer string.
            - "sources": A list of source dicts, each containing:
                - "filename": Source file name.
                - "text": The chunk text snippet (first 200 chars).
                - "score": Relevance score if available, else 0.0.
            - "llm_used": Which LLM produced the answer (e.g. "ollama/mistral").
    """
    try:
        logger.info(f"Querying knowledge base: '{question[:100]}...'")
        response = query_engine.query(question)

        sources = []
        if response.source_nodes:
            for node in response.source_nodes:
                metadata = node.node.metadata or {}
                filename = metadata.get("file_name") or metadata.get("filename", "Unknown")
                text_snippet = node.node.get_content()[:200]
                score = node.score if node.score is not None else 0.0

                sources.append({
                    "filename": filename,
                    "text": text_snippet,
                    "score": float(score),
                })

        logger.info(f"Query returned {len(sources)} sources")
        return {
            "answer": str(response),
            "sources": sources,
            "llm_used": _active_llm_name,
        }

    except Exception as e:
        logger.error(f"Error querying knowledge base: {e}", exc_info=True)
        return {
            "answer": f"Error: {str(e)}",
            "sources": [],
            "llm_used": "none",
        }
