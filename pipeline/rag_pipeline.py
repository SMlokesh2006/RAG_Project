import os
import logging
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv

from config import LLM_MODEL

logger = logging.getLogger(__name__)


def get_query_engine(index):
    """
    Create a query engine from a LlamaIndex VectorStoreIndex.

    Sets up the OpenAI LLM using OPENAI_API_KEY from .env and configures
    the query engine with similarity_top_k=5 and response_mode="compact".

    Args:
        index: A LlamaIndex VectorStoreIndex object.

    Returns:
        A configured query engine ready to answer questions.
    """
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_key_here":
        logger.error("OPENAI_API_KEY is not set or still has the placeholder value. "
                      "Please update .env with a valid API key.")
        raise ValueError(
            "OpenAI API key is not configured. "
            "Please set a valid OPENAI_API_KEY in the .env file."
        )

    llm = OpenAI(
        model=LLM_MODEL,
        api_key=api_key,
    )
    Settings.llm = llm
    logger.info(f"Initialized OpenAI LLM with model '{LLM_MODEL}'")

    query_engine = index.as_query_engine(
        similarity_top_k=5,
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
                - "score": Relevance score if available.
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
        }

    except Exception as e:
        logger.error(f"Error querying knowledge base: {e}", exc_info=True)
        return {
            "answer": f"Error: {str(e)}",
            "sources": [],
        }
