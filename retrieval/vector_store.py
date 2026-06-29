import os
import logging
import faiss
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from config import INDEX_DIR, EMBED_MODEL

logger = logging.getLogger(__name__)

def build_vector_index(nodes: list):
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
    
    d = 384
    faiss_index = faiss.IndexFlatL2(d)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    index = VectorStoreIndex(
        nodes=nodes, 
        storage_context=storage_context, 
        embed_model=embed_model
    )
    
    os.makedirs(INDEX_DIR, exist_ok=True)
    index.storage_context.persist(persist_dir=INDEX_DIR)
    logger.info(f"Built and persisted vector index with {len(nodes)} nodes to '{INDEX_DIR}'")
    
    return index

def load_vector_index():
    if not os.path.exists(INDEX_DIR) or not os.listdir(INDEX_DIR):
        logger.info(f"No existing index found at '{INDEX_DIR}'")
        return None
    
    try:
        logger.info(f"Loading existing vector index from '{INDEX_DIR}'...")
        vector_store = FaissVectorStore.from_persist_dir(INDEX_DIR)
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store, persist_dir=INDEX_DIR
        )
        embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
        
        index = load_index_from_storage(
            storage_context=storage_context,
            embed_model=embed_model
        )
        logger.info("Vector index loaded successfully")
        return index
    except Exception as e:
        logger.error(f"Failed to load vector index from '{INDEX_DIR}': {e}", exc_info=True)
        return None
