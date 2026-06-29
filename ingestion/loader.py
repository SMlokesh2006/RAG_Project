import os
import logging
import pandas as pd
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.node_parser import SentenceSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP, UPLOAD_DIR

logger = logging.getLogger(__name__)

def load_documents(file_paths: list[str]) -> list:
    all_documents = []
    
    for file_path in file_paths:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            logger.info(f"Loading PDF: {file_path}")
            reader = SimpleDirectoryReader(input_files=[file_path])
            docs = reader.load_data()
            all_documents.extend(docs)
            logger.info(f"Loaded {len(docs)} document(s) from PDF")
        elif ext == ".csv":
            logger.info(f"Loading CSV: {file_path}")
            df = pd.read_csv(file_path)
            for idx, row in df.iterrows():
                text = "\n".join([f"{col}: {val}" for col, val in row.items()])
                doc = Document(
                    text=text,
                    metadata={"filename": os.path.basename(file_path), "row_number": idx + 1}
                )
                all_documents.append(doc)
            logger.info(f"Loaded {len(df)} rows from CSV")
        else:
            logger.warning(f"Unsupported file type '{ext}' for: {file_path}")

    logger.info(f"Total documents loaded: {len(all_documents)}")
    return all_documents

def chunk_documents(documents: list) -> list:
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    nodes = splitter.get_nodes_from_documents(documents)
    logger.info(f"Chunked {len(documents)} documents into {len(nodes)} nodes "
                f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return nodes

def save_uploaded_file(uploaded_file) -> str:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    logger.info(f"Saved uploaded file: {file_path}")
    return file_path
