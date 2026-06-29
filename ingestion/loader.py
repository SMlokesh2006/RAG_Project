import os
import logging
import pandas as pd
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP, UPLOAD_DIR

logger = logging.getLogger(__name__)

MIN_PAGE_TEXT_LENGTH = 20


def _load_pdf_with_fitz(file_path: str) -> list[Document]:
    """Extract text from a PDF using PyMuPDF (fitz). Returns a list of Documents, one per page."""
    import fitz  # PyMuPDF

    documents = []
    with fitz.open(file_path) as pdf:
        for page_num in range(len(pdf)):
            page = pdf[page_num]
            text = page.get_text().strip()
            if len(text) < MIN_PAGE_TEXT_LENGTH:
                logger.debug(f"Skipping page {page_num + 1} of {file_path} (text too short: {len(text)} chars)")
                continue
            doc = Document(
                text=text,
                metadata={
                    "filename": os.path.basename(file_path),
                    "page_number": page_num + 1,
                    "source": file_path,
                    "type": "pdf",
                },
            )
            documents.append(doc)
    return documents


def _load_pdf_with_pypdf(file_path: str) -> list[Document]:
    """Fallback: extract text from a PDF using pypdf. Returns a list of Documents, one per page."""
    from pypdf import PdfReader

    documents = []
    reader = PdfReader(file_path)
    for page_num, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if len(text) < MIN_PAGE_TEXT_LENGTH:
            logger.debug(f"Skipping page {page_num + 1} of {file_path} (text too short: {len(text)} chars)")
            continue
        doc = Document(
            text=text,
            metadata={
                "filename": os.path.basename(file_path),
                "page_number": page_num + 1,
                "source": file_path,
                "type": "pdf",
            },
        )
        documents.append(doc)
    return documents


def _load_pdf(file_path: str) -> list[Document]:
    """Load a PDF using PyMuPDF first, falling back to pypdf on failure."""
    try:
        docs = _load_pdf_with_fitz(file_path)
        logger.info(f"Loaded PDF with PyMuPDF: {file_path}")
        return docs
    except Exception as e:
        logger.warning(f"PyMuPDF failed for {file_path}: {e}. Falling back to pypdf.")

    try:
        docs = _load_pdf_with_pypdf(file_path)
        logger.info(f"Loaded PDF with pypdf: {file_path}")
        return docs
    except Exception as e:
        logger.error(f"pypdf also failed for {file_path}: {e}")
        return []


def load_documents(file_paths: list[str]) -> list:
    all_documents = []

    for file_path in file_paths:
        ext = os.path.splitext(file_path)[1].lower()
        docs_from_file = []

        if ext == ".pdf":
            logger.info(f"Loading PDF: {file_path}")
            docs_from_file = _load_pdf(file_path)
        elif ext == ".csv":
            logger.info(f"Loading CSV: {file_path}")
            df = pd.read_csv(file_path)
            for idx, row in df.iterrows():
                text = "\n".join([f"{col}: {val}" for col, val in row.items()])
                doc = Document(
                    text=text,
                    metadata={
                        "filename": os.path.basename(file_path),
                        "row_number": idx + 1,
                        "source": file_path,
                        "type": "csv",
                    },
                )
                docs_from_file.append(doc)
            logger.info(f"Loaded {len(df)} rows from CSV")
        else:
            logger.warning(f"Unsupported file type '{ext}' for: {file_path}")

        # --- Validation ---
        count = len(docs_from_file)
        print(f"[Loader] {os.path.basename(file_path)}: {count} document(s) loaded")
        if count == 0:
            print(f"[Loader] ⚠️  WARNING: Zero documents loaded from {file_path}")
        else:
            preview = docs_from_file[0].text[:200]
            print(f"[Loader] First doc preview: {preview}")

        all_documents.extend(docs_from_file)

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
