import os

UPLOAD_DIR = "data/uploads"
INDEX_DIR = "indexes"
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "app.log")
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # free HuggingFace model
LLM_MODEL = "gpt-3.5-turbo"
