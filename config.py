import os
from dotenv import load_dotenv

load_dotenv()

UPLOAD_DIR = "data/uploads"
INDEX_DIR = "indexes"
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "app.log")
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # free HuggingFace model

# ── LLM Settings ──────────────────────────────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # "ollama" or "gemini"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
OLLAMA_BASE_URL = "http://localhost:11434"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
SIMILARITY_TOP_K = 5
