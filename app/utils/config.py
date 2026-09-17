"""Central configuration for the AI Campus Helpdesk, loaded from environment variables."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# --- LLM provider settings ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # "ollama" or "cloud"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

# Cloud LLM (OpenAI-compatible API), used only when LLM_PROVIDER=cloud
CLOUD_LLM_API_KEY = os.getenv("CLOUD_LLM_API_KEY", "")
CLOUD_LLM_BASE_URL = os.getenv("CLOUD_LLM_BASE_URL", "https://api.openai.com/v1")
CLOUD_LLM_MODEL = os.getenv("CLOUD_LLM_MODEL", "gpt-4o-mini")

# --- RAG settings ---
CHROMA_PERSIST_DIRECTORY = os.getenv(
    "CHROMA_PERSIST_DIRECTORY", str(BASE_DIR / "data" / "chroma")
)
KNOWLEDGE_BASE_DIR = os.getenv("KNOWLEDGE_BASE_DIR", str(BASE_DIR / "knowledge_base"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "700"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
TOP_K = int(os.getenv("TOP_K", "4"))

# Cosine distance (0 = identical, 2 = opposite) above which a retrieved chunk is
# considered too unrelated to the question to be used as context. Chosen from
# empirical testing: in-domain questions in this project scored ~0.3-0.45,
# out-of-domain questions scored ~0.73-0.87.
RELEVANCE_DISTANCE_THRESHOLD = float(os.getenv("RELEVANCE_DISTANCE_THRESHOLD", "0.6"))

# --- API settings ---
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# --- Departments known to the system ---
DEPARTMENTS = ["Academic", "Hostel", "Administration", "Student Services"]
