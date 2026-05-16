import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
CHROMA_DIR = MODEL_DIR / "chroma_store"


def count_tokens(text: str) -> int:
    return max(1, len(text) // 4)

LOCAL_SERVER_HOST = os.environ.get("LOCAL_SERVER_HOST", "0.0.0.0")
LOCAL_SERVER_PORT = int(os.environ.get("LOCAL_SERVER_PORT", "8000"))
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:1b")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
CLOUD_MODEL = os.environ.get("CLOUD_MODEL", "gpt-3.5-turbo")

CHROMA_COLLECTION_NAME = "local_agent_docs"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LOCAL_COMPLEXITY_THRESHOLD = int(os.environ.get("LOCAL_COMPLEXITY_THRESHOLD", "3"))
PREFER_LOCAL = os.environ.get("PREFER_LOCAL", "true").lower() == "true"
