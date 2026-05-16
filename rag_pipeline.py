from pathlib import Path
from typing import List, Dict, Any, Optional
import requests

import chromadb
from sentence_transformers import SentenceTransformer

from config import CHROMA_COLLECTION_NAME, EMBEDDING_MODEL_NAME, CHROMA_DIR, LOCAL_SERVER_PORT

LOCAL_LLM_API = f"http://127.0.0.1:{LOCAL_SERVER_PORT}/v1/chat/completions"


class LocalTextStore:
    def __init__(self, persist_dir: Optional[Path | str] = None):
        self.persist_dir = Path(persist_dir or CHROMA_DIR)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        
        self.embeddings = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

    def ingest_documents(self, docs: List[str], ids: List[str] = None):
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(docs))]
        
        embeddings = self.embeddings.encode(docs, show_progress_bar=False).tolist()
        self.collection.add(documents=docs, ids=ids, embeddings=embeddings)

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        query_embedding = self.embeddings.encode([query], show_progress_bar=False).tolist()
        results = self.collection.query(query_embeddings=query_embedding, n_results=top_k)
        return results["documents"][0] if results["documents"] else []


class RAGAgent:
    def __init__(self, store: LocalTextStore):
        self.store = store

    def _call_llm(self, prompt: str, max_tokens: int = 512) -> str:
        try:
            response = requests.post(LOCAL_LLM_API, json={
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
            }, timeout=120)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception:
            return "[ERROR] LLM unavailable"

    def answer(self, query: str, max_tokens: int = 512) -> str:
        context = self.store.retrieve(query, top_k=4)
        context_text = "\n---\n".join(context) if context else "[No relevant documents]"
        prompt = f"Context:\n{context_text}\n\nQuestion: {query}\nAnswer:"
        return self._call_llm(prompt, max_tokens)


def sample_documents() -> List[str]:
    return [
        "ChromaDB is a lightweight local vector store for retrieval-augmented generation.",
        "Quantizing models reduces memory and compute requirements while preserving reasoning.",
        "An OpenAI-compatible API allows drop-in replacement for existing applications.",
        "The Hybrid Router routes simple tasks locally and complex tasks to cloud.",
        "TPS (tokens-per-second) measures local inference performance.",
        "Air-gapped systems operate without internet, relying on local resources.",
    ]


if __name__ == "__main__":
    store = LocalTextStore()
    store.ingest_documents(sample_documents())
    
    agent = RAGAgent(store)
    query = "How does quantization work?"
    answer = agent.answer(query)
    print(f"Q: {query}\nA: {answer}")
