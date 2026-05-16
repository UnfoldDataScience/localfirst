import requests
import time
from typing import Dict, Any, Optional

from config import (
    LOCAL_SERVER_PORT, LOCAL_COMPLEXITY_THRESHOLD, PREFER_LOCAL,
    OPENAI_API_KEY, OPENAI_API_BASE, CLOUD_MODEL
)

LOCAL_API = f"http://127.0.0.1:{LOCAL_SERVER_PORT}/v1/chat/completions"


class TaskComplexityAnalyzer:
    SIMPLE = ["summarize", "translate", "keywords", "short answer", "format", "list", "yes or no"]
    COMPLEX = ["analyze", "design", "explain how", "why", "reasoning", "compare", "strategy", "code"]
    
    @staticmethod
    def analyze(prompt: str) -> int:
        lower = prompt.lower()
        words = len(prompt.split())
        
        score = min(10, max(1, words // 30))
        score -= sum(1 for kw in TaskComplexityAnalyzer.SIMPLE if kw in lower) * 1.5
        score += sum(1 for kw in TaskComplexityAnalyzer.COMPLEX if kw in lower) * 1.5
        
        if any(x in lower for x in ["code", "algorithm", "math", "formula"]):
            score += 2
        
        return min(10, max(1, int(score)))


class HybridRouter:
    def __init__(self, local_endpoint: str = LOCAL_API, cloud_key: Optional[str] = None,
                 cloud_model: str = CLOUD_MODEL, prefer_local: bool = PREFER_LOCAL,
                 threshold: int = LOCAL_COMPLEXITY_THRESHOLD):
        self.local = local_endpoint
        self.cloud_key = cloud_key or OPENAI_API_KEY
        self.cloud_model = cloud_model
        self.prefer_local = prefer_local
        self.threshold = threshold
        self.stats = {"total": 0, "local": 0, "cloud": 0, "errors": 0}

    def decide_backend(self, prompt: str, force_local: bool = False, force_cloud: bool = False) -> str:
        if force_local:
            return "local"
        if force_cloud and self.cloud_key:
            return "cloud"
        
        complexity = TaskComplexityAnalyzer.analyze(prompt)
        return "local" if (complexity <= self.threshold) else "cloud"

    def _call_local(self, prompt: str, max_tokens: int = 256, temp: float = 0.7) -> Dict[str, Any]:
        start = time.time()
        response = requests.post(self.local, json={
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temp,
        }, timeout=120)
        response.raise_for_status()
        return {"response": response.json(), "backend": "local", "latency": time.time() - start}

    def _call_cloud(self, prompt: str, max_tokens: int = 256, temp: float = 0.7) -> Dict[str, Any]:
        if not self.cloud_key:
            raise RuntimeError("Cloud API key not configured")
        
        start = time.time()
        response = requests.post(
            f"{OPENAI_API_BASE}/chat/completions",
            json={
                "model": self.cloud_model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temp,
            },
            headers={"Authorization": f"Bearer {self.cloud_key}", "Content-Type": "application/json"},
            timeout=120
        )
        response.raise_for_status()
        return {"response": response.json(), "backend": "cloud", "latency": time.time() - start}

    def route(self, prompt: str, max_tokens: int = 256, temp: float = 0.7,
              force_local: bool = False, force_cloud: bool = False) -> Dict[str, Any]:
        self.stats["total"] += 1
        backend = self.decide_backend(prompt, force_local, force_cloud)
        
        try:
            if backend == "local":
                result = self._call_local(prompt, max_tokens, temp)
                self.stats["local"] += 1
                return result
            else:
                result = self._call_cloud(prompt, max_tokens, temp)
                self.stats["cloud"] += 1
                return result
        except Exception as e:
            self.stats["errors"] += 1
            if backend == "local" and self.cloud_key:
                try:
                    result = self._call_cloud(prompt, max_tokens, temp)
                    self.stats["cloud"] += 1
                    return {**result, "fallback": "local"}
                except Exception as ce:
                    raise RuntimeError(f"Both backends failed. Local: {e}, Cloud: {ce}")
            raise RuntimeError(f"{backend} inference failed: {e}")


if __name__ == "__main__":
    router = HybridRouter(prefer_local=True)
    tests = [
        ("Summarize this in one sentence", "simple"),
        ("Top 3 programming languages?", "medium"),
        ("Design a distributed system", "complex"),
    ]
    for prompt, expected in tests:
        score = TaskComplexityAnalyzer.analyze(prompt)
        backend = router.decide_backend(prompt)
        print(f"{expected:8} | Complexity: {score:2}/10 | Routes to: {backend}")
    print(f"\nStats: {router.stats}")
