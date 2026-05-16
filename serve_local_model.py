import time
import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import requests
from typing import Any, Dict, Optional, List

from config import LOCAL_SERVER_HOST, LOCAL_SERVER_PORT, OLLAMA_HOST, OLLAMA_MODEL, count_tokens

app = FastAPI(title="Local Agent Server")

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    model: str = OLLAMA_MODEL
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95

class CompletionRequest(BaseModel):
    prompt: str
    model: str = OLLAMA_MODEL
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7


def send_to_ollama_chat(messages: List[Dict], model: str, max_tokens: int, 
                        temperature: float, top_p: float) -> Dict[str, Any]:
    data = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "top_p": top_p, "num_predict": max_tokens}
    }
    response = requests.post(f"{OLLAMA_HOST}/api/chat", json=data, timeout=120)
    response.raise_for_status()
    content = response.json()["message"]["content"]
    
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": count_tokens(str(messages)),
            "completion_tokens": count_tokens(content),
            "total_tokens": count_tokens(str(messages) + content),
        }
    }


def send_to_ollama_completion(prompt: str, model: str, max_tokens: int,
                              temperature: float, top_p: float) -> Dict[str, Any]:
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "top_p": top_p, "num_predict": max_tokens}
    }
    response = requests.post(f"{OLLAMA_HOST}/api/generate", json=data, timeout=120)
    response.raise_for_status()
    text = response.json()["response"]
    
    return {
        "id": f"cmpl-{uuid.uuid4().hex[:12]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"text": text, "index": 0, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": count_tokens(prompt),
            "completion_tokens": count_tokens(text),
            "total_tokens": count_tokens(prompt + text),
        }
    }


@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [{"id": OLLAMA_MODEL, "object": "model", "owned_by": "ollama"}]}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    try:
        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        return send_to_ollama_chat(messages, req.model, req.max_tokens or 512, 
                                   req.temperature or 0.7, req.top_p or 0.95)
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail=f"Ollama not running at {OLLAMA_HOST}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/completions")
def completions(req: CompletionRequest):
    try:
        return send_to_ollama_completion(req.prompt, req.model, req.max_tokens or 512,
                                        req.temperature or 0.7, req.top_p or 0.95)
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail=f"Ollama not running at {OLLAMA_HOST}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    try:
        requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        return {"status": "healthy"}
    except:
        return {"status": "degraded"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=LOCAL_SERVER_HOST, port=LOCAL_SERVER_PORT)
