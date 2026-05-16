import requests
from openai import OpenAI
from rag_pipeline import LocalTextStore, RAGAgent, sample_documents
from hybrid_router import HybridRouter, TaskComplexityAnalyzer
from config import LOCAL_SERVER_PORT

LOCAL_API = f"http://127.0.0.1:{LOCAL_SERVER_PORT}"


def demo_rag():
    print("\n" + "="*60)
    print("DEMO 1: RAG PIPELINE".center(60))
    print("="*60)
    
    store = LocalTextStore()
    store.ingest_documents(sample_documents())
    print("\nDocuments ingested into ChromaDB")
    
    agent = RAGAgent(store)
    
    queries = ["What is ChromaDB?", "How does quantization work?"]
    for q in queries:
        try:
            a = agent.answer(q)
            print(f"\nQ: {q}\nA: {a[:150]}...")
        except Exception as e:
            print(f"Error: {e}")


def demo_router():
    print("\n" + "="*60)
    print("DEMO 2: HYBRID ROUTER".center(60))
    print("="*60)
    
    router = HybridRouter(prefer_local=True)
    
    tests = [
        ("Translate 'hello' to Spanish", "simple"),
        ("Top 5 ML frameworks?", "medium"),
        ("Design a distributed system", "complex"),
    ]
    
    print("\nComplexity Analysis:")
    for prompt, expected in tests:
        score = TaskComplexityAnalyzer.analyze(prompt)
        backend = router.decide_backend(prompt)
        print(f"  {expected:8} | Score: {score:2}/10 | Routes to: {backend}")
    
    prompt = "Summarize quantization in one sentence"
    try:
        result = router.route(prompt, max_tokens=100)
        backend = result["backend"]
        latency = result["latency"]
        response = result["response"]["choices"][0]["message"]["content"]
        print(f"\nRequest: {prompt}")
        print(f"Backend: {backend} | Latency: {latency:.2f}s")
        print(f"Response: {response[:150]}...")
    except Exception as e:
        print(f"Error: {e}")


def demo_api():
    print("\n" + "="*60)
    print("DEMO 3: LOCAL API".center(60))
    print("="*60)
    
    try:
        response = requests.get(f"{LOCAL_API}/health", timeout=5)
        health = response.json()
        print(f"\nHealth: {health['status']}")
    except Exception as e:
        print(f"Server not running: {e}")
        return
    
    try:
        response = requests.post(f"{LOCAL_API}/v1/chat/completions", json={
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "What is AI?"}
            ],
            "max_tokens": 100,
        }, timeout=120)
        result = response.json()
        answer = result["choices"][0]["message"]["content"]
        usage = result["usage"]
        
        print(f"\nChat Response: {answer[:150]}...")
        print(f"Tokens - Prompt: {usage['prompt_tokens']}, Completion: {usage['completion_tokens']}")
    except Exception as e:
        print(f"Error: {e}")


def demo_openai_swap():
    print("\n" + "="*60)
    print("DEMO 4: ONE-LINE SWAP".center(60))
    print("="*60)

    print("\n--- BEFORE (Cloud) ---")
    print("client = OpenAI(base_url='https://api.openai.com/v1', api_key='sk-...')")

    print("\n--- AFTER (Local) ---")
    print("client = OpenAI(base_url='http://localhost:8000/v1', api_key='ignored')")

    print("\nUsing the OpenAI Python client pointed at localhost...")
    try:
        client = OpenAI(base_url=f"http://localhost:{LOCAL_SERVER_PORT}/v1", api_key="ignored")
        response = client.chat.completions.create(
            model="gemma3:1b",
            messages=[{"role": "user", "content": "What is quantization in one sentence?"}],
            max_tokens=100,
        )
        print(f"\nResponse: {response.choices[0].message.content}")
        print(f"Model: {response.model} | Tokens: {response.usage.total_tokens}")
        print("\n✓ OpenAI client works against local server — zero other code changes needed.")
    except Exception as e:
        print(f"Error: {e}")


def demo_comparison():
    print("\n" + "="*60)
    print("DEMO 5: LOCAL VS CLOUD".center(60))
    print("="*60)
    
    print("\nLocal Inference:")
    print("  ✓ No API key | ✓ Privacy | ✓ Offline | ✓ 10-50 TPS")
    print("  ✗ Limited by hardware | ✗ Smaller models")
    
    print("\nCloud Inference (OpenAI):")
    print("  ✓ Unlimited compute | ✓ Better models | ✓ Managed service")
    print("  ✗ Costs per request | ✗ Internet required | ✗ Data to cloud")
    
    print("\nHybrid (This Stack):")
    print("  ✓ Best of both | ✓ Cost optimized | ✓ Fallback support")


def main():
    print("\n" + "="*60)
    print("LOCAL AGENT STACK - DEMO".center(60))
    print("="*60)
    
    demo_rag()
    demo_router()
    demo_api()
    demo_openai_swap()
    demo_comparison()
    
    print("\n" + "="*60)
    print("DEMO COMPLETE".center(60))
    print("="*60)
    print("\nNext: python benchmark_tps.py\n")


if __name__ == "__main__":
    main()
