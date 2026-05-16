import time
import requests
import statistics
from typing import Dict, List, Any

from config import LOCAL_SERVER_PORT, count_tokens

LOCAL_API = f"http://127.0.0.1:{LOCAL_SERVER_PORT}/v1/chat/completions"

PROMPTS = {
    "short": "Summarize the key differences between ML and deep learning.",
    "medium": "Explain how transformer-based language models work with attention mechanisms.",
    "long": "Design a real-time fraud detection ML system covering data, features, models, serving, monitoring, and retraining.",
}


def measure_tps(prompt: str, iterations: int = 3) -> Dict[str, Any]:
    total_tokens = 0
    total_time = 0.0
    token_counts = []
    latencies = []
    
    print(f"  Running {iterations} iterations...")
    
    for i in range(iterations):
        try:
            start = time.time()
            response = requests.post(LOCAL_API, json={
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 256,
                "temperature": 0.7,
            }, timeout=120)
            elapsed = time.time() - start
            
            response.raise_for_status()
            text = response.json()["choices"][0]["message"]["content"]
            tokens = count_tokens(text)
            
            total_tokens += tokens
            total_time += elapsed
            token_counts.append(tokens)
            latencies.append(elapsed)
            
            tps = tokens / elapsed if elapsed > 0 else 0
            print(f"    Iter {i+1}: {tokens} tokens, {elapsed:.2f}s, {tps:.2f} TPS")
        except Exception as e:
            print(f"    Iter {i+1}: ERROR - {e}")
            return {"error": str(e)}
    
    avg_tps = total_tokens / total_time if total_time > 0 else 0
    
    return {
        "total_tokens": total_tokens,
        "total_time": total_time,
        "avg_tps": avg_tps,
        "avg_latency": statistics.mean(latencies),
        "min_latency": min(latencies),
        "max_latency": max(latencies),
        "std_latency": statistics.stdev(latencies) if len(latencies) > 1 else 0,
    }


def benchmark_suite() -> Dict[str, Any]:
    print("="*60)
    print("LOCAL INFERENCE BENCHMARKING".center(60))
    print("="*60)
    
    results = {}
    for prompt_type, prompt in PROMPTS.items():
        print(f"\nBenchmarking {prompt_type.upper()}...")
        result = measure_tps(prompt, iterations=3)
        results[prompt_type] = result
    
    print("\n" + "="*60)
    print("SUMMARY".center(60))
    print("="*60)
    
    for prompt_type, result in results.items():
        if "error" in result:
            print(f"\n{prompt_type.upper()}: ERROR - {result['error']}")
        else:
            print(f"\n{prompt_type.upper()}:")
            print(f"  Avg TPS: {result['avg_tps']:.2f}")
            print(f"  Latency: {result['avg_latency']:.2f}s ± {result['std_latency']:.2f}s")
            print(f"  Total: {result['total_tokens']} tokens in {result['total_time']:.2f}s")
    
    return results


if __name__ == "__main__":
    benchmark_suite()
