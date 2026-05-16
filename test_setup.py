#!/usr/bin/env python

import sys
import subprocess
from pathlib import Path

def test_imports():
    print("\n" + "="*60)
    print("TESTING IMPORTS".center(60))
    print("="*60)

    packages = {
        'fastapi': 'FastAPI',
        'uvicorn': 'Uvicorn',
        'pydantic': 'Pydantic',
        'requests': 'Requests',
        'chromadb': 'ChromaDB',
        'sentence_transformers': 'Sentence Transformers',
        'numpy': 'NumPy',
    }

    all_ok = True
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError as e:
            print(f"  ✗ {name}: {e}")
            all_ok = False

    return all_ok


def test_local_modules():
    print("\n" + "="*60)
    print("TESTING LOCAL MODULES".center(60))
    print("="*60)

    modules = [
        'config',
        'serve_local_model',
        'rag_pipeline',
        'hybrid_router',
        'benchmark_tps',
        'example_usage',
    ]

    all_ok = True
    for module in modules:
        try:
            code = (Path(__file__).parent / f"{module}.py").read_text()
            compile(code, f"{module}.py", 'exec')
            print(f"  ✓ {module}.py")
        except SyntaxError as e:
            print(f"  ✗ {module}.py: {e}")
            all_ok = False

    return all_ok


def test_file_structure():
    print("\n" + "="*60)
    print("TESTING FILE STRUCTURE".center(60))
    print("="*60)

    base_dir = Path(__file__).parent
    required_files = [
        'config.py',
        'serve_local_model.py',
        'rag_pipeline.py',
        'hybrid_router.py',
        'benchmark_tps.py',
        'example_usage.py',
        'requirements.txt',
        'README.md',
        'SETUP_GUIDE.md',
        '.env.example',
    ]

    all_ok = True
    for file in required_files:
        path = base_dir / file
        if path.exists():
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} (NOT FOUND)")
            all_ok = False

    return all_ok


def test_configuration():
    print("\n" + "="*60)
    print("TESTING CONFIGURATION".center(60))
    print("="*60)

    try:
        from config import (
            LOCAL_SERVER_HOST, LOCAL_SERVER_PORT, OLLAMA_HOST, OLLAMA_MODEL,
            CHROMA_COLLECTION_NAME, EMBEDDING_MODEL_NAME
        )

        print(f"  ✓ LOCAL_SERVER_HOST: {LOCAL_SERVER_HOST}")
        print(f"  ✓ LOCAL_SERVER_PORT: {LOCAL_SERVER_PORT}")
        print(f"  ✓ OLLAMA_HOST: {OLLAMA_HOST}")
        print(f"  ✓ OLLAMA_MODEL: {OLLAMA_MODEL}")
        print(f"  ✓ CHROMA_COLLECTION_NAME: {CHROMA_COLLECTION_NAME}")
        print(f"  ✓ EMBEDDING_MODEL_NAME: {EMBEDDING_MODEL_NAME}")

        return True
    except Exception as e:
        print(f"  ✗ Configuration error: {e}")
        return False


def test_core_classes():
    print("\n" + "="*60)
    print("TESTING CORE CLASSES".center(60))
    print("="*60)

    try:
        from hybrid_router import TaskComplexityAnalyzer
        complexity = TaskComplexityAnalyzer.analyze("Summarize this text")
        print(f"  ✓ TaskComplexityAnalyzer (test: {complexity}/10)")

        from rag_pipeline import LocalTextStore
        store = LocalTextStore()
        print(f"  ✓ LocalTextStore")

        from rag_pipeline import RAGAgent
        RAGAgent(store)
        print(f"  ✓ RAGAgent")

        from hybrid_router import HybridRouter
        HybridRouter()
        print(f"  ✓ HybridRouter")

        return True
    except Exception as e:
        print(f"  ✗ Class instantiation error: {e}")
        return False


def test_environment():
    print("\n" + "="*60)
    print("TESTING ENVIRONMENT".center(60))
    print("="*60)

    print(f"  ✓ Python version: {sys.version.split()[0]}")
    print(f"  ✓ Python path: {sys.executable}")

    try:
        result = subprocess.run(
            ['ollama', '--version'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            print(f"  ✓ Ollama: {result.stdout.strip()}")
        else:
            print(f"  ⚠ Ollama: Not responding correctly")
    except FileNotFoundError:
        print(f"  ⚠ Ollama: Not found in PATH (install from ollama.ai)")
    except Exception as e:
        print(f"  ⚠ Ollama: {e}")

    return True


def main():
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + "LOCAL AGENT STACK - SETUP VALIDATION".center(58) + "║")
    print("╚" + "="*58 + "╝")

    results = {
        "File Structure": test_file_structure(),
        "Imports": test_imports(),
        "Local Modules": test_local_modules(),
        "Configuration": test_configuration(),
        "Core Classes": test_core_classes(),
        "Environment": test_environment(),
    }

    print("\n" + "="*60)
    print("SUMMARY".center(60))
    print("="*60)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {test_name}")

    all_passed = all(results.values())

    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED!".center(60))
        print("\nNext steps:".center(60))
        print("  1. Start Ollama: ollama serve".center(60))
        print("  2. Start local server: python serve_local_model.py".center(60))
        print("  3. Run demo: python example_usage.py".center(60))
    else:
        print("✗ SOME TESTS FAILED".center(60))
        print("\nPlease fix the issues above, then run again.".center(60))
    print("="*60 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
