"""MESA reference adapters.

Each adapter wraps a different memory architecture and implements the
MemoryAdapter interface (reset / inject / ask / stored_facts).

Adapters:
  KeywordAdapter  — LLM extraction + TF-IDF retrieval. No heavy deps.
  Mem0Adapter     — Mem0 (mem0ai) extraction and retrieval.
  ChromaAdapter   — ChromaDB vector store + LLM answering.

All adapters require an OpenAI-compatible inference client. Pass it at
construction time so you can point to any local or cloud endpoint.
"""
