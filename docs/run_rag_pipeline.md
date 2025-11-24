# RAG Pipeline

> **Part 3:** Extract named entities using Retrieval-Augmented Generation (RAG)

---

## Overview

This pipeline uses RAG to extract Vietnamese NER by retrieving similar examples from a corpus before generation. It combines semantic search with LLM generation for improved accuracy.

---

## Quick Start

```bash
python scripts/run_rag_pipeline.py
```

---

## Configuration

```python
from src.config import NERRagConfig

config = NERRagConfig(
    model_name="Qwen/Qwen3-4B-Instruct-2507",
    embedding_model="AITeamVN/Vietnamese_Embedding_v2",
    rerank_model="AITeamVN/Vietnamese_Reranker",
    top_k_retrieval=3
)
```

**Parameters:**
- `top_k_retrieval`: Number of similar examples to retrieve (default: 3)
- `embedding_model`: Model for semantic search
- `rerank_model`: Model for retrieval reranking

---

## Input Requirements

| File | Path | Description |
|------|------|-------------|
| Test Data (Corpus) | `data/processed/test.json` | Examples for retrieval |
| Test Data | `data/processed/test.json` | Test dataset |