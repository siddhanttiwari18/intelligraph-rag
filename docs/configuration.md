# IntelliGraph-RAG — Configuration Guide

This guide documents the settings configuration schemas, environment variable prefixes, and LLM pricing rate variables.

---

## 1. Core Settings Schema (`Settings` class)

Central configurations are managed in `rag/config/config.py`. They support a default hierarchy:
1. Hardcoded python defaults.
2. Overrides loaded from the local `./rag_storage/config.json` file.
3. System environment overrides starting with `RAG_` (e.g. `RAG_LLM_MODEL`).

### Parameters Matrix

| Attribute | Type | Default | CLI / Env Key | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| `llm_model` | `str` | `deepseek-chat` | `RAG_LLM_MODEL` | Active model endpoint. |
| `embed_model` | `str` | `all-MiniLM-L6-v2` | `RAG_EMBED_MODEL` | Local SentenceTransformer name. |
| `parent_size` | `int` | `1500` | `RAG_PARENT_SIZE` | Chunk size of parent documents. |
| `parent_overlap`| `int` | `200` | `RAG_PARENT_OVERLAP` | Overlap size of parent documents. |
| `child_size` | `int` | `400` | `RAG_CHILD_SIZE` | Chunk size of child retrieval documents. |
| `child_overlap` | `int` | `50` | `RAG_CHILD_OVERLAP` | Overlap size of child retrieval documents. |
| `retrieve_k` | `int` | `20` | `RAG_RETRIEVE_K` | Initial retrieval candidate count. |
| `rerank_top_n` | `int` | `4` | `RAG_RERANK_TOP_N` | Chunks sent to the generator. |
| `confidence_threshold` | `float`| `0.6` | `RAG_CONFIDENCE_THRESHOLD` | Threshold before routing to planning nodes. |
| `max_retrieval_iterations`| `int` | `2` | `RAG_MAX_RETRIEVAL_ITERATIONS` | Maximum loops for complex queries. |
| `max_planner_depth`| `int` | `3` | `RAG_MAX_PLANNER_DEPTH` | Max sub-questions generated per loop. |
| `max_retrieved_chunks` | `int` | `20` | `RAG_MAX_RETRIEVED_CHUNKS` | Overall cumulative retrieval limit. |
| `agent_trace_visibility` | `bool` | `True` | `RAG_AGENT_TRACE_VISIBILITY` | Toggle trace displaying in chat messages. |
| `max_depth` | `int` | `2` | `RAG_MAX_DEPTH` | BFS graph traversal depth limits. |
| `graph_enabled` | `bool` | `True` | `RAG_GRAPH_ENABLED` | Toggle knowledge graph RAG layer. |
| `cache_ttl` | `float`| `300.0` | `RAG_CACHE_TTL` | Cache expiry time in seconds. |

---

## 2. Environment Variables & Secret Configuration

To configure connection parameters and secret keys:

```bash
# Set DeepSeek key (Required)
DEEPSEEK_API_KEY=sk-your-deepseek-api-key-here

# Optional: Override setting parameters in shell environment
export RAG_LLM_MODEL="gpt-4o"
export RAG_CONFIDENCE_THRESHOLD=0.75
```

---

## 3. Configurable Cost Rates & Presets

LLM pricing rates are configured dynamically in the UI under **Resource & Cost Insights** and saved to `./rag_storage/pricing_config.json`. 

### Rate Variables
Pricing calculations map to rates defined **per 1 Million tokens**:

* **Input Cost / 1M** (`input_rate`): Cost for non-cached prompt tokens.
* **Output Cost / 1M` (`output_rate`): Cost for completion/generation tokens.
* **Cached Input Cost / 1M** (`cached_input_rate`): Cost for prompt tokens hit by DeepSeek/OpenAI cache matches.
* **Embedding Cost / 1M** (`embedding_rate`): Cost of character tokens mapped to local embed indexes.

### Built-in Presets

The following pricing rates are pre-configured in the platform:

| Preset Name | Input Rate ($) | Output Rate ($) | Cached Input ($) | Embedding ($) |
| :--- | :---: | :---: | :---: | :---: |
| **DeepSeek Chat** | 0.27 | 1.10 | 0.14 | 0.02 |
| **GPT-4o** | 5.00 | 15.00 | 2.50 | 0.02 |
| **GPT-4.1 (GPT-4-turbo)** | 10.00 | 30.00 | 5.00 | 0.02 |
| **Claude Sonnet** | 3.00 | 15.00 | 0.30 | 0.02 |
| **Gemini** | 0.075 | 0.30 | 0.0375 | 0.02 |
| **Custom** | 1.00 | 2.00 | 0.50 | 0.02 |
