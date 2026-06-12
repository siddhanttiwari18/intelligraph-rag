# IntelliGraph-RAG — System Architecture

This document details the layered structure, data pipelines, and workflow orchestration of the **IntelliGraph-RAG** platform.

---

## 1. System Layout Overview

The codebase is organized into modular packages to isolate concerns and enforce boundary constraints between processing stages:

```text
IntelliGraph-RAG/
├── app.py                      # UI Presentation layer (Streamlit dashboard and widgets)
└── rag/
    ├── config/                 # Central settings configurations and env checks
    ├── utils/                  # Safe security checks and standard exception templates
    ├── services/               # Caching, background executors, and file sessions
    ├── models/                 # Neural sentence transformers and token interceptor wrappers
    ├── retrieval/              # Hybrid BM25/Vector databases and rerankers
    ├── graph/                  # Local NetworkX store, entity & relationship builders
    ├── analytics/              # Usage stats and latency trackers
    ├── agent/                  # State workflows planners and sufficiency metrics
    ├── pipelines/              # Segmented ingest and query engines
    └── pipeline.py             # Entrypoint RAGPipeline orchestrator
```

---

## 2. Ingest Pipeline Flow

When a document (PDF, TXT, or MD) is indexed, the Ingest Pipeline processes it sequentially:

```mermaid
sequenceDiagram
    autonumber
    participant U as User Upload
    participant I as IngestPipeline
    participant S as Security Module
    participant P as Document Parser
    participant V as Hybrid Vector Store
    participant G as Graph Builder
    participant T as Telemetry Service

    U->>I: Upload file name and bytes
    I->>S: validate_filename() & validate_file_content()
    S-->>I: File verified (no traversal, valid header)
    I->>P: load_document_pages()
    Note right of P: Fallback OCR runs if PDF text density is low.<br/>Tables are parsed into Markdown.
    P-->>I: Document pages list
    I->>I: parent_child_chunk_documents()
    I->>V: add_documents(child_chunks)
    Note right of V: Computes local embeddings and<br/>builds FAISS & BM25 indices.
    I->>G: extract_entities() & extract_relationships()
    Note right of G: Rule-based regex matches first.<br/>LLM triggers for complex turns.
    G->>G: add_entity() & add_relationship()
    Note right of G: Normalizes entity names to merge duplicates.
    I->>T: record_ingestion() (Success)
    I-->>U: Complete (chunks count, table counts)
```

---

## 3. Query Pipeline Flow (Agentic RAG)

When a query is submitted, the system adaptively routes the execution based on semantic classification and context evaluations:

```mermaid
graph TD
    Start[User Query] --> Rewriter[Query Rewriting: context history compression]
    Rewriter --> Classifier{Query Classifier}
    
    Classifier -->|Semantic Strategy| VectorOnly[Vector + BM25 Hybrid Retrieval]
    Classifier -->|Relationship Strategy| GraphOnly[Knowledge Graph BFS Retrieval]
    Classifier -->|Hybrid Strategy| HybridSearch[Vector + Graph Combined Retrieval]
    
    VectorOnly --> Rerank[Cross-Encoder Re-ranking]
    HybridSearch --> Rerank
    
    GraphOnly --> Evaluate[Evidence Sufficiency Evaluator]
    Rerank --> Evaluate
    
    Evaluate --> Suff{Confidence >= Threshold?}
    
    Suff -->|Yes / Simple Path| Answer[LLM Answer Generation with Citations]
    Suff -->|No / Complex Path| Plan{Planner Iteration < Max?}
    
    Plan -->|Yes| SubQs[Generate Sub-Questions]
    SubQs --> SearchLoops[Search vector/graph layers for missing details]
    SearchLoops --> Evaluate
    
    Plan -->|No| Answer
    
    Answer --> LogTelemetry[Log telemetries and token counts]
    LogTelemetry --> End[Response Sent to User]
```

---

## 4. Key Architectural Design Decisions

* **Unified API Facade**: `RAGPipeline` (in `rag/pipeline.py`) implements a facade pattern, acting as the single entrypoint for `app.py` while delegating implementation internals to `IngestPipeline` and `QueryPipeline`.
* **State Object Isolation**: State variables are encapsulated within `WorkflowState` and mutated inside individual nodes of the `RAGWorkflow` pipeline, preventing side effects.
* **Synchronized Settings**: `Settings` acts as the single source of truth for runtime configurations, mapping overrides from `config.json` and environmental prefixes `RAG_` dynamically.
