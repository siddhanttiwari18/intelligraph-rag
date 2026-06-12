import os
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from rag.config.config import Settings
from rag.models.embeddings import LocalEmbeddings
from rag.models.llm import create_llm_client, TokenTrackerWrapper
from rag.retrieval.vector_store import HybridStore
from rag.retrieval.re_ranker import Reranker
from rag.services.sessions import SessionManager
from rag.analytics.telemetry import TelemetryService
from rag.graph.graph_store import GraphStore
from rag.graph.graph_retriever import GraphRetriever
from rag.graph.entity_extractor import EntityExtractor
from rag.graph.relationship_extractor import RelationshipExtractor
from rag.pipelines.ingest_pipeline import IngestPipeline
from rag.pipelines.query_pipeline import QueryPipeline

logger = logging.getLogger("rag_platform")


class RAGPipeline:
    """Consolidated orchestrator representing the RAG platform.
    
    Acts as the entrypoint for app.py, delegating queries to QueryPipeline and
    ingest/deletion commands to IngestPipeline.
    """
    def __init__(
        self,
        persist_dir: str = "./rag_storage",
        llm_model: Optional[str] = None,
        parent_size: Optional[int] = None,
        parent_overlap: Optional[int] = None,
        child_size: Optional[int] = None,
        child_overlap: Optional[int] = None,
        retrieve_k: Optional[int] = None,
        rerank_top_n: Optional[int] = None,
    ):
        # 1. Load centralized settings
        self.settings = Settings(persist_dir=persist_dir)
        
        # Override settings if explicit parameters are passed
        if llm_model:
            self.settings.llm_model = llm_model
        if parent_size is not None:
            self.settings.parent_size = parent_size
        if parent_overlap is not None:
            self.settings.parent_overlap = parent_overlap
        if child_size is not None:
            self.settings.child_size = child_size
        if child_overlap is not None:
            self.settings.child_overlap = child_overlap
        if retrieve_k is not None:
            self.settings.retrieve_k = retrieve_k
        if rerank_top_n is not None:
            self.settings.rerank_top_n = rerank_top_n

        self.persist_dir = persist_dir
        self.retrieve_k = self.settings.retrieve_k
        self.rerank_top_n = self.settings.rerank_top_n
        self.parent_size = self.settings.parent_size
        self.parent_overlap = self.settings.parent_overlap
        self.child_size = self.settings.child_size
        self.child_overlap = self.settings.child_overlap
        self.llm_model = self.settings.llm_model

        # 2. Instantiate core services & models
        self.embeddings = LocalEmbeddings(model_name=self.settings.embed_model)
        self.vector_store = HybridStore(
            embeddings=self.embeddings,
            persist_dir=persist_dir,
        )
        self.session_manager = SessionManager(persist_dir=persist_dir)
        self.telemetry = TelemetryService(persist_dir=persist_dir)
        
        self.graph_store = GraphStore(persist_dir=persist_dir)
        self.graph_retriever = GraphRetriever(self.graph_store, max_depth=self.settings.max_depth)
        self.entity_extractor = EntityExtractor()
        self.relationship_extractor = RelationshipExtractor()

        self._llm = None
        self._cross_encoder = None

        # Config dictionary matches expectations of WorkflowState / RAGWorkflow
        self.config = {
            "confidence_threshold": self.settings.confidence_threshold,
            "max_retrieval_iterations": self.settings.max_retrieval_iterations,
            "max_planner_depth": self.settings.max_planner_depth,
            "max_retrieved_chunks": self.settings.max_retrieved_chunks,
            "agent_trace_visibility": self.settings.agent_trace_visibility,
            "max_depth": self.settings.max_depth,
            "graph_enabled": self.settings.graph_enabled,
        }

        # 3. Instantiate segmented pipelines
        self.ingest_pipeline = IngestPipeline(
            vector_store=self.vector_store,
            graph_store=self.graph_store,
            graph_retriever=self.graph_retriever,
            entity_extractor=self.entity_extractor,
            relationship_extractor=self.relationship_extractor,
            telemetry=self.telemetry,
            llm=self.llm
        )
        
        self.query_pipeline = QueryPipeline(
            vector_store=self.vector_store,
            graph_store=self.graph_store,
            graph_retriever=self.graph_retriever,
            cross_encoder=self.cross_encoder,
            llm=self.llm,
            telemetry=self.telemetry,
            config=self.config
        )

    def set_config(self, config: dict) -> None:
        """Updates runtime and persistent configuration."""
        self.config.update(config)
        self.settings.update(config)
        self.rerank_top_n = min(4, self.config.get("max_retrieved_chunks", 20))
        
        # Keep sub-components synchronized
        self.graph_retriever.max_depth = self.config.get("max_depth", 2)
        if hasattr(self, "query_pipeline") and self.query_pipeline:
            self.query_pipeline.set_config(self.config)

    @property
    def llm(self) -> TokenTrackerWrapper:
        if self._llm is None:
            self._llm = create_llm_client(self.llm_model)
            # Link llm to IngestPipeline once created
            if hasattr(self, "ingest_pipeline") and self.ingest_pipeline:
                self.ingest_pipeline.llm = self._llm
        return self._llm

    @property
    def cross_encoder(self) -> Reranker:
        if self._cross_encoder is None:
            self._cross_encoder = Reranker()
            # Link to QueryPipeline once created
            if hasattr(self, "query_pipeline") and self.query_pipeline:
                self.query_pipeline.cross_encoder = self._cross_encoder
        return self._cross_encoder

    # Ingestion delegations
    def ingest_file(self, file_name: str, file_bytes: bytes) -> dict:
        return self.ingest_pipeline.ingest_file(
            file_name=file_name,
            file_bytes=file_bytes,
            parent_size=self.parent_size,
            parent_overlap=self.parent_overlap,
            child_size=self.child_size,
            child_overlap=self.child_overlap
        )

    def delete_document(self, filename: str) -> dict:
        return self.ingest_pipeline.delete_document(filename)

    def reindex_document(self, filename: str, file_bytes: bytes) -> dict:
        self.delete_document(filename)
        return self.ingest_file(filename, file_bytes)

    def get_documents_metadata(self) -> List[dict]:
        return self.vector_store.get_documents_metadata()

    # Query delegations
    def expand_query(self, query: str) -> List[str]:
        return self.query_pipeline.expand_query(query)

    def rewrite_query(self, query: str, history: List[dict]) -> str:
        return self.query_pipeline.rewrite_query(query, history)

    def retrieve(self, question: str) -> List[dict]:
        return self.query_pipeline.retrieve(question)

    def _build_context(self, chunks: List[dict]) -> str:
        return self.query_pipeline._build_context(chunks)

    def ask(self, question: str, history: List[dict] = None) -> dict:
        return self.query_pipeline.ask(question, history)

    def rebuild_graph(self) -> dict:
        return self.ingest_pipeline.rebuild_graph()

    def get_stats(self) -> dict:
        return {"indexed_chunks": self.vector_store.get_chunk_count()}

    def clear_index(self) -> None:
        logger.info("Clearing index database and graph caches...")
        self.vector_store.clear()
        self.graph_store.clear()
        self.graph_retriever.clear_cache()
        
        # Invalidate platform cache
        from rag.services.cache import platform_cache
        platform_cache.invalidate()
