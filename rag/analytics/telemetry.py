import json
import time
import os
import re
from pathlib import Path

class TelemetryService:
    def __init__(self, persist_dir: str = "./rag_storage"):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.persist_dir / "telemetry.jsonl"
        self.pricing_path = self.persist_dir / "pricing_config.json"
        
        self.events = []
        self.load_events()
        self.load_pricing_config()
        
        # Cache for aggregated metrics
        self._cache = None
        self._cache_time = 0.0
        self.cache_ttl = 3.0 # seconds cache validity

    def load_events(self) -> None:
        self.events = []
        if self.log_path.exists():
            try:
                with open(self.log_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                self.events.append(json.loads(line))
                            except Exception:
                                pass
            except Exception as e:
                print(f"Error loading telemetry.jsonl: {e}")

    def load_pricing_config(self) -> None:
        # Define built-in presets
        self.presets = {
            "DeepSeek Chat": {
                "provider_name": "DeepSeek Chat",
                "input_rate": 0.27,
                "output_rate": 1.10,
                "cached_input_rate": 0.14,
                "embedding_rate": 0.02
            },
            "GPT-4o": {
                "provider_name": "GPT-4o",
                "input_rate": 5.00,
                "output_rate": 15.00,
                "cached_input_rate": 2.50,
                "embedding_rate": 0.02
            },
            "GPT-4.1 (GPT-4-turbo)": {
                "provider_name": "GPT-4.1 (GPT-4-turbo)",
                "input_rate": 10.00,
                "output_rate": 30.00,
                "cached_input_rate": 5.00,
                "embedding_rate": 0.02
            },
            "Claude Sonnet": {
                "provider_name": "Claude Sonnet",
                "input_rate": 3.00,
                "output_rate": 15.00,
                "cached_input_rate": 0.30,
                "embedding_rate": 0.02
            },
            "Gemini": {
                "provider_name": "Gemini",
                "input_rate": 0.075,
                "output_rate": 0.30,
                "cached_input_rate": 0.0375,
                "embedding_rate": 0.02
            },
            "Custom": {
                "provider_name": "Custom",
                "input_rate": 1.00,
                "output_rate": 2.00,
                "cached_input_rate": 0.50,
                "embedding_rate": 0.02
            }
        }
        
        # Default to DeepSeek Chat
        self.active_pricing = self.presets["DeepSeek Chat"].copy()
        
        if self.pricing_path.exists():
            try:
                with open(self.pricing_path, "r", encoding="utf-8") as f:
                    stored = json.load(f)
                    active_preset = stored.get("active_preset", "DeepSeek Chat")
                    if active_preset in self.presets:
                        self.active_pricing = self.presets[active_preset].copy()
                    
                    config = stored.get("config", {})
                    if config:
                        self.active_pricing.update(config)
            except Exception as e:
                print(f"Error loading pricing_config.json: {e}")

    def save_pricing_config(self, active_preset: str, config: dict) -> None:
        try:
            self.active_pricing = {
                "provider_name": active_preset,
                "input_rate": float(config.get("input_rate", 0.0)),
                "output_rate": float(config.get("output_rate", 0.0)),
                "cached_input_rate": float(config.get("cached_input_rate", 0.0)),
                "embedding_rate": float(config.get("embedding_rate", 0.0))
            }
            
            with open(self.pricing_path, "w", encoding="utf-8") as f:
                json.dump({
                    "active_preset": active_preset,
                    "config": self.active_pricing
                }, f, indent=2)
            
            self._cache = None  # invalidate cache
        except Exception as e:
            print(f"Error saving pricing config: {e}")

    def save_event(self, event_type: str, data: dict) -> None:
        event = {
            "event_type": event_type,
            "timestamp": time.time(),
            "data": data
        }
        self.events.append(event)
        self._cache = None  # invalidate cache
        
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
        except Exception as e:
            print(f"Error saving telemetry event: {e}")

    def record_query(self, query: str, classification: str, query_type: str, success: bool,
                     total_response_time: float, retrieval_latency: float, graph_retrieval_latency: float,
                     re_ranking_time: float, embedding_time: float, chunks_retrieved: int,
                     chunks_sent_to_llm: int, context_size_chars: int, token_estimate: int,
                     sub_questions_count: int, retrieval_iterations: int, graph_queries_count: int,
                     matched_entities: list[str], accessed_documents: list[str], error_message: str = None,
                     input_tokens: int = 0, output_tokens: int = 0, cached_input_tokens: int = 0,
                     embedding_tokens: int = 0) -> None:
        
        # Calculate pricing rates per token
        in_rate = self.active_pricing.get("input_rate", 0.0) / 1000000.0
        out_rate = self.active_pricing.get("output_rate", 0.0) / 1000000.0
        cached_rate = self.active_pricing.get("cached_input_rate", 0.0) / 1000000.0
        emb_rate = self.active_pricing.get("embedding_rate", 0.0) / 1000000.0
        
        non_cached_input = max(0, input_tokens - cached_input_tokens)
        llm_cost = (non_cached_input * in_rate) + (cached_input_tokens * cached_rate) + (output_tokens * out_rate)
        embedding_cost = embedding_tokens * emb_rate
        total_cost = llm_cost + embedding_cost

        self.save_event("query", {
            "query": query,
            "classification": classification,
            "query_type": query_type,
            "success": success,
            "total_response_time": total_response_time,
            "retrieval_latency": retrieval_latency,
            "graph_retrieval_latency": graph_retrieval_latency,
            "re_ranking_time": re_ranking_time,
            "embedding_time": embedding_time,
            "chunks_retrieved": chunks_retrieved,
            "chunks_sent_to_llm": chunks_sent_to_llm,
            "context_size_chars": context_size_chars,
            "token_estimate": token_estimate,
            "sub_questions_count": sub_questions_count,
            "retrieval_iterations": retrieval_iterations,
            "graph_queries_count": graph_queries_count,
            "matched_entities": matched_entities,
            "accessed_documents": accessed_documents,
            "error_message": error_message,
            
            # Pricing Metrics
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_input_tokens": cached_input_tokens,
            "embedding_tokens": embedding_tokens,
            "llm_cost": llm_cost,
            "embedding_cost": embedding_cost,
            "total_cost": total_cost,
            "provider_name": self.active_pricing.get("provider_name", "DeepSeek Chat")
        })

    def record_ingestion(self, file_name: str, file_type: str, file_size_bytes: int,
                         chunks_count: int, tables_extracted: int, ocr_processed: bool,
                         ocr_success: bool, success: bool, error_message: str = None) -> None:
        self.save_event("ingestion", {
            "file_name": file_name,
            "file_type": file_type,
            "file_size_bytes": file_size_bytes,
            "chunks_count": chunks_count,
            "tables_extracted": tables_extracted,
            "ocr_processed": ocr_processed,
            "ocr_success": ocr_success,
            "success": success,
            "error_message": error_message
        })

    def record_error(self, error_type: str, error_message: str, context: dict = None) -> None:
        self.save_event("error", {
            "error_type": error_type,
            "error_message": error_message,
            "context": context or {}
        })

    def clear(self) -> None:
        self.events = []
        self._cache = None
        if self.log_path.exists():
            try:
                self.log_path.unlink()
            except Exception:
                pass

    def get_metrics(self, pipeline=None) -> dict:
        now = time.time()
        if self._cache and (now - self._cache_time) < self.cache_ttl:
            return self._cache

        # Filter events
        queries = [e["data"] for e in self.events if e["event_type"] == "query"]
        ingestions = [e["data"] for e in self.events if e["event_type"] == "ingestion"]
        errors = [e["data"] for e in self.events if e["event_type"] == "error"]

        total_queries = len(queries)
        total_agent_executions = sum(1 for q in queries if q.get("query_type") == "Complex")
        total_graph_queries = sum(1 for q in queries if q.get("classification") in ("Relationship", "Hybrid") or q.get("graph_queries_count", 0) > 0)
        
        # Ingestion aggregates
        ocr_docs_processed = sum(1 for i in ingestions if i.get("ocr_processed"))
        ocr_successes = sum(1 for i in ingestions if i.get("ocr_processed") and i.get("ocr_success"))
        tables_extracted = sum(i.get("tables_extracted", 0) for i in ingestions)
        
        # System Resource sizes
        vector_store_size_bytes = 0
        store_path = self.persist_dir / "store.json"
        faiss_path = self.persist_dir / "faiss.index"
        if store_path.exists():
            vector_store_size_bytes += store_path.stat().st_size
        if faiss_path.exists():
            vector_store_size_bytes += faiss_path.stat().st_size

        graph_store_size_bytes = 0
        graph_path = self.persist_dir / "graph.json"
        if graph_path.exists():
            graph_store_size_bytes += graph_path.stat().st_size

        telemetry_size_bytes = 0
        if self.log_path.exists():
            telemetry_size_bytes += self.log_path.stat().st_size

        # Node / Relationship Counts from pipeline if available
        graph_nodes = 0
        graph_relationships = 0
        connected_components = 0
        orphan_nodes = 0
        avg_degree = 0.0
        relationship_density = 0.0
        top_entity_types = {}
        most_connected_entities = []
        entity_distribution = {}

        if pipeline and hasattr(pipeline, "graph_store"):
            G = pipeline.graph_store.graph
            graph_nodes = len(G.nodes())
            graph_relationships = len(G.edges())
            if graph_nodes > 0:
                import networkx as nx
                try:
                    connected_components = nx.number_weakly_connected_components(G)
                except Exception:
                    connected_components = 0
                orphan_nodes = len([n for n in G.nodes() if G.degree(n) == 0])
                avg_degree = (2.0 * graph_relationships / graph_nodes)
                if graph_nodes > 1:
                    relationship_density = graph_relationships / (graph_nodes * (graph_nodes - 1))
                
                # entity types
                for node, attrs in G.nodes(data=True):
                    etype = attrs.get("entity_type", "Unknown")
                    top_entity_types[etype] = top_entity_types.get(etype, 0) + 1
                
                # most connected
                degrees = list(G.degree())
                degrees.sort(key=lambda x: x[1], reverse=True)
                most_connected_entities = [{"entity": node, "connections": deg} for node, deg in degrees[:10]]
                entity_distribution = top_entity_types

        # Performance Metrics
        success_queries = [q for q in queries if q.get("success", True)]
        total_success = len(success_queries)
        
        avg_response_time = sum(q.get("total_response_time", 0.0) for q in queries) / total_queries if total_queries > 0 else 0.0
        avg_retrieval_latency = sum(q.get("retrieval_latency", 0.0) for q in queries) / total_queries if total_queries > 0 else 0.0
        avg_graph_retrieval_latency = sum(q.get("graph_retrieval_latency", 0.0) for q in queries) / total_queries if total_queries > 0 else 0.0
        avg_reranking_time = sum(q.get("re_ranking_time", 0.0) for q in queries) / total_queries if total_queries > 0 else 0.0
        avg_embedding_time = sum(q.get("embedding_time", 0.0) for q in queries) / total_queries if total_queries > 0 else 0.0
        avg_chunks_retrieved = sum(q.get("chunks_retrieved", 0) for q in queries) / total_queries if total_queries > 0 else 0.0
        
        # Percentiles
        response_times = sorted([q.get("total_response_time", 0.0) for q in queries])
        p95_response_time = response_times[int(len(response_times) * 0.95)] if response_times else 0.0
        p99_response_time = response_times[int(len(response_times) * 0.99)] if response_times else 0.0
        
        retrieval_success_rate = (total_success / total_queries) * 100 if total_queries > 0 else 100.0

        # Classification Distribution
        class_dist = {"Relationship": 0, "Semantic": 0, "Hybrid": 0}
        for q in queries:
            c = q.get("classification", "Semantic")
            class_dist[c] = class_dist.get(c, 0) + 1

        # Ingestion analytics
        total_ingestions = len(ingestions)
        avg_doc_size = sum(i.get("file_size_bytes", 0) for i in ingestions) / total_ingestions if total_ingestions > 0 else 0.0
        doc_types = {}
        for i in ingestions:
            t = i.get("file_type", ".txt")
            doc_types[t] = doc_types.get(t, 0) + 1

        ocr_success_rate = (ocr_successes / ocr_docs_processed) * 100 if ocr_docs_processed > 0 else 100.0

        # Token & Cost Monitoring Calculations
        in_rate = self.active_pricing.get("input_rate", 0.0) / 1000000.0
        out_rate = self.active_pricing.get("output_rate", 0.0) / 1000000.0
        cached_rate = self.active_pricing.get("cached_input_rate", 0.0) / 1000000.0
        emb_rate = self.active_pricing.get("embedding_rate", 0.0) / 1000000.0

        total_input_tokens = 0
        total_output_tokens = 0
        total_cached_input_tokens = 0
        total_embedding_tokens = 0
        total_llm_cost = 0.0
        total_embedding_cost = 0.0
        total_ai_cost = 0.0
        daily_cost = 0.0
        
        today_date = time.strftime("%Y-%m-%d", time.localtime(now))
        first_query_time = None

        for event in self.events:
            if event["event_type"] == "query":
                q = event["data"]
                timestamp = event.get("timestamp", now)
                if first_query_time is None or timestamp < first_query_time:
                    first_query_time = timestamp
                
                # Fetch query metrics (with legacy fallback checks)
                q_input = q.get("input_tokens")
                q_output = q.get("output_tokens")
                q_cached = q.get("cached_input_tokens", 0)
                q_emb = q.get("embedding_tokens")
                
                # Legacy compatibility fallback
                if q_input is None or q_output is None:
                    est_total = q.get("token_estimate", 0)
                    if est_total > 0:
                        q_input = int(est_total * 0.8)
                        q_output = int(est_total * 0.2)
                    else:
                        context_chars = q.get("context_size_chars", 0)
                        query_chars = len(q.get("query", ""))
                        q_input = max(1, (query_chars + context_chars) // 4)
                        q_output = 100
                    q_cached = 0
                
                if q_emb is None:
                    q_emb = q.get("context_size_chars", 0) // 4
                
                # Re-calculate costs dynamically using the current active settings
                non_cached = max(0, q_input - q_cached)
                q_llm_cost = (non_cached * in_rate) + (q_cached * cached_rate) + (q_output * out_rate)
                q_emb_cost = q_emb * emb_rate
                q_tot_cost = q_llm_cost + q_emb_cost
                
                total_input_tokens += q_input
                total_output_tokens += q_output
                total_cached_input_tokens += q_cached
                total_embedding_tokens += q_emb
                total_llm_cost += q_llm_cost
                total_embedding_cost += q_emb_cost
                total_ai_cost += q_tot_cost
                
                query_date = time.strftime("%Y-%m-%d", time.localtime(timestamp))
                if query_date == today_date:
                    daily_cost += q_tot_cost

        # Monthly Cost Projection
        monthly_cost_projection = 0.0
        if first_query_time is not None:
            elapsed_seconds = now - first_query_time
            elapsed_days = max(1.0, elapsed_seconds / 86400.0)
            monthly_cost_projection = (total_ai_cost / elapsed_days) * 30.0

        avg_tokens = (total_input_tokens + total_output_tokens) / total_queries if total_queries > 0 else 0.0
        avg_context = sum(q.get("context_size_chars", 0) for q in queries) / total_queries if total_queries > 0 else 0.0
        avg_chunks_sent_llm = sum(q.get("chunks_sent_to_llm", 0) for q in queries) / total_queries if total_queries > 0 else 0.0

        # Agent analytics
        agent_queries = [q for q in queries if q.get("query_type") == "Complex"]
        avg_agent_iterations = sum(q.get("retrieval_iterations", 0) for q in agent_queries) / len(agent_queries) if agent_queries else 0.0
        avg_sub_questions = sum(q.get("sub_questions_count", 0) for q in agent_queries) / len(agent_queries) if agent_queries else 0.0

        # Frequency tracking
        matched_entities = {}
        accessed_docs = {}
        common_tech = {}
        tech_keywords = {"kafka", "postgresql", "postgres", "mongodb", "mysql", "oracle", "redis", "sqlite", "docker", "kubernetes", "k8s", "ipv6", "ipv4", "http", "https", "grpc", "rest", "graphql", "python", "java", "golang", "react"}

        for q in queries:
            for ent in q.get("matched_entities", []):
                matched_entities[ent] = matched_entities.get(ent, 0) + 1
                if ent.lower() in tech_keywords:
                    common_tech[ent] = common_tech.get(ent, 0) + 1
            for doc in q.get("accessed_documents", []):
                accessed_docs[doc] = accessed_docs.get(doc, 0) + 1

        top_entities = sorted(matched_entities.items(), key=lambda x: x[1], reverse=True)[:10]
        top_docs = sorted(accessed_docs.items(), key=lambda x: x[1], reverse=True)[:10]
        top_tech = sorted(common_tech.items(), key=lambda x: x[1], reverse=True)[:10]

        # Daily Query Volume
        daily_volume = {}
        for event in self.events:
            if event["event_type"] == "query":
                dt = time.strftime("%Y-%m-%d", time.localtime(event["timestamp"]))
                daily_volume[dt] = daily_volume.get(dt, 0) + 1
        
        sorted_daily = sorted(daily_volume.items())

        # Error tracking
        failed_retrievals = sum(1 for err in errors if err.get("error_type") == "retrieval")
        failed_ocr = sum(1 for err in errors if err.get("error_type") == "ocr")
        failed_graph = sum(1 for err in errors if err.get("error_type") == "graph_build")
        failed_queries = sum(1 for err in errors if err.get("error_type") == "query")
        agent_errors = sum(1 for err in errors if err.get("error_type") == "agent")

        self._cache = {
            "total_queries": total_queries,
            "total_agent_executions": total_agent_executions,
            "total_graph_queries": total_graph_queries,
            "graph_nodes": graph_nodes,
            "graph_relationships": graph_relationships,
            "connected_components": connected_components,
            "orphan_nodes": orphan_nodes,
            "avg_degree": avg_degree,
            "relationship_density": relationship_density,
            "top_entity_types": top_entity_types,
            "most_connected_entities": most_connected_entities,
            "entity_distribution": entity_distribution,
            
            "avg_response_time": avg_response_time,
            "p95_response_time": p95_response_time,
            "p99_response_time": p99_response_time,
            "avg_retrieval_latency": avg_retrieval_latency,
            "avg_graph_retrieval_latency": avg_graph_retrieval_latency,
            "avg_reranking_time": avg_reranking_time,
            "avg_embedding_time": avg_embedding_time,
            "avg_chunks_retrieved": avg_chunks_retrieved,
            "retrieval_success_rate": retrieval_success_rate,
            
            "class_distribution": class_dist,
            
            "ocr_documents_processed": ocr_docs_processed,
            "ocr_success_rate": ocr_success_rate,
            "tables_extracted": tables_extracted,
            "avg_document_size": avg_doc_size,
            "document_type_distribution": doc_types,
            
            # Pricing & Detailed Token Metrics
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_cached_input_tokens": total_cached_input_tokens,
            "total_embedding_tokens": total_embedding_tokens,
            "total_tokens_consumed": total_input_tokens + total_output_tokens,
            "avg_tokens_per_query": avg_tokens,
            
            "total_llm_cost": total_llm_cost,
            "total_embedding_cost": total_embedding_cost,
            "total_ai_cost": total_ai_cost,
            "daily_cost": daily_cost,
            "monthly_cost_projection": monthly_cost_projection,
            
            "active_pricing": self.active_pricing,
            "pricing_presets": self.presets,
            
            "avg_context_size": avg_context,
            "avg_chunks_sent_llm": avg_chunks_sent_llm,
            
            "avg_agent_iterations": avg_agent_iterations,
            "avg_sub_questions": avg_sub_questions,
            
            "top_entities": top_entities,
            "top_documents": top_docs,
            "top_technologies": top_tech,
            
            "daily_volume": sorted_daily,
            
            "resource_vector_store_size_bytes": vector_store_size_bytes,
            "resource_graph_store_size_bytes": graph_store_size_bytes,
            "resource_telemetry_size_bytes": telemetry_size_bytes,
            
            "failed_retrievals": failed_retrievals,
            "failed_ocr": failed_ocr,
            "failed_graph_builds": failed_graph,
            "failed_queries": failed_queries,
            "agent_errors": agent_errors,
            "error_summaries": errors[-10:]
        }
        self._cache_time = now
        return self._cache
