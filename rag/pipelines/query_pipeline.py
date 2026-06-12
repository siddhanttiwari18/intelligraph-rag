import os
import time
import logging
from typing import Dict, Any, List, Optional

from rag.analytics.tracker import tracker
from rag.services.cache import platform_cache
from rag.agent.workflow import WorkflowState, RAGWorkflow

logger = logging.getLogger("rag_platform")


class QueryPipeline:
    """Orchestrates query expansion, conversational query rewriting, 
    hybrid retrieval (vector + BM25), re-ranking, and the agentic RAG workflow.
    """
    def __init__(self, vector_store, graph_store, graph_retriever, cross_encoder, llm, telemetry, config: dict):
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.graph_retriever = graph_retriever
        self.cross_encoder = cross_encoder
        self.llm = llm
        self.telemetry = telemetry
        self.config = config
        self.rerank_top_n = min(4, self.config.get("max_retrieved_chunks", 20))

    def set_config(self, config: dict) -> None:
        self.config.update(config)
        self.rerank_top_n = min(4, self.config.get("max_retrieved_chunks", 20))
        # Keep graph retriever depth synced with configuration
        self.graph_retriever.max_depth = self.config.get("max_depth", 2)

    def expand_query(self, query: str) -> List[str]:
        # Try to retrieve from cache
        cached_expansion = platform_cache.get("query_expansion", query)
        if cached_expansion:
            logger.info(f"Query expansion cache hit for: {query}")
            return cached_expansion

        try:
            logger.info(f"Running LLM query expansion for: {query}")
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if api_key:
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are a search query expansion assistant. Given a user question, "
                            "generate up to 3 alternate phrasings or synonyms to improve vector search and keyword retrieval. "
                            "Format the output strictly as a list of queries, one per line. "
                            "Do not include numbers, markdown lists, or any other explanations. Just the raw queries."
                        ),
                    },
                    {"role": "user", "content": f"Original query: {query}"},
                ]
                response = self.llm.invoke(messages)
                lines = response.content.strip().split("\n")
                expanded = [query]
                for line in lines:
                    line_clean = line.strip().lstrip("0123456789.-*• ").strip()
                    if line_clean and line_clean != query:
                        expanded.append(line_clean)
                
                result = expanded[:4]
                # Cache the result
                platform_cache.set("query_expansion", query, result)
                return result
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}. Falling back to original query.")
        return [query]

    def rewrite_query(self, query: str, history: List[dict]) -> str:
        if not history:
            return query
        try:
            logger.info("Rewriting conversational query based on chat history...")
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if api_key:
                # Format recent history turns (context window compression to last 6 messages)
                history_str = ""
                for msg in history[-6:]:
                    role = "User" if msg["role"] == "user" else "Assistant"
                    history_str += f"{role}: {msg['content']}\n"
                
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are a conversational query rewriting assistant. Given the chat history and the user's latest follow-up question, "
                            "rewrite the follow-up question into a standalone, self-contained search query. "
                            "The rewritten query should incorporate relevant details from the chat history so it can be searched in a vector database. "
                            "Do NOT answer the question. Just output the rewritten query, nothing else."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Chat History:\n{history_str}\nFollow-up Question: {query}\n\nStandalone Query:",
                    },
                ]
                response = self.llm.invoke(messages)
                rewritten = response.content.strip()
                if rewritten:
                    logger.info(f"Rewrote follow-up query: '{query}' -> '{rewritten}'")
                    return rewritten
        except Exception as e:
            logger.warning(f"Query rewriting failed: {e}. Falling back to original query.")
        return query

    def retrieve(self, question: str) -> List[dict]:
        # Try to retrieve from cache
        cached_retrieval = platform_cache.get("retrieval", question)
        if cached_retrieval:
            logger.info(f"Retrieval cache hit for query: {question}")
            return cached_retrieval

        # 1. Run Query Expansion
        queries = self.expand_query(question)

        # 2. Retrieve candidates for each query
        all_candidates = {}
        for q in queries:
            candidates = self.vector_store.search(q, k=20)
            for c in candidates:
                c_id = c["chunk_id"]
                # Deduplicate: keep the one with the highest fused score across queries
                if c_id not in all_candidates or c["fused_score"] > all_candidates[c_id]["fused_score"]:
                    all_candidates[c_id] = c

        merged_candidates = list(all_candidates.values())
        if not merged_candidates:
            return []

        # 3. Re-rank using Cross-Encoder based on the ORIGINAL question
        top_chunks = self.cross_encoder.rerank(question, merged_candidates, self.rerank_top_n)

        # Cache the result
        platform_cache.set("retrieval", question, top_chunks)
        return top_chunks

    def _build_context(self, chunks: List[dict]) -> str:
        parts = []
        for i, chunk in enumerate(chunks, start=1):
            chunk_type = chunk.get("chunk_type", "text")
            label = "Table" if chunk_type == "table" else "Source"
            parts.append(
                f"[{i}] {label}: {chunk['filename']}, Page: {chunk['page_number']}, Chunk Reference: {chunk['chunk_id']}\n"
                f"{chunk['parent_text']}"
            )
        return "\n\n---\n\n".join(parts)

    def ask(self, question: str, history: List[dict] = None) -> dict:
        if history is None:
            history = []

        t0_total = time.time()
        tracker.start_track()
        
        try:
            # Run state-based RAG workflow (Agentic classification + adaptive graph routing)
            state = WorkflowState(question, history)
            workflow = RAGWorkflow(self, self.config)
            final_state = workflow.run(state)

            # 1. Compile vector sources
            sources = []
            for chunk in final_state.retrieved_chunks:
                sources.append({
                    "source": chunk["filename"],
                    "page": chunk["page_number"],
                    "chunk_ref": chunk["chunk_id"],
                    "excerpt": chunk["text"],
                    "chunk_type": chunk.get("chunk_type", "text"),
                    "semantic_score": chunk.get("semantic_score", 0.0),
                    "bm25_score": chunk.get("bm25_score", 0.0),
                    "fused_score": chunk.get("fused_score", 0.0),
                    "cross_score": chunk.get("cross_score", 0.0),
                    "source_rank": chunk.get("source_rank", 1),
                })

            # 2. Compile graph citations with traceability to source documents
            seen_refs = {s["chunk_ref"] for s in sources}
            for src in final_state.graph_sources:
                ref = src.get("chunk_ref")
                if ref and ref not in seen_refs:
                    seen_refs.add(ref)
                    
                    excerpt_text = "Graph relationship connection."
                    chunk_match = next((c for c in self.vector_store.chunks if c["chunk_id"] == ref), None)
                    if chunk_match:
                        excerpt_text = chunk_match["text"]

                    sources.append({
                        "source": src.get("source_document", "unknown"),
                        "page": src.get("page_number", 1),
                        "chunk_ref": ref,
                        "excerpt": excerpt_text,
                        "chunk_type": "graph",
                        "semantic_score": 0.0,
                        "bm25_score": 0.0,
                        "fused_score": 1.0,
                        "cross_score": 0.0,
                        "source_rank": 999,  # Graph items are supplementary
                    })

            # Calculate metrics to log
            total_response_time = time.time() - t0_total
            track = tracker.get_track()
            
            # Matched entities in query
            matched_entities = self.graph_retriever.detect_entities(question)
            
            # Accessed docs
            accessed_docs = list(final_state.documents_consulted)
            
            # Context size in chars
            context_text = final_state.graph_context + "\n" + "".join(c.get("parent_text") or c["text"] for c in final_state.retrieved_chunks)
            context_size_chars = len(context_text)
            
            # Token estimate: prompt + context + answer
            hist_chars = sum(len(m["content"]) for m in history)
            token_estimate = (len(question) + context_size_chars + len(final_state.answer) + hist_chars) // 4
            
            # Record successful query
            self.telemetry.record_query(
                query=question,
                classification=final_state.query_classification,
                query_type=final_state.query_type,
                success=True,
                total_response_time=total_response_time,
                retrieval_latency=track["retrieval_latency"],
                graph_retrieval_latency=track["graph_retrieval_latency"],
                re_ranking_time=track["re_ranking_time"],
                embedding_time=track["embedding_time"],
                chunks_retrieved=len(final_state.retrieved_chunks),
                chunks_sent_to_llm=min(self.rerank_top_n, len(final_state.retrieved_chunks)),
                context_size_chars=context_size_chars,
                token_estimate=token_estimate,
                sub_questions_count=len(final_state.sub_questions),
                retrieval_iterations=final_state.iterations,
                graph_queries_count=track["graph_queries_count"],
                matched_entities=matched_entities,
                accessed_documents=accessed_docs,
                input_tokens=track.get("input_tokens", 0),
                output_tokens=track.get("output_tokens", 0),
                cached_input_tokens=track.get("cached_input_tokens", 0),
                embedding_tokens=track.get("embedding_tokens", 0)
            )

            return {
                "answer": final_state.answer,
                "sources": sources,
                "agent_trace": final_state.agent_trace.to_dict(),
            }
            
        except Exception as e:
            total_response_time = time.time() - t0_total
            track = tracker.get_track()
            self.telemetry.record_query(
                query=question,
                classification="Semantic",
                query_type="Simple",
                success=False,
                total_response_time=total_response_time,
                retrieval_latency=track["retrieval_latency"],
                graph_retrieval_latency=track["graph_retrieval_latency"],
                re_ranking_time=track["re_ranking_time"],
                embedding_time=track["embedding_time"],
                chunks_retrieved=0,
                chunks_sent_to_llm=0,
                context_size_chars=0,
                token_estimate=len(question) // 4,
                sub_questions_count=0,
                retrieval_iterations=0,
                graph_queries_count=track["graph_queries_count"],
                matched_entities=[],
                accessed_documents=[],
                error_message=str(e),
                input_tokens=track.get("input_tokens", 0),
                output_tokens=track.get("output_tokens", 0),
                cached_input_tokens=track.get("cached_input_tokens", 0),
                embedding_tokens=track.get("embedding_tokens", 0)
            )
            self.telemetry.record_error("query", str(e), {"query": question})
            raise e
