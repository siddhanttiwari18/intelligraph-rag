import time
from sentence_transformers import CrossEncoder
from rag.analytics.tracker import tracker


class Reranker:
    """Wraps CrossEncoder for context chunk re-ranking, logging latency to the platform tracker."""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[dict], top_n: int) -> list[dict]:
        """Re-ranks candidates using cross-encoder scores and returns the top_n items."""
        if not candidates:
            return []
            
        t0 = time.time()
        try:
            pairs = [[query, c["text"]] for c in candidates]
            scores = self.model.predict(pairs)

            for c, score in zip(candidates, scores):
                c["cross_score"] = float(score)

            # Sort by cross-encoder score descending
            candidates.sort(key=lambda x: x["cross_score"], reverse=True)
            
            # Take the top N reranked chunks
            top_chunks = candidates[:top_n]

            # Attach source rankings (1-indexed)
            for rank, chunk in enumerate(top_chunks, start=1):
                chunk["source_rank"] = rank
                
            return top_chunks
        finally:
            tracker.add_re_ranking_time(time.time() - t0)
