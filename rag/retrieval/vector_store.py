import json
import time
from pathlib import Path
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from rag.analytics.tracker import tracker


class HybridStore:
    def __init__(self, embeddings, persist_dir: str = "./rag_storage"):
        self.embeddings = embeddings
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.store_path = self.persist_dir / "store.json"
        self.faiss_path = self.persist_dir / "faiss.index"

        self.chunks = []
        self.faiss_index = None
        self.bm25 = None

        self.load()

    def load(self) -> None:
        # Load chunks from store.json
        if self.store_path.exists():
            try:
                with open(self.store_path, "r", encoding="utf-8") as f:
                    self.chunks = json.load(f)
            except Exception as e:
                print(f"Error loading store.json: {e}")
                self.chunks = []

        # Load FAISS index or rebuild it if chunks exist
        if self.chunks:
            if self.faiss_path.exists():
                try:
                    self.faiss_index = faiss.read_index(str(self.faiss_path))
                except Exception as e:
                    print(f"Error reading faiss.index: {e}, rebuilding...")
                    self.rebuild_faiss()
            else:
                self.rebuild_faiss()

            # Initialize BM25
            self.rebuild_bm25()
        else:
            self.faiss_index = None
            self.bm25 = None

    def rebuild_faiss(self) -> None:
        if not self.chunks:
            self.faiss_index = None
            return

        embeddings = []
        valid_chunks = []
        for c in self.chunks:
            if "embedding" in c:
                embeddings.append(c["embedding"])
                valid_chunks.append(c)
            else:
                # Fallback if embedding is missing
                emb = self.embeddings.embed_query(c["text"])
                c["embedding"] = emb
                embeddings.append(emb)
                valid_chunks.append(c)

        self.chunks = valid_chunks
        embedding_dim = len(embeddings[0])
        self.faiss_index = faiss.IndexFlatL2(embedding_dim)
        self.faiss_index.add(np.array(embeddings, dtype=np.float32))

    def rebuild_bm25(self) -> None:
        if not self.chunks:
            self.bm25 = None
            return
        tokenized_corpus = [c["text"].lower().split() for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def save(self) -> None:
        # Save chunks metadata
        with open(self.store_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

        # Save FAISS index
        if self.faiss_index is not None:
            faiss.write_index(self.faiss_index, str(self.faiss_path))
        else:
            if self.faiss_path.exists():
                self.faiss_path.unlink(missing_ok=True)

    def add_documents(self, new_chunks: list[dict]) -> int:
        if not new_chunks:
            return 0

        # Generate embeddings for new chunks
        texts = [c["text"] for c in new_chunks]
        embeddings = self.embeddings.embed_documents(texts)

        for chunk, emb in zip(new_chunks, embeddings):
            chunk["embedding"] = emb
            self.chunks.append(chunk)

        self.rebuild_faiss()
        self.rebuild_bm25()
        self.save()
        return len(new_chunks)

    def delete_document(self, filename: str) -> int:
        initial_count = len(self.chunks)
        self.chunks = [c for c in self.chunks if c["filename"] != filename]
        deleted_count = initial_count - len(self.chunks)

        if deleted_count > 0:
            if self.chunks:
                self.rebuild_faiss()
                self.rebuild_bm25()
            else:
                self.faiss_index = None
                self.bm25 = None
            self.save()
        return deleted_count

    def search(self, query: str, k: int = 20, semantic_weight: float = 0.7) -> list[dict]:
        t0 = time.time()
        try:
            if not self.chunks or self.faiss_index is None or self.bm25 is None:
                return []

            # 1. Semantic search via FAISS
            query_vector = self.embeddings.embed_query(query)
            search_k = min(len(self.chunks), max(k * 2, 20))
            distances, indices = self.faiss_index.search(
                np.array([query_vector], dtype=np.float32), search_k
            )

            semantic_results = {}
            for dist, idx in zip(distances[0], indices[0]):
                if idx == -1 or idx >= len(self.chunks):
                    continue
                # Convert L2 distance to similarity score
                sim = 1.0 / (1.0 + float(dist))
                semantic_results[idx] = sim

            # 2. BM25 search
            tokenized_query = query.lower().split()
            bm25_scores = self.bm25.get_scores(tokenized_query)

            # Get top search_k indices
            bm25_indices = np.argsort(bm25_scores)[::-1][:search_k]
            bm25_results = {}
            for idx in bm25_indices:
                score = float(bm25_scores[idx])
                bm25_results[idx] = score

            # 3. Score Fusion with Min-Max normalization
            all_indices = set(semantic_results.keys()).union(set(bm25_results.keys()))
            if not all_indices:
                return []

            # Min-max values for semantic
            sem_values = list(semantic_results.values())
            min_sem, max_sem = (min(sem_values), max(sem_values)) if sem_values else (0.0, 1.0)

            # Min-max values for BM25
            bm_values = list(bm25_results.values())
            min_bm, max_bm = (min(bm_values), max(bm_values)) if bm_values else (0.0, 1.0)

            fused_candidates = []
            for idx in all_indices:
                # Normalize Semantic Score
                sem_score = semantic_results.get(idx, 0.0)
                if max_sem > min_sem:
                    norm_sem = (sem_score - min_sem) / (max_sem - min_sem)
                else:
                    norm_sem = 1.0 if sem_score > 0 else 0.0

                # Normalize BM25 Score
                bm_score = bm25_results.get(idx, 0.0)
                if max_bm > min_bm:
                    norm_bm = (bm_score - min_bm) / (max_bm - min_bm)
                else:
                    norm_bm = 1.0 if bm_score > 0 else 0.0

                # Fused score
                fused_score = semantic_weight * norm_sem + (1.0 - semantic_weight) * norm_bm

                chunk = self.chunks[idx].copy()
                # Strip embedding for light serialization and transfer
                chunk.pop("embedding", None)
                chunk["fused_score"] = fused_score
                chunk["semantic_score"] = norm_sem
                chunk["bm25_score"] = norm_bm

                fused_candidates.append(chunk)

            # Sort and return top k
            fused_candidates.sort(key=lambda x: x["fused_score"], reverse=True)
            return fused_candidates[:k]
        finally:
            tracker.add_retrieval_latency(time.time() - t0)

    def get_chunk_count(self) -> int:
        return len(self.chunks)

    def get_documents_metadata(self) -> list[dict]:
        docs = {}
        for c in self.chunks:
            fn = c["filename"]
            if fn not in docs:
                docs[fn] = {
                    "filename": fn,
                    "upload_date": c["upload_date"],
                    "document_type": c["document_type"],
                    "pdf_type": c.get("pdf_type", "Digital PDF"),
                    "chunks": 0,
                    "characters": 0,
                }
            docs[fn]["chunks"] += 1
            docs[fn]["characters"] += len(c["text"])
        return list(docs.values())

    def clear(self) -> None:
        self.chunks = []
        self.faiss_index = None
        self.bm25 = None
        if self.store_path.exists():
            self.store_path.unlink()
        if self.faiss_path.exists():
            self.faiss_path.unlink()
