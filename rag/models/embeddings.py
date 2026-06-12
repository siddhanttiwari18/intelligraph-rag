import time
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from rag.analytics.tracker import tracker


class LocalEmbeddings(Embeddings):
    """Lightweight local embeddings via SentenceTransformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        # Load the model locally
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        t0 = time.time()
        embeddings = self.model.encode(texts, show_progress_bar=False)
        tracker.add_embedding_time(time.time() - t0)
        tokens = sum(len(text) for text in texts) // 4
        tracker.add_embedding_tokens(tokens)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        t0 = time.time()
        embedding = self.model.encode(text, show_progress_bar=False)
        tracker.add_embedding_time(time.time() - t0)
        tokens = len(text) // 4
        tracker.add_embedding_tokens(tokens)
        return embedding.tolist()
