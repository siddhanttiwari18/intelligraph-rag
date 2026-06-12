import threading

class PipelineTracker:
    def __init__(self):
        self.local = threading.local()

    def start_track(self):
        self.local.embedding_time = 0.0
        self.local.re_ranking_time = 0.0
        self.local.retrieval_latency = 0.0
        self.local.graph_retrieval_latency = 0.0
        self.local.graph_queries_count = 0
        self.local.input_tokens = 0
        self.local.output_tokens = 0
        self.local.cached_input_tokens = 0
        self.local.embedding_tokens = 0

    def add_embedding_time(self, t: float):
        if hasattr(self.local, "embedding_time"):
            self.local.embedding_time += t

    def add_re_ranking_time(self, t: float):
        if hasattr(self.local, "re_ranking_time"):
            self.local.re_ranking_time += t

    def add_retrieval_latency(self, t: float):
        if hasattr(self.local, "retrieval_latency"):
            self.local.retrieval_latency += t

    def add_graph_retrieval_latency(self, t: float):
        if hasattr(self.local, "graph_retrieval_latency"):
            self.local.graph_retrieval_latency += t

    def inc_graph_queries(self):
        if hasattr(self.local, "graph_queries_count"):
            self.local.graph_queries_count += 1

    def add_llm_tokens(self, input_tokens: int, output_tokens: int, cached_input_tokens: int = 0):
        if hasattr(self.local, "input_tokens"):
            self.local.input_tokens += input_tokens
        if hasattr(self.local, "output_tokens"):
            self.local.output_tokens += output_tokens
        if hasattr(self.local, "cached_input_tokens"):
            self.local.cached_input_tokens += cached_input_tokens

    def add_embedding_tokens(self, tokens: int):
        if hasattr(self.local, "embedding_tokens"):
            self.local.embedding_tokens += tokens

    def get_track(self) -> dict:
        return {
            "embedding_time": getattr(self.local, "embedding_time", 0.0),
            "re_ranking_time": getattr(self.local, "re_ranking_time", 0.0),
            "retrieval_latency": getattr(self.local, "retrieval_latency", 0.0),
            "graph_retrieval_latency": getattr(self.local, "graph_retrieval_latency", 0.0),
            "graph_queries_count": getattr(self.local, "graph_queries_count", 0),
            "input_tokens": getattr(self.local, "input_tokens", 0),
            "output_tokens": getattr(self.local, "output_tokens", 0),
            "cached_input_tokens": getattr(self.local, "cached_input_tokens", 0),
            "embedding_tokens": getattr(self.local, "embedding_tokens", 0),
        }

tracker = PipelineTracker()
