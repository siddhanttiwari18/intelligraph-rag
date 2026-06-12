import os
import json
from pathlib import Path


class Settings:
    def __init__(self, persist_dir: str = "./rag_storage"):
        self.persist_dir = Path(persist_dir)
        self.config_path = self.persist_dir / "config.json"
        
        # 1. Default Configurations
        self.llm_model = "deepseek-chat"
        self.embed_model = "all-MiniLM-L6-v2"
        self.parent_size = 1500
        self.parent_overlap = 200
        self.child_size = 400
        self.child_overlap = 50
        self.retrieve_k = 20
        self.rerank_top_n = 4
        
        # Agent parameters
        self.confidence_threshold = 0.6
        self.max_retrieval_iterations = 2
        self.max_planner_depth = 3
        self.max_retrieved_chunks = 20
        self.agent_trace_visibility = True
        
        # Graph parameters
        self.max_depth = 2
        self.graph_enabled = True
        
        # Caching parameters
        self.cache_ttl = 300.0
        self.cache_max_size = 128
        
        # Initial load and overrides
        self.load()

    def load(self) -> None:
        """Loads configuration from JSON file first, then applies environment overrides."""
        # 1. Load from local configuration file if present
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    stored = json.load(f)
                    for k, v in stored.items():
                        if hasattr(self, k):
                            setattr(self, k, v)
            except Exception as e:
                print(f"Warning: Failed to load local config.json: {e}")

        # 2. Load and override from environment variables (Prefix: RAG_)
        for key in list(self.__dict__.keys()):
            if key in ("persist_dir", "config_path"):
                continue
            env_key = f"RAG_{key.upper()}"
            env_val = os.getenv(env_key)
            if env_val is not None:
                # Convert type based on default attribute type
                default_val = getattr(self, key)
                try:
                    if isinstance(default_val, bool):
                        setattr(self, key, env_val.lower() in ("true", "1", "yes"))
                    elif isinstance(default_val, int):
                        setattr(self, key, int(env_val))
                    elif isinstance(default_val, float):
                        setattr(self, key, float(env_val))
                    else:
                        setattr(self, key, env_val)
                except Exception as e:
                    print(f"Warning: Failed to parse env override {env_key}={env_val}: {e}")

    def save(self) -> None:
        """Saves current memory settings back to local JSON config file."""
        try:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            data = {}
            for k, v in self.__dict__.items():
                if k in ("persist_dir", "config_path"):
                    continue
                data[k] = v
                
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save config to local file: {e}")

    def update(self, config_dict: dict) -> None:
        """Updates internal configuration from a dictionary and persists to file."""
        for k, v in config_dict.items():
            if hasattr(self, k) and k not in ("persist_dir", "config_path"):
                setattr(self, k, v)
        self.save()
