import os
import unittest
import tempfile
import json
from pathlib import Path
from rag.config.config import Settings


class TestSettings(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.persist_dir = self.temp_dir.name

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_default_values(self):
        settings = Settings(persist_dir=self.persist_dir)
        self.assertEqual(settings.llm_model, "deepseek-chat")
        self.assertEqual(settings.parent_size, 1500)
        self.assertEqual(settings.parent_overlap, 200)

    def test_env_overrides(self):
        os.environ["RAG_LLM_MODEL"] = "gpt-4o"
        os.environ["RAG_PARENT_SIZE"] = "2000"
        os.environ["RAG_GRAPH_ENABLED"] = "False"
        
        try:
            settings = Settings(persist_dir=self.persist_dir)
            self.assertEqual(settings.llm_model, "gpt-4o")
            self.assertEqual(settings.parent_size, 2000)
            self.assertFalse(settings.graph_enabled)
        finally:
            del os.environ["RAG_LLM_MODEL"]
            del os.environ["RAG_PARENT_SIZE"]
            del os.environ["RAG_GRAPH_ENABLED"]

    def test_save_and_load(self):
        settings = Settings(persist_dir=self.persist_dir)
        settings.llm_model = "custom-model"
        settings.parent_size = 999
        settings.save()

        # Load again in a new Settings instance
        new_settings = Settings(persist_dir=self.persist_dir)
        self.assertEqual(new_settings.llm_model, "custom-model")
        self.assertEqual(new_settings.parent_size, 999)


if __name__ == "__main__":
    unittest.main()
