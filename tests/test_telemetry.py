import unittest
import tempfile
import threading
import time
from pathlib import Path

from rag.analytics.telemetry import TelemetryService
from rag.analytics.tracker import tracker


class TestTelemetryAndTracker(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.telemetry = TelemetryService(persist_dir=self.test_dir.name)

    def tearDown(self):
        self.test_dir.cleanup()

    def test_pricing_presets_loading(self):
        # By default, DeepSeek Chat should be loaded
        self.assertEqual(self.telemetry.active_pricing["provider_name"], "DeepSeek Chat")
        self.assertEqual(self.telemetry.active_pricing["input_rate"], 0.27)
        self.assertEqual(self.telemetry.active_pricing["output_rate"], 1.10)

    def test_save_pricing_config(self):
        # Test changing LLM rates to GPT-4o preset
        rates = {
            "input_rate": 5.00,
            "output_rate": 15.00,
            "cached_input_rate": 2.50,
            "embedding_rate": 0.05
        }
        self.telemetry.save_pricing_config("GPT-4o", rates)
        
        # Verify changes in memory
        self.assertEqual(self.telemetry.active_pricing["provider_name"], "GPT-4o")
        self.assertEqual(self.telemetry.active_pricing["input_rate"], 5.00)
        self.assertEqual(self.telemetry.active_pricing["embedding_rate"], 0.05)
        
        # Verify persistence by loading a fresh service instance
        new_service = TelemetryService(persist_dir=self.test_dir.name)
        self.assertEqual(new_service.active_pricing["provider_name"], "GPT-4o")
        self.assertEqual(new_service.active_pricing["input_rate"], 5.00)

    def test_record_query_pricing_math(self):
        # Setup active pricing
        rates = {
            "input_rate": 1.00,        # $1 per 1M tokens ($0.000001 per token)
            "output_rate": 2.00,       # $2 per 1M tokens ($0.000002 per token)
            "cached_input_rate": 0.50, # $0.50 per 1M tokens
            "embedding_rate": 0.10     # $0.10 per 1M tokens
        }
        self.telemetry.save_pricing_config("Custom", rates)

        # Record a query with specific token splits:
        # 10,000 input tokens (4,000 of which are cached), 5,000 output tokens, 20,000 embedding tokens
        self.telemetry.record_query(
            query="Test query",
            classification="Semantic",
            query_type="Simple",
            success=True,
            total_response_time=1.0,
            retrieval_latency=0.2,
            graph_retrieval_latency=0.0,
            re_ranking_time=0.1,
            embedding_time=0.1,
            chunks_retrieved=3,
            chunks_sent_to_llm=3,
            context_size_chars=4000,
            token_estimate=3750,
            sub_questions_count=0,
            retrieval_iterations=1,
            graph_queries_count=0,
            matched_entities=[],
            accessed_documents=[],
            input_tokens=10000,
            output_tokens=5000,
            cached_input_tokens=4000,
            embedding_tokens=20000
        )

        metrics = self.telemetry.get_metrics()
        
        # Assert token counts
        self.assertEqual(metrics["total_input_tokens"], 10000)
        self.assertEqual(metrics["total_output_tokens"], 5000)
        self.assertEqual(metrics["total_cached_input_tokens"], 4000)
        self.assertEqual(metrics["total_embedding_tokens"], 20000)
        
        # Assert cost calculation logic:
        # Non-cached input = 10,000 - 4,000 = 6,000 tokens.
        # LLM Cost = (6,000 * $1/1M) + (4,000 * $0.50/1M) + (5,000 * $2/1M)
        #          = $0.006 + $0.002 + $0.010 = $0.0180
        # Embedding Cost = 20,000 * $0.10/1M = $0.0020
        # Total AI Cost = $0.0180 + $0.0020 = $0.0200
        self.assertAlmostEqual(metrics["total_llm_cost"], 0.0180, places=6)
        self.assertAlmostEqual(metrics["total_embedding_cost"], 0.0020, places=6)
        self.assertAlmostEqual(metrics["total_ai_cost"], 0.0200, places=6)
        self.assertAlmostEqual(metrics["daily_cost"], 0.0200, places=6)

    def test_legacy_logs_backward_compatibility(self):
        # Save a custom raw log with missing input_tokens and output_tokens (simulating legacy record)
        legacy_event = {
            "event_type": "query",
            "timestamp": time.time(),
            "data": {
                "query": "What is Kafka?",
                "classification": "Semantic",
                "query_type": "Simple",
                "success": True,
                "total_response_time": 1.2,
                "retrieval_latency": 0.2,
                "graph_retrieval_latency": 0.0,
                "re_ranking_time": 0.1,
                "embedding_time": 0.1,
                "chunks_retrieved=5": 5,
                "chunks_sent_to_llm": 4,
                "context_size_chars": 8000,
                "token_estimate": 2500, # Legacy token estimate
                "sub_questions_count": 0,
                "retrieval_iterations": 1,
                "graph_queries_count": 0,
                "matched_entities": [],
                "accessed_documents": []
            }
        }
        self.telemetry.events.append(legacy_event)

        # Retrieve metrics
        metrics = self.telemetry.get_metrics()
        
        # Verify fallback calculations:
        # Input tokens split: 80% of 2500 = 2000
        # Output tokens split: 20% of 2500 = 500
        # Embedding tokens: context_size_chars // 4 = 8000 // 4 = 2000
        self.assertEqual(metrics["total_input_tokens"], 2000)
        self.assertEqual(metrics["total_output_tokens"], 500)
        self.assertEqual(metrics["total_embedding_tokens"], 2000)

    def test_tracker_thread_safety(self):
        thread_errors = []

        def run_thread_1():
            try:
                tracker.start_track()
                tracker.add_embedding_time(0.5)
                tracker.add_retrieval_latency(1.0)
                tracker.add_llm_tokens(500, 100, 50)
                tracker.add_embedding_tokens(800)
                
                time.sleep(0.1)
                
                t_val = tracker.get_track()
                if t_val["input_tokens"] != 500:
                    thread_errors.append(f"Thread 1 input_tokens expected 500, got {t_val['input_tokens']}")
                if t_val["output_tokens"] != 100:
                    thread_errors.append(f"Thread 1 output_tokens expected 100, got {t_val['output_tokens']}")
                if t_val["cached_input_tokens"] != 50:
                    thread_errors.append(f"Thread 1 cached_input_tokens expected 50, got {t_val['cached_input_tokens']}")
                if t_val["embedding_tokens"] != 800:
                    thread_errors.append(f"Thread 1 embedding_tokens expected 800, got {t_val['embedding_tokens']}")
            except Exception as e:
                thread_errors.append(f"Thread 1 error: {e}")

        def run_thread_2():
            try:
                tracker.start_track()
                tracker.add_embedding_time(0.8)
                tracker.add_retrieval_latency(2.0)
                tracker.add_llm_tokens(1200, 300, 200)
                tracker.add_embedding_tokens(1500)
                
                time.sleep(0.1)
                
                t_val = tracker.get_track()
                if t_val["input_tokens"] != 1200:
                    thread_errors.append(f"Thread 2 input_tokens expected 1200, got {t_val['input_tokens']}")
                if t_val["output_tokens"] != 300:
                    thread_errors.append(f"Thread 2 output_tokens expected 300, got {t_val['output_tokens']}")
                if t_val["cached_input_tokens"] != 200:
                    thread_errors.append(f"Thread 2 cached_input_tokens expected 200, got {t_val['cached_input_tokens']}")
                if t_val["embedding_tokens"] != 1500:
                    thread_errors.append(f"Thread 2 embedding_tokens expected 1500, got {t_val['embedding_tokens']}")
            except Exception as e:
                thread_errors.append(f"Thread 2 error: {e}")

        t1 = threading.Thread(target=run_thread_1)
        t2 = threading.Thread(target=run_thread_2)

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertEqual(len(thread_errors), 0, "\n".join(thread_errors))


if __name__ == "__main__":
    unittest.main()
