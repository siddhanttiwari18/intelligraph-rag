import time
import unittest
from rag.services.cache import RAGCache


class TestRAGCache(unittest.TestCase):
    def test_cache_set_get(self):
        cache = RAGCache(default_ttl=60)
        cache.set("retrieval", "key1", "value1")
        self.assertEqual(cache.get("retrieval", "key1"), "value1")

    def test_cache_expiry(self):
        # Cache with a very short TTL of 0.1 seconds
        cache = RAGCache(default_ttl=0.1)
        cache.set("retrieval", "key1", "value1")
        
        # Immediate get should work
        self.assertEqual(cache.get("retrieval", "key1"), "value1")
        
        # Wait for expiration
        time.sleep(0.2)
        self.assertIsNone(cache.get("retrieval", "key1"))

    def test_cache_invalidation_by_type(self):
        cache = RAGCache()
        cache.set("retrieval", "k1", "v1")
        cache.set("query_expansion", "k2", "v2")
        
        cache.invalidate(cache_type="retrieval")
        self.assertIsNone(cache.get("retrieval", "k1"))
        self.assertEqual(cache.get("query_expansion", "k2"), "v2")

    def test_cache_global_invalidation(self):
        cache = RAGCache()
        cache.set("retrieval", "k1", "v1")
        cache.set("query_expansion", "k2", "v2")
        
        cache.clear()
        self.assertIsNone(cache.get("retrieval", "k1"))
        self.assertIsNone(cache.get("query_expansion", "k2"))


if __name__ == "__main__":
    unittest.main()
