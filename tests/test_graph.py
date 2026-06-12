import unittest
import tempfile
from pathlib import Path
from rag.graph.graph_store import GraphStore


class TestGraphStore(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.graph_store = GraphStore(persist_dir=self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_canonical_name_resolution(self):
        # Exact match
        name1 = "Project Atlas"
        self.assertEqual(self.graph_store.get_canonical_name(name1, "Project"), "Project Atlas")

        # Set up a canonical node in the graph
        meta = {"source_document": "doc1.txt", "page_number": 1, "chunk_ref": "c1"}
        self.graph_store.add_entity("Project Atlas", "Project", meta)
        
        # Word subset/alias match should resolve to canonical
        self.assertEqual(self.graph_store.get_canonical_name("Atlas", "Project"), "Project Atlas")
        self.assertEqual(self.graph_store.get_canonical_name("Project Atlas", "Project"), "Project Atlas")

    def test_add_entity_and_relationship(self):
        meta = {"source_document": "doc1.txt", "page_number": 1, "chunk_ref": "c1"}
        self.graph_store.add_entity("Alice Smith", "Person", meta)
        self.graph_store.add_entity("Team Alpha", "Team", meta)
        
        self.graph_store.add_relationship("Alice Smith", "Team Alpha", "works_on", meta)
        
        nodes = self.graph_store.get_nodes()
        edges = self.graph_store.get_edges()
        
        self.assertEqual(len(nodes), 2)
        self.assertEqual(len(edges), 1)
        
        # Check source document traceability
        self.assertEqual(edges[0]["source"], "Alice Smith")
        self.assertEqual(edges[0]["target"], "Team Alpha")
        self.assertEqual(edges[0]["relation_type"], "works_on")
        self.assertEqual(edges[0]["sources"][0]["source_document"], "doc1.txt")


if __name__ == "__main__":
    unittest.main()
