import time
from rag.analytics.tracker import tracker

class GraphRetriever:
    def __init__(self, graph_store, max_depth: int = 2, max_relations: int = 10):
        self.graph_store = graph_store
        self.max_depth = max_depth
        self.max_relations = max_relations
        # Simple in-memory cache for lookups
        self.cache = {}

    def clear_cache(self) -> None:
        self.cache.clear()

    def detect_entities(self, query: str) -> list[str]:
        # Perform case-insensitive search for entity nodes or their aliases appearing in the query text
        query_lower = query.lower()
        matched_nodes = []
        
        # Build candidate matches (string to match, canonical node name)
        candidates = []
        for node, attrs in self.graph_store.graph.nodes(data=True):
            candidates.append((node, node))
            for alias in attrs.get("aliases", []):
                candidates.append((alias, node))
                
        # Sort candidates by matching name length descending so we match longer names first
        candidates.sort(key=lambda x: len(x[0]), reverse=True)
        
        for match_name, canonical_node in candidates:
            if len(match_name) > 2 and match_name.lower() in query_lower:
                if canonical_node not in matched_nodes:
                    matched_nodes.append(canonical_node)
                # Remove matched substring from query to avoid double matching smaller parts
                query_lower = query_lower.replace(match_name.lower(), "")
                
        return matched_nodes

    def retrieve(self, query: str, max_depth: int | None = None) -> dict:
        t0 = time.time()
        tracker.inc_graph_queries()
        try:
            depth = max_depth if max_depth is not None else self.max_depth
            cache_key = (query, depth)
            
            # 1. Cache lookup
            if cache_key in self.cache:
                return self.cache[cache_key]

            # 2. Match entities in query
            matched_nodes = self.detect_entities(query)
            if not matched_nodes:
                return {"context": "", "sources": [], "paths": []}

            # 3. Traverse paths using BFS
            visited = set()
            queue = [(node, 0) for node in matched_nodes]
            paths = []
            supporting_sources = []

            while queue:
                curr, curr_depth = queue.pop(0)
                if curr in visited or curr_depth >= depth:
                    continue
                visited.add(curr)

                # Get neighbors (both incoming and outgoing to support bidirectionality)
                successors = list(self.graph_store.graph.successors(curr))
                predecessors = list(self.graph_store.graph.predecessors(curr))
                
                # Combine neighbors and cap per-node relation traversal count to prevent explosion
                neighbors = [(n, "forward") for n in successors] + [(n, "reverse") for n in predecessors]
                neighbors = neighbors[:self.max_relations]

                for neighbor, direction in neighbors:
                    if neighbor not in visited:
                        if direction == "forward":
                            attrs = self.graph_store.graph.edges[curr, neighbor]
                            rel_type = attrs.get("relation_type", "associated_with")
                            sources = attrs.get("sources", [])
                            source_node, target_node = curr, neighbor
                        else:
                            attrs = self.graph_store.graph.edges[neighbor, curr]
                            rel_type = attrs.get("relation_type", "associated_with")
                            sources = attrs.get("sources", [])
                            source_node, target_node = neighbor, curr

                        paths.append({
                            "source": source_node,
                            "target": target_node,
                            "relation_type": rel_type,
                            "sources": sources
                        })
                        
                        for s in sources:
                            supporting_sources.append(s)

                        queue.append((neighbor, curr_depth + 1))

            # 4. Format textual context representing paths
            context_lines = []
            for p in paths:
                src = p["source"]
                tgt = p["target"]
                rel = p["relation_type"]
                src_type = self.graph_store.graph.nodes[src].get("entity_type", "Unknown")
                tgt_type = self.graph_store.graph.nodes[tgt].get("entity_type", "Unknown")
                
                evidences = []
                for s in p["sources"]:
                    evidences.append(f"{s['source_document']} (Page {s['page_number']})")
                    
                context_lines.append(
                    f"- [{src_type}] '{src}' is connected to [{tgt_type}] '{tgt}' via relation '{rel}' "
                    f"(Evidence: {', '.join(evidences)})"
                )

            context_text = ""
            if context_lines:
                context_text = "Knowledge Graph Connections:\n" + "\n".join(context_lines)

            result = {
                "context": context_text,
                "sources": supporting_sources,
                "paths": paths
            }
            
            # 5. Populate cache
            self.cache[cache_key] = result
            return result
        finally:
            tracker.add_graph_retrieval_latency(time.time() - t0)
