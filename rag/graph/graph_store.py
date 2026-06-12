import json
import re
from pathlib import Path
import networkx as nx
from networkx.readwrite import json_graph


class GraphStore:
    def __init__(self, persist_dir: str = "./rag_storage"):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.store_path = self.persist_dir / "graph.json"
        self.graph = nx.DiGraph()
        self.load()

    def load(self) -> None:
        if self.store_path.exists():
            try:
                with open(self.store_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.graph = json_graph.node_link_graph(data)
            except Exception as e:
                print(f"Error loading graph.json: {e}")
                self.graph = nx.DiGraph()
        else:
            self.graph = nx.DiGraph()

    def save(self) -> None:
        try:
            data = json_graph.node_link_data(self.graph)
            with open(self.store_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving graph.json: {e}")

    def get_canonical_name(self, name: str, entity_type: str = "Unknown") -> str:
        name_clean = name.strip()
        if not name_clean:
            return ""

        name_lower = name_clean.lower()
        words_new = set(w.lower() for w in re.findall(r'\b\w+\b', name_clean))
        if not words_new:
            return name_clean

        for existing_node, attrs in list(self.graph.nodes(data=True)):
            existing_type = attrs.get("entity_type", "Unknown")
            
            # Check type compatibility: same type, or one of them is "Unknown"
            types_compatible = (
                entity_type == "Unknown" or 
                existing_type == "Unknown" or 
                entity_type.lower() == existing_type.lower()
            )
            if not types_compatible:
                continue

            # Case 1: Exact case-insensitive match
            if existing_node.lower() == name_lower:
                return existing_node

            # Case 2: Check aliases of existing node
            aliases = attrs.get("aliases", [])
            if any(a.lower() == name_lower for a in aliases):
                return existing_node

            # Case 3: Word subset match (excluding purely generic/type terms to be safe)
            words_existing = set(w.lower() for w in re.findall(r'\b\w+\b', existing_node))
            if words_existing and (words_new.issubset(words_existing) or words_existing.issubset(words_new)):
                # The longer one is the canonical name!
                if len(existing_node) >= len(name_clean):
                    return existing_node
                else:
                    # New name is longer/more specific -> rename existing to new
                    self._rename_node(existing_node, name_clean)
                    return name_clean

        return name_clean

    def _rename_node(self, old_name: str, new_name: str) -> None:
        if old_name == new_name or not self.graph.has_node(old_name):
            return
        
        attrs = dict(self.graph.nodes[old_name])
        
        if self.graph.has_node(new_name):
            new_attrs = self.graph.nodes[new_name]
            # Merge sources
            sources = attrs.get("sources", [])
            new_sources = new_attrs.get("sources", [])
            for s in sources:
                if not any(ns.get("chunk_ref") == s.get("chunk_ref") for ns in new_sources):
                    new_sources.append(s)
            new_attrs["sources"] = new_sources
            new_attrs["confidence"] = max(new_attrs.get("confidence", 0.0), attrs.get("confidence", 0.0))
            
            # Merge aliases
            aliases = set(new_attrs.get("aliases", []))
            aliases.update(attrs.get("aliases", []))
            aliases.add(old_name)
            new_attrs["aliases"] = list(aliases)
            
            if new_attrs.get("entity_type") == "Unknown" and attrs.get("entity_type") != "Unknown":
                new_attrs["entity_type"] = attrs["entity_type"]
        else:
            self.graph.add_node(new_name, **attrs)
            aliases = set(attrs.get("aliases", []))
            aliases.add(old_name)
            self.graph.nodes[new_name]["aliases"] = list(aliases)
            self.graph.nodes[new_name]["canonical_name"] = new_name
            self.graph.nodes[new_name]["display_name"] = new_name

        # Move edges
        in_edges = list(self.graph.in_edges(old_name, data=True))
        for u, _, edge_data in in_edges:
            u_new = new_name if u == old_name else u
            if self.graph.has_edge(u_new, new_name):
                existing_edge_data = self.graph.edges[u_new, new_name]
                for s in edge_data.get("sources", []):
                    if not any(es.get("chunk_ref") == s.get("chunk_ref") for es in existing_edge_data.get("sources", [])):
                        existing_edge_data.setdefault("sources", []).append(s)
                existing_edge_data["confidence"] = max(existing_edge_data.get("confidence", 0.0), edge_data.get("confidence", 0.0))
            else:
                self.graph.add_edge(u_new, new_name, **edge_data)

        out_edges = list(self.graph.out_edges(old_name, data=True))
        for _, v, edge_data in out_edges:
            v_new = new_name if v == old_name else v
            if self.graph.has_edge(new_name, v_new):
                existing_edge_data = self.graph.edges[new_name, v_new]
                for s in edge_data.get("sources", []):
                    if not any(es.get("chunk_ref") == s.get("chunk_ref") for es in existing_edge_data.get("sources", [])):
                        existing_edge_data.setdefault("sources", []).append(s)
                existing_edge_data["confidence"] = max(existing_edge_data.get("confidence", 0.0), edge_data.get("confidence", 0.0))
            else:
                self.graph.add_edge(new_name, v_new, **edge_data)

        self.graph.remove_node(old_name)

    def add_entity(self, name: str, entity_type: str, source_metadata: dict, confidence: float = 1.0) -> None:
        name = name.strip()
        if not name:
            return

        canonical_name = self.get_canonical_name(name, entity_type)

        if self.graph.has_node(canonical_name):
            node_attrs = self.graph.nodes[canonical_name]
            
            # Prefer explicit types over Unknown
            if node_attrs.get("entity_type") == "Unknown" and entity_type != "Unknown":
                node_attrs["entity_type"] = entity_type

            # Append source reference if not already tracked
            sources = node_attrs.get("sources", [])
            ref = source_metadata.get("chunk_ref")
            if not any(s.get("chunk_ref") == ref for s in sources):
                sources.append(source_metadata)
                node_attrs["sources"] = sources
            
            # Add name to aliases if it differs
            aliases = set(node_attrs.get("aliases", []))
            if name.lower() != canonical_name.lower():
                aliases.add(name)
            node_attrs["aliases"] = list(aliases)
            
            # Update to maximum confidence
            node_attrs["confidence"] = max(node_attrs.get("confidence", 0.0), confidence)
        else:
            self.graph.add_node(
                canonical_name,
                entity_type=entity_type,
                canonical_name=canonical_name,
                display_name=canonical_name,
                aliases=[name] if name.lower() != canonical_name.lower() else [],
                sources=[source_metadata],
                confidence=confidence
            )

    def add_relationship(self, source: str, target: str, relation_type: str, source_metadata: dict, confidence: float = 1.0) -> None:
        source = source.strip()
        target = target.strip()
        if not source or not target:
            return

        # Resolve canonical names for endpoints
        source_canonical = self.get_canonical_name(source, "Unknown")
        target_canonical = self.get_canonical_name(target, "Unknown")

        # Ensure nodes exist
        if not self.graph.has_node(source_canonical):
            self.add_entity(source_canonical, "Unknown", source_metadata, confidence=confidence)
        if not self.graph.has_node(target_canonical):
            self.add_entity(target_canonical, "Unknown", source_metadata, confidence=confidence)

        if self.graph.has_edge(source_canonical, target_canonical):
            edge_attrs = self.graph.edges[source_canonical, target_canonical]
            sources = edge_attrs.get("sources", [])
            ref = source_metadata.get("chunk_ref")
            if not any(s.get("chunk_ref") == ref for s in sources):
                sources.append(source_metadata)
                self.graph.edges[source_canonical, target_canonical]["sources"] = sources
            
            # Update to maximum confidence
            self.graph.edges[source_canonical, target_canonical]["confidence"] = max(edge_attrs.get("confidence", 0.0), confidence)
        else:
            self.graph.add_edge(
                source_canonical,
                target_canonical,
                relation_type=relation_type,
                sources=[source_metadata],
                confidence=confidence
            )

    def delete_document_references(self, filename: str) -> None:
        # 1. Clean edges
        edges_to_remove = []
        for u, v, d in self.graph.edges(data=True):
            sources = d.get("sources", [])
            cleaned_sources = [s for s in sources if s.get("source_document") != filename]
            if not cleaned_sources:
                edges_to_remove.append((u, v))
            else:
                self.graph.edges[u, v]["sources"] = cleaned_sources

        for u, v in edges_to_remove:
            self.graph.remove_edge(u, v)

        # 2. Clean nodes
        nodes_to_remove = []
        for node, d in self.graph.nodes(data=True):
            sources = d.get("sources", [])
            cleaned_sources = [s for s in sources if s.get("source_document") != filename]
            if not cleaned_sources:
                nodes_to_remove.append(node)
            else:
                self.graph.nodes[node]["sources"] = cleaned_sources

        for node in nodes_to_remove:
            self.graph.remove_node(node)

        # 3. Clean up orphans (nodes with no edges AND type "Unknown" or no sources)
        orphans = [
            n for n in self.graph.nodes()
            if self.graph.degree(n) == 0 and (
                self.graph.nodes[n].get("entity_type") == "Unknown"
                or not self.graph.nodes[n].get("sources")
            )
        ]
        for node in orphans:
            self.graph.remove_node(node)

        self.save()

    def get_nodes(self) -> list[dict]:
        nodes = []
        for node, attrs in self.graph.nodes(data=True):
            nodes.append({
                "id": node,
                "canonical_name": attrs.get("canonical_name", node),
                "display_name": attrs.get("display_name", node),
                "aliases": attrs.get("aliases", []),
                "entity_type": attrs.get("entity_type", "Unknown"),
                "sources": attrs.get("sources", []),
                "confidence": attrs.get("confidence", 1.0),
            })
        return nodes

    def get_edges(self) -> list[dict]:
        edges = []
        for u, v, attrs in self.graph.edges(data=True):
            edges.append({
                "source": u,
                "target": v,
                "relation_type": attrs.get("relation_type", "associated_with"),
                "sources": attrs.get("sources", []),
                "confidence": attrs.get("confidence", 1.0),
            })
        return edges

    def clear(self) -> None:
        self.graph = nx.DiGraph()
        if self.store_path.exists():
            self.store_path.unlink(missing_ok=True)
