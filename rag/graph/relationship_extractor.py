import json
import re

RELATIONSHIP_TRIGGERS = [
    "owned by", "connects to", "depends on", "uses", "works on",
    "supports", "affects", "impacted", "used by", "associated with",
    "works at", "employed by", "built", "created", "developed", "implements", "implement",
    "disrupts", "disrupted", "impacts", "connect to", "trained on", "train on", "evaluates on",
    "dataset", "data set"
]

class RelationshipExtractor:
    def __init__(self):
        # Local regex patterns for direct sentences (ordered from most specific to most generic)
        self.rules = [
            # E.g. "HTTPS uses Port 443" or "HTTP listens on Port 80"
            (re.compile(r'\b([A-Z][a-zA-Z0-9_\-]+(?:\s+[A-Z][a-zA-Z0-9_\-]+)*)\s+(?:uses|use|listens\s+on)\s+(?:port\s+)?(\d+)\b', re.IGNORECASE), "uses", "Call Type", "Port"),
            # E.g. "PaperX uses ImageNet dataset"
            (re.compile(r'\b([A-Z][a-zA-Z0-9_\-]+\s+Paper|Paper\s+[A-Z][a-zA-Z0-9_\-]+)\s+(?:uses|evaluates\s+on)\s+([A-Z][a-zA-Z0-9_\-]+(?:\s+[A-Z][a-zA-Z0-9_\-]+)*)\b', re.IGNORECASE), "uses", "Research Paper", "Dataset"),
            # E.g. "ModelA is trained on DatasetB"
            (re.compile(r'\b([A-Z][a-zA-Z0-9_\-]+(?:\s+[A-Z][a-zA-Z0-9_\-]+)*)\s+(?:is\s+)?trained\s+on\s+([A-Z][a-zA-Z0-9_\-]+(?:\s+[A-Z][a-zA-Z0-9_\-]+)*)\b', re.IGNORECASE), "trained_on", "Model", "Dataset"),
            # E.g. "Person X works at Org Y"
            (re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:works\s+at|is\s+employed\s+by)\s+([A-Z][a-zA-Z0-9_\-]+(?:\s+[A-Z][a-zA-Z0-9_\-]+)*)\b', re.IGNORECASE), "works_at", "Person", "Organization"),
            # E.g. "Person X built Project Y"
            (re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:built|created|developed)\s+([A-Z][a-zA-Z0-9_\-]+(?:\s+[A-Z][a-zA-Z0-9_\-]+)*)\b', re.IGNORECASE), "built", "Person", "Project"),
            # E.g. "Service X is owned by Team A"
            (re.compile(r'\b([A-Z][a-zA-Z0-9_\-]+(?:\s+[A-Z][a-zA-Z0-9_\-]+)*)\s+(?:is\s+)?owned\s+by\s+([A-Z][a-zA-Z0-9_\-]+(?:\s+[A-Z][a-zA-Z0-9_\-]+)*)\b', re.IGNORECASE), "owned_by", "Service", "Team"),
            # E.g. "Incident X affects Service Y"
            (re.compile(r'\b([Ii]ncident\s+[A-Z0-9\-]+)\s+(?:affects|impacts|disrupts)\s+([A-Z][a-zA-Z0-9_\-]+(?:\s+[A-Z][a-zA-Z0-9_\-]+)*)\b', re.IGNORECASE), "affects", "Incident", "Service"),
            # E.g. "Application Y connects to Database Z"
            (re.compile(r'\b([A-Z][a-zA-Z0-9_\-]+(?:\s+[A-Z][a-zA-Z0-9_\-]+)*)\s+(?:connects\s+to|connect\s+to)\s+([A-Z][a-zA-Z0-9_\-]+(?:\s+[A-Z][a-zA-Z0-9_\-]+)*)\b', re.IGNORECASE), "connects_to", "Application", "Database"),
            # E.g. "Application A implements Protocol B"
            (re.compile(r'\b([A-Z][a-zA-Z0-9_\-]+(?:\s+[A-Z][a-zA-Z0-9_\-]+)*)\s+(?:uses|use|implements|implement)\s+([A-Z][a-zA-Z0-9_\-]+(?:\s+[A-Z][a-zA-Z0-9_\-]+)*)\b', re.IGNORECASE), "uses", "Application", "Protocol"),
            # E.g. "Project A uses Technology B"
            (re.compile(r'\b([A-Z][a-zA-Z0-9_\-]+(?:\s+[A-Z][a-zA-Z0-9_\-]+)*)\s+(?:uses|use|utilizes|utilize)\s+([A-Z][a-zA-Z0-9_\-]+(?:\s+[A-Z][a-zA-Z0-9_\-]+)*)\b', re.IGNORECASE), "uses", "Project", "Technology"),
        ]

    def extract_relationships(self, text: str, pipeline) -> list[dict]:
        # 1. Cost-efficient trigger check
        # If no keywords are present, bypass extraction to save token cost and computation
        if not any(trigger in text.lower() for trigger in RELATIONSHIP_TRIGGERS):
            return []

        # Setup entity validation helper
        if hasattr(pipeline, "entity_extractor") and pipeline.entity_extractor is not None:
            is_valid = pipeline.entity_extractor.is_valid_entity
        else:
            from rag.graph.entity_extractor import EntityExtractor
            is_valid = EntityExtractor().is_valid_entity

        relationships = []
        seen = set()

        # 2. Local rule-based extraction
        for pattern, rel_type, src_type, tgt_type in self.rules:
            matches = pattern.findall(text)
            for match in matches:
                src, tgt = match[0].strip(), match[1].strip()
                if src and tgt:
                    # Validate endpoints
                    if not is_valid(src) or not is_valid(tgt):
                        continue

                    key = (src.lower(), tgt.lower(), rel_type)
                    if key not in seen:
                        seen.add(key)
                        relationships.append({
                            "source": src,
                            "source_type": src_type,
                            "target": tgt,
                            "target_type": tgt_type,
                            "relation_type": rel_type,
                            "confidence": 1.0  # Rule-based extractions have high confidence
                        })

        # 3. Fallback to LLM for complex texts where rule-based is empty
        # We only call the LLM for chunks that have triggers but no clean regex matches.
        if not relationships:
            try:
                if hasattr(pipeline, "llm") and pipeline.llm is not None:
                    messages = [
                        {
                            "role": "system",
                            "content": (
                                "You are a graph relationship extractor. Extract entities and direct relationships from the provided text chunk.\n"
                                "Focus ONLY on relationships between: Person, Team, Organization, Project, Service, Application, Technology, API, Database, Protocol, Incident, Location, Port, Dataset, Model, Research Paper.\n"
                                "Valid relations: works_on, uses, connects_to, owned_by, affects, used_by, supports, works_at, built, trained_on.\n"
                                "Respond strictly with a JSON array of objects. Each relationship MUST have a confidence score between 0.0 and 1.0 based on clarity of text evidence. "
                                "Format the response in this structure:\n"
                                "[\n"
                                "  {\n"
                                "    \"source\": \"entity name\",\n"
                                "    \"source_type\": \"entity type\",\n"
                                "    \"target\": \"entity name\",\n"
                                "    \"target_type\": \"entity type\",\n"
                                "    \"relation_type\": \"works_on/uses/connects_to/owned_by/affects/used_by/supports/works_at/built/trained_on\",\n"
                                "    \"confidence\": 0.85\n"
                                "  }\n"
                                "]"
                            ),
                        },
                        {"role": "user", "content": f"Text:\n{text}"},
                    ]
                    response = pipeline.llm.invoke(messages)
                    res_text = response.content.strip()
                    # Clean JSON output
                    if "[" in res_text and "]" in res_text:
                        res_text = res_text[res_text.find("["):res_text.rfind("]")+1]
                    res_list = json.loads(res_text)
                    for r in res_list:
                        src = r.get("source", "").strip()
                        tgt = r.get("target", "").strip()
                        rel = r.get("relation_type", "associated_with").strip()
                        conf = r.get("confidence", 0.8)
                        try:
                            conf = float(conf)
                        except Exception:
                            conf = 0.8

                        if src and tgt:
                            # Validate endpoints
                            if not is_valid(src) or not is_valid(tgt):
                                continue

                            key = (src.lower(), tgt.lower(), rel)
                            if key not in seen:
                                seen.add(key)
                                relationships.append({
                                    "source": src,
                                    "source_type": r.get("source_type", "Unknown"),
                                    "target": tgt,
                                    "target_type": r.get("target_type", "Unknown"),
                                    "relation_type": rel,
                                    "confidence": conf
                                })
            except Exception as e:
                print(f"LLM relationship extraction failed: {e}")

        return relationships
