import re

ENTITY_TYPES = [
    "Person", "Team", "Organization", "Project", "Service",
    "Application", "Technology", "API", "Database", "Protocol",
    "Incident", "Location"
]

# Quick static mappings for common technical entities
TECH_KEYWORDS = {
    "kafka": "Technology",
    "postgresql": "Database",
    "postgres": "Database",
    "mongodb": "Database",
    "mysql": "Database",
    "oracle": "Database",
    "redis": "Database",
    "sqlite": "Database",
    "docker": "Technology",
    "kubernetes": "Technology",
    "k8s": "Technology",
    "ipv6": "Protocol",
    "ipv4": "Protocol",
    "http": "Protocol",
    "https": "Protocol",
    "grpc": "Protocol",
    "rest": "Protocol",
    "graphql": "Protocol",
    "python": "Technology",
    "java": "Technology",
    "golang": "Technology",
    "react": "Technology",
    "aws": "Organization",
    "azure": "Organization",
    "gcp": "Organization",
}

class EntityExtractor:
    def __init__(self):
        # Setup regex patterns for dynamic detection
        self.patterns = {
            "Team": re.compile(r'\b([A-Z][a-zA-Z0-9_\-]+(?:\s+[A-Z][a-zA-Z0-9_\-]+)*\s+[Tt]eam)\b|\b([Tt]eam\s+[A-Z][a-zA-Z0-9_\-]+(?:\s+[A-Z][a-zA-Z0-9_\-]+)*)\b'),
            "Service": re.compile(r'\b([A-Z][a-zA-Z0-9_\-]+(?:\s+[A-Z][a-zA-Z0-9_\-]+)*\s+[Ss]ervice)\b|\b([A-Z][a-zA-Z0-9_\-]+(?:\s+[A-Z][a-zA-Z0-9_\-]+)*\s+[Mm]icroservice)\b'),
            "Application": re.compile(r'\b([A-Z][a-zA-Z0-9_\-]+(?:\s+[A-Z][a-zA-Z0-9_\-]+)*\s+[Aa]pp)\b|\b([A-Z][a-zA-Z0-9_\-]+(?:\s+[A-Z][a-zA-Z0-9_\-]+)*\s+[Aa]pplication)\b'),
            "Project": re.compile(r'\b([Pp]roject\s+[A-Z][a-zA-Z0-9_\-]+(?:\s+[A-Z][a-zA-Z0-9_\-]+)*)\b|\b([A-Z][a-zA-Z0-9_\-]+(?:\s+[A-Z][a-zA-Z0-9_\-]+)*\s+[Pp]roject)\b'),
            "Database": re.compile(r'\b([A-Z][a-zA-Z0-9_\-]+(?:\s+[A-Z][a-zA-Z0-9_\-]+)*\s+DB)\b|\b([A-Z][a-zA-Z0-9_\-]+(?:\s+[A-Z][a-zA-Z0-9_\-]+)*\s+[Dd]atabase)\b'),
            "Incident": re.compile(r'\b(Incident\s+[A-Z0-9\-]+)\b', re.IGNORECASE),
            "Person": re.compile(r'\b(?:Mr\.|Ms\.|Dr\.)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'),
        }

    def is_valid_entity(self, name: str) -> bool:
        name_clean = name.strip()
        if not name_clean:
            return False

        # Rule 1: Filter short words unless they are valid short technical terms or numeric (ports)
        if len(name_clean) < 4:
            if not name_clean.isdigit():
                if name_clean.lower() not in {"aws", "gcp", "api", "k8s", "db", "app", "cpu", "ram", "os", "ip", "ssl", "tls", "tcp", "udp", "ssh", "git", "npm", "pip", "mac", "win", "lnx"}:
                    return False

        # Rule 2: Filter stop words
        STOP_WORDS = {
            "the", "a", "an", "and", "or", "but", "if", "then", "else", "when", 
            "at", "by", "for", "with", "about", "against", "between", "into", 
            "through", "during", "before", "after", "above", "below", "to", 
            "from", "up", "down", "in", "out", "on", "off", "over", "under", 
            "again", "further", "then", "once", "here", "there", "when", 
            "where", "why", "how", "all", "any", "both", "each", "few", "more", 
            "most", "other", "some", "such", "no", "nor", "not", "only", "own", 
            "same", "so", "than", "too", "very", "s", "t", "can", "will", 
            "just", "don", "should", "now"
        }
        if name_clean.lower() in STOP_WORDS:
            return False

        # Rule 3: Filter generic terms
        GENERIC_TERMS = {
            "app", "application", "service", "services", "system", "systems",
            "database", "databases", "db", "project", "projects", "team", "teams",
            "person", "user", "users", "file", "files", "code", "data", "test",
            "name", "names", "entity", "entities", "relationship", "relationships",
            "connection", "connections", "server", "servers", "client", "clients",
            "software", "hardware", "process", "processes", "function", "functions",
            "class", "classes", "object", "objects", "method", "methods", "module",
            "modules", "package", "packages", "library", "libraries", "framework",
            "frameworks", "tool", "tools", "technology", "technologies", "platform",
            "platforms", "component", "components", "architecture", "architectures",
            "document", "documents", "text", "chunk", "chunks", "page", "pages",
            "file name", "file names", "upload", "uploads", "source", "sources",
            "port", "ports", "dataset", "datasets", "model", "models", "paper", "papers",
            "year", "years"
        }
        if name_clean.lower() in GENERIC_TERMS:
            return False

        # Rule 4: OCR Noise filter
        # Must contain at least one letter OR be a valid port number (excluding common year numbers)
        if not any(c.isalpha() for c in name_clean):
            if name_clean.isdigit():
                val = int(name_clean)
                if not (1 <= val <= 65535) or (1900 <= val <= 2100):
                    return False
            else:
                return False

        # Check for weird symbols typical of OCR misreading
        if re.search(r'[^a-zA-Z0-9\s\-\.\'_\(\)]', name_clean):
            return False

        return True

    def extract_entities(self, text: str) -> list[dict]:
        entities = []
        seen = set()

        # 1. Rule-based static keyword lookups
        words = re.findall(r'\b[a-zA-Z0-9\-]+\b', text)
        for w in words:
            wl = w.lower()
            if wl in TECH_KEYWORDS:
                ent_name = w if wl != "postgres" else "PostgreSQL"
                ent_type = TECH_KEYWORDS[wl]
                if not self.is_valid_entity(ent_name):
                    continue
                key = (ent_name.lower(), ent_type)
                if key not in seen:
                    seen.add(key)
                    entities.append({"name": ent_name, "type": ent_type})

        # 2. Pattern-based regex extractions
        for ent_type, pattern in self.patterns.items():
            matches = pattern.findall(text)
            for match in matches:
                # regex findall returns tuples if there are multiple capture groups
                val = next((v for v in match if v), "").strip()
                if val and len(val) > 1:
                    if not self.is_valid_entity(val):
                        continue
                    key = (val.lower(), ent_type)
                    if key not in seen:
                        seen.add(key)
                        entities.append({"name": val, "type": ent_type})

        return entities
