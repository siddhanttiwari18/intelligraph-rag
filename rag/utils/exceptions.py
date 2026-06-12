class RAGException(Exception):
    """Base exception for all RAG platform errors."""
    pass


class ConfigurationError(RAGException):
    """Raised when configuration or environment validation fails."""
    pass


class LLMError(RAGException):
    """Raised when an LLM provider API call fails or times out."""
    pass


class RetrievalError(RAGException):
    """Raised when vector search, BM25, or re-ranking retrieval stage fails."""
    pass


class GraphError(RAGException):
    """Raised when knowledge graph store modification or relationship extraction fails."""
    pass


class SecurityError(RAGException):
    """Raised when input validation, path traversal check, or magic bytes check fails."""
    pass


class OCRError(RAGException):
    """Raised when OCR loader processing fails."""
    pass


class UploadError(RAGException):
    """Raised when file upload validation or parsing fails."""
    pass
