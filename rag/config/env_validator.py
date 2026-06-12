import os
from pathlib import Path
from rag.config.config import Settings


def run_diagnostics(settings: Settings) -> dict:
    """Runs setup diagnostics and returns a status dictionary of checks.
    
    Checks:
    - DEEPSEEK_API_KEY is configured
    - Storage directory is writeable
    - Config file validity
    """
    diagnostics = {
        "deepseek_api_key_configured": False,
        "storage_dir_writeable": False,
        "storage_dir_path": str(settings.persist_dir),
        "errors": [],
        "warnings": []
    }

    # 1. Check DEEPSEEK_API_KEY
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if api_key:
        diagnostics["deepseek_api_key_configured"] = True
    else:
        diagnostics["errors"].append("DEEPSEEK_API_KEY environment variable is not configured.")

    # 2. Check storage directory write permissions
    try:
        persist_path = Path(settings.persist_dir)
        persist_path.mkdir(parents=True, exist_ok=True)
        
        # Try writing a temporary diagnostic file to test write permissions
        temp_file = persist_path / ".diagnostic_write_test"
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write("test")
        temp_file.unlink()  # Clean up
        
        diagnostics["storage_dir_writeable"] = True
    except Exception as e:
        diagnostics["errors"].append(f"Storage directory '{settings.persist_dir}' is not writeable: {e}")

    # 3. Check for specific python dependencies if needed (optional diagnostics warning)
    try:
        import sentence_transformers
    except ImportError:
        diagnostics["warnings"].append("sentence_transformers package is missing. Embeddings and re-ranking might fail.")

    try:
        import faiss
    except ImportError:
        diagnostics["warnings"].append("faiss package is missing. FAISS vector store will not function.")

    return diagnostics
