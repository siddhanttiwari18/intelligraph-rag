import re
from pathlib import Path
from rag.utils.exceptions import SecurityError


def validate_filename(filename: str) -> str:
    """Sanitize the filename to prevent directory traversal attacks.
    
    Ensures the name contains only valid characters and has no folder navigation parts.
    """
    if not filename:
        raise SecurityError("Filename cannot be empty.")
        
    # Block traversal sequences
    if ".." in filename or "/" in filename or "\\" in filename:
        raise SecurityError(f"Directory traversal attempt detected in filename: {filename}")
        
    # Keep only alphanumeric characters, dots, dashes, and underscores
    cleaned = re.sub(r"[^\w\.\-]", "_", filename)
    
    # Ensure there is a valid extension
    if not Path(cleaned).suffix:
        raise SecurityError(f"Filename must contain a valid extension: {filename}")
        
    return cleaned


def validate_file_content(file_bytes: bytes, filename: str) -> bool:
    """Validate file content using magic bytes to match the declared extension."""
    if not file_bytes:
        raise SecurityError("File content is empty.")

    suffix = Path(filename).suffix.lower()

    if suffix == ".pdf":
        # PDF files must start with %PDF- (hex: 25 50 44 46 2d)
        if not file_bytes.startswith(b"%PDF"):
            raise SecurityError(f"Security validation failed: File '{filename}' declares PDF extension but header magic bytes do not match %PDF.")
    elif suffix in (".txt", ".md", ".json", ".csv"):
        # Validate that the bytes represent clean UTF-8 text strings without null bytes
        try:
            # Check for null bytes which indicate binary files
            if b"\x00" in file_bytes[:1024]:
                raise SecurityError(f"Security validation failed: Plain text file '{filename}' contains binary null bytes.")
            file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            try:
                file_bytes.decode("latin-1")
            except Exception:
                raise SecurityError(f"Security validation failed: Plain text file '{filename}' does not contain valid text encoding.")
    else:
        # Unknown extension - block for strict security unless explicitly supported
        raise SecurityError(f"Security validation failed: Unsupported file extension '{suffix}' for filename '{filename}'.")
        
    return True
