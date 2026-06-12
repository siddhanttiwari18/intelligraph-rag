import logging
import sys
from pathlib import Path

# Define logging format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"

def setup_logger(log_dir: str = "./rag_storage", log_level: int = logging.INFO) -> logging.Logger:
    """Configures and returns the platform logger.
    
    Creates a file handler pointing to `log_dir/platform.log` and a console handler.
    """
    logger = logging.getLogger("rag_platform")
    logger.setLevel(log_level)
    
    # Avoid duplicate handlers if setup is called multiple times
    if logger.handlers:
        return logger

    # Ensure log directory exists
    log_path = Path(log_dir)
    try:
        log_path.mkdir(parents=True, exist_ok=True)
        file_path = log_path / "platform.log"
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Warning: Could not create file logging handler: {e}", file=sys.stderr)

    # Console handler
    console_handler = sys.stdout
    stream_handler = logging.StreamHandler(console_handler)
    stream_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(stream_handler)

    return logger

# Create default active logger instance
logger = setup_logger()
