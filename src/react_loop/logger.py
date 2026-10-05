import logging
import os
from pathlib import Path
from typing import Optional

LOG_DIR = Path("logs")

def setup_session_logger(session_id: str, level: int = logging.INFO) -> logging.Logger:
    """Create a logger that writes to a session-specific file."""
    LOG_DIR.mkdir(exist_ok=True)
    
    logger = logging.getLogger(f"react_loop.session.{session_id}")
    logger.setLevel(level)
    
    # Avoid duplicate handlers if the logger is reused
    if not logger.handlers:
        log_file = LOG_DIR / f"session_{session_id}.log"
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger
