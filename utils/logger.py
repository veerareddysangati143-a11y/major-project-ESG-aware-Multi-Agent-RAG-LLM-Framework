"""
Logger Module for ESG Multi-Agent System.
Provides a unified logging setup across all agents, models, and pipelines.
"""

import sys
import logging
from pathlib import Path
from config import LOG_LEVEL, LOG_FILE


def setup_logger(name: str = "esg_multi_agent") -> logging.Logger:
    """
    Creates or retrieves a logger instance with console and file handlers.
    
    Args:
        name (str): The name of the module/logger.
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    
    # Avoid adding duplicate handlers if logger is already set up
    if logger.handlers:
        return logger

    numeric_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning(f"Could not initialize file handler for logging: {e}")

    return logger


# Default logger for project-wide use
logger = setup_logger()
