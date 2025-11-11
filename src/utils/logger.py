"""
Logging configuration for HYCO-PhiFlow.

This module sets up a centralized logging system with:
- Console output with color coding
- File output with rotation
- Different log levels for different modules
- Structured logging with context
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """Formatter that adds color to console output."""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record):
        # Shorten module name to last component
        name_parts = record.name.split(".")
        if len(name_parts) > 2:
            # Keep only the last 2 parts (e.g., training.synthetic_trainer)
            record.name = ".".join(name_parts[-2:])

        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logger(
    name: str,
    log_dir: Optional[Path] = None,
    level: int = logging.INFO,
    log_to_file: bool = True,
    log_to_console: bool = True,
) -> logging.Logger:
    """
    Set up a logger with file and console handlers.

    Args:
        name: Logger name (usually __name__ of the module)
        log_dir: Directory for log files (default: logs/)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to log to file
        log_to_console: Whether to log to console

    Returns:
        Configured logger instance

    Example:
        >>> logger = setup_logger(__name__)
        >>> logger.info("Training started")
        >>> logger.error("Model failed to load", exc_info=True)
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # Don't propagate to root logger (avoids Hydra duplicate)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Console handler with color
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        # Simplified format: just [module] LEVEL: message
        console_format = ColoredFormatter("[%(name)s] %(levelname)s: %(message)s")
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)

    # File handler with rotation
    if log_to_file:
        if log_dir is None:
            log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f'{name.replace(".", "_")}_{timestamp}.log'

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # File gets all messages
        file_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

        logger.debug(f"Logging to file: {log_file}")

    return logger


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Get or create a logger with default configuration.

    Args:
        name: Logger name (use __name__)
        level: Override default logging level

    Returns:
        Logger instance
    """
    if level is None:
        level = logging.INFO

    return setup_logger(name, level=level)
