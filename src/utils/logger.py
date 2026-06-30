"""
logger.py — Centralized loguru logger configuration.
Import setup_logger() once at app startup.
"""

import sys
from loguru import logger
from pathlib import Path


def setup_logger(log_level: str = "INFO", log_to_file: bool = True) -> None:
    """
    Configure loguru for the project.

    Args:
        log_level: Minimum log level to capture ("DEBUG", "INFO", "WARNING", "ERROR").
        log_to_file: Whether to also write logs to logs/app.log.
    """
    # Remove default handler
    logger.remove()

    # Console handler — colorized
    logger.add(
        sys.stdout,
        level=log_level,
        colorize=True,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
    )

    # File handler — optional
    if log_to_file:
        log_dir = Path(__file__).resolve().parents[2] / "logs"
        log_dir.mkdir(exist_ok=True)
        logger.add(
            log_dir / "app.log",
            level=log_level,
            rotation="10 MB",
            retention="7 days",
            compression="zip",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} — {message}",
        )
