import logging
import sys
from rich.console import Console
from rich.logging import RichHandler

console = Console()


def get_logger(name: str = "omnirag") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = RichHandler(console=console, rich_tracebacks=True, show_time=True, show_path=False)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.propagate = False
    return logger


logger = get_logger("omnirag")
