"""Central logging configuration: rotating file log plus a console stream."""
import logging
import os
from logging.handlers import RotatingFileHandler

from app.core.paths import LOG_FILE, ensure_dirs

_configured = False

CONSOLE_FORMAT = "%(asctime)s  %(levelname)-7s %(name)-28s %(message)s"
FILE_FORMAT = "%(asctime)s  %(levelname)-7s %(name)s  %(message)s"


def is_dev_mode() -> bool:
    return os.getenv("APP_ENV", "PROD").strip().upper() == "DEV"


def configure_logging(force: bool = False) -> None:
    global _configured
    if _configured and not force:
        return

    ensure_dirs()
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)

    console_level = logging.DEBUG if is_dev_mode() else logging.INFO
    root.setLevel(logging.DEBUG)

    console = logging.StreamHandler()
    console.setLevel(console_level)
    console.setFormatter(logging.Formatter(CONSOLE_FORMAT, datefmt="%H:%M:%S"))
    root.addHandler(console)

    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=2_000_000, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(FILE_FORMAT))
    root.addHandler(file_handler)

    for noisy in ("httpx", "httpcore", "urllib3", "huggingface_hub", "openai", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True
    logging.getLogger(__name__).debug("Logging configured (dev_mode=%s)", is_dev_mode())


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)


def read_recent_logs(lines: int = 200) -> str:
    if not LOG_FILE.exists():
        return "No log file yet."
    with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as handle:
        return "".join(handle.readlines()[-lines:])
