"""Single source of truth for every path the application reads or writes."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
LOGS_DIR = ROOT / "logs"
WEIGHTS_DIR = ROOT / "models"

CONFIG_FILE = DATA_DIR / "config.json"
SECRETS_FILE = DATA_DIR / "secrets.json"
REGISTRY_FILE = DATA_DIR / "models_registry.json"
ARCHIVE_DIR = DATA_DIR / "archive"
TEST_RUNS_DIR = DATA_DIR / "test_runs"
LOG_FILE = LOGS_DIR / "app.log"

LEGACY_CONFIG_FILE = ROOT / "config.json"
LEGACY_ARCHIVE_DIR = ROOT / "archive"
LEGACY_TEST_RUNS_DIR = ROOT / "test_runs"


def ensure_dirs() -> None:
    for directory in (DATA_DIR, LOGS_DIR, WEIGHTS_DIR, ARCHIVE_DIR, TEST_RUNS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
