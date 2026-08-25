"""Inventory of GGUF models downloaded to models/ — what they are and where they came from."""
import json
from datetime import datetime
from pathlib import Path

from app.core.logging_setup import get_logger
from app.core.paths import REGISTRY_FILE, WEIGHTS_DIR, ensure_dirs

logger = get_logger(__name__)


def _read() -> list[dict]:
    if not REGISTRY_FILE.exists():
        return []
    try:
        with open(REGISTRY_FILE, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError):
        logger.warning("Model registry unreadable; rebuilding from disk.")
        return []


def _write(entries: list[dict]) -> None:
    ensure_dirs()
    with open(REGISTRY_FILE, "w", encoding="utf-8") as handle:
        json.dump(entries, handle, indent=2, ensure_ascii=False)


def scan_disk() -> list[Path]:
    if not WEIGHTS_DIR.exists():
        return []
    return sorted(path for path in WEIGHTS_DIR.rglob("*.gguf") if path.is_file())


def list_models() -> list[dict]:
    """Registry entries reconciled against what is actually on disk."""
    entries = {entry["path"]: entry for entry in _read()}
    reconciled: list[dict] = []

    for path in scan_disk():
        key = str(path)
        entry = entries.get(key, {})
        reconciled.append(
            {
                "path": key,
                "filename": path.name,
                "repo_id": entry.get("repo_id", "(found on disk)"),
                "size_gb": round(path.stat().st_size / 1024**3, 2),
                "downloaded_at": entry.get(
                    "downloaded_at",
                    datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
                ),
            }
        )

    _write(reconciled)
    return reconciled


def record(repo_id: str, filename: str, path: str) -> dict:
    entries = [entry for entry in _read() if entry.get("path") != path]
    entry = {
        "path": path,
        "filename": filename,
        "repo_id": repo_id,
        "size_gb": round(Path(path).stat().st_size / 1024**3, 2) if Path(path).exists() else 0.0,
        "downloaded_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    entries.append(entry)
    _write(entries)
    logger.info("Registered model %s (%s)", filename, repo_id)
    return entry


def delete_model(path: str) -> None:
    target = Path(path)
    resolved = target.resolve()
    if not str(resolved).startswith(str(WEIGHTS_DIR.resolve())):
        raise ValueError("Refusing to delete a file outside the models directory.")
    if resolved.exists():
        resolved.unlink()
        logger.info("Deleted model file %s", resolved)
    _write([entry for entry in _read() if entry.get("path") != path])
