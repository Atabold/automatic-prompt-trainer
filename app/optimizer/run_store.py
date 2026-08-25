"""Per-run artifacts on disk: one folder per optimization run, readable by the History tab."""
import json
from datetime import datetime
from pathlib import Path

from app.core.logging_setup import get_logger
from app.core.paths import LEGACY_TEST_RUNS_DIR, TEST_RUNS_DIR, ensure_dirs

logger = get_logger(__name__)


def migrate_legacy_runs() -> None:
    if not LEGACY_TEST_RUNS_DIR.exists() or not LEGACY_TEST_RUNS_DIR.is_dir():
        return
    ensure_dirs()
    for item in LEGACY_TEST_RUNS_DIR.iterdir():
        target = TEST_RUNS_DIR / item.name
        if not target.exists():
            item.rename(target)
    try:
        LEGACY_TEST_RUNS_DIR.rmdir()
        logger.info("Migrated test_runs/ -> %s", TEST_RUNS_DIR)
    except OSError:
        pass


class RunStore:
    """Writes summary/history/details/logs for one run and keeps them current mid-run."""

    def __init__(self, run_id: str):
        ensure_dirs()
        self.run_id = run_id
        self.run_dir = TEST_RUNS_DIR / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.started_at = datetime.now()
        self.events: list[dict] = []
        self.api_calls = 0

    def event(self, level: str, message: str, data: dict | None = None) -> None:
        entry = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "level": level,
            "message": message,
        }
        if data:
            entry["data"] = data
        self.events.append(entry)
        logger.log(
            {"ERROR": 40, "WARN": 30, "INFO": 20}.get(level, 10),
            "[%s] %s", self.run_id, message,
        )
        self._dump("events.json", self.events)

    def count_api_call(self, call_type: str, model: str) -> None:
        self.api_calls += 1
        self.event("DEBUG", f"API call #{self.api_calls} ({call_type})", {"model": model})

    def _dump(self, name: str, payload) -> None:
        with open(self.run_dir / name, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    def write_summary(self, summary: dict) -> None:
        summary = {
            "run_id": self.run_id,
            "started_at": self.started_at.isoformat(timespec="seconds"),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "duration_seconds": round((datetime.now() - self.started_at).total_seconds(), 1),
            "api_calls": self.api_calls,
            **summary,
        }
        self._dump("summary.json", summary)

    def write_history(self, history: list[dict]) -> None:
        self._dump("history.json", history)


LEGACY_FIELDS = {
    "start_time": "started_at",
    "end_time": "finished_at",
    "api_calls_made": "api_calls",
    "test_cases_evaluated": "test_cases",
}


def _normalize(summary: dict, run_id: str) -> dict:
    """Runs written by the pre-restructure logger used different key names."""
    for old, new in LEGACY_FIELDS.items():
        if old in summary and new not in summary:
            summary[new] = summary.pop(old)
    summary.setdefault("run_id", run_id)
    summary.setdefault("base_score", 0.0)
    summary.setdefault("status", "unknown")
    return summary


def _read_summary(run_dir: Path) -> dict | None:
    for name in ("summary.json", "state.json"):
        target = run_dir / name
        if not target.exists():
            continue
        try:
            with open(target, "r", encoding="utf-8") as handle:
                return _normalize(json.load(handle), run_dir.name)
        except (json.JSONDecodeError, OSError):
            logger.debug("Skipping unreadable run summary at %s", target)
    return None


def list_runs() -> list[dict]:
    migrate_legacy_runs()
    runs = []
    for path in sorted(TEST_RUNS_DIR.glob("*/"), reverse=True):
        summary = _read_summary(path)
        if summary:
            runs.append(summary)
    return runs


def load_run(run_id: str) -> dict:
    run_dir = TEST_RUNS_DIR / run_id
    payload: dict = {"run_id": run_id, "summary": _read_summary(run_dir) or {}}
    for name, key in (("history.json", "history"), ("events.json", "events"), ("logs.json", "events")):
        target = run_dir / name
        if target.exists() and not payload.get(key):
            try:
                with open(target, "r", encoding="utf-8") as handle:
                    payload[key] = json.load(handle)
            except (json.JSONDecodeError, OSError):
                payload[key] = None
    return payload


def delete_run(run_id: str) -> None:
    import shutil

    run_dir = (TEST_RUNS_DIR / run_id).resolve()
    if not str(run_dir).startswith(str(TEST_RUNS_DIR.resolve())) or not run_dir.exists():
        raise ValueError(f"Unknown run '{run_id}'.")
    shutil.rmtree(run_dir)
    logger.info("Deleted run %s", run_id)


def new_run_id() -> str:
    return datetime.now().strftime("run_%Y%m%d_%H%M%S")
