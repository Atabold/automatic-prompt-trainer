"""Application configuration: load, save, archive, and edit test cases.

Every save snapshots the previous config into data/archive/ (capped, oldest pruned) so
an LLM-driven edit is always recoverable. Secrets never enter this file — see
app/core/secrets_store.py.
"""
import json
import os
import shutil
from copy import deepcopy
from datetime import datetime

from app.core.logging_setup import get_logger
from app.core.paths import (
    ARCHIVE_DIR,
    CONFIG_FILE,
    LEGACY_ARCHIVE_DIR,
    LEGACY_CONFIG_FILE,
    ensure_dirs,
)
from app.llm.providers import DEFAULT_PROVIDER, get_provider

logger = get_logger(__name__)

ARCHIVE_RETENTION = 25

DEFAULT_CONFIG = {
    "base_prompt": """You are a text formatter. Clean up the given spoken/voice-input text by:
- keeping the input language exactly
- resolving self-corrections (use the last correction)
- removing filler words and false starts
- fixing punctuation and capitalization
- formatting lists, paragraphs, and emails automatically
Return only the cleaned text.""",
    "evaluation_criteria": """Rate the output from 0 to 100 based on:
1. Language preservation: output must be in the same language as input.
2. Self-correction handling: the last correction should be used.
3. Filler words removed.
4. Punctuation, capitalization, and formatting correct.
5. The output should be concise and ready to send.""",
    "weights": {
        "language": 0.3,
        "filler": 0.2,
        "judge": 0.5,
    },
    "test_cases": [],
    "llm": {
        "provider": DEFAULT_PROVIDER,
        "base_url": "",
        "model": "",
        "roles": {"judge": "", "optimizer": "", "assistant": ""},
    },
    "run": {
        "max_iterations": 5,
        "threshold": 0.95,
        "pass_threshold": 0.95,
        "candidates_per_iteration": 2,
        "sample_size": 0,
    },
    "model_runtime": {
        "active_model_path": "",
        "n_ctx": 4096,
        "n_threads": max(os.cpu_count() or 8, 4),
        "n_batch": 512,
        "max_tokens": 512,
        "temperature": 0.2,
    },
}

_cache: dict | None = None


def _seed_test_cases() -> list[dict]:
    try:
        from data.seed_test_cases import TEST_CASES

        return deepcopy(TEST_CASES)
    except Exception:
        logger.warning("No seed test cases available; starting with an empty suite.")
        return []


def _fill_defaults(config: dict) -> dict:
    """Backfill any key added since the config file was written."""
    for key, default in DEFAULT_CONFIG.items():
        if key not in config:
            config[key] = deepcopy(default)
        elif isinstance(default, dict) and isinstance(config[key], dict):
            for sub_key, sub_default in default.items():
                config[key].setdefault(sub_key, deepcopy(sub_default))
    provider = get_provider(config["llm"].get("provider", DEFAULT_PROVIDER))
    if not config["llm"].get("base_url"):
        config["llm"]["base_url"] = provider.base_url
    if not config["llm"].get("model"):
        config["llm"]["model"] = provider.default_model
    return config


def _migrate_legacy() -> None:
    """One-time move of the flat-layout config/archive into data/."""
    ensure_dirs()
    if not CONFIG_FILE.exists() and LEGACY_CONFIG_FILE.exists():
        shutil.move(str(LEGACY_CONFIG_FILE), str(CONFIG_FILE))
        logger.info("Migrated config.json -> %s", CONFIG_FILE)
    if LEGACY_ARCHIVE_DIR.exists() and LEGACY_ARCHIVE_DIR.is_dir():
        for item in LEGACY_ARCHIVE_DIR.iterdir():
            target = ARCHIVE_DIR / item.name
            if not target.exists():
                shutil.move(str(item), str(target))
        try:
            LEGACY_ARCHIVE_DIR.rmdir()
        except OSError:
            pass


def load_config(refresh: bool = False) -> dict:
    global _cache
    if _cache is not None and not refresh:
        return _cache

    _migrate_legacy()
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r", encoding="utf-8") as handle:
            config = json.load(handle)
        _cache = _fill_defaults(config)
        return _cache

    config = deepcopy(DEFAULT_CONFIG)
    config["test_cases"] = _seed_test_cases()
    _cache = _fill_defaults(config)
    save_config(_cache)
    logger.info("Created a fresh config with %d seed test cases.", len(config["test_cases"]))
    return _cache


def _prune_archive() -> None:
    snapshots = sorted(ARCHIVE_DIR.glob("config_*.json"))
    for stale in snapshots[:-ARCHIVE_RETENTION]:
        try:
            stale.unlink()
        except OSError:
            logger.debug("Could not prune archive file %s", stale)


def save_config(config: dict) -> dict:
    global _cache
    ensure_dirs()
    if CONFIG_FILE.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        shutil.copy(CONFIG_FILE, ARCHIVE_DIR / f"config_{timestamp}.json")
        _prune_archive()
    with open(CONFIG_FILE, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)
    _cache = config
    return config


def update_config(changes: dict) -> dict:
    config = deepcopy(load_config())
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            config[key].update(value)
        else:
            config[key] = value
    return save_config(config)


# ---------------------------------------------------------------- test cases

def list_test_cases() -> list[dict]:
    return load_config().get("test_cases", [])


def find_test_case(case_id: str) -> dict | None:
    return next((case for case in list_test_cases() if case.get("id") == case_id), None)


def add_test_case(case_id: str, language: str, input_text: str, expected: str) -> dict:
    if find_test_case(case_id):
        raise ValueError(f"Test case '{case_id}' already exists. Use update instead.")
    config = deepcopy(load_config())
    case = {"id": case_id, "language": language, "input": input_text, "expected": expected}
    config["test_cases"].append(case)
    save_config(config)
    logger.info("Added test case %s (%s)", case_id, language)
    return case


def update_test_case(case_id: str, **fields) -> dict:
    config = deepcopy(load_config())
    for case in config["test_cases"]:
        if case.get("id") == case_id:
            for key, value in fields.items():
                if value is not None and key in {"language", "input", "expected"}:
                    case[key] = value
            save_config(config)
            logger.info("Updated test case %s", case_id)
            return case
    raise ValueError(f"Test case '{case_id}' not found.")


def remove_test_case(case_id: str) -> str:
    config = deepcopy(load_config())
    remaining = [case for case in config["test_cases"] if case.get("id") != case_id]
    if len(remaining) == len(config["test_cases"]):
        raise ValueError(f"Test case '{case_id}' not found.")
    config["test_cases"] = remaining
    save_config(config)
    logger.info("Removed test case %s", case_id)
    return case_id


def test_case_stats() -> dict:
    cases = list_test_cases()
    languages: dict[str, int] = {}
    for case in cases:
        languages[case.get("language", "unknown")] = languages.get(case.get("language", "unknown"), 0) + 1
    return {"total": len(cases), "by_language": dict(sorted(languages.items()))}


# ---------------------------------------------------------------- archive

def list_archive() -> list[dict]:
    entries = []
    for path in sorted(ARCHIVE_DIR.glob("config_*.json"), reverse=True):
        stat = path.stat()
        entries.append(
            {
                "name": path.name,
                "path": str(path),
                "saved_at": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                "size_kb": round(stat.st_size / 1024, 1),
            }
        )
    return entries


def restore_archive(name: str) -> dict:
    source = ARCHIVE_DIR / name
    if not source.exists():
        raise ValueError(f"Archive '{name}' not found.")
    with open(source, "r", encoding="utf-8") as handle:
        config = _fill_defaults(json.load(handle))
    save_config(config)
    logger.info("Restored configuration from %s", name)
    return config
