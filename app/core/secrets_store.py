"""API keys live here, never in config.json.

config.json is snapshotted into data/archive/ on every save, so putting secrets in it
would scatter plaintext keys across dozens of archive files. This store is a single
gitignored file, seeded once from environment variables for backwards compatibility.
"""
import json
import os
import stat

from app.core.logging_setup import get_logger
from app.core.paths import SECRETS_FILE, ensure_dirs
from app.llm.providers import PROVIDERS

logger = get_logger(__name__)

HF_TOKEN_ENV = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN")


def _empty() -> dict:
    return {"provider_keys": {}, "hf_token": ""}


def _read() -> dict:
    if not SECRETS_FILE.exists():
        return _empty()
    try:
        with open(SECRETS_FILE, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, OSError):
        logger.warning("Could not read %s; starting from an empty secret store.", SECRETS_FILE)
        return _empty()
    data.setdefault("provider_keys", {})
    data.setdefault("hf_token", "")
    return data


def _write(data: dict) -> None:
    ensure_dirs()
    with open(SECRETS_FILE, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
    try:
        os.chmod(SECRETS_FILE, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass


def _env_key(provider_key: str) -> str:
    for provider in PROVIDERS:
        if provider.key == provider_key:
            for name in provider.api_key_env:
                value = os.getenv(name, "").strip()
                if value:
                    return value
    return ""


def get_api_key(provider_key: str) -> str:
    """Stored key wins; environment variables are the fallback."""
    stored = _read()["provider_keys"].get(provider_key, "").strip()
    return stored or _env_key(provider_key)


def set_api_key(provider_key: str, api_key: str) -> None:
    data = _read()
    api_key = (api_key or "").strip()
    if api_key:
        data["provider_keys"][provider_key] = api_key
    else:
        data["provider_keys"].pop(provider_key, None)
    _write(data)
    logger.info("API key for provider '%s' %s.", provider_key, "saved" if api_key else "cleared")


def get_hf_token() -> str:
    stored = _read().get("hf_token", "").strip()
    if stored:
        return stored
    for name in HF_TOKEN_ENV:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""


def set_hf_token(token: str) -> None:
    data = _read()
    data["hf_token"] = (token or "").strip()
    _write(data)
    logger.info("Hugging Face token %s.", "saved" if data["hf_token"] else "cleared")


def configured_providers() -> set[str]:
    data = _read()["provider_keys"]
    return {key for key, value in data.items() if value.strip()}


def mask(secret: str) -> str:
    if not secret:
        return ""
    if len(secret) <= 8:
        return "•" * len(secret)
    return f"{secret[:4]}{'•' * 8}{secret[-4:]}"
