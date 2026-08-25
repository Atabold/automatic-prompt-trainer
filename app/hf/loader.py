"""Loads and hot-swaps the local GGUF model used for formatting."""
import threading
from pathlib import Path

from app.core.config import load_config, update_config
from app.core.logging_setup import get_logger
from app.hf import registry

logger = get_logger(__name__)

_model = None
_model_path: Path | None = None
_lock = threading.Lock()


class NoModelAvailable(RuntimeError):
    """Raised when no GGUF file exists to load."""


def active_model_path() -> Path | None:
    """The configured model, else the first one on disk."""
    configured = load_config().get("model_runtime", {}).get("active_model_path", "")
    if configured and Path(configured).exists():
        return Path(configured)
    found = registry.scan_disk()
    return found[0] if found else None


def set_active_model(path: str) -> Path:
    target = Path(path)
    if not target.exists():
        raise ValueError(f"Model file not found: {path}")
    update_config({"model_runtime": {"active_model_path": str(target)}})
    unload_model()
    logger.info("Active model set to %s", target.name)
    return target


def is_loaded() -> bool:
    return _model is not None


def loaded_model_name() -> str:
    return _model_path.name if _model_path else "none"


def unload_model() -> None:
    global _model, _model_path
    with _lock:
        if _model is not None:
            logger.info("Unloading model %s", _model_path.name if _model_path else "?")
        _model = None
        _model_path = None


def load_model(model_path: str | None = None):
    global _model, _model_path

    target = Path(model_path) if model_path else active_model_path()
    if target is None:
        raise NoModelAvailable(
            "No GGUF model found. Open the Models tab and download one from Hugging Face."
        )
    if not target.exists():
        raise NoModelAvailable(f"Model file missing: {target}")

    with _lock:
        if _model is not None and _model_path == target:
            return _model

        from llama_cpp import Llama  # imported lazily: heavy, and not needed for API-only work

        runtime = load_config().get("model_runtime", {})
        logger.info("Loading %s (n_ctx=%s, n_threads=%s)", target.name,
                    runtime.get("n_ctx", 4096), runtime.get("n_threads", 8))
        _model = Llama(
            model_path=str(target),
            n_ctx=int(runtime.get("n_ctx", 4096)),
            n_threads=int(runtime.get("n_threads", 8)),
            n_batch=int(runtime.get("n_batch", 512)),
            verbose=False,
        )
        _model_path = target
        logger.info("Model ready: %s", target.name)
        return _model


def format_text(model, system_prompt: str, text: str, max_tokens: int | None = None,
                temperature: float | None = None) -> str:
    runtime = load_config().get("model_runtime", {})
    prompt = f"{system_prompt}\n\nText: {text}\n\nFormatted output:"
    output = model(
        prompt,
        max_tokens=int(max_tokens if max_tokens is not None else runtime.get("max_tokens", 512)),
        temperature=float(temperature if temperature is not None else runtime.get("temperature", 0.2)),
        stop=["\n\n", "###"],
        echo=False,
    )
    return output["choices"][0]["text"].strip()
