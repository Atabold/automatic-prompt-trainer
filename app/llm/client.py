"""Builds the OpenAI-compatible client for whichever provider is configured."""
from dataclasses import dataclass

from openai import OpenAI

from app.core import secrets_store
from app.core.config import load_config
from app.core.logging_setup import get_logger
from app.llm.providers import Provider, get_provider

logger = get_logger(__name__)

ROLES = ("judge", "optimizer", "assistant")


class ProviderNotConfigured(RuntimeError):
    """Raised when the selected provider is missing an API key or a base URL."""


@dataclass
class LLMTarget:
    client: OpenAI
    model: str
    provider: Provider
    role: str

    @property
    def label(self) -> str:
        return f"{self.provider.label} · {self.model}"


def resolve_settings(role: str = "judge") -> tuple[Provider, str, str]:
    """Return (provider, base_url, model) for a role, falling back to the shared default."""
    llm = load_config().get("llm", {})
    provider = get_provider(llm.get("provider", ""))
    base_url = (llm.get("base_url") or provider.base_url).strip()
    model = (llm.get("roles", {}).get(role) or llm.get("model") or provider.default_model).strip()
    return provider, base_url, model


def build_target(role: str = "judge") -> LLMTarget:
    provider, base_url, model = resolve_settings(role)
    api_key = secrets_store.get_api_key(provider.key)

    if provider.requires_key and not api_key:
        raise ProviderNotConfigured(
            f"No API key for {provider.label}. Open the Settings tab, paste your key, and save."
        )
    if not base_url:
        raise ProviderNotConfigured(
            f"No base URL configured for {provider.label}. Set it in the Settings tab."
        )
    if not model:
        raise ProviderNotConfigured(
            f"No model selected for {provider.label}. Choose one in the Settings tab."
        )

    client = OpenAI(api_key=api_key or "not-needed", base_url=base_url, timeout=120.0, max_retries=2)
    return LLMTarget(client=client, model=model, provider=provider, role=role)


def chat(role: str, messages: list[dict], **kwargs):
    """Single chat completion call for a role. Returns (response, target)."""
    target = build_target(role)
    logger.debug("LLM call role=%s provider=%s model=%s", role, target.provider.key, target.model)
    response = target.client.chat.completions.create(
        model=target.model, messages=messages, **kwargs
    )
    return response, target


def test_connection(provider_key: str, base_url: str, model: str, api_key: str) -> tuple[bool, str]:
    """Fire one minimal request so the user finds out now, not mid-run."""
    provider = get_provider(provider_key)
    base_url = (base_url or provider.base_url).strip()
    model = (model or provider.default_model).strip()
    api_key = (api_key or secrets_store.get_api_key(provider_key)).strip()

    if provider.requires_key and not api_key:
        return False, "No API key provided."
    if not base_url:
        return False, "No base URL provided."
    if not model:
        return False, "No model name provided."

    try:
        client = OpenAI(api_key=api_key or "not-needed", base_url=base_url, timeout=30.0, max_retries=0)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with the single word: ok"}],
            max_tokens=5,
            temperature=0,
        )
        reply = (response.choices[0].message.content or "").strip()
        logger.info("Connection test OK for %s / %s", provider.label, model)
        return True, f"Connected to {provider.label} — `{model}` replied: “{reply[:40]}”"
    except Exception as exc:
        logger.warning("Connection test failed for %s / %s: %s", provider.label, model, exc)
        return False, f"{type(exc).__name__}: {exc}"


def fetch_models(provider_key: str, base_url: str, api_key: str) -> tuple[list[str], str]:
    """Ask the provider what models it serves. Falls back to the curated list."""
    provider = get_provider(provider_key)
    base_url = (base_url or provider.base_url).strip()
    api_key = (api_key or secrets_store.get_api_key(provider_key)).strip()
    fallback = list(provider.suggested_models)

    if not base_url or (provider.requires_key and not api_key):
        return fallback, "Using the built-in list — add a key and base URL to fetch live."

    try:
        client = OpenAI(api_key=api_key or "not-needed", base_url=base_url, timeout=30.0, max_retries=0)
        names = sorted({item.id for item in client.models.list().data})
        if not names:
            return fallback, "Provider returned no models; using the built-in list."
        return names, f"Fetched {len(names)} models from {provider.label}."
    except Exception as exc:
        logger.info("Model listing unavailable for %s: %s", provider.label, exc)
        return fallback, f"Could not list models ({type(exc).__name__}); using the built-in list."
