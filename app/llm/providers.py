"""Registry of OpenAI-compatible LLM providers usable as judge / optimizer / assistant.

Every provider here exposes an OpenAI-style ``/chat/completions`` endpoint, so a single
``openai.OpenAI(api_key=..., base_url=...)`` client covers all of them. ``suggested_models``
is only a starting point for the dropdowns — the Settings tab can refresh the list live
from each provider's ``/models`` endpoint, and every model field accepts a custom value.
"""
from dataclasses import dataclass

DIRECT = "direct"
ROUTER = "router"
LOCAL = "local"
CUSTOM = "custom"

CATEGORIES: tuple[tuple[str, str, str], ...] = (
    (DIRECT, "API providers", "First-party endpoints from the labs that build the models."),
    (ROUTER, "Routers & gateways", "One key, many vendors — useful for comparing models cheaply."),
    (LOCAL, "Local", "Runs on your own machine. No key, no data leaves the box."),
    (CUSTOM, "Custom", "Any other endpoint that speaks the OpenAI chat-completions API."),
)

CATEGORY_LABELS = {key: label for key, label, _ in CATEGORIES}


@dataclass(frozen=True)
class Provider:
    key: str
    label: str
    base_url: str
    default_model: str
    suggested_models: tuple[str, ...]
    category: str = DIRECT
    api_key_env: tuple[str, ...] = ()
    keys_url: str = ""
    docs_url: str = ""
    editable_base_url: bool = False
    requires_key: bool = True
    supports_tools: bool = True
    notes: str = ""


PROVIDERS: tuple[Provider, ...] = (
    Provider(
        key="mistral",
        label="Mistral",
        base_url="https://api.mistral.ai/v1",
        default_model="mistral-medium-latest",
        suggested_models=(
            "mistral-large-latest",
            "mistral-medium-latest",
            "mistral-small-latest",
            "magistral-medium-latest",
            "magistral-small-latest",
            "ministral-8b-latest",
            "ministral-3b-latest",
            "codestral-latest",
            "open-mistral-nemo",
        ),
        api_key_env=("MISTRAL_API_KEY",),
        keys_url="https://console.mistral.ai/api-keys",
        docs_url="https://docs.mistral.ai/api/",
    ),
    Provider(
        key="anthropic",
        label="Anthropic (Claude)",
        base_url="https://api.anthropic.com/v1",
        default_model="claude-sonnet-5",
        suggested_models=(
            "claude-opus-5",
            "claude-sonnet-5",
            "claude-fable-5",
            "claude-haiku-4-5-20251001",
        ),
        api_key_env=("ANTHROPIC_API_KEY",),
        keys_url="https://console.anthropic.com/settings/keys",
        docs_url="https://docs.claude.com/en/api/openai-sdk",
        notes="Served through Anthropic's OpenAI-SDK compatibility layer.",
    ),
    Provider(
        key="deepseek",
        label="DeepSeek",
        base_url="https://api.deepseek.com",
        default_model="deepseek-chat",
        suggested_models=(
            "deepseek-chat",
            "deepseek-reasoner",
        ),
        api_key_env=("DEEPSEEK_API_KEY",),
        keys_url="https://platform.deepseek.com/api_keys",
        docs_url="https://api-docs.deepseek.com/",
    ),
    Provider(
        key="qwen",
        label="Qwen (Alibaba DashScope)",
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        default_model="qwen-plus",
        suggested_models=(
            "qwen-max",
            "qwen-plus",
            "qwen-turbo",
            "qwen3-max",
            "qwen3-coder-plus",
        ),
        api_key_env=("DASHSCOPE_API_KEY", "QWEN_API_KEY"),
        keys_url="https://bailian.console.alibabacloud.com/",
        docs_url="https://www.alibabacloud.com/help/en/model-studio/compatibility-of-openai-with-dashscope",
        editable_base_url=True,
        notes="Use the CN endpoint (dashscope.aliyuncs.com) if your account is mainland-China based.",
    ),
    Provider(
        key="openai",
        label="OpenAI",
        base_url="https://api.openai.com/v1",
        default_model="gpt-4.1-mini",
        suggested_models=(
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-4o",
            "gpt-4o-mini",
            "o4-mini",
        ),
        api_key_env=("OPENAI_API_KEY",),
        keys_url="https://platform.openai.com/api-keys",
        docs_url="https://platform.openai.com/docs/api-reference",
    ),
    Provider(
        key="xai",
        label="Grok (xAI)",
        base_url="https://api.x.ai/v1",
        default_model="grok-4-fast",
        suggested_models=(
            "grok-4",
            "grok-4-fast",
            "grok-3",
            "grok-3-mini",
        ),
        api_key_env=("XAI_API_KEY", "GROK_API_KEY"),
        keys_url="https://console.x.ai/",
        docs_url="https://docs.x.ai/docs/api-reference",
    ),
    Provider(
        key="moonshot",
        label="Kimi (Moonshot AI)",
        base_url="https://api.moonshot.ai/v1",
        default_model="kimi-latest",
        suggested_models=(
            "kimi-latest",
            "kimi-k2-turbo-preview",
            "moonshot-v1-8k",
            "moonshot-v1-32k",
            "moonshot-v1-128k",
        ),
        api_key_env=("MOONSHOT_API_KEY", "KIMI_API_KEY"),
        keys_url="https://platform.moonshot.ai/console/api-keys",
        docs_url="https://platform.moonshot.ai/docs/api/chat",
        editable_base_url=True,
        notes="Mainland-China accounts use https://api.moonshot.cn/v1 instead.",
    ),
    Provider(
        key="meta",
        label="Meta (Llama API)",
        base_url="https://api.llama.com/compat/v1",
        default_model="Llama-4-Maverick-17B-128E-Instruct-FP8",
        suggested_models=(
            "Llama-4-Maverick-17B-128E-Instruct-FP8",
            "Llama-4-Scout-17B-16E-Instruct-FP8",
            "Llama-3.3-70B-Instruct",
            "Llama-3.3-8B-Instruct",
        ),
        api_key_env=("LLAMA_API_KEY", "META_API_KEY"),
        keys_url="https://llama.developer.meta.com/",
        docs_url="https://llama.developer.meta.com/docs/",
    ),
    Provider(
        key="cortecs",
        label="Cortecs (Europe's LLM router)",
        base_url="https://api.cortecs.ai/v1",
        default_model="mistral-medium-3.5",
        category=ROUTER,
        suggested_models=(
            "mistral-medium-3.5",
            "claude-sonnet-5",
            "claude-opus-5",
            "deepseek-v4-flash-0731",
            "glm-5.2",
            "kimi-k3",
            "gpt-5.4",
            "gemini-3.7-flash",
            "qwen3.8-27b",
        ),
        api_key_env=("CORTECS_API_KEY",),
        keys_url="https://cortecs.ai/userArea/console?tab=billing",
        docs_url="https://docs.cortecs.ai/quickstart",
        notes="EU-hosted, GDPR-focused gateway to 150+ endpoints across 10+ vendors. "
              "Model ids change often — use Refresh list to pull the live catalogue.",
    ),
    Provider(
        key="groq",
        label="Groq",
        base_url="https://api.groq.com/openai/v1",
        default_model="llama-3.3-70b-versatile",
        category=ROUTER,
        suggested_models=(
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "moonshotai/kimi-k2-instruct",
            "openai/gpt-oss-120b",
            "qwen/qwen3-32b",
        ),
        api_key_env=("GROQ_API_KEY",),
        keys_url="https://console.groq.com/keys",
        docs_url="https://console.groq.com/docs/openai",
    ),
    Provider(
        key="together",
        label="Together AI",
        base_url="https://api.together.xyz/v1",
        default_model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        category=ROUTER,
        suggested_models=(
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "deepseek-ai/DeepSeek-V3",
            "Qwen/Qwen2.5-72B-Instruct-Turbo",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
        ),
        api_key_env=("TOGETHER_API_KEY",),
        keys_url="https://api.together.xyz/settings/api-keys",
        docs_url="https://docs.together.ai/docs/openai-api-compatibility",
    ),
    Provider(
        key="openrouter",
        label="OpenRouter",
        base_url="https://openrouter.ai/api/v1",
        default_model="anthropic/claude-sonnet-4.5",
        category=ROUTER,
        suggested_models=(
            "anthropic/claude-sonnet-4.5",
            "openai/gpt-4.1-mini",
            "mistralai/mistral-medium-3.1",
            "deepseek/deepseek-chat",
            "meta-llama/llama-3.3-70b-instruct",
            "qwen/qwen3-max",
            "x-ai/grok-4-fast",
            "moonshotai/kimi-k2",
        ),
        api_key_env=("OPENROUTER_API_KEY",),
        keys_url="https://openrouter.ai/keys",
        docs_url="https://openrouter.ai/docs/quickstart",
        notes="Single key, routes to almost every vendor. Good fallback when a direct key is unavailable.",
    ),
    Provider(
        key="ollama",
        label="Ollama (local)",
        base_url="http://localhost:11434/v1",
        default_model="llama3.2",
        category=LOCAL,
        suggested_models=(
            "llama3.2",
            "qwen2.5",
            "mistral",
            "gemma3",
        ),
        api_key_env=(),
        keys_url="",
        docs_url="https://docs.ollama.com/openai",
        editable_base_url=True,
        requires_key=False,
        notes="Runs on your machine. No API key needed; make sure `ollama serve` is running.",
    ),
    Provider(
        key="custom",
        label="Custom (OpenAI-compatible)",
        base_url="",
        default_model="",
        suggested_models=(),
        category=CUSTOM,
        api_key_env=(),
        keys_url="",
        docs_url="",
        editable_base_url=True,
        notes="Point this at any endpoint that speaks the OpenAI chat-completions API.",
    ),
)

PROVIDERS_BY_KEY: dict[str, Provider] = {provider.key: provider for provider in PROVIDERS}

DEFAULT_PROVIDER = "mistral"


def get_provider(key: str) -> Provider:
    return PROVIDERS_BY_KEY.get(key, PROVIDERS_BY_KEY[DEFAULT_PROVIDER])


def providers_in(category: str) -> list[Provider]:
    return [provider for provider in PROVIDERS if provider.category == category]


def provider_choices(category: str | None = None) -> list[tuple[str, str]]:
    """(label, key) pairs for a Gradio dropdown, optionally limited to one category."""
    pool = providers_in(category) if category else list(PROVIDERS)
    return [(provider.label, provider.key) for provider in pool]


def category_choices() -> list[tuple[str, str]]:
    return [(label, key) for key, label, _ in CATEGORIES]


def category_description(category: str) -> str:
    return next((note for key, _, note in CATEGORIES if key == category), "")
