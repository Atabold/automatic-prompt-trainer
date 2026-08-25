"""Registry of OpenAI-compatible LLM providers usable as judge / optimizer / assistant.

Every provider here exposes an OpenAI-style ``/chat/completions`` endpoint, so a single
``openai.OpenAI(api_key=..., base_url=...)`` client covers all of them. ``suggested_models``
is only a starting point for the dropdowns — the Settings tab can refresh the list live
from each provider's ``/models`` endpoint, and every model field accepts a custom value.

Model ids last checked against provider documentation on 2026-08-25. Catalogues move fast;
when a name here 404s, use **Refresh list** in Settings to pull the provider's live ids.
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
        default_model="mistral-medium-2604",
        suggested_models=(
            "mistral-medium-2604",       # Mistral Medium 3.5
            "mistral-large-2512",        # Mistral Large 3
            "mistral-small-2603",        # Mistral Small 4
            "ministral-3-14b-2512",
            "ministral-3-8b-2512",
            "ministral-3-3b-2512",
            "codestral-latest",
        ),
        api_key_env=("MISTRAL_API_KEY",),
        keys_url="https://console.mistral.ai/api-keys",
        docs_url="https://docs.mistral.ai/api/",
        notes="The `-latest` aliases (mistral-medium-latest, …) still resolve but point at "
              "older snapshots than the dated ids above.",
    ),
    Provider(
        key="anthropic",
        label="Anthropic (Claude)",
        base_url="https://api.anthropic.com/v1",
        default_model="claude-opus-5",
        suggested_models=(
            "claude-opus-5",
            "claude-sonnet-5",
            "claude-fable-5",
            "claude-opus-4-8",
            "claude-haiku-4-5",
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
        default_model="deepseek-v4-pro",
        suggested_models=(
            "deepseek-v4-pro",              # GA, serves DeepSeek-V4-Pro-0813
            "deepseek-v4-flash",           # cheaper, public beta (V4-Flash-0731)
            "deepseek-v4-flash-vision-exp",
        ),
        api_key_env=("DEEPSEEK_API_KEY",),
        keys_url="https://platform.deepseek.com/api_keys",
        docs_url="https://api-docs.deepseek.com/",
        notes="The old deepseek-chat / deepseek-reasoner aliases were discontinued on "
              "2026-07-24 — use the V4 ids.",
    ),
    Provider(
        key="zai",
        label="GLM (Z.ai)",
        base_url="https://api.z.ai/api/paas/v4",
        default_model="glm-5.3",
        suggested_models=(
            "glm-5.3",
            "glm-5.2",
            "glm-5.1",
            "glm-5-turbo",
            "glm-4.7",
            "glm-4.7-flash",
            "glm-4.6",
            "glm-4.6v",      # vision
        ),
        api_key_env=("ZAI_API_KEY", "GLM_API_KEY"),
        keys_url="https://z.ai/manage-apikey/apikey-list",
        docs_url="https://docs.z.ai/guides/develop/http/introduction",
        editable_base_url=True,
        notes="Z.ai also exposes a pure OpenAI-compatible endpoint at "
              "https://api.z.ai/api/openai/v1 — swap the base URL if a call is rejected. "
              "GLM Coding Plan subscriptions use their own dedicated endpoint.",
    ),
    Provider(
        key="qwen",
        label="Qwen (Alibaba DashScope)",
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        default_model="qwen3.7-plus",
        suggested_models=(
            "qwen3.8-max",
            "qwen3.7-max",
            "qwen3.7-plus",
            "qwen3-max",
            "qwen-plus-latest",
            "qwen-turbo",
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
        default_model="gpt-5.6-terra",
        suggested_models=(
            "gpt-5.6-sol",     # flagship (also aliased as gpt-5.6)
            "gpt-5.6-terra",   # balanced
            "gpt-5.6-luna",    # cheapest
        ),
        api_key_env=("OPENAI_API_KEY",),
        keys_url="https://platform.openai.com/api-keys",
        docs_url="https://developers.openai.com/api/docs/models",
        notes="The GPT-4.x / 4o / o-series names this app used to suggest are superseded by "
              "the GPT-5.6 Sol / Terra / Luna tiers.",
    ),
    Provider(
        key="xai",
        label="Grok (xAI)",
        base_url="https://api.x.ai/v1",
        default_model="grok-4.6",
        suggested_models=(
            "grok-4.6",
            "grok-4.5",
            "grok-4.3",
            "grok-4.20-0309-reasoning",
            "grok-4.20-0309-non-reasoning",
        ),
        api_key_env=("XAI_API_KEY", "GROK_API_KEY"),
        keys_url="https://console.x.ai/",
        docs_url="https://docs.x.ai/developers/models",
    ),
    Provider(
        key="moonshot",
        label="Kimi (Moonshot AI)",
        base_url="https://api.moonshot.ai/v1",
        default_model="kimi-k3",
        suggested_models=(
            "kimi-k3",
            "kimi-k2.7-code",
            "kimi-k2.7-code-highspeed",
            "kimi-k2.6",
        ),
        api_key_env=("MOONSHOT_API_KEY", "KIMI_API_KEY"),
        keys_url="https://platform.kimi.ai/console/api-keys",
        docs_url="https://platform.kimi.ai/docs/models",
        editable_base_url=True,
        notes="Mainland-China accounts use https://api.moonshot.cn/v1 instead. "
              "kimi-latest and the kimi-k2 series are already discontinued; "
              "kimi-k2.5 and the moonshot-v1-* models sunset on 2026-08-31.",
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
        default_model="openai/gpt-oss-120b",
        category=ROUTER,
        suggested_models=(
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "groq/compound",
            "groq/compound-mini",
        ),
        api_key_env=("GROQ_API_KEY",),
        keys_url="https://console.groq.com/keys",
        docs_url="https://console.groq.com/docs/openai",
    ),
    Provider(
        key="together",
        label="Together AI",
        base_url="https://api.together.xyz/v1",
        default_model="deepseek-ai/DeepSeek-V4-Pro",
        category=ROUTER,
        suggested_models=(
            "deepseek-ai/DeepSeek-V4-Pro",
            "deepseek-ai/DeepSeek-V4-Flash-0731",
            "moonshotai/Kimi-K3",
            "zai-org/GLM-5.2",
            "Qwen/Qwen3.7-Plus",
            "openai/gpt-oss-120b",
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        ),
        api_key_env=("TOGETHER_API_KEY",),
        keys_url="https://api.together.xyz/settings/api-keys",
        docs_url="https://docs.together.ai/docs/openai-api-compatibility",
    ),
    Provider(
        key="openrouter",
        label="OpenRouter",
        base_url="https://openrouter.ai/api/v1",
        default_model="anthropic/claude-sonnet-5",
        category=ROUTER,
        suggested_models=(
            "anthropic/claude-sonnet-5",
            "anthropic/claude-opus-5",
            "deepseek/deepseek-v4-pro",
            "deepseek/deepseek-v4-flash",
            "z-ai/glm-5.3",
            "z-ai/glm-5.2",
            "openai/gpt-oss-120b",
            "moonshotai/kimi-k3",
            "x-ai/grok-4.6",
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
        default_model="gpt-oss:20b",
        category=LOCAL,
        suggested_models=(
            "gpt-oss:20b",
            "qwen3-coder:30b",
            "qwen3:30b",
            "gemma4",
            "mistral-small3.1",
            "llama3.3:70b",
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
