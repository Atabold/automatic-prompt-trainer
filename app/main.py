"""Gradio entry point. Composes the tabs and the global status header."""
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

from app.core import secrets_store
from app.core.config import load_config, test_case_stats
from app.core.logging_setup import configure_logging, get_logger, is_dev_mode
from app.core.paths import ensure_dirs
from app.hf import loader
from app.llm.providers import get_provider
from app.optimizer.run_store import migrate_legacy_runs
from app.ui import (
    assistant_tab,
    format_tab,
    history_tab,
    models_tab,
    optimizer_tab,
    settings_tab,
    suite_tab,
)
from app.ui import components as ui
from app.ui.theme import APP_SUBTITLE, APP_TITLE, CSS, build_theme

logger = get_logger(__name__)


def render_header() -> str:
    config = load_config(refresh=True)
    provider = get_provider(config["llm"].get("provider", ""))
    model = config["llm"].get("model", provider.default_model)
    has_key = bool(secrets_store.get_api_key(provider.key)) or not provider.requires_key
    active_model = loader.active_model_path()
    stats = test_case_stats()

    return (
        '<div id="pt-header">'
        f'<div class="pt-brand"><h1>{APP_TITLE}</h1>'
        f'<p class="pt-sub">{APP_SUBTITLE}</p></div>'
        + ui.pills([
            ui.pill("Evaluator", f"{provider.label} · {model}" if has_key else f"{provider.label} · no key",
                    "ok" if has_key else "bad"),
            ui.pill("Local model", active_model.name if active_model else "none downloaded",
                    "ok" if active_model else "warn"),
            ui.pill("Test cases", str(stats["total"]), "ok" if stats["total"] else "warn"),
            ui.pill("Mode", "DEV" if is_dev_mode() else "PROD"),
        ])
        + "</div>"
    )


def build_app() -> gr.Blocks:
    configure_logging()
    ensure_dirs()
    migrate_legacy_runs()
    load_config()

    with gr.Blocks(title=APP_TITLE, fill_width=True) as demo:
        header = gr.HTML(render_header)

        with gr.Tabs():
            with gr.Tab("Format"):
                prompt_box = format_tab.build()
            with gr.Tab("Optimizer"):
                optimizer_tab.build(prompt_box)
            with gr.Tab("Test suite"):
                suite_tab.build(prompt_box)
            with gr.Tab("History"):
                history_tab.build()
            with gr.Tab("Models"):
                models_tab.build(header, render_header)
            with gr.Tab("Assistant"):
                assistant_tab.build(prompt_box)
            with gr.Tab("Settings"):
                settings_tab.build(header, render_header)

    return demo


def main() -> None:
    demo = build_app()
    logger.info("Starting %s", APP_TITLE)
    demo.launch(theme=build_theme(), css=CSS, favicon_path=None)


if __name__ == "__main__":
    main()
