"""Format: run the local model over one piece of text with the current prompt."""
import time

import gradio as gr

from app.core.config import load_config, update_config
from app.core.logging_setup import get_logger
from app.evaluation import judge
from app.hf import loader
from app.ui import components as ui

logger = get_logger(__name__)

SAMPLE = (
    "ummm so let's meet at 5 pm... actually 6 pm. also please bring the, uh, documents "
    "and the second thing check the price first then compare quality"
)


def _format(text: str, prompt: str):
    if not (text or "").strip():
        yield ui.status("Nothing to format", "Paste some raw text first.", "warn"), "", ""
        return

    started = time.time()
    yield ui.status("Loading model…", loader.loaded_model_name(), "running", 0.15), "", ""

    try:
        model = loader.load_model()
    except Exception as exc:
        yield ui.status("Model unavailable", str(exc), "bad"), "", ""
        return

    yield ui.status("Formatting…", f"{loader.loaded_model_name()} · {len(text)} characters",
                    "running", 0.6), "", ""

    try:
        output = loader.format_text(model, prompt or load_config()["base_prompt"], text)
    except Exception as exc:
        logger.exception("Formatting failed")
        yield ui.status("Formatting failed", f"{type(exc).__name__}: {exc}", "bad"), "", ""
        return

    elapsed = time.time() - started
    detected = judge.detect_language(output)
    metrics = ui.stats([
        ui.stat("Time", f"{elapsed:.1f}s"),
        ui.stat("Input", f"{len(text)} ch"),
        ui.stat("Output", f"{len(output)} ch"),
        ui.stat("Language", detected or "—"),
        ui.stat("Filler words", "none" if not judge.FILLER_PATTERN.search(output) else "found",
                tone="ok" if not judge.FILLER_PATTERN.search(output) else "bad"),
    ])
    yield ui.status("Done", f"Formatted in {elapsed:.1f}s", "ok", 1.0), output, metrics


def _save_prompt(prompt: str):
    update_config({"base_prompt": prompt})
    return ui.status("Base prompt saved", "It is now the default everywhere, including runs.", "ok")


def _reload_prompt():
    return load_config(refresh=True)["base_prompt"]


def build() -> gr.Textbox:
    config = load_config()

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### System prompt")
            prompt_box = gr.Textbox(
                label="Base prompt",
                value=config["base_prompt"],
                lines=12,
                show_label=False,
            )
            with gr.Row():
                save_prompt_btn = gr.Button("Save as default", variant="secondary")
                reload_prompt_btn = gr.Button("Reload saved")
            prompt_status = gr.HTML("")

        with gr.Column(scale=1):
            gr.Markdown("### Input")
            input_box = gr.Textbox(
                label="Raw spoken text",
                placeholder=SAMPLE,
                lines=8,
                show_label=False,
            )
            format_btn = gr.Button("Format text", variant="primary")
            gr.Examples(examples=[[SAMPLE]], inputs=[input_box], label="Example")

    gr.Markdown("### Result")
    run_status = gr.HTML(ui.status("Idle", "Nothing running.", ""))
    output_box = gr.Textbox(label="Formatted output", lines=8, buttons=["copy"])
    metrics_html = gr.HTML("")

    format_btn.click(_format, inputs=[input_box, prompt_box], outputs=[run_status, output_box, metrics_html])
    save_prompt_btn.click(_save_prompt, inputs=prompt_box, outputs=prompt_status)
    reload_prompt_btn.click(_reload_prompt, outputs=prompt_box)

    return prompt_box
