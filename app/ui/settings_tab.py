"""Settings: which LLM provider judges and optimizes, its API key, and the HF token."""
import gradio as gr

from app.core import secrets_store
from app.core.config import load_config, update_config
from app.core.logging_setup import read_recent_logs
from app.llm import client
from app.llm.providers import (
    CATEGORIES,
    PROVIDERS,
    category_choices,
    category_description,
    get_provider,
    provider_choices,
    providers_in,
)
from app.ui import components as ui


def _provider_note(provider) -> str:
    parts = []
    if provider.notes:
        parts.append(provider.notes)
    if provider.keys_url:
        parts.append(f'<a href="{provider.keys_url}" target="_blank">Get an API key ↗</a>')
    if provider.docs_url:
        parts.append(f'<a href="{provider.docs_url}" target="_blank">Docs ↗</a>')
    if not provider.requires_key:
        parts.append("No API key required.")
    return f'<div class="pt-hint">{" · ".join(parts)}</div>' if parts else ""


def _on_category_change(category: str):
    """Switching category narrows the provider list and jumps to its first entry."""
    choices = provider_choices(category)
    first = choices[0][1] if choices else None
    return (gr.update(choices=choices, value=first),
            f'<div class="pt-hint">{category_description(category)}</div>',
            *_on_provider_change(first))


def _on_provider_change(provider_key: str):
    provider = get_provider(provider_key)
    stored_key = secrets_store.get_api_key(provider.key)
    models = list(provider.suggested_models)
    return (
        gr.update(value=provider.base_url, interactive=True),
        gr.update(value=stored_key, placeholder="Not set" if provider.requires_key else "Not required"),
        gr.update(choices=models, value=provider.default_model or None),
        _provider_note(provider),
        ui.status("Not tested yet", f"{provider.label} selected — save, then test the connection.", "warn"),
    )


def _refresh_models(provider_key: str, base_url: str, api_key: str):
    models, message = client.fetch_models(provider_key, base_url, api_key)
    return gr.update(choices=models), ui.status("Model list", message, "ok" if models else "warn")


def _test(provider_key: str, base_url: str, model: str, api_key: str):
    ok, message = client.test_connection(provider_key, base_url, model, api_key)
    return ui.status("Connection OK" if ok else "Connection failed", message, "ok" if ok else "bad")


def _save(provider_key: str, base_url: str, model: str, api_key: str,
          judge_model: str, optimizer_model: str, assistant_model: str, hf_token: str):
    provider = get_provider(provider_key)
    secrets_store.set_api_key(provider.key, api_key)
    secrets_store.set_hf_token(hf_token)
    update_config({
        "llm": {
            "provider": provider.key,
            "base_url": (base_url or provider.base_url).strip(),
            "model": (model or provider.default_model).strip(),
            "roles": {
                "judge": (judge_model or "").strip(),
                "optimizer": (optimizer_model or "").strip(),
                "assistant": (assistant_model or "").strip(),
            },
        }
    })
    detail = f"{provider.label} · {model}"
    if any((judge_model, optimizer_model, assistant_model)):
        detail += " (with per-role overrides)"
    return ui.status("Settings saved", detail, "ok")


def build(header: gr.HTML, render_header) -> None:
    config = load_config()
    llm = config["llm"]
    provider = get_provider(llm.get("provider", ""))
    stored_key = secrets_store.get_api_key(provider.key)

    gr.Markdown("### Evaluation provider")
    gr.Markdown(
        "The judge, the prompt optimizer, and the assistant all run on this provider. "
        "Any OpenAI-compatible endpoint works — pick a preset or use **Custom**.",
        elem_classes=["pt-hint"],
    )

    with gr.Row():
        with gr.Column(scale=3):
            category_radio = gr.Radio(
                label="Provider type",
                choices=category_choices(),
                value=provider.category,
            )
            category_note = gr.HTML(
                f'<div class="pt-hint">{category_description(provider.category)}</div>'
            )
            provider_dropdown = gr.Dropdown(
                label="Provider",
                choices=provider_choices(provider.category),
                value=provider.key,
            )
            provider_note = gr.HTML(_provider_note(provider))
            api_key_box = gr.Textbox(
                label="API key",
                value=stored_key,
                type="password",
                placeholder="Not set" if provider.requires_key else "Not required",
                info="Stored in data/secrets.json on this machine only — never in config.json.",
            )
            base_url_box = gr.Textbox(
                label="Base URL",
                value=llm.get("base_url", provider.base_url),
                info="Edit only if you use a regional endpoint or a self-hosted gateway.",
            )
            with gr.Row():
                model_dropdown = gr.Dropdown(
                    label="Default model",
                    choices=list(provider.suggested_models) or [llm.get("model", "")],
                    value=llm.get("model", provider.default_model),
                    allow_custom_value=True,
                    scale=4,
                )
                refresh_btn = gr.Button("Refresh list", scale=1)

            with gr.Accordion("Per-role model overrides (optional)", open=False):
                gr.Markdown(
                    "Leave blank to use the default model. Judging runs once per test case, "
                    "so a cheap fast model there saves the most.",
                    elem_classes=["pt-hint"],
                )
                judge_model = gr.Textbox(label="Judge model", value=llm.get("roles", {}).get("judge", ""),
                                         placeholder="same as default")
                optimizer_model = gr.Textbox(label="Optimizer model", value=llm.get("roles", {}).get("optimizer", ""),
                                             placeholder="same as default")
                assistant_model = gr.Textbox(label="Assistant model", value=llm.get("roles", {}).get("assistant", ""),
                                             placeholder="same as default")

        with gr.Column(scale=2):
            connection_status = gr.HTML(
                ui.status("Not tested yet", "Save your key, then run a connection test.", "warn")
            )
            test_btn = gr.Button("Test connection")
            save_btn = gr.Button("Save settings", variant="primary")
            gr.Markdown("### Hugging Face", elem_classes=["pt-hint"])
            hf_token_box = gr.Textbox(
                label="Hugging Face token",
                value=secrets_store.get_hf_token(),
                type="password",
                placeholder="Optional — needed for gated or private repos",
            )
            gr.HTML(
                '<div class="pt-hint">'
                '<a href="https://huggingface.co/settings/tokens" target="_blank">Create a token ↗</a>'
                "</div>"
            )

    with gr.Accordion("Provider reference", open=False):
        for category_key, category_label, category_note_text in CATEGORIES:
            pool = providers_in(category_key)
            if not pool:
                continue
            gr.Markdown(f"**{category_label}** — {category_note_text}", elem_classes=["pt-hint"])
            gr.HTML(
                ui.table(
                    ["Provider", "Base URL", "Default model", "Key needed"],
                    [
                        [
                            ui.cell(item.label),
                            ui.cell(item.base_url or "you supply it", mono=True),
                            ui.cell(item.default_model or "—", mono=True),
                            "yes" if item.requires_key else "no",
                        ]
                        for item in pool
                    ],
                )
            )

    with gr.Accordion("Application log", open=False):
        log_view = gr.Code(value=read_recent_logs(120), language=None, label="logs/app.log (last 120 lines)")
        gr.Button("Reload log").click(lambda: read_recent_logs(120), outputs=log_view)

    provider_outputs = [base_url_box, api_key_box, model_dropdown, provider_note, connection_status]
    category_radio.change(
        _on_category_change,
        inputs=category_radio,
        outputs=[provider_dropdown, category_note, *provider_outputs],
    )
    provider_dropdown.change(
        _on_provider_change,
        inputs=provider_dropdown,
        outputs=provider_outputs,
    )
    refresh_btn.click(
        _refresh_models,
        inputs=[provider_dropdown, base_url_box, api_key_box],
        outputs=[model_dropdown, connection_status],
    )
    test_btn.click(
        _test,
        inputs=[provider_dropdown, base_url_box, model_dropdown, api_key_box],
        outputs=connection_status,
    )
    save_btn.click(
        _save,
        inputs=[provider_dropdown, base_url_box, model_dropdown, api_key_box,
                judge_model, optimizer_model, assistant_model, hf_token_box],
        outputs=connection_status,
    ).then(render_header, outputs=header)
