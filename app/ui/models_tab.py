"""Models: download any GGUF repo from Hugging Face, pick a quant, choose the active model."""
import gradio as gr

from app.core.config import load_config, update_config
from app.core.logging_setup import get_logger
from app.hf import downloader, loader, registry
from app.ui import components as ui

logger = get_logger(__name__)

EXAMPLES = [
    "unsloth/gemma-3-4b-it-GGUF",
    "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF",
    "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/blob/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
]


def _installed_table() -> str:
    models = registry.list_models()
    if not models:
        return ui.empty("No models downloaded yet. Paste a Hugging Face repo id or URL above.")
    active = loader.active_model_path()
    rows = []
    for item in models:
        is_active = active is not None and str(active) == item["path"]
        rows.append([
            ui.cell(item["filename"], mono=True),
            ui.cell(item["repo_id"]),
            f"{item['size_gb']:.2f} GB",
            ui.cell(item["downloaded_at"]),
            '<span class="pt-badge pass">active</span>' if is_active else "",
        ])
    return ui.table(["File", "Source repo", "Size", "Downloaded", ""], rows, numeric_columns={2})


def _model_choices() -> list[tuple[str, str]]:
    return [
        (f"{item['filename']}  ·  {item['size_gb']:.2f} GB", item["path"])
        for item in registry.list_models()
    ]


def _fetch_files(reference: str):
    try:
        ref, files = downloader.list_gguf_files(reference)
    except Exception as exc:
        return (
            gr.update(choices=[], value=None),
            ui.status("Could not read that repo", str(exc), "bad"),
            gr.update(value=""),
            None,
        )

    choices = [
        (f"{item['quant']:<8}  {item['filename']}  ·  {item['size']}"
         + ("  (already downloaded)" if item["downloaded"] else ""), item["filename"])
        for item in files
    ]
    preselect = ref.filename or downloader.recommend(files)
    detail = f"{len(files)} GGUF file(s) · revision {ref.revision}"
    if ref.filename:
        detail += f" · URL pointed at {ref.filename}"
    return (
        gr.update(choices=choices, value=preselect),
        ui.status(f"Found {ref.repo_id}", detail, "ok"),
        gr.update(value=ref.repo_id),
        {"repo_id": ref.repo_id, "revision": ref.revision,
         "sizes": {item["filename"]: item["size_bytes"] for item in files}},
    )


def _download(repo_state: dict, filename: str):
    if not repo_state or not filename:
        yield ui.status("Nothing to download", "Fetch a repo and pick a file first.", "warn"), gr.update(), gr.update()
        return

    total = (repo_state.get("sizes") or {}).get(filename)
    job = downloader.start_download(repo_state["repo_id"], filename, repo_state.get("revision", "main"), total)

    for update in downloader.poll(job, interval=1.0):
        if update.status == "downloading":
            done_gb = update.downloaded_bytes / 1024**3
            total_gb = (update.total_bytes or 0) / 1024**3
            detail = f"{done_gb:.2f} GB of {total_gb:.2f} GB" if total_gb else f"{done_gb:.2f} GB"
            yield (
                ui.status(f"Downloading {filename}", detail, "running", update.fraction),
                gr.update(),
                gr.update(),
            )
        elif update.status == "pending":
            yield ui.status("Starting download…", filename, "running", 0.0), gr.update(), gr.update()

    if job.status == "completed":
        yield (
            ui.status("Download complete", job.message, "ok", 1.0),
            _installed_table(),
            gr.update(choices=_model_choices(), value=job.local_path),
        )
    else:
        yield ui.status("Download failed", job.message, "bad"), _installed_table(), gr.update()


def _activate(path: str, header_unused=None):
    if not path:
        return ui.status("No model selected", "Pick one from the list first.", "warn"), _installed_table()
    try:
        target = loader.set_active_model(path)
    except Exception as exc:
        return ui.status("Could not activate", str(exc), "bad"), _installed_table()
    return (
        ui.status("Active model set", f"{target.name} — it loads on the next format or optimizer run.", "ok"),
        _installed_table(),
    )


def _delete(path: str):
    if not path:
        return ui.status("No model selected", "Pick one from the list first.", "warn"), _installed_table(), gr.update()
    try:
        if loader.active_model_path() and str(loader.active_model_path()) == path:
            loader.unload_model()
            update_config({"model_runtime": {"active_model_path": ""}})
        registry.delete_model(path)
    except Exception as exc:
        return ui.status("Delete failed", str(exc), "bad"), _installed_table(), gr.update()
    choices = _model_choices()
    return (
        ui.status("Model deleted", "The .gguf file was removed from models/.", "ok"),
        _installed_table(),
        gr.update(choices=choices, value=choices[0][1] if choices else None),
    )


def _save_runtime(n_ctx, n_threads, n_batch, max_tokens, temperature):
    update_config({"model_runtime": {
        "n_ctx": int(n_ctx), "n_threads": int(n_threads), "n_batch": int(n_batch),
        "max_tokens": int(max_tokens), "temperature": float(temperature),
    }})
    loader.unload_model()
    return ui.status("Runtime settings saved", "The model will reload with these values.", "ok")


def build(header: gr.HTML, render_header) -> None:
    runtime = load_config().get("model_runtime", {})
    choices = _model_choices()
    active = loader.active_model_path()

    gr.Markdown("### Download a model from Hugging Face")
    gr.Markdown(
        "Paste a repo id (`owner/repo`), a repo URL, or a direct link to one `.gguf` file. "
        "This app runs GGUF/llama.cpp models — repos whose names end in **-GGUF**.",
        elem_classes=["pt-hint"],
    )

    with gr.Row():
        reference_box = gr.Textbox(
            label="Repo id or URL",
            placeholder="unsloth/gemma-3-4b-it-GGUF   or   https://huggingface.co/…",
            scale=4,
        )
        fetch_btn = gr.Button("Fetch files", variant="secondary", scale=1)
    gr.Examples(examples=[[item] for item in EXAMPLES], inputs=[reference_box], label="Examples")

    repo_state = gr.State(None)
    resolved_repo = gr.Textbox(visible=False)

    file_dropdown = gr.Dropdown(
        label="Quantization / file",
        choices=[],
        info="Q4_K_M is the usual quality-to-size sweet spot and is preselected when present.",
    )
    download_btn = gr.Button("Download selected file", variant="primary")
    download_status = gr.HTML(ui.status("Ready", "Nothing downloading.", ""))

    gr.Markdown("### Installed models")
    installed_html = gr.HTML(_installed_table())
    with gr.Row():
        installed_dropdown = gr.Dropdown(
            label="Select a model",
            choices=choices,
            value=str(active) if active else None,
            scale=3,
        )
        activate_btn = gr.Button("Set as active", scale=1)
        delete_btn = gr.Button("Delete file", variant="stop", scale=1)
    manage_status = gr.HTML(
        ui.status("Active model", active.name if active else "none — download one above",
                  "ok" if active else "warn")
    )

    with gr.Accordion("Runtime settings (llama.cpp)", open=False):
        with gr.Row():
            n_ctx = gr.Number(label="Context window", value=runtime.get("n_ctx", 4096), precision=0)
            n_threads = gr.Number(label="Threads", value=runtime.get("n_threads", 8), precision=0)
            n_batch = gr.Number(label="Batch size", value=runtime.get("n_batch", 512), precision=0)
        with gr.Row():
            max_tokens = gr.Number(label="Max output tokens", value=runtime.get("max_tokens", 512), precision=0)
            temperature = gr.Slider(label="Temperature", minimum=0.0, maximum=1.5,
                                    step=0.05, value=runtime.get("temperature", 0.2))
        runtime_btn = gr.Button("Save runtime settings")
        runtime_status = gr.HTML("")

    fetch_btn.click(
        _fetch_files,
        inputs=reference_box,
        outputs=[file_dropdown, download_status, resolved_repo, repo_state],
    )
    reference_box.submit(
        _fetch_files,
        inputs=reference_box,
        outputs=[file_dropdown, download_status, resolved_repo, repo_state],
    )
    download_btn.click(
        _download,
        inputs=[repo_state, file_dropdown],
        outputs=[download_status, installed_html, installed_dropdown],
    ).then(render_header, outputs=header)
    activate_btn.click(
        _activate,
        inputs=installed_dropdown,
        outputs=[manage_status, installed_html],
    ).then(render_header, outputs=header)
    delete_btn.click(
        _delete,
        inputs=installed_dropdown,
        outputs=[manage_status, installed_html, installed_dropdown],
    ).then(render_header, outputs=header)
    runtime_btn.click(
        _save_runtime,
        inputs=[n_ctx, n_threads, n_batch, max_tokens, temperature],
        outputs=runtime_status,
    )
