"""History: browse every past optimization run stored under data/test_runs/."""
import gradio as gr

from app.core.config import load_config
from app.optimizer import run_store
from app.ui import components as ui

STATUS_TONE = {"completed": "ok", "running": "running", "cancelled": "warn", "failed": "bad"}


def _runs_table() -> str:
    runs = run_store.list_runs()
    if not runs:
        return ui.empty("No runs recorded yet. Start one from the Optimizer tab.")
    rows = []
    for run in runs:
        base = run.get("base_score", 0.0) or 0.0
        best = run.get("best_score", 0.0) or 0.0
        delta = best - base
        rows.append([
            ui.cell(run.get("run_id", "?"), mono=True),
            ui.cell(run.get("started_at", "").replace("T", " ")),
            f'<span class="pt-badge {"pass" if run.get("status") == "completed" else "fail"}">'
            f'{ui.cell(run.get("status", "?"))}</span>',
            f"{base:.3f}",
            f"{best:.3f}",
            f"{delta:+.3f}",
            str(run.get("iterations", 0)),
            f"{run.get('duration_seconds', 0):.0f}s",
        ])
    return ui.table(
        ["Run", "Started", "Status", "Baseline", "Best", "Delta", "Iterations", "Duration"],
        rows, numeric_columns={3, 4, 5, 6, 7},
    )


def _run_choices() -> list[str]:
    return [run.get("run_id", "") for run in run_store.list_runs() if run.get("run_id")]


def _refresh():
    choices = _run_choices()
    return _runs_table(), gr.update(choices=choices, value=choices[0] if choices else None)


def _load(run_id: str):
    empty = ui.empty("Select a run to inspect it.")
    if not run_id:
        return empty, empty, "", empty

    payload = run_store.load_run(run_id)
    summary = payload.get("summary") or {}
    history = payload.get("history") or []
    events = payload.get("events") or []
    pass_threshold = float(load_config().get("run", {}).get("pass_threshold", 0.95))

    base = summary.get("base_score", 0.0) or 0.0
    best = summary.get("best_score", 0.0) or 0.0
    stats = ui.stats([
        ui.stat("Status", summary.get("status", "?"),
                tone=STATUS_TONE.get(summary.get("status", ""), "")),
        ui.stat("Baseline", f"{base:.3f}"),
        ui.stat("Best", f"{best:.3f}", tone="ok" if best > base else ""),
        ui.stat("Improvement", f"{best - base:+.3f}", tone="ok" if best > base else ""),
        ui.stat("Test cases", str(summary.get("test_cases", 0))),
        ui.stat("API calls", str(summary.get("api_calls", 0))),
        ui.stat("Duration", f"{summary.get('duration_seconds', 0):.0f}s"),
        ui.stat("Model", summary.get("model", "—")),
    ])

    rows, best_rows = [], set()
    running_best = -1.0
    for index, generation in enumerate(history):
        score = generation.get("score", 0.0)
        outputs = generation.get("outputs", [])
        passed = sum(1 for item in outputs if item.get("score", 0) >= pass_threshold)
        if score > running_best:
            running_best = score
            best_rows.add(index)
        rows.append([
            ui.cell(generation.get("label", "?")),
            f"{score:.3f}",
            f"{passed}/{len(outputs)}",
            ui.cell((generation.get("prompt", "")[:80] + "…") if len(generation.get("prompt", "")) > 80
                    else generation.get("prompt", "")),
        ])
    generations = ui.table(["Generation", "Score", "Passed", "Prompt preview"], rows,
                           numeric_columns={1, 2}, highlight_rows=best_rows)

    log_rows = [
        [ui.cell(event.get("timestamp", "").split("T")[-1]),
         ui.cell(event.get("level", "")),
         ui.cell(event.get("message", ""))]
        for event in events if event.get("level") != "DEBUG"
    ]
    log_table = ui.table(["Time", "Level", "Event"], log_rows)

    return stats, generations, summary.get("best_prompt", ""), log_table


def build() -> None:
    choices = _run_choices()

    with gr.Row():
        gr.Markdown("### Past runs")
        refresh_btn = gr.Button("Refresh", scale=0)
    runs_html = gr.HTML(_runs_table())

    run_selector = gr.Dropdown(label="Inspect a run", choices=choices,
                               value=choices[0] if choices else None)
    detail_stats = gr.HTML(ui.empty("Select a run to inspect it."))

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Generations")
            detail_generations = gr.HTML(ui.empty("—"))
        with gr.Column():
            gr.Markdown("### Best prompt from this run")
            detail_prompt = gr.Textbox(lines=10, show_label=False)

    with gr.Accordion("Run log", open=False):
        detail_log = gr.HTML(ui.empty("—"))

    refresh_btn.click(_refresh, outputs=[runs_html, run_selector])
    run_selector.change(_load, inputs=run_selector,
                        outputs=[detail_stats, detail_generations, detail_prompt, detail_log])
