"""Optimizer: run the improvement loop with live progress and a working stop button."""
import threading
import time
from dataclasses import asdict

import gradio as gr

from app.core.config import load_config, update_config
from app.core.logging_setup import get_logger
from app.optimizer import optimizer
from app.ui import components as ui

logger = get_logger(__name__)

_active_cancel: optimizer.CancelToken | None = None

PHASE_TONE = {
    "loading": "running", "baseline": "running", "generating": "running", "evaluating": "running",
    "done": "ok", "cancelled": "warn", "failed": "bad",
}
PHASE_TITLE = {
    "starting": "Starting…", "loading": "Loading model", "baseline": "Scoring baseline",
    "generating": "Generating candidates", "evaluating": "Evaluating candidate",
    "done": "Run complete", "cancelled": "Run cancelled", "failed": "Run failed",
}


def _live_stats(snapshot: dict) -> str:
    base = snapshot.get("base_score", 0.0)
    best = snapshot.get("best_score", 0.0)
    delta = best - base
    return ui.stats([
        ui.stat("Baseline", f"{base:.3f}" if base else "—"),
        ui.stat("Best so far", f"{best:.3f}" if best else "—",
                tone="ok" if delta > 0 else ""),
        ui.stat("Improvement", f"{delta:+.3f}" if base else "—",
                tone="ok" if delta > 0 else ("bad" if delta < 0 else "")),
        ui.stat("Iteration", f"{snapshot.get('iteration', 0)}/{snapshot.get('max_iterations', 0)}"),
        ui.stat("Judge calls", str(snapshot.get("api_calls", 0)), "optimizer requests"),
        ui.stat("Elapsed", f"{snapshot.get('elapsed', 0.0):.0f}s"),
    ])


def _live_status(snapshot: dict) -> str:
    phase = snapshot.get("phase", "idle")
    title = PHASE_TITLE.get(phase, phase.title())
    detail = snapshot.get("message", "")
    if phase in {"baseline", "evaluating"} and snapshot.get("test_total"):
        detail = f"{detail}  ·  test {snapshot['test_index']}/{snapshot['test_total']}"
    return ui.status(title, detail, PHASE_TONE.get(phase, ""), snapshot.get("fraction"))


def _generation_rows(history: list[dict], pass_threshold: float) -> tuple[str, list[str], set[int]]:
    rows, labels, best_rows = [], [], set()
    running_best = -1.0
    for index, generation in enumerate(history):
        outputs = generation.get("outputs", [])
        passed = sum(1 for item in outputs if item.get("score", 0) >= pass_threshold)
        score = generation.get("score", 0.0)
        if score > running_best:
            running_best = score
            best_rows.add(index)
        label = generation.get("label", f"Iteration {generation.get('iteration', 0)}")
        labels.append(f"{label} — {score:.3f}")
        rows.append([
            ui.cell(label),
            f"{score:.3f}",
            f"{passed}/{len(outputs)}",
            ui.cell((generation.get("prompt", "")[:90] + "…") if len(generation.get("prompt", "")) > 90
                    else generation.get("prompt", "")),
        ])
    return (
        ui.table(["Generation", "Score", "Passed", "Prompt preview"], rows,
                 numeric_columns={1, 2}, highlight_rows=best_rows),
        labels,
        best_rows,
    )


def _run(base_prompt: str, max_iterations, threshold, candidates, sample_size, pass_threshold):
    global _active_cancel

    update_config({"run": {
        "max_iterations": int(max_iterations), "threshold": float(threshold),
        "candidates_per_iteration": int(candidates), "sample_size": int(sample_size),
        "pass_threshold": float(pass_threshold),
    }})

    cancel = optimizer.CancelToken()
    _active_cancel = cancel
    snapshot: dict = {"phase": "starting", "message": "Preparing run…", "fraction": 0.0}
    box: dict = {}

    def on_progress(progress: optimizer.Progress) -> None:
        snapshot.update(asdict(progress))

    def worker() -> None:
        box["result"] = optimizer.run_optimization(
            on_progress=on_progress, cancel=cancel, base_prompt=base_prompt or None
        )

    thread = threading.Thread(target=worker, name="optimizer-run", daemon=True)
    thread.start()

    idle_table = ui.empty("Generations appear here once the run finishes.")
    while thread.is_alive():
        yield (
            _live_status(snapshot), _live_stats(snapshot), idle_table,
            gr.update(), gr.update(), gr.update(), gr.update(interactive=True), gr.update(),
        )
        time.sleep(0.6)
    thread.join()
    _active_cancel = None

    result = box.get("result")
    if result is None:
        yield (
            ui.status("Run failed", "The run thread ended unexpectedly. Check logs/app.log.", "bad"),
            _live_stats(snapshot), idle_table, gr.update(), gr.update(), gr.update(),
            gr.update(interactive=False), [],
        )
        return

    table_html, labels, _ = _generation_rows(result.history, float(pass_threshold))
    best_index = max(range(len(result.history)),
                     key=lambda i: result.history[i].get("score", 0.0)) if result.history else 0
    selected_label = labels[best_index] if labels else None
    cards = ui.test_cards(result.history[best_index].get("outputs", []),
                          float(pass_threshold)) if result.history else ui.empty("No results.")

    final_status = {
        "completed": ui.status("Run complete",
                               f"Best {result.best_score:.3f} (baseline {result.base_score:.3f}) · run {result.run_id}",
                               "ok", 1.0),
        "cancelled": ui.status("Run cancelled",
                               f"Kept best {result.best_score:.3f} · run {result.run_id}", "warn"),
        "failed": ui.status("Run failed", result.error or "See logs/app.log", "bad"),
    }.get(result.status, ui.status(result.status, "", ""))

    yield (
        final_status,
        _live_stats({**snapshot, "best_score": result.best_score, "base_score": result.base_score}),
        table_html,
        gr.update(choices=labels, value=selected_label),
        cards,
        result.best_prompt,
        gr.update(interactive=False),
        result.history,
    )


def _stop():
    if _active_cancel is not None:
        _active_cancel.cancel()
        return ui.status("Stopping…", "Finishing the current test case, then shutting down cleanly.", "warn")
    return ui.status("Nothing running", "", "")


def _select_generation(label: str, history: list[dict], pass_threshold: float):
    if not label or not history:
        return ui.empty("Select a generation to inspect its test results.")
    for index, generation in enumerate(history):
        candidate_label = f"{generation.get('label', '')} — {generation.get('score', 0.0):.3f}"
        if candidate_label == label:
            return ui.test_cards(generation.get("outputs", []), float(pass_threshold))
    return ui.empty("Generation not found.")


def _apply_prompt(prompt: str):
    if not (prompt or "").strip():
        return ui.status("Nothing to apply", "Run the optimizer first.", "warn"), gr.update()
    update_config({"base_prompt": prompt})
    return ui.status("Base prompt updated", "The winning prompt is now the default.", "ok"), prompt


def build(prompt_box: gr.Textbox) -> None:
    config = load_config()
    run_config = config.get("run", {})

    with gr.Accordion("Run settings", open=False):
        with gr.Row():
            max_iterations = gr.Slider(label="Iterations", minimum=1, maximum=15, step=1,
                                       value=run_config.get("max_iterations", 5),
                                       info="Improvement rounds")
            candidates = gr.Slider(label="Candidates per iteration", minimum=1, maximum=5, step=1,
                                   value=run_config.get("candidates_per_iteration", 2))
        with gr.Row():
            threshold = gr.Slider(label="Stop at score", minimum=0.5, maximum=1.0, step=0.01,
                                  value=run_config.get("threshold", 0.95),
                                  info="Finish early once the average reaches this")
            pass_threshold = gr.Slider(label="Pass threshold", minimum=0.5, maximum=1.0, step=0.01,
                                       value=run_config.get("pass_threshold", 0.95),
                                       info="A test counts as passed at or above this score")
        sample_size = gr.Slider(
            label="Test cases per evaluation", minimum=0,
            maximum=max(len(config.get("test_cases", [])), 10), step=5,
            value=run_config.get("sample_size", 0),
            info="0 = use every case. A smaller random sample makes runs much cheaper and faster.",
        )

    with gr.Row():
        run_btn = gr.Button("Run optimization", variant="primary", scale=3)
        stop_btn = gr.Button("Stop", variant="stop", scale=1, interactive=False)

    status_html = gr.HTML(ui.status("Idle", "Configure the run, then start.", ""))
    stats_html = gr.HTML(_live_stats({}))

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Generations")
            generations_html = gr.HTML(ui.empty("No run yet."))
            generation_selector = gr.Dropdown(label="Inspect a generation", choices=[])
        with gr.Column(scale=1):
            gr.Markdown("### Test results")
            gr.Markdown("Lowest scores first, so failures are the first thing you see.",
                        elem_classes=["pt-hint"])
            cards_html = gr.HTML(ui.empty("Run the optimizer to see per-test results."))

    gr.Markdown("### Winning prompt")
    best_prompt_box = gr.Textbox(label="Best prompt", lines=10, show_label=False)
    with gr.Row():
        apply_btn = gr.Button("Apply as base prompt", variant="primary")
    apply_status = gr.HTML("")

    history_state = gr.State([])

    run_event = run_btn.click(
        lambda: (gr.update(interactive=False), gr.update(interactive=True)),
        outputs=[run_btn, stop_btn],
    ).then(
        _run,
        inputs=[prompt_box, max_iterations, threshold, candidates, sample_size, pass_threshold],
        outputs=[status_html, stats_html, generations_html, generation_selector,
                 cards_html, best_prompt_box, stop_btn, history_state],
    ).then(
        lambda: gr.update(interactive=True), outputs=run_btn,
    )

    stop_btn.click(_stop, outputs=status_html)

    generation_selector.change(
        _select_generation,
        inputs=[generation_selector, history_state, pass_threshold],
        outputs=cards_html,
    )
    apply_btn.click(_apply_prompt, inputs=best_prompt_box, outputs=[apply_status, prompt_box])
