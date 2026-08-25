"""Test suite: inspect and hand-edit the cases the optimizer scores against."""
import gradio as gr

from app.core import config as config_store
from app.core.config import load_config, restore_archive, list_archive
from app.ui import components as ui

COLUMNS = ["id", "language", "input", "expected"]


def _language_choices() -> list[str]:
    stats = config_store.test_case_stats()
    return ["all"] + list(stats["by_language"].keys())


def _rows(language: str = "all") -> list[list[str]]:
    cases = config_store.list_test_cases()
    if language and language != "all":
        cases = [case for case in cases if case.get("language") == language]
    return [[case.get(column, "") for column in COLUMNS] for case in cases]


def _overview() -> str:
    config = load_config(refresh=True)
    stats = config_store.test_case_stats()
    weights = config.get("weights", {})
    top_languages = ", ".join(f"{code} ({count})" for code, count in
                              sorted(stats["by_language"].items(), key=lambda item: -item[1])[:6])
    return ui.stats([
        ui.stat("Test cases", str(stats["total"]), top_languages),
        ui.stat("Languages", str(len(stats["by_language"]))),
        ui.stat("Weights", f"{weights.get('language', 0):.2f} / {weights.get('filler', 0):.2f} / {weights.get('judge', 0):.2f}",
                "language / filler / judge"),
    ])


def _save(rows) -> tuple[str, str]:
    records = rows.values.tolist() if hasattr(rows, "values") else list(rows or [])
    cases, seen = [], set()
    for row in records:
        values = [str(value).strip() if value is not None else "" for value in list(row)[:4]]
        while len(values) < 4:
            values.append("")
        case_id, language, input_text, expected = values
        if not case_id or not input_text:
            continue
        if case_id in seen:
            return ui.status("Duplicate id", f"'{case_id}' appears more than once.", "bad"), _overview()
        seen.add(case_id)
        cases.append({"id": case_id, "language": language or "en",
                      "input": input_text, "expected": expected})

    if not cases:
        return ui.status("Nothing saved", "The suite would be empty — add at least one case.", "bad"), _overview()

    config_store.update_config({"test_cases": cases})
    return (
        ui.status("Test suite saved", f"{len(cases)} cases written; the previous version is archived.", "ok"),
        _overview(),
    )


def _archive_table() -> str:
    entries = list_archive()
    if not entries:
        return ui.empty("No archived configurations yet.")
    return ui.table(
        ["Snapshot", "Saved", "Size"],
        [[ui.cell(entry["name"], mono=True), ui.cell(entry["saved_at"]), f"{entry['size_kb']} KB"]
         for entry in entries],
        numeric_columns={2},
    )


def _restore(name: str, prompt_unused=None):
    if not name:
        return ui.status("Pick a snapshot", "", "warn"), gr.update(), _overview(), gr.update()
    try:
        config = restore_archive(name)
    except Exception as exc:
        return ui.status("Restore failed", str(exc), "bad"), gr.update(), _overview(), gr.update()
    return (
        ui.status("Configuration restored", f"Loaded {name}", "ok"),
        _rows("all"),
        _overview(),
        config["base_prompt"],
    )


def build(prompt_box: gr.Textbox) -> None:
    gr.Markdown("### Test suite")
    gr.Markdown(
        "These are the cases every prompt is scored against. Edit inline, or ask the Assistant to "
        "add and curate them for you.",
        elem_classes=["pt-hint"],
    )
    overview_html = gr.HTML(_overview())

    with gr.Row():
        language_filter = gr.Dropdown(label="Filter by language", choices=_language_choices(),
                                      value="all", scale=1)
        gr.Markdown("")

    table = gr.Dataframe(
        headers=["id", "language", "input", "expected"],
        datatype=["str", "str", "str", "str"],
        value=_rows(),
        interactive=True,
        wrap=True,
        column_widths=["12%", "10%", "39%", "39%"],
        max_height=520,
        label="Cases",
        show_search="filter",
    )
    with gr.Row():
        save_btn = gr.Button("Save test suite", variant="primary")
        reload_btn = gr.Button("Discard changes")
    save_status = gr.HTML("")

    with gr.Accordion("Configuration snapshots", open=False):
        gr.Markdown(
            "Every save archives the previous configuration, so an assistant edit is always reversible.",
            elem_classes=["pt-hint"],
        )
        archive_html = gr.HTML(_archive_table())
        with gr.Row():
            archive_selector = gr.Dropdown(
                label="Snapshot",
                choices=[entry["name"] for entry in list_archive()],
                scale=3,
            )
            refresh_archive_btn = gr.Button("Refresh", scale=1)
            restore_btn = gr.Button("Restore", variant="stop", scale=1)
        restore_status = gr.HTML("")

    language_filter.change(lambda language: _rows(language), inputs=language_filter, outputs=table)
    save_btn.click(_save, inputs=table, outputs=[save_status, overview_html])
    reload_btn.click(lambda language: _rows(language), inputs=language_filter, outputs=table)
    refresh_archive_btn.click(
        lambda: (_archive_table(), gr.update(choices=[entry["name"] for entry in list_archive()])),
        outputs=[archive_html, archive_selector],
    )
    restore_btn.click(
        _restore,
        inputs=archive_selector,
        outputs=[restore_status, table, overview_html, prompt_box],
    )
