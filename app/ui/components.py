"""HTML fragment builders shared by the tabs — the app's visual vocabulary."""
from html import escape


def _tone_for(score: float, threshold: float) -> str:
    if score >= threshold:
        return "ok"
    if score >= threshold * 0.8:
        return "warn"
    return "bad"


def pill(label: str, value: str, tone: str = "") -> str:
    tone_class = f" {tone}" if tone else ""
    return (
        f'<span class="pt-pill{tone_class}"><span class="pt-dot"></span>'
        f"{escape(label)} <b>{escape(value)}</b></span>"
    )


def pills(items: list[str]) -> str:
    return f'<div class="pt-pills">{"".join(items)}</div>'


def stat(label: str, value: str, note: str = "", tone: str = "") -> str:
    tone_class = f" {tone}" if tone else ""
    note_html = f'<div class="pt-stat-note">{escape(note)}</div>' if note else ""
    return (
        f'<div class="pt-stat{tone_class}">'
        f'<div class="pt-stat-label">{escape(label)}</div>'
        f'<div class="pt-stat-value">{escape(value)}</div>'
        f"{note_html}</div>"
    )


def stats(tiles: list[str]) -> str:
    return f'<div class="pt-stats">{"".join(tiles)}</div>'


def status(title: str, detail: str = "", tone: str = "", fraction: float | None = None) -> str:
    spinner = '<div class="pt-spinner"></div>' if tone == "running" else ""
    bar = ""
    if fraction is not None:
        bar = f'<div class="pt-bar"><span style="width:{max(0.0, min(fraction, 1.0)) * 100:.1f}%"></span></div>'
    detail_html = f'<span class="pt-status-detail">{escape(detail)}</span>' if detail else ""
    return (
        f'<div><div class="pt-status {tone}">{spinner}'
        f'<span class="pt-status-title">{escape(title)}</span>{detail_html}</div>{bar}</div>'
    )


def empty(message: str) -> str:
    return f'<div class="pt-empty">{escape(message)}</div>'


def field(label: str, value: str, extra_class: str = "") -> str:
    return (
        f'<div class="pt-field {extra_class}">'
        f'<div class="pt-field-label">{escape(label)}</div>'
        f"<pre>{escape(value or '—')}</pre></div>"
    )


def test_card(result: dict, threshold: float, open_by_default: bool = False) -> str:
    score = result.get("score", 0.0)
    passed = score >= threshold
    state = "pass" if passed else "fail"
    preview = (result.get("output") or result.get("error") or "").replace("\n", " ")[:70]

    subscores = "".join(
        f"<span>{escape(name)} {value:.2f}</span>"
        for name, value in (
            ("language", result.get("language_score", 0.0)),
            ("filler", result.get("filler_score", 0.0)),
            ("judge", result.get("judge_score", 0.0)),
        )
    )
    detected = result.get("detected_language")
    if detected:
        subscores += f"<span>detected {escape(str(detected))}</span>"

    body = [
        field("Input", result.get("input", "")),
        field("Expected", result.get("expected", "")),
        field("Model output", result.get("output", "") or result.get("error", ""), "out"),
        f'<div class="pt-subscores">{subscores}</div>',
    ]

    return (
        f'<details class="pt-card {state}"{" open" if open_by_default else ""}>'
        f'<summary><span class="pt-badge {state}">{state}</span>'
        f'<span class="pt-id">{escape(str(result.get("id", "?")))}</span>'
        f'<span class="pt-preview">{escape(preview)}</span>'
        f'<span class="pt-score">{score:.3f}</span></summary>'
        f'<div class="pt-body">{"".join(body)}</div></details>'
    )


def test_cards(results: list[dict], threshold: float, failures_first: bool = True) -> str:
    if not results:
        return empty("No test results yet.")
    ordered = sorted(results, key=lambda item: item.get("score", 0.0)) if failures_first else results
    cards = [test_card(item, threshold, open_by_default=index == 0 and item.get("score", 0) < threshold)
             for index, item in enumerate(ordered)]
    return f'<div class="pt-cards">{"".join(cards)}</div>'


def results_summary(results: list[dict], threshold: float) -> str:
    if not results:
        return empty("Run the optimizer to see results here.")
    passed = sum(1 for item in results if item.get("score", 0.0) >= threshold)
    average = sum(item.get("score", 0.0) for item in results) / len(results)
    worst = min(results, key=lambda item: item.get("score", 0.0))
    return stats([
        stat("Average score", f"{average:.3f}", tone=_tone_for(average, threshold)),
        stat("Passed", f"{passed}/{len(results)}", f"threshold {threshold:.2f}",
             tone="ok" if passed == len(results) else ""),
        stat("Weakest case", str(worst.get("id", "—")), f"score {worst.get('score', 0.0):.3f}", tone="bad"),
    ])


def table(headers: list[str], rows: list[list[str]], numeric_columns: set[int] | None = None,
          highlight_rows: set[int] | None = None) -> str:
    if not rows:
        return empty("Nothing to show yet.")
    numeric_columns = numeric_columns or set()
    highlight_rows = highlight_rows or set()

    head = "".join(f"<th>{escape(header)}</th>" for header in headers)
    body = []
    for index, row in enumerate(rows):
        cells = "".join(
            f'<td class="{"num" if column in numeric_columns else ""}">{cell}</td>'
            for column, cell in enumerate(row)
        )
        body.append(f'<tr class="{"best" if index in highlight_rows else ""}">{cells}</tr>')
    return f'<table class="pt-table"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table>'


def cell(text: str, mono: bool = False) -> str:
    escaped = escape(str(text))
    return f'<span class="pt-mono">{escaped}</span>' if mono else escaped
