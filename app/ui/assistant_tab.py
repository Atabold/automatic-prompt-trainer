"""Assistant: manage prompts, criteria, weights, and test cases by talking to the LLM."""
import gradio as gr

from app.assistant import agent
from app.core.config import load_config
from app.llm import client
from app.ui import components as ui

SUGGESTIONS = [
    "Show me a summary of the current configuration.",
    "Make the evaluation criteria stricter about punctuation and capitalization.",
    "Add three Italian test cases about scheduling a meeting, with realistic filler words.",
    "Weight the judge score at 60% and language at 25%.",
    "List the German test cases and remove any that are duplicates.",
    "Reduce the test suite to a 30-case sample per evaluation so runs are cheaper.",
]


def _actions_panel(actions: list[str]) -> str:
    if not actions:
        return ""
    items = "".join(f'<span class="pt-pill ok"><span class="pt-dot"></span>{action}</span>'
                    for action in actions)
    return f'<div class="pt-pills">{items}</div>'


def _respond(message: str, history: list[dict]):
    message = (message or "").strip()
    if not message:
        return history, "", "", gr.update()

    history = list(history) + [{"role": "user", "content": message}]
    try:
        reply, actions, changed = agent.respond(message, history[:-1])
    except client.ProviderNotConfigured as exc:
        history.append({"role": "assistant", "content": f"⚠️ {exc}"})
        return history, "", ui.status("Provider not configured", str(exc), "bad"), gr.update()
    except Exception as exc:
        history.append({"role": "assistant", "content": f"⚠️ {type(exc).__name__}: {exc}"})
        return history, "", ui.status("Assistant error", str(exc), "bad"), gr.update()

    history.append({"role": "assistant", "content": reply})
    status = (
        ui.status("Configuration updated", f"{len(actions)} change(s) applied and archived", "ok")
        if changed else ui.status("No changes made", "The assistant only read the configuration.", "")
    )
    return history, "", status + _actions_panel(actions), _config_snapshot()


def _config_snapshot() -> str:
    config = load_config(refresh=True)
    weights = config.get("weights", {})
    from app.core.config import test_case_stats

    stats = test_case_stats()
    languages = ", ".join(f"{code} {count}" for code, count in list(stats["by_language"].items())[:8])
    return ui.stats([
        ui.stat("Test cases", str(stats["total"]), languages),
        ui.stat("Language weight", f"{weights.get('language', 0):.2f}"),
        ui.stat("Filler weight", f"{weights.get('filler', 0):.2f}"),
        ui.stat("Judge weight", f"{weights.get('judge', 0):.2f}"),
    ])


def build(prompt_box: gr.Textbox) -> None:
    gr.Markdown("### Configuration assistant")
    gr.Markdown(
        "Ask in plain language. The assistant edits the configuration through tools, so changes are "
        "surgical, validated, and archived — it never rewrites the whole file blindly.",
        elem_classes=["pt-hint"],
    )

    snapshot_html = gr.HTML(_config_snapshot())

    chatbot = gr.Chatbot(
        label="Assistant",
        height=440,
        placeholder="Ask for a configuration change…",
        buttons=["copy", "copy_all"],
    )
    with gr.Row():
        message_box = gr.Textbox(placeholder="e.g. add five Spanish test cases about email dictation",
                                 show_label=False, scale=5)
        send_btn = gr.Button("Send", variant="primary", scale=1)
    action_status = gr.HTML("")
    gr.Examples(examples=[[item] for item in SUGGESTIONS], inputs=[message_box], label="Try one of these")

    def _sync_prompt():
        return load_config(refresh=True)["base_prompt"]

    for trigger in (send_btn.click, message_box.submit):
        trigger(
            _respond,
            inputs=[message_box, chatbot],
            outputs=[chatbot, message_box, action_status, snapshot_html],
        ).then(_sync_prompt, outputs=prompt_box)
