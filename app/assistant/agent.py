"""Conversational configuration assistant driven by tool calling."""
from app.assistant import tools
from app.core.logging_setup import get_logger
from app.llm import client

logger = get_logger(__name__)

MAX_TOOL_ROUNDS = 6

SYSTEM_PROMPT = """You are the configuration assistant for a prompt-optimization workbench.

The workbench runs a local GGUF model that reformats messy spoken/voice-input text, and scores its
output against a suite of multilingual test cases using three signals: language preservation, filler
removal, and an LLM judge that follows the configured evaluation criteria.

You manage that configuration through the tools you have been given. Rules:
- Call get_config_summary first whenever you need to know the current state. Never guess it.
- Make surgical changes: change only what the user asked for, and keep everything else intact.
- When rewriting the base prompt or criteria, always send the COMPLETE new text, not a diff.
- When adding test cases, follow the existing id convention (language prefix + number, e.g. en_161)
  and call list_test_cases first if you need to see what ids are taken.
- If a request is ambiguous or destructive (e.g. "delete all the Italian cases"), ask before acting.
- After making changes, reply in plain language: say what changed and why it helps. Keep it short.
- Do not output raw JSON at the user; the tools handle persistence."""


def _to_openai_messages(history: list[dict]) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for item in history:
        role = item.get("role")
        content = item.get("content", "")
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})
    return messages


def respond(message: str, history: list[dict]) -> tuple[str, list[str], bool]:
    """Run the tool-calling loop. Returns (reply, actions_taken, config_changed)."""
    messages = _to_openai_messages(history)
    messages.append({"role": "user", "content": message})

    actions: list[str] = []
    changed = False

    for round_index in range(MAX_TOOL_ROUNDS):
        response, target = client.chat(
            "assistant",
            messages,
            tools=tools.TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=0.2,
        )
        choice = response.choices[0].message
        tool_calls = choice.tool_calls or []

        if not tool_calls:
            reply = (choice.content or "").strip() or "(no reply)"
            logger.info("Assistant replied after %d tool round(s) via %s", round_index, target.label)
            return reply, actions, changed

        messages.append({
            "role": "assistant",
            "content": choice.content or "",
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.function.name, "arguments": call.function.arguments},
                }
                for call in tool_calls
            ],
        })

        for call in tool_calls:
            result, summary, did_change = tools.execute(call.function.name, call.function.arguments)
            changed = changed or did_change
            if did_change or "fail" in summary.lower() or "error" in summary.lower():
                actions.append(summary)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

    return (
        "I ran out of tool steps before finishing. Try breaking the request into smaller pieces.",
        actions,
        changed,
    )
