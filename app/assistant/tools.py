"""Tools the configuration assistant can call.

Function calling replaces the old "ask the model to emit the whole config as JSON" trick:
edits are surgical, failures are explicit, and every change is reported back to the user.
"""
import json

from app.core import config as config_store
from app.core.logging_setup import get_logger

logger = get_logger(__name__)


def _get_config_summary() -> dict:
    config = config_store.load_config(refresh=True)
    stats = config_store.test_case_stats()
    return {
        "base_prompt": config["base_prompt"],
        "evaluation_criteria": config["evaluation_criteria"],
        "weights": config["weights"],
        "test_case_count": stats["total"],
        "test_cases_by_language": stats["by_language"],
        "run_settings": config.get("run", {}),
    }


def _update_base_prompt(prompt: str) -> dict:
    config_store.update_config({"base_prompt": prompt})
    return {"ok": True, "base_prompt": prompt}


def _update_evaluation_criteria(criteria: str) -> dict:
    config_store.update_config({"evaluation_criteria": criteria})
    return {"ok": True, "evaluation_criteria": criteria}


def _update_weights(language: float, filler: float, judge: float) -> dict:
    total = language + filler + judge
    if total <= 0:
        raise ValueError("Weights must add up to more than zero.")
    weights = {
        "language": round(language / total, 4),
        "filler": round(filler / total, 4),
        "judge": round(judge / total, 4),
    }
    config_store.update_config({"weights": weights})
    return {"ok": True, "weights": weights, "note": "Normalized to sum to 1.0."}


def _update_run_settings(max_iterations=None, threshold=None, candidates_per_iteration=None,
                         sample_size=None) -> dict:
    changes = {
        key: value
        for key, value in {
            "max_iterations": max_iterations,
            "threshold": threshold,
            "candidates_per_iteration": candidates_per_iteration,
            "sample_size": sample_size,
        }.items()
        if value is not None
    }
    if not changes:
        raise ValueError("No run settings supplied.")
    config_store.update_config({"run": changes})
    return {"ok": True, "run": config_store.load_config()["run"]}


def _list_test_cases(language: str = "", limit: int = 20) -> dict:
    cases = config_store.list_test_cases()
    if language:
        cases = [case for case in cases if case.get("language") == language]
    return {
        "total_matching": len(cases),
        "showing": min(limit, len(cases)),
        "test_cases": cases[:limit],
    }


def _add_test_case(id: str, language: str, input: str, expected: str) -> dict:
    return {"ok": True, "added": config_store.add_test_case(id, language, input, expected)}


def _update_test_case(id: str, language: str = None, input: str = None, expected: str = None) -> dict:
    return {"ok": True, "updated": config_store.update_test_case(
        id, language=language, input=input, expected=expected)}


def _remove_test_case(id: str) -> dict:
    return {"ok": True, "removed": config_store.remove_test_case(id)}


HANDLERS = {
    "get_config_summary": _get_config_summary,
    "update_base_prompt": _update_base_prompt,
    "update_evaluation_criteria": _update_evaluation_criteria,
    "update_weights": _update_weights,
    "update_run_settings": _update_run_settings,
    "list_test_cases": _list_test_cases,
    "add_test_case": _add_test_case,
    "update_test_case": _update_test_case,
    "remove_test_case": _remove_test_case,
}

# Short, human-readable confirmations shown in the "Changes applied" panel.
SUMMARIES = {
    "get_config_summary": lambda args, out: "Read the current configuration",
    "list_test_cases": lambda args, out: f"Listed {out.get('showing', 0)} of {out.get('total_matching', 0)} test cases",
    "update_base_prompt": lambda args, out: "Updated the base prompt",
    "update_evaluation_criteria": lambda args, out: "Updated the evaluation criteria",
    "update_weights": lambda args, out: f"Updated weights to {out.get('weights')}",
    "update_run_settings": lambda args, out: f"Updated run settings: {', '.join(args)}",
    "add_test_case": lambda args, out: f"Added test case {out.get('added', {}).get('id')}",
    "update_test_case": lambda args, out: f"Updated test case {out.get('updated', {}).get('id')}",
    "remove_test_case": lambda args, out: f"Removed test case {out.get('removed')}",
}

READ_ONLY = {"get_config_summary", "list_test_cases"}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_config_summary",
            "description": "Read the current configuration: base prompt, evaluation criteria, scoring weights, run settings, and test-case counts by language. Call this before making changes so you know what you are editing.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_base_prompt",
            "description": "Replace the formatter's system prompt with a new full prompt.",
            "parameters": {
                "type": "object",
                "properties": {"prompt": {"type": "string", "description": "The complete new system prompt."}},
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_evaluation_criteria",
            "description": "Replace the criteria text the judge model uses to score outputs.",
            "parameters": {
                "type": "object",
                "properties": {"criteria": {"type": "string", "description": "The complete new criteria text."}},
                "required": ["criteria"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_weights",
            "description": "Set the scoring weights. They are normalized to sum to 1.0 automatically.",
            "parameters": {
                "type": "object",
                "properties": {
                    "language": {"type": "number", "description": "Weight for language preservation."},
                    "filler": {"type": "number", "description": "Weight for filler-word removal."},
                    "judge": {"type": "number", "description": "Weight for the LLM judge score."},
                },
                "required": ["language", "filler", "judge"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_run_settings",
            "description": "Change optimizer run settings. Only supply the fields you want to change.",
            "parameters": {
                "type": "object",
                "properties": {
                    "max_iterations": {"type": "integer", "description": "How many improvement rounds to run."},
                    "threshold": {"type": "number", "description": "Stop early once the score reaches this (0-1)."},
                    "candidates_per_iteration": {"type": "integer", "description": "Prompt variations per round."},
                    "sample_size": {"type": "integer", "description": "Random subset of test cases per evaluation; 0 means use all."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_test_cases",
            "description": "List existing test cases, optionally filtered by language code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "language": {"type": "string", "description": "Language code such as en, it, fr. Omit for all."},
                    "limit": {"type": "integer", "description": "Maximum number to return (default 20)."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_test_case",
            "description": "Add one new test case to the suite.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Unique id, e.g. en_161."},
                    "language": {"type": "string", "description": "Language code, e.g. en."},
                    "input": {"type": "string", "description": "Raw spoken-style input text."},
                    "expected": {"type": "string", "description": "The correctly formatted output."},
                },
                "required": ["id", "language", "input", "expected"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_test_case",
            "description": "Modify fields of an existing test case. Only supply what changes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Id of the test case to change."},
                    "language": {"type": "string"},
                    "input": {"type": "string"},
                    "expected": {"type": "string"},
                },
                "required": ["id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remove_test_case",
            "description": "Delete a test case by id.",
            "parameters": {
                "type": "object",
                "properties": {"id": {"type": "string", "description": "Id of the test case to delete."}},
                "required": ["id"],
            },
        },
    },
]


def execute(name: str, arguments: str) -> tuple[str, str, bool]:
    """Run one tool call. Returns (json_result_for_model, human_summary, changed_config)."""
    handler = HANDLERS.get(name)
    if handler is None:
        return json.dumps({"error": f"Unknown tool '{name}'."}), f"Unknown tool '{name}'", False

    try:
        parsed = json.loads(arguments) if arguments else {}
    except json.JSONDecodeError as exc:
        return json.dumps({"error": f"Malformed arguments: {exc}"}), f"{name}: bad arguments", False

    try:
        output = handler(**parsed)
    except TypeError as exc:
        return json.dumps({"error": f"Bad arguments for {name}: {exc}"}), f"{name}: bad arguments", False
    except Exception as exc:
        logger.warning("Tool %s failed: %s", name, exc)
        return json.dumps({"error": str(exc)}), f"{name} failed: {exc}", False

    summary = SUMMARIES.get(name, lambda args, out: name)(list(parsed.keys()), output)
    return json.dumps(output, ensure_ascii=False, default=str), summary, name not in READ_ONLY
