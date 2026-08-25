"""Iteratively improves the formatter's system prompt against the test suite.

The run reports structured progress through a callback and honours a cancel flag, so the
UI can show exactly where it is and stop it cleanly instead of leaving a thread running.
"""
import json
import random
import threading
import time
from dataclasses import asdict, dataclass, field

from app.core.config import load_config
from app.core.logging_setup import get_logger
from app.evaluation import judge
from app.hf import loader
from app.llm import client
from app.optimizer.run_store import RunStore, new_run_id

logger = get_logger(__name__)

CANDIDATE_SYSTEM = "You are an expert prompt engineer for a text formatting model."


@dataclass
class Progress:
    run_id: str = ""
    phase: str = "idle"
    message: str = ""
    fraction: float = 0.0
    iteration: int = 0
    max_iterations: int = 0
    candidate: int = 0
    candidates_total: int = 0
    test_index: int = 0
    test_total: int = 0
    base_score: float = 0.0
    best_score: float = 0.0
    current_score: float = 0.0
    api_calls: int = 0
    elapsed: float = 0.0


@dataclass
class RunResult:
    run_id: str = ""
    status: str = "running"
    best_prompt: str = ""
    best_score: float = 0.0
    base_score: float = 0.0
    history: list[dict] = field(default_factory=list)
    error: str = ""


class CancelToken:
    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()


class Cancelled(Exception):
    pass


def _sample(cases: list[dict], size: int) -> list[dict]:
    if size and 0 < size < len(cases):
        return random.sample(cases, size)
    return cases


def generate_candidates(base_prompt: str, best_prompt: str, best_score: float,
                        count: int, store: RunStore) -> list[str]:
    user = f"""Current best prompt:
{best_prompt}

Best score: {best_score:.3f}

Propose {count} improved prompt variations. They must be meaningfully different from each other and better at:
- preserving the input language exactly
- resolving self-corrections (keep the last correction)
- removing filler words and false starts
- fixing punctuation, capitalization, and formatting of lists/paragraphs/emails

Return ONLY a JSON array of strings, e.g. ["prompt one", "prompt two"]."""

    response, target = client.chat(
        "optimizer",
        [
            {"role": "system", "content": CANDIDATE_SYSTEM},
            {"role": "user", "content": user},
        ],
        temperature=0.8,
    )
    store.count_api_call("candidate_generation", target.model)

    content = (response.choices[0].message.content or "").strip()
    content = content.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        parsed = json.loads(content)
        if isinstance(parsed, list) and parsed:
            return [str(item) for item in parsed[:count]]
    except json.JSONDecodeError:
        logger.info("Candidate response was not JSON; falling back to line splitting.")

    lines = [line.strip().strip('"') for line in content.splitlines() if len(line.strip()) > 30]
    return lines[:count] or [base_prompt]


def evaluate_prompt(model, prompt: str, cases: list[dict], on_progress, progress: Progress,
                    cancel: CancelToken) -> tuple[float, list[dict]]:
    outputs = []
    total = len(cases)
    progress.test_total = total

    for index, case in enumerate(cases, start=1):
        if cancel.cancelled:
            raise Cancelled()
        progress.test_index = index
        progress.message = f"Evaluating test {index}/{total} · {case.get('id', '?')}"
        on_progress(progress)

        try:
            output = loader.format_text(model, prompt, case["input"])
            error = ""
        except Exception as exc:
            output, error = "", f"{type(exc).__name__}: {exc}"
            logger.warning("Formatting failed for %s: %s", case.get("id"), error)

        breakdown = judge.score_case(case, output) if not error else {
            "score": 0.0, "language_score": 0.0, "filler_score": 0.0,
            "judge_score": 0.0, "detected_language": "n/a",
        }
        outputs.append({
            "id": case.get("id", f"case_{index}"),
            "language": case.get("language", ""),
            "input": case["input"],
            "expected": case["expected"],
            "output": output,
            "error": error,
            **breakdown,
        })

    average = sum(item["score"] for item in outputs) / len(outputs) if outputs else 0.0
    return average, outputs


def run_optimization(on_progress=None, cancel: CancelToken | None = None,
                     base_prompt: str | None = None) -> RunResult:
    config = load_config(refresh=True)
    run_config = config.get("run", {})
    max_iterations = int(run_config.get("max_iterations", 5))
    threshold = float(run_config.get("threshold", 0.95))
    candidates_per_iteration = int(run_config.get("candidates_per_iteration", 2))

    cancel = cancel or CancelToken()
    on_progress = on_progress or (lambda _p: None)

    run_id = new_run_id()
    store = RunStore(run_id)
    result = RunResult(run_id=run_id)
    started = time.time()

    progress = Progress(run_id=run_id, phase="starting", max_iterations=max_iterations,
                        candidates_total=candidates_per_iteration)

    def report(phase: str | None = None, fraction: float | None = None, message: str | None = None) -> None:
        if phase:
            progress.phase = phase
        if fraction is not None:
            progress.fraction = min(max(fraction, 0.0), 1.0)
        if message:
            progress.message = message
        progress.api_calls = store.api_calls
        progress.elapsed = time.time() - started
        on_progress(progress)

    cases = _sample(config.get("test_cases", []), int(run_config.get("sample_size", 0)))
    if not cases:
        result.status = "failed"
        result.error = "No test cases configured. Add some in the Test Suite tab."
        store.event("ERROR", result.error)
        store.write_summary({"status": "failed", "error": result.error})
        report("failed", 1.0, result.error)
        return result

    best_prompt = base_prompt or config["base_prompt"]
    history: list[dict] = []

    try:
        report("loading", 0.01, "Loading the local formatter model…")
        model = loader.load_model()
        store.event("INFO", f"Model loaded: {loader.loaded_model_name()}")
        store.event("INFO", f"Run configured: {max_iterations} iterations, "
                            f"{candidates_per_iteration} candidates each, {len(cases)} test cases")

        report("baseline", 0.03, "Scoring the baseline prompt…")
        progress.iteration = 0
        base_score, base_outputs = evaluate_prompt(model, best_prompt, cases, on_progress, progress, cancel)
        best_score = base_score
        progress.base_score = base_score
        progress.best_score = best_score
        progress.current_score = base_score

        history.append({
            "iteration": 0, "candidate": 0, "label": "Baseline", "prompt": best_prompt,
            "score": base_score, "outputs": base_outputs, "is_best": True,
        })
        store.event("INFO", f"Baseline score: {base_score:.3f}")
        store.write_history(history)
        store.write_summary({"status": "running", "best_score": best_score,
                             "base_score": base_score, "best_prompt": best_prompt,
                             "iterations": 0, "test_cases": len(cases)})

        total_steps = max(max_iterations * candidates_per_iteration, 1)
        completed_steps = 0

        for iteration in range(1, max_iterations + 1):
            if cancel.cancelled:
                raise Cancelled()
            if best_score >= threshold:
                store.event("INFO", f"Threshold {threshold:.2f} reached — stopping early.")
                break

            progress.iteration = iteration
            report("generating", 0.05 + 0.9 * completed_steps / total_steps,
                   f"Iteration {iteration}/{max_iterations} · asking the optimizer for new prompts…")
            candidates = generate_candidates(config["base_prompt"], best_prompt, best_score,
                                             candidates_per_iteration, store)
            store.event("INFO", f"Iteration {iteration}: generated {len(candidates)} candidates")

            for index, candidate in enumerate(candidates, start=1):
                if cancel.cancelled:
                    raise Cancelled()
                progress.candidate = index
                progress.candidates_total = len(candidates)
                report("evaluating", 0.05 + 0.9 * completed_steps / total_steps,
                       f"Iteration {iteration}/{max_iterations} · candidate {index}/{len(candidates)}")

                score, outputs = evaluate_prompt(model, candidate, cases, on_progress, progress, cancel)
                completed_steps += 1
                progress.current_score = score
                improved = score > best_score
                if improved:
                    store.event("INFO", f"New best: {best_score:.3f} -> {score:.3f}")
                    best_score, best_prompt = score, candidate
                    progress.best_score = best_score

                history.append({
                    "iteration": iteration, "candidate": index,
                    "label": f"Iteration {iteration} · candidate {index}",
                    "prompt": candidate, "score": score, "outputs": outputs, "is_best": improved,
                })
                store.write_history(history)
                store.write_summary({"status": "running", "best_score": best_score,
                                     "base_score": base_score, "best_prompt": best_prompt,
                                     "iterations": iteration, "test_cases": len(cases)})
                report("evaluating", 0.05 + 0.9 * completed_steps / total_steps)

            time.sleep(0.5)

        result.status = "completed"
        result.best_prompt = best_prompt
        result.best_score = best_score
        result.base_score = base_score
        result.history = history
        store.event("INFO", f"Run complete. Best score {best_score:.3f} (baseline {base_score:.3f})")
        report("done", 1.0, f"Finished — best score {best_score:.3f}")

    except Cancelled:
        result.status = "cancelled"
        result.best_prompt = best_prompt
        result.best_score = progress.best_score
        result.base_score = progress.base_score
        result.history = history
        store.event("WARN", "Run cancelled by the user.")
        report("cancelled", progress.fraction, "Run cancelled.")

    except client.ProviderNotConfigured as exc:
        result.status = "failed"
        result.error = str(exc)
        store.event("ERROR", result.error)
        report("failed", progress.fraction, result.error)

    except Exception as exc:
        result.status = "failed"
        result.error = f"{type(exc).__name__}: {exc}"
        logger.exception("Optimization run %s failed", run_id)
        store.event("ERROR", result.error)
        report("failed", progress.fraction, result.error)

    store.write_history(history)
    store.write_summary({
        "status": result.status,
        "error": result.error,
        "best_score": result.best_score,
        "base_score": result.base_score,
        "best_prompt": result.best_prompt,
        "iterations": progress.iteration,
        "test_cases": len(cases),
        "model": loader.loaded_model_name(),
        "cache": judge.judge_cache_stats(),
        "progress": asdict(progress),
    })
    return result
