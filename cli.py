"""Headless entry points.

    python cli.py optimize [--iterations N] [--sample N]
    python cli.py download <repo-id-or-url> [--file NAME]
    python cli.py models
    python cli.py runs
"""
import argparse
import sys

from dotenv import load_dotenv

load_dotenv()

from app.core.config import load_config, update_config
from app.core.logging_setup import configure_logging, get_logger
from app.core.paths import ensure_dirs

logger = get_logger(__name__)


def cmd_optimize(args: argparse.Namespace) -> int:
    from app.optimizer import optimizer

    changes = {}
    if args.iterations is not None:
        changes["max_iterations"] = args.iterations
    if args.sample is not None:
        changes["sample_size"] = args.sample
    if changes:
        update_config({"run": changes})

    last_message = ""

    def on_progress(progress: optimizer.Progress) -> None:
        nonlocal last_message
        line = f"[{progress.fraction * 100:5.1f}%] {progress.phase}: {progress.message}"
        if line != last_message:
            print(line, flush=True)
            last_message = line

    result = optimizer.run_optimization(on_progress=on_progress)
    print(f"\nStatus:   {result.status}")
    if result.error:
        print(f"Error:    {result.error}")
        return 1
    print(f"Run id:   {result.run_id}")
    print(f"Baseline: {result.base_score:.3f}")
    print(f"Best:     {result.best_score:.3f}  ({result.best_score - result.base_score:+.3f})")
    print(f"\nBest prompt:\n{result.best_prompt}")
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    from app.hf import downloader

    ref, files = downloader.list_gguf_files(args.reference)
    print(f"{ref.repo_id}: {len(files)} GGUF file(s)")
    for item in files:
        print(f"  {item['quant']:<8} {item['size']:>10}  {item['filename']}")

    filename = args.file or ref.filename or downloader.recommend(files)
    if not filename:
        print("No file to download.")
        return 1

    sizes = {item["filename"]: item["size_bytes"] for item in files}
    print(f"\nDownloading {filename}…")
    job = downloader.start_download(ref.repo_id, filename, ref.revision, sizes.get(filename))
    for update in downloader.poll(job, interval=2.0):
        if update.status == "downloading" and update.total_bytes:
            print(f"  {update.fraction * 100:5.1f}%  "
                  f"{update.downloaded_bytes / 1024**3:.2f}/{update.total_bytes / 1024**3:.2f} GB",
                  flush=True)
    print(job.message)
    return 0 if job.status == "completed" else 1


def cmd_models(_: argparse.Namespace) -> int:
    from app.hf import loader, registry

    active = loader.active_model_path()
    models = registry.list_models()
    if not models:
        print("No models downloaded. Try: python cli.py download https://huggingface.co/google/gemma-4-E2B-it")
        return 0
    for item in models:
        marker = "*" if active and str(active) == item["path"] else " "
        print(f"{marker} {item['filename']:<50} {item['size_gb']:>6.2f} GB  {item['repo_id']}")
    return 0


def cmd_runs(_: argparse.Namespace) -> int:
    from app.optimizer import run_store

    runs = run_store.list_runs()
    if not runs:
        print("No runs recorded yet.")
        return 0
    for run in runs:
        print(f"{run.get('run_id', '?'):<22} {run.get('status', '?'):<10} "
              f"base {run.get('base_score', 0):.3f} -> best {run.get('best_score', 0):.3f}  "
              f"({run.get('duration_seconds', 0):.0f}s)")
    return 0


def main() -> int:
    configure_logging()
    ensure_dirs()
    load_config()

    parser = argparse.ArgumentParser(prog="cli.py", description="Prompt Trainer command line")
    sub = parser.add_subparsers(dest="command", required=True)

    optimize = sub.add_parser("optimize", help="run one optimization pass")
    optimize.add_argument("--iterations", type=int)
    optimize.add_argument("--sample", type=int, help="test cases per evaluation; 0 uses all")
    optimize.set_defaults(func=cmd_optimize)

    download = sub.add_parser("download", help="download a GGUF model from Hugging Face")
    download.add_argument("reference", help="owner/repo, a hub URL, or a direct .gguf link")
    download.add_argument("--file", help="exact .gguf filename to fetch")
    download.set_defaults(func=cmd_download)

    sub.add_parser("models", help="list downloaded models").set_defaults(func=cmd_models)
    sub.add_parser("runs", help="list past optimization runs").set_defaults(func=cmd_runs)

    args = parser.parse_args()
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except Exception as exc:
        logger.exception("Command failed")
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
