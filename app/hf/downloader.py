"""Download any GGUF model from Hugging Face by pasting a repo id or a URL."""
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote, urlparse

from huggingface_hub import HfApi, hf_hub_download

from app.core import secrets_store
from app.core.logging_setup import get_logger
from app.core.paths import WEIGHTS_DIR, ensure_dirs
from app.hf import registry

logger = get_logger(__name__)

_REPO_ID = re.compile(r"^[A-Za-z0-9][\w.\-]*/[\w.\-]+$")


@dataclass
class RepoRef:
    repo_id: str
    filename: str | None = None
    revision: str = "main"


def parse_reference(text: str) -> RepoRef:
    """Accepts 'owner/repo', a hub URL, or a direct blob/resolve link to one file."""
    text = (text or "").strip().strip("\"'")
    if not text:
        raise ValueError("Paste a Hugging Face repo id or URL first.")

    if _REPO_ID.match(text):
        return RepoRef(repo_id=text)

    if "://" not in text:
        text = f"https://{text}"

    parsed = urlparse(text)
    if "huggingface.co" not in parsed.netloc and "hf.co" not in parsed.netloc:
        raise ValueError(f"'{parsed.netloc or text}' is not a Hugging Face URL.")

    parts = [unquote(part) for part in parsed.path.strip("/").split("/") if part]
    if parts and parts[0] in {"models", "spaces", "datasets"}:
        parts = parts[1:]
    if len(parts) < 2:
        raise ValueError("That URL does not contain an owner/repo pair.")

    repo_id = f"{parts[0]}/{parts[1]}"
    rest = parts[2:]
    revision, filename = "main", None
    if rest and rest[0] in {"blob", "resolve", "tree"} and len(rest) >= 2:
        revision = rest[1]
        remainder = "/".join(rest[2:])
        if remainder.endswith(".gguf"):
            filename = remainder
    return RepoRef(repo_id=repo_id, filename=filename, revision=revision)


def _human_size(num_bytes: int | None) -> str:
    if not num_bytes:
        return "—"
    gigabytes = num_bytes / 1024**3
    if gigabytes >= 1:
        return f"{gigabytes:.2f} GB"
    return f"{num_bytes / 1024**2:.0f} MB"


def list_gguf_files(reference: str) -> tuple[RepoRef, list[dict]]:
    """Return every .gguf file in the repo, largest metadata we can get, cheapest call first."""
    ref = parse_reference(reference)
    api = HfApi(token=secrets_store.get_hf_token() or None)

    sizes: dict[str, int | None] = {}
    try:
        for item in api.list_repo_tree(ref.repo_id, revision=ref.revision, recursive=True):
            path = getattr(item, "path", "")
            if path.endswith(".gguf"):
                sizes[path] = getattr(item, "size", None)
    except Exception as exc:
        logger.info("list_repo_tree failed for %s (%s); falling back to list_repo_files.", ref.repo_id, exc)
        for path in api.list_repo_files(ref.repo_id, revision=ref.revision):
            if path.endswith(".gguf"):
                sizes[path] = None

    if not sizes:
        raise ValueError(
            f"No .gguf files found in {ref.repo_id}. This app runs GGUF (llama.cpp) models — "
            "look for a repo whose name usually ends in -GGUF."
        )

    files = [
        {
            "filename": name,
            "quant": _quant_label(name),
            "size_bytes": size,
            "size": _human_size(size),
            "downloaded": (WEIGHTS_DIR / name).exists(),
        }
        for name, size in sorted(sizes.items())
    ]
    logger.info("Found %d GGUF files in %s", len(files), ref.repo_id)
    return ref, files


_QUANT = re.compile(r"(IQ\d[\w]*|Q\d_[\w]+|Q\d|BF16|F16|F32)", re.IGNORECASE)


def _quant_label(filename: str) -> str:
    match = _QUANT.search(Path(filename).stem)
    return match.group(0).upper() if match else "—"


def recommend(files: list[dict]) -> str | None:
    """Q4_K_M is the usual quality/size sweet spot; fall back sensibly."""
    for preference in ("Q4_K_M", "Q4_K_S", "Q5_K_M", "Q8_0"):
        for item in files:
            if preference in item["filename"].upper():
                return item["filename"]
    return files[0]["filename"] if files else None


@dataclass
class DownloadJob:
    repo_id: str
    filename: str
    revision: str = "main"
    total_bytes: int | None = None
    downloaded_bytes: int = 0
    status: str = "pending"
    message: str = ""
    local_path: str = ""
    _thread: threading.Thread | None = field(default=None, repr=False)

    @property
    def fraction(self) -> float:
        if not self.total_bytes:
            return 0.0
        return min(self.downloaded_bytes / self.total_bytes, 1.0)

    @property
    def done(self) -> bool:
        return self.status in {"completed", "failed"}


def _measure(destination: Path, filename: str) -> int:
    """Bytes on disk for this download, including huggingface_hub's .incomplete staging files."""
    final = destination / filename
    if final.exists():
        return final.stat().st_size
    staging = destination / ".cache" / "huggingface" / "download"
    if staging.exists():
        return sum(path.stat().st_size for path in staging.rglob("*.incomplete") if path.is_file())
    return 0


def start_download(repo_id: str, filename: str, revision: str = "main", total_bytes: int | None = None) -> DownloadJob:
    ensure_dirs()
    job = DownloadJob(repo_id=repo_id, filename=filename, revision=revision, total_bytes=total_bytes)

    def worker() -> None:
        job.status = "downloading"
        job.message = f"Downloading {filename} from {repo_id}…"
        logger.info("Starting download %s :: %s", repo_id, filename)
        try:
            path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                local_dir=str(WEIGHTS_DIR),
                token=secrets_store.get_hf_token() or None,
            )
            job.local_path = path
            job.downloaded_bytes = Path(path).stat().st_size
            job.total_bytes = job.total_bytes or job.downloaded_bytes
            registry.record(repo_id, filename, path)
            job.status = "completed"
            job.message = f"Downloaded to {path}"
            logger.info("Download complete: %s", path)
        except Exception as exc:
            job.status = "failed"
            job.message = f"{type(exc).__name__}: {exc}"
            logger.exception("Download failed for %s :: %s", repo_id, filename)

    thread = threading.Thread(target=worker, name=f"hf-download-{filename}", daemon=True)
    job._thread = thread
    thread.start()
    return job


def poll(job: DownloadJob, interval: float = 1.0):
    """Yield the job as it progresses, measuring bytes on disk between checks."""
    while not job.done:
        job.downloaded_bytes = max(job.downloaded_bytes, _measure(WEIGHTS_DIR, job.filename))
        yield job
        time.sleep(interval)
    yield job
