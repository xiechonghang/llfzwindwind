#!/usr/bin/env python3
"""Download Hugging Face model metadata and code, with optional weights.

Features:
  - Mirror presets (default: OpenCSG; also hf-mirror, hf official)
  - Optional weight download (--include-weights / -w)
  - List mode: read a text file with one repo_id per line (--list-file)
  - Resumable downloads via HTTP Range
  - Concurrent downloads (--workers)
  - Per-request timeout (--timeout)
  - Retry with exponential backoff (--retries)
  - Per-file progress bar (--no-progress to disable)
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath


# Edit this value when you want to run the script without command-line arguments.
DEFAULT_MODEL_ID = "Qwen/Qwen3.8-Flash-Next"

# Preset mirrors. Key = alias used by --mirror, value = base URL.
MIRRORS = {
    "opencsg":   "https://hub.opencsg.com/hf",
    "hf-mirror": "https://hf-mirror.com",
    "hf":        "https://huggingface.co",
}

# Default mirror alias. Override with --mirror or --endpoint.
DEFAULT_MIRROR = "opencsg"
DEFAULT_ENDPOINT = MIRRORS[DEFAULT_MIRROR]

DEFAULT_WORKERS = 4
DEFAULT_TIMEOUT = 60          # seconds, per request
DEFAULT_RETRIES = 3           # total attempts
RETRY_BACKOFF_BASE = 2.0      # seconds; sleep = base ** attempt
CHUNK_SIZE = 1024 * 1024      # 1 MiB

WEIGHT_SUFFIXES = {
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
    ".h5",
    ".msgpack",
    ".onnx",
    ".gguf",
    ".tflite",
    ".ot",
    ".pb",
    ".pkl",
    ".pickle",
    ".npy",
    ".npz",
}

# A small lock so concurrent prints don't interleave mid-line.
_PRINT_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def normalize_endpoint(endpoint: str) -> str:
    endpoint = endpoint.strip().rstrip("/")
    if not endpoint.startswith(("http://", "https://")):
        raise ValueError(f"endpoint must start with http:// or https://: {endpoint}")
    return endpoint


def resolve_endpoint(args: argparse.Namespace) -> str:
    """Resolve the effective endpoint.

    Priority: --endpoint > --mirror > HF_ENDPOINT env > DEFAULT_MIRROR.
    """
    if args.endpoint:
        return normalize_endpoint(args.endpoint)
    if args.mirror:
        return normalize_endpoint(MIRRORS[args.mirror])
    env_endpoint = os.environ.get("HF_ENDPOINT")
    if env_endpoint:
        return normalize_endpoint(env_endpoint)
    return normalize_endpoint(DEFAULT_ENDPOINT)


def auth_headers(token: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token else {}


def is_weight(path: str) -> bool:
    name = PurePosixPath(path).name.lower()
    return any(name.endswith(suffix) for suffix in WEIGHT_SUFFIXES)


def safe_destination(root: Path, relative_path: str) -> Path:
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe repository path: {relative_path}")
    return root.joinpath(*relative.parts)


def split_repo_id(repo_id: str) -> tuple[str, str]:
    parts = repo_id.split("/")
    if len(parts) != 2 or not all(parts) or any(part in {".", ".."} for part in parts):
        raise ValueError("repo_id must have the form 'organization/model-name'")
    return parts[0], parts[1]


def build_file_url(endpoint: str, repo_id: str, revision: str, path: str) -> str:
    quoted_repo = "/".join(urllib.parse.quote(part, safe="") for part in repo_id.split("/"))
    quoted_revision = urllib.parse.quote(revision, safe="")
    quoted_path = "/".join(urllib.parse.quote(part, safe="") for part in PurePosixPath(path).parts)
    return f"{endpoint}/{quoted_repo}/resolve/{quoted_revision}/{quoted_path}?download=true"


def human_size(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:6.1f}{unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:6.1f}PB"


def human_speed(bps: float) -> str:
    return f"{human_size(bps)}/s"


# ---------------------------------------------------------------------------
# Retry-aware HTTP open
# ---------------------------------------------------------------------------

class RetryExhausted(Exception):
    pass


def open_with_retry(
    request: urllib.request.Request,
    *,
    timeout: float,
    retries: int,
    label: str,
):
    """Open a request with retries on transient failures. Caller must close."""
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as error:
            # 4xx (except 408/429) are not retryable.
            if error.code < 500 and error.code not in (408, 429):
                raise
            last_error = error
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as error:
            last_error = error

        if attempt < retries:
            delay = RETRY_BACKOFF_BASE ** attempt
            with _PRINT_LOCK:
                print(
                    f"  ! retry {attempt}/{retries - 1} for {label} after error: {last_error} "
                    f"(sleep {delay:.1f}s)",
                    file=sys.stderr,
                )
            time.sleep(delay)

    raise RetryExhausted(f"{label}: giving up after {retries} attempts: {last_error}")


def request_json(url: str, token: str | None, timeout: float, retries: int) -> dict:
    request = urllib.request.Request(url, headers=auth_headers(token))
    with open_with_retry(request, timeout=timeout, retries=retries, label=f"GET {url}") as response:
        return json.load(response)


def _probe_remote_size(url: str, token: str | None, timeout: float, retries: int) -> int | None:
    request = urllib.request.Request(url, method="HEAD", headers=auth_headers(token))
    try:
        with open_with_retry(request, timeout=timeout, retries=retries, label=f"HEAD {url}") as response:
            length = response.headers.get("Content-Length")
            return int(length) if length is not None else None
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError, RetryExhausted):
        return None


# ---------------------------------------------------------------------------
# Progress bar
# ---------------------------------------------------------------------------

class ProgressBar:
    """A single-line progress bar that redraws in place. Thread-safe via a lock."""

    def __init__(self, label: str, total: int | None, *, enabled: bool = True) -> None:
        self.label = label
        self.total = total
        self.enabled = enabled
        self.start = time.monotonic()
        self._last_render = 0.0
        self._last_len = 0
        if self.enabled and self.total is not None:
            self.render(0)

    def update(self, done: int, *, force: bool = False) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if not force and now - self._last_render < 0.1:
            return
        self._last_render = now
        self.render(done)

    def render(self, done: int) -> None:
        elapsed = max(time.monotonic() - self.start, 1e-6)
        speed = done / elapsed
        if self.total is None or self.total <= 0:
            line = f"  {self.label}: {human_size(done)} @ {human_speed(speed)}"
        else:
            ratio = min(done / self.total, 1.0)
            bar_width = 24
            filled = int(ratio * bar_width)
            bar = "█" * filled + "░" * (bar_width - filled)
            line = (
                f"  {self.label}: [{bar}] {ratio * 100:5.1f}%  "
                f"{human_size(done)}/{human_size(self.total)} @ {human_speed(speed)}"
            )
        pad = max(self._last_len - len(line), 0)
        with _PRINT_LOCK:
            sys.stderr.write("\r" + line + " " * pad)
            sys.stderr.flush()
        self._last_len = len(line)

    def finish(self, done: int) -> None:
        if not self.enabled:
            return
        self.render(done)
        with _PRINT_LOCK:
            sys.stderr.write("\n")
            sys.stderr.flush()


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_file(
    endpoint: str,
    repo_id: str,
    revision: str,
    path: str,
    destination: Path,
    token: str | None,
    *,
    timeout: float,
    retries: int,
    show_progress: bool,
) -> str:
    """Download one file with resume + retry + progress. Returns status string."""
    url = build_file_url(endpoint, repo_id, revision, path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")

    remote_size = _probe_remote_size(url, token, timeout, retries)

    # Skip if already complete and matches remote size.
    if destination.exists() and remote_size is not None and destination.stat().st_size == remote_size:
        return "skipped"

    # If local full file is the wrong size, delete and redownload.
    if destination.exists() and remote_size is not None and destination.stat().st_size != remote_size:
        destination.unlink(missing_ok=True)

    resume_from = temporary.stat().st_size if temporary.exists() else 0
    if remote_size is not None and resume_from > remote_size:
        temporary.unlink(missing_ok=True)
        resume_from = 0

    headers = auth_headers(token)
    if resume_from > 0:
        headers["Range"] = f"bytes={resume_from}-"

    request = urllib.request.Request(url, headers=headers)

    with open_with_retry(request, timeout=timeout, retries=retries, label=f"GET {path}") as response:
        status = response.status
        if resume_from > 0 and status == 206:
            mode = "ab"
            written_start = resume_from
        else:
            mode = "wb"
            written_start = 0

        content_length = response.headers.get("Content-Length")
        if content_length is not None:
            try:
                remaining = int(content_length)
            except ValueError:
                remaining = None
        else:
            remaining = None

        if written_start == 0 and remaining is not None:
            total_size: int | None = remaining
        elif remote_size is not None:
            total_size = remote_size
        elif remaining is not None:
            total_size = written_start + remaining
        else:
            total_size = None

        bar = ProgressBar(path, total_size, enabled=show_progress)
        bar.update(written_start, force=True)

        with temporary.open(mode) as output:
            done = written_start
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                output.write(chunk)
                done += len(chunk)
                bar.update(done)
            bar.finish(done)

    if remote_size is not None:
        actual = temporary.stat().st_size
        if actual != remote_size:
            raise OSError(
                f"size mismatch after download: {path} got {actual}, expected {remote_size}"
            )

    temporary.replace(destination)
    return "resumed" if written_start > 0 else "downloaded"


def _download_one(
    endpoint: str,
    repo_id: str,
    revision: str,
    path: str,
    model_dir: Path,
    token: str | None,
    index: int,
    total: int,
    *,
    timeout: float,
    retries: int,
    show_progress: bool,
) -> tuple[str, str]:
    try:
        result = download_file(
            endpoint, repo_id, revision, path, safe_destination(model_dir, path), token,
            timeout=timeout, retries=retries, show_progress=show_progress,
        )
        with _PRINT_LOCK:
            print(f"[{index}/{total}] {result:10s} {path}")
        return path, result
    except Exception as error:  # noqa: BLE001
        with _PRINT_LOCK:
            print(f"[{index}/{total}] FAILED     {path}: {error}", file=sys.stderr)
        return path, f"error: {error}"


# ---------------------------------------------------------------------------
# Repo-level processing
# ---------------------------------------------------------------------------

def process_repo(
    repo_id: str,
    *,
    endpoint: str,
    revision: str,
    output_root: Path,
    token: str | None,
    include_weights: bool,
    workers: int,
    timeout: float,
    retries: int,
    show_progress: bool,
) -> tuple[int, int]:
    """Download one repo. Returns (ok_count, fail_count)."""
    organization, model_name = split_repo_id(repo_id)
    encoded_repo = "/".join(urllib.parse.quote(part, safe="") for part in repo_id.split("/"))
    encoded_revision = urllib.parse.quote(revision, safe="")
    api_url = (
        f"{endpoint}/api/models/{encoded_repo}/revision/{encoded_revision}"
        "?expand[]=siblings&expand[]=sha"
    )

    metadata = request_json(api_url, token, timeout, retries)
    resolved_revision = metadata.get("sha") or revision
    files = [item["rfilename"] for item in metadata.get("siblings", [])]

    if include_weights:
        selected = list(files)
        excluded_weights: list[str] = []
    else:
        selected = [p for p in files if not is_weight(p)]
        excluded_weights = [p for p in files if is_weight(p)]

    model_dir = output_root / organization / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    total = len(selected)
    results: dict[str, str] = {}

    if total == 0:
        print(f"[{repo_id}] no files to download (siblings list is empty).", file=sys.stderr)
    else:
        common_kwargs = dict(timeout=timeout, retries=retries, show_progress=show_progress)
        if workers == 1:
            for index, path in enumerate(selected, start=1):
                p, r = _download_one(
                    endpoint, repo_id, resolved_revision, path, model_dir, token,
                    index, total, **common_kwargs,
                )
                results[p] = r
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(
                        _download_one,
                        endpoint, repo_id, resolved_revision, path, model_dir, token,
                        index, total, **common_kwargs,
                    )
                    for index, path in enumerate(selected, start=1)
                ]
                for future in concurrent.futures.as_completed(futures):
                    p, r = future.result()
                    results[p] = r

    failed = {p: r for p, r in results.items() if r.startswith("error:")}
    succeeded = {p: r for p, r in results.items() if not r.startswith("error:")}

    manifest = {
        "repo_id": repo_id,
        "organization": organization,
        "model_name": model_name,
        "revision": resolved_revision,
        "endpoint": endpoint,
        "include_weights": include_weights,
        "workers": workers,
        "timeout": timeout,
        "retries": retries,
        "downloaded_files": sorted(succeeded.keys()),
        "download_results": succeeded,
        "failed_files": failed,
        "excluded_weight_files": excluded_weights,
    }
    (model_dir / "metadata_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(
        f"[{repo_id}] {len(succeeded)}/{total} files OK, {len(failed)} failed. "
        f"Output: {model_dir}"
    )
    return len(succeeded), len(failed)


def read_repo_list(list_file: Path) -> list[str]:
    """Read repo ids from a text file.

    Ignores blank lines and lines starting with '#'.
    """
    if not list_file.exists():
        raise ValueError(f"list file not found: {list_file}")
    repos: list[str] = []
    for raw in list_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        repos.append(line)
    if not repos:
        raise ValueError(f"list file is empty: {list_file}")
    return repos


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download model configs, tokenizer files, docs, and code; "
                    "weights are excluded unless --include-weights is given. "
                    "Use --list-file to batch-download multiple repos."
    )
    parser.add_argument(
        "repo_id",
        nargs="?",
        default=DEFAULT_MODEL_ID,
        help=f"Hugging Face repository (default: {DEFAULT_MODEL_ID}); ignored when --list-file is used",
    )
    parser.add_argument("--revision", default="main", help="branch, tag, or commit (default: main)")
    parser.add_argument("--output-root", type=Path, default=Path("models"))
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face token; defaults to HF_TOKEN (only needed for gated/private repos)",
    )
    parser.add_argument(
        "--mirror",
        choices=sorted(MIRRORS.keys()),
        default=None,
        help=(
            "Preset mirror alias; "
            f"choices: {', '.join(sorted(MIRRORS.keys()))} "
            f"(default: {DEFAULT_MIRROR}). "
            "Overrides --endpoint if both are given."
        ),
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        help=(
            "Explicit Hugging Face-compatible endpoint URL. "
            "If omitted, uses --mirror, then HF_ENDPOINT, then "
            f"{DEFAULT_MIRROR} ({DEFAULT_ENDPOINT})."
        ),
    )
    parser.add_argument(
        "-w", "--include-weights",
        action="store_true",
        help="Also download weight files (default: skip all weights in single-repo mode)",
    )
    parser.add_argument(
        "--list-file",
        type=Path,
        default=None,
        help=(
            "Path to a text file with one repo_id per line. "
            "In list mode, weights are downloaded by default "
            "unless --no-weights is given."
        ),
    )
    parser.add_argument(
        "--no-weights",
        action="store_true",
        help="In list mode, skip weights (overrides the default include-weights behavior).",
    )
    parser.add_argument(
        "-j", "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of concurrent download workers (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"Per-request timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRIES,
        help=f"Max attempts per request, including the first (default: {DEFAULT_RETRIES})",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable per-file progress bars",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.workers < 1:
        print("--workers must be >= 1", file=sys.stderr)
        return 2
    if args.retries < 1:
        print("--retries must be >= 1", file=sys.stderr)
        return 2
    if args.timeout <= 0:
        print("--timeout must be > 0", file=sys.stderr)
        return 2

    try:
        endpoint = resolve_endpoint(args)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2

    show_progress = not args.no_progress

    # ----- list mode -----
    if args.list_file is not None:
        try:
            repos = read_repo_list(args.list_file)
        except ValueError as error:
            print(str(error), file=sys.stderr)
            return 2

        # In list mode, weights are ON by default; --no-weights turns them off.
        include_weights = not args.no_weights
        print(
            f"List mode: {len(repos)} repos from {args.list_file} "
            f"(include_weights={include_weights}, endpoint={endpoint})"
        )

        total_ok = 0
        total_fail = 0
        failed_repos: list[str] = []
        for idx, repo_id in enumerate(repos, start=1):
            print(f"\n=== [{idx}/{len(repos)}] {repo_id} ===")
            try:
                ok, fail = process_repo(
                    repo_id,
                    endpoint=endpoint,
                    revision=args.revision,
                    output_root=args.output_root,
                    token=args.token,
                    include_weights=include_weights,
                    workers=args.workers,
                    timeout=args.timeout,
                    retries=args.retries,
                    show_progress=show_progress,
                )
                total_ok += ok
                total_fail += fail
                if fail > 0:
                    failed_repos.append(repo_id)
            except (OSError, ValueError, urllib.error.HTTPError, RetryExhausted) as error:
                print(f"[{repo_id}] download failed: {error}", file=sys.stderr)
                failed_repos.append(repo_id)

        print(
            f"\nAll done. {total_ok} files OK, {total_fail} files failed "
            f"across {len(repos)} repos."
        )
        if failed_repos:
            print("Repos with failures:", file=sys.stderr)
            for r in failed_repos:
                print(f"  - {r}", file=sys.stderr)
            return 1
        return 0

    # ----- single-repo mode -----
    try:
        ok, fail = process_repo(
            args.repo_id,
            endpoint=endpoint,
            revision=args.revision,
            output_root=args.output_root,
            token=args.token,
            include_weights=args.include_weights,
            workers=args.workers,
            timeout=args.timeout,
            retries=args.retries,
            show_progress=show_progress,
        )
    except (OSError, ValueError, urllib.error.HTTPError, RetryExhausted) as error:
        print(f"download failed: {error}", file=sys.stderr)
        return 1

    print(f"\nDone. endpoint: {endpoint}")
    return 1 if fail > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
