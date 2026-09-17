from __future__ import annotations

import contextlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import tempfile

DEFAULT_MODEL = "Qwen/Qwen3.5-9B"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = PROJECT_ROOT / "models"


def repo_parts(repo: str) -> tuple[str, str]:
    # Keep the existing models/<organization>/<name> convention.
    parts = repo.split("/")
    if len(parts) != 2 or any(not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", p) for p in parts):
        raise ValueError("Use an organization/model repository ID")
    return parts[0], parts[1]


def safe_path(root: Path, relative: str) -> Path:
    p = PurePosixPath(relative)
    if not relative or p.is_absolute() or ".." in p.parts or "\\" in relative:
        raise ValueError(f"Unsafe repository path: {relative}")
    result = root.joinpath(*p.parts)
    if not result.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes directory: {relative}")
    return result


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".manifest-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextlib.contextmanager
def directory_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".hf-onnx.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        try:
            owner = lock.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError):
            owner = "unknown"
        raise RuntimeError(
            f"Model directory is locked (recorded PID: {owner}; lock: {lock}). "
            "The previous operation may still be running or suspended. "
            "Inspect it with ps; if suspended, resume it with fg in its original terminal. "
            "Remove the lock only after confirming the previous process has exited."
        ) from error
    try:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        yield
    finally:
        lock.unlink(missing_ok=True)


def environment() -> dict:
    packages = {}
    for name in ("huggingface-hub", "torch", "transformers", "optimum-onnx", "onnx", "onnxruntime", "onnxruntime-genai", "safetensors", "numpy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages}


def model_path(value: str, root: Path = DEFAULT_ROOT) -> Path:
    direct = Path(value).expanduser()
    if direct.is_dir():
        return direct.resolve()
    return root.joinpath(*repo_parts(value)).resolve()
