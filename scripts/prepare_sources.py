#!/usr/bin/env python3
"""Prepare the current main branches of the core Hugging Face source checkouts.

In a normal Git clone this initializes and updates the two submodules from their
configured remote branches. It also supports source archives, where submodule
metadata is unavailable, by cloning the branches described in
``sources/sources.json``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CONFIG = ROOT / "sources" / "sources.json"
SOURCE_NAMES = ("transformers", "diffusers")


def run(command: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def output(command: list[str], *, cwd: Path | None = None) -> str:
    return subprocess.check_output(command, cwd=cwd, text=True).strip()


def is_git_checkout() -> bool:
    try:
        return output(["git", "rev-parse", "--show-toplevel"], cwd=ROOT) == str(ROOT)
    except (OSError, subprocess.CalledProcessError):
        return False


def verify_source(name: str, spec: dict[str, str]) -> None:
    path = ROOT / spec["path"]
    check_path = path / spec["check_path"]
    if not check_path.is_dir():
        raise RuntimeError(f"{name} source is incomplete: missing {check_path}")
    try:
        revision = output(["git", "-C", str(path), "rev-parse", "HEAD"])
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"{name} source is not a Git checkout: {path}") from exc
    print(f"ready: {name} ({revision[:12]}) -> {path}")


def prepare_from_submodules(specs: dict[str, dict[str, str]]) -> None:
    paths = [specs[name]["path"] for name in SOURCE_NAMES]
    run(["git", "submodule", "sync", "--", *paths], cwd=ROOT)
    run(
        ["git", "submodule", "update", "--init", "--remote", "--recursive", "--", *paths],
        cwd=ROOT,
    )


def prepare_from_archive(specs: dict[str, dict[str, str]]) -> None:
    if shutil.which("git") is None:
        raise RuntimeError("git is required to prepare the source repositories")
    for name in SOURCE_NAMES:
        spec = specs[name]
        path = ROOT / spec["path"]
        if path.exists():
            if not (path / ".git").exists():
                raise RuntimeError(
                    f"{path} already exists but is not a Git checkout; remove it or choose another directory"
                )
            run(["git", "-C", str(path), "fetch", "origin", spec["branch"]])
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            run(
                [
                    "git",
                    "clone",
                    "--filter=blob:none",
                    "--branch",
                    spec["branch"],
                    spec["url"],
                    str(path),
                ]
            )
        run(["git", "-C", str(path), "checkout", "--detach", f"origin/{spec['branch']}"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive-mode",
        action="store_true",
        help="force direct clone/update mode, useful for a GitHub ZIP archive",
    )
    args = parser.parse_args()

    if not SOURCE_CONFIG.is_file():
        print(f"missing source configuration: {SOURCE_CONFIG}", file=sys.stderr)
        return 2
    specs = json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))
    missing = [name for name in SOURCE_NAMES if name not in specs]
    if missing:
        print(f"source configuration is missing: {', '.join(missing)}", file=sys.stderr)
        return 2

    try:
        if is_git_checkout() and not args.archive_mode:
            prepare_from_submodules(specs)
        else:
            prepare_from_archive(specs)
        for name in SOURCE_NAMES:
            verify_source(name, specs[name])
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"prepare_sources failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
