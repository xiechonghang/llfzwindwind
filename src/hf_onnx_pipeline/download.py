from __future__ import annotations

from pathlib import Path, PurePosixPath
import shutil

from .common import directory_lock, read_json, repo_parts, safe_path, write_json
from .graphs import external_files

MANIFEST = ".hf-onnx-download.json"
METADATA_SUFFIXES = {".json", ".txt", ".md", ".py", ".jinja", ".model", ".tiktoken", ".yaml", ".yml"}


def choose_files(files: list[str], mode: str, graph_names: list[str] | None = None) -> dict:
    graphs = sorted(f for f in files if f.endswith(".onnx"))
    if mode == "onnx" and not graphs:
        raise ValueError("The specified repository has no ONNX files")
    if mode in ("auto", "onnx") and graphs:
        selected = graph_names or graphs
        if graph_names is None and len(graphs) > 1:
            raise ValueError("Multiple ONNX graphs/variants found. Select the complete desired graph set with repeated --onnx-file: " + ", ".join(graphs))
        if any(g not in graphs for g in selected):
            raise ValueError("Selected ONNX file does not exist in this revision")
        metadata = [f for f in files if PurePosixPath(f).suffix.lower() in METADATA_SUFFIXES]
        return {"kind": "onnx", "graphs": sorted(set(selected)), "files": sorted(set(metadata + selected)),
                "external_data": "Resolved from graph references during download; bytes not included in initial estimate"}
    if graph_names:
        raise ValueError("--onnx-file requires an ONNX download")
    if mode == "metadata":
        return {"kind": "metadata", "graphs": [], "files": sorted(f for f in files if PurePosixPath(f).suffix.lower() in METADATA_SUFFIXES)}
    # Full original snapshot, including every weight shard and component.
    return {"kind": "original", "graphs": [], "files": sorted(files)}


def pinned_revision(destination: Path, repo: str, requested: str | None) -> str:
    prior = None
    for name in (MANIFEST, "metadata_manifest.json"):
        if (destination / name).is_file():
            prior = read_json(destination / name)
            break
    if prior:
        if prior.get("repo_id") != repo:
            raise ValueError("Destination belongs to another repository; use another --output-root")
        # An explicit branch is resolved later; exact mismatch is checked against its SHA.
        return requested or prior["revision"]
    return requested or "main"


def plan_download(repo: str, root: Path, mode: str = "auto", revision: str | None = None,
                  onnx_repo: str | None = None, graph_names: list[str] | None = None) -> dict:
    from huggingface_hub import HfApi

    model_dir = root.joinpath(*repo_parts(repo)).resolve()
    source = onnx_repo or repo
    repo_parts(source)
    if onnx_repo and mode not in ("auto", "onnx"):
        raise ValueError("--onnx-repo can only be used with --mode auto or onnx")
    # Alternate source is explicitly requested and has its own provenance.
    hint = model_dir / "onnx" / "downloaded" if onnx_repo else model_dir
    prior_onnx = model_dir / "onnx/downloaded" / MANIFEST
    if not onnx_repo and prior_onnx.is_file() and not (model_dir / MANIFEST).is_file():
        if read_json(prior_onnx).get("repo_id") == source:
            hint = model_dir / "onnx/downloaded"
    rev = pinned_revision(hint, source, revision)
    info = HfApi().model_info(source, revision=rev, files_metadata=True)
    if not info.sha:
        raise ValueError("Hub did not return an immutable revision")
    inventory = {s.rfilename: s.size for s in info.siblings}
    selection = choose_files(list(inventory), "onnx" if onnx_repo else mode, graph_names)
    destination = model_dir / "onnx/downloaded" if selection["kind"] == "onnx" else model_dir
    for legacy in (MANIFEST, "metadata_manifest.json"):
        previous = destination / legacy
        if previous.is_file():
            old = read_json(previous)
            if old.get("repo_id") != source or old.get("revision") != info.sha:
                raise ValueError("Refusing to mix repository revisions in an existing directory; use another --output-root")
    if (destination / "config.json").is_file() and not any((destination / n).is_file() for n in (MANIFEST, "metadata_manifest.json")):
        raise ValueError("Existing model has no revision manifest; use another --output-root to avoid mixing versions")
    for f in selection["files"]:
        safe_path(destination, f)
        if f in {MANIFEST, ".hf-onnx.lock", "metadata_manifest.json"}:
            raise ValueError(f"Repository uses reserved local manifest path: {f}")
    sizes = [inventory[f] for f in selection["files"]]
    return {"requested_repo_id": repo, "repo_id": source, "revision": info.sha,
            "destination": str(destination), **selection,
            "known_download_bytes": sum(s for s in sizes if s is not None),
            "unknown_size_files": sum(s is None for s in sizes),
            "sizes": {f: inventory[f] for f in selection["files"]}}


def check_weights(root: Path) -> None:
    weights = [p for p in root.rglob("*") if p.is_file() and "onnx" not in p.relative_to(root).parts
               and ".cache" not in p.relative_to(root).parts
               and (p.suffix == ".safetensors" or p.name.startswith(("pytorch_model", "diffusion_pytorch_model")) and p.suffix == ".bin")]
    if not weights:
        raise ValueError("No original PyTorch/safetensors weights found")
    for index in root.rglob("*.index.json"):
        if "onnx" in index.relative_to(root).parts or ".cache" in index.relative_to(root).parts:
            continue
        for shard in set(read_json(index).get("weight_map", {}).values()):
            if not safe_path(index.parent, shard).is_file():
                raise ValueError(f"Missing weight shard referenced by {index.name}: {shard}")


def execute_download(plan: dict, max_bytes: int | None = None) -> dict:
    from huggingface_hub import HfApi, snapshot_download

    destination = Path(plan["destination"])
    if max_bytes is not None and plan["known_download_bytes"] > max_bytes:
        raise ValueError("Download exceeds --max-gb; use plan to inspect the inventory")
    if max_bytes is not None and plan["unknown_size_files"]:
        raise ValueError("Cannot enforce --max-gb with unknown file sizes")
    if plan["known_download_bytes"] > shutil.disk_usage(destination.parent if destination.parent.exists() else Path.cwd()).free:
        raise ValueError("Insufficient disk space for the snapshot")
    with directory_lock(destination):
        # Repeat identity guard inside the lock, including a resumed failed transfer.
        for name in (MANIFEST, "metadata_manifest.json"):
            if (destination / name).is_file():
                previous = read_json(destination / name)
                if previous.get("revision") != plan["revision"] or previous.get("repo_id") != plan["repo_id"]:
                    raise ValueError("Destination revision changed after planning")
        state = {**plan, "status": "downloading"}
        write_json(destination / MANIFEST, state)
        try:
            def fetch(names):
                snapshot_download(repo_id=plan["repo_id"], revision=plan["revision"],
                                  local_dir=destination, allow_patterns=names)

            fetch(plan["files"])
            files = set(plan["files"])
            if plan["kind"] == "onnx":
                external = set()
                for graph in plan["graphs"]:
                    external.update(external_files(safe_path(destination, graph), destination))
                info = HfApi().model_info(plan["repo_id"], revision=plan["revision"], files_metadata=True)
                inventory = {s.rfilename: s.size for s in info.siblings}
                if external - inventory.keys():
                    raise ValueError(f"Missing external files in repository: {sorted(external - inventory.keys())}")
                for f in external:
                    safe_path(destination, f)
                    if f in {MANIFEST, ".hf-onnx.lock", "metadata_manifest.json"}:
                        raise ValueError(f"External data references reserved path: {f}")
                sizes = [inventory[f] for f in files | external]
                if max_bytes is not None and (any(s is None for s in sizes) or sum(sizes) > max_bytes):
                    raise ValueError("External weights exceed --max-gb or have unknown size; graph metadata is retained for inspection")
                if external:
                    fetch(sorted(external))
                files.update(external)
                plan = {**plan, "sizes": {f: inventory[f] for f in files}}
            for f in files:
                p = safe_path(destination, f)
                if not p.is_file():
                    raise ValueError(f"Incomplete download: {f}")
                expected = plan["sizes"].get(f)
                if expected is not None and p.stat().st_size != expected:
                    raise ValueError(f"Size mismatch: {f}")
            if plan["kind"] == "original":
                check_weights(destination)
            state.update(status="download_complete", files=sorted(files),
                         validation="not_run", sizes={f: safe_path(destination, f).stat().st_size for f in files})
            write_json(destination / MANIFEST, state)
            return state
        except Exception as error:
            state.update(status="failed", error=str(error))
            write_json(destination / MANIFEST, state)
            raise
