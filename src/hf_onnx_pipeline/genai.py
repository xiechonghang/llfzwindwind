"""Optional Microsoft Model Builder adapter, installed in a separate environment."""
from __future__ import annotations

import importlib
import os
from pathlib import Path
import subprocess
import sys

from .common import directory_lock, environment, read_json, write_json
from .download import MANIFEST, check_weights
from .graphs import inspect_graphs


def check_genai(model: Path) -> dict:
    config = read_json(model / "config.json")
    report = {"backend": "genai", "model": str(model), "architectures": config.get("architectures", []),
              "environment": environment(), "supported": False,
              "meaning": "Builder class availability only; model export and runtime validation are separate"}
    if config.get("architectures") != ["Qwen3_5ForConditionalGeneration"]:
        report["reason"] = "This adapter currently targets the dense Qwen3.5 conditional-generation architecture"
        return report
    try:
        from transformers import AutoConfig
        AutoConfig.from_pretrained(str(model), local_files_only=True)
        module = importlib.import_module("onnxruntime_genai.models.builders.qwen")
        getattr(module, "Qwen35Model")
        importlib.import_module("onnxruntime_genai.models.builder")
        report["supported"] = True
        report["builder_source"] = module.__file__
    except (ImportError, AttributeError, ValueError, OSError) as error:
        report["reason"] = str(error)
    return report


def builder_command(model: Path, output: Path, precision: str, provider: str) -> list[str]:
    if precision not in {"fp32", "fp16", "int4"}:
        raise ValueError("GenAI adapter precision must be fp32, fp16 or int4")
    if provider not in {"cpu", "cuda"}:
        raise ValueError("GenAI adapter provider must be cpu or cuda")
    return [sys.executable, "-m", "onnxruntime_genai.models.builder", "-i", str(model),
            "-o", str(output), "-p", precision, "-e", provider,
            "-c", str(output / "builder-cache"), "--extra_options", "hf_remote=false", "exclude_embeds=false"]


def export_genai(model: Path, output: Path, precision: str = "fp32", provider: str = "cpu") -> dict:
    compatibility = check_genai(model)
    if not compatibility["supported"]:
        raise ValueError("GenAI preflight failed: " + compatibility.get("reason", "unsupported"))
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty")
    source = read_json(model / MANIFEST) if (model / MANIFEST).is_file() else {"source": "local_unmanaged", "revision": None}
    if "status" in source and (source["status"] != "download_complete" or source.get("kind") != "original"):
        raise ValueError("Download the complete original snapshot first")
    check_weights(model)
    command = builder_command(model, output, precision, provider)
    with directory_lock(output):
        state = {"status": "exporting", "backend": "genai", "source": source,
                 "compatibility": compatibility, "command": command,
                 "scope": "Text decoder with embeddings; optional MTP according to builder/config. Vision is not exported by this adapter.",
                 "runtime_requirement": "Matching ONNX Runtime GenAI and execution provider; may contain com.microsoft operators"}
        write_json(output / "export_manifest.json", state)
        try:
            env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
            with (output / "builder.log").open("w", encoding="utf-8") as log:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            if not (output / "genai_config.json").is_file():
                raise ValueError("Builder did not produce genai_config.json")
            graphs = sorted(p.relative_to(output).as_posix() for p in output.rglob("*.onnx") if "builder-cache" not in p.relative_to(output).parts)
            state["validation"] = inspect_graphs(output, graphs)
            state["status"] = "exported_structure_checked"
            write_json(output / "export_manifest.json", state)
            return state
        except Exception as error:
            state.update(status="failed", error=str(error))
            write_json(output / "export_manifest.json", state)
            raise
