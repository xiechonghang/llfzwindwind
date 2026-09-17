from __future__ import annotations

import os
from pathlib import Path

from .common import directory_lock, environment, read_json, write_json
from .download import MANIFEST, check_weights
from .graphs import inspect_graphs


def check_export(model: Path, task: str) -> dict:
    config = read_json(model / "config.json")
    report = {"model": str(model), "model_type": config.get("model_type"), "task": task,
              "environment": environment(), "supported": False,
              "meaning": "Registry check only; actual export and numerical correctness require execution"}
    try:
        import transformers
        from optimum.exporters.tasks import TasksManager
        # Import ONNX configs to populate the backend registry.
        from optimum.exporters.onnx import model_configs  # noqa: F401

        report["transformers_import_path"] = transformers.__file__
        supported = TasksManager.get_supported_tasks_for_model_type(
            config["model_type"], exporter="onnx", library_name="transformers")
        report["available_tasks"] = sorted(supported)
        report["supported"] = task in supported
        if not report["supported"]:
            report["reason"] = "Task is not registered for this architecture; a model-specific adapter is required"
    except (ImportError, KeyError, ValueError) as error:
        report["reason"] = str(error)
    return report


def export_model(model: Path, output: Path, task: str, dtype: str = "fp32",
                 opset: int | None = None, sequence_length: int = 8,
                 trust_remote_code: bool = False) -> dict:
    compatibility = check_export(model, task)
    if not compatibility["supported"]:
        raise ValueError("Export preflight failed: " + compatibility.get("reason", "unsupported"))
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty; choose a new output to preserve previous exports")
    if (model / MANIFEST).is_file():
        downloaded = read_json(model / MANIFEST)
        if downloaded.get("status") != "download_complete" or downloaded.get("kind") != "original":
            raise ValueError("Download a complete original snapshot before exporting")
    else:
        downloaded = {"source": "local_unmanaged", "revision": None}
    check_weights(model)
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    with directory_lock(output):
        report = {"status": "exporting", "source": downloaded, "compatibility": compatibility,
                  "dtype": dtype, "requested_opset": opset, "sequence_length": sequence_length,
                  "task": task, "device": "cpu", "trust_remote_code": trust_remote_code}
        manifest = output / "export_manifest.json"
        write_json(manifest, report)
        try:
            # Do not silently fill missing checkpoint files from the network.
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            from optimum.exporters.onnx import main_export
            main_export(str(model), output=output, task=task, dtype=dtype, opset=opset,
                        device="cpu", local_files_only=True, trust_remote_code=trust_remote_code,
                        do_validation=True, sequence_length=sequence_length, batch_size=1)
            graphs = sorted(p.relative_to(output).as_posix() for p in output.rglob("*.onnx"))
            report["validation"] = inspect_graphs(output, graphs)
            # Some backend versions only log numerical validation failures. Do not
            # infer parity from main_export returning successfully.
            report["exporter_validation"] = "requested; use validate --reference for independently enforced text parity"
            report["status"] = "exported_structure_checked"
            write_json(manifest, report)
            return report
        except Exception as error:
            report.update(status="failed", error=str(error))
            write_json(manifest, report)
            raise
