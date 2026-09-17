from __future__ import annotations

from pathlib import Path

from .common import environment, read_json, write_json
from .download import MANIFEST
from .graphs import inspect_graphs


def compare_text(reference: Path, output: Path, atol: float, rtol: float) -> dict:
    """Teacher-forced multi-step parity; both models receive identical token IDs."""
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM
    from optimum.onnxruntime import ORTModelForCausalLM

    config = read_json(reference / "config.json")
    if "vision_config" in config:
        raise ValueError("Automatic text parity supports text-only causal LMs. Multimodal/hybrid adapters need model-specific fixtures.")
    pt = AutoModelForCausalLM.from_pretrained(str(reference), local_files_only=True, torch_dtype=torch.float32).eval()
    ort = ORTModelForCausalLM.from_pretrained(str(output), local_files_only=True, provider="CPUExecutionProvider")
    records = []
    with torch.inference_mode():
        for length in (2, 5, 9):
            ids = torch.arange(1, length + 1).reshape(1, -1) % config["vocab_size"]
            full_ids = ids
            mask = torch.ones_like(ids)
            pt_cache = ort_cache = None
            for step in range(3):
                if getattr(ort, "use_cache", False):
                    left = pt(input_ids=ids, attention_mask=mask, past_key_values=pt_cache, use_cache=True)
                    right = ort(input_ids=ids, attention_mask=mask, past_key_values=ort_cache)
                    pt_cache, ort_cache = left.past_key_values, right.past_key_values
                else:
                    left = pt(input_ids=full_ids, attention_mask=mask, use_cache=False)
                    right = ort(input_ids=full_ids, attention_mask=mask)
                a = left.logits.detach().cpu().numpy()
                b = right.logits.detach().cpu().numpy()
                if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
                    raise ValueError("Invalid numerical output shape or non-finite values")
                delta = np.abs(a - b)
                records.append({"prompt_length": length, "step": step,
                                "max_abs_error": float(delta.max()), "mean_abs_error": float(delta.mean())})
                if not np.allclose(a, b, atol=atol, rtol=rtol):
                    raise ValueError(f"Numerical mismatch at length={length}, step={step}: max error={delta.max()}")
                ids = left.logits[:, -1].argmax(-1, keepdim=True)
                full_ids = torch.cat((full_ids, ids), dim=1)
                mask = torch.ones_like(full_ids)
    return {"status": "passed", "atol": atol, "rtol": rtol,
            "cache_tested": bool(getattr(ort, "use_cache", False)), "cases": records,
            "scope": "batch 1, prompt lengths 2/5/9, 3 teacher-forced steps; not an exhaustive equivalence proof"}


def validate_model(root: Path, graph_names: list[str] | None = None, runtime: bool = False,
                   reference: Path | None = None, atol: float = 1e-4, rtol: float = 1e-3) -> dict:
    if not root.is_dir():
        raise ValueError("ONNX directory does not exist")
    if atol < 0 or rtol < 0:
        raise ValueError("Tolerances must be nonnegative")
    if graph_names is None and (root / MANIFEST).is_file():
        snapshot = read_json(root / MANIFEST)
        if snapshot.get("status") != "download_complete":
            raise ValueError("Snapshot download has not completed")
        graphs = snapshot.get("graphs", [])
    else:
        graphs = graph_names or sorted(p.relative_to(root).as_posix() for p in root.rglob("*.onnx"))
    report = {"status": "running", "environment": environment()}
    try:
        report.update(inspect_graphs(root, graphs, runtime=runtime))
        if reference is not None:
            report["numerical_validation"] = compare_text(reference, root, atol, rtol)
        report["status"] = "passed"
        write_json(root / "validation.json", report)
        return report
    except Exception as error:
        report.update(status="failed", error=str(error))
        write_json(root / "validation.json", report)
        raise
