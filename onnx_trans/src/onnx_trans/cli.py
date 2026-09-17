from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch

from .qwen_image import QwenImageBlockWrapper, as_numpy, build_qwen_image_block, fixture, tensor_stats


def _dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")


def export_qwen_image(args: argparse.Namespace) -> int:
    model_dir = Path(args.model_dir).resolve()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    log = out / "run.log"
    try:
        block = build_qwen_image_block(model_dir, args.layer, args.device)
        dim = int(block.config["num_attention_heads"]) * int(block.config["attention_head_dim"])
        inputs = fixture(dim, args.image_tokens, args.text_tokens, args.seed)
        wrapper = QwenImageBlockWrapper(block.model).eval()
        with torch.no_grad():
            reference = wrapper(*inputs)

        _dump(out / "plan.json", {
            "model_dir": str(model_dir), "layer": args.layer, "architecture": "QwenImageTransformerBlock",
            "source_implementation": "sources/diffusers/src/diffusers/models/transformers/transformer_qwenimage.py",
            "dim": dim, "image_tokens": args.image_tokens, "text_tokens": args.text_tokens,
            "device": args.device, "source_dtype": sorted({x["source_dtype"] for x in block.weights.values()}),
            "weights": len(block.weights),
        })
        _dump(out / "source_manifest.json", {
            "model_config": str(model_dir / "config.json"),
            "weight_index": str(next(model_dir.glob("*.safetensors.index.json"))),
            "source_file": "sources/diffusers/src/diffusers/models/transformers/transformer_qwenimage.py",
            "adapter": "onnx_trans.qwen_image._QwenImageBlock (source-aligned minimal block)",
            "remote_code": False,
        })
        _dump(out / "weights_manifest.json", block.weights)
        _dump(out / "io_spec.json", {
            "shape_mode": "static_fixture_for_first_export",
            "inputs": [
                {"name": "hidden_states", "shape": [1, args.image_tokens, dim], "dtype": "source"},
                {"name": "encoder_hidden_states", "shape": [1, args.text_tokens, dim], "dtype": "source"},
                {"name": "encoder_hidden_states_mask", "shape": [1, args.text_tokens], "dtype": "bool"},
                {"name": "temb", "shape": [1, dim], "dtype": "source"},
                {"name": "img_freqs", "shape": [args.image_tokens, 128], "dtype": "float32"},
                {"name": "txt_freqs", "shape": [args.text_tokens, 128], "dtype": "float32"},
            ], "outputs": ["encoder_hidden_states", "hidden_states"],
        })
        _dump(out / "precision_report.json", {
            "policy": "preserve", "source": sorted({x["source_dtype"] for x in block.weights.values()}),
            "loaded": sorted({x["loaded_dtype"] for x in block.weights.values()}),
            "activations": tensor_stats(reference[0]), "auxiliary_inputs": "RoPE frequencies are float32 by source design",
        })
        torch.onnx.export(
            wrapper, inputs, os.fspath(out / "model.onnx"), input_names=[
                "hidden_states", "encoder_hidden_states", "encoder_hidden_states_mask", "temb", "img_freqs", "txt_freqs"
            ], output_names=["encoder_hidden_states_out", "hidden_states_out"],
            opset_version=args.opset, dynamo=True, external_data=True, optimize=False,
            report=True, artifacts_dir=os.fspath(out / "diagnostics"),
        )
        model = onnx.load(out / "model.onnx", load_external_data=True)
        onnx.checker.check_model(model, full_check=False)
        initializer_dtypes = sorted({onnx.TensorProto.DataType.Name(x.data_type) for x in model.graph.initializer})
        _dump(out / "precision_report.json", {
            "policy": "preserve", "source": sorted({x["source_dtype"] for x in block.weights.values()}),
            "loaded": sorted({x["loaded_dtype"] for x in block.weights.values()}),
            "onnx_initializers": initializer_dtypes,
            "activations": tensor_stats(reference[0]), "auxiliary_inputs": "RoPE angles are float32 by source design",
        })
        validation: dict[str, Any] = {"onnx_checker": "passed", "runtime": "not-run", "numerical": "not-run"}
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(os.fspath(out / "model.onnx"), providers=["CPUExecutionProvider"])
            validation["runtime"] = "loaded"
            try:
                result = session.run(None, {n: as_numpy(x) for n, x in zip(
                    ["hidden_states", "encoder_hidden_states", "encoder_hidden_states_mask", "temb", "img_freqs", "txt_freqs"], inputs
                )})
                errors = [float(np.max(np.abs(result[i].astype(np.float32) - as_numpy(reference[i]).astype(np.float32)))) for i in range(2)]
                validation.update({"numerical": "passed", "max_abs_error": errors})
            except Exception as exc:
                validation.update({"numerical": "failed", "numerical_error": repr(exc)})
        except Exception as exc:
            validation.update({"runtime": "failed", "runtime_error": repr(exc)})
        _dump(out / "validation.json", validation)
        log.write_text("export completed\n" + json.dumps(validation, indent=2) + "\n")
        print(json.dumps({"output": str(out), **validation}, ensure_ascii=False, indent=2))
        return 0 if validation["onnx_checker"] == "passed" else 2
    except Exception:
        log.write_text(traceback.format_exc())
        traceback.print_exc()
        return 2


def doctor(_: argparse.Namespace) -> int:
    print(json.dumps({"python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "onnx": onnx.__version__}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="onnx-trans")
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("doctor"); d.set_defaults(func=doctor)
    e = sub.add_parser("export-qwen-image")
    e.add_argument("--model-dir", required=True); e.add_argument("--layer", type=int, default=0)
    e.add_argument("--output", required=True); e.add_argument("--device", default="cpu")
    e.add_argument("--image-tokens", type=int, default=4); e.add_argument("--text-tokens", type=int, default=4)
    e.add_argument("--seed", type=int, default=17); e.add_argument("--opset", type=int, default=18)
    e.set_defaults(func=export_qwen_image)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
