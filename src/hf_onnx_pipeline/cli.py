from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .common import DEFAULT_MODEL, DEFAULT_ROOT, environment, model_path, write_json


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Hugging Face → existing ONNX or original checkpoint → ONNX")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Report Python and installed dependency versions")
    for name in ("plan", "download"):
        s = sub.add_parser(name)
        s.add_argument("repo", nargs="?", default=DEFAULT_MODEL)
        s.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
        s.add_argument("--revision", help="Default: reuse existing manifest revision, otherwise main")
        s.add_argument("--mode", choices=("auto", "original", "metadata", "onnx"), default="auto")
        s.add_argument("--onnx-repo", help="Explicit alternate ONNX repository; never searched automatically")
        s.add_argument("--onnx-file", action="append", help="Repo-relative graph; repeat for multi-component models")
        s.add_argument("--save-plan", type=Path)
        if name == "download":
            s.add_argument("--max-gb", type=float, help="Upper bound on total selected bytes, including external tensors")
    for name in ("check", "export"):
        s = sub.add_parser(name)
        s.add_argument("model", nargs="?", default=DEFAULT_MODEL)
        s.add_argument("--task", default="image-text-to-text", help="Default targets Qwen3.5; for plain LMs use text-generation-with-past")
        if name == "check":
            s.add_argument("--report", type=Path)
        else:
            s.add_argument("--output", type=Path)
            s.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="fp32")
            s.add_argument("--opset", type=int)
            s.add_argument("--sequence-length", type=int, default=8)
            s.add_argument("--trust-remote-code", action="store_true")
    for name in ("check-genai", "export-genai"):
        s = sub.add_parser(name, help="Optional dense Qwen3.5 adapter for Microsoft ORT GenAI")
        s.add_argument("model", nargs="?", default=DEFAULT_MODEL)
        if name == "check-genai":
            s.add_argument("--report", type=Path)
        else:
            s.add_argument("--output", type=Path)
            s.add_argument("--precision", choices=("fp32", "fp16", "int4"), default="fp32")
            s.add_argument("--provider", choices=("cpu", "cuda"), default="cpu")
    s = sub.add_parser("validate")
    s.add_argument("directory", type=Path)
    s.add_argument("--onnx-file", action="append")
    s.add_argument("--runtime", action="store_true", help="Also load each graph in CPU ONNX Runtime; does not prove inference parity")
    s.add_argument("--reference", type=Path, help="Original text-only causal LM directory for numerical and decode parity")
    s.add_argument("--atol", type=float, default=1e-4)
    s.add_argument("--rtol", type=float, default=1e-3)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "doctor":
            result = environment()
        elif args.command in ("plan", "download"):
            from .download import execute_download, plan_download
            result = plan_download(args.repo, args.output_root, args.mode, args.revision, args.onnx_repo, args.onnx_file)
            if args.save_plan:
                write_json(args.save_plan, result)
            if args.command == "download":
                if args.max_gb is not None and args.max_gb <= 0:
                    raise ValueError("--max-gb must be positive")
                result = execute_download(result, None if args.max_gb is None else int(args.max_gb * 10**9))
        elif args.command in ("check", "export"):
            from .export import check_export, export_model
            model = model_path(args.model)
            if args.command == "check":
                result = check_export(model, args.task)
                if args.report:
                    write_json(args.report, result)
                print(json.dumps(result, ensure_ascii=False, indent=2))
                return 0 if result["supported"] else 2
            result = export_model(model, (args.output or model / "onnx/exported").resolve(), args.task,
                                  args.dtype, args.opset, args.sequence_length, args.trust_remote_code)
        elif args.command in ("check-genai", "export-genai"):
            from .genai import check_genai, export_genai
            model = model_path(args.model)
            if args.command == "check-genai":
                result = check_genai(model)
                if args.report:
                    write_json(args.report, result)
                print(json.dumps(result, ensure_ascii=False, indent=2))
                return 0 if result["supported"] else 2
            result = export_genai(model, (args.output or model / "onnx/genai").resolve(), args.precision, args.provider)
        else:
            from .validate import validate_model
            result = validate_model(args.directory.resolve(), args.onnx_file, args.runtime,
                                    args.reference, args.atol, args.rtol)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
