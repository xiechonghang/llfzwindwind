#!/usr/bin/env python3
"""Compatibility entrypoint; install the project in .venv-onnx first."""
import sys
from hf_onnx_pipeline.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["download", *sys.argv[1:]]))
