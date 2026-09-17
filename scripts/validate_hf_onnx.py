#!/usr/bin/env python3
import sys
from hf_onnx_pipeline.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["validate", *sys.argv[1:]]))
