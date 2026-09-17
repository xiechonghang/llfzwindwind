# Model metadata

This directory stores Hugging Face model configuration, tokenizer, documentation,
and implementation files for architecture analysis. Model weights are intentionally
excluded.

The default model name is defined at the top of
[`scripts/download_hf_metadata.py`](../scripts/download_hf_metadata.py). Edit
`DEFAULT_MODEL_ID` and run the script without arguments:

```bash
python3 scripts/download_hf_metadata.py
```

For a one-off download, pass the model name directly:

```bash
python3 scripts/download_hf_metadata.py owner/model-name
```

The repository ID is always interpreted as `organization/model-name`, so for
example:

- `Qwen/Qwen3.8-Flash-Next` → `models/Qwen/Qwen3.8-Flash-Next/`
- `THUDM/GLM-4.5` → `models/THUDM/GLM-4.5/`
- `google/gemma-3-27b-it` → `models/google/gemma-3-27b-it/`

Dots, hyphens, and version suffixes in model names are supported.

The downloader excludes common weight formats including `.safetensors`, `.bin`,
`.pt`, `.pth`, `.ckpt`, `.h5`, `.msgpack`, `.onnx`, `.gguf`, `.tflite`,
`.ot`, `.pb`, `.pkl`, `.pickle`, `.npy`, and `.npz`.
It writes `metadata_manifest.json` with the resolved commit and the exact lists of
downloaded and excluded files.
