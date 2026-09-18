# Hugging Face Model Analysis and ONNX Export

本仓库用于分析本地 Hugging Face 模型结构，并按层导出核心 Transformer 网络为 ONNX。

## ONNX 功能

`onnx_trans/` 是按层导出的独立工程，当前以 Qwen-Image 的单个 DiT block 为实例：

```bash
.venv-onnx/bin/python -m pip install -e onnx_trans
.venv-onnx/bin/python -m pip install onnxruntime

PYTHONPATH=onnx_trans/src \
.venv-onnx/bin/python -m onnx_trans.cli export-qwen-image \
  --model-dir models/Qwen/Qwen-Image/transformer \
  --layer 0 \
  --output outputs/onnx_trans/qwen-image/block-0
```

导出器只加载目标层的 safetensors 权重，并默认保留源权重 dtype。Qwen-Image 的
BF16 参数会生成 BF16 ONNX initializer，不会静默转换成 FP32。完整的依赖、输入输出、
产物和验证说明见 [`onnx_trans/README.md`](onnx_trans/README.md)。

## 目录说明

- `onnx_trans/`：按层导出 ONNX 的工程和 Qwen-Image 适配器。
- `src/`、`scripts/`、`tests/`：原有模型下载、导出、验证和测试代码。
- `sources/transformers/`、`sources/diffusers/`：通过 submodule 引用的上游源码。
- `models/`、`reports/`、`outputs/`：本地数据和生成产物，不提交模型权重或报告文件。

## 准备源码仓库

GitHub 主仓库不直接存储 Transformers 和 Diffusers 的源码内容。克隆本仓库后，在根目录执行下面的一条命令即可初始化两个源码仓库，并检出工程锁定的提交：

```bash
python3 scripts/prepare_sources.py
```

如果是从 GitHub ZIP 下载的主仓库（没有 `.git` 目录），同一条命令会自动使用 `git clone` 下载源码：

```bash
python3 scripts/prepare_sources.py --archive-mode
```

源码版本记录在 [`sources/sources.lock.json`](sources/sources.lock.json) 中。需要准备 HiDream 官方源码时，仍可使用主仓库的子模块命令：

```bash
git submodule update --init --recursive sources/HiDream-O1-Image
```
