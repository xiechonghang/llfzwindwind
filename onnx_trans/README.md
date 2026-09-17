# ONNX Trans

`onnx_trans` 用于把本地 Hugging Face / Diffusers 模型的核心 Transformer block
导出为 ONNX。工程按层加载权重，避免先把完整 LLM 或 DiT 模型加载到内存中。

当前实例是 `models/Qwen/Qwen-Image/transformer` 的
`QwenImageTransformerBlock`。首版支持导出指定的单个 DiT block，输入输出保持
Tensor-only 接口，适合后续接入完整推理流水线。

## 精度规则

默认策略是 `preserve`：源 checkpoint 中的参数是什么 dtype，ONNX initializer 就
保留什么 dtype。Qwen-Image 本地权重为 BF16，因此导出图的 initializer 也是
`BFLOAT16`，不会自动转成 FP32、FP16 或其他格式。

RoPE 角度输入按照源码的计算语义使用 FP32；hidden states、文本条件、时间条件和
模型参数跟随源权重 dtype。

## 安装

建议使用项目已有的独立环境：

```bash
python3.12 -m venv .venv-onnx
.venv-onnx/bin/python -m pip install -e onnx_trans
.venv-onnx/bin/python -m pip install onnxruntime
```

核心依赖包括 PyTorch、safetensors、ONNX、onnxscript 和 NumPy。导出器使用本地
`sources/diffusers` 中 Qwen-Image 实现的 block 结构；不会启用 remote code，也不会
联网补齐缺失权重。

检查环境：

```bash
.venv-onnx/bin/onnx-trans doctor
```

## 导出 Qwen-Image 单层 DiT

```bash
.venv-onnx/bin/onnx-trans export-qwen-image \
  --model-dir models/Qwen/Qwen-Image/transformer \
  --layer 0 \
  --output outputs/onnx_trans/qwen-image/block-0
```

命令会从 safetensors 索引中定位并读取
`transformer_blocks.0.*`，构造一个 block，然后使用 PyTorch 的 dynamo ONNX
导出器生成图和 external data。可以修改 `--layer` 导出其他层：

```bash
.venv-onnx/bin/onnx-trans export-qwen-image \
  --model-dir models/Qwen/Qwen-Image/transformer \
  --layer 12 \
  --image-tokens 16 \
  --text-tokens 128 \
  --output outputs/onnx_trans/qwen-image/block-12
```

首版使用固定的测试形状（默认 batch=1、图像 token=4、文本 token=4），用于先验证
真实权重和计算图。动态 batch、动态序列长度和完整 RoPE 生成将在此基础上扩展。

## 导出目录

每次运行使用一个独立目录，主要文件如下：

```text
block-0/
├── model.onnx                  # ONNX 图
├── model.onnx.data             # external data，通常占主要体积
├── plan.json                   # 模型、层号、输入规格和源 dtype
├── source_manifest.json        # 配置、权重索引和源码适配记录
├── weights_manifest.json       # 每个参数的文件、shape 和 dtype
├── io_spec.json                # 输入输出 Tensor 契约
├── precision_report.json       # 源权重、加载参数和 ONNX initializer dtype
├── validation.json             # checker、runtime 和数值验证状态
├── run.log                     # 失败时的完整 traceback
└── diagnostics/                # PyTorch ONNX 导出报告
```

## 输入输出

Qwen-Image block 的输入为：

- `hidden_states`: `[batch, image_tokens, 3072]`
- `encoder_hidden_states`: `[batch, text_tokens, 3072]`
- `encoder_hidden_states_mask`: `[batch, text_tokens]`，布尔 mask
- `temb`: `[batch, 3072]`
- `img_freqs`: `[image_tokens, 128]`，FP32 RoPE 角度
- `txt_freqs`: `[text_tokens, 128]`，FP32 RoPE 角度

输出顺序为：

1. 更新后的文本流 `encoder_hidden_states_out`
2. 更新后的图像流 `hidden_states_out`

## 验证结果的解释

`validation.json` 分开记录三个状态：

- `onnx_checker=passed`：图结构和 external data 引用有效。
- `runtime=loaded`：当前 ONNX Runtime 能加载该图。
- `numerical=passed`：PyTorch block 与 ONNX 输出完成数值比较。

某些 CPU 版 ONNX Runtime 不支持包含 BF16 的 `SplitToSequence` 等算子。此时应保留
BF16 图，并在 `validation.json` 中记录运行时失败原因；不能把 FP32 诊断图当成源
模型精度的交付结果。

## 当前边界

- 当前适配器只覆盖 Qwen-Image 的单个双流 DiT block。
- 还未导出完整 transformer、VAE、文本编码器或 pipeline。
- 当前导出为固定输入 shape；动态 shape 需要进一步处理 RoPE 和 attention 的符号维度。
- FP8、INT8、INT4 等量化权重需要额外处理 scale、zero point 和专用算子，不能直接
  当作普通 BF16/FP16 参数导出。
- 运行时验证取决于 ONNX Runtime 对 BF16 算子和目标设备的支持。
