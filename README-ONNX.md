# Hugging Face 下载与 ONNX 工程

独立于 HF Model Architecture Analyzer。复用 `models/<组织>/<模型名>/` 目录约定，旧 metadata 下载器与报告生成器不变。

默认目标：`Qwen/Qwen3.5-9B`。**工程可运行不代表该模型已经完成 ONNX 导出。** 通用入口使用 Optimum；另提供 Microsoft ONNX Runtime GenAI 的 Qwen3.5 文本导出入口。先执行对应 `check` / `check-genai`，不能把 `qwen3_5` 冒充成 `qwen3`。查看 `outputs/onnx-pipeline/` 中本机检查记录（执行后生成）。

## 环境

使用 Python 3.11 或 3.12，独立安装，避免混用本仓库 `transformers/` 和 `sources/` 中用于分析的源码。

```bash
python3.12 -m venv .venv-onnx
.venv-onnx/bin/python -m pip install -e '.[export,test]'
.venv-onnx/bin/hf-onnx doctor
```

仅下载原始模型或已有 ONNX：`pip install -e .`，不安装 PyTorch。基础依赖为 Hub 和 ONNX；ONNX 用于解析图中外部权重的真实路径。执行 `validate --runtime` 还需安装 `onnxruntime`。文本数值校验需要 `export` extra。

导出依赖：PyTorch、Transformers、Optimum ONNX、ONNX、ONNX Runtime、safetensors、numpy、onnxscript。当前实现使用 CPU；没有 CUDA/FlashAttention 要求。首次运行验证通过后，使用 `requirements-onnx-lock.txt` 重建已验证环境；锁文件是本机 Python/平台快照，不保证所有平台都有相同 wheel。

本机已完成 Optimum 环境安装和 19 项测试。复现该环境可先运行 `python -m pip install -r requirements-onnx-lock.txt`，再运行 `python -m pip install -e . --no-deps`。此锁文件不用于 GenAI 环境。

## 推荐流程

所有命令从项目根目录执行。`plan` 查询官网文件清单，不下载权重；默认复用已有 manifest 的 commit，否则解析 main 并固定 SHA。

```bash
.venv-onnx/bin/hf-onnx plan --save-plan outputs/onnx-pipeline/qwen35-download-plan.json
.venv-onnx/bin/hf-onnx download --mode metadata
.venv-onnx/bin/hf-onnx check --report outputs/onnx-pipeline/qwen35-compatibility.json
```

`check` 只读 config 和导出注册表，不加载权重。返回 0 表示已注册，2 表示当前架构/任务不受支持；注册通过也不能保证真实模型一定导出成功。

确认后端支持或完成专用适配后，下载完整原始快照并导出：

```bash
.venv-onnx/bin/hf-onnx download --max-gb 25
.venv-onnx/bin/hf-onnx export --task image-text-to-text
```

默认 `auto` 优先同一指定仓库内的 ONNX；无 ONNX 才下载原始完整仓库。`--mode original` 强制完整快照（包括仓库里的其他格式）；`--mode metadata` 仅下载配置、tokenizer、processor、文档和代码等元数据，不能执行真实权重导出。

### Qwen3.5 的可选 GenAI 后端

Microsoft 的官方 Model Builder 源码有 `Qwen3_5ForConditionalGeneration → Qwen35Model` 分支。新增入口使用它导出**包含 embedding 的文本 decoder**，可能按配置额外生成 MTP；不代表完整视觉模型导出。它可能使用 `com.microsoft` 算子，需要对应 ORT GenAI / execution provider。当前入口尚未执行 Qwen3.5-9B 实权重导出或数值校验。

Optimum 与新版 Transformers 的依赖约束可能冲突，GenAI 使用另一环境，不要同时安装两个 extra：

```bash
python3.12 -m venv .venv-genai
.venv-genai/bin/python -m pip install -e '.[genai]'
.venv-genai/bin/hf-onnx check-genai --report outputs/onnx-pipeline/qwen35-genai-compatibility.json
.venv-genai/bin/hf-onnx download --max-gb 25
.venv-genai/bin/hf-onnx export-genai --precision fp32 --provider cpu
```

`check-genai` 检查本地 AutoConfig 能否加载，以及已安装包中 Qwen35Model 和 builder 是否可导入。官方 main 的支持不能证明某个发布 wheel 已包含该分支；若检查失败，报告具体版本/缺失项，不自动拉取未固定版本的源码。平台没有合适 wheel 时需要在受支持环境执行。

输出为 `onnx/genai/`，包含 `builder.log`、`genai_config.json`、图和外部数据。可以显式选择 `--precision int4`，但不会默认量化或把量化结果声称为 FP32 等价。前向计算正确性仍需专用 GenAI 测试，普通 `--reference` 不支持这类产物。

### 已有 ONNX

```bash
.venv-onnx/bin/hf-onnx plan OWNER/MODEL
.venv-onnx/bin/hf-onnx download OWNER/MODEL --onnx-file onnx/model.onnx
```

多个 ONNX 时不会随意混合精度或只挑一个视觉/解码组件；命令列出候选并要求使用重复的 `--onnx-file` 明确整套所需计算图。同时下载 tokenizer/processor/运行配置等元数据，解析并下载每个图引用的外部权重，保留仓库相对路径。元数据可以含多个变体的配置：实际运行仍需按模型说明选对应配置。自动识别图的任务、量化精度、设备兼容性不属于第一版能力。

只有显式指定 `--onnx-repo OWNER/CONVERTED-MODEL` 才使用其他来源的 ONNX；不会自动搜索社区仓库或混用官方配置。`--revision` 此时作用于 ONNX 来源，manifest 同时记录请求模型与实际来源，不声称已验证两者权重等价。

已有 ONNX 无法加载时保留错误，用户可通过 `--mode original` 选择原始导出流程；不会自动重复下载大型权重。

### 导出与验证

```bash
.venv-onnx/bin/hf-onnx export OWNER/TEXT-MODEL --task text-generation-with-past
.venv-onnx/bin/hf-onnx validate models/OWNER/TEXT-MODEL/onnx/exported --runtime --reference models/OWNER/TEXT-MODEL
```

`export` 使用真实本地权重，默认 FP32、batch 1、示例序列长度 8，保留后端默认动态维度和模型专用 opset。加载缺文件不自动联网补齐。允许 `--dtype`、`--opset`、`--sequence-length`、`--output`。不覆盖非空产物目录；失败后的目录保留日志，重试指定新目录。自定义仓库代码需要显式 `--trust-remote-code`。

验证分开记录：

- 下载完成：文件存在、大小匹配、索引分片齐全；尚未证明图正确。
- 结构检查：ONNX checker 通过、外部权重存在。
- `--runtime`：CPU ORT session 成功加载；尚未运行测试输入。
- `--reference`：对纯文本 causal LM 执行 PyTorch/ORT logits 对比。batch 1，初始长度 2/5/9，各 3 次前向；若 ONNX 提供 cache，则分别维护两端缓存，使用相同参考 token 测连续 decode。误差/非有限值导致非零退出码。

多模态与混合递归状态模型（包括 Qwen3.5）的自动数值验证尚需专用输入和 adapter，不能套用普通 KV-cache 验证。现有 ONNX 若要求 ORT GenAI、自定义算子、GPU 等，CPU session 检查可能失败，应据原模型要求增加对应 runtime。

Optimum 某些版本会把数值校验失败降为日志；本工程不把导出函数返回等同于数值验证通过。`export_manifest.json` 标记 `exported_structure_checked`；单独的 `validation.json` 明确数值校验是否执行。

## 目录与入口

- `src/hf_onnx_pipeline/`：共享实现、下载计划、外部数据解析、导出、验证、CLI。
- `scripts/download_hf_model.py`、`export_hf_onnx.py`、`validate_hf_onnx.py`：兼容脚本入口，先安装工程后使用。
- `models/<org>/<name>/`：原始快照，`.hf-onnx-download.json` 记录来源、commit、文件与状态。
- `models/<org>/<name>/onnx/downloaded/`：已有 ONNX 快照，保留来源仓库内部目录。
- `models/<org>/<name>/onnx/exported/`：新导出图和外部数据、导出/校验记录。
- `tests/`：策略、revision、防路径越界、缺失分片、外部数据、失败恢复，以及本地 tiny GPT2 真实导出/数值测试。

旧 `download_hf_metadata.py` 保留。新下载器复用了其参数/目录/版本记录设计，改用官方 Hub 客户端；不导入 skill 内脚本。旧 metadata manifest 的 SHA 可用于补齐同版本权重。已有目录无版本记录，或新旧 commit 不同，则要求另选 `--output-root`，不覆盖混合文件。

下载使用 `.hf-onnx.lock` 防并发，失败保留 Hub 下载缓存，可重新运行恢复。进程被强制杀死后可能残留 lock；核实其中 PID 已退出再移除。`--max-gb` 是完整选定文件体积限制，而非本次新增流量；已有 ONNX 的外部数据体积需要先读取图才能确定，超限时可能已经下载了图与元数据，但不会继续下载大权重。

## 测试

```bash
.venv-onnx/bin/python -m pytest -q
```

集成测试在临时目录创建随机初始化 tiny GPT2，实际导出并验证数值与 cache。它只验证工程链路，不是 Qwen3.5-9B 的替代模型实验，也不证明 Qwen3.5 已完成导出。

当前验证结果：19 passed，0 skipped。测试覆盖三个初始输入长度和连续 cache 更新。后端有 TorchScript 导出弃用和 tracing 警告；这些测试不证明所有动态维度/分支都能泛化。详细结果保存在 `outputs/onnx-pipeline/tests.xml` 和 `build-status.json`。Qwen3.5-9B 已下载 23,069,667 字节元数据，尚未下载权重或生成其 ONNX。

## 官方接口依据

- [Hub 下载 API](https://huggingface.co/docs/huggingface_hub/guides/download)
- [Optimum ONNX 安装](https://huggingface.co/docs/optimum-onnx/en/installation)
- [导出接口](https://huggingface.co/docs/optimum-onnx/en/onnx/package_reference/export)
- [Qwen3.5-9B 官方仓库](https://huggingface.co/Qwen/Qwen3.5-9B/tree/main)
- [Microsoft Model Builder](https://github.com/microsoft/onnxruntime-genai/blob/main/src/python/py/models/README.md)
- [Qwen3.5 builder 分支](https://github.com/microsoft/onnxruntime-genai/blob/main/src/python/py/models/builder.py)
