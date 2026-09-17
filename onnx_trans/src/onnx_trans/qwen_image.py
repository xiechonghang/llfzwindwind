from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open


@dataclass
class QwenImageBlock:
    model: torch.nn.Module
    layer: int
    config: dict[str, Any]
    weights: dict[str, dict[str, Any]]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        variance = x.float().pow(2).mean(-1, keepdim=True)
        return (x * torch.rsqrt(variance + self.eps)).to(input_dtype) * self.weight


class _GeluMLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.ModuleList([nn.Module(), nn.Identity(), nn.Linear(dim * 4, dim)])
        self.net[0].proj = nn.Linear(dim, dim * 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net[2](F.gelu(self.net[0].proj(x), approximate="tanh"))


class _QwenImageBlock(nn.Module):
    """Minimal source-aligned copy of QwenImageTransformerBlock.

    It intentionally contains only the block dependencies, so importing this project does
    not require the full Diffusers optional image/web stack.
    """

    def __init__(self, dim: int, num_attention_heads: int, attention_head_dim: int, qk_norm: str = "rms_norm", eps: float = 1e-6, zero_cond_t: bool = False):
        super().__init__()
        self.dim, self.heads, self.head_dim = dim, num_attention_heads, attention_head_dim
        self.img_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = nn.Module()
        for name in ("to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj"):
            setattr(self.attn, name, nn.Linear(dim, dim))
        self.attn.norm_q = _RMSNorm(attention_head_dim, eps)
        self.attn.norm_k = _RMSNorm(attention_head_dim, eps)
        self.attn.norm_added_q = _RMSNorm(attention_head_dim, eps)
        self.attn.norm_added_k = _RMSNorm(attention_head_dim, eps)
        self.attn.to_out = nn.ModuleList([nn.Linear(dim, dim)])
        self.attn.to_add_out = nn.Linear(dim, dim)
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_mlp = _GeluMLP(dim)
        self.txt_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        self.txt_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_mlp = _GeluMLP(dim)

    @staticmethod
    def _modulate(x: torch.Tensor, params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shift, scale, gate = params.chunk(3, dim=-1)
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1), gate.unsqueeze(1)

    def forward(self, hidden_states, encoder_hidden_states, encoder_hidden_states_mask, temb, image_rotary_emb):
        img_mod1, img_mod2 = self.img_mod(temb).chunk(2, dim=-1)
        txt_mod1, txt_mod2 = self.txt_mod(temb).chunk(2, dim=-1)
        img_x, img_gate1 = self._modulate(self.img_norm1(hidden_states), img_mod1)
        txt_x, txt_gate1 = self._modulate(self.txt_norm1(encoder_hidden_states), txt_mod1)
        iq, ik, iv = self.attn.to_q(img_x), self.attn.to_k(img_x), self.attn.to_v(img_x)
        tq, tk, tv = self.attn.add_q_proj(txt_x), self.attn.add_k_proj(txt_x), self.attn.add_v_proj(txt_x)
        b, si, _ = iq.shape
        st = tq.shape[1]
        iq, ik, iv = [z.unflatten(-1, (-1, self.head_dim)) for z in (iq, ik, iv)]
        tq, tk, tv = [z.unflatten(-1, (-1, self.head_dim)) for z in (tq, tk, tv)]
        iq, ik, tq, tk = self.attn.norm_q(iq), self.attn.norm_k(ik), self.attn.norm_added_q(tq), self.attn.norm_added_k(tk)
        img_freqs, txt_freqs = image_rotary_emb
        def rope(x, f):
            c, s = f[None, :, None, :].cos(), f[None, :, None, :].sin()
            xr, xi = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
            rot = torch.stack((-xi, xr), dim=-1).flatten(-2)
            return (x.float() * c + rot.float() * s).to(x.dtype)
        iq, ik, tq, tk = rope(iq, img_freqs), rope(ik, img_freqs), rope(tq, txt_freqs), rope(tk, txt_freqs)
        q, k, v = torch.cat((tq, iq), 1), torch.cat((tk, ik), 1), torch.cat((tv, iv), 1)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        mask = None
        if encoder_hidden_states_mask is not None:
            full = torch.cat((encoder_hidden_states_mask.bool(), torch.ones((b, si), dtype=torch.bool, device=q.device)), 1)
            mask = full[:, None, None, :]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(b, st + si, self.dim).to(q.dtype)
        txt_attn = self.attn.to_add_out(out[:, :st])
        img_attn = self.attn.to_out[0](out[:, st:])
        hidden_states = hidden_states + img_gate1 * img_attn
        encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn
        img_x, img_gate2 = self._modulate(self.img_norm2(hidden_states), img_mod2)
        txt_x, txt_gate2 = self._modulate(self.txt_norm2(encoder_hidden_states), txt_mod2)
        return encoder_hidden_states + txt_gate2 * self.txt_mlp(txt_x), hidden_states + img_gate2 * self.img_mlp(img_x)


def _source_import() -> Any:
    return _QwenImageBlock


def _weight_index(model_dir: Path) -> tuple[dict[str, str], Path]:
    indexes = sorted(model_dir.glob("*.safetensors.index.json"))
    if not indexes:
        raise FileNotFoundError(f"no safetensors index in {model_dir}")
    index_path = indexes[0]
    data = _load_json(index_path)
    return data["weight_map"], index_path


def _torch_dtype(name: str) -> torch.dtype:
    return {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "F64": torch.float64}[name]


def build_qwen_image_block(model_dir: str | Path, layer: int = 0, device: str = "cpu") -> QwenImageBlock:
    """Construct one block and load only its indexed safetensors parameters."""
    model_dir = Path(model_dir).resolve()
    config = _load_json(model_dir / "config.json")
    weight_map, index_path = _weight_index(model_dir)
    prefix = f"transformer_blocks.{layer}."
    selected = {k: v for k, v in weight_map.items() if k.startswith(prefix)}
    if not selected:
        raise KeyError(f"no weights found for layer {layer} in {index_path}")

    block_cls = _source_import()
    dim = int(config["num_attention_heads"]) * int(config["attention_head_dim"])
    block = block_cls(
        dim=dim,
        num_attention_heads=int(config["num_attention_heads"]),
        attention_head_dim=int(config["attention_head_dim"]),
        qk_norm="rms_norm",
        eps=1e-6,
        zero_cond_t=bool(config.get("zero_cond_t", False)),
    ).to(device=device)
    block.eval()

    # Convert module parameters to the source checkpoint dtype before copying. This avoids a
    # transient FP32 state_dict and makes an accidental precision change visible.
    metadata: dict[str, dict[str, Any]] = {}
    files = sorted(set(selected.values()))
    handles = {f: safe_open(model_dir / f, framework="pt", device=device) for f in files}
    try:
        state = dict(block.named_parameters())
        buffers = dict(block.named_buffers())
        all_tensors = {**state, **buffers}
        missing: list[str] = []
        for full_name, filename in selected.items():
            name = full_name[len(prefix) :]
            if name not in all_tensors:
                raise KeyError(f"checkpoint tensor {full_name} has no target module parameter {name}")
            tensor = handles[filename].get_tensor(full_name)
            target = all_tensors[name]
            if target.shape != tensor.shape:
                raise ValueError(f"shape mismatch for {full_name}: {tuple(tensor.shape)} != {tuple(target.shape)}")
            target.data = tensor.to(device=device)
            metadata[full_name] = {
                "file": filename,
                "shape": list(tensor.shape),
                "source_dtype": str(tensor.dtype).replace("torch.", ""),
                "loaded_dtype": str(target.dtype).replace("torch.", ""),
            }
        for name in all_tensors:
            full = prefix + name
            if full not in selected:
                missing.append(full)
        if missing:
            raise KeyError(f"missing {len(missing)} layer tensors, first: {missing[:4]}")
    finally:
        for handle in handles.values():
            handle.__exit__(None, None, None)

    return QwenImageBlock(model=block, layer=layer, config=config, weights=metadata)


class QwenImageBlockWrapper(torch.nn.Module):
    """Tensor-only boundary around the Diffusers block for ONNX export."""

    def __init__(self, block: torch.nn.Module):
        super().__init__()
        self.block = block

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor,
        temb: torch.Tensor,
        img_freqs: torch.Tensor,
        txt_freqs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.block(
            hidden_states,
            encoder_hidden_states,
            encoder_hidden_states_mask,
            temb,
            (img_freqs, txt_freqs),
        )


def fixture(dim: int, image_tokens: int = 4, text_tokens: int = 4, seed: int = 17) -> tuple[torch.Tensor, ...]:
    g = torch.Generator(device="cpu").manual_seed(seed)
    # RoPE frequency inputs are float32 in the source implementation, while block activations
    # and conditions follow the checkpoint dtype.
    dtype = torch.bfloat16
    return (
        torch.randn((1, image_tokens, dim), generator=g, dtype=dtype),
        torch.randn((1, text_tokens, dim), generator=g, dtype=dtype),
        torch.ones((1, text_tokens), dtype=torch.bool),
        torch.randn((1, dim), generator=g, dtype=dtype),
        torch.randn((image_tokens, 128), generator=g, dtype=torch.float32),
        torch.randn((text_tokens, 128), generator=g, dtype=torch.float32),
    )


def tensor_stats(x: torch.Tensor) -> dict[str, Any]:
    y = x.detach().float().cpu()
    return {"dtype": str(x.dtype).replace("torch.", ""), "shape": list(x.shape), "max_abs": float(y.abs().max())}


def as_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()
