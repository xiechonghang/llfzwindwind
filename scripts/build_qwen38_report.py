#!/usr/bin/env python3
"""Build the source-aligned Qwen3.8-Flash-Next architecture report."""

from __future__ import annotations

import html
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / ".agents/skills/hf-model-architecture/assets/report-template.html"
OUT = ROOT / "reports/Qwen/Qwen3.8-Flash-Next/report.html"


def node(node_id: str, kind: str, x: int, y: int, w: int, h: int, title: str, shape: str) -> str:
    rx = ' rx="10"' if kind == "tensor-node" else ""
    return f'''<g class="{kind}"><rect x="{x}" y="{y}" width="{w}" height="{h}"{rx}/><text x="{x+w/2}" y="{y+22}"><tspan class="node-id">{node_id}</tspan><tspan x="{x+w/2}" dy="18">{html.escape(title)}</tspan><tspan class="shape-label" x="{x+w/2}" dy="17">{html.escape(shape)}</tspan></text></g>'''


def flow(points: list[tuple[int, int]]) -> str:
    commands = [f"M {points[0][0]} {points[0][1]}"] + [f"L {x} {y}" for x, y in points[1:]]
    return f'<path class="flow" marker-end="url(#arrow)" d="{" ".join(commands)}"/>'


def svg(nodes: list[str], flows: list[str], height: int, labels: list[tuple[int, int, str]] | None = None) -> str:
    lane_labels = "".join(
        f'<text class="lane-label" x="{x}" y="{y}">{html.escape(label)}</text>' for x, y, label in (labels or [])
    )
    return f'''<div class="graph-scroll"><svg class="architecture-graph" viewBox="0 0 820 {height}" role="img">
    <defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor"/></marker></defs>
    {lane_labels}{''.join(flows)}{''.join(nodes)}
    </svg></div>'''


def code(lines: list[tuple[str, str]]) -> str:
    return "\n".join(f'<span class="code-map">{tag}</span>{html.escape(text)}' for tag, text in lines)


def pair(title: str, diagram: str, source: str, line_start: int, line_end: int, kind: str, lines: list[tuple[str, str]]) -> str:
    href = f"../../../{source}#L{line_start}"
    return f'''<div class="graph-code-pair">
      <div class="graph-panel"><h4>{html.escape(title)}</h4><div class="graph-legend"><span><i class="tensor-key"></i>张量（圆角）</span><span><i></i>运算（直角）</span></div>{diagram}</div>
      <div class="code-panel"><h4><a href="{href}">{html.escape(source)} · L{line_start}–L{line_end}</a><span class="excerpt-kind">{html.escape(kind)}</span></h4><pre><code>{code(lines)}</code></pre></div>
    </div>'''


def top_level_pair() -> str:
    ns = [
        node("TL1", "tensor-node", 20, 45, 150, 64, "input_ids", "[B,S]"),
        node("TL2", "op-node", 215, 45, 165, 64, "embed_tokens", "[B,S] → [B,S,2560]"),
        node("TL3", "op-node", 430, 10, 175, 76, "visual + masked_scatter", "patches → [B,S,2560]"),
        node("TL4", "op-node", 430, 120, 175, 76, "repeat(hc_count=4)", "[B,S,2560] → [B,S,10240]"),
        node("TL5", "op-node", 645, 75, 150, 76, "48 decoder layers", "36 DeltaNet + 12 QSA"),
        node("TL6", "op-node", 430, 250, 175, 76, "hyper_connection_mixer", "[B,S,10240] → [B,S,2560]"),
        node("TL7", "op-node", 645, 250, 150, 76, "lm_head", "[B,S,2560] → [B,S,248320]"),
    ]
    fs = [
        flow([(170,77),(215,77)]), flow([(380,77),(430,48)]), flow([(380,77),(430,158)]),
        flow([(605,48),(645,105)]), flow([(605,158),(645,105)]), flow([(720,151),(720,220),(605,288)]),
        flow([(605,288),(645,288)]),
    ]
    return pair("顶层多模态与 4-stream 文本路径", svg(ns, fs, 360),
        "sources/transformers/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py", 2249, 2464, "source-aligned derivation",
        [("TL1", "if inputs_embeds is None: inputs_embeds = self.get_input_embeddings()(input_ids)"),
         ("TL2", "inputs_embeds = self.language_model.embed_tokens(input_ids)"),
         ("TL3", "inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)"),
         ("TL4", "hidden_states = inputs_embeds.repeat(1, 1, self.config.hc_count)"),
         ("TL5", "for decoder_layer in self.layers[: self.config.num_hidden_layers]: hidden_states = decoder_layer(...)"),
         ("TL6", "hidden_states = self.hyper_connection_mixer(hidden_states)"),
         ("TL7", "logits = self.lm_head(hidden_states[:, slice_indices, :])")])


def decoder_pair(prefix: str, title: str, mixer: str) -> str:
    ns = [
        node(prefix+"1", "tensor-node", 20, 40, 170, 68, "hidden_states", "[B,S,10240]"),
        node(prefix+"2", "op-node", 240, 30, 185, 86, "attn_hyper_connection", "read [B,S,2560] + write [B,S,4]"),
        node(prefix+"3", "op-node", 480, 30, 155, 86, mixer, "[B,S,2560] → [B,S,2560]"),
        node(prefix+"4", "op-node", 675, 30, 125, 86, "gated inject", "→ [B,S,10240]"),
        node(prefix+"5", "op-node", 240, 180, 185, 86, "mlp_hyper_connection", "read [B,S,2560] + write [B,S,4]"),
        node(prefix+"6", "op-node", 480, 180, 155, 86, "SparseMoE", "512→top10 + shared"),
        node(prefix+"7", "tensor-node", 675, 180, 125, 86, "layer output", "[B,S,10240]"),
    ]
    fs = [flow([(190,74),(240,74)]),flow([(425,74),(480,74)]),flow([(635,74),(675,74)]),flow([(738,116),(738,145),(240,223)]),flow([(425,223),(480,223)]),flow([(635,223),(675,223)])]
    return pair(title, svg(ns, fs, 300), "sources/transformers/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py", 1192, 1245, "source-aligned derivation",
        [(prefix+"1", "hidden_states: torch.Tensor  # [B,S,4*H]"),
         (prefix+"2", "hidden_states, hyper_input, injection_weights = self.attn_hyper_connection(hidden_states)"),
         (prefix+"3", ("hidden_states = self.linear_attn(hidden_states, cache_params=past_key_values, ...)" if prefix == "DL" else "hidden_states, _ = self.self_attn(hidden_states, position_embeddings, ..., past_key_values=...)")),
         (prefix+"4", "hidden_states = hyper_input + (hidden_states.unsqueeze(-2) * injection_weights.unsqueeze(-1)).flatten(-2)"),
         (prefix+"5", "hidden_states, hyper_input, injection_weights = self.mlp_hyper_connection(hidden_states)"),
         (prefix+"6", "hidden_states = self.mlp(hidden_states)"),
         (prefix+"7", "hidden_states = hyper_input + (hidden_states.unsqueeze(-2) * injection_weights.unsqueeze(-1)).flatten(-2)")])


def full_attention_pair() -> str:
    ns = [
        node("FA1", "tensor-node", 20, 40, 140, 64, "hidden_states", "[B,Sq,2560]"),
        node("FA2", "op-node", 200, 10, 165, 76, "index_qk_proj", "2560→640 = 4×128 + 1×128"),
        node("FA3", "op-node", 400, 10, 175, 76, "4-token block pool", "raw_keys [B,Skv,128] → [Nb,128]"),
        node("FA4", "op-node", 615, 10, 180, 76, "scores.topk", "[Nb,4]→512 blocks→≤2051 tokens"),
        node("FA5Q", "op-node", 200, 125, 165, 76, "q_proj + split gate", "2560→12288→Q/G [B,Sq,6144]"),
        node("FA5K", "op-node", 200, 230, 165, 76, "k_proj", "2560→512→[B,2,Sq,256]"),
        node("FA5V", "op-node", 200, 335, 165, 76, "v_proj", "2560→512→[B,2,Sq,256]"),
        node("FA6", "op-node", 400, 145, 175, 76, "q/k norm + partial RoPE", "64 rotary + 192 non-rotary"),
        node("FA7", "op-node", 400, 250, 175, 76, "cache update + repeat_kv", "2 KV heads → logical 24 heads"),
        node("FA8", "op-node", 615, 145, 180, 76, "attn_weights = matmul", "[B,24,Sq,256]×[B,24,256,Skv]"),
        node("FA9", "op-node", 615, 250, 180, 76, "mask + softmax", "[B,24,Sq,Skv], ≤2051 visible"),
        node("FA10", "op-node", 615, 355, 180, 76, "attn_output = matmul", "P×V → [B,24,Sq,256]"),
        node("FA11", "op-node", 400, 440, 175, 76, "sigmoid(gate) + o_proj", "6144→2560"),
        node("FA12", "tensor-node", 615, 450, 180, 64, "attention output", "[B,Sq,2560]"),
    ]
    fs = [
        flow([(160,72),(200,48)]),flow([(365,48),(400,48)]),flow([(575,48),(615,48)]),
        flow([(160,72),(200,163)]),flow([(160,72),(180,268),(200,268)]),flow([(160,72),(180,373),(200,373)]),
        flow([(365,163),(400,183)]),flow([(365,268),(400,288)]),flow([(575,183),(615,183)]),
        flow([(575,288),(615,183)]),flow([(705,86),(705,125),(705,250)]),flow([(705,221),(705,250)]),
        flow([(705,326),(705,355)]),flow([(365,373),(520,373),(615,393)]),flow([(705,431),(575,478)]),
        flow([(365,163),(365,478),(400,478)]),flow([(575,478),(615,482)])
    ]
    return pair("QSA：索引器分支与 gated GQA 主分支", svg(ns, fs, 550, [(282,118,"Q / gate"),(282,223,"K"),(282,328,"V")]),
        "sources/transformers/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py", 611, 839, "source-aligned derivation + eager fallback",
        [("FA1", "batch_size, seq_length, _ = hidden_states.shape"),
         ("FA2", "qk = self.index_qk_proj(hidden_states); q, token_k = torch.split(qk, [4*128, 1*128], dim=-1)"),
         ("FA3", "key_groups = raw_keys.index_select(...).view(num_complete_blocks, 4, 128); pooled_keys = key_groups.float().mean(dim=1)"),
         ("FA4", "scores = relu(matmul(q.float(), block_key_states.float().T).T).sum(-1) / sqrt(128); selected_block_indices = scores.topk(min(512, num_complete_blocks), dim=0).indices"),
         ("FA5Q", "query_states, gate = torch.chunk(self.q_proj(hidden_states).view(B, Sq, -1, 512), 2, dim=-1)"),
         ("FA5K", "key_states = self.k_norm(self.k_proj(hidden_states).view(B, Sq, 2, 256)).transpose(1, 2)"),
         ("FA5V", "value_states = self.v_proj(hidden_states).view(B, Sq, 2, 256).transpose(1, 2)"),
         ("FA6", "query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)"),
         ("FA7", "key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx); key_states = repeat_kv(key_states, 12)"),
         ("FA8", "attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling"),
         ("FA9", "attn_weights = softmax(attn_weights + attention_mask, dim=-1, dtype=torch.float32).to(query.dtype)"),
         ("FA10", "attn_output = torch.matmul(attn_weights, value_states)"),
         ("FA11", "attn_output = self.o_proj(attn_output.reshape(B, Sq, 6144) * torch.sigmoid(gate))"),
         ("FA12", "return attn_output, attn_weights")])


def linear_attention_pair() -> str:
    ns = [
        node("LA1", "tensor-node", 20, 35, 140, 64, "hidden_states", "[B,S,2560]"),
        node("LA2", "op-node", 200, 5, 175, 76, "in_proj_qkv", "2560→10240"),
        node("LA3", "op-node", 200, 105, 175, 76, "in_proj_z", "2560→6144=[48×128]"),
        node("LA4", "op-node", 200, 205, 175, 76, "in_proj_a / in_proj_b", "2560→48 each"),
        node("LA5", "op-node", 415, 5, 175, 76, "depthwise causal conv1d", "[B,10240,S], kernel=4"),
        node("LA6", "op-node", 415, 110, 175, 76, "split + repeat Q/K", "Q,K [B,S,16,128]→48 heads"),
        node("LA7", "op-node", 415, 215, 175, 76, "beta=sigmoid(b), g(a)", "[B,S,48] float32 decay"),
        node("LA8", "op-node", 630, 100, 170, 100, "gated delta rule", "Q,K,V [B,S,48,128]\nstate [B,48,128,128]"),
        node("LA9", "op-node", 630, 245, 170, 86, "RMSNormGated(z)", "[B*S*48,128]"),
        node("LA10", "op-node", 415, 355, 175, 76, "out_proj", "6144→2560"),
        node("LA11", "tensor-node", 630, 365, 170, 64, "linear_attn output", "[B,S,2560]"),
    ]
    fs = [flow([(160,67),(200,43)]),flow([(160,67),(200,143)]),flow([(160,67),(180,243),(200,243)]),flow([(375,43),(415,43)]),flow([(590,43),(590,110)]),flow([(590,148),(630,148)]),flow([(375,143),(610,143),(630,280)]),flow([(375,243),(415,253)]),flow([(590,253),(630,168)]),flow([(715,200),(715,245)]),flow([(715,331),(590,393)]),flow([(590,393),(630,397)])]
    return pair("Gated DeltaNet：卷积 + 48-head recurrent state", svg(ns, fs, 465),
        "sources/transformers/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py", 403, 563, "source-aligned derivation + recurrent fallback",
        [("LA1", "hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)"),
         ("LA2", "mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)"),
         ("LA3", "z = self.in_proj_z(hidden_states).reshape(B, S, 48, 128)"),
         ("LA4", "b = self.in_proj_b(hidden_states); a = self.in_proj_a(hidden_states)"),
         ("LA5", "mixed_qkv = causal_conv1d_fn(cache.update_conv_state(mixed_qkv, ...), self.conv1d.weight, activation='silu')"),
         ("LA6", "query, key, value = torch.split(mixed_qkv, [2048,2048,6144], dim=-1); query = query.reshape(B,S,16,128).repeat_interleave(3, dim=2)"),
         ("LA7", "beta = b.sigmoid(); g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)"),
         ("LA8", "core_attn_out, last_recurrent_state = (torch_recurrent_gated_delta_rule if cached_S_eq_1 else torch_chunk_gated_delta_rule)(query,key,value,g=g,beta=beta,...)") ,
         ("LA9", "core_attn_out = self.norm(core_attn_out.reshape(-1,128), z.reshape(-1,128))"),
         ("LA10", "output = self.out_proj(core_attn_out.reshape(B,S,6144))"),
         ("LA11", "return output")])


def moe_pair() -> str:
    ns = [
        node("MOE1", "tensor-node", 20, 70, 145, 64, "hidden_states", "[B,S,2560] → [T,2560]"),
        node("MOE2", "op-node", 210, 15, 165, 76, "gate = F.linear", "[T,2560]×[2560,512]→[T,512]"),
        node("MOE3", "op-node", 420, 15, 175, 76, "softmax + topk", "[T,512]→weights/ids [T,10]"),
        node("MOE4", "op-node", 210, 130, 165, 76, "dispatch payload", "Σe Ne = T×10"),
        node("MOE5G", "op-node", 420, 115, 175, 76, "experts.gate_up_proj", "[Ne,2560]→gate/up [Ne,640]"),
        node("MOE5D", "op-node", 630, 115, 170, 76, "experts.down_proj", "[Ne,640]→[Ne,2560]"),
        node("MOE6", "op-node", 420, 235, 175, 86, "weighted index_add_", "[T,10] weights → [T,2560]"),
        node("MOE7", "op-node", 210, 280, 165, 86, "shared_expert + sigmoid gate", "2560→640→2560"),
        node("MOE8", "op-node", 630, 275, 170, 76, "routed + shared", "[T,2560] → [B,S,2560]"),
    ]
    fs = [flow([(165,102),(210,53)]),flow([(375,53),(420,53)]),flow([(165,102),(210,168)]),flow([(595,53),(595,95),(507,115)]),flow([(375,168),(420,153)]),flow([(595,153),(630,153)]),flow([(715,191),(715,225),(595,278)]),flow([(165,102),(190,323),(210,323)]),flow([(375,323),(630,313)]),flow([(595,278),(630,313)])]
    return pair("MoE：router 与 token payload 分离，10 routed + 1 shared", svg(ns, fs, 400),
        "sources/transformers/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py", 842, 937, "source-aligned derivation",
        [("MOE1", "hidden_states_reshaped = hidden_states.view(-1, hidden_dim)"),
         ("MOE2", "router_logits = F.linear(hidden_states, self.weight)  # [T,512]"),
         ("MOE3", "router_probs = softmax(router_logits, dtype=float, dim=-1); router_top_value, router_indices = torch.topk(router_probs, 10, dim=-1); router_top_value /= router_top_value.sum(-1, keepdim=True)"),
         ("MOE4", "expert_mask = one_hot(router_indices, num_classes=512).permute(2,1,0); top_k_pos, token_idx = torch.where(expert_mask[expert_idx])"),
         ("MOE5G", "gate, up = F.linear(hidden_states[token_idx], self.gate_up_proj[expert_idx]).chunk(2, dim=-1)"),
         ("MOE5D", "current_hidden_states = F.linear(silu(gate) * up, self.down_proj[expert_idx])"),
         ("MOE6", "final_hidden_states.index_add_(0, token_idx, current_hidden_states * top_k_weights[token_idx, top_k_pos, None])"),
         ("MOE7", "shared_expert_output = sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)"),
         ("MOE8", "expert_output = (expert_output + shared_expert_output).reshape(B,S,2560)")])


def residual_pair() -> str:
    ns = [
        node("RN1", "tensor-node", 20, 20, 160, 64, "hyper_input", "[B,S,4×2560]"),
        node("RN2", "op-node", 220, 5, 180, 86, "hc_norm + low-rank read gate", "10240→320→10240"),
        node("RN3", "op-node", 445, 5, 160, 86, "weighted mean", "[B,S,4,2560]→[B,S,2560]"),
        node("RN4", "op-node", 640, 5, 160, 86, "block_inject_weight", "10240→4; 2·sigmoid"),
        node("RN5", "op-node", 445, 125, 160, 86, "branch block F", "[B,S,2560]→[B,S,2560]"),
        node("RN6", "op-node", 640, 125, 160, 86, "inject + residual", "hyper_input + F×write_gate"),
        node("RN7", "op-node", 20, 270, 160, 86, "hashed bigram/trigram IDs", "16 heads; context 2"),
        node("RN8", "op-node", 220, 270, 180, 86, "ngram_embedding", "[B,S,16]→[B,S,2560]"),
        node("RN9", "op-node", 445, 255, 160, 100, "key/value projections + gate", "key [B,S,4,2560]; value [B,S,2560]"),
        node("RN10", "op-node", 640, 270, 160, 86, "dilated depthwise conv", "kernel 4, dilation 3 → [B,S,10240]"),
        node("RN11", "tensor-node", 640, 390, 160, 64, "PLE injection", "only one-indexed layer 2"),
    ]
    fs = [flow([(180,52),(220,48)]),flow([(400,48),(445,48)]),flow([(400,48),(640,48)]),flow([(525,91),(525,125)]),flow([(605,168),(640,168)]),flow([(720,91),(720,125)]),flow([(180,313),(220,313)]),flow([(400,313),(445,305)]),flow([(605,305),(640,313)]),flow([(720,356),(720,390)])]
    return pair("Gated Residual 与第 2 层 PLE 注入", svg(ns, fs, 490),
        "sources/transformers/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py", 941, 1190, "source-aligned derivation",
        [("RN1", "hyper_input_normed = self.hc_norm(hyper_input)  # [...,10240] grouped by H=2560"),
         ("RN2", "input_mix_weight = sigmoid(self.input_mix_weight_up(silu(self.input_mix_weight_down(hyper_input_normed) / 4)))"),
         ("RN3", "mixed_input = (input_mix_weight.unflatten(-1,(4,2560)) * hyper_input_normed.unflatten(-1,(4,2560))).mean(dim=-2)"),
         ("RN4", "injection_weights = 2 * sigmoid(self.block_inject_weight(hyper_input_normed) / 4)"),
         ("RN5", "hidden_states = self.linear_attn(...) or self.self_attn(...) or self.mlp(...)"),
         ("RN6", "hidden_states = hyper_input + (hidden_states.unsqueeze(-2) * injection_weights.unsqueeze(-1)).flatten(-2)"),
         ("RN7", "shifted_tokens = [...]; ngram_ids = remainder(xor(mixed token ids), per_head_prime_vocab)"),
         ("RN8", "embeddings = self.ple_embedding(input_ids, past_key_values)  # 16 heads × 160 dims"),
         ("RN9", "key_normed = norm_key(key_proj(embeddings)).unflatten(-1,(4,2560)); value = value_proj(embeddings); gate = signed_sqrt((key_normed*query_normed).sum(-1)/sqrt(2560))"),
         ("RN10", "output = gated_value + self._short_conv(norm_conv(gated_value.flatten(-2)), past_key_values)"),
         ("RN11", "if self.ple is not None: hidden_states = hidden_states + self.ple(hidden_states, ple_input_ids, ...)")])


def cache_pair() -> str:
    ns = [
        node("KV1", "tensor-node", 20, 35, 150, 64, "prefill/decode input", "current Sq tokens"),
        node("KV2K", "op-node", 210, 5, 175, 76, "DynamicIndexedLayer.keys", "append [B,2,Sq,256]→[B,2,Skv,256]"),
        node("KV2V", "op-node", 210, 105, 175, 76, "DynamicIndexedLayer.values", "append [B,2,Sq,256]→[B,2,Skv,256]"),
        node("KV3", "op-node", 210, 205, 175, 76, "indexer_keys", "append [B,Sq,128]→[B,Skv,128]"),
        node("KV4", "op-node", 430, 45, 175, 86, "QSA mask/read", "≤2051 visible; fallback QK axis=Skv"),
        node("KV5", "op-node", 210, 335, 175, 76, "DeltaNet conv state", "[B,10240,4] / linear layer"),
        node("KV6", "op-node", 430, 335, 175, 76, "recurrent state", "[B,48,128,128] float32"),
        node("KV7", "op-node", 210, 445, 175, 86, "PLE states (layer 2 only)", "conv [B,10240,9] + ids [B,2]"),
        node("KV8", "op-node", 640, 190, 160, 86, "beam reorder / crop", "batch index_select; sequence crop"),
    ]
    fs = [flow([(170,67),(210,43)]),flow([(170,67),(190,143),(210,143)]),flow([(170,67),(190,243),(210,243)]),flow([(385,43),(430,88)]),flow([(385,143),(430,88)]),flow([(385,243),(430,88)]),flow([(170,67),(190,373),(210,373)]),flow([(385,373),(430,373)]),flow([(170,67),(190,488),(210,488)]),flow([(605,88),(640,233)]),flow([(605,373),(640,233)]),flow([(385,488),(640,233)])]
    return pair("Hybrid cache：QSA 的 dense history + DeltaNet 常数状态 + PLE", svg(ns, fs, 565),
        "sources/transformers/src/transformers/cache_utils.py", 107, 1090, "source-aligned derivation",
        [("KV1", "if use_cache and past_key_values is None: past_key_values = DynamicCache(config=self.config)"),
         ("KV2K", "self.keys = torch.cat([self.keys, key_states], dim=-2)"),
         ("KV2V", "self.values = torch.cat([self.values, value_states], dim=-2)"),
         ("KV3", "self.indexer_keys = torch.cat([self.indexer_keys, indexer_key_states], dim=1)"),
         ("KV4", "selected_token_mask = scatter(selected_token_indices, True)[..., :kv_length].unsqueeze(1); attention_mask &= selected_token_mask"),
         ("KV5", "self.conv_states[state_idx].copy_(full_conv_states[..., -conv_kernel_size:])"),
         ("KV6", "self.recurrent_states[state_idx].copy_(recurrent_states)"),
         ("KV7", "state_idx=1 stores PLE dilated-conv context length 9; state_idx=2 stores ngram token context length 2"),
         ("KV8", "keys/values/indexer_keys/conv_states/recurrent_states = tensor.index_select(0, beam_idx); crop removes rejected tail")])


def body() -> str:
    return f'''
<section id="identity"><h2>1. 身份、范围与依据</h2>
<p class="lead"><b>结论：</b>这是一个面向图像/视频/文本条件生成的 Qwen4-Exp 实验架构 checkpoint；详细分析聚焦实际驱动解码器的 <code>text_config</code>，并把视觉塔作为顶层输入路径说明。</p>
<div class="grid"><div class="card"><b>Checkpoint</b><code>Qwen/Qwen3.8-Flash-Next</code><br>revision <code>f5d08274bafd880402bd16f5e3e6c514136ec06c</code></div><div class="card"><b>入口类</b><code>Qwen4ExpForConditionalGeneration</code><br><code>model_type=qwen4_exp</code></div><div class="card"><b>本地 Transformers</b><code>5.16.0.dev0</code><br>commit <code>36deb0b53ed0863f4b4dfdea23dcaec7f3df3701</code></div><div class="card"><b>配置记录版本</b><code>5.8.0.dev0</code><br>存在版本差异；语义以本地源码为准</div></div>
<p class="source">模型配置：<a href="../../../models/Qwen/Qwen3.8-Flash-Next/config.json">config.json</a>；模型卡参数口径：<a href="../../../models/Qwen/Qwen3.8-Flash-Next/README.md#L41">README L41–71</a>；证据包：<a href="evidence.json">evidence.json</a>。</p>
<div class="callout">模型卡给出的参数口径是 <b>125B、每 token 激活 6B，另有 51B n-gram embedding 与 4B MTP</b>。权重索引记录 1658 个张量、131 个 shard、总字节约 360 GB；下载目录只保留 metadata 和 index，没有 shard，因此没有执行权重级数值验证。</div></section>

<section id="hyperparameters"><h2>2. 已配置超参数与派生维度</h2><p class="lead"><b>结论：</b>48 层按 <code>12 × (3 DeltaNet + 1 QSA)</code> 排列，隐藏状态不是单路 2560，而是层间常驻四路、拼接宽度 10240。</p>
<table><thead><tr><th>模块</th><th>配置值</th><th>派生结果</th></tr></thead><tbody>
<tr><td>文本主干</td><td>V=248320, H=2560, L=48, bf16</td><td>token embedding [248320,2560]；LM head 未 tied</td></tr>
<tr><td>层调度</td><td>原始数组每第 4 层为 <code>full_attention</code></td><td><code>__post_init__</code> 重写为 36×<code>linear_attention</code> + 12×<code>qwen_sparse_attention</code></td></tr>
<tr><td>QSA 主注意力</td><td>Nq=24, Nkv=2, head_dim=256</td><td>GQA repeat=12；Q/K rotary=256×0.25=64，non-rotary=192</td></tr>
<tr><td>QSA indexer</td><td>4 Q heads, 1 K head, dim=128, ratio=4, budget=2048</td><td>512 个完整 block；最多 2048 + 尾部 3 个 token 可见</td></tr>
<tr><td>Gated DeltaNet</td><td>16 QK heads, 48 V heads, dim=128, conv=4</td><td>key_dim=2048，value_dim=6144，conv_dim=10240；Q/K repeat 3×</td></tr>
<tr><td>MoE</td><td>E=512, K=10, Ie=640 + shared Ie=640</td><td>router [T,512]；每 token 10 routed + 1 shared SwiGLU</td></tr>
<tr><td>Gated Residual</td><td>hc_count=4, rank=320</td><td>层间状态 [B,S,10240]；每个子层读 [B,S,2560]，写回四路</td></tr>
<tr><td>PLE</td><td>one-indexed layer 2；bigram+trigram，各 8 heads</td><td>16×160=2560；embedding 51,200,245,760 参数，bf16 约 95.37 GiB</td></tr>
<tr><td>位置/上下文</td><td>max=262144, rope_theta=10,000,000, M-RoPE [11,11,10]</td><td>原生 262K；模型卡称可用 scaling 扩到 1M</td></tr>
</tbody></table>
<details><summary>48 层实际运行时调度</summary><p><code>L0–2 DeltaNet, L3 QSA</code>，此 4 层模式重复到 <code>L44–46 DeltaNet, L47 QSA</code>。PLE 只在 zero-based <code>L1</code>（配置 one-indexed layer 2）注入。</p></details></section>

<section id="top-level"><h2>3. 顶层模型</h2><p class="lead"><b>结论：</b>27-block 视觉塔把图像/视频 patch 合并成 2560 维 token，并替换序列中的 placeholder；文本 embedding 随后扩成四路 residual state，48 层后由最终 hyper mixer 收回 2560 维，再进 untied LM head。这里<b>没有标准 final RMSNorm</b>。</p>{top_level_pair()}</section>

<section id="decoder"><h2>4. Decoder 层与执行顺序</h2><p class="lead"><b>结论：</b>两类层共享同一拓扑：可选 PLE 注入 → attention-side gated residual read → mixer → gated write-back → MoE-side gated residual read → MoE → gated write-back。区别只在 mixer 是 DeltaNet 还是 QSA。</p>{decoder_pair('DL','线性注意力层（36 层）','linear_attn / GatedDeltaNet')}{decoder_pair('DF','QSA 层（12 层）','self_attn / QSA')}</section>

<section id="attention"><h2>5. Attention</h2><p class="lead"><b>结论：</b>QSA 是“微块索引 + gated GQA”。Indexer 用 4-token 平均池化，把最多 512 个 block（2048 token）加未满 block 尾部写成 mask；主注意力有 24Q/2KV、256 维 head 和 sigmoid 输出门。</p>
<div class="callout"><b>实现 caveat：</b>本地 fallback 并未先 gather 选中的 K/V；它对完整 <code>S_kv</code> 做 <code>Q @ Kᵀ</code>，再叠加 QSA mask。因此这里区分“逻辑稀疏可见集（≤2051 token）”与“fallback 物理计算轴（完整 S_kv）”。服务框架的专用 kernel 可能真正稀疏化计算，但本报告没有据此推断。</div>{full_attention_pair()}
<h3>Gated DeltaNet 线性注意力</h3><p>36 个线性层不保存 tokenwise KV 历史，而以深度卷积状态和 48 个 128×128 的 recurrent matrix 表示历史。单 token decode 选择 recurrent rule；prefill/chunk 选择 chunk rule。</p>{linear_attention_pair()}</section>

<section id="ffn-moe"><h2>6. MoE</h2><p class="lead"><b>结论：</b>每一层都有 512 个 routed experts，softmax 后 top-10 并重新归一化；共享 expert 永远执行，并由一个标量 sigmoid gate 控制。fallback 没有 capacity limit 或 token dropping。</p>{moe_pair()}
<p>每个 routed expert 的矩阵参数为 <code>2×2560×640 + 2560×640 = 4,915,200</code>；完整 routed bank 每层 <code>512×4,915,200 = 2,516,582,400</code> 参数。每 token 只有 10 个 routed expert 的 MLP 参与，再加一个 shared expert；router 本身仍计算全部 512 分数。</p></section>

<section id="residual"><h2>7. Gated Residual、归一化与 PLE</h2><p class="lead"><b>结论：</b>状态始终是四路拼接，而非标准残差。read gate 是逐元素 [4,2560]，write gate 是每路一个标量 [4]；两者都由当前 10240 维状态产生。PLE 只在第 2 层，把 hashed bigram/trigram 特征投影并注入所有四路。</p>{residual_pair()}
<p>RMSNorm 在 float32 中计算，且权重采用 <code>(1 + weight)</code> offset 参数化。PLE 的 16 个 hash head 分别使用约 20M 的不同 prime vocabulary，合计 padded vocabulary 320,001,536，每 head 160 维；源码还把这张约 95.37 GiB 的 bf16 表标为可跳过自动 device placement。</p></section>

<section id="kv-cache"><h2>8. KV Cache</h2><p class="lead"><b>结论：</b>这是三类存储并存的 hybrid cache：12 个 QSA 层保存 dense K/V + dense indexer keys；36 个 DeltaNet 层保存常数大小卷积/递归状态；第 2 层 PLE 额外保存 9-token 卷积上下文和 2 个 token id。</p>{cache_pair()}
<table><thead><tr><th>存储</th><th>每层形状</th><th>B=1 估算</th></tr></thead><tbody>
<tr><td>QSA K/V</td><td>K,V 各 [B,2,S,256]</td><td>1024 bf16 elements / token / QSA layer</td></tr>
<tr><td>QSA indexer</td><td>[B,S,128]</td><td>128 bf16 elements / token / QSA layer</td></tr>
<tr><td>12 层合计 tokenwise cache</td><td>12×1152 elements/token</td><td>27 KiB/token；32K≈0.844 GiB，128K≈3.375 GiB，262144≈6.75 GiB</td></tr>
<tr><td>DeltaNet recurrent</td><td>[B,48,128,128] float32</td><td>3 MiB/layer；36 层≈108 MiB</td></tr>
<tr><td>DeltaNet conv</td><td>[B,10240,4] bf16</td><td>80 KiB/layer；36 层≈2.813 MiB</td></tr>
<tr><td>PLE extra</td><td>[B,10240,9] bf16 + [B,2] int64</td><td>约 180 KiB，仅 L1</td></tr>
</tbody></table><p class="source">估算不含 allocator、mask、临时广播、position_ids 与 fused-kernel workspace；静态 cache 会按 max length 预分配。</p></section>

<section id="sources"><h2>9. 源码对应与可编辑图</h2><p class="lead"><b>结论：</b>运行实现是自动生成的 <code>modeling_qwen4_exp.py</code>；<code>modular_qwen4_exp.py</code>解释了它从 Qwen3.5/Qwen3-Next 家族继承的作者层结构。所有关键运行分支都在本地 generated modeling 文件中可直接追踪。</p>
<table><thead><tr><th>符号</th><th>位置</th><th>作用</th></tr></thead><tbody>
<tr><td>Qwen4ExpTextConfig.__post_init__</td><td><a href="../../../sources/transformers/src/transformers/models/qwen4_exp/configuration_qwen4_exp.py#L166">configuration L166</a></td><td>把 full_attention 重写成 qwen_sparse_attention；设 3 个 conv-state slots</td></tr>
<tr><td>Qwen4ExpTextGatedDeltaNet.forward</td><td><a href="../../../sources/transformers/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py#L450">modeling L450</a></td><td>线性注意力、卷积与 recurrent cache</td></tr>
<tr><td>Qwen4ExpTextQSAIndexer.forward</td><td><a href="../../../sources/transformers/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py#L631">modeling L631</a></td><td>微块池化、top-512 和 selected-token mask</td></tr>
<tr><td>Qwen4ExpTextAttention.forward</td><td><a href="../../../sources/transformers/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py#L785">modeling L785</a></td><td>gated GQA、RoPE、KV update、attention interface</td></tr>
<tr><td>Qwen4ExpTextSparseMoeBlock.forward</td><td><a href="../../../sources/transformers/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py#L927">modeling L927</a></td><td>routed/shared expert 并行路径</td></tr>
<tr><td>Qwen4ExpTextGatedResidual.forward</td><td><a href="../../../sources/transformers/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py#L952">modeling L952</a></td><td>逐元素 read gate 与逐路 write gate</td></tr>
<tr><td>Qwen4ExpTextPLELayer.forward</td><td><a href="../../../sources/transformers/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py#L1169">modeling L1169</a></td><td>n-gram features、stream gating 与 dilated conv</td></tr>
<tr><td>Qwen4ExpTextModel.forward</td><td><a href="../../../sources/transformers/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py#L1337">modeling L1337</a></td><td>mask、position cache、4-stream loop、最终 mix</td></tr>
<tr><td>DynamicIndexedLayer</td><td><a href="../../../sources/transformers/src/transformers/cache_utils.py#L319">cache_utils L319</a></td><td>dense K/V 外加 indexer key history</td></tr>
</tbody></table>
<h3>Editable Excalidraw scenes</h3><ul><li><a href="diagrams/01-top-level.excalidraw">01 top-level</a></li><li><a href="diagrams/02-decoder-linear.excalidraw">02 decoder linear</a></li><li><a href="diagrams/03-decoder-full-attention.excalidraw">03 decoder QSA</a></li><li><a href="diagrams/04-full-attention.excalidraw">04 QSA attention</a></li><li><a href="diagrams/05-gated-deltanet.excalidraw">05 Gated DeltaNet</a></li><li><a href="diagrams/06-moe.excalidraw">06 MoE</a></li><li><a href="diagrams/07-residual-norm.excalidraw">07 Gated Residual + PLE</a></li><li><a href="diagrams/08-kv-cache.excalidraw">08 cache</a></li></ul></section>

<section id="unknowns"><h2>10. 假设、兼容性与未决项</h2><p class="lead"><b>结论：</b>结构与形状已由 config、source 和 weight index 三方交叉核对；仍未验证的是权重数值、专用 serving kernel 的真实 gather/fusion 布局，以及训练时 MTP 的执行实现。</p><ul>
<li>本报告的运行语义来自 Transformers <code>5.16.0.dev0</code> / commit <code>36deb0b5</code>，checkpoint 记录 <code>5.8.0.dev0</code>；版本差异已显式保留。</li>
<li>checkpoint index 含 <code>mtp.*</code>，但本地 conditional-generation 类将其列为 unexpected/ignored，主生成 forward 不实例化 MTP；因此仅记录 4B MTP 权重事实，不把它画入推理主路径。</li>
<li>QSA 的专用 vLLM/SGLang/TokenSpeed kernel 可能在 block 选择后真正 gather K/V。本地 Transformers fallback 只产生 mask，报告没有把外部 kernel 行为当作已验证事实。</li>
<li>参数和 cache 字节估算采用 config dtype；权重 shards 未下载，未逐张量验证 dtype，也未运行 logits/吞吐 benchmark。</li>
</ul><p class="source">分析日期：2026-08-27（Asia/Shanghai）。本报告区分 inference-time 当前 token 维度 <code>S_q</code>、历史 <code>S_past</code> 与完整物理 cache 维度 <code>S_kv</code>。</p></section>
'''


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    template = TEMPLATE.read_text(encoding="utf-8")
    rendered = template.replace("{{MODEL_TITLE}}", "Qwen3.8-Flash-Next · 实例化架构分析")
    rendered = rendered.replace("{{GENERATED_AT}}", "2026-08-27 · Asia/Shanghai")
    rendered = rendered.replace("{{REPORT_BODY}}", body())
    rendered = rendered.replace(
        "main { min-width:0; }",
        "main { min-width:0; overflow:hidden; }\n"
        "    section,.card,.graph-panel,.code-panel { max-width:100%; overflow-wrap:anywhere; }\n"
        "    .source a,.code-panel h4 a { word-break:break-all; }",
    )
    OUT.write_text(rendered, encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
