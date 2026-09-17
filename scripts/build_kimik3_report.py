#!/usr/bin/env python3
"""Build the source-aligned Kimi K3 architecture report from local evidence."""

from __future__ import annotations

import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/moonshotai/Kimi-K3"
REPORT = ROOT / "reports/moonshotai/Kimi-K3"
TEMPLATE = ROOT / ".agents/skills/hf-model-architecture/assets/report-template.html"
SRC_K3 = "../../../models/moonshotai/Kimi-K3/modeling_kimi_k3.py"
SRC_LINEAR = "../../../models/moonshotai/Kimi-K3/modeling_kimi_linear.py"
SRC_CONFIG = "../../../models/moonshotai/Kimi-K3/configuration_kimi_k3.py"


def node(node_id, x, y, w, h, kind, *lines):
    return {"id": node_id, "x": x, "y": y, "w": w, "h": h, "kind": kind, "lines": lines}


def svg(nodes, edges, height, marker):
    parts = [
        f'<svg class="architecture-graph" viewBox="0 0 760 {height}">',
        f'<defs><marker id="{marker}" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="currentColor"/></marker></defs>',
    ]
    lookup = {item["id"]: item for item in nodes}
    for start, end in edges:
        a, b = lookup[start], lookup[end]
        x1, y1 = a["x"] + a["w"] / 2, a["y"] + a["h"]
        x2, y2 = b["x"] + b["w"] / 2, b["y"]
        mid = (y1 + y2) / 2
        parts.append(f'<path class="flow" marker-end="url(#{marker})" d="M{x1},{y1} V{mid} H{x2} V{y2}"/>')
    for item in nodes:
        cls = "tensor-node" if item["kind"] == "tensor" else "op-node"
        rx = ' rx="12"' if item["kind"] == "tensor" else ""
        parts.append(f'<g class="{cls}"><rect x="{item["x"]}" y="{item["y"]}" width="{item["w"]}" height="{item["h"]}"{rx}/>' )
        total = len(item["lines"])
        start_y = item["y"] + item["h"] / 2 - (total - 1) * 8
        for index, line in enumerate(item["lines"]):
            css = "node-id" if index == 0 else ("shape-label" if "[" in line or "→" in line or "×" in line else "")
            parts.append(f'<text class="{css}" x="{item["x"] + item["w"] / 2}" y="{start_y + index * 17}"><tspan class="{css}">{html.escape(str(line))}</tspan></text>')
        parts.append('</g>')
    parts.append('</svg>')
    return "".join(parts)


def code_panel(title, path, line_start, line_end, rows, kind="trimmed verbatim"):
    rendered = []
    for node_id, source in rows:
        rendered.append(f'<span class="code-map">{html.escape(node_id)}</span>{html.escape(source)}')
    code = "\n".join(rendered)
    return (
        f'<div class="code-panel"><h4><a href="{path}#L{line_start}">{html.escape(title)}</a> '
        f'· L{line_start}–L{line_end}<span class="excerpt-kind">{kind}</span></h4>'
        f'<pre><code>{code}</code></pre></div>'
    )


def pair(title, graph, panel):
    return (
        '<div class="graph-code-pair"><div class="graph-panel">'
        f'<h4>{html.escape(title)}</h4><div class="graph-legend"><span><i class="tensor-key"></i>张量</span><span><i></i>算子</span></div>'
        f'<div class="graph-scroll">{graph}</div></div>{panel}</div>'
    )


def build_body():
    top_nodes = [
        node("TL1", 30, 35, 220, 68, "tensor", "TL1 input_ids", "[B,S]"),
        node("TL2", 510, 35, 220, 68, "tensor", "TL2 pixel_values", "[L_img,3,14,14] patches"),
        node("TL3", 500, 145, 240, 88, "op", "TL3 vision_tower", "27× MoonViT, H_v=1024", "12 heads × 128"),
        node("TL4", 500, 275, 240, 88, "op", "TL4 mm_projector", "2×2 merge: 4096 → 7168", "image_features [N_img,7168]"),
        node("TL5", 25, 275, 230, 88, "op", "TL5 embed_tokens", "[B,S] × [163840,7168]", "→ [B,S,7168]"),
        node("TL6", 260, 410, 240, 88, "op", "TL6 merge image/text", "placeholder expansion", "→ inputs_embeds [B,S*,7168]"),
        node("TL7", 260, 545, 240, 88, "op", "TL7 KimiLinearModel", "93 decoder layers", "69 KDA + 24 MLA"),
        node("TL8", 260, 680, 240, 78, "op", "TL8 norm + lm_head", "7168 → 163840", "logits [B,S*,V]"),
    ]
    top = svg(top_nodes, [("TL2","TL3"),("TL3","TL4"),("TL1","TL5"),("TL4","TL6"),("TL5","TL6"),("TL6","TL7"),("TL7","TL8")], 800, "top-arrow")
    top_code = code_panel("KimiK3ForConditionalGeneration.__init__/forward", SRC_K3, 899, 1218, [
        ("TL1", "input_ids: torch.LongTensor | None = None,"),
        ("TL2", "pixel_values: torch.FloatTensor | list[torch.FloatTensor] | None = None,"),
        ("TL3", "self.vision_tower = MoonViT3dPretrainedModel(vt_config)"),
        ("TL4", "self.mm_projector = PatchMergerMLPV2(proj_config)"),
        ("TL7", "self.language_model = KimiLinearForCausalLM(config.text_config)"),
        ("TL5", "inputs_embeds = self.get_input_embeddings()(input_ids)"),
        ("TL3", "image_features = self._extract_image_features(pixel_values, grid_thws)"),
        ("TL4", "image_features = self.mm_projector(image_features)"),
        ("TL6", "inputs_embeds, attention_mask, labels, position_ids = self._merge_input_ids_with_image_features(...)"),
        ("TL7", "outputs = self.language_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, ...)"),
        ("TL8", "# KimiLinearForCausalLM: logits = self.lm_head(self.model(...)[0])"),
    ], "ordered trimmed verbatim")

    dl_nodes = [
        node("DL1", 255, 25, 250, 68, "tensor", "DL1 hidden_states", "[B,S,7168]"),
        node("DL2", 255, 125, 250, 76, "op", "DL2 attention-res mix", "bank + prefix softmax mix", "[B,S,7168]"),
        node("DL3", 255, 235, 250, 68, "op", "DL3 input_layernorm", "[B,S,7168] → same"),
        node("DL4", 255, 335, 250, 82, "op", "DL4 KimiDeltaAttention", "KDA recurrent operator", "[B,S,7168] → same"),
        node("DL5", 255, 455, 250, 76, "op", "DL5 prefix_sum + attn", "[B,S,7168] + same"),
        node("DL6", 255, 565, 250, 76, "op", "DL6 MLP-res mix + norm", "bank + prefix; RMSNorm", "[B,S,7168]"),
        node("DL7", 255, 675, 250, 82, "op", "DL7 dense FFN / latent MoE", "layer 0 dense; layers 1–92 MoE", "[B,S,7168] → same"),
        node("DL8", 255, 795, 250, 76, "tensor", "DL8 prefix_sum", "[B,S,7168]"),
    ]
    dl = svg(dl_nodes, [(f"DL{i}",f"DL{i+1}") for i in range(1,8)], 900, "dl-arrow")
    dl_code = code_panel("KimiDecoderLayer._forward_attn_residual（KDA 分支）", SRC_LINEAR, 973, 1046, [
        ("DL1", "prefix_sum = hidden_states"),
        ("DL2", "hidden_states = _apply_attn_res(prefix_sum.view(-1, hidden_size), block_residual, self.self_attention_res_proj, self.self_attention_res_norm)"),
        ("DL3", "hidden_states = self.input_layernorm(hidden_states)"),
        ("DL4", "hidden_states = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask, cache_params=past_key_values, ...)"),
        ("DL5", "prefix_sum = prefix_sum + hidden_states"),
        ("DL6", "hidden_states = _apply_attn_res(prefix_sum.view(-1, hidden_size), block_residual, self.mlp_res_proj, self.mlp_res_norm)"),
        ("DL6", "hidden_states = self.post_attention_layernorm(hidden_states)"),
        ("DL7", "hidden_states = self.block_sparse_moe(hidden_states) if hasattr(self, 'block_sparse_moe') else self.mlp(hidden_states)"),
        ("DL8", "prefix_sum = prefix_sum + hidden_states"),
    ])

    df_nodes = [
        node("DF1", 255, 30, 250, 68, "tensor", "DF1 hidden_states", "[B,S,7168]"),
        node("DF2", 255, 140, 250, 76, "op", "DF2 attention-res mix", "bank + prefix → [B,S,7168]"),
        node("DF3", 255, 260, 250, 68, "op", "DF3 input_layernorm", "[B,S,7168] → same"),
        node("DF4", 255, 370, 250, 82, "op", "DF4 KimiMLAAttention", "causal dense attention", "[B,S,7168] → same"),
        node("DF5", 255, 500, 250, 76, "op", "DF5 prefix_sum + attn", "[B,S,7168] + same"),
        node("DF6", 255, 620, 250, 76, "op", "DF6 mix → norm → MoE", "896选16 + shared expert"),
        node("DF7", 255, 740, 250, 68, "tensor", "DF7 prefix_sum", "[B,S,7168]"),
    ]
    df = svg(df_nodes, [(f"DF{i}",f"DF{i+1}") for i in range(1,7)], 850, "df-arrow")
    df_code = code_panel("KimiDecoderLayer._forward_attn_residual（MLA 分支）", SRC_LINEAR, 883, 1046, [
        ("DF1", "prefix_sum = hidden_states"),
        ("DF2", "hidden_states = _apply_attn_res(prefix_sum.view(-1, hidden_size), block_residual, self.self_attention_res_proj, self.self_attention_res_norm)"),
        ("DF3", "hidden_states = self.input_layernorm(hidden_states)"),
        ("DF4", "# configured when not is_kda_layer: self.self_attn = KimiMLAAttention(config, layer_idx)"),
        ("DF4", "hidden_states = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask, past_key_values=past_key_values, ...)"),
        ("DF5", "prefix_sum = prefix_sum + hidden_states"),
        ("DF6", "hidden_states = self.post_attention_layernorm(_apply_attn_res(...))"),
        ("DF6", "hidden_states = self.block_sparse_moe(hidden_states)"),
        ("DF7", "prefix_sum = prefix_sum + hidden_states"),
    ], "ordered trimmed verbatim")

    fa_nodes = [
        node("FA1", 255, 20, 250, 66, "tensor", "FA1 hidden_states", "[B,Sq,7168]"),
        node("FA2", 20, 135, 210, 90, "op", "FA2 q_a → norm → q_b", "7168→1536→18432", "q [B,96,Sq,192]"),
        node("FA3", 275, 135, 210, 90, "op", "FA3 kv_a_proj_with_mqa", "7168→576 = 512+64", "compressed_kv [B,Sq,576]"),
        node("FA4", 530, 135, 210, 90, "op", "FA4 g_proj.sigmoid", "7168→12288", "gate [B,Sq,12288]"),
        node("FA5", 275, 275, 210, 98, "op", "FA5 kv_b_proj", "512→24576", "K_nope,V [B,96,Sq,128]"),
        node("FA6", 20, 425, 210, 92, "tensor", "FA6 query_states", "cat 128+64", "[B,96,Sq,192]"),
        node("FA7", 275, 425, 210, 92, "op", "FA7 cache.update", "K [B,96,Skv,192]", "V [B,96,Skv,128]"),
        node("FA8", 110, 565, 250, 94, "op", "FA8 scores = einsum", "[B,96,Sq,192] × Kᵀ", "→ [B,96,Sq,Skv]"),
        node("FA9", 110, 705, 250, 94, "op", "FA9 softmax; probs @ V", "[B,96,Sq,Skv] × V", "→ [B,Sq,96,128]"),
        node("FA10", 360, 830, 250, 94, "op", "FA10 output gate ×", "[B,Sq,12288] × gate", "→ [B,Sq,12288]"),
        node("FA11", 360, 965, 250, 82, "op", "FA11 o_proj", "12288→7168", "[B,Sq,7168]"),
    ]
    fa_edges = [("FA1","FA2"),("FA1","FA3"),("FA1","FA4"),("FA3","FA5"),("FA2","FA6"),("FA5","FA7"),("FA6","FA8"),("FA7","FA8"),("FA8","FA9"),("FA7","FA9"),("FA9","FA10"),("FA4","FA10"),("FA10","FA11")]
    fa = svg(fa_nodes, fa_edges, 1090, "fa-arrow")
    fa_code = code_panel("KimiMLAAttention.forward + eager fallback", SRC_LINEAR, 418, 473, [
        ("FA1", "batch_size, seq_length = hidden_states.shape[:-1]"),
        ("FA2", "q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))"),
        ("FA2", "q_states = q_states.view(query_shape).transpose(1, 2)"),
        ("FA3", "compressed_kv = self.kv_a_proj_with_mqa(hidden_states)"),
        ("FA5", "k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(key_shape).transpose(1, 2)"),
        ("FA6", "query_states = torch.cat((q_pass, q_rot), dim=-1)"),
        ("FA7", "key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)"),
        ("FA8", "scores = torch.einsum('bhqd,bhkd->bhqk', query, key) * scaling"),
        ("FA9", "probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)"),
        ("FA9", "out = torch.einsum('bhqk,bhkd->bhqd', probs, value).transpose(1, 2).contiguous()"),
        ("FA4", "g = self.g_proj(hidden_states).sigmoid()"),
        ("FA10", "attn_output = attn_output * g"),
        ("FA11", "attn_output = self.o_proj(attn_output)"),
    ], "ordered trimmed verbatim + eager fallback")

    la_nodes = [
        node("LA1", 255, 20, 250, 66, "tensor", "LA1 hidden_states", "[B,Sq,7168]"),
        node("LA2", 15, 125, 220, 92, "op", "LA2 q/k/v_proj", "3 × (7168→12288)", "[B,Sq,12288] each"),
        node("LA3", 270, 125, 220, 92, "op", "LA3 f_a/f_b + b_proj", "g [B,Sq,96,128]", "beta [B,Sq,96]"),
        node("LA4", 525, 125, 220, 92, "op", "LA4 g_proj", "7168→12288", "output gate [B,Sq,96,128]"),
        node("LA5", 15, 275, 220, 92, "op", "LA5 q/k/v_conv1d", "kernel=4; SiLU", "q,k,v [B,Sq,96,128]"),
        node("LA6", 145, 430, 300, 110, "op", "LA6 chunk_kda / recurrent", "Q,K,V,g,beta + state", "→ o [B,Sq,96,128]", "+ fixed recurrent state"),
        node("LA7", 400, 605, 250, 92, "op", "LA7 o_norm(o,g)", "FusedRMSNormGated", "[B,Sq,96,128]"),
        node("LA8", 400, 745, 250, 82, "op", "LA8 reshape + o_proj", "12288→7168", "[B,Sq,7168]"),
    ]
    la_edges = [("LA1","LA2"),("LA1","LA3"),("LA1","LA4"),("LA2","LA5"),("LA5","LA6"),("LA3","LA6"),("LA6","LA7"),("LA4","LA7"),("LA7","LA8")]
    la = svg(la_nodes, la_edges, 870, "la-arrow")
    la_code = code_panel("KimiDeltaAttention.forward", SRC_LINEAR, 580, 659, [
        ("LA1", "batch_size, q_len, _ = hidden_states.shape"),
        ("LA2", "q_proj_states, k_proj_states, v_proj_states = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)"),
        ("LA5", "q, conv_state_q = self.q_conv1d(x=q_proj_states, cache=conv_state_q, output_final_state=use_cache, ...)"),
        ("LA5", "k, conv_state_k = self.k_conv1d(...); v, conv_state_v = self.v_conv1d(...)"),
        ("LA3", "g = rearrange(self.f_b_proj(self.f_a_proj(hidden_states)), '... (h d) -> ... h d', d=self.head_dim)"),
        ("LA3", "beta = self.b_proj(hidden_states).float()"),
        ("LA6", "o, recurrent_state = chunk_kda(q=q, k=k, v=v, g=g, beta=beta, A_log=self.A_log, dt_bias=self.dt_bias, initial_state=recurrent_state, ...)"),
        ("LA6", "# Sq==1 with cache selects fused_recurrent_kda(...)"),
        ("LA4", "g = self.g_proj(hidden_states)"),
        ("LA7", "o = self.o_norm(o, g)"),
        ("LA8", "o = self.o_proj(rearrange(o, 'b t h d -> b t (h d)'))"),
    ], "ordered trimmed verbatim; fla-core kernel interface")

    moe_nodes = [
        node("MOE1", 255, 20, 250, 66, "tensor", "MOE1 hidden_states", "[B,S,7168]; T=B·S"),
        node("MOE2", 20, 125, 210, 92, "op", "MOE2 gate F.linear", "[T,7168] × [896,7168]", "logits [T,896]"),
        node("MOE3", 275, 125, 210, 92, "op", "MOE3 payload down_proj", "7168→3584", "payload [T,3584]"),
        node("MOE4", 530, 125, 210, 92, "op", "MOE4 shared_experts", "7168→2×6144→7168", "shared [B,S,7168]"),
        node("MOE5", 20, 275, 210, 102, "op", "MOE5 sigmoid + bias + topk", "[T,896] → ids [T,16]", "weights [T,16], renorm"),
        node("MOE6", 160, 435, 300, 110, "op", "MOE6 dispatch to 896 experts", "Σe Ne=T·16", "each 3584→2×3072→3584", "SiTU(w1,w3) → w2"),
        node("MOE7", 160, 595, 300, 102, "op", "MOE7 weighted scatter/sum", "[T,16,3584] × [T,16,1]", "→ [T,3584]"),
        node("MOE8", 300, 755, 280, 92, "op", "MOE8 norm + up + add shared", "3584→7168 + shared", "[B,S,7168]"),
    ]
    moe_edges = [("MOE1","MOE2"),("MOE1","MOE3"),("MOE1","MOE4"),("MOE2","MOE5"),("MOE3","MOE6"),("MOE5","MOE6"),("MOE6","MOE7"),("MOE5","MOE7"),("MOE7","MOE8"),("MOE4","MOE8")]
    moe = svg(moe_nodes, moe_edges, 890, "moe-arrow")
    moe_code = code_panel("KimiMoEGate + KimiSparseMoeBlock", SRC_LINEAR, 703, 873, [
        ("MOE1", "hidden_states = hidden_states.view(-1, h)"),
        ("MOE2", "logits = F.linear(hidden_states.float(), self.weight.float(), None)"),
        ("MOE5", "scores = logits.sigmoid(); scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)"),
        ("MOE5", "_, topk_idx = torch.topk(scores_for_choice, k=16, dim=-1, sorted=False)"),
        ("MOE5", "topk_weight = scores.gather(1, topk_idx); topk_weight /= topk_weight.sum(-1, keepdim=True) + 1e-20"),
        ("MOE3", "hidden_states = self.routed_expert_down_proj(hidden_states)"),
        ("MOE6", "sorted_tokens = x[topk_ids.view(-1).argsort() // topk_ids.shape[1]]; expert_out = expert(tokens_for_this_expert)"),
        ("MOE7", "final_out = new_x.view(*topk_ids.shape, -1).mul_(topk_weight.unsqueeze(-1)).sum(dim=1)"),
        ("MOE8", "y = self.routed_expert_up_proj(self.routed_expert_norm(y))"),
        ("MOE4", "y = y + self.shared_experts(identity)"),
    ], "ordered trimmed verbatim")

    rn_nodes = [
        node("RN1", 245, 20, 270, 72, "tensor", "RN1 prefix_sum", "[T,7168]"),
        node("RN2", 20, 145, 260, 90, "tensor", "RN2 block_residual", "[T,M,7168]", "M grows at layers 0,12,…,84"),
        node("RN3", 360, 145, 280, 90, "op", "RN3 cat candidates", "v=[bank,prefix]", "[T,M+1,7168]"),
        node("RN4", 360, 285, 280, 100, "op", "RN4 norm + score", "scores=(RMS(v)·(norm.weight×proj.weight))", "[T,M+1]"),
        node("RN5", 360, 435, 280, 96, "op", "RN5 softmax + matmul", "[T,1,M+1] × [T,M+1,7168]", "→ [T,7168]"),
        node("RN6", 235, 590, 290, 90, "op", "RN6 prefix accumulation", "prefix_sum += attn / MLP", "[B,S,7168]"),
    ]
    rn_edges = [("RN1","RN3"),("RN2","RN3"),("RN3","RN4"),("RN4","RN5"),("RN1","RN6"),("RN5","RN6")]
    rn = svg(rn_nodes, rn_edges, 730, "rn-arrow")
    rn_code = code_panel("_apply_attn_res + block boundary", SRC_LINEAR, 987, 1088, [
        ("RN1", "prefix_sum = hidden_states"),
        ("RN2", "if self.layer_idx % self.attn_res_block_size == 0: block_residual = torch.cat([block_residual, prefix_sum.view(-1, hidden_size).unsqueeze(1)], dim=1)"),
        ("RN3", "v = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)"),
        ("RN4", "k = v.float() * torch.rsqrt(v.float().pow(2).mean(-1, keepdim=True) + norm.variance_epsilon)"),
        ("RN4", "scores = (k * (norm.weight.float() * proj.weight.squeeze(0).float())).sum(-1)"),
        ("RN5", "probs = scores.softmax(-1).unsqueeze(1)"),
        ("RN5", "hidden_states = torch.matmul(probs, v.float()).squeeze(1)"),
        ("RN6", "prefix_sum = prefix_sum + hidden_states"),
    ], "ordered trimmed verbatim")

    kv_nodes = [
        node("KV1", 20, 20, 220, 88, "tensor", "KV1 MLA new K/V", "K [B,96,Sq,192]", "V [B,96,Sq,128]"),
        node("KV2", 20, 165, 220, 98, "op", "KV2 torch.cat(dim=2)", "past + current", "S_kv=S_past+Sq"),
        node("KV3", 20, 320, 220, 92, "tensor", "KV3 dense MLA cache", "K [B,96,S_kv,192]", "V [B,96,S_kv,128]"),
        node("KV4", 500, 20, 220, 88, "tensor", "KV4 KDA current", "q/k/v conv inputs", "[B,Sq,12288]"),
        node("KV5", 500, 165, 220, 102, "op", "KV5 ShortConvolution", "read/update q,k,v states", "kernel=4; fixed-size"),
        node("KV6", 500, 325, 220, 100, "op", "KV6 KDA kernel", "read/update recurrent_state", "sequence-length independent"),
        node("KV7", 245, 505, 270, 100, "op", "KV7 reorder_cache", "index_select batch axis", "MLA K/V + KDA conv/recurrent"),
    ]
    kv_edges = [("KV1","KV2"),("KV2","KV3"),("KV4","KV5"),("KV5","KV6"),("KV3","KV7"),("KV6","KV7")]
    kv = svg(kv_nodes, kv_edges, 650, "kv-arrow")
    kv_code = code_panel("KimiDynamicCache.update + KDA state update", SRC_LINEAR, 157, 649, [
        ("KV1", "def update(self, key_states, value_states, layer_idx, cache_kwargs=None):"),
        ("KV2", "self.key_cache[layer_idx] = key_states if old is None else torch.cat([old, key_states], dim=2)"),
        ("KV2", "self.value_cache[layer_idx] = value_states if old is None else torch.cat([old, value_states], dim=2)"),
        ("KV3", "return self.key_cache[layer_idx], self.value_cache[layer_idx]"),
        ("KV4", "q_proj_states, k_proj_states, v_proj_states = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)"),
        ("KV5", "q, conv_state_q = self.q_conv1d(x=q_proj_states, cache=conv_state_q, output_final_state=use_cache, ...)"),
        ("KV6", "o, recurrent_state = fused_recurrent_kda(..., initial_state=recurrent_state, output_final_state=True, ...)"),
        ("KV6", "cache_params.recurrent_states[self.layer_idx] = recurrent_state; cache_params.conv_states[self.layer_idx] = (conv_state_q, conv_state_k, conv_state_v)"),
        ("KV7", "cache_tensor = cache_tensor.index_select(0, beam_idx)"),
    ], "ordered trimmed verbatim; external state layout unresolved")

    diagrams = [
        ("01-top-level.excalidraw", "顶层多模态路径"), ("02-decoder-linear.excalidraw", "KDA Decoder 层"),
        ("03-decoder-full-attention.excalidraw", "MLA Decoder 层"), ("04-full-attention.excalidraw", "MLA 全注意力"),
        ("05-gated-deltanet.excalidraw", "KDA 线性注意力"), ("06-moe.excalidraw", "Latent MoE"),
        ("07-residual-norm.excalidraw", "Attention Residual"), ("08-kv-cache.excalidraw", "混合缓存"),
    ]
    diagram_links = "".join(f'<li><a href="diagrams/{name}">{label}</a></li>' for name, label in diagrams)
    full_layers = "4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,84,88,92,93（1-based）"
    return f'''
<section id="identity"><h2>1. 身份、范围与结论摘要</h2>
<p class="lead"><b>结论：</b>Kimi K3 是一个视觉—语言条件生成模型；文本核心是 93 层、隐藏维 7168 的混合 Decoder，由 69 个 Kimi Delta Attention（KDA）层与 24 个 MLA 风格全注意力层组成。它同时使用 attention-residual 候选库、第一层稠密 SiTU FFN，以及后 92 层的 896 选 16 latent MoE。</p>
<div class="grid"><div class="card"><b>Checkpoint</b><code>moonshotai/Kimi-K3</code><br>a590ce090cb0…</div><div class="card"><b>模型规模索引</b>1.560 TB 参数文件总字节（96 shards；权重未下载）</div><div class="card"><b>文本层</b>93 = 69 KDA + 24 MLA</div><div class="card"><b>上下文配置</b>1,048,576 tokens</div></div>
<p class="callout"><b>来源边界：</b>本地 Transformers 提交 <code>36deb0b53ed0…</code> 没有 <code>kimi_k3</code>/<code>kimi_linear</code> 原生实现；本报告以 checkpoint 仓库 revision 随附的 remote code 为权威执行语义。配置要求 Transformers ≥4.56.0，并记录版本 4.56.2。</p>
<p class="source">分析日期：2026-08-28。形状默认推理态；<code>B</code> 批量，<code>S_q</code> 当前调用 token 数，<code>S_past</code> 历史长度，<code>T=B·S_q</code>。</p></section>

<section id="hyperparameters"><h2>2. 实际生效的配置与层调度</h2>
<p class="lead"><b>结论：</b>配置中的层列表是 1-based，源码用 <code>(layer_idx+1)</code> 查询。MLA 层为 {full_layers}；其余 69 层是 KDA。第 0 层用稠密 FFN，层 1–92 每层均实例化 MoE。</p>
<table><thead><tr><th>类别</th><th>配置值</th><th>派生/解释</th></tr></thead><tbody>
<tr><td>词表 / embedding</td><td>V=163840, H=7168, untied</td><td>embedding [163840,7168]；LM head [163840,7168]</td></tr>
<tr><td>MLA</td><td>Nq=Nkv=96; q-rank=1536; kv-rank=512</td><td>Dqk=128+64=192；Dv=128；Q out=18432；KV-b out=24576</td></tr>
<tr><td>KDA</td><td>96 heads, D=128, conv kernel=4</td><td>q/k/v projection 各 12288；prefill chunk、单 token decode recurrent</td></tr>
<tr><td>MoE</td><td>E=896, K=16, latent=3584, Ie=3072</td><td>router [T,896]；共享专家等效 intermediate=6144</td></tr>
<tr><td>稠密 FFN</td><td>I=33792, SiTU β=4, linear β=25</td><td>仅 layer_idx=0；7168→2×33792→7168</td></tr>
<tr><td>残差 / norm</td><td>block size=12; RMS eps=1e-5</td><td>边界层 0,12,…,84 保存 8 个 bank 候选</td></tr>
<tr><td>视觉</td><td>27 layers, H=1024, QKV=1536, 12 heads</td><td>14×14 patch；2×2 merger 将 4×1024 投影到 7168</td></tr>
<tr><td>量化</td><td>MXFP4 group=32</td><td>routed expert w1/w2/w3 在 index 中为 packed；attention、shared expert、LM head 等被 ignore</td></tr>
</tbody></table>
<details><summary>关键维度公式</summary><pre>MLA: D_qk = D_nope + D_rope = 128 + 64 = 192
Q projection = 96 × 192 = 18,432
KV-b projection = 96 × (128 + 128) = 24,576
KDA projection = 96 × 128 = 12,288
MLA cache elements/token/layer = 96 × (192 + 128) = 30,720</pre></details></section>

<section id="top-level"><h2>3. 顶层模型图 + 源码</h2><p class="lead"><b>结论：</b>图像走 MoonViT→2×2 patch merger，文本走 embedding；placeholder 被扩展为图像 token 后，统一送入同一个 KimiLinear Causal LM。纯文本时视觉支路不执行。</p>{pair("多模态输入到 logits", top, top_code)}</section>

<section id="decoder"><h2>4. 两种 Decoder 层</h2><p class="lead"><b>结论：</b>KDA 与 MLA 层共享同一 attention-residual + pre-norm + FFN/MoE 骨架，只替换 self-attention 算子和 mask。KDA 使用二维 padding mask；MLA 使用 causal mask。</p>{pair("KDA 层（69 层）", dl, dl_code)}{pair("MLA 全注意力层（24 层）", df, df_code)}</section>

<section id="attention"><h2>5. Attention：MLA 投影与 KDA 递归</h2>
<p class="lead"><b>结论：</b>全注意力层采用 MLA 风格低秩投影，但运行时展开为 96 个 K/V 头并把展开张量写入缓存，因此不是节省 KV cache 的 latent-cache 实现。<code>q_rot/k_rot</code> 虽保留 64 维命名，本 remote code 没有调用 RoPE，<code>position_ids</code> 也未在 MLA forward 中使用；这是一项必须显式保留的兼容性疑点。</p>
{pair("MLA 风格全注意力：Q / KV / gate 并行", fa, fa_code)}
<p class="lead"><b>KDA 结论：</b>69 层不形成 <code>[S_q,S_kv]</code> 注意力矩阵；Q/K/V 先经 kernel=4 的短卷积，再交给 <code>fla-core</code> 的 KDA 核。prefill/chunk 调 <code>chunk_kda</code>，缓存存在且 <code>S_q=1</code> 时调 <code>fused_recurrent_kda</code>。</p>
{pair("Kimi Delta Attention：短卷积 + 固定状态", la, la_code)}</section>

<section id="ffn-moe"><h2>6. FFN / MoE</h2><p class="lead"><b>结论：</b>router 和 token payload 独立分叉；选择分数是 <code>sigmoid(logits)+correction_bias</code>，但实际权重从未加 bias 的 sigmoid scores 中 gather，再归一化。<code>num_expert_group=topk_group=1</code>，所以 grouped-top-k 分支不执行。每个 token 激活 16 个 routed experts，另始终执行一个等效 2 个专家宽度的 shared expert。</p>{pair("896 选 16 latent MoE + shared expert", moe, moe_code)}<p class="source">第一层例外：<code>KimiMLP</code> 使用两条并行投影 <code>gate_proj/up_proj</code>，拼接为 [B,S,67584] 后由 SiTU 拆半并相乘，再经 <code>down_proj</code> 返回 [B,S,7168]。</p></section>

<section id="residual"><h2>7. Attention Residual 与归一化</h2><p class="lead"><b>结论：</b>这不是标准单一残差。每 12 层将一个 prefix 快照加入 bank；每次 attention 前（bank 非空时）和每次 MLP 前，用一个学习投影对 bank 与当前 prefix 的 RMS-normalized 候选打分，softmax 后加权混合。attention/MLP 输出仍加到未混合的 <code>prefix_sum</code> 上。最终输出再用 8 个 bank 快照 + 当前状态做一次 9-way 混合。</p>{pair("候选库 softmax 混合与 prefix 累积", rn, rn_code)}</section>

<section id="kv-cache"><h2>8. 混合 Cache 行为</h2><p class="lead"><b>结论：</b>24 个 MLA 层缓存随序列线性增长的展开 K/V；69 个 KDA 层只维护三组短卷积状态与一个 recurrent state，长度不随上下文增长。两类状态都支持 beam 维重排。</p>{pair("MLA 密集 KV 与 KDA 固定状态", kv, kv_code)}
<div class="callout"><b>MLA cache 量级：</b>每 token、每 MLA 层 30,720 elements；24 层、bf16 合计 1,474,560 bytes/token ≈ 1.406 MiB/token。若真的缓存满 1,048,576 token，单 batch 仅 MLA K/V 即约 <b>1,440 GiB</b>，未计分配器和临时张量。KDA 的精确 recurrent/conv 物理布局由外部 <code>fla-core</code> 决定，本仓库只能确定其固定长度性质与读写接口。</div></section>

<section id="sources"><h2>9. 源码对应</h2><p class="lead"><b>结论：</b>模型是 remote-code-only；参数 index 只佐证模块实例化与 packed 权重命名，不用于推断控制流。</p>
<table><thead><tr><th>符号</th><th>角色</th><th>来源</th></tr></thead><tbody>
<tr><td>KimiK3Config</td><td>将 top-level dict 转为 text/vision nested config</td><td><a href="{SRC_CONFIG}#L229">configuration_kimi_k3.py:229</a></td></tr>
<tr><td>KimiK3ForConditionalGeneration.forward</td><td>视觉特征、placeholder 合并、语言模型调用</td><td><a href="{SRC_K3}#L1113">modeling_kimi_k3.py:1113</a></td></tr>
<tr><td>KimiLinearModel.forward</td><td>cache、mask、93 层循环、最终 residual mix/norm</td><td><a href="{SRC_LINEAR}#L1138">modeling_kimi_linear.py:1138</a></td></tr>
<tr><td>KimiMLAAttention.forward</td><td>低秩 Q/KV、缓存、attention、output gate</td><td><a href="{SRC_LINEAR}#L405">modeling_kimi_linear.py:405</a></td></tr>
<tr><td>KimiDeltaAttention.forward</td><td>短卷积与 fla-core KDA</td><td><a href="{SRC_LINEAR}#L543">modeling_kimi_linear.py:543</a></td></tr>
<tr><td>KimiSparseMoeBlock.forward</td><td>latent payload、专家 dispatch、shared merge</td><td><a href="{SRC_LINEAR}#L815">modeling_kimi_linear.py:815</a></td></tr>
<tr><td>_apply_attn_res</td><td>候选残差 softmax 加权</td><td><a href="{SRC_LINEAR}#L1075">modeling_kimi_linear.py:1075</a></td></tr>
<tr><td>KimiDynamicCache</td><td>MLA K/V 与 KDA 状态容器</td><td><a href="{SRC_LINEAR}#L120">modeling_kimi_linear.py:120</a></td></tr>
</tbody></table></section>

<section id="unknowns"><h2>10. 假设、兼容边界与开放问题</h2><p class="lead"><b>结论：</b>配置选择、Python 调用链和外部 kernel 接口已确定；以下细节需要对应 serving runtime 或 <code>fla-core</code> 版本才能封闭。</p><ul>
<li>本地 Transformers 无原生 Kimi K3；checkpoint remote code 要求 <code>transformers≥4.56.0</code> 与额外 <code>fla-core</code>。</li>
<li>MLA 中 64 维变量名为 <code>q_rot/k_rot</code>，但源码未应用 RoPE。报告不假设部署端会隐式补上位置变换。</li>
<li><code>flash_attention_2</code> 是默认实际后端；图中的 eager QK/softmax/PV 是接口语义的本地 fallback，FlashAttention 可能融合这些算子。</li>
<li>KDA recurrent state 的精确物理 layout、dtype 与内存量取决于外部 fla-core kernel；这里只断言调用点、逻辑 head 维度和序列长度不增长。</li>
<li>权重索引总大小为 1,560,860,324,864 bytes，但 96 个 shard 被 metadata 下载脚本排除；本分析不实例化或运行完整权重。</li>
</ul></section>

<section id="excalidraw"><h2>11. 可编辑 Excalidraw 场景</h2><p class="lead"><b>结论：</b>每个主要计算图都导出为 version-2 可编辑场景，右侧带对应源码面板。</p><ul>{diagram_links}</ul></section>
'''


def build_state():
    return {
        "schema_version": 1,
        "analysis_date": "2026-08-28",
        "model": {"repo_id": "moonshotai/Kimi-K3", "revision": "a590ce090cb049c93a33dfe8c208ec652aa20503", "config_path": "models/moonshotai/Kimi-K3/config.json", "architecture": ["KimiK3ForConditionalGeneration", "KimiLinearForCausalLM"], "model_type": ["kimi_k3", "kimi_linear"]},
        "transformers": {"commit": "36deb0b53ed0863f4b4dfdea23dcaec7f3df3701", "source_directories": [], "compatibility_notes": ["No native kimi_k3/kimi_linear directory exists in the local Transformers checkout; execution semantics come from checkpoint remote code.", "Remote code asserts transformers>=4.56.0; config records 4.56.2.", "KDA kernels and state layouts are external to the repository in fla-core."]},
        "dimension_symbols": {"B": "batch size", "S_q": "current tokens", "S_past": "cached prior tokens", "S_kv": "S_past+S_q in MLA", "T": "B*S_q", "H": 7168, "N_q": 96, "N_kv": 96, "D_nope": 128, "D_rope_named": 64, "D_qk": 192, "D_v": 128, "R_q": 1536, "R_kv": 512, "E": 896, "K": 16, "H_latent_moe": 3584, "I_e": 3072},
        "configured_layer_schedule": [
            {"layers_0_based": "all except 3,7,11,...,87,91,92", "count": 69, "attention": "KimiDeltaAttention/KDA"},
            {"layers_0_based": [3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63,67,71,75,79,83,87,91,92], "count": 24, "attention": "KimiMLAAttention"},
            {"layers_0_based": [0], "ffn": "dense KimiMLP, I=33792"},
            {"layers_0_based": "1..92", "count": 92, "ffn": "KimiSparseMoeBlock, E=896 K=16 plus shared expert"},
        ],
        "conclusions": {
            "top_level": ["Vision tower is 27-layer MoonViT at width 1024; PatchMergerMLPV2 maps 2x2 merged patches from 4096 to text width 7168.", "Text and projected image features form one embedding sequence for a 93-layer KimiLinear decoder and untied LM head."],
            "attention": ["69 layers use KDA with short convolutions and a fixed recurrent state; 24 layers use MLA-style low-rank projections and dense causal attention.", "MLA runtime expands K and V to 96 heads and caches expanded tensors, so cache storage is not latent-compressed.", "The remote code does not apply RoPE to q_rot/k_rot despite their names; position_ids are unused in KimiMLAAttention.forward."],
            "ffn_or_moe": ["Layer 0 is dense SiTU FFN; layers 1-92 use latent MoE.", "Router chooses 16 of 896 experts using sigmoid scores plus correction bias for selection; gathered uncorrected scores are renormalized.", "Routed payload is projected 7168->3584, processed by 3584->3072->3584 experts, normalized and projected back; a parallel shared 7168->6144->7168 expert is added."],
            "residual": ["Every 12 layers a prefix snapshot is appended to a residual bank; learned softmax mixing over bank plus current prefix feeds attention and MLP.", "Eight snapshots are accumulated at layers 0,12,...,84; final output mixes those eight with the current state."],
            "kv_cache": ["MLA caches K [B,96,S_kv,192] and V [B,96,S_kv,128] with unbounded cat on sequence axis.", "KDA caches three short-convolution states and one fixed recurrent state per KDA layer; exact physical layout is owned by external fla-core.", "At bf16, 24-layer MLA cache costs 1,474,560 bytes per token and 1,440 GiB at 1,048,576 tokens for B=1."],
        },
        "source_symbols": [
            {"symbol": "KimiK3ForConditionalGeneration.forward", "path": "models/moonshotai/Kimi-K3/modeling_kimi_k3.py", "line": 1113, "role": "multimodal input merge and LM call"},
            {"symbol": "KimiLinearModel.forward", "path": "models/moonshotai/Kimi-K3/modeling_kimi_linear.py", "line": 1138, "role": "93-layer decoder loop and masks"},
            {"symbol": "KimiMLAAttention.forward", "path": "models/moonshotai/Kimi-K3/modeling_kimi_linear.py", "line": 405, "role": "MLA projections, cache and attention"},
            {"symbol": "KimiDeltaAttention.forward", "path": "models/moonshotai/Kimi-K3/modeling_kimi_linear.py", "line": 543, "role": "KDA convolution and kernel calls"},
            {"symbol": "KimiSparseMoeBlock.forward", "path": "models/moonshotai/Kimi-K3/modeling_kimi_linear.py", "line": 815, "role": "latent routed and shared MoE"},
            {"symbol": "_apply_attn_res", "path": "models/moonshotai/Kimi-K3/modeling_kimi_linear.py", "line": 1075, "role": "attention-residual candidate mixing"},
            {"symbol": "KimiDynamicCache", "path": "models/moonshotai/Kimi-K3/modeling_kimi_linear.py", "line": 120, "role": "hybrid cache container"},
        ],
        "graph_code_pairs": [
            {"section": "top_level", "graph_nodes": [f"TL{i}" for i in range(1,9)], "path": "models/moonshotai/Kimi-K3/modeling_kimi_k3.py", "line_start": 899, "line_end": 1218, "excerpt_kind": "ordered trimmed verbatim"},
            {"section": "decoder_kda", "graph_nodes": [f"DL{i}" for i in range(1,9)], "path": "models/moonshotai/Kimi-K3/modeling_kimi_linear.py", "line_start": 973, "line_end": 1046, "excerpt_kind": "trimmed verbatim"},
            {"section": "decoder_mla", "graph_nodes": [f"DF{i}" for i in range(1,8)], "path": "models/moonshotai/Kimi-K3/modeling_kimi_linear.py", "line_start": 883, "line_end": 1046, "excerpt_kind": "ordered trimmed verbatim"},
            {"section": "attention", "graph_nodes": [f"FA{i}" for i in range(1,12)], "path": "models/moonshotai/Kimi-K3/modeling_kimi_linear.py", "line_start": 311, "line_end": 473, "excerpt_kind": "ordered trimmed verbatim plus fallback"},
            {"section": "kda", "graph_nodes": [f"LA{i}" for i in range(1,9)], "path": "models/moonshotai/Kimi-K3/modeling_kimi_linear.py", "line_start": 580, "line_end": 659, "excerpt_kind": "ordered trimmed verbatim"},
            {"section": "moe", "graph_nodes": [f"MOE{i}" for i in range(1,9)], "path": "models/moonshotai/Kimi-K3/modeling_kimi_linear.py", "line_start": 703, "line_end": 873, "excerpt_kind": "ordered trimmed verbatim"},
            {"section": "residual", "graph_nodes": [f"RN{i}" for i in range(1,7)], "path": "models/moonshotai/Kimi-K3/modeling_kimi_linear.py", "line_start": 987, "line_end": 1088, "excerpt_kind": "ordered trimmed verbatim"},
            {"section": "kv_cache", "graph_nodes": [f"KV{i}" for i in range(1,8)], "path": "models/moonshotai/Kimi-K3/modeling_kimi_linear.py", "line_start": 157, "line_end": 649, "excerpt_kind": "ordered trimmed verbatim and external interface"},
        ],
        "assumptions": ["The eager attention function is used to explain unfused attention semantics; default runtime selects flash_attention_2.", "Cache memory estimate assumes bf16, batch size 1 and excludes allocator/kernel temporary storage."],
        "open_questions": ["Does the production runtime apply positional transforms absent from the checkpoint remote code?", "What exact fla-core version and recurrent state layout are used in deployment?"],
        "conversation_followups": [],
    }


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    template = TEMPLATE.read_text(encoding="utf-8")
    template = template.replace("</style>", ".big{color:var(--accent);font-size:21px;font-weight:750}.warning{border-left-color:#b57b22}</style>")
    output = template.replace("{{MODEL_TITLE}}", "Kimi K3（moonshotai/Kimi-K3）")
    output = output.replace("{{GENERATED_AT}}", "2026-08-28")
    output = output.replace("{{REPORT_BODY}}", build_body())
    (REPORT / "report.html").write_text(output, encoding="utf-8")
    (REPORT / "analysis-state.json").write_text(json.dumps(build_state(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {REPORT / 'report.html'}")
    print(f"Wrote {REPORT / 'analysis-state.json'}")


if __name__ == "__main__":
    main()
