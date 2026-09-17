#!/usr/bin/env python3
"""Build the model-specific HiDream-O1-Image architecture report."""

from __future__ import annotations

import html
import json
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "HiDream-ai" / "HiDream-O1-Image"
TEMPLATE = ROOT / ".agents/skills/hf-model-architecture/assets/report-template.html"


def node(node_id, x, y, w, h, label, tensor=True):
    lines = label.split("\n")
    klass = "tensor-node" if tensor else "op-node"
    rx = 11 if tensor else 0
    start_y = y + h / 2 - (len(lines) - 1) * 8
    spans = [f'<tspan class="node-id" x="{x+w/2}" dy="0">{html.escape(node_id)}</tspan>']
    for i, line in enumerate(lines):
        spans.append(f'<tspan class="shape-label" x="{x+w/2}" dy="{16 if i == 0 else 15}">{html.escape(line)}</tspan>')
    return (
        f'<g class="{klass}"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}"/>'
        f'<text x="{x+w/2}" y="{start_y}">{"".join(spans)}</text></g>'
    )


def edge(points):
    path = "M " + " L ".join(f"{x} {y}" for x, y in points)
    return f'<path class="flow" marker-end="url(#arrow)" d="{path}"/>'


def svg(width, height, nodes, edges, labels=()):
    lane_labels = "".join(
        f'<text class="lane-label" x="{x}" y="{y}">{html.escape(label)}</text>' for x, y, label in labels
    )
    return f'''<div class="graph-scroll"><svg class="architecture-graph" viewBox="0 0 {width} {height}" role="img">
      <defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor"/></marker></defs>
      {lane_labels}{''.join(edges)}{''.join(nodes)}
    </svg></div>'''


def pair(title, graph, source, line_range, code, kind="trimmed verbatim"):
    return f'''<div class="graph-code-pair">
      <div class="graph-panel"><h4>{html.escape(title)}</h4>
        <div class="graph-legend"><span><i class="tensor-key"></i>张量</span><span><i></i>计算</span></div>{graph}
      </div>
      <div class="code-panel"><h4><a href="{source}#L{line_range.split('-')[0]}">{html.escape(source)}</a> · L{line_range}<span class="excerpt-kind">{kind}</span></h4>
        <pre><code>{code}</code></pre>
      </div>
    </div>'''


def esc_code(raw):
    return html.escape(raw, quote=False)


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    top_graph = svg(760, 760, [
        node("TL1", 20, 35, 190, 64, "prompt / reference images\ntext + optional RGB", True),
        node("TL2", 285, 35, 190, 64, "noise z [B,Nx,3072]\nNx=(H/32)(W/32)", True),
        node("TL3", 550, 35, 190, 64, "timestep [B]\nscalar flow time", True),
        node("TL4", 20, 135, 190, 72, "processor + embed_tokens\n[B,St,4096]", False),
        node("TL5", 285, 135, 190, 72, "x_embedder\n3072→1024→4096", False),
        node("TL6", 550, 135, 190, 72, "t_embedder1\n256→4096→4096", False),
        node("TL7", 265, 255, 230, 72, "inputs_embeds\n[B,S=St+Nx,4096]", True),
        node("TL8", 265, 365, 230, 84, "language_model.layers ×36\nmixed causal/full attention", False),
        node("TL9", 265, 495, 230, 72, "final_layer2.linear\n4096→3072 per token", False),
        node("TL10", 265, 610, 230, 72, "x_pred target patches\n[B,Nx,3072]", True),
        node("TL11", 530, 610, 210, 72, "scheduler step ×50\nupdated z [B,Nx,3072]", False),
    ], [
        edge([(115,99),(115,135)]), edge([(380,99),(380,135)]), edge([(645,99),(645,135)]),
        edge([(115,207),(115,230),(330,230),(330,255)]), edge([(380,207),(380,255)]),
        edge([(645,207),(645,230),(430,230),(430,255)]), edge([(380,327),(380,365)]),
        edge([(380,449),(380,495)]), edge([(380,567),(380,610)]), edge([(495,646),(530,646)]),
        edge([(635,682),(635,720),(380,720),(380,682)]),
    ])
    top_code = (
        '<span class="code-map">TL1</span>' + esc_code("template_caption = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)\n") +
        '<span class="code-map">TL2</span>' + esc_code("z = rearrange(noise, 'B C (H p1) (W p2) -> B (H W) (C p1 p2)', p1=32, p2=32)\n") +
        '<span class="code-map">TL3</span>' + esc_code("t_pixeldit = 1.0 - step_t.float() / 1000.0\n") +
        '<span class="code-map">TL4</span>' + esc_code("inputs_embeds = self.get_input_embeddings()(input_ids)\n") +
        '<span class="code-map">TL6</span>' + esc_code("t_emb = self.t_embedder1(timestep)\ninputs_embeds = torch.where(tms_mask_3d, t_emb_expanded, inputs_embeds)\n") +
        '<span class="code-map">TL5</span>' + esc_code("vinputs_embedded = self.x_embedder(vinputs)\n") +
        '<span class="code-map">TL7</span>' + esc_code("inputs_embeds = torch.cat([inputs_embeds, vinputs_embedded], dim=1)\n") +
        '<span class="code-map">TL8</span>' + esc_code("hidden_states, mid_results = self._run_decoder_flash(inputs_embeds, position_ids, token_types, ...)\n") +
        '<span class="code-map">TL9</span>' + esc_code("x_pred = self.final_layer2(hidden_states)\n") +
        '<span class="code-map">TL10</span>' + esc_code("x_pred = outputs.x_pred\nreturn x_pred[0, sample['vinput_mask'][0]].unsqueeze(0)\n") +
        '<span class="code-map">TL11</span>' + esc_code("z = sched.step(model_output.float(), step_t, z.float(), return_dict=False)[0]")
    )

    call_graph = svg(980, 1040, [
        node("MC1", 300, 20, 380, 70, "Qwen3VLForConditionalGeneration.forward\n顶层包装器 · 1 instance", False),
        node("MC2", 300, 125, 380, 70, "self.model → Qwen3VLModel.forward\n选择 text/vision/image-generation 分支", False),
        node("MC3", 300, 230, 380, 76, "vinputs is not None\n→ Qwen3VLModel._forward_generation", False),
        node("MC4", 300, 345, 380, 82, "embed_tokens + t_embedder1 + x_embedder\ncat → inputs_embeds [B,S,4096]", False),
        node("MC5", 300, 470, 380, 76, "_run_decoder_flash\n3D MRoPE + mixed attention loop", False),
        node("MC6", 300, 585, 380, 76, "Qwen3VLTextDecoderLayer.forward ×36\n同构层 schedule: 0…35", False),
        node("MC7", 35, 705, 270, 78, "input_layernorm → self_attn\nGQA: 32Q / 8KV", False),
        node("MC8", 355, 705, 270, 78, "post_attention_layernorm\n→ self.mlp", False),
        node("MC9", 675, 705, 270, 78, "Qwen3VLTextMLP.forward\nDense SwiGLU", False),
        node("MC10", 35, 845, 200, 76, "gate_proj + SiLU\n4096→12288", False),
        node("MC11", 275, 845, 200, 76, "up_proj\n4096→12288", False),
        node("MC12", 515, 845, 200, 76, "elementwise multiply\n[B,S,12288]", False),
        node("MC13", 755, 845, 200, 76, "down_proj\n12288→4096", False),
        node("MC14", 300, 960, 380, 62, "residual add → next DecoderLayer\n最终 RMSNorm → final_layer2", True),
    ], [
        edge([(490,90),(490,125)]), edge([(490,195),(490,230)]), edge([(490,306),(490,345)]),
        edge([(490,427),(490,470)]), edge([(490,546),(490,585)]),
        edge([(490,661),(490,685),(170,685),(170,705)]),
        edge([(490,661),(490,705)]), edge([(490,661),(490,685),(810,685),(810,705)]),
        edge([(810,783),(810,815),(135,815),(135,845)]),
        edge([(810,783),(810,815),(375,815),(375,845)]),
        edge([(235,883),(515,883)]), edge([(475,883),(515,883)]), edge([(715,883),(755,883)]),
        edge([(855,921),(855,940),(490,940),(490,960)]),
    ], [(170,695,"Attention 子调用"),(490,695,"FFN 入口"),(810,695,"FFN 实现")])
    call_code = (
        '<span class="code-map">MC1</span>' + esc_code("class Qwen3VLForConditionalGeneration(...):\n    def forward(..., vinputs=None, timestep=None, token_types=None, ...):\n") +
        '<span class="code-map">MC2</span>' + esc_code("        outputs = self.model(..., vinputs=vinputs, timestep=timestep, token_types=token_types, ...)\n") +
        '<span class="code-map">MC3</span>' + esc_code("if vinputs is not None:\n    return self._forward_generation(...)\n") +
        '<span class="code-map">MC4</span>' + esc_code("inputs_embeds = self.get_input_embeddings()(input_ids)\nt_emb = self.t_embedder1(timestep)\nvinputs_embedded = self.x_embedder(vinputs)\ninputs_embeds = torch.cat([inputs_embeds, vinputs_embedded], dim=1)\n") +
        '<span class="code-map">MC5</span>' + esc_code("hidden_states, mid_results = self._run_decoder_flash(inputs_embeds, position_ids, token_types, ...)\n") +
        '<span class="code-map">MC6</span>' + esc_code("for layer_idx, decoder_layer in enumerate(text_model.layers):\n    hidden_states = _flash_layer_forward(hidden_states, decoder_layer, cos, sin, idx_ar)\n") +
        '<span class="code-map">MC7</span>' + esc_code("hidden_states = self.input_layernorm(hidden_states)\nhidden_states, _ = self.self_attn(...)\nhidden_states = residual + hidden_states\n") +
        '<span class="code-map">MC8</span>' + esc_code("hidden_states = self.post_attention_layernorm(hidden_states)\nhidden_states = self.mlp(hidden_states)\n") +
        '<span class="code-map">MC9</span>' + esc_code("def forward(self, x):\n") +
        '<span class="code-map">MC10</span>' + esc_code("    gate = self.act_fn(self.gate_proj(x))\n") +
        '<span class="code-map">MC11</span>' + esc_code("    up = self.up_proj(x)\n") +
        '<span class="code-map">MC12</span>' + esc_code("    fused = gate * up\n") +
        '<span class="code-map">MC13</span>' + esc_code("    down_proj = self.down_proj(fused)\n") +
        '<span class="code-map">MC14</span>' + esc_code("hidden_states = residual + hidden_states\n...\nhidden_states = text_model.norm(hidden_states)\nx_pred = self.final_layer2(hidden_states)")
    )

    vision_graph = svg(760, 610, [
        node("VE1", 25, 35, 210, 70, "reference pixel_values\n[Np,3·2·16·16]", True),
        node("VE2", 275, 35, 210, 70, "patch_embed Conv3d\n1536→1152", False),
        node("VE3", 525, 35, 210, 70, "vision tokens\n[Np,1152]", True),
        node("VE4", 275, 160, 210, 70, "VisionBlock ×27\nMHA 16 heads + MLP", False),
        node("VE5", 25, 285, 210, 80, "DeepStack taps\nblocks 8,16,24", False),
        node("VE6", 275, 285, 210, 80, "merger 2×2 patches\n4608→4608→4096", False),
        node("VE7", 525, 285, 210, 80, "image_embeds\n[Np/4,4096]", True),
        node("VE8", 25, 430, 210, 80, "deepstack embeds ×3\n[Np/4,4096]", True),
        node("VE9", 275, 430, 210, 80, "masked_scatter\nreplace image placeholders", False),
        node("VE10", 525, 430, 210, 80, "decoder input + injections\nlayer 0/1/2", True),
    ], [
        edge([(235,70),(275,70)]), edge([(485,70),(525,70)]), edge([(630,105),(630,130),(380,130),(380,160)]),
        edge([(380,230),(380,285)]), edge([(275,195),(130,195),(130,285)]), edge([(485,325),(525,325)]),
        edge([(130,365),(130,430)]), edge([(630,365),(630,400),(380,400),(380,430)]),
        edge([(235,470),(275,470)]), edge([(485,470),(525,470)]),
    ])
    vision_code = (
        '<span class="code-map">VE1</span>' + esc_code("hidden_states = hidden_states.view(-1, 3, 2, 16, 16)\n") +
        '<span class="code-map">VE2</span>' + esc_code("hidden_states = self.proj(hidden_states).view(-1, self.embed_dim)\n") +
        '<span class="code-map">VE3</span>' + esc_code("# derived after patch_embed: hidden_states [N_patches, 1152]\n") +
        '<span class="code-map">VE4</span>' + esc_code("for layer_num, blk in enumerate(self.blocks):\n    hidden_states = blk(hidden_states, ...)\n") +
        '<span class="code-map">VE5</span>' + esc_code("if layer_num in self.deepstack_visual_indexes:\n    deepstack_feature = self.deepstack_merger_list[...](hidden_states)\n") +
        '<span class="code-map">VE6</span>' + esc_code("hidden_states = self.merger(hidden_states)\n") +
        '<span class="code-map">VE7</span>' + esc_code("return hidden_states, deepstack_feature_lists\n") +
        '<span class="code-map">VE8</span>' + esc_code("# deepstack_feature_lists: three tensors [N_patches/4, 4096]\n") +
        '<span class="code-map">VE9</span>' + esc_code("inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)\n") +
        '<span class="code-map">VE10</span>' + esc_code("hidden_states = text_model._deepstack_process(hidden_states, visual_pos_masks, deepstack_visual_embeds[layer_idx])")
    )

    decoder_graph = svg(760, 560, [
        node("DL1", 260, 25, 240, 65, "hidden_states hℓ\n[B,S,4096]", True),
        node("DL2", 260, 125, 240, 62, "input_layernorm\nRMSNorm ε=1e-6", False),
        node("DL3", 260, 220, 240, 70, "self_attn\n[B,S,4096]", False),
        node("DL4", 260, 325, 240, 64, "residual + attention\nh′=hℓ+Attn(Norm(hℓ))", False),
        node("DL5", 25, 430, 210, 70, "post_attention_layernorm\n[B,S,4096]", False),
        node("DL6", 275, 430, 210, 70, "SwiGLU MLP\n4096→12288→4096", False),
        node("DL7", 525, 430, 210, 70, "layer output hℓ+1\nh′+MLP(Norm(h′))", True),
    ], [
        edge([(380,90),(380,125)]), edge([(380,187),(380,220)]), edge([(380,290),(380,325)]),
        edge([(260,57),(180,57),(180,357),(260,357)]), edge([(380,389),(380,410),(130,410),(130,430)]),
        edge([(235,465),(275,465)]), edge([(485,465),(525,465)]), edge([(380,389),(620,389),(620,430)]),
    ])
    decoder_code = (
        '<span class="code-map">DL1</span>' + esc_code("residual = hidden_states\n") +
        '<span class="code-map">DL2</span>' + esc_code("hidden_states = self.input_layernorm(hidden_states)\n") +
        '<span class="code-map">DL3</span>' + esc_code("hidden_states, _ = self.self_attn(hidden_states=hidden_states, ...)\n") +
        '<span class="code-map">DL4</span>' + esc_code("hidden_states = residual + hidden_states\nresidual = hidden_states\n") +
        '<span class="code-map">DL5</span>' + esc_code("hidden_states = self.post_attention_layernorm(hidden_states)\n") +
        '<span class="code-map">DL6</span>' + esc_code("hidden_states = self.mlp(hidden_states)\n") +
        '<span class="code-map">DL7</span>' + esc_code("hidden_states = residual + hidden_states")
    )

    attn_graph = svg(920, 790, [
        node("FA1", 340, 25, 240, 62, "hidden_states\n[B,S,4096]", True),
        node("FA2", 20, 130, 250, 76, "q_proj + q_norm\n[B,S,32,128]", False),
        node("FA3", 335, 130, 250, 76, "k_proj + k_norm\n[B,S,8,128]", False),
        node("FA4", 650, 130, 250, 76, "v_proj\n[B,S,8,128]", False),
        node("FA5", 20, 255, 250, 72, "3D interleaved MRoPE\nq [B,S,32,128]", False),
        node("FA6", 335, 255, 250, 72, "3D interleaved MRoPE\nk [B,S,8,128]", False),
        node("FA7", 20, 380, 250, 80, "AR causal FlashAttention\n[B,Sar,32,128]", False),
        node("FA8", 335, 380, 250, 80, "full FlashAttention\n[B,S,32,128]", False),
        node("FA9", 650, 380, 250, 80, "GQA inside kernel\n32 Q : 8 KV = 4:1", False),
        node("FA10", 335, 515, 250, 80, "index replace AR positions\nout_full[:,idx_ar]=out_ar", False),
        node("FA11", 335, 650, 250, 76, "reshape + o_proj\n[B,S,4096]", False),
    ], [
        edge([(460,87),(460,108),(145,108),(145,130)]), edge([(460,87),(460,130)]), edge([(460,87),(460,108),(775,108),(775,130)]),
        edge([(145,206),(145,255)]), edge([(460,206),(460,255)]),
        edge([(145,327),(145,380)]), edge([(460,327),(460,380)]),
        edge([(775,206),(775,350),(520,350),(520,380)]), edge([(775,206),(775,380)]),
        edge([(270,420),(300,420),(300,555),(335,555)]), edge([(460,460),(460,515)]), edge([(775,460),(775,490),(585,490),(585,555)]),
        edge([(460,595),(460,650)]),
    ], [(145,115,"Q lane"),(460,115,"K lane"),(775,115,"V lane")])
    attn_code = (
        '<span class="code-map">FA1</span>' + esc_code("input_shape = hidden_states.shape[:-1]\n") +
        '<span class="code-map">FA2</span>' + esc_code("q = attn.q_norm(attn.q_proj(hidden_states).view(hidden_shape))\n") +
        '<span class="code-map">FA3</span>' + esc_code("k = attn.k_norm(attn.k_proj(hidden_states).view(hidden_shape))\n") +
        '<span class="code-map">FA4</span>' + esc_code("v = attn.v_proj(hidden_states).view(hidden_shape)\n") +
        '<span class="code-map">FA5</span>' + esc_code("q_r = q.transpose(1, 2)\n") +
        '<span class="code-map">FA6</span>' + esc_code("q_r, k_r = apply_rotary_pos_emb(q_r, k_r, cos_pe, sin_pe)\n") +
        '<span class="code-map">FA7</span>' + esc_code("result_ar = _flash_attn_func(q[:, idx_ar], k[:, idx_ar], v[:, idx_ar], causal=True)\n") +
        '<span class="code-map">FA8</span>' + esc_code("result_full = _flash_attn_func(q, k, v, causal=False)\n") +
        '<span class="code-map">FA9</span>' + esc_code("# derived: Nq/Nkv = 32/8 = 4 query heads per KV head\n") +
        '<span class="code-map">FA10</span>' + esc_code("out_full[:, idx_ar] = out_ar\n") +
        '<span class="code-map">FA11</span>' + esc_code("attn_output = attn.o_proj(out_full.reshape(*input_shape, -1).contiguous())")
    )

    ffn_graph = svg(760, 480, [
        node("FF1", 260, 25, 240, 65, "x [B,S,4096]\npost-attention RMSNorm", True),
        node("FF2", 40, 145, 250, 74, "gate_proj + SiLU\n4096→12288", False),
        node("FF3", 470, 145, 250, 74, "up_proj\n4096→12288", False),
        node("FF4", 260, 280, 240, 70, "elementwise product\n[B,S,12288]", False),
        node("FF5", 260, 395, 240, 62, "down_proj\n12288→4096", False),
    ], [
        edge([(380,90),(380,120),(165,120),(165,145)]), edge([(380,90),(380,120),(595,120),(595,145)]),
        edge([(165,219),(165,250),(320,250),(320,280)]), edge([(595,219),(595,250),(440,250),(440,280)]),
        edge([(380,350),(380,395)]),
    ], [(165,132,"gate lane"),(595,132,"up lane")])
    ffn_code = (
        '<span class="code-map">FF1</span>' + esc_code("def forward(self, x):\n") +
        '<span class="code-map">FF2</span>' + esc_code("    gated = self.act_fn(self.gate_proj(x))\n") +
        '<span class="code-map">FF3</span>' + esc_code("    up = self.up_proj(x)\n") +
        '<span class="code-map">FF4</span>' + esc_code("    fused = gated * up\n") +
        '<span class="code-map">FF5</span>' + esc_code("    return self.down_proj(fused)")
    )

    residual_graph = svg(760, 470, [
        node("RN1", 35, 40, 210, 68, "hℓ [B,S,4096]", True),
        node("RN2", 275, 40, 210, 68, "RMSNorm(hℓ)\nfloat32 variance", False),
        node("RN3", 515, 40, 210, 68, "Attention\n[B,S,4096]", False),
        node("RN4", 275, 175, 210, 68, "h′ = hℓ + attention\nserial pre-norm residual", True),
        node("RN5", 35, 305, 210, 68, "RMSNorm(h′)\nfloat32 variance", False),
        node("RN6", 275, 305, 210, 68, "SwiGLU MLP\n[B,S,4096]", False),
        node("RN7", 515, 305, 210, 68, "hℓ+1 = h′ + mlp\n[B,S,4096]", True),
    ], [
        edge([(245,74),(275,74)]), edge([(485,74),(515,74)]), edge([(620,108),(620,140),(380,140),(380,175)]),
        edge([(140,108),(140,209),(275,209)]), edge([(380,243),(380,275),(140,275),(140,305)]),
        edge([(245,339),(275,339)]), edge([(485,339),(515,339)]), edge([(380,243),(620,243),(620,305)]),
    ])
    residual_code = (
        '<span class="code-map">RN1</span>' + esc_code("residual = hidden_states\n") +
        '<span class="code-map">RN2</span>' + esc_code("hidden_states = self.input_layernorm(hidden_states)\n") +
        '<span class="code-map">RN3</span>' + esc_code("hidden_states, _ = self.self_attn(...)\n") +
        '<span class="code-map">RN4</span>' + esc_code("hidden_states = residual + hidden_states\nresidual = hidden_states\n") +
        '<span class="code-map">RN5</span>' + esc_code("hidden_states = self.post_attention_layernorm(hidden_states)\n") +
        '<span class="code-map">RN6</span>' + esc_code("hidden_states = self.mlp(hidden_states)\n") +
        '<span class="code-map">RN7</span>' + esc_code("hidden_states = residual + hidden_states")
    )

    cache_graph = svg(760, 480, [
        node("KV1", 35, 35, 210, 68, "image denoising call\n[B,S,4096]", True),
        node("KV2", 275, 35, 210, 68, "_run_decoder_flash\nno past_key_values", False),
        node("KV3", 515, 35, 210, 68, "recompute Q/K/V ×36\nfor every denoise step", False),
        node("KV4", 35, 180, 210, 76, "standard LM path only\nDynamicCache.update", False),
        node("KV5", 275, 180, 210, 76, "K,V per layer\n[B,8,Spast,128] each", True),
        node("KV6", 515, 180, 210, 76, "2048 elements/token/layer\n4 KiB bf16", True),
        node("KV7", 275, 335, 210, 76, "generation behavior\nactive cache = 0 bytes", True),
    ], [
        edge([(245,69),(275,69)]), edge([(485,69),(515,69)]),
        edge([(140,103),(140,180)]), edge([(245,218),(275,218)]), edge([(485,218),(515,218)]),
        edge([(620,103),(620,310),(380,310),(380,335)]),
    ], [(380,330,"selected image-generation branch")])
    cache_code = (
        '<span class="code-map">KV1</span>' + esc_code("if use_flash_attn:\n") +
        '<span class="code-map">KV2</span>' + esc_code("    hidden_states, mid_results = self._run_decoder_flash(inputs_embeds, position_ids, token_types, ...)\n") +
        '<span class="code-map">KV3</span>' + esc_code("# _run_decoder_flash executes all 36 layers on the complete sequence\n") +
        '<span class="code-map">KV4</span>' + esc_code("# standard language path (not selected by image pipeline):\nkey_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)\n") +
        '<span class="code-map">KV5</span>' + esc_code("# derived cache tensors: [B,8,S_past,128] for K and V\n") +
        '<span class="code-map">KV6</span>' + esc_code("# 2*8*128 = 2048 elements = 4096 bytes/token/layer in bf16\n") +
        '<span class="code-map">KV7</span>' + esc_code("outputs = self.language_model(..., use_cache=False, ...)")
    )

    body = f'''
<section id="identity">
  <h2>1. 身份、范围与依据</h2>
  <p class="lead"><b>结论：</b>HiDream-O1-Image 是一个 8.8049B 参数的原生像素生成模型。它借用了 Qwen3-VL 的视觉编码器与 36 层文本解码器，但官方运行时对该类做了生成扩展；标准 Transformers 的同名类并不能独立解释 checkpoint 中的像素路径。</p>
  <div class="callout">核心区别是：目标图像不经过 VAE，也不离散化成视觉词表。32×32 RGB patch 直接投影到 4096 维共享 token 空间，模型每个去噪步直接预测 3072 维原始像素 patch。</div>
  <div class="grid">
    <div class="card"><b>Checkpoint</b><code>HiDream-ai/HiDream-O1-Image</code><br>revision <code>0b0901d…</code></div>
    <div class="card"><b>官方运行时</b><code>sources/HiDream-O1-Image</code><br>commit <code>2c2d29f…</code></div>
    <div class="card"><b>本地 Transformers</b>commit <code>36deb0b…</code><br>config 声明 <code>4.57.0.dev0</code></div>
    <div class="card"><b>分析范围</b>推理时 text-to-image、editing / reference conditioning；训练分支仅在必要处说明。</div>
  </div>
  <p class="source">证据：<a href="../../../models/HiDream-ai/HiDream-O1-Image/config.json">config.json</a>、<a href="../../../models/HiDream-ai/HiDream-O1-Image/model.safetensors.index.json">权重索引</a>、<a href="../../../sources/HiDream-O1-Image/models/qwen3_vl_transformers.py">官方自定义运行时</a>。分析日期：{date.today().isoformat()}。</p>
</section>

<section id="hyperparameters">
  <h2>2. 配置与派生维度</h2>
  <p class="lead"><b>结论：</b>主干是 dense 36-layer GQA decoder，而不是 MoE；视觉条件编码器有 27 层。目标图像在 2048×2048 时产生 64×64=4096 个原始像素 token。</p>
  <div class="grid">
    <div class="card"><b>总参数量</b>8,804,887,792（权重索引元数据）</div>
    <div class="card"><b>Decoder</b>H=4096，L=36，I=12288，SwiGLU</div>
    <div class="card"><b>Attention</b>Nq=32，Nkv=8，D=128，GQA 比 4:1</div>
    <div class="card"><b>像素 patch</b>32×32×3=3072；bottleneck 1024</div>
    <div class="card"><b>Vision encoder</b>Hᵥ=1152，27 层，16 heads，Iᵥ=4304</div>
    <div class="card"><b>位置编码</b>interleaved 3D MRoPE；theta=5,000,000；section=[24,20,20]</div>
    <div class="card"><b>标准采样</b>undistilled full：50 步，CFG=5，shift=3</div>
    <div class="card"><b>数值格式</b>checkpoint 索引大小约 35.22 GB（≈FP32）；官方加载后转 BF16。</div>
  </div>
  <h3>精确参数核算</h3>
  <table><thead><tr><th>部分</th><th>参数</th><th>说明</th></tr></thead><tbody>
    <tr><td>token embedding</td><td>622,329,856</td><td>151,936×4,096</td></tr>
    <tr><td>36 decoder layers</td><td>6,946,071,552</td><td>GQA attention + SwiGLU + norms</td></tr>
    <tr><td>final decoder norm</td><td>4,096</td><td>RMSNorm</td></tr>
    <tr><td>vision encoder + mergers</td><td>576,388,336</td><td>27 blocks + main/deepstack mergers</td></tr>
    <tr><td>raw-pixel / timestep / pixel head</td><td>37,764,096</td><td>x_embedder 7.34M + t_embedder 17.83M + final 12.59M</td></tr>
    <tr><td>untied LM head</td><td>622,329,856</td><td>虽图像路径不使用，权重仍在 checkpoint 中</td></tr>
    <tr><th>合计</th><th>8,804,887,792</th><th>与索引完全一致</th></tr>
  </tbody></table>
  <details><summary>查看完整配置</summary><pre>{html.escape((ROOT/'models/HiDream-ai/HiDream-O1-Image/config.json').read_text())}</pre></details>
</section>

<section id="top-level">
  <h2>3. 顶层生成路径</h2>
  <p class="lead"><b>结论：</b>每个采样步都把文本、一个时间步 token 和当前噪声图像 patch 拼成同一序列，完整运行 36 层，再由线性头直接输出 x₀ 像素 patch。</p>
  {pair('去噪一步：统一 token 空间', top_graph, '../../../sources/HiDream-O1-Image/models/qwen3_vl_transformers.py', '1428-1617', top_code)}
  <p>2048² 例子：<code>Nx=(2048/32)²=4096</code>，每个 patch 输入/输出均为 <code>3×32×32=3072</code>。pipeline 将 <code>x_pred</code> 转为速度 <code>v=(x₀-z)/σ</code>，CFG 后交给 scheduler 更新 <code>z</code>。full 模型默认执行 50 次，因此 decoder 也完整执行 50 次（启用 CFG 时条件/无条件各一次）。</p>
</section>

<section id="model-callgraph">
  <h2>4. modeling.py 模型调用树</h2>
  <p class="lead"><b>结论：</b>官方文件虽然仍叫 <code>qwen3_vl_transformers.py</code>，但真正图像生成调用链是明确的层级组合：顶层 wrapper → HiDream generation model → 36× DecoderLayer → Decoder FFN → SwiGLU 算子。下面同时给出模块层级、实例数量、张量形状和对应源码。</p>
  {pair('从顶层模型展开到 Decoder FFN / SwiGLU', call_graph, '../../../sources/HiDream-O1-Image/models/qwen3_vl_transformers.py', '480-539,1257-1398,1400-1617,1808-1912', call_code, 'ordered trimmed excerpts + explicit one-line expansion')}
  <h3>按 Python 对象展开</h3>
  <pre>Qwen3VLForConditionalGeneration                       ×1
└── model: Qwen3VLModel                              ×1
    ├── visual: Qwen3VLVisionModel                   ×1（仅参考图条件）
    │   ├── blocks: Qwen3VLVisionBlock               ×27
    │   ├── merger: Qwen3VLVisionPatchMerger         ×1
    │   └── deepstack_merger_list                    ×3
    ├── language_model: Qwen3VLTextModel              ×1
    │   ├── layers: Qwen3VLTextDecoderLayer           ×36
    │   │   ├── input_layernorm: RMSNorm              ×1/layer
    │   │   ├── self_attn: Qwen3VLTextAttention       ×1/layer
    │   │   │   ├── q_proj / k_proj / v_proj
    │   │   │   ├── q_norm / k_norm + 3D MRoPE
    │   │   │   └── two-pass FlashAttention → o_proj
    │   │   ├── post_attention_layernorm: RMSNorm     ×1/layer
    │   │   └── mlp: Qwen3VLTextMLP                   ×1/layer
    │   │       └── down_proj(SiLU(gate_proj(x)) × up_proj(x))
    │   └── norm: RMSNorm                             ×1
    ├── x_embedder: BottleneckPatchEmbed              ×1
    ├── t_embedder1: TimestepEmbedder                 ×1
    └── final_layer2: FinalLayer                      ×1
Qwen3VLForConditionalGeneration.lm_head               ×1（图像生成分支不调用）</pre>
  <p class="source">特别注意：<code>lm_head</code> 虽占 622.33M 参数，但当 <code>vinputs is not None</code> 时 wrapper 在返回 <code>x_pred</code> 后直接结束，不计算 vocabulary logits。</p>
</section>

<section id="vision">
  <h2>5. 参考图条件编码</h2>
  <p class="lead"><b>结论：</b>编辑/个性化时同一参考图走两条路径：低分辨率副本进入 16×16 Qwen3-VL vision encoder 提供语义条件；原始参考图按 32×32 patch 与目标噪声一起进入 Pixel-UiT。纯 T2I 推理不会调用 vision encoder。</p>
  {pair('参考图语义分支与 DeepStack', vision_graph, '../../../sources/HiDream-O1-Image/models/qwen3_vl_transformers.py', '728-778,1201-1215,1439-1451', vision_code)}
  <p>Vision encoder 的 block 8/16/24 特征分别经 2×2 merger 后，注入 decoder 的第 0/1/2 层输出处；“tap 层号”和“注入层号”并不相同。主 merger 将每 4 个 1152 维 patch 拼为 4608，再投影到 4096。</p>
</section>

<section id="decoder">
  <h2>6. Decoder 层</h2>
  <p class="lead"><b>结论：</b>36 层完全同构，都是串行 pre-norm：RMSNorm→self-attention→残差，再 RMSNorm→SwiGLU→残差；没有 MoE、AdaLN 或并行 FFN。</p>
  {pair('单层残差拓扑', decoder_graph, '../../../sources/HiDream-O1-Image/models/qwen3_vl_transformers.py', '496-539', decoder_code)}
</section>

<section id="attention">
  <h2>7. 混合因果 / 双向 Attention</h2>
  <p class="lead"><b>结论：</b>图像生成实际选择自定义 two-pass FlashAttention。文本 AR token 只能因果地看过去文本；时间步与目标/参考像素 token 能双向看完整序列。注意力是 32Q/8KV 的 GQA，不使用缓存。</p>
  {pair('Two-pass GQA：AR causal + all-token full', attn_graph, '../../../sources/HiDream-O1-Image/models/qwen3_vl_transformers.py', '1291-1350', attn_code)}
  <div class="derived-note">形状推导：<code>q_proj: 4096→32×128=4096</code>；<code>k_proj/v_proj: 4096→8×128=1024</code>。FlashAttention 内部让每组 4 个 query heads 共享一个 K/V head。非 flash fallback 会显式 <code>repeat_kv</code>，并构造 <code>[B,1,S,S]</code> mask。</div>
</section>

<section id="ffn-moe">
  <h2>8. Dense SwiGLU FFN</h2>
  <p class="lead"><b>结论：</b>每层 FFN 是 dense SwiGLU，不存在 router、top-k 或 expert dispatch。gate/up 两条支路必须并行计算，到逐元素乘法才汇合。</p>
  {pair('SwiGLU 的 gate/up 并行支路', ffn_graph, '../../../sources/HiDream-O1-Image/models/qwen3_vl_transformers.py', '480-493', ffn_code, 'derived from verbatim one-liner')}
  <p>每层 FFN 权重为 <code>3×4096×12288=150,994,944</code> 参数，占单层约 78.3%；这是模型参数量的主体。</p>
</section>

<section id="residual">
  <h2>9. 残差与归一化</h2>
  <p class="lead"><b>结论：</b>残差拓扑是标准串行 pre-norm。RMSNorm 在 float32 中计算方差后转回输入 dtype；Q/K 还各自做 head_dim=128 的 RMSNorm。</p>
  {pair('Pre-norm residual 的两次串行更新', residual_graph, '../../../sources/HiDream-O1-Image/models/qwen3_vl_transformers.py', '358-372,519-539', residual_code)}
</section>

<section id="kv-cache">
  <h2>10. KV Cache</h2>
  <p class="lead"><b>结论：</b>图像生成路径的有效 KV cache 为零。每个去噪步改变全部图像 token，且生成 token 需要全局双向注意力，所以代码每一步重算完整 Q/K/V；config 的 <code>use_cache=true</code> 只对标准自回归语言路径有意义。</p>
  {pair('生成分支与未选择的标准 cache 分支', cache_graph, '../../../sources/HiDream-O1-Image/models/qwen3_vl_transformers.py', '455-458,1569-1607', cache_code)}
  <p>若单独使用标准 LM 自回归分支，DynamicCache 每层保存 K/V <code>[B,8,S_past,128]</code>，每 token 每层为 2048 个元素，即 BF16 4 KiB；36 层合计约 144 KiB/token（batch=1，不含管理开销）。这不是 HiDream 图像 pipeline 的显存占用。</p>
</section>

<section id="sources">
  <h2>11. 源码对应与可编辑图</h2>
  <p class="lead"><b>结论：</b>Checkpoint 的标准身份映射到 Qwen3-VL，但决定图像生成语义的是官方 fork 中的 <code>Qwen3VLModel._forward_generation</code> 与 <code>_run_decoder_flash</code>。</p>
  <table><thead><tr><th>符号</th><th>本地源码</th><th>作用</th></tr></thead><tbody>
    <tr><td>generate_image</td><td><a href="../../../sources/HiDream-O1-Image/models/pipeline.py#L107">pipeline.py:L107</a></td><td>patchify、CFG、50-step scheduler loop</td></tr>
    <tr><td>Qwen3VLModel.__init__</td><td><a href="../../../sources/HiDream-O1-Image/models/qwen3_vl_transformers.py#L1033">qwen3_vl_transformers.py:L1033</a></td><td>实例化 vision/text、像素与时间头</td></tr>
    <tr><td>_forward_generation</td><td><a href="../../../sources/HiDream-O1-Image/models/qwen3_vl_transformers.py#L1400">qwen3_vl_transformers.py:L1400</a></td><td>统一序列与像素预测</td></tr>
    <tr><td>_run_decoder_flash</td><td><a href="../../../sources/HiDream-O1-Image/models/qwen3_vl_transformers.py#L1257">qwen3_vl_transformers.py:L1257</a></td><td>mixed causal/full two-pass attention</td></tr>
    <tr><td>Qwen3VLTextDecoderLayer</td><td><a href="../../../sources/HiDream-O1-Image/models/qwen3_vl_transformers.py#L496">qwen3_vl_transformers.py:L496</a></td><td>36 层统一 residual block</td></tr>
    <tr><td>标准 Transformers 对照</td><td><a href="../../../transformers/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L1260">modeling_qwen3_vl.py:L1260</a></td><td>不含 HiDream 的 raw-pixel generation fork</td></tr>
  </tbody></table>
  <h3>可编辑 Excalidraw 场景</h3>
  <p><a href="diagrams/01-top-level.excalidraw">顶层</a> · <a href="diagrams/02-diagram.excalidraw">modeling.py 调用树</a> · <a href="diagrams/03-diagram.excalidraw">视觉条件</a> · <a href="diagrams/02-decoder-layer.excalidraw">Decoder</a> · <a href="diagrams/04-full-attention.excalidraw">Attention</a> · <a href="diagrams/06-diagram.excalidraw">FFN</a> · <a href="diagrams/07-residual-norm.excalidraw">残差</a> · <a href="diagrams/08-kv-cache.excalidraw">KV cache</a></p>
</section>

<section id="unknowns">
  <h2>12. 兼容性、假设与待确认项</h2>
  <p class="lead"><b>结论：</b>结构本身已由 config、运行时代码与权重名交叉确认；剩余不确定性主要来自运行环境和 checkpoint 之外的训练细节。</p>
  <ul>
    <li><b>必须使用官方 fork：</b>直接用本地 Transformers 的标准 <code>Qwen3VLForConditionalGeneration</code> 会缺少 <code>x_embedder/t_embedder1/final_layer2</code> 与生成分支。</li>
    <li><b>FlashAttention 是默认必选：</b>pipeline 硬编码 <code>use_flash_attn=True</code>；未安装 FA2/FA3 会触发 assertion，README 建议手动改为 fallback。</li>
    <li><b>版本偏差：</b>config 记录 4.57.0.dev0；本地 Transformers checkout 更新于另一 commit，官方 fork 又复制并修改了生成文件。报告以官方 fork 的实际调用为准。</li>
    <li><b>权重精度：</b>索引总大小精确等于参数数×4，说明发布权重按 4-byte 存储；官方推理显式加载成 BF16。未读取被排除的 safetensors shard header，但元数据与计算一致。</li>
    <li><b>训练损失：</b>下载的推理仓库不公开完整训练 loop；可确认推理输出 x₀ 并换算 flow velocity，但不把训练目标的更多细节当作源码事实。</li>
  </ul>
  <details><summary>原始证据与状态文件</summary><p><a href="evidence.json">evidence.json</a> · <a href="analysis-state.json">analysis-state.json</a></p></details>
</section>
'''

    rendered = TEMPLATE.read_text(encoding="utf-8")
    rendered = rendered.replace("{{MODEL_TITLE}}", "HiDream-O1-Image · Pixel-level UiT 架构分析")
    rendered = rendered.replace("{{GENERATED_AT}}", date.today().isoformat())
    rendered = rendered.replace("{{REPORT_BODY}}", body)
    (REPORT_DIR / "report.html").write_text(rendered, encoding="utf-8")

    state = {
        "schema_version": 1,
        "model": {
            "repo_id": "HiDream-ai/HiDream-O1-Image",
            "revision": "0b0901d99f200389e138c61946af1185f5f49a13",
            "config_path": "models/HiDream-ai/HiDream-O1-Image/config.json",
            "architecture": ["Qwen3VLForConditionalGeneration (HiDream fork)"],
            "model_type": ["qwen3_vl", "qwen3_vl_text"],
            "parameter_count": 8804887792,
        },
        "transformers": {
            "commit": "36deb0b53ed0863f4b4dfdea23dcaec7f3df3701",
            "source_directories": ["transformers/src/transformers/models/qwen3_vl"],
            "official_runtime_commit": "2c2d29ff729e48f33e41f49edfdbd81d5ac103b4",
            "compatibility_notes": [
                "Config says transformers 4.57.0.dev0; official runtime is a modified copied Qwen3-VL implementation.",
                "Standard Transformers lacks the raw-pixel generation modules and mixed-mask denoising path.",
            ],
        },
        "dimension_symbols": {
            "B": "batch size", "St": "text sequence length including one timestep token",
            "Nx": "target 32x32 RGB patch count", "S": "St+Nx (+ reference raw patches when present)",
            "H": 4096, "N_q": 32, "N_kv": 8, "D_qk": 128, "D_v": 128,
            "I": 12288, "P": 32, "patch_dim": 3072, "pixel_bottleneck": 1024,
            "H_v": 1152, "vision_layers": 27, "decoder_layers": 36,
        },
        "configured_layer_schedule": [{"layers": "0-35", "type": "Qwen3VLTextDecoderLayer", "count": 36}],
        "conclusions": {
            "top_level": [
                "Raw RGB patches, text tokens and timestep embedding share one 4096-dimensional sequence.",
                "A linear head predicts 32x32x3=3072 raw-pixel values per selected target token; there is no VAE.",
                "2048x2048 generation uses 4096 target pixel tokens and 50 denoising steps in the full recipe.",
            ],
            "attention": [
                "32-query/8-KV-head grouped-query attention with head_dim 128.",
                "Selected inference branch is two-pass FlashAttention: causal AR-only pass plus bidirectional all-token pass, with AR outputs replaced.",
                "Interleaved 3D MRoPE uses sections [24,20,20] and theta 5,000,000.",
            ],
            "ffn_or_moe": ["Dense SwiGLU 4096->12288->4096; no MoE/router/expert routing."],
            "residual": ["Serial pre-norm residual attention then pre-norm residual MLP."],
            "kv_cache": [
                "Image generation uses no KV cache and recomputes the full sequence at every denoising step.",
                "The unused standard AR path would store K/V [B,8,S_past,128] per layer.",
            ],
            "vision_conditioning": [
                "Optional references use both a Qwen3-VL semantic vision path and a raw 32x32-patch Pixel-UiT path.",
                "Vision taps at blocks 8,16,24 are injected after decoder layers 0,1,2.",
            ],
        },
        "source_symbols": [
            {"symbol":"generate_image","path":"sources/HiDream-O1-Image/models/pipeline.py","line":107,"role":"inference and scheduler loop"},
            {"symbol":"Qwen3VLForConditionalGeneration.forward","path":"sources/HiDream-O1-Image/models/qwen3_vl_transformers.py","line":1852,"role":"top-level wrapper and image-generation early return"},
            {"symbol":"Qwen3VLModel.__init__","path":"sources/HiDream-O1-Image/models/qwen3_vl_transformers.py","line":1033,"role":"raw-pixel modules"},
            {"symbol":"Qwen3VLModel._run_decoder_flash","path":"sources/HiDream-O1-Image/models/qwen3_vl_transformers.py","line":1257,"role":"mixed attention"},
            {"symbol":"Qwen3VLModel._forward_generation","path":"sources/HiDream-O1-Image/models/qwen3_vl_transformers.py","line":1400,"role":"denoising forward"},
            {"symbol":"Qwen3VLTextDecoderLayer.forward","path":"sources/HiDream-O1-Image/models/qwen3_vl_transformers.py","line":508,"role":"decoder residual topology"},
            {"symbol":"Qwen3VLTextMLP.forward","path":"sources/HiDream-O1-Image/models/qwen3_vl_transformers.py","line":491,"role":"dense SwiGLU"},
        ],
        "graph_code_pairs": [
            {"section":"top-level","graph_nodes":[f"TL{i}" for i in range(1,12)],"path":"sources/HiDream-O1-Image/models/qwen3_vl_transformers.py","line_start":1428,"line_end":1617,"excerpt_kind":"trimmed verbatim"},
            {"section":"model-callgraph","graph_nodes":[f"MC{i}" for i in range(1,15)],"path":"sources/HiDream-O1-Image/models/qwen3_vl_transformers.py","line_start":480,"line_end":1912,"excerpt_kind":"ordered trimmed excerpts plus explicit one-line expansion"},
            {"section":"vision","graph_nodes":[f"VE{i}" for i in range(1,11)],"path":"sources/HiDream-O1-Image/models/qwen3_vl_transformers.py","line_start":728,"line_end":1451,"excerpt_kind":"trimmed verbatim"},
            {"section":"decoder","graph_nodes":[f"DL{i}" for i in range(1,8)],"path":"sources/HiDream-O1-Image/models/qwen3_vl_transformers.py","line_start":496,"line_end":539,"excerpt_kind":"trimmed verbatim"},
            {"section":"attention","graph_nodes":[f"FA{i}" for i in range(1,12)],"path":"sources/HiDream-O1-Image/models/qwen3_vl_transformers.py","line_start":1291,"line_end":1350,"excerpt_kind":"trimmed verbatim plus derived GQA note"},
            {"section":"ffn","graph_nodes":[f"FF{i}" for i in range(1,6)],"path":"sources/HiDream-O1-Image/models/qwen3_vl_transformers.py","line_start":480,"line_end":493,"excerpt_kind":"derived expansion of verbatim one-liner"},
            {"section":"residual","graph_nodes":[f"RN{i}" for i in range(1,8)],"path":"sources/HiDream-O1-Image/models/qwen3_vl_transformers.py","line_start":358,"line_end":539,"excerpt_kind":"trimmed verbatim"},
            {"section":"kv-cache","graph_nodes":[f"KV{i}" for i in range(1,8)],"path":"sources/HiDream-O1-Image/models/qwen3_vl_transformers.py","line_start":455,"line_end":1607,"excerpt_kind":"selected and unselected branches plus derivation"},
        ],
        "model_call_hierarchy": [
            {"caller":"Qwen3VLForConditionalGeneration.forward","callee":"Qwen3VLModel.forward","path":"sources/HiDream-O1-Image/models/qwen3_vl_transformers.py","line":1886,"count":1,"selected":True},
            {"caller":"Qwen3VLModel.forward","callee":"Qwen3VLModel._forward_generation","path":"sources/HiDream-O1-Image/models/qwen3_vl_transformers.py","line":1650,"count":1,"selected":True},
            {"caller":"Qwen3VLModel._forward_generation","callee":"Qwen3VLModel._run_decoder_flash","path":"sources/HiDream-O1-Image/models/qwen3_vl_transformers.py","line":1569,"count":1,"selected":True},
            {"caller":"Qwen3VLModel._run_decoder_flash","callee":"Qwen3VLTextDecoderLayer.forward","path":"sources/HiDream-O1-Image/models/qwen3_vl_transformers.py","line":1368,"count":36,"selected":True},
            {"caller":"Qwen3VLTextDecoderLayer.forward","callee":"Qwen3VLTextMLP.forward","path":"sources/HiDream-O1-Image/models/qwen3_vl_transformers.py","line":537,"count":36,"selected":True},
            {"caller":"Qwen3VLTextMLP.forward","callee":"gate_proj/SiLU/up_proj/product/down_proj","path":"sources/HiDream-O1-Image/models/qwen3_vl_transformers.py","line":492,"count":36,"selected":True},
            {"caller":"Qwen3VLForConditionalGeneration.forward","callee":"lm_head","path":"sources/HiDream-O1-Image/models/qwen3_vl_transformers.py","line":1914,"count":1,"selected":False,"note":"bypassed when vinputs is provided"},
        ],
        "assumptions": [
            "Checkpoint storage precision inferred from total_size == total_parameters*4; shards were intentionally excluded from metadata download.",
            "Inference analysis follows the official main-branch runtime commit cloned locally, not the generic Transformers class.",
        ],
        "open_questions": ["Full training loss and data curriculum are not present in the inference repository."],
        "conversation_followups": [],
    }
    (REPORT_DIR / "analysis-state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
