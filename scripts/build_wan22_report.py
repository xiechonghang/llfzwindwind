#!/usr/bin/env python3
"""Build the model-specific WAN2.2 T2V-A14B Diffusers architecture report."""

from __future__ import annotations

import html
import json
import subprocess
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "Wan-AI" / "Wan2.2-T2V-A14B-Diffusers"
REPORT_DIR = ROOT / "reports" / "Wan-AI" / "Wan2.2-T2V-A14B-Diffusers"
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
    return '<path class="flow" marker-end="url(#arrow)" d="M ' + " L ".join(f"{x} {y}" for x, y in points) + '"/>'


def svg(width, height, nodes, edges, labels=()):
    lane_labels = "".join(
        f'<text class="lane-label" x="{x}" y="{y}">{html.escape(label)}</text>' for x, y, label in labels
    )
    return f'''<div class="graph-scroll"><svg class="architecture-graph" viewBox="0 0 {width} {height}" role="img">
      <defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor"/></marker></defs>
      {lane_labels}{''.join(edges)}{''.join(nodes)}
    </svg></div>'''


def esc(raw):
    return html.escape(raw, quote=False)


def mapped(items):
    return "".join(f'<span class="code-map">{node_id}</span>{esc(code)}' for node_id, code in items)


def pair(title, graph, source, line_range, code, kind="trimmed verbatim"):
    first = line_range.split(",")[0].split("-")[0]
    return f'''<div class="graph-code-pair">
      <div class="graph-panel"><h4>{html.escape(title)}</h4>
        <div class="graph-legend"><span><i class="tensor-key"></i>张量</span><span><i></i>计算</span></div>{graph}
      </div>
      <div class="code-panel"><h4><a href="{source}#L{first}">{html.escape(source)}</a> · L{line_range}<span class="excerpt-kind">{kind}</span></h4>
        <pre><code>{code}</code></pre>
      </div>
    </div>'''


def git_commit(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    configs = {
        "model_index": json.loads((MODEL_DIR / "model_index.json").read_text()),
        "transformer": json.loads((MODEL_DIR / "transformer/config.json").read_text()),
        "transformer_2": json.loads((MODEL_DIR / "transformer_2/config.json").read_text()),
        "text_encoder": json.loads((MODEL_DIR / "text_encoder/config.json").read_text()),
        "vae": json.loads((MODEL_DIR / "vae/config.json").read_text()),
        "scheduler": json.loads((MODEL_DIR / "scheduler/scheduler_config.json").read_text()),
        "manifest": json.loads((MODEL_DIR / "metadata_manifest.json").read_text()),
    }
    diffusers_commit = git_commit(ROOT / "sources/diffusers")
    transformers_commit = git_commit(ROOT / "transformers")

    top_graph = svg(920, 760, [
        node("TL1", 20, 30, 235, 70, "prompt / negative_prompt\nT5 token ids [B,M≤512]", True),
        node("TL2", 342, 30, 235, 70, "Gaussian latents\n[B,16,F,HV,WV]", True),
        node("TL3", 665, 30, 235, 70, "scheduler timestep t\nscalar expanded to [B]", True),
        node("TL4", 20, 145, 235, 75, "UMT5EncoderModel ×24\nprompt_embeds [B,512,4096]", False),
        node("TL5", 342, 145, 235, 75, "selected DiT forward\nnoise_pred [B,16,F,HV,WV]", False),
        node("TL6", 665, 145, 235, 75, "boundary: 0.875×1000=875\nt ≥875 ? high : low", False),
        node("TL7", 115, 285, 285, 82, "high-noise transformer\n40 blocks · 14,288,491,584 params", False),
        node("TL8", 520, 285, 285, 82, "low-noise transformer_2\n40 blocks · 14,288,491,584 params", False),
        node("TL9", 317, 430, 285, 82, "selected expert output\nnoise_pred [B,16,F,HV,WV]", True),
        node("TL10", 317, 555, 285, 76, "CFG + UniPC scheduler.step\nupdated latents, next t", False),
        node("TL11", 317, 675, 285, 62, "AutoencoderKLWan.decode\nvideo [B,3,81,H,W]", False),
    ], [
        edge([(137,100),(137,145)]), edge([(460,100),(460,145)]), edge([(782,100),(782,145)]),
        edge([(137,220),(137,250),(257,250),(257,285)]), edge([(460,220),(460,250),(257,250),(257,285)]),
        edge([(460,220),(460,250),(662,250),(662,285)]), edge([(782,220),(782,250),(662,250),(662,285)]),
        edge([(400,326),(440,326),(440,430)]), edge([(520,326),(480,326),(480,430)]),
        edge([(460,512),(460,555)]), edge([(317,593),(270,593),(270,405),(460,405),(460,430)]), edge([(460,631),(460,675)]),
    ], [(257,275,"t ≥ 875"),(662,275,"t < 875")])
    top_code = mapped([
        ("TL1", "text_inputs = self.tokenizer(..., max_length=max_sequence_length, ...)\n"),
        ("TL4", "prompt_embeds = self.text_encoder(text_input_ids, mask).last_hidden_state\n"),
        ("TL2", "latents = self.prepare_latents(..., num_channels_latents, height, width, num_frames, ...)\n"),
        ("TL3", "for i, t in enumerate(timesteps):\n"),
        ("TL6", "boundary_timestep = self.config.boundary_ratio * self.scheduler.config.num_train_timesteps\n"),
        ("TL7", "if boundary_timestep is None or t >= boundary_timestep:\n    current_model = self.transformer\n"),
        ("TL8", "else:\n    current_model = self.transformer_2\n"),
        ("TL5", "noise_pred = current_model(hidden_states=latent_model_input, timestep=timestep, encoder_hidden_states=prompt_embeds)[0]\n"),
        ("TL9", "noise_pred = noise_uncond + current_guidance_scale * (noise_pred - noise_uncond)\n"),
        ("TL10", "latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]\n"),
        ("TL11", "video = self.vae.decode(latents, return_dict=False)[0]\n"),
    ])

    call_graph = svg(1040, 1080, [
        node("MC1", 315, 20, 410, 65, "WanPipeline.__call__ · ×1\nT2V inference wrapper", False),
        node("MC2", 315, 120, 410, 70, "UMT5EncoderModel.forward · ×1 or ×2\nencoder [B,512,4096]", False),
        node("MC3", 315, 225, 410, 75, "time router · one expert per step\nWanTransformer3DModel.forward", False),
        node("MC4", 315, 340, 410, 75, "patch_embedding + condition_embedder\n[B,L,5120] + [B,512,5120]", False),
        node("MC5", 315, 455, 410, 70, "blocks: WanTransformerBlock ×40\nhomogeneous schedule 0…39", False),
        node("MC6", 25, 575, 300, 80, "attn1: WanAttention\nself MHA + 3D RoPE", False),
        node("MC7", 370, 575, 300, 80, "attn2: WanAttention\ntext cross MHA", False),
        node("MC8", 715, 575, 300, 80, "ffn: FeedForward\ndense GELU MLP", False),
        node("MC9", 25, 710, 215, 78, "to_q / to_k / to_v\n5120→5120", False),
        node("MC10", 270, 710, 215, 78, "dispatch_attention_fn\nQKᵀ→softmax→PV", False),
        node("MC11", 515, 710, 215, 78, "to_out[0]\n5120→5120", False),
        node("MC12", 760, 710, 240, 78, "net[0].proj + GELU(tanh)\n5120→13824", False),
        node("MC13", 760, 840, 240, 78, "net[1] Dropout(0)\nnet[2] Linear 13824→5120", False),
        node("MC14", 315, 845, 360, 82, "gated residual merges ×3\nblock output [B,L,5120]", True),
        node("MC15", 315, 970, 360, 80, "norm_out → proj_out → unpatchify\n[B,16,F,HV,WV]", False),
    ], [
        edge([(520,85),(520,120)]), edge([(520,190),(520,225)]), edge([(520,300),(520,340)]), edge([(520,415),(520,455)]),
        edge([(520,525),(520,550),(175,550),(175,575)]), edge([(520,525),(520,575)]), edge([(520,525),(520,550),(865,550),(865,575)]),
        edge([(175,655),(175,710)]), edge([(325,749),(270,749)]), edge([(485,749),(515,749)]),
        edge([(865,655),(865,710)]), edge([(880,788),(880,840)]), edge([(175,788),(175,815),(400,815),(400,845)]),
        edge([(622,788),(622,815),(590,815),(590,845)]), edge([(760,879),(675,879)]), edge([(495,927),(495,970)]),
    ])
    call_code = mapped([
        ("MC1", "def __call__(..., num_inference_steps=50, ...):\n"),
        ("MC2", "prompt_embeds = self.text_encoder(...).last_hidden_state\n"),
        ("MC3", "current_model = self.transformer if t >= boundary_timestep else self.transformer_2\n"),
        ("MC4", "hidden_states = self.patch_embedding(hidden_states).flatten(2).transpose(1, 2)\n... = self.condition_embedder(timestep, encoder_hidden_states, ...)\n"),
        ("MC5", "for block in self.blocks:\n    hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)\n"),
        ("MC6", "attn_output = self.attn1(norm_hidden_states, None, None, rotary_emb)\n"),
        ("MC7", "attn_output = self.attn2(norm_hidden_states, encoder_hidden_states, None, None)\n"),
        ("MC8", "ff_output = self.ffn(norm_hidden_states)\n"),
        ("MC9", "query = attn.to_q(hidden_states); key = attn.to_k(...); value = attn.to_v(...)\n"),
        ("MC10", "hidden_states = dispatch_attention_fn(query, key, value, is_causal=False, ...)\n"),
        ("MC11", "hidden_states = attn.to_out[0](hidden_states)\n"),
        ("MC12", "hidden_states = self.proj(hidden_states)\nhidden_states = self.gelu(hidden_states)\n"),
        ("MC13", "for module in self.net:\n    hidden_states = module(hidden_states)\n"),
        ("MC14", "hidden_states = hidden_states + attn_output * gate_msa\n...\nhidden_states = hidden_states + ff_output * c_gate_msa\n"),
        ("MC15", "hidden_states = self.proj_out(self.norm_out(hidden_states.float()) * (1 + scale) + shift)\noutput = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)\n"),
    ])

    block_graph = svg(900, 820, [
        node("BL1", 300, 20, 300, 62, "hidden_states hℓ [B,L,5120]", True),
        node("BL2", 20, 125, 250, 78, "timestep_proj chunk(6)\nshift/scale/gate ×2 [B,1,5120]", False),
        node("BL3", 325, 125, 250, 78, "norm1 + affine modulation\n[B,L,5120]", False),
        node("BL4", 630, 125, 250, 78, "attn1 self + 3D RoPE\n[B,L,5120]", False),
        node("BL5", 325, 250, 250, 72, "h′ = hℓ + gate_msa·attn\n[B,L,5120]", False),
        node("BL6", 20, 375, 250, 72, "norm2 affine LayerNorm\n[B,L,5120]", False),
        node("BL7", 325, 375, 250, 72, "attn2 cross(text)\n[B,L,5120]", False),
        node("BL8", 630, 375, 250, 72, "h″ = h′ + cross_attn\n[B,L,5120]", False),
        node("BL9", 20, 520, 250, 78, "norm3 + affine modulation\n[B,L,5120]", False),
        node("BL10", 325, 520, 250, 78, "ffn GELU MLP\n5120→13824→5120", False),
        node("BL11", 630, 520, 250, 78, "hℓ+1 = h″ + c_gate·ffn\n[B,L,5120]", True),
        node("BL12", 325, 680, 250, 70, "next block (×40 total)\n[B,L,5120]", True),
    ], [
        edge([(450,82),(450,125)]), edge([(270,164),(325,164)]), edge([(575,164),(630,164)]), edge([(755,203),(755,225),(450,225),(450,250)]),
        edge([(450,82),(290,82),(290,286),(325,286)]), edge([(450,322),(450,345),(145,345),(145,375)]), edge([(270,411),(325,411)]),
        edge([(575,411),(630,411)]), edge([(755,447),(755,485),(145,485),(145,520)]), edge([(270,559),(325,559)]), edge([(575,559),(630,559)]),
        edge([(755,598),(755,640),(450,640),(450,680)]),
    ])
    block_code = mapped([
        ("BL1", "def forward(self, hidden_states, encoder_hidden_states, temb, rotary_emb):\n"),
        ("BL2", "shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (self.scale_shift_table + temb.float()).chunk(6, dim=1)\n"),
        ("BL3", "norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)\n"),
        ("BL4", "attn_output = self.attn1(norm_hidden_states, None, None, rotary_emb)\n"),
        ("BL5", "hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)\n"),
        ("BL6", "norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)\n"),
        ("BL7", "attn_output = self.attn2(norm_hidden_states, encoder_hidden_states, None, None)\n"),
        ("BL8", "hidden_states = hidden_states + attn_output\n"),
        ("BL9", "norm_hidden_states = (self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(hidden_states)\n"),
        ("BL10", "ff_output = self.ffn(norm_hidden_states)\n"),
        ("BL11", "hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)\n"),
        ("BL12", "return hidden_states\n"),
    ])

    self_graph = svg(960, 780, [
        node("SA1", 355, 20, 250, 62, "norm_hidden_states [B,L,5120]", True),
        node("SA2", 20, 130, 270, 78, "to_q + norm_q + unflatten\n[B,L,40,128]", False),
        node("SA3", 345, 130, 270, 78, "to_k + norm_k + unflatten\n[B,L,40,128]", False),
        node("SA4", 670, 130, 270, 78, "to_v + unflatten\n[B,L,40,128]", False),
        node("SA5", 20, 270, 270, 76, "3D RoPE (t,h,w=44,42,42)\nquery [B,L,40,128]", False),
        node("SA6", 345, 270, 270, 76, "3D RoPE (t,h,w=44,42,42)\nkey [B,L,40,128]", False),
        node("SA7", 240, 410, 480, 86, "dispatch_attention_fn · bidirectional MHA\nQKᵀ: [B,40,L,128]×[B,40,128,L]→[B,40,L,L]\nsoftmax; P×V→[B,L,40,128]", False),
        node("SA8", 355, 555, 250, 72, "flatten heads\n[B,L,5120]", False),
        node("SA9", 355, 680, 250, 72, "to_out[0] + dropout(0)\n[B,L,5120]", True),
    ], [
        edge([(480,82),(480,105),(155,105),(155,130)]), edge([(480,82),(480,130)]), edge([(480,82),(480,105),(805,105),(805,130)]),
        edge([(155,208),(155,270)]), edge([(480,208),(480,270)]), edge([(155,346),(155,380),(360,380),(360,410)]),
        edge([(480,346),(480,410)]), edge([(805,208),(805,380),(600,380),(600,410)]), edge([(480,496),(480,555)]), edge([(480,627),(480,680)]),
    ], [(155,118,"Q lane"),(480,118,"K lane"),(805,118,"V lane")])
    self_code = mapped([
        ("SA1", "query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)\n"),
        ("SA2", "query = attn.norm_q(query)\nquery = query.unflatten(2, (attn.heads, -1))\n"),
        ("SA3", "key = attn.norm_k(key)\nkey = key.unflatten(2, (attn.heads, -1))\n"),
        ("SA4", "value = value.unflatten(2, (attn.heads, -1))\n"),
        ("SA5", "query = apply_rotary_emb(query, *rotary_emb)\n"),
        ("SA6", "key = apply_rotary_emb(key, *rotary_emb)\n"),
        ("SA7", "hidden_states = dispatch_attention_fn(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False, ...)\n"),
        ("SA8", "hidden_states = hidden_states.flatten(2, 3)\n"),
        ("SA9", "hidden_states = attn.to_out[0](hidden_states)\nhidden_states = attn.to_out[1](hidden_states)\n"),
    ])

    cross_graph = svg(960, 700, [
        node("CA1", 70, 20, 300, 65, "video queries x [B,L,5120]", True),
        node("CA2", 590, 20, 300, 65, "text context c [B,M=512,5120]", True),
        node("CA3", 40, 140, 280, 78, "to_q + norm_q + unflatten\n[B,L,40,128]", False),
        node("CA4", 355, 140, 280, 78, "to_k + norm_k + unflatten\n[B,512,40,128]", False),
        node("CA5", 670, 140, 250, 78, "to_v + unflatten\n[B,512,40,128]", False),
        node("CA6", 240, 300, 480, 86, "dispatch_attention_fn · no RoPE · noncausal\nQKᵀ→scores [B,40,L,512]\nsoftmax; P×V→[B,L,40,128]", False),
        node("CA7", 355, 455, 250, 72, "flatten heads\n[B,L,5120]", False),
        node("CA8", 355, 580, 250, 72, "to_out[0]\n[B,L,5120]", True),
    ], [
        edge([(220,85),(220,140)]), edge([(740,85),(740,110),(495,110),(495,140)]), edge([(740,85),(740,140)]),
        edge([(180,218),(180,270),(360,270),(360,300)]), edge([(495,218),(495,300)]), edge([(795,218),(795,270),(600,270),(600,300)]),
        edge([(480,386),(480,455)]), edge([(480,527),(480,580)]),
    ], [(180,128,"query lane"),(495,128,"text K lane"),(795,128,"text V lane")])
    cross_code = mapped([
        ("CA1", "query = attn.to_q(hidden_states)\n"),
        ("CA2", "key = attn.to_k(encoder_hidden_states)\nvalue = attn.to_v(encoder_hidden_states)\n"),
        ("CA3", "query = attn.norm_q(query).unflatten(2, (attn.heads, -1))\n"),
        ("CA4", "key = attn.norm_k(key).unflatten(2, (attn.heads, -1))\n"),
        ("CA5", "value = value.unflatten(2, (attn.heads, -1))\n"),
        ("CA6", "hidden_states = dispatch_attention_fn(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, ...)\n"),
        ("CA7", "hidden_states = hidden_states.flatten(2, 3)\n"),
        ("CA8", "hidden_states = attn.to_out[0](hidden_states)\n"),
    ])

    ffn_graph = svg(780, 500, [
        node("FF1", 250, 20, 280, 65, "norm_hidden_states [B,L,5120]", True),
        node("FF2", 250, 125, 280, 78, "net[0].proj Linear\n[B,L,5120]×[5120,13824]\n→[B,L,13824]", False),
        node("FF3", 250, 245, 280, 70, "GELU approximate='tanh'\n[B,L,13824]", False),
        node("FF4", 250, 355, 280, 70, "net[1] Dropout(0)\nnet[2] Linear 13824→5120", False),
        node("FF5", 250, 455, 280, 40, "ff_output [B,L,5120]", True),
    ], [edge([(390,85),(390,125)]),edge([(390,203),(390,245)]),edge([(390,315),(390,355)]),edge([(390,425),(390,455)])])
    ffn_code = mapped([
        ("FF1", "def forward(self, hidden_states):\n"),
        ("FF2", "hidden_states = self.proj(hidden_states)\n"),
        ("FF3", "hidden_states = F.gelu(hidden_states, approximate=self.approximate)\n"),
        ("FF4", "for module in self.net:\n    hidden_states = module(hidden_states)\n"),
        ("FF5", "return hidden_states\n"),
    ])

    residual_graph = svg(840, 640, [
        node("RN1", 280, 20, 280, 62, "hℓ [B,L,5120]", True),
        node("RN2", 30, 130, 245, 72, "FP32LayerNorm norm1\nnon-affine", False),
        node("RN3", 300, 130, 245, 72, "(1+scale_msa)·x+shift_msa\nAdaLN modulation", False),
        node("RN4", 570, 130, 245, 72, "gate_msa·self_attn(x)\n[B,L,5120]", False),
        node("RN5", 300, 255, 245, 72, "h′ = hℓ + gated self-attn\n[B,L,5120]", True),
        node("RN6", 30, 385, 245, 72, "norm2 → cross-attn\nh″ = h′ + cross", False),
        node("RN7", 300, 385, 245, 72, "norm3 + c_scale/c_shift\n[B,L,5120]", False),
        node("RN8", 570, 385, 245, 72, "c_gate_msa·FFN\n[B,L,5120]", False),
        node("RN9", 300, 530, 245, 72, "hℓ+1 = h″ + gated FFN\n[B,L,5120]", True),
    ], [
        edge([(420,82),(420,105),(152,105),(152,130)]),edge([(275,166),(300,166)]),edge([(545,166),(570,166)]),
        edge([(692,202),(692,230),(422,230),(422,255)]),edge([(280,51),(10,51),(10,291),(300,291)]),
        edge([(422,327),(422,350),(152,350),(152,385)]),edge([(275,421),(300,421)]),edge([(545,421),(570,421)]),
        edge([(692,457),(692,495),(422,495),(422,530)]),
    ])
    residual_code = mapped([
        ("RN1", "origin_dtype = inputs.dtype\n"),
        ("RN2", "F.layer_norm(inputs.float(), ..., self.eps).to(origin_dtype)\n"),
        ("RN3", "norm_hidden_states = self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa\n"),
        ("RN4", "attn_output = self.attn1(norm_hidden_states, None, None, rotary_emb) * gate_msa\n"),
        ("RN5", "hidden_states = hidden_states.float() + attn_output\n"),
        ("RN6", "hidden_states = hidden_states + self.attn2(self.norm2(hidden_states.float()), encoder_hidden_states, None, None)\n"),
        ("RN7", "norm_hidden_states = self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa\n"),
        ("RN8", "ff_output = self.ffn(norm_hidden_states) * c_gate_msa\n"),
        ("RN9", "hidden_states = (hidden_states.float() + ff_output.float()).type_as(hidden_states)\n"),
    ])

    cache_graph = svg(860, 590, [
        node("KV1", 290, 20, 280, 65, "latents at step t\n[B,16,F,HV,WV]", True),
        node("KV2", 290, 120, 280, 70, "patch + 40 blocks\nrecompute Q/K/V", False),
        node("KV3", 20, 245, 245, 80, "self-attn K/V\n[B,L,40,128] ephemeral", True),
        node("KV4", 307, 245, 245, 80, "text cross-attn K/V\n[B,512,40,128] recomputed", True),
        node("KV5", 595, 245, 245, 80, "no past_key_values\nno update/append/reorder", False),
        node("KV6", 290, 375, 280, 75, "scheduler.step changes all latents\nnext step t′", False),
        node("KV7", 20, 495, 245, 72, "cache_context('cond/uncond')\ncontext label only by default", False),
        node("KV8", 595, 495, 245, 72, "VAE feat_cache\ncausal-conv chunk cache, not KV", False),
    ], [
        edge([(430,85),(430,120)]),edge([(430,190),(430,220),(142,220),(142,245)]),edge([(430,190),(430,245)]),
        edge([(430,190),(430,220),(717,220),(717,245)]),edge([(142,325),(142,350),(430,350),(430,375)]),
        edge([(430,325),(430,375)]),edge([(430,450),(430,475),(142,475),(142,495)]),edge([(430,450),(430,475),(717,475),(717,495)]),
    ])
    cache_code = mapped([
        ("KV1", "latent_model_input = latents.to(transformer_dtype)\n"),
        ("KV2", "noise_pred = current_model(hidden_states=latent_model_input, timestep=timestep, ...)[0]\n"),
        ("KV3", "query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)\n"),
        ("KV4", "key = attn.to_k(encoder_hidden_states); value = attn.to_v(encoder_hidden_states)\n"),
        ("KV5", "# WanAttention.__call__ has no past_key_values/cache_position argument\n"),
        ("KV6", "latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]\n"),
        ("KV7", "with current_model.cache_context('cond'):\n    noise_pred = current_model(...)\n"),
        ("KV8", "out = self.decoder(x[..., i:i+1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx)\n"),
    ])

    body = f'''
<section id="identity">
  <h2>1. 身份、范围与依据</h2>
  <p class="lead"><b>结论：</b><code>Wan-AI/Wan2.2-T2V-A14B-Diffusers</code> 是一个 Diffusers 复合 pipeline。核心不是单个 14B 模型内部的 token router，而是两个结构相同、权重不同的 14.288B 视频 DiT，沿扩散时间轴二选一执行。</p>
  <div class="grid">
    <div class="card"><b>仓库 revision</b><code>{configs['manifest']['revision'][:12]}</code>（完整值见 evidence）</div>
    <div class="card"><b>顶层类</b><code>WanPipeline</code>；边界比例 0.875</div>
    <div class="card"><b>本地 Diffusers</b><code>{diffusers_commit[:12]}</code></div>
    <div class="card"><b>本地 Transformers</b><code>{transformers_commit[:12]}</code></div>
    <div class="card"><b>配置版本</b>Diffusers 0.35.0.dev0；Transformers 4.48.0.dev0</div>
    <div class="card"><b>分析日期</b>{date.today().isoformat()}；推理路径</div>
  </div>
  <p class="source">现有 skill 证据脚本要求顶层 <code>config.json</code>，该仓库只有 <code>model_index.json</code> 与组件配置，故报告记录脚本兼容性失败并逐组件收集证据。权重 shard 按项目规则未下载，索引足以交叉验证模块名、层数和总字节数。</p>
</section>

<section id="hyperparameters">
  <h2>2. 配置与派生维度</h2>
  <p class="lead"><b>结论：</b>每个专家都是 40 层、隐藏宽度 5120 的 dense DiT；每层含一次全局视频 self-attention、一次文本 cross-attention 与 dense GELU FFN。发布索引精确对应每专家 14,288,491,584 个 FP32 参数。</p>
  <div class="grid">
    <div class="card"><b>DiT hidden</b><code>H=40×128=5120</code></div>
    <div class="card"><b>层数</b>40 / expert；两个 expert</div>
    <div class="card"><b>Attention</b>40 heads，head dim 128，MHA</div>
    <div class="card"><b>FFN</b><code>5120→13824→5120</code>，GELU(tanh)</div>
    <div class="card"><b>3D patch</b><code>(1,2,2)</code>；16 latent channels</div>
    <div class="card"><b>3D RoPE</b><code>D=(44 time,42 height,42 width)</code></div>
    <div class="card"><b>Text</b>UMT5-XXL encoder，24 层，4096 hidden，512 tokens</div>
    <div class="card"><b>VAE</b>16 channels；temporal ×4，spatial ×8</div>
  </div>
  <h3>精确参数核算（单个 DiT 专家）</h3>
  <table><thead><tr><th>部分</th><th>公式</th><th>参数</th></tr></thead><tbody>
    <tr><td>单个 WanTransformerBlock</td><td><code>8H² + 2HI + 21H + I</code></td><td>351,394,304</td></tr>
    <tr><td>40 blocks</td><td><code>40×351,394,304</code></td><td>14,055,772,160</td></tr>
    <tr><td>patch embedding</td><td><code>5120×(16×1×2×2)+5120</code></td><td>332,800</td></tr>
    <tr><td>time + text condition</td><td>two-layer time MLP + 6H projection + two-layer text projection</td><td>232,048,640</td></tr>
    <tr><td>output projection + AdaLN table</td><td><code>5120×64+64 + 2×5120</code></td><td>337,984</td></tr>
    <tr><th>单专家合计</th><th>索引 total_size / 4</th><th>14,288,491,584</th></tr>
    <tr><th>两个 DiT 参数库</th><th>高噪声 + 低噪声</th><th>28,576,983,168</th></tr>
  </tbody></table>
  <p>默认 480×832、81 帧：latent 为 <code>[B,16,21,60,104]</code>，patch 后 <code>L=21×30×52=32,760</code>。720×1280 时 <code>L=21×45×80=75,600</code>。self-attention 逻辑 score 分别是 <code>[B,40,32760,32760]</code> 或 <code>[B,40,75600,75600]</code>，实际后端通常避免完整物化。</p>
  <details><summary>查看组件配置</summary><pre>{html.escape(json.dumps({k:v for k,v in configs.items() if k != 'manifest'}, ensure_ascii=False, indent=2))}</pre></details>
</section>

<section id="top-level">
  <h2>3. 顶层生成路径</h2>
  <p class="lead"><b>结论：</b>UMT5 对正提示编码一次，启用 CFG 时再对负提示编码一次；每个扩散步按 <code>t≥875</code> 选高噪声专家，否则选低噪声专家。同一个已选专家顺序执行条件与无条件两次。</p>
  {pair('T2V pipeline 与时间轴专家路由', top_graph, '../../../sources/diffusers/src/diffusers/pipelines/wan/pipeline_wan.py', '158-197,535-668', top_code)}
  <div class="derived-note"><b>MoE 分类：</b>这是 diffusion-timestep expert routing。没有 learned router logits、top-k、token dispatch 或容量因子；“每步激活 14B、参数库约 28.6B”是准确表述。</div>
</section>

<section id="model-callgraph">
  <h2>4. 源码对齐调用树</h2>
  <p class="lead"><b>结论：</b>选中的运行路径从 <code>WanPipeline.__call__</code> 展开到一个 <code>WanTransformer3DModel</code>、40 个同构 block，再下钻至 self/cross attention 的 QKV 与 dense GELU FFN 的两次线性投影。</p>
  {pair('从 pipeline 到注意力与 FFN 叶算子', call_graph, '../../../sources/diffusers/src/diffusers/models/transformers/transformer_wan.py', '39-56,78-162,420-504,507-735', call_code, 'ordered trimmed excerpts')}
  <pre>WanPipeline                                                   ×1
├── tokenizer: T5TokenizerFast                               ×1
├── text_encoder: UMT5EncoderModel                           ×1
│   └── encoder.block: UMT5Block                             ×24
│       ├── SelfAttention: 64-head MHA + relative bias
│       └── DenseReluDense: gated-GELU 4096→10240→4096
├── transformer: WanTransformer3DModel (high-noise)          ×1
├── transformer_2: WanTransformer3DModel (low-noise)         ×1
│   └── selected expert per timestep                         ×1 active
│       └── blocks: WanTransformerBlock                      ×40
│           ├── attn1: WanAttention → to_q/to_k/to_v → dispatch_attention_fn → to_out
│           ├── attn2: WanAttention → video Q, text K/V → dispatch_attention_fn → to_out
│           └── ffn: FeedForward → proj → GELU(tanh) → Dropout(0) → Linear
└── vae: AutoencoderKLWan                                    ×1</pre>
</section>

<section id="decoder">
  <h2>5. WanTransformerBlock</h2>
  <p class="lead"><b>结论：</b>40 层完全同构。执行顺序是 gated AdaLN self-attention → 无门控 cross-attention 残差 → gated AdaLN dense FFN；并非标准两子层 Transformer。</p>
  {pair('单个 block 的三段串行残差', block_graph, '../../../sources/diffusers/src/diffusers/models/transformers/transformer_wan.py', '462-504', block_code)}
</section>

<section id="attention">
  <h2>6. Self-attention 与文本 Cross-attention</h2>
  <p class="lead"><b>结论：</b>两种注意力都是 40-head MHA（Q/K/V head 数均 40）。视频 self-attention 使用 3D RoPE；文本 cross-attention 不使用 RoPE，K/V 长度固定填充到 512。两者均 non-causal。</p>
  {pair('视频全局 self-attention', self_graph, '../../../sources/diffusers/src/diffusers/models/transformers/transformer_wan.py', '39-56,93-162', self_code)}
  {pair('视频 query × UMT5 文本 context', cross_graph, '../../../sources/diffusers/src/diffusers/models/transformers/transformer_wan.py', '39-56,93-162', cross_code)}
  <p><code>qk_norm="rms_norm_across_heads"</code> 的实际实现是在 unflatten 前对最后一维 5120 做 <code>torch.nn.RMSNorm</code>，随后 reshape 为 40×128。配置中的 <code>added_kv_proj_dim=null</code> 使 I2V 图像 K/V 分支不实例化。</p>
</section>

<section id="ffn-moe">
  <h2>7. Dense FFN 与 MoE 边界</h2>
  <p class="lead"><b>结论：</b>block 内 FFN 是普通两层 GELU MLP，不是 GEGLU/SwiGLU，也没有专家路由。WAN2.2 的专家选择只发生在 pipeline 的时间步边界。</p>
  {pair('Dense GELU FFN 叶算子', ffn_graph, '../../../sources/diffusers/src/diffusers/models/attention.py', '1682-1742', ffn_code, 'trimmed verbatim + activation definition')}
  <p>单层 FFN 参数：<code>5120×13824+13824 + 13824×5120+5120 = 141,576,704</code>。与常见 gated FFN 不同，它只有一条升维支路，无 gate/up 并行乘法。</p>
</section>

<section id="residual">
  <h2>8. 残差与归一化</h2>
  <p class="lead"><b>结论：</b>self-attention 与 FFN 都由时间条件产生的 gate 控制，并在归一化后施加 scale/shift；cross-attention 使用带 affine 的 FP32 LayerNorm，但没有显式时间 gate。</p>
  {pair('FP32 AdaLN 与三次残差更新', residual_graph, '../../../sources/diffusers/src/diffusers/models/transformers/transformer_wan.py', '469-502', residual_code, 'trimmed verbatim + FP32LayerNorm:L84-93')}
</section>

<section id="kv-cache">
  <h2>9. Cache 行为</h2>
  <p class="lead"><b>结论：</b>默认 T2V 去噪没有自回归 KV cache：每个 timestep 都对变化后的全部 latent token 重算 Q/K/V。<code>cache_context('cond'/'uncond')</code> 只是给可选 Diffusers cache hook 命名上下文；未显式 <code>enable_cache</code> 时不缓存。</p>
  {pair('无 KV append 的逐步重计算', cache_graph, '../../../sources/diffusers/src/diffusers/pipelines/wan/pipeline_wan.py', '589-635,656-668', cache_code, 'ordered trimmed excerpts + explicit absence')}
  <p>VAE decode 内部的 <code>feat_cache</code> 用于因果 3D 卷积按帧解码，它不是 Transformer KV cache，也不参与 denoising block attention。</p>
</section>

<section id="sources">
  <h2>10. 源码对应与可编辑图</h2>
  <p class="lead"><b>结论：</b>Diffusers 源码决定视频 DiT 与 pipeline 语义；Transformers 源码只决定 UMT5 encoder。权重索引为 40 个 block、两个独立 checkpoint 和关键模块名提供交叉验证。</p>
  <table><thead><tr><th>符号</th><th>本地源码</th><th>作用</th></tr></thead><tbody>
    <tr><td>WanPipeline.__call__</td><td><a href="../../../sources/diffusers/src/diffusers/pipelines/wan/pipeline_wan.py#L383">pipeline_wan.py:L383</a></td><td>时间步专家选择、CFG、scheduler、VAE</td></tr>
    <tr><td>WanTransformer3DModel.forward</td><td><a href="../../../sources/diffusers/src/diffusers/models/transformers/transformer_wan.py#L629">transformer_wan.py:L629</a></td><td>patchify、40-block loop、unpatchify</td></tr>
    <tr><td>WanTransformerBlock.forward</td><td><a href="../../../sources/diffusers/src/diffusers/models/transformers/transformer_wan.py#L462">transformer_wan.py:L462</a></td><td>AdaLN/gated residual topology</td></tr>
    <tr><td>WanAttnProcessor.__call__</td><td><a href="../../../sources/diffusers/src/diffusers/models/transformers/transformer_wan.py#L78">transformer_wan.py:L78</a></td><td>QKV、RoPE、attention dispatch</td></tr>
    <tr><td>FeedForward / GELU</td><td><a href="../../../sources/diffusers/src/diffusers/models/attention.py#L1682">attention.py:L1682</a> · <a href="../../../sources/diffusers/src/diffusers/models/activations.py#L65">activations.py:L65</a></td><td>dense FFN 叶算子</td></tr>
    <tr><td>UMT5EncoderModel</td><td><a href="../../../transformers/src/transformers/models/umt5/modeling_umt5.py#L1084">modeling_umt5.py:L1084</a></td><td>24-layer prompt encoder</td></tr>
    <tr><td>AutoencoderKLWan.decode</td><td><a href="../../../sources/diffusers/src/diffusers/models/autoencoders/autoencoder_kl_wan.py#L1187">autoencoder_kl_wan.py:L1187</a></td><td>latent → video，causal-conv feature cache</td></tr>
  </tbody></table>
  <h3>可编辑 Excalidraw 场景</h3>
  <p><a href="diagrams/01-top-level.excalidraw">顶层路由</a> · <a href="diagrams/02-diagram.excalidraw">调用树</a> · <a href="diagrams/03-diagram.excalidraw">block</a> · <a href="diagrams/04-diagram.excalidraw">self-attention</a> · <a href="diagrams/05-diagram.excalidraw">cross-attention</a> · <a href="diagrams/06-diagram.excalidraw">FFN</a> · <a href="diagrams/07-residual-norm.excalidraw">残差</a> · <a href="diagrams/08-kv-cache.excalidraw">cache</a></p>
</section>

<section id="unknowns">
  <h2>11. 兼容性、假设与待确认项</h2>
  <p class="lead"><b>结论：</b>结构与参数量已由组件 config、源码和权重索引闭环验证；不确定性集中在运行时 attention backend 与用户是否额外启用 cache/并行 hook。</p>
  <ul>
    <li><b>版本偏差：</b>checkpoint 记录 Diffusers 0.35.0.dev0、Transformers 4.48.0.dev0，本地 checkout 是更晚 commit。报告以当前本地运行时源码为语义依据，并标记兼容性差异。</li>
    <li><b>attention backend：</b><code>dispatch_attention_fn</code> 可选择 PyTorch SDPA、FlashAttention 等后端；报告画的是接口保证的逻辑张量流，不声称 score 矩阵一定物化。</li>
    <li><b>步数分配：</b>结构上可精确确认阈值 875；高/低噪声专家各运行多少实际 step 取决于 scheduler 生成的 timesteps 与用户指定步数。</li>
    <li><b>缓存：</b>模型继承 <code>CacheMixin</code>，但仓库配置不启用具体 cache hook；若用户运行时调用 <code>enable_cache</code>，行为会改变。</li>
    <li><b>权重精度：</b>每个 DiT 索引 <code>total_size=57,153,966,336</code>，恰为参数数×4，说明发布 shard 是 4-byte 存储；README 示例加载时转换为 BF16。</li>
  </ul>
  <details><summary>原始证据与状态文件</summary><p><a href="evidence.json">evidence.json</a> · <a href="analysis-state.json">analysis-state.json</a></p></details>
</section>
'''

    rendered = TEMPLATE.read_text(encoding="utf-8")
    rendered = rendered.replace("{{MODEL_TITLE}}", "WAN2.2 T2V-A14B · 时间轴双专家视频 DiT")
    rendered = rendered.replace("{{GENERATED_AT}}", date.today().isoformat())
    rendered = rendered.replace("{{REPORT_BODY}}", body)
    rendered = rendered.replace("基于具体配置与本地 Transformers 源码", "基于具体组件配置与本地 Diffusers / Transformers 源码")
    rendered = rendered.replace("</style>", ".code-panel h4 a { overflow-wrap:anywhere; word-break:break-word; }\n  </style>")
    (REPORT_DIR / "report.html").write_text(rendered, encoding="utf-8")

    evidence = {
        "schema_version": 1,
        "collection": {
            "skill_helper_command": "python3 .agents/skills/hf-model-architecture/scripts/collect_model_evidence.py Wan2.2-T2V-A14B-Diffusers",
            "helper_result": "failed: composite Diffusers repository has model_index.json but no top-level config.json",
            "fallback": "component configs and local Diffusers/Transformers sources collected directly",
        },
        "repository": configs["manifest"],
        "component_configs": {k: v for k, v in configs.items() if k != "manifest"},
        "source_revisions": {"diffusers": diffusers_commit, "transformers": transformers_commit},
        "weight_indexes": {
            "transformer": {"total_size": 57153966336, "tensor_names": 1095, "block_ids": list(range(40))},
            "transformer_2": {"total_size": 57153966336, "tensor_names": 1095, "block_ids": list(range(40))},
            "text_encoder": {"total_size": 11361820672, "tensor_names": 242, "block_ids": list(range(24))},
        },
        "matched_sources": [
            "sources/diffusers/src/diffusers/pipelines/wan/pipeline_wan.py",
            "sources/diffusers/src/diffusers/models/transformers/transformer_wan.py",
            "sources/diffusers/src/diffusers/models/attention.py",
            "sources/diffusers/src/diffusers/models/activations.py",
            "sources/diffusers/src/diffusers/models/autoencoders/autoencoder_kl_wan.py",
            "transformers/src/transformers/models/umt5/modeling_umt5.py",
        ],
    }
    (REPORT_DIR / "evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    state = {
        "schema_version": 1,
        "model": {
            "repo_id": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            "revision": configs["manifest"]["revision"],
            "config_path": "models/Wan-AI/Wan2.2-T2V-A14B-Diffusers/model_index.json",
            "architecture": ["WanPipeline", "WanTransformer3DModel", "UMT5EncoderModel", "AutoencoderKLWan"],
            "model_type": ["diffusers-pipeline", "wan-transformer-3d", "umt5"],
            "parameter_count_per_dit_expert": 14288491584,
            "parameter_count_two_dit_experts": 28576983168,
        },
        "transformers": {
            "commit": transformers_commit,
            "diffusers_commit": diffusers_commit,
            "source_directories": ["transformers/src/transformers/models/umt5", "sources/diffusers/src/diffusers/pipelines/wan", "sources/diffusers/src/diffusers/models/transformers"],
            "compatibility_notes": [
                "Checkpoint records diffusers 0.35.0.dev0 and transformers 4.48.0.dev0; local source revisions are newer.",
                "The skill collector expects a top-level config.json and does not directly support this composite Diffusers repository.",
            ],
        },
        "dimension_symbols": {
            "B": "batch size", "M": "padded text length, 512", "F": "latent frames", "HV": "latent height", "WV": "latent width",
            "L": "F*(HV/2)*(WV/2) video patch tokens", "H": 5120, "N_q": 40, "N_kv": 40, "D_qk": 128, "D_v": 128,
            "I": 13824, "P": [1,2,2], "C_latent": 16, "text_input_dim": 4096, "transformer_layers": 40, "text_layers": 24,
        },
        "configured_layer_schedule": [
            {"model": "transformer", "noise_stage": "high", "layers": "0-39", "type": "WanTransformerBlock", "count": 40},
            {"model": "transformer_2", "noise_stage": "low", "layers": "0-39", "type": "WanTransformerBlock", "count": 40},
        ],
        "conclusions": {
            "top_level": [
                "Two full 14.288B DiT experts are routed by diffusion timestep threshold 875; one expert is active per model evaluation.",
                "This is timestep-level MoE, not learned token-level top-k routing.",
                "UMT5-XXL encodes prompts; AutoencoderKLWan maps between 16-channel latents and RGB video.",
            ],
            "attention": [
                "Each block has 40-head MHA self-attention over all video tokens and 40-head MHA cross-attention to 512 text tokens.",
                "Self-attention uses 3D RoPE split 44/42/42; cross-attention has no RoPE.",
                "Attention backend is runtime-dispatched; logical computation is dense and non-causal.",
            ],
            "ffn_or_moe": [
                "Block FFN is dense GELU(tanh) 5120->13824->5120, not gated and not MoE.",
                "MoE boundary exists only at pipeline timestep routing between transformer and transformer_2.",
            ],
            "residual": ["Self-attention and FFN use timestep-conditioned scale/shift and output gates; cross-attention is an ungated residual after affine LayerNorm."],
            "kv_cache": [
                "Default denoising has no autoregressive KV cache; all Q/K/V are recomputed each timestep.",
                "cache_context labels optional Diffusers cache hooks and is inert unless a cache is explicitly enabled.",
                "VAE feat_cache is causal-convolution state, not attention KV cache.",
            ],
        },
        "source_symbols": [
            {"symbol":"WanPipeline._get_t5_prompt_embeds","path":"sources/diffusers/src/diffusers/pipelines/wan/pipeline_wan.py","line":158,"role":"prompt encoding"},
            {"symbol":"WanPipeline.__call__","path":"sources/diffusers/src/diffusers/pipelines/wan/pipeline_wan.py","line":383,"role":"timestep expert routing and denoising loop"},
            {"symbol":"WanTransformer3DModel.forward","path":"sources/diffusers/src/diffusers/models/transformers/transformer_wan.py","line":629,"role":"DiT forward"},
            {"symbol":"WanTransformerBlock.forward","path":"sources/diffusers/src/diffusers/models/transformers/transformer_wan.py","line":462,"role":"block residual topology"},
            {"symbol":"WanAttnProcessor.__call__","path":"sources/diffusers/src/diffusers/models/transformers/transformer_wan.py","line":78,"role":"self/cross attention"},
            {"symbol":"FeedForward.forward","path":"sources/diffusers/src/diffusers/models/attention.py","line":1736,"role":"dense FFN"},
            {"symbol":"GELU.forward","path":"sources/diffusers/src/diffusers/models/activations.py","line":87,"role":"FFN input projection and activation"},
            {"symbol":"UMT5EncoderModel.forward","path":"transformers/src/transformers/models/umt5/modeling_umt5.py","line":1129,"role":"text encoder wrapper"},
            {"symbol":"AutoencoderKLWan._decode","path":"sources/diffusers/src/diffusers/models/autoencoders/autoencoder_kl_wan.py","line":1187,"role":"video decode and convolution feature cache"},
        ],
        "graph_code_pairs": [
            {"section":"top-level","graph_nodes":[f"TL{i}" for i in range(1,12)],"path":"sources/diffusers/src/diffusers/pipelines/wan/pipeline_wan.py","line_start":158,"line_end":668,"excerpt_kind":"ordered trimmed excerpts"},
            {"section":"model-callgraph","graph_nodes":[f"MC{i}" for i in range(1,16)],"path":"sources/diffusers/src/diffusers/models/transformers/transformer_wan.py","line_start":39,"line_end":735,"excerpt_kind":"ordered trimmed excerpts"},
            {"section":"decoder","graph_nodes":[f"BL{i}" for i in range(1,13)],"path":"sources/diffusers/src/diffusers/models/transformers/transformer_wan.py","line_start":462,"line_end":504,"excerpt_kind":"trimmed verbatim"},
            {"section":"self-attention","graph_nodes":[f"SA{i}" for i in range(1,10)],"path":"sources/diffusers/src/diffusers/models/transformers/transformer_wan.py","line_start":39,"line_end":162,"excerpt_kind":"trimmed verbatim plus logical attention derivation"},
            {"section":"cross-attention","graph_nodes":[f"CA{i}" for i in range(1,9)],"path":"sources/diffusers/src/diffusers/models/transformers/transformer_wan.py","line_start":39,"line_end":162,"excerpt_kind":"trimmed verbatim plus logical attention derivation"},
            {"section":"ffn","graph_nodes":[f"FF{i}" for i in range(1,6)],"path":"sources/diffusers/src/diffusers/models/attention.py","line_start":1682,"line_end":1742,"excerpt_kind":"trimmed verbatim plus activation definition"},
            {"section":"residual","graph_nodes":[f"RN{i}" for i in range(1,10)],"path":"sources/diffusers/src/diffusers/models/transformers/transformer_wan.py","line_start":469,"line_end":502,"excerpt_kind":"trimmed verbatim"},
            {"section":"kv-cache","graph_nodes":[f"KV{i}" for i in range(1,9)],"path":"sources/diffusers/src/diffusers/pipelines/wan/pipeline_wan.py","line_start":589,"line_end":668,"excerpt_kind":"ordered trimmed excerpts plus explicit absence"},
        ],
        "model_call_hierarchy": [
            {"caller":"WanPipeline.__call__","callee":"UMT5EncoderModel.forward","path":"sources/diffusers/src/diffusers/pipelines/wan/pipeline_wan.py","line":536,"count":1,"selected":True},
            {"caller":"WanPipeline.__call__","callee":"WanTransformer3DModel.forward (selected expert)","path":"sources/diffusers/src/diffusers/pipelines/wan/pipeline_wan.py","line":615,"count":1,"selected":True},
            {"caller":"WanTransformer3DModel.forward","callee":"WanTransformerBlock.forward","path":"sources/diffusers/src/diffusers/models/transformers/transformer_wan.py","line":704,"count":40,"selected":True},
            {"caller":"WanTransformerBlock.forward","callee":"WanAttention.forward (attn1)","path":"sources/diffusers/src/diffusers/models/transformers/transformer_wan.py","line":489,"count":40,"selected":True},
            {"caller":"WanTransformerBlock.forward","callee":"WanAttention.forward (attn2)","path":"sources/diffusers/src/diffusers/models/transformers/transformer_wan.py","line":494,"count":40,"selected":True},
            {"caller":"WanTransformerBlock.forward","callee":"FeedForward.forward","path":"sources/diffusers/src/diffusers/models/transformers/transformer_wan.py","line":501,"count":40,"selected":True},
            {"caller":"FeedForward.forward","callee":"GELU.proj -> GELU(tanh) -> Dropout -> Linear","path":"sources/diffusers/src/diffusers/models/attention.py","line":1740,"count":40,"selected":True},
            {"caller":"WanPipeline.__call__","callee":"AutoencoderKLWan.decode","path":"sources/diffusers/src/diffusers/pipelines/wan/pipeline_wan.py","line":667,"count":1,"selected":True},
        ],
        "assumptions": [
            "Logical attention score shapes are derived from dispatch_attention_fn interface; a fused backend may not materialize scores.",
            "Published DiT storage precision is inferred from exact index total_size == derived parameter_count * 4.",
            "No optional Diffusers cache hook is enabled because no enable_cache call or cache config is present in the downloaded repository.",
        ],
        "open_questions": ["Exact high/low expert step counts depend on runtime scheduler timesteps and user-selected inference step count."],
        "conversation_followups": [],
    }
    (REPORT_DIR / "analysis-state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
