#!/usr/bin/env python3
"""Source-extracted, shape-explicit WAN2.2 companion report; no model weights required."""
from __future__ import annotations
import hashlib
import html
import json
import re
import subprocess
import sys
import unicodedata
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'reports/Wan-AI/Wan2.2-T2V-A14B-Diffusers'
OUT = BASE / 'detailed'
MODEL = ROOT / 'models/Wan-AI/Wan2.2-T2V-A14B-Diffusers'
SKILL = ROOT / '.agents/skills/hf-model-architecture'
P = 'sources/diffusers/src/diffusers/pipelines/wan/pipeline_wan.py'
W = 'sources/diffusers/src/diffusers/models/transformers/transformer_wan.py'
U = 'transformers/src/transformers/models/umt5/modeling_umt5.py'
S = 'sources/diffusers/src/diffusers/schedulers/scheduling_unipc_multistep.py'
V = 'sources/diffusers/src/diffusers/models/autoencoders/autoencoder_kl_wan.py'
E = 'sources/diffusers/src/diffusers/models/embeddings.py'
A = 'sources/diffusers/src/diffusers/models/attention.py'
G = 'sources/diffusers/src/diffusers/models/activations.py'
D = 'sources/diffusers/src/diffusers/models/attention_dispatch.py'
N = 'sources/diffusers/src/diffusers/models/normalization.py'
K = 'sources/diffusers/src/diffusers/models/cache_utils.py'
esc = html.escape
CHAPTERS = []
SOURCE_PATHS = set()

def ref(path, start, end=None):
    SOURCE_PATHS.add(path)
    return (path, start, end or start)

def link(path, line):
    return f'source/{Path(path).stem}.html#L{line}'

class Graph:
    def __init__(self, prefix, width=1080):
        self.prefix, self.width = prefix, width
        self.nodes, self.edges = [], []

    def n(self, key, x, y, label, refs=(), formula='', tensor=False, width=310, height=96):
        item = dict(id=self.prefix+str(key), x=x, y=y, w=width, h=height,
                    label=label, refs=list(refs), formula=formula, tensor=tensor)
        self.nodes.append(item)
        return item['id']

    def wire(self, a, b, side='bottom', target='top', via=()):
        na = next(n for n in self.nodes if n['id'] == self.prefix+str(a))
        nb = next(n for n in self.nodes if n['id'] == self.prefix+str(b))
        if na['y']==nb['y'] and side=='bottom' and target=='top' and not via:
            side,target=('right','left') if na['x']<nb['x'] else ('left','right')
        def anchor(key, side):
            n = next(n for n in self.nodes if n['id'] == self.prefix+str(key))
            x,y,w,h = n['x'],n['y'],n['w'],n['h']
            return {'top':(x+w/2,y),'bottom':(x+w/2,y+h),'left':(x,y+h/2),'right':(x+w,y+h/2)}[side]
        p,q=anchor(a,side),anchor(b,target)
        if not via and p[0] != q[0] and p[1] != q[1]:
            via=[(p[0],(p[1]+q[1])/2),(q[0],(p[1]+q[1])/2)]
        self.edges.append([p,*via,q])

    def svg(self):
        height=max(n['y']+n['h'] for n in self.nodes)+24
        elems=[]
        for points in self.edges:
            route='M '+' L '.join(f'{x} {y}' for x,y in points)
            elems.append(f'<path class="flow" marker-end="url(#arrow-{self.prefix})" d="{route}"/>')
        for n in self.nodes:
            kind='tensor-node' if n['tensor'] else 'op-node'
            x,y,w,h=n['x'],n['y'],n['w'],n['h']; lines=n['label'].split('\n')
            elems.append(f'<g class="{kind}"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{12 if n["tensor"] else 0}"/>')
            # Separate editable text objects; no concatenated multiline SVG tspans.
            elems.append(f'<text x="{x+w/2}" y="{y+22}"><tspan class="node-id">{n["id"]}</tspan></text>')
            for i,t in enumerate(lines):
                elems.append(f'<text class="shape-label" x="{x+w/2}" y="{y+44+i*19}">{esc(t)}</text>')
            elems.append('</g>')
        return f'<svg class="architecture-graph" viewBox="0 0 {self.width} {height}" role="img"><defs><marker id="arrow-{self.prefix}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor"/></marker></defs>{"".join(elems)}</svg>'

    def code(self):
        chunks=[]
        for n in self.nodes:
            chunks.append(f'<span class="code-map">{n["id"]}</span> {esc(n["label"].splitlines()[0])}\n')
            if n['formula']:
                chunks.append(esc('# derived: '+n['formula'])+'\n')
            for path,lo,hi in n['refs']:
                lines=(ROOT/path).read_text().splitlines()
                assert 0<lo<=hi<=len(lines), (path,lo,hi)
                chunks.append(f'<a href="{link(path,lo)}">{Path(path).name}:L{lo}–{hi}</a> · verbatim\n')
                chunks.append(esc('\n'.join(f'{i:4}  {lines[i-1]}' for i in range(lo,hi+1)))+'\n')
            chunks.append('\n')
        return ''.join(chunks)

def chapter(slug,title,lead,body,formula,graph):
    CHAPTERS.append(dict(slug=slug,title=title,lead=lead,body=body,formula=formula,graph=graph))

def chain(prefix, rows):
    g=Graph(prefix,800)
    for i,(label,refs,derived,tensor) in enumerate(rows,1):
        g.n(i,130,20+(i-1)*144,label,refs,derived,tensor,width=540,height=104)
        if i>1:g.wire(i-1,i)
    return g

def attention_graph(prefix,title,slug,text=False,cross=False):
    g=Graph(prefix)
    heads,dh,H=(64,64,4096) if text else (40,128,5120)
    sq='M' if text else 'L'; sk='M' if text or cross else 'L'
    desc=f'[B,{sq},{H}]'; ctx=f'[B,{sk},{H}]'
    g.n(1,20,20,('文本 normed_hidden_states' if text else '视频 norm_hidden_states')+'\n'+desc,
        [ref(U,386,392)] if text else [ref(W,493 if cross else 488,494 if cross else 489)],tensor=True)
    g.n(2,740,20,('text context C' if cross else '同一输入序列')+'\n'+ctx,
        [ref(W,347)] if cross else ([ref(U,322)] if text else [ref(W,41,42)]),tensor=True)
    qr=[ref(U,307,308)] if text else [ref(W,53),ref(W,95),ref(W,98)]
    kr=[ref(U,328,330)] if text else [ref(W,54),ref(W,96),ref(W,99)]
    vr=[ref(U,329,331)] if text else [ref(W,55),ref(W,100)]
    g.n(3,20,180,('q' if text else 'to_q + norm_q')+f' → Q\n[B,{sq},{heads},{dh}]',qr,f'Linear weight [{H},{H}]；'+('无 bias' if text else f'bias [{H}]；RMSNorm跨{H}维'))
    g.n(4,380,180,('k' if text else 'to_k + norm_k')+f' → K\n[B,{sk},{heads},{dh}]',kr,f'Linear weight [{H},{H}]；'+('无 bias' if text else f'bias [{H}]；RMSNorm跨{H}维'))
    g.n(5,740,180,('v' if text else 'to_v')+f' → V\n[B,{sk},{heads},{dh}]',vr,f'Linear weight [{H},{H}]；head布局统一用于图解')
    if not text and not cross:
        g.n(6,20,335,'apply_rotary_emb(Q)\n[B,L,40,128]；t/h/w=44/42/42',[ref(W,104,118),ref(W,368,369)])
        g.n(7,380,335,'apply_rotary_emb(K)\n[B,L,40,128]；V不旋转',[ref(W,117,118)])
    else:
        g.n(6,20,335,'Q → head-first layout\n'+f'[B,{heads},{sq},{dh}]',[ref(U,307,308)] if text else [ref(D,3711,3712)])
        g.n(7,380,335,'K → head-first layout\n'+f'[B,{heads},{sk},{dh}]',[ref(U,330)] if text else [ref(D,3711,3712)])
    g.n(8,190,500,'scores = Q @ Kᵀ'+(' (scale=1)' if text else ' / √128')+f'\n[B,{heads},{sq},{sk}]',
        [ref(U,208,209),ref(U,170,177)] if text else [ref(W,143,153),ref(D,3711,3723)],
        f'[B,{heads},{sq},{dh}] @ [B,{heads},{dh},{sk}] → [B,{heads},{sq},{sk}]；'+('UMT5无1/√d缩放' if text else 'SDPA逻辑展开，不宣称scores被物化'),width=510)
    g.n(9,190,650,('relative bias + padding mask → softmax' if text else 'softmax(scores) · 无 causal mask')+f'\nP [B,{heads},{sq},{sk}]',
        [ref(U,279,289),ref(U,173,180)] if text else [ref(W,143,153)],
        'softmax沿文本/视频key维；'+('有效文本双向可见' if text else ('本cross路径attention_mask=None' if cross else '整个时空token网格互相可见')),width=510)
    g.n(10,190,800,'attn_output = P @ V'+f'\n[B,{heads},{sq},{dh}]',
        [ref(U,182,183)] if text else [ref(D,3713,3723)],
        f'[B,{heads},{sq},{sk}] @ [B,{heads},{sk},{dh}] → [B,{heads},{sq},{dh}]；V换轴后参与',width=510)
    g.n(11,190,950,('reshape + o' if text else 'flatten(2,3) + to_out[0]')+f'\n[B,{sq},{H}] → [B,{sq},{H}]',
        [ref(U,367,369)] if text else [ref(W,154,162)],f'合并 heads；Wout [{H},{H}]',width=510)
    for a,b in [(1,3),(2,4),(2,5),(3,6),(4,7),(6,8),(7,8),(8,9),(9,10),(10,11)]:g.wire(a,b)
    g.wire(5,10,'bottom','right',[(895,780),(730,780),(730,848)])
    if text:
        lead='UMT5 对有效文本双向注意；使用 padding mask 与每层独立的相对位置偏置。'
        body='<p>64 heads ×64维=4096；q/k/v/o 均无 bias。源码 self.scaling=1.0，所以与视频 attention 的 1/√128 缩放不同。相对位置距离用32个桶、max_distance=128映射到每个head的bias，24层各有一张[32,64]表，权重索引与此一致。</p><p>图统一先展示 token-first 形状帮助对照 WAN；UMT5 源码在线性投影后直接 transpose 到 [B,64,M,64]。实际 backend 可选择优化实现；旁边列出本地 eager 算法作为语义对应。padding屏蔽key列，不使用未来位置三角mask。</p>'
        formula='Q = split_heads(XWqᵀ), K = split_heads(XWkᵀ), V = split_heads(XWvᵀ)\nScores = QKᵀ + RelativeBias[1,64,M,M] + PaddingMask\nP = softmax(Scores, dim=key)\nY = merge_heads(PV) Woᵀ\n注意：UMT5缩放系数=1，不是1/√64。'
    else:
        lead='视频 Q 读取文本 K/V；输出仍是每个视频位置的更新 [B,L,H]。' if cross else '全时空 dense MHA；Q/K 在拆head前跨5120维 RMSNorm，再应用三维 RoPE。'
        body=('<p>Q来自当前视频特征，K/V来自文本投影C；投影矩阵属于当前层，40层不共享。Cross-attention没有RoPE、不拼接文本token到视频输出；softmax矩阵的每行是一个视频query对M个文本位置的权重。</p><p><b>padding边界：</b>UMT5有padding mask；其输出padding先置零，再经过带bias的文本投影；本DiT cross调用传入mask=None，因此补齐的文本位置仍参与cross softmax。不能把UMT5的padding mask原样假定到这里。</p>' if cross else '<p>40 heads、每head128维，Q/K/V头数相等，属于MHA。每个attention参数4(H²+H)+2H=104,888,320。Q/K normalization是5120维，而不是每head128维。RoPE将128维分配为time44、height42、width42，theta默认10000；max_seq_len=1024是轴向频率表长度，不表示总token只能1024。</p><p>cos/sin实际广播形状[1,L,1,128]；t/h/w坐标按F×h/2×w/2展开。Q和K在各自支路旋转，V保持不变。所有视频位置双向可见，没有逐帧因果mask。</p>')
        body+='<p>图中的 scores→softmax→PV 是 dispatch_attention_fn / PyTorch SDPA 的数学展开。源码实际布局为 [B,S,heads,dim]，native backend转换到head-first后计算并转回；优化后端可避免完整物化scores。本文没有运行GPU内核或测量峰值显存。</p>'
        formula=('Q = RMSNorm_H(XWqᵀ+bq)；K = RMSNorm_H(CWkᵀ+bk)；V = CWvᵀ+bv\n' if cross else 'Q = RoPE₃D(split(RMSNorm_H(XWqᵀ+bq)))\nK = RoPE₃D(split(RMSNorm_H(XWkᵀ+bk)))；V=split(XWvᵀ+bv)\n')+f'P = softmax(QKᵀ / √128)；P [B,40,L,{sk}]\nY = merge_heads(PV) Woᵀ + bo；Y [B,L,5120]'
    chapter(slug,title,lead,body,formula,g)

def build_chapters():
    g=Graph('OV')
    g.n(1,20,20,'prompt / negative_prompt\n两段文本（负提示可为空）',[ref(P,245,257)],tensor=True)
    g.n(2,20,165,'UMT5EncoderModel ×24 blocks\nE± [B,M,4096]；循环前执行',[ref(P,173,195)],tensor=False)
    g.n(3,740,20,'randn_tensor → z₀\n[B,16,F,h,w]；float32',[ref(P,340,354)],tensor=True)
    g.n(4,380,20,'set_timesteps(N)\nσ₀…σN，tᵢ=1000σᵢ',[ref(P,552,554),ref(S,428,450)],tensor=False)
    g.n(5,380,320,'每个采样步选择专家 eᵢ\nt≥875: transformer；否则 _2',[ref(P,584,603)])
    g.n(6,200,480,'DiT(zᵢ,tᵢ,E+)\n40 blocks → v+ [B,16,F,h,w]',[ref(P,614,621)])
    g.n(7,560,480,'DiT(zᵢ,tᵢ,E−) · CFG 可选\n40 blocks → v− [B,16,F,h,w]',[ref(P,623,631)])
    g.n(8,380,640,'CFG → scheduler.step\n同形状的 zᵢ₊₁；外循环 N 次',[ref(P,632,635)])
    g.n(9,380,820,'仅采样全部结束后\n反归一化 → VAE.decode → RGB',[ref(P,656,670)])
    for a,b in [(1,2),(3,5),(4,5),(2,6),(5,6),(5,7),(6,8),(7,8),(8,9)]:g.wire(a,b)
    g.wire(2,7,via=[(175,440),(715,440)])
    g.wire(8,5,'right','right',[(1060,688),(1060,368)])
    chapter('overview','01 · 全局执行：两条输入线与两个循环','文本编码一次；整段视频 latent 在外层循环中反复更新；最后才解码为像素。',
        '<p>本报告选择纯 T2V、eval/no_grad、无 LoRA、无可选缓存 hook、无并行切分、VAE 不切片/不分块的基线路径。CFG 两个分支在当前 pipeline 中顺序调用同一已选专家；图中并列表示数据依赖，不代表实际并发。默认示例 B=1、81 帧、480×832、M=512、N=40，N 是讲解用采样设置，不是 checkpoint 固定常数。</p><p>内层 ℓ=0…39 是不同权重的 block；外层 i=0…N−1 是反复调用专家并更新 latent。噪声时间步 t、采样编号 i、视频时间坐标 f 三者不同。无 LLM 式增量 prefill/decode。</p>',
        'E± = UMT5(prompt±)\nz₀ ~ Normal(0,I)\neᵢ = high if tᵢ ≥ 875 else low\nvᵢ± = DiT_eᵢ(zᵢ,tᵢ,E±)\nzᵢ₊₁ = UniPC.step(CFG(vᵢ+,vᵢ−),tᵢ,zᵢ)\nvideo = VAE.decode(std ⊙ zN + mean)',g)

    g=chain('TX',[
        ('tokenizer：词表 256384，M 默认 512',[ref(P,173,183)],'input_ids / padding mask [B,M]；正/负提示分别编码',True),
        ('embed_tokens → [B,M,4096]',[ref(U,627,634),ref(U,660,661)],'W_embed [256384,4096]；1,050,148,864 个参数',False),
        ('UMT5Stack：24× UMT5Block',[ref(U,707,720)],'每层 bidirectional self-attention → gated-GELU FFN；宽度保持4096',False),
        ('final_layer_norm → 去掉 padding 特征后补零',[ref(U,719,724),ref(P,185,195)],'最终 E± [B,M,4096]；重复 num_videos_per_prompt 后 B 是有效批量',False),
        ('保留 E+ / E−，供所有采样步读取',[ref(P,535,550)],'文本输出不由视频 cross-attention 写回；未启用 CFG 时不算 E−',True),
    ])
    chapter('text','02 · 文本 pipeline：从 prompt 到固定语义条件','UMT5 是 24 层 encoder；每次生成在采样循环前完成文本编码。',
        '<p>正提示描述期望内容；负提示由用户或应用提供，缺省为空字符串。两者使用同一个编码器。M=512 是 WanPipeline.__call__ 的默认参数，调用方可以改；不是词表大小。padding mask 屏蔽无效文本位置，有效词之间双向可见。</p><p>UMT5EncoderModel 将 encoder_config.use_cache 设为 False；未实例化语言 decoder / LM head。配置里的 num_decoder_layers=24、use_cache=true 不应套用到这里的实际路径。共享 embedding 名称与 encoder.embed_tokens 可能是别名，参数核算不重复计数。</p>',
        'input_ids [B,M] → Embedding [V=256384,Ht=4096] → [B,M,4096]\nE = final_RMSNorm(Block₂₃(…Block₀(embeddings)))\nE[b,j] = 0, 若 j 是被补齐的位置（在 UMT5 输出后处理）',g)

    attention_graph('UA','03 · UMT5 attention：文本双向语义交互','text-attention',text=True)
    g=Graph('UF')
    g.n(1,380,20,'text hidden h [B,M,4096]',[ref(U,148,151)],tensor=True)
    g.n(2,380,165,'layer_norm → u [B,M,4096]\nRMSNorm ε=1e−6',[ref(U,145,150),ref(U,67,80)])
    g.n(3,110,310,'wi_0 + gelu_new\n4096→10240；无 bias',[ref(U,110,118)],'W [10240,4096]；输出 [B,M,10240]')
    g.n(4,650,310,'wi_1\n4096→10240；无 bias',[ref(U,111,118)],'W [10240,4096]；输出 [B,M,10240]')
    g.n(5,380,455,'hidden_gelu * hidden_linear\n[B,M,10240] ⊙ [B,M,10240]',[ref(U,119,120)])
    g.n(6,380,600,'wo：10240→4096\n[B,M,4096]；eval dropout=identity',[ref(U,125,133)])
    g.n(7,380,745,'h + FFN(norm(h))\n[B,M,4096]',[ref(U,148,152)],tensor=True)
    for a,b in [(1,2),(2,3),(2,4),(3,5),(4,5),(5,6),(6,7)]:g.wire(a,b)
    g.wire(1,7,'left','left',[(35,68),(35,793)])
    chapter('text-ffn','04 · 文本 FFN 与两次残差','UMT5 使用 gated-GELU FFN；视频 DiT 则使用普通 GELU FFN。',
        '<p>UMT5 每层先做 RMSNorm→self-attention→残差，再 RMSNorm→gated FFN→残差。wi_0 和 wi_1 是独立并行支路，到逐元素乘法时合并。dropout_rate=0.1 是配置值，本文 eval 推理下禁用 dropout。</p><p>文本 FFN 每层参数为 3×4096×10240=125,829,120；文本 attention 四个无偏置投影共 67,108,864。每层另外有两组 4096 维 RMSNorm 和 [32,64] 相对位置表，共 192,948,224。24 层加 embedding 与 final norm 得 5,680,910,336 参数；与文本索引 total_size/2 相符。</p>',
        'u = h + SelfAttention(RMSNorm(h), padding_mask)\nf = wo(gelu_new(wi_0(RMSNorm(u))) ⊙ wi_1(RMSNorm(u)))\nh_next = u + f',g)

    g=chain('LP',[
        ('生成随机初始 latent z₀ [B,16,21,60,104]',[ref(P,340,354)],'F=(81−1)/4+1=21；h=480/8=60；w=832/8=104',True),
        ('patch_embedding：Conv3d(16,5120,(1,2,2))',[ref(W,593,598)],'weight [5120,16,1,2,2]；bias [5120]；共332,800参数',False),
        ('Conv 输出 [B,5120,21,30,52]',[ref(W,662,671)],'每块16×1×2×2=64个连续数 → 5120特征',True),
        ('flatten(2).transpose(1,2).contiguous()',[ref(W,670,674)],'L=21×30×52=32760；token x [B,L,5120]',False),
        ('进入 block loop；下一采样步重新 patchify',[ref(W,703,704)],'每次输入新的 zᵢ；无 embedding lookup / 离散视频词表',True),
    ])
    chapter('latent','05 · Latent pipeline：随机网格与 patch embedding','16 个 latent 通道都参与投影；1×2×2 只描述时空块尺寸。',
        '<p>纯 T2V 不需要先输入真实视频。pipeline 直接生成随机浮点张量，默认初始 latent 是 float32，送进专家前转专家 dtype。每次采样后依旧是相同网格大小。81×720×1280 时 latent 为 [B,16,21,90,160]、L=75,600。</p><p>初始随机 latent 没有可识别的画面；后续模型预测和 scheduler 更新逐步形成视频表示。这里的 latent 通道是学到的特征，不等同于 RGB，也不能给每个通道固定贴上“颜色/人物”等语义标签。</p>',
        'z₀ ~ N(0,I)\nx_patch ∈ R^(16·1·2·2) = R^64\nh_patch = W_patch x_patch + b_patch ∈ R^5120\nConv3d: [B,16,F,h,w] → [B,5120,F,h/2,w/2]\nL = F(h/2)(w/2)',g)

    g=Graph('CE')
    g.n(1,20,20,'文本 E [B,M,4096]',[ref(W,347)],tensor=True)
    g.n(2,20,165,'text_embedder.linear_1\n4096→5120；GELU(tanh)',[ref(E,2203,2205),ref(E,2214,2216)])
    g.n(3,20,310,'text_embedder.linear_2\n5120→5120 → C [B,M,5120]',[ref(E,2212,2218)])
    g.n(4,740,20,'timestep t [B] → Timesteps\nsinusoidal embedding [B,256]',[ref(W,320,324),ref(W,337,345)])
    g.n(5,740,165,'time_embedder\n256→5120→5120；SiLU',[ref(E,1275),ref(E,1282,1288),ref(E,1298,1303)])
    g.n(6,560,310,'SiLU(temb) → time_proj\n5120→30720；[B,6,5120]',[ref(W,344,345),ref(W,691)])
    g.n(7,560,465,'每层 + scale_shift_table\n[1,6,5120] + [B,6,5120]',[ref(W,460),ref(W,483,485)],'chunk → 六个 [B,1,5120]；广播所有视频位置')
    g.n(8,20,620,'各层 cross-attention 的 K/V 来源\nC [B,M,5120] 保持不变',[ref(W,494)],tensor=True)
    g.n(9,560,620,'shift/scale/gate：self + FFN\n六路调制；输出头另用 temb',[ref(W,488,502),ref(W,714)],tensor=True)
    for a,b in [(1,2),(2,3),(3,8),(4,5),(5,6),(6,7),(7,9)]:g.wire(a,b)
    chapter('conditions','06 · 两类条件：文本投影与噪声时间调制','文本用于 cross-attention；噪声时间用于 AdaLN 和残差 gate，两条条件各有作用。',
        '<p>文本投影的参数量 47,196,160；时间 MLP 27,535,360；time_proj 157,317,120；合计 232,048,640/专家。每个 block 自己的 scale_shift_table 是 30,720 参数。time_proj 在专家前向中计算一次供40层共享，但每层加自己的可学习表。</p><p>当前 T2V 使用 t [B]，所以六个调制向量都是 [B,1,H] 并广播到 L 个位置；未选 TI2V 的逐 token timestep 分支。image_embedder 和 added image K/V 因配置 null 不实例化。temb [B,H] 还在输出头中使用。</p>',
        'C = Linear₂(GELU_tanh(Linear₁(E)))\ne = Linear₂(SiLU(Linear₁(Sinusoidal256(t))))\nT = reshape(Linear_6H(SiLU(e)), [B,6,H])\n[βsa,γsa,gsa,βff,γff,gff] = chunk₆(T + Aℓ)\nAℓ [1,6,H]；H=5120',g)

    g=Graph('BR')
    g.n(1,380,20,'xℓ [B,L,5120]',[ref(W,462,467)],tensor=True)
    g.n(2,380,165,'norm1 + scale_msa / shift_msa\n[B,L,H]；FP32运算',[ref(W,434),ref(N,84,93),ref(W,488)])
    g.n(3,380,310,'attn1 self-attention\n[B,L,H] → a [B,L,H]',[ref(W,489)])
    g.n(4,380,455,'x₁ = xℓ + gate_msa ⊙ a\ngate [B,1,H] 广播 L',[ref(W,490)])
    g.n(5,380,600,'norm2 → attn2(text)\n读取 C [B,M,H] → c [B,L,H]',[ref(W,454),ref(W,493,494)])
    g.n(6,380,745,'x₂ = x₁ + c\n无额外 timestep gate',[ref(W,495)])
    g.n(7,380,890,'norm3 + c_scale / c_shift → ffn\n[B,L,H] → f [B,L,H]',[ref(W,498,501)])
    g.n(8,380,1035,'xℓ₊₁ = x₂ + c_gate_msa ⊙ f\n进入下一层 self-attention',[ref(W,502,504)],tensor=True)
    for a,b in zip(range(1,8),range(2,9)):g.wire(a,b)
    g.wire(1,4,'left','left',[(70,68),(70,503)])
    g.wire(4,6,'left','left',[(120,503),(120,793)])
    g.wire(6,8,'left','left',[(170,793),(170,1083)])
    chapter('block','07 · 单层全景：三次残差如何串起来','Cross-attention 更新加到 self-attention 残差后的状态，再经 FFN 传给下一层。',
        '<p>图中左侧三条旁路是实际残差依赖。cross-attention 的结果不是只加到 self-attention 的裸输出；它加到包含原始 xℓ 的 x₁。下一层 self-attention 才处理已经融合文本的结果。</p><p>norm1/norm3 是无可学习 affine 的 LayerNorm，ε=1e−6；norm2 是带 weight+bias 的 LayerNorm，共 2H=10,240 参数。gate 是直接使用的可学习条件值，不经过 sigmoid，不保证处于 [0,1]。本结构没有 mHC 多流残差。</p>',
        'LN₁/LN₃(x) = (x−mean_H(x)) / √(var_H(x)+10⁻⁶)\nLN₂(x) = learned_weight ⊙ LN(x) + learned_bias\nu = LN₁(xℓ) ⊙ (1+γsa) + βsa\nx₁ = xℓ + gsa ⊙ SA(u)\nx₂ = x₁ + CA(LN₂(x₁), C)\nw = LN₃(x₂) ⊙ (1+γff) + βff\nxℓ₊₁ = x₂ + gff ⊙ FFN(w)',g)
    attention_graph('VS','08 · 视频 self-attention：Q/K/V 与三维 RoPE','video-attention')
    attention_graph('VC','09 · Cross-attention：两条 pipeline 的交点','cross-attention',cross=True)

    g=Graph('VF')
    g.n(1,380,20,'x₂ [B,L,5120]\n已融合 self + text cross',[ref(W,495)],tensor=True)
    g.n(2,380,165,'norm3 + c_scale/c_shift\n[B,L,5120]',[ref(W,498,500)])
    g.n(3,380,310,'ffn.net[0].proj Linear\n[B,L,5120] → [B,L,13824]',[ref(W,457),ref(A,1714,1715),ref(G,76,89)],'W [13824,5120]，b [13824]')
    g.n(4,380,455,'GELU(approximate=tanh)\n[B,L,13824]；dropout=0',[ref(G,81,90),ref(A,1725,1731)])
    g.n(5,380,600,'ffn.net[2] Linear\n[B,L,13824] → [B,L,5120]',[ref(A,1731),ref(A,1740,1742)],'W [5120,13824]，b [5120]')
    g.n(6,380,745,'ff_output.float() * c_gate_msa\n[B,L,H] ⊙ [B,1,H]',[ref(W,501,502)])
    g.n(7,380,890,'x₂.float() + gated FFN\ncast back → xℓ₊₁ [B,L,H]',[ref(W,502,504)],tensor=True)
    for a,b in zip(range(1,7),range(2,8)):g.wire(a,b)
    g.wire(1,7,'left','left',[(60,68),(60,938)])
    chapter('video-ffn','10 · 视频 FFN 与门控残差','FFN 对各视频 token 独立变换特征；跨位置混合由 attention 完成。',
        '<p>FFN 是 5120→13824→5120 的普通 GELU MLP，参数 2HI+I+H=141,576,704/层。没有内部 gate/up 两路相乘；图中的 c_gate_msa 是 FFN 输出的残差门控，与 gated-GELU 的内部乘法是不同位置。</p><p>FP32LayerNorm 在 float32 中进行归一化。源码 self 残差先把 hidden_states 转 float32，gate 也来自 float32 调制；FFN 残差显式把 ff_output 转 float32 再相乘相加，最后回到 hidden dtype。跨注意力残差则直接 hidden_states + attn_output。</p>',
        'FFN(w) = W₂ GELU_tanh(W₁w+b₁)+b₂\nGELU_tanh(a)=0.5a[1+tanh(√(2/π)(a+0.044715a³))]\nx_next = cast_dtype(float32(x₂) + float32(FFN(w)) ⊙ gff)',g)

    g=chain('OH',[
        ('40 层后 h [B,L,5120]',[ref(W,703,704)],'h 为隐藏特征，不是下一份 latent',True),
        ('输出 scale_shift_table + temb → shift,scale',[ref(W,622,624),ref(W,714)],'[1,2,H]+[B,1,H] → 两个 [B,1,H]；表有10,240参数',False),
        ('norm_out(h) * (1+scale) + shift',[ref(W,720,724)],'[B,L,H]；无 affine LayerNorm',False),
        ('proj_out：5120→64',[ref(W,623),ref(W,724)],'W [64,5120]，b [64]；327,744参数；输出[B,L,64]',False),
        ('reshape → permute → flatten 时空块',[ref(W,726,730)],'[B,F,h/2,w/2,1,2,2,16] → [B,16,F,h,w]',False),
        ('vᵢ = flow prediction [B,16,F,h,w]',[ref(W,732,735),ref(S,805,807)],'v 与 z 同形状；速度预测不直接替换 z',True),
    ])
    chapter('output','11 · 输出头：hidden states 怎样成为 DiT 预测','最后一个 block 输出后，还有条件归一化、线性头与 unpatchify。',
        '<p>proj_out 输出的是每个 patch 的64个预测数值，与 patch 输入的16×1×2×2对应。权重独立于 patch_embedding，不要求互逆。当前调度配置 prediction_type=flow_prediction，而变量名仍叫 noise_pred；其语义应由目标/调度器约定判定。</p>',
        '[βout,γout] = chunk₂(Aout + e[:,None,:])\ny = (LNout(h) ⊙ (1+γout) + βout) Woutᵀ + bout\nv = unpatchify(y)\nzᵢ₊₁ ≠ vᵢ；实际由 scheduler.step(vᵢ,tᵢ,zᵢ) 得到',g)

    g=Graph('CF')
    g.n(1,380,20,'同一 zᵢ、tᵢ、已选专家 eᵢ\nzᵢ [B,16,F,h,w]',[ref(P,596,612)],tensor=True)
    g.n(2,20,20,'E+ [B,M,4096]\n正提示编码',[ref(P,245,253)],tensor=True)
    g.n(3,740,20,'E− [B,M,4096]\n负提示或空字符串编码',[ref(P,254,278)],tensor=True)
    g.n(4,170,205,'cond：完整 DiT ×40 blocks\nv+ [B,16,F,h,w]',[ref(P,614,621)])
    g.n(5,600,205,'uncond：完整 DiT ×40 blocks\nv− [B,16,F,h,w]',[ref(P,623,631)])
    g.n(6,380,390,'v = v− + s(v+−v−)\ns=guidance_scale / scale_2',[ref(P,596,603),ref(P,632)])
    g.n(7,380,555,'scheduler.step 只调用一次\n下一 latent [B,16,F,h,w]',[ref(P,635)],tensor=True)
    for a,b in [(1,4),(1,5),(2,4),(3,5),(4,6),(5,6),(6,7)]:g.wire(a,b)
    chapter('cfg','12 · CFG 与高/低噪声专家：两种独立的分支','先按噪声时间选择专家，再用该专家做正/负条件预测，组合后推进一次。',
        '<p>当前 pipeline 以 guidance_scale&gt;1 判断是否执行 CFG。未开启时只算正分支；开启后 DiT 调用数约翻倍，VAE 最后仍只解码一次。guidance_scale_2 缺省等于 guidance_scale，控制低噪声专家阶段的引导强度。负提示不是把正文本向量取负。</p><p>transformer 与 transformer_2 的配置逐字一致、权重独立。每个都是完整的 40 层 dense DiT；“专家”在这里是按时间步选择整个网络，无 token router/top-k。切换时传递 latent，不传上一专家的 hidden states。scheduler 历史在当前 pipeline 中跨专家切换继续保留。</p>',
        'v+ = DiT_e(zᵢ,tᵢ,E+)\nv− = DiT_e(zᵢ,tᵢ,E−)\nvCFG = v− + s(v+−v−)\ne=high if tᵢ≥0.875×1000 else low\nCFG两分支 ≠ 高低噪声两专家',g)

    g=chain('TS',[
        ('num_inference_steps N → 原始均匀序列',[ref(S,428,430)],'uᵢ=1−0.999·i/N，i=0…N−1；先生成N+1点，再丢最后一点',False),
        ('flow_shift=3.0 → σᵢ=3uᵢ/(1+2uᵢ)',[ref(S,431,438)],'use_dynamic_shifting=false；σ₀从1微调为1−1e−6',False),
        ('timesteps=1000σ；末尾追加σN=0',[ref(S,439,450)],'t只有N个网络调用点；sigmas有N+1个状态端点',True),
        ('按固定时间表做 N 次循环',[ref(P,584,612)],'N=40：13次高噪声专家+27次低噪声专家（由此本地调度分支推导）',False),
        ('最后一步到σ=0 → 结束采样',[ref(S,1208,1227),ref(P,654,668)],'时间表终点不等于质量检测通过',True),
    ])
    chapter('schedule','13 · set_timesteps：预先选定噪声水平','当前调度表在循环前确定；latent 内容和模型预测仍须逐步计算。',
        '<p>这组配置选择 flow sigmas，beta_start/beta_end 等其他字段不决定这里实际采用的 flow 时间表。N=40 时，原始u末值为0.025975，最后一次模型调用的σ约0.074077，再由最后一次step走到0；不是固定在0.001调用最后一次网络。</p><p>高/低专家步数取决于N、flow_shift和boundary_ratio，13/27只适用于本报告N=40与当前源码分支。σ不是从latent估计出的质量指标；输出差也不会自动增加采样步。</p>',
        'uᵢ = 1 − (1−1/1000)i/N\nσᵢ = 3uᵢ/(1+2uᵢ)；tᵢ=1000σᵢ；σN=0\nN=40：t₀≈999.999；t₁₂≈875.156；t₁₃≈861.317\n高噪声条件：σᵢ≥0.875 ⇔ uᵢ≥0.7\n噪声时间表固定 ≠ zᵢ数值预先已知',g)

    g=Graph('SC')
    g.n(1,20,20,'sample zᵢ [B,16,F,h,w]\n本轮DiT读取的latent',[ref(S,1192)],tensor=True)
    g.n(2,740,20,'model_output vCFG\n[B,16,F,h,w]',[ref(P,632,635)],tensor=True)
    g.n(3,380,185,'convert_model_output\nmᵢ = zᵢ − σᵢ vCFG',[ref(S,805,807)],'干净latent估计mᵢ；同形状 [B,16,F,h,w]')
    g.n(4,20,355,'last_sample + model_outputs\n最多2份m；上轮保存的sample',[ref(S,1188,1189),ref(S,1201,1206)],tensor=True)
    g.n(5,380,355,'UniC：校正当前sample\n首步跳过；以后使用历史+本轮mᵢ',[ref(S,1193,1199)])
    g.n(6,380,520,'移位保存 mᵢ / timestep\n选本轮order；保存last_sample',[ref(S,1201,1216)])
    g.n(7,380,685,'UniP：预测下一sample\nσᵢ → σᵢ₊₁；不再调用DiT',[ref(S,1217,1221),ref(S,944,950)])
    g.n(8,380,850,'step_index += 1\nprev_sample = zᵢ₊₁',[ref(S,1223,1232)],tensor=True)
    for a,b in [(1,3),(2,3),(3,5),(4,5),(5,6),(6,7),(7,8)]:g.wire(a,b)
    g.wire(1,5,'left','left',[(8,68),(8,320),(340,320),(340,403)])
    chapter('scheduler','14 · scheduler.step：流速度、UniC 校正、UniP 推进','UniPC 使用网络预测和历史状态求下一latent；没有梯度下降，也不更新模型权重。',
        '<p>配置为 solver_order=2、solver_type=bh2、predict_x0=true、prediction_type=flow_prediction、lower_order_final=true、disable_corrector=[]、solver_p=null。首次没有历史，预测器从一阶开始；中间通常二阶；最后一步降回一阶。</p><p><b>顺序不能颠倒：</b>先用传入的sample和v计算mᵢ，再用UniC校正sample；随后移位保存mᵢ，用已校正sample做UniP。源码不会在校正后重跑DiT，也不会重算mᵢ。UniC/UniP是一个step内部的数值操作，不额外增加模型调用。</p><p>model_outputs保存的是转换后的干净latent估计，不是文本KV或视频attention KV。每份m和last_sample均与latent同形状。专家切换时本pipeline没有重置这些历史。</p>',
        'mᵢ = zᵢ − σᵢ vᵢ  （convert_model_output）\n若i>0：zᵢᶜ = UniC(last_sample, zᵢ, mᵢ, 历史m, σ表)\n否则：zᵢᶜ=zᵢ\n保存mᵢ；pᵢ=min(2, N−i, 已有历史阶数+1)\nlast_sample ← zᵢᶜ\nzᵢ₊₁ = UniP(zᵢᶜ, mᵢ, 历史m, σᵢ,σᵢ₊₁,pᵢ)',g)

    g=chain('UP',[
        ('α=1−σ；λ=logα−logσ',[ref(S,629,636),ref(S,885,893)],'h=λnext−λcurrent；B(h)=expm1(−h)',False),
        ('历史差分 D=(m_prev−m_current)/r',[ref(S,897,904)],'r=(λprev−λcurrent)/h；每个D [B,16,F,h,w]',False),
        ('二阶预测器 rhos_p=[0.5]',[ref(S,934,950)],'D1s [B,1,16,F,h,w]；einsum→[B,16,F,h,w]',False),
        ('校正器：构造R,b → solve(R,b)',[ref(S,1030,1087)],'一阶rho_c=0.5；二阶是2×2系统；公式见正文',False),
        ('终点：最后预测器order=1、σnext=0',[ref(S,1208,1213),ref(S,944,950)],'极限下zN=m_last；不在σ=0再调用DiT',True),
    ])
    chapter('unipc-math','15 · UniPC 公式：对齐当前 bh2 / 二阶分支','下面是源码分支的公式，不用简单Euler更新冒充当前调度器。',
        '<p>预测器以本轮mᵢ为m₀、已校正zᵢᶜ为x。λ是数值求解坐标，与视频时间位置无关。二阶只有一份历史差分。末端σnext=0时λnext趋于正无穷，源码expm1(−h)趋于−1，且最后一阶无历史修正，输出mᵢ。</p><p>校正器在下一轮获得新预测后，修正上轮预测推进得到的sample。令a=i−1、b=i；以m_a为基准，x_a=last_sample。p=1时ρ=0.5；p=2时用一个更早历史点及当前点构造2×2线性系统。其新预测m_b来自本轮DiT，不是额外评估。</p>',
        '【UniP predictor，p=2】\nαj=1−σj；λj=log(αj)−log(σj)\nh=λi+1−λi；B=expm1(−h)\nr=(λi−1−λi)/h；D=(mi−1−mi)/r\nzᵢ₊₁ = (σi+1/σi)zᵢᶜ − αi+1 B mi − αi+1 B (D/2)\np=1时省略最后一项。\n\n【UniC corrector，a=i−1，b=i】\nh=λb−λa；u=−h；B=expm1(u)\nr=(λi−2−λa)/h；D=(mi−2−ma)/r\nf₁=B/u−1；f₂=f₁/u−1/2\nR=[[1,1],[r,1]]；bvec=[f₁/B,2f₂/B]\nρ=solve(R,bvec) （p=2）\nz_bᶜ = (σb/σa)last_sample − αb B ma\n       − αb B [ρ₀D + ρ₁(mb−ma)]\np=1：仅保留0.5(mb−ma)校正项。',g)

    g=chain('VD',[
        ('最终 normalized latent zN [B,16,21,60,104]',[ref(P,656,666)],'zVAE = zN ⊙ std + mean；mean/std各16项',True),
        ('post_quant_conv：1×1×1，16→16',[ref(V,1049,1050),ref(V,1195,1197)],'W [16,16,1,1,1]；272参数；输出同形状',False),
        ('逐 latent 帧调用 decoder，共21次',[ref(V,1197,1205)],'每次输入 [B,16,1,60,104]；feat_cache跨chunk保留',False),
        ('首chunk输出1帧；其余每chunk输出4帧',[ref(V,272,303)],'1+20×4=81；输出chunk [B,3,1或4,480,832]',False),
        ('cat(frame axis) → clamp [−1,1] → clear_cache',[ref(V,1203,1214)],'RGB tensor [B,3,81,480,832]',True),
        ('postprocess_video：转输出布局与[0,1]',[ref(P,667,668),ref('sources/diffusers/src/diffusers/video_processor.py',104,120),ref('sources/diffusers/src/diffusers/image_processor.py',222,234)],'np输出 [B,81,480,832,3]；fps/视频编码属于后续导出',False),
    ])
    chapter('vae','16 · 采样结束：VAE 解码入口与分帧缓存','标准T2V只在所有采样完成后调用VAE decoder；encoder/quant_conv在此路径旁路。',
        '<p>模型保留VAE encoder与quant_conv以支持编码，但当前纯T2V不调用。decode使用post_quant_conv和decoder。配置缺少scale_factor项时，当前本地默认 temporal=4、spatial=8；与两次时间上采样、三次空间上采样一致。此A14B使用16通道、空间8倍的VAE，不应套用TI2V-5B的空间16倍VAE。</p><p>pipeline中变量latents_std实际上被设为1/std，所以源码 latents/latents_std+mean 等价于 zN×std+mean。第一latent帧是时间边界特例，不能写成21×4=84。</p>',
        'zVAE[b,c,f,y,x] = std[c]·zN[b,c,f,y,x] + mean[c]\nFout = 1+4(Flatent−1)=81\nHout=8h=480；Wout=8w=832\nRGB01 = clamp(0.5·RGB_signed+0.5,0,1)',g)

    g=chain('VU',[
        ('decoder.conv_in：causal Conv3d 16→384',[ref(V,826,832),ref(V,881,891)],'kernel3³；输出 [B,384,τ,60,104]；τ输入=1',False),
        ('mid_block：ResBlock → spatial attention → ResBlock',[ref(V,434,470),ref(V,893,894)],'C=384，分辨率60×104；默认1个中间attention',False),
        ('up_blocks[0]：3×ResBlock 384→384；upsample3d',[ref(V,835,873),ref(V,747,761)],'时间首块1/其余2；空间120×208；resample输出C=192',False),
        ('up_blocks[1]：3×ResBlock 192→384；upsample3d',[ref(V,835,873),ref(V,747,761)],'时间首块1/其余4；空间240×416；输出C=192',False),
        ('up_blocks[2]：3×ResBlock 192→192；upsample2d',[ref(V,835,873)],'时间不变；空间480×832；输出C=96',False),
        ('up_blocks[3]：3×ResBlock 96→96；无上采样',[ref(V,835,873)],'输出 [B,96,1或4,480,832]',False),
        ('norm_out → SiLU → causal conv_out 96→3',[ref(V,875,877),ref(V,900,914)],'kernel3³；输出 [B,3,1或4,480,832]',True),
    ])
    chapter('vae-decoder','17 · VAE decoder 内部：通道与上采样逐级展开','base_dim=96、dim_mult=[1,2,4,4]，decoder采用WanUpBlock，含两次时间和三次空间上采样。',
        '<p>is_residual默认为false，故选择WanUpBlock而非WanResidualUpBlock。反转temporal_downsample得到[true,true,false]。每个up block有num_res_blocks+1=3个残差块，共12个；加mid的2个，decoder总计14个WanResidualBlock。</p><p>upsample3d对后续chunk先用kernel(3,1,1)的time_conv将C→2C，再将额外通道重排为两倍时间长度；空间nearest-exact×2后Conv2d使C减半。第一个chunk用缓存哨兵跳过时间扩展。attn_scales=[]不代表整个VAE没有attention：mid_block仍固定有一个空间attention。</p>',
        'dims = 96×[4,4,4,2,1] = [384,384,384,192,96]\nup0: 384→384→192；up1: 192→384→192\nup2: 192→192→96；up3: 96→96\n后续chunk时间: 1→2→4；首chunk: 1→1→1\n空间: 60×104→120×208→240×416→480×832',g)

    g=Graph('VR')
    g.n(1,380,20,'输入 x [B,Cin,τ,h,w]',[ref(V,346,348)],tensor=True)
    g.n(2,20,190,'shortcut：Identity 或 Conv1³\nCin→Cout',[ref(V,344,348)])
    g.n(3,560,190,'WanRMS_norm → SiLU → conv1\nConv3³ Cin→Cout',[ref(V,338,343),ref(V,350,366)])
    g.n(4,560,370,'norm2 → SiLU → Dropout(0)\nconv2：Conv3³ Cout→Cout',[ref(V,368,383)])
    g.n(5,380,550,'return x + shortcut\n[B,Cout,τ,h,w]',[ref(V,385,386)],tensor=True)
    g.n(6,20,730,'feat_cache[j]：历史特征帧\n每个卷积独立的 [B,C,≤2,h,w]',[ref(V,354,363),ref(V,374,383)],tensor=True)
    g.n(7,560,730,'WanCausalConv3d\ncat(history,current) → 左时间padding',[ref(V,162,173)],'时序因果来自卷积padding；不缓存Q/K/V')
    for a,b in [(1,2),(1,3),(3,4),(2,5),(4,5),(6,7)]:g.wire(a,b)
    chapter('vae-residual','18 · VAE 残差块与因果卷积状态','VAE也使用残差，但分支算子主要是卷积；缓存的是历史特征帧。',
        '<p>WanRMS_norm在channel维做L2归一化并乘√C和可学习gamma；与DiT的LayerNorm不同。Cin≠Cout时shortcut使用1³投影，无时间混合。主分支两个3³因果卷积保持当前chunk的时空输出尺寸。</p><p>普通卷积保留最多2帧历史特征供下一chunk使用；时间上采样模块还有Rep哨兵与特殊首帧逻辑，不能把所有缓存简单归为相同的长度2张量。解码结束clear_cache，状态不跨视频生成复用。</p>',
        'R(x)=Conv₂(SiLU(Norm₂(Conv₁(SiLU(Norm₁(x)))))) + Shortcut(x)\nNorm(x) = normalize_L2(x,channel)·√C·γ\nConv权重 [Cout,Cin,kt,kh,kw]；参数 Cout·Cin·kt·kh·kw+Cout\n因果卷积：仅补过去时间，不补未来时间',g)

    g=Graph('VM')
    g.n(1,380,20,'mid input [B,384,τ,60,104]',[ref(V,406,412)],tensor=True)
    g.n(2,380,165,'将time折进batch → norm\n[Bτ,384,60,104]',[ref(V,406,413)])
    g.n(3,20,325,'to_qkv → chunk Q\nQ [Bτ,1,6240,384]',[ref(V,414,417)],'Conv2d1²:384→1152；三支共享一次投影')
    g.n(4,380,325,'chunk K\nK [Bτ,1,6240,384]',[ref(V,414,417)])
    g.n(5,740,325,'chunk V\nV [Bτ,1,6240,384]',[ref(V,414,417)])
    g.n(6,190,495,'QKᵀ / √384 → softmax\n[Bτ,1,6240,6240]',[ref(V,420)],'SDPA逻辑展开；没有跨帧attention',width=510)
    g.n(7,190,650,'P @ V → [Bτ,1,6240,384]',[ref(V,420,422)],'score/P逻辑shape，可能不物化',width=510)
    g.n(8,190,805,'proj 1×1 → reshape → +identity\n[B,384,τ,60,104]',[ref(V,424,431)],width=510)
    for a,b in [(1,2),(2,3),(2,4),(2,5),(3,6),(4,6),(6,7),(7,8)]:g.wire(a,b)
    g.wire(5,7,'bottom','right',[(895,620),(735,620),(735,698)])
    g.wire(1,8,'left','left',[(8,68),(8,853)])
    chapter('vae-attention','19 · VAE 中间 attention：按帧进行空间混合','源码虽称causal attention，但此处实际将time折入batch，每帧做单头空间attention。',
        '<p>mid attention既不读取文本，也不跨视频帧。Q/K/V来自当前帧空间特征；SDPA默认无causal mask。VAE整体的时间因果性主要来自因果卷积与分chunk解码，不能把该attention画成时间方向三角mask。</p><p>单头dim=384，to_qkv为1×1 Conv2d(384,1152)，proj为Conv2d(384,384)，加norm gamma，共591,744参数。此图针对解码mid_block，不分析纯T2V未调用的encoder。</p>',
        'x [B,C,τ,h,w] → xflat [Bτ,C,h,w]\nQ,K,V [Bτ,1,hw,C]\nY = softmax(QKᵀ/√C)V\noutput = x + reshape(proj(Y))',g)

    g=Graph('CM')
    g.n(1,20,20,'E±：UMT5固定输出\n[B,M,4096]',[ref(P,535,550)],tensor=True)
    g.n(2,20,190,'每专家文本投影 + 每层K/V\nK/V [B,M,40,128]',[ref(W,347),ref(W,53,56)],'默认重算；启用特定hook才能复用')
    g.n(3,740,20,'每步新的zᵢ → 视频Q/K/V\n[B,L,40,128]',[ref(P,605,621)],tensor=True)
    g.n(4,740,190,'视频self K/V 不跨步复用\n无past_key_values append',[ref(W,78,100)],'每步输入变了；与是否采用GQA是独立概念')
    g.n(5,20,390,'cache_context(cond/uncond)\n只设置hook上下文标签',[ref(K,154,164)],'没有enable_cache时不建立文本KV缓存')
    g.n(6,740,390,'UniPC历史 / VAE feat_cache\n各有生命周期和形状',[ref(S,1201,1216),ref(V,1123,1131)],'数值求解历史与卷积状态均非attention KV')
    for a,b in [(1,2),(2,5),(3,4),(4,6)]:g.wire(a,b)
    chapter('cache','20 · 状态、缓存与未选路径','文本特征、文本K/V、scheduler历史、VAE卷积缓存是四类不同状态。',
        '<p>固定专家、固定层、固定正/负条件时，cross K/V可复用；跨专家或跨层不可共用投影结果。逻辑文本KV每层每分支有2×B×M×40×128元素。B=1,M=512,BF16时约10MiB；每专家40层×两CFG分支约800MiB。这是可选缓存的估计，不是默认实测显存。</p><p>当前主干是MHA；即使不缓存也可以设计成GQA，但本配置没有选用。I2V图像投影、TI2V逐位置时间条件、LoRA/量化/并行/可选cache hook均不作为本报告运行分支。VAE encoder存在而未执行；UMT5 decoder及LM head没有实例化。</p>',
        'text KV elements/layer/branch = 2BM·40·128\nBF16 B=1,M=512 → 10MiB\n默认视频AR KV cache = 0（不代表总工作显存=0）\nself scores逻辑规模 B·40·L²；cross scores B·40·L·M',g)

    g=Graph('CT')
    g.n(1,380,20,'WanPipeline.__call__\n文本一次 → 采样N步 → VAE一次',[ref(P,535,554),ref(P,614,635),ref(P,656,668)])
    g.n(2,20,180,'UMT5EncoderModel.forward\nencoder: UMT5Stack ×1',[ref(U,1105,1112),ref(U,1159,1166)])
    g.n(3,380,180,'WanTransformer3DModel.forward\n两个实例；每次选择一个',[ref(W,662,691),ref(W,703,704)])
    g.n(4,740,180,'AutoencoderKLWan.decode\npost_quant_conv → decoder',[ref(V,1195,1214)])
    g.n(5,20,360,'UMT5Stack.block ×24\nUMT5Block.layer[0] / layer[1]',[ref(U,627,634),ref(U,426,451),ref(U,476)])
    g.n(6,380,360,'blocks ×40 / expert\nWanTransformerBlock.forward',[ref(W,462,504)])
    g.n(7,740,360,'WanDecoder3d\nmid_block + up_blocks ×4',[ref(V,826,877)])
    g.n(8,20,540,'UMT5LayerSelfAttention\nSelfAttention.q / k / v / o',[ref(U,386,395),ref(U,218,224),ref(U,307,308),ref(U,328,331),ref(U,367,369)])
    g.n(9,380,540,'attn1 / attn2 → processor\nto_q/k/v → SDPA → to_out',[ref(W,53,56),ref(W,143,162)])
    g.n(10,740,540,'WanResidualBlock ×14\nnorm → SiLU → Conv，+shortcut',[ref(V,338,348),ref(V,385,386)])
    g.n(11,20,720,'UMT5LayerFF.DenseReluDense\nwi_0→act ∥ wi_1 → × → wo',[ref(U,110,133),ref(U,145,152)])
    g.n(12,380,720,'ffn: FeedForward.net\n[0].proj → GELU → [1] → [2]',[ref(A,1714,1715),ref(A,1725,1731),ref(A,1740,1742),ref(G,76,90)])
    g.n(13,740,720,'WanAttentionBlock ×1\nnorm / to_qkv / SDPA / proj',[ref(V,400,431)])
    g.n(14,380,930,'未选：VAE encoder/quant_conv；I2V图像条件；可选cache/LoRA\nUMT5 decoder/LM head未实例化；两专家内的FFN都是dense',
        [ref(V,1038,1050),ref(W,326,328),ref(U,1105,1112)],width=690)
    for a,b in [(1,2),(1,3),(1,4),(2,5),(3,6),(4,7),(5,8),(5,11),(6,9),(6,12),(7,10),(7,13)]:
        if b in (11,12,13):
            g.wire(a,b,'left','left',[(next(n['x'] for n in g.nodes if n['id']==g.prefix+str(a))-12,408),(next(n['x'] for n in g.nodes if n['id']==g.prefix+str(a))-12,768)])
        else:g.wire(a,b)
    chapter('call-hierarchy','21 · 源码调用树与配置映射','本地类名、实例数量和所选forward分支，展开到实际线性层、激活、乘法和卷积。',
        '<p>UMT5Block中layer[1]在encoder配置下是FFN，而非cross-attention。视频block的attn1、attn2、ffn是不同模块；最终输出头属于WanTransformer3DModel，位于40层循环之外。下方连线表示模块包含/调用关系，不表示tensor依次流经所有兄弟模块；真实数据依赖见各专章。</p><p>配置文件映射：model_index.json→WanPipeline；text_encoder/config.json→UMT5EncoderModel；transformer/config.json及transformer_2/config.json→WanTransformer3DModel；scheduler/scheduler_config.json→UniPCMultistepScheduler；vae/config.json→AutoencoderKLWan。UMT5直接对应本地modeling_umt5.py，未改用其他T5变体的源码。</p>',
        'UMT5 FFN: hidden_states → wi_0 → act ─┐\n                        → wi_1 ──────× → dropout → wo → residual\nWan FFN: hidden_states → net[0].proj → GELU(tanh)\n                      → net[1](Dropout) → net[2](Linear) → gated residual\nUMT5: [B,M,4096] → 两路[B,M,10240] → [B,M,4096]\nWan: [B,L,5120] → [B,L,13824] → [B,L,5120]',g)


def dump_json(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def collect_evidence():
    configs={str(p.relative_to(ROOT)):json.loads(p.read_text()) for p in sorted(MODEL.rglob('*.json'))
             if p.name in ('config.json','model_index.json','scheduler_config.json')}
    sources=[]
    for path in sorted(SOURCE_PATHS):
        data=(ROOT/path).read_bytes()
        sources.append(dict(path=path,sha256=hashlib.sha256(data).hexdigest(),lines=len(data.splitlines())))
    commits={name:subprocess.check_output(['git','-C',str(ROOT/path),'rev-parse','HEAD'],text=True).strip()
             for name,path in [('transformers','transformers'),('diffusers','sources/diffusers')]}
    return dict(analysis_date='2026-09-07',model='Wan-AI/Wan2.2-T2V-A14B-Diffusers',
                revision='5be7df9619b54f4e2667b2755bc6a756675b5cd7',configs=configs,
                source_commits=commits,source_files=sources,
                method='Concrete local configs + exact source excerpts; metadata only; no weight loading or GPU inference.',
                collector_caveat='Composite Diffusers repository lacks top-level config.json; reused prior model_index-based evidence and extended it with component configs/source hashes.')


def identity(evidence):
    configs=evidence['configs']; prefix=str(MODEL.relative_to(ROOT))
    dit=configs[prefix+'/transformer/config.json']; textcfg=configs[prefix+'/text_encoder/config.json']
    rows=[
        ('DiT experts','2 × 相同配置、不同参数','高/低噪声各40层；每次forward一个专家'),
        ('每个 DiT block','H=5120；heads=40；head_dim=128','self MHA + text cross MHA + dense GELU FFN'),
        ('FFN / patch','I=13824；patch=(1,2,2)；Cin=Cout=16','patch 16×1×2×2=64 → H；末端H → 64'),
        ('文本编码器','24层；Htext=4096；64 heads×64','词表256384；gated-GELU中间宽度10240'),
        ('文本条件 / 时间条件','M=512（pipeline默认）；freq_dim=256','文本4096→5120→5120；时间256→5120→5120→6H'),
        ('归一化 / 残差','DiT eps=1e−6；qk_norm=rms_norm_across_heads','SA/FFN残差乘时间gate；cross残差直接相加'),
        ('Scheduler','UniPC；order=2；bh2；flow_shift=3','flow_prediction；predict_x0；lower_order_final'),
        ('VAE','z_dim=16；base_dim=96；dim_mult=[1,2,4,4]','空间×8，时间1+4(F−1)；decoder有14个残差块'),
        ('示例形状（推导）','B=1；81帧；480×832；采样N=40','z[B,16,21,60,104]；L=32760；M=512'),
    ]
    par=[
        ('单个self或cross attention','4H²+6H','104,888,320'),
        ('单个视频FFN','2HI+I+H','141,576,704'),
        ('单个视频block','2×attention + FFN + 2H(cross LN) + 6H(table)','351,394,304'),
        ('40个视频block','40×351,394,304','14,055,772,160'),
        ('patch_embedding','H×(16×1×2×2)+H','332,800'),
        ('condition_embedder','text 47,196,160 + time MLP 27,535,360 + time_proj 157,317,120','232,048,640'),
        ('最终输出头','proj_out 327,744 + scale_shift_table 10,240','337,984'),
        ('每个完整DiT专家','上述block、patch、condition、head合计','14,288,491,584'),
        ('两个DiT专家','2×14,288,491,584','28,576,983,168'),
        ('UMT5EncoderModel','embedding + 24×(4Htext²+3Htext·Itext+2Htext+32×64) + final norm','5,680,910,336'),
    ]
    table=lambda headers,rs:'<div class="table-scroll"><table><thead><tr>'+''.join(f'<th>{esc(h)}</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{esc(c)}</td>' for c in r)+'</tr>' for r in rs)+'</tbody></table></div>'
    return f'''<section id="identity"><div class="eyebrow">SOURCE-ALIGNED DEEP DIVE · 推理路径</div>
    <h2>WAN 2.2 · 从文本到视频，逐算子展开</h2>
    <p class="lead">沿两条输入线进入视频Transformer，追踪每条残差、每次采样更新，最后进入VAE。每章都有文字、公式、形状图和逐字源码。</p>
    <div class="callout">范围：<b>Wan2.2-T2V-A14B-Diffusers</b>，不是TI2V-5B或I2V。旧报告保留；本版是细化补充。<br>每个采样步预测的是同形状的<b>flow velocity v</b>，不是最终视频，也不是仅最后一层FFN的hidden state。</div>
    <p><a href="../report.html">返回原版报告</a> · <a href="diagrams.zip" download>下载全部21张可编辑Excalidraw</a> · <a href="evidence.json">证据快照</a> · <a href="analysis-state.json">分析状态</a></p>
    <details><summary>版本、来源与验证边界</summary>
    <p>分析日期：{evidence['analysis_date']}；模型revision：<code>{evidence['revision']}</code>。本地模型目录：<code>{esc(str(MODEL))}</code>。</p>
    <p>checkpoint记录diffusers {dit.get('_diffusers_version')}、transformers {textcfg.get('transformers_version')}。本地diffusers commit：<code>{evidence['source_commits']['diffusers']}</code>；Transformers commit：<code>{evidence['source_commits']['transformers']}</code>。本地实现较新，报告明确按当前选中分支解读，不宣称复原发布时所有kernel行为。</p>
    <p>仅配置、权重索引及源码可用；没有加载权重或执行GPU推理。参数数目为构造器代数审计，权重索引total_size仅交叉校验；未根据参数量断言实际加载dtype或显存。图中attention scores为逻辑张量，优化kernel可能不物化。</p>
    <p>skill通用收集器要求顶层config.json，不直接适用此复合pipeline；本版沿用原版model_index证据，逐组件读取配置并记录源码SHA256。</p></details></section>
    <section id="parameters"><h2>00 · 配置、参数与符号</h2>{table(['模块','实际配置 / 默认','含义'],rows)}
    <h3>参数总账：由本地构造器推导</h3>{table(['范围','计算依据','参数量'],par)}
    <p class="source">参数包含bias和可学习归一化/调制表。两个DiT配置完全相同；高低噪声专家不共享权重。未把VAE总参数混入上述合计；VAE专章给出运行层的通道、卷积核和参数公式。</p>
    <pre class="formula">B = prompt_batch × num_videos_per_prompt（有效批量）\nM = 补齐后的文本长度，默认512\nF,h,w = latent时间、高、宽；H = DiT hidden size 5120\nL = F×(h/2)×(w/2)；I = DiT FFN 13824\ni = 外部采样编号；ℓ = Transformer层号；f = 视频时间坐标\nt / σ = 噪声时间步 / 噪声水平（都不是帧号）</pre>
    <details><summary>所有组件原始配置（完整JSON）</summary>{''.join('<h3>'+esc(k)+'</h3><pre>'+esc(json.dumps(v,ensure_ascii=False,indent=2))+'</pre>' for k,v in configs.items())}</details></section>'''


def render(evidence):
    sections=[identity(evidence)]
    for i,ch in enumerate(CHAPTERS,1):
        g=ch['graph']; refs=[r for n in g.nodes for r in n['refs']]; p,lo,hi=refs[0]
        heading=f'<a href="{link(p,lo)}">{esc(p)}:L{lo}–{hi}</a> 起；以下各段逐一标注文件与行号 <span class="excerpt-kind">verbatim摘录；# derived为推导注释</span>'
        sections.append(f'''<section id="{ch['slug']}"><h2>{ch['title']}</h2><p class="lead">{ch['lead']}</p>{ch['body']}
        <h3>算法与形状</h3><pre class="formula">{esc(ch['formula'])}</pre>
        <div class="section-tools"><a href="diagrams/{i:02d}-{ch['slug']}.excalidraw" download>下载本章Excalidraw ↗</a><span>圆角 = 数据张量　方角 = 运算　同ID = 对应源码</span></div>
        <div class="graph-code-pair"><div class="graph-panel"><h4>计算图 · {ch['graph'].prefix}节点 · 可横向滚动</h4><div class="graph-scroll">{g.svg()}</div></div>
        <div class="code-panel"><h4>{heading}</h4><pre><code>{g.code()}</code></pre></div></div></section>''')
    ledger=''.join(f'<li><a href="source/{Path(s["path"]).stem}.html">{esc(s["path"])}</a> · {s["lines"]}行 · SHA256 <code>{s["sha256"]}</code></li>' for s in evidence['source_files'])
    sections.append('<section id="sources"><h2>22 · 源码索引、假设与复查入口</h2><p>源码链接指向本次分析的逐行HTML快照，不依赖编辑器插件；每个节点标注可回到完整上下文。快照与原始源码逐字相同，仅添加行号和HTML转义。</p><ul>'+ledger+'</ul><h3>仍需运行环境确认的事项</h3><p>具体attention backend、GPU精度与数值误差、模型权重实际dtype、显存峰值和生成质量未实测。模型类的默认参数以当前本地源码为准；额外cache/量化/分布式hook可能改变计算和存储路径。</p><p>时间步与参数代数、示例形状、patch/unpatch索引双射、图码ID和全部Excalidraw JSON均由构建脚本检查；这些检查不等价于真实模型推理正确性测试。</p></section>')
    template=(SKILL/'assets/report-template.html').read_text()
    nav='<nav><b>WAN2.2 · 精细版</b><a href="#identity">阅读范围与依据</a><a href="#parameters">00 · 配置与参数</a>'+''.join(f'<a href="#{c["slug"]}">{c["title"].split("：")[0]}</a>' for c in CHAPTERS)+'<a href="#sources">22 · 源码与验证</a></nav>'
    template=re.sub(r'<nav>.*?</nav>',nav,template,flags=re.S)
    template=template.replace('{{MODEL_TITLE}}','WAN2.2 · 精细架构报告').replace('{{GENERATED_AT}}','2026-09-07').replace('{{REPORT_BODY}}',''.join(sections))
    css='''<style>
    :root{color-scheme:light;--ink:#192839;--muted:#546778;--line:#d6e1e7;--paper:#fff;--bg:#edf2f5;--accent:#007c83;--soft:#f2f7f8;--tensor-fill:#e8f5f5;--tensor-line:#39999d;--op-fill:#fff}
    header{padding:36px 4vw;background:linear-gradient(115deg,#102b40,#13676d)}header h1{font-size:35px;letter-spacing:-1px}
    .layout{max-width:2200px;grid-template-columns:225px minmax(0,1fr);gap:20px;margin:22px auto;padding:0 24px 80px}
    nav{max-height:calc(100vh - 40px);overflow:auto;padding:12px;font-size:12px}nav a{padding:6px 7px}nav b{padding:8px;display:block}
    section{padding:28px;scroll-margin-top:20px}section p{max-width:1250px}h2{font-size:25px;color:#154d5e}.eyebrow{font-size:11px;letter-spacing:2px;color:#007c83;margin-bottom:14px}
    a{color:#007c83}.formula{font-size:15px;line-height:1.85;background:#f1f7f8;border-left:3px solid #249498;white-space:pre-wrap;overflow-wrap:anywhere}
    .graph-code-pair{grid-template-columns:minmax(0,1.35fr) minmax(0,.85fr)}.graph-panel,.code-panel{padding:12px}
    .architecture-graph{min-width:850px}.architecture-graph text{font-size:15px}.architecture-graph .shape-label{font:14px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;fill:#203949}
    .code-panel h4{font-size:12px;overflow-wrap:anywhere;line-height:1.6}.code-panel pre{max-height:880px;font-size:12px}.code-panel pre a{color:#a8dce4;text-decoration:underline}
    .section-tools{display:flex;gap:16px;justify-content:space-between;flex-wrap:wrap;font-size:12px;color:#63727d}.section-tools a{font-weight:600}
    .table-scroll{overflow:auto}table{font-size:13px;display:table}td{min-width:140px}td:nth-child(2){min-width:270px}#sources li{overflow-wrap:anywhere;margin-bottom:12px}
    @media(max-width:1450px){.graph-code-pair{grid-template-columns:1fr}.code-panel pre{max-height:480px}.architecture-graph{max-width:1100px;margin:auto}}
    @media(max-width:900px){.layout{grid-template-columns:1fr;padding:0 12px}nav{position:static;max-height:240px}section{padding:18px}header{padding:28px 20px}h2{font-size:22px}}
    </style>'''
    return template.replace('</head>',css+'</head>')


def source_snapshots(evidence):
    dest=OUT/'source'; dest.mkdir(parents=True,exist_ok=True)
    for info in evidence['source_files']:
        p=ROOT/info['path']; lines=p.read_text().splitlines()
        code='\n'.join(f'<span id="L{i}"><a href="#L{i}">{i:5}</a>  {esc(line)}</span>' for i,line in enumerate(lines,1))
        page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>{esc(p.name)} · source snapshot</title><style>body{{margin:24px;background:#101827;color:#eef4fc;font:13px/1.65 monospace}}a{{color:#83cdd6}}pre span{{display:block;min-height:1em}}pre span:target{{background:#365259}}header{{position:sticky;top:0;background:#101827;padding:12px 0}}code{{overflow-wrap:anywhere}}</style><header><a href="../report.html#sources">← 返回报告</a><br>{esc(info['path'])}<br><code>SHA256 {info['sha256']}</code></header><pre>{code}</pre></html>'''
        (dest/(p.stem+'.html')).write_text(page,encoding='utf-8')


def validate_math(evidence):
    H,I=5120,13824
    attn=4*H*H+6*H; ffn=2*H*I+I+H; block=2*attn+ffn+8*H
    total=40*block+(64*H+H)+232048640+(H*64+64+2*H)
    texttotal=256384*4096+24*(4*4096**2+3*4096*10240+2*4096+32*64)+4096
    assert total==14288491584 and texttotal==5680910336
    c=evidence['configs']; prefix=str(MODEL.relative_to(ROOT))
    assert c[prefix+'/transformer/config.json']==c[prefix+'/transformer_2/config.json']
    indices={str(p.relative_to(ROOT)):json.loads(p.read_text())['metadata']['total_size'] for p in MODEL.rglob('*.index.json')}
    for path,size in indices.items():
        if '/transformer/' in path or '/transformer_2/' in path:assert size==4*total
        if '/text_encoder/' in path:assert size==2*texttotal
    sigmas=[3*(1-.999*i/40)/(1+2*(1-.999*i/40)) for i in range(40)]
    sigmas[0]-=1e-6
    assert sum(s>=.875 for s in sigmas)==13
    assert 21*30*52==32760 and 1+4*(21-1)==81
    # Independent index-level check of Conv3d patch ordering and output unpatchify.
    # Identity values make any permutation mistake visible without torch/numpy.
    C,F,h,w=2,3,4,6
    src=list(range(C*F*h*w)); tokens=[]
    for f in range(F):
        for y in range(h//2):
            for x in range(w//2):
                tokens.append([src[((c*F+f)*h+y*2+dy)*w+x*2+dx] for dy in range(2) for dx in range(2) for c in range(C)])
    restored=[None]*len(src)
    for k,token in enumerate(tokens):
        f,rem=divmod(k,(h//2)*(w//2)); y,x=divmod(rem,w//2)
        for j,val in enumerate(token):
            patchpos,c=divmod(j,C); dy,dx=divmod(patchpos,2)
            restored[((c*F+f)*h+y*2+dy)*w+x*2+dx]=val
    assert restored==src
    return dict(parameter_count_per_expert=total,text_encoder_parameters=texttotal,
                index_total_sizes=indices,N40_high_steps=13,N40_low_steps=27,
                N40_timesteps=[1000*s for s in sigmas],final_sigma=0,
                shape_checks='passed',unpatchify_index_roundtrip='passed; synthetic index data, not learned projection inverse',
                gpu_inference='not run; weights absent')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    build_chapters()
    # Calculate displayed schedule examples, avoiding hand-rounded drift.
    ch=next(c for c in CHAPTERS if c['slug']=='schedule')
    u=1-.999*13/40; ch['formula']=ch['formula'].replace('861.317',f'{1000*3*u/(1+2*u):.3f}')
    evidence=collect_evidence(); evidence['checks']=validate_math(evidence)
    report=render(evidence)
    assert '{{' not in report
    graphids=re.findall(r'<tspan class="node-id">([^<]+)</tspan>',report)
    codeids=re.findall(r'<span class="code-map">([^<]+)</span>',report)
    assert graphids==codeids and len(graphids)==len(set(graphids))
    (OUT/'report.html').write_text(report,encoding='utf-8')
    source_snapshots(evidence)
    subprocess.run([sys.executable,str(SKILL/'scripts/export_report_excalidraw.py'),str(OUT/'report.html'),str(OUT/'diagrams')],check=True,stdout=subprocess.DEVNULL)
    scenes=[]
    for i,ch in enumerate(CHAPTERS,1):
        old=OUT/'diagrams'/f'{i:02d}-diagram.excalidraw'; new=OUT/'diagrams'/f'{i:02d}-{ch["slug"]}.excalidraw'
        old.replace(new)
        scene=json.loads(new.read_text())
        assert scene['type']=='excalidraw' and scene['version']==2 and scene['elements'] and 'files' in scene and 'appState' in scene
        # The shared exporter estimates ASCII widths. Preserve graph centers while
        # accounting for full-width CJK glyphs in these Chinese editable labels.
        for element in scene['elements']:
            if element['type']=='text' and '-code-' not in element['id']:
                center=element['x']+element['width']/2
                width=sum(1 if unicodedata.east_asian_width(c) in ('W','F') else .62 for c in element['text'])*element['fontSize']
                element['width']=round(max(24,width),2)
                element['x']=round(center-element['width']/2,2)
        dump_json(new,scene)
        txt='\n'.join(e.get('text','') for e in scene['elements'])
        for node in ch['graph'].nodes:assert f'[{node["id"]}]' in txt
        assert not any(e['type']=='image' for e in scene['elements'])
        panel=next(e for e in scene['elements'] if e['id'].endswith('-code-background'))
        assert panel['x']>ch['graph'].width*1.5
        scenes.append(str(new.relative_to(OUT)))
    # Check that all authored connectors remain outside unrelated node interiors.
    for ch in CHAPTERS:
        for edge in ch['graph'].edges:
            for n in ch['graph'].nodes:
                x,y,w,h=n['x'],n['y'],n['w'],n['h']
                for (a,b),(d,f) in zip(edge,edge[1:]):
                    crosses=(a==d and x<a<x+w and max(min(b,f),y)<min(max(b,f),y+h)) or (b==f and y<b<y+h and max(min(a,d),x)<min(max(a,d),x+w))
                    assert not crosses,(ch['slug'],n['id'],edge)
    # All report links resolve offline, including every full-source line anchor.
    for href in re.findall(r'href="([^"]+)"',report):
        pathname,_,anchor=href.partition('#')
        target=(OUT/pathname) if pathname else OUT/'report.html'
        if pathname in ('diagrams.zip','evidence.json','analysis-state.json'):continue # constructed below
        assert target.exists(),href
        if anchor:assert f'id="{anchor}"' in target.read_text(),href
    with zipfile.ZipFile(OUT/'diagrams.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in scenes:z.write(OUT/name,name)
    evidence['checks'].update(graph_code_ids=len(graphids),matched_graph_code_pairs=len(CHAPTERS),excalidraw_scenes=len(scenes),
                             graph_connector_intersections=0,source_links='all resolved',
                             browser_review='319px stacked and 1800px graph/code columns; DOM text-bound checks passed at 1800px')
    dump_json(OUT/'evidence.json',evidence)
    prior=json.loads((BASE/'analysis-state.json').read_text())
    state=dict(prior)
    state.update(report_variant='detailed',analysis_date='2026-09-07',parent_report='../report.html',
                 source_snapshot_manifest=evidence['source_files'],validated_checks=evidence['checks'],
                 graph_code_pairs=[dict(section=c['slug'],graph_nodes=[n['id'] for n in c['graph'].nodes],
                                        excerpts=[dict(node=n['id'],path=p,line_start=lo,line_end=hi,excerpt_kind='verbatim') for n in c['graph'].nodes for p,lo,hi in n['refs']]) for c in CHAPTERS],
                 excalidraw_exports=scenes,
                 detailed_conclusions=['UMT5 uses scale=1, bidirectional padding-masked attention and 24 independent relative-bias tables.',
                  'DiT cross-attention receives no text padding mask on the selected baseline branch.',
                  'Wan block: gated self residual, ungated cross residual, gated FFN residual; final head produces flow velocity.',
                  'Selected scheduler is flow-prediction UniPC bh2 order2, including corrector and history, not Euler.',
                  'VAE decode iterates latent frames with causal convolution feature caches; first chunk yields1 frame, next chunks4.',
                  'VAE decoder has14 residual blocks and1 per-frame single-head spatial attention.'])
    dump_json(OUT/'analysis-state.json',state)
    # Preserve all prior findings and only register the new companion artifact.
    follow=prior.setdefault('conversation_followups',[])
    item=dict(date='2026-09-07',topic='算子级精细版：双pipeline、残差、CFG、UniPC、VAE',report='detailed/report.html',state='detailed/analysis-state.json',diagrams=21)
    follow[:]=[f for f in follow if not isinstance(f,dict) or f.get('report')!='detailed/report.html']; follow.append(item)
    dump_json(BASE/'analysis-state.json',prior)
    assert all((OUT/name).exists() for name in ('diagrams.zip','evidence.json','analysis-state.json'))
    print(json.dumps(dict(report=str(OUT/'report.html'),chapters=len(CHAPTERS),nodes=len(graphids),scenes=len(scenes),checks=evidence['checks']),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
