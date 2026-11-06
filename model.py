"""手写 decoder-only GPT（字符级）：所有结构均为手写张量实现。

对应论文：Vaswani et al. 2017 "Attention Is All You Need" 与 Radford et al. 2019 "GPT-2"。
本文件刻意不使用 nn.Transformer / nn.MultiheadAttention / nn.Linear 之外的任何封装层，
其中 Linear 只作为纯线性映射（y = xW^T + b）使用；注意力、归一化、位置编码、掩码、
残差连接、参数初始化全部手写。

各组件与论文小节对应关系：
- TokenEmbedding / AbsolutePositionalEmbedding / SinusoidalPE : GPT-2 §2.1、Vaswani §3.5（正弦编码）
- RotaryEmbedding (RoPE)        : Su et al. 2021 "RoFormer" §3.2.2
- CausalSelfAttention           : Vaswani §3.2.1（缩放点积注意力）+ §3.2.2（多头）
- FeedForward                   : Vaswani §3.3（FFN，比例为 4x，GPT-2 用 GELU）
- LayerNorm                     : Ba et al. 2016；Pre-LN 变体见 GPT-2 §2.3 / Xiong et al. 2020
- Block（残差 + 归一化位置）     : Vaswani §3.1（残差）；Pre-LN vs Post-LN 见 Xiong et al. 2020 §3
- GPT.lm_head + 权重共享         : Vaswani §3.4 / GPT-2 §2.1（tied embedding）
- GPT.generate + KV cache       : 推理加速，非论文内容；见 Shazeer 2019 "Fast Transformer Decoding"
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    """模型超参数。所有消融都由这里的字段切换，保证“只变一个因素”。"""

    vocab_size: int = 5000
    block_size: int = 192          # 上下文长度
    n_layer: int = 4
    n_head: int = 6
    n_embd: int = 192
    dropout: float = 0.1
    pos_emb: str = "learned"       # learned | sinusoidal | rope
    norm_placement: str = "pre"    # pre | post
    norm_type: str = "layernorm"   # layernorm | rmsnorm（Zhang & Sennrich 2019）
    bias: bool = True              # Linear/LayerNorm 是否带偏置（GPT-2 带）

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0, "n_embd 必须能被 n_head 整除"
        assert self.pos_emb in ("learned", "sinusoidal", "rope")
        assert self.norm_placement in ("pre", "post")
        assert self.norm_type in ("layernorm", "rmsnorm")

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head


class LayerNorm(nn.Module):
    """手写 LayerNorm（Ba et al. 2016）：对最后一维做零均值单位方差归一化。

    与 nn.LayerNorm 的区别仅在于这里显式写了 mean/var，便于展示实现细节。
    """

    def __init__(self, dim: int, eps: float = 1e-5, bias: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        xhat = (x - mu) / torch.sqrt(var + self.eps)
        out = xhat * self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


class RMSNorm(nn.Module):
    """手写 RMSNorm（Zhang & Sennrich 2019, "Root Mean Square Layer Normalization" §3）。

    与 LayerNorm 的唯一差别：不减均值、不加 bias，只按均方根缩放。
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class SinusoidalPE(nn.Module):
    """Vaswani §3.5 的正弦位置编码（不可学习，作为绝对位置编码的对照）。"""

    def __init__(self, block_size: int, n_embd: int):
        super().__init__()
        pe = torch.zeros(block_size, n_embd)
        pos = torch.arange(block_size, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, n_embd, 2, dtype=torch.float32) * (-math.log(10000.0) / n_embd))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, pos_offset: int, seq_len: int) -> torch.Tensor:
        return self.pe[pos_offset : pos_offset + seq_len]


class RotaryEmbedding(nn.Module):
    """RoPE（Su et al. 2021）：把绝对位置写成 q/k 的旋转，使注意力分数只依赖相对距离。

    采用 GPT-NeoX 的“前后半轴”旋转约定：
        [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
    旋转角 theta_i = pos * base^(-2i/d)，base=10000。
    """

    def __init__(self, head_dim: int, block_size: int, base: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE 要求 head_dim 为偶数"
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        pos = torch.arange(block_size, dtype=torch.float32)
        freqs = torch.outer(pos, inv_freq)                 # (T, d/2)
        self.register_buffer("cos", torch.cos(freqs), persistent=False)
        self.register_buffer("sin", torch.sin(freqs), persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, pos_offset: int):
        """q,k: (B, H, T, D) -> 旋转后的 q,k"""
        t = q.size(-2)
        cos = self.cos[pos_offset : pos_offset + t].to(q.dtype)  # (T, D/2)
        sin = self.sin[pos_offset : pos_offset + t].to(q.dtype)
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]

        def rotate(x):
            x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
            return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

        return rotate(q), rotate(k)


class CausalSelfAttention(nn.Module):
    """Vaswani §3.2：多头缩放点积自注意力 + 因果掩码。

    手写部分：
      1) QKV 线性投影（每个 head 独立切分）
      2) 注意力分数 qk^T / sqrt(d_head)，加因果 mask
      3) softmax 后加权 V
      4) 输出投影
    KV cache：推理时把历史 K/V 拼接进当前步，避免重复计算（见 forward 的 kv_cache 参数）。
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.head_dim
        self.dropout = cfg.dropout
        d = cfg.n_embd
        self.c_attn = nn.Linear(d, 3 * d, bias=cfg.bias)   # 一次算 q,k,v
        self.c_proj = nn.Linear(d, d, bias=cfg.bias)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, kv_cache=None, pos_offset=0, rope: RotaryEmbedding | None = None):
        B, T, C = x.shape
        H, D = self.n_head, self.head_dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, H, D).transpose(1, 2)             # (B,H,T,D)
        k = k.view(B, T, H, D).transpose(1, 2)
        v = v.view(B, T, H, D).transpose(1, 2)

        if rope is not None:
            q, k = rope(q, k, pos_offset)

        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)              # 拼接历史 K/V
            v = torch.cat([past_v, v], dim=2)
            new_cache = (k, v)
        else:
            new_cache = (k, v) if T > 0 else None

        Tk = k.size(2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(D)     # (B,H,T,Tk)

        # 因果掩码：第 i 个 query 只能看到 [0, offset+i] 的 key
        if T > 1:
            offset = Tk - T
            idx = torch.arange(T, device=x.device).unsqueeze(1) + offset
            key_idx = torch.arange(Tk, device=x.device).unsqueeze(0)
            att = att.masked_fill(key_idx > idx, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v                                        # (B,H,T,D)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y, new_cache


class FeedForward(nn.Module):
    """Vaswani §3.3：两层 MLP，隐层 4d，GELU 激活（GPT-2 的约定）。"""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x), approximate="tanh")))


class Block(nn.Module):
    """Vaswani §3.1：残差连接。归一化位置由 cfg.norm_placement 决定。

    pre : x = x + attn(LN(x));  x = x + ffn(LN(x))     （GPT-2 默认，深层更稳）
    post: x = LN(x + attn(x));  x = LN(x + ffn(x))     （原始 Transformer）
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        make_norm = (lambda: RMSNorm(cfg.n_embd)) if cfg.norm_type == "rmsnorm" \
            else (lambda: LayerNorm(cfg.n_embd, bias=cfg.bias))
        self.ln1 = make_norm()
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = make_norm()
        self.ffn = FeedForward(cfg)
        self.pre = cfg.norm_placement == "pre"

    def forward(self, x, kv_cache=None, pos_offset=0, rope=None):
        if self.pre:
            h, cache = self.attn(self.ln1(x), kv_cache, pos_offset, rope)
            x = x + h
            x = x + self.ffn(self.ln2(x))
        else:
            h, cache = self.attn(x, kv_cache, pos_offset, rope)
            x = self.ln1(x + h)
            x = self.ln2(x + self.ffn(x))
        return x, cache


class GPT(nn.Module):
    """完整 decoder-only GPT。词表 embedding 与输出头共享权重（GPT-2 §2.1）。"""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.learned_pe = nn.Parameter(torch.zeros(cfg.block_size, cfg.n_embd)) if cfg.pos_emb == "learned" else None
        self.sinusoidal_pe = SinusoidalPE(cfg.block_size, cfg.n_embd) if cfg.pos_emb == "sinusoidal" else None
        self.rope = RotaryEmbedding(cfg.head_dim, cfg.block_size) if cfg.pos_emb == "rope" else None
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.bias:
            self.lm_head.bias = None
        self.lm_head.weight = self.tok_emb.weight            # 权重共享

        self.apply(self._init_weights)
        # 残差分支输出投影按 1/sqrt(2L) 缩放（GPT-2 论文原文做法），抑制深层方差累积
        std = 0.02 / math.sqrt(2 * cfg.n_layer)
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=std)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def num_parameters(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
            if self.learned_pe is not None:
                n -= self.learned_pe.numel()
        return n

    def forward(self, idx, targets=None, kv_cache=None, pos_offset=0):
        """idx: (B,T) token id。kv_cache: list[tuple(k,v)] 每层一个，推理时逐 token decode 用。

        返回 (logits, loss, new_caches)。训练时 kv_cache=None。
        """
        B, T = idx.shape
        x = self.tok_emb(idx)

        # —— 位置信息 ——
        if self.learned_pe is not None:
            x = x + self.learned_pe[pos_offset : pos_offset + T]
        elif self.sinusoidal_pe is not None:
            x = x + self.sinusoidal_pe(pos_offset, T)
        # rope 不在这里加，而是在每层注意力内部旋转 q/k
        x = self.drop(x)

        new_caches = [] if kv_cache is not None else None
        for i, block in enumerate(self.blocks):
            c = kv_cache[i] if (kv_cache is not None and i < len(kv_cache)) else None
            x, new_c = block(x, c, pos_offset, self.rope)
            if kv_cache is not None:
                new_caches.append(new_c)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100
            )
        return logits, loss, new_caches

    @staticmethod
    def _sample_next(logits, temperature: float, top_k: int | None, generator=None):
        logits = logits[:, -1, :].clone()
        if temperature <= 0:
            return logits.argmax(dim=-1, keepdim=True)
        logits = logits / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=generator)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 0.8,
                 top_k: int | None = 40, use_cache: bool = True, generator=None):
        """自回归采样。use_cache=True：prefill 一次，之后每步只喂 1 个 token 并复用历史 K/V。"""
        self.eval()
        total = idx.size(1) + max_new_tokens
        assert total <= self.cfg.block_size, (
            f"prompt({idx.size(1)}) + max_new_tokens({max_new_tokens}) 超出 block_size({self.cfg.block_size})"
        )

        if not use_cache:
            for _ in range(max_new_tokens):
                logits, _, _ = self(idx, kv_cache=None, pos_offset=0)
                nxt = self._sample_next(logits, temperature, top_k, generator)
                idx = torch.cat([idx, nxt], dim=1)
            return idx

        # prefill：整段 prompt 一次前向，建立每层的 K/V cache
        logits, _, caches = self(idx, kv_cache=[], pos_offset=0)
        pos = idx.size(1)
        for i in range(max_new_tokens):
            nxt = self._sample_next(logits, temperature, top_k, generator)
            idx = torch.cat([idx, nxt], dim=1)
            if i == max_new_tokens - 1:
                break
            # decode：只前向 1 个 token，位置从 pos 开始，历史 K/V 从 cache 复用
            logits, _, caches = self(idx[:, -1:], kv_cache=caches, pos_offset=pos)
            pos += 1
        return idx
