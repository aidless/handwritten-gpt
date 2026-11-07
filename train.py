"""训练循环：手写 AdamW、warmup + cosine 调度、梯度裁剪、验证、采样、画曲线。

用法：
    python train.py --preset tiny          # 主实验（CPU 约 1 小时）
    python train.py --preset ablation      # 消融实验用的统一预算
    python train.py --preset debug         # 冒烟测试

设计说明（对应论文/工程实践）：
- AdamW（Loshchilov & Hutter 2019）：weight decay 只作用于 2 维以上参数（矩阵），
  LN/偏置不做衰减。这里手写实现，不用 torch.optim。
- 学习率：线性 warmup + 余弦衰减到峰值的 10%（GPT-2/GPT-3 的常见配置）。
- 梯度裁剪：全局 L2 范数裁剪到 1.0，手写实现，避免依赖 nn.utils。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

try:                      # DCU 容器可能没装 matplotlib；画图只是产物，不参与训练
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    plt = None
    _HAS_MPL = False

import numpy as np
import torch

import data as data_lib
from model import GPT, GPTConfig

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT / "experiments"

PRESETS = {
    # 主实验：6.1M 参数，3 个 epoch 左右，8 线程 CPU 约 1 小时
    "tiny": dict(n_layer=6, n_head=8, n_embd=256, block_size=256, batch_size=12,
                 steps=3000, warmup=200, lr=3e-4, dropout=0.1, eval_every=100,
                 weight_decay=0.1, grad_clip=1.0, threads=8),
    # 消融统一预算：比主实验小一档，保证 5 组实验能在 CPU 上跑完；除被消融因素外完全相同
    "ablation": dict(n_layer=4, n_head=6, n_embd=192, block_size=192, batch_size=16,
                     steps=1600, warmup=160, lr=3e-4, dropout=0.1, eval_every=100,
                     weight_decay=0.1, grad_clip=1.0, threads=8),
    "debug": dict(n_layer=2, n_head=4, n_embd=64, block_size=64, batch_size=8,
                  steps=30, warmup=10, lr=3e-4, dropout=0.0, eval_every=10,
                  weight_decay=0.1, grad_clip=1.0, threads=8),
    "xl": dict(n_layer=12, n_head=10, n_embd=640, block_size=512, batch_size=16,
               steps=2000, warmup=200, lr=2e-4, dropout=0.1, eval_every=100,
               weight_decay=0.1, grad_clip=1.0, threads=8),
    # 120M 档：只在 DCU/GPU 上用（CPU 上单步会到几十秒），结构与 xl 同族，只放大 d 与层数
    "xl120m": dict(n_layer=16, n_head=12, n_embd=768, block_size=512, batch_size=16,
                   steps=2000, warmup=200, lr=1.8e-4, dropout=0.1, eval_every=100,
                   weight_decay=0.1, grad_clip=1.0, threads=8),
}

PROMPTS = ["春江潮水连海平", "白日依山尽", "秋风", "明月", "山中"]


class AdamW:
    """手写 decoupled weight decay AdamW。"""

    def __init__(self, params, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1):
        self.params = [p for p in params if p.requires_grad]
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self, lr: float) -> None:
        self.t += 1
        b1, b2 = self.betas
        bc1 = 1.0 - b1 ** self.t
        bc2 = 1.0 - b2 ** self.t
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            g = p.grad
            m.mul_(b1).add_(g, alpha=1.0 - b1)
            v.mul_(b2).addcmul_(g, g, value=1.0 - b2)
            if p.dim() >= 2 and self.weight_decay > 0:
                p.mul_(1.0 - lr * self.weight_decay)
            denom = (v.sqrt() / math.sqrt(bc2)).add_(self.eps)
            p.addcdiv_(m / bc1, denom, value=-lr)


def clip_grad_norm_(params, max_norm: float) -> float:
    """手写全局范数梯度裁剪，返回裁剪前的范数。"""
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return 0.0
    total = torch.sqrt(sum((g * g).sum() for g in grads))
    norm = float(total)
    if norm > max_norm:
        scale = max_norm / (norm + 1e-6)
        for g in grads:
            g.mul_(scale)
    return norm


def lr_at(step: int, peak: float, warmup: int, total: int, min_ratio: float = 0.1) -> float:
    """线性 warmup + 余弦衰减。"""
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return peak * (min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))


@torch.no_grad()
def evaluate(model: GPT, batches, device="cpu") -> float:
    model.eval()
    total, n = 0.0, 0
    for x, y in batches:
        _, loss, _ = model(x.to(device), y.to(device))
        total += float(loss)
        n += 1
    model.train()
    return total / max(1, n)


@torch.no_grad()
def sample_texts(model: GPT, tok, prompts, max_new: int = 120, seed: int = 1234, device: str = "cpu"):
    model.eval()
    max_new = min(max_new, model.cfg.block_size - max(len(p) for p in prompts))
    results = []
    for p in prompts:
        g = torch.Generator(device=device).manual_seed(seed)
        ids = torch.tensor([tok.encode(p)], dtype=torch.long, device=device)
        out = model.generate(ids, max_new_tokens=max_new, temperature=0.8, top_k=40,
                             use_cache=True, generator=g)
        results.append({"prompt": p, "output": tok.decode(out[0].tolist())})
    model.train()
    return results


def plot_curves(csv_path: Path, png_path: Path, title: str) -> None:
    if not _HAS_MPL:
        print("[plot] 跳过：matplotlib 未安装（csv/meta/pt 照常产出）")
        return
    rows = list(csv.DictReader(csv_path.open()))
    steps = [int(r["step"]) for r in rows]
    train = [float(r["train_loss"]) for r in rows]
    val_steps = [int(r["step"]) for r in rows if r["val_loss"]]
    val = [float(r["val_loss"]) for r in rows if r["val_loss"]]
    # 简单滑动平均让训练曲线可读
    win = max(1, len(train) // 80)
    smooth = np.convolve(train, np.ones(win) / win, mode="valid")

    plt.figure(figsize=(7, 4.2))
    plt.plot(steps[win - 1:], smooth, label="train loss (smoothed)", color="#2b6cb0")
    plt.plot(val_steps, val, label="val loss", color="#c53030", marker="o", ms=3)
    plt.xlabel("step")
    plt.ylabel("cross-entropy loss")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=140)
    plt.close()


def _save_ckpt(path: Path, model, opt, step: int, steps_total: int, cfg_dict: dict, seed: int,
               device: str, rng, tokens: int, t_train: float,
               loss_history: list, val_history: list) -> None:
    """原子写断点：model + Adam(m/v/t) + step + 三套 RNG（含数据批 RNG），供 --resume 续跑。

    先写 .tmp 再 os.replace，避免崩溃时留下半截文件把续跑带偏。
    """
    payload = {
        "model": model.state_dict(),
        "opt_m": opt.m, "opt_v": opt.v, "opt_t": opt.t,
        "step": step, "steps_total": steps_total,
        "cfg_dict": cfg_dict, "seed": seed,
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if device.startswith("cuda") else [],
        "rng_np": rng.bit_generator.state,
        "tokens": tokens, "t_train": t_train,
        "loss_history": loss_history, "val_history": val_history,
    }
    tmp = str(path) + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def run(cfg_dict: dict, seed: int, out_dir: Path, tag: str, pos_emb: str, norm: str,
        steps: int | None = None, dropout: float | None = None,
        norm_type: str = "layernorm", device: str = "cpu", amp_dtype=None,
        resume: bool = False, ckpt_every: int = 500) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = steps or cfg_dict["steps"]
    if dropout is not None:
        cfg_dict = dict(cfg_dict, dropout=dropout)

    torch.manual_seed(seed)
    torch.set_num_threads(cfg_dict["threads"])

    stats = data_lib.prepare_corpus()
    ds = data_lib.make_datasets(cfg_dict["block_size"], seed=1337)
    rng = np.random.default_rng(seed)

    cfg = GPTConfig(
        vocab_size=ds["vocab_size"],
        block_size=cfg_dict["block_size"],
        n_layer=cfg_dict["n_layer"],
        n_head=cfg_dict["n_head"],
        n_embd=cfg_dict["n_embd"],
        dropout=cfg_dict["dropout"],
        pos_emb=pos_emb,
        norm_placement=norm,
        norm_type=norm_type,
    )
    model = GPT(cfg).to(device)
    model.train()
    opt = AdamW(model.parameters(), weight_decay=cfg_dict["weight_decay"])

    # ---- 断点续跑（--resume，默认关闭，不改既有行为）----
    ckpt_path = out_dir / f"{tag}_ckpt.pt"
    start_step, tokens, t_train = 0, 0, 0.0
    loss_history, val_history = [], []
    if resume and ckpt_path.exists():
        # map_location="cpu"：RNG 状态是 CPU ByteTensor，跟着 model 一起 map 到 cuda 会导致 set_rng_state 失败
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if int(ck["steps_total"]) != steps:
            raise RuntimeError(f"ckpt 的 steps_total={ck['steps_total']} 与本次 steps={steps} 不一致，拒绝续跑")
        model.load_state_dict(ck["model"])
        opt.m = [x.to(device) for x in ck["opt_m"]]
        opt.v = [x.to(device) for x in ck["opt_v"]]
        opt.t = int(ck["opt_t"])
        start_step = int(ck["step"])
        tokens = int(ck["tokens"])
        t_train = float(ck["t_train"])
        loss_history = list(ck["loss_history"])
        val_history = list(ck["val_history"])
        torch.set_rng_state(ck["rng_torch"].cpu().to(torch.uint8))
        if device.startswith("cuda"):
            torch.cuda.set_rng_state_all([t.cpu().to(torch.uint8) for t in ck["rng_cuda"]])
        rng.bit_generator.state = ck["rng_np"]
        print(f"[run:{tag}] RESUME step={start_step}/{steps} tokens={tokens}", flush=True)

    val_batches = data_lib.fixed_eval_batches(ds["val"], cfg_dict["batch_size"], cfg.block_size, n_batches=10)
    train_eval_batches = data_lib.fixed_eval_batches(ds["train"], cfg_dict["batch_size"], cfg.block_size, n_batches=10, seed=777)

    csv_path = out_dir / f"{tag}.csv"
    header = ["step", "lr", "train_loss", "grad_norm", "val_loss", "elapsed_s", "tokens_per_s"]
    if start_step > 0 and csv_path.exists():
        # 续跑：先截断到 ckpt 步数再追加，保证 CSV 不重复
        keep = [r for r in csv.DictReader(csv_path.open()) if int(r["step"]) <= start_step]
        log_csv = csv_path.open("w", newline="")
        writer = csv.writer(log_csv)
        writer.writerow(header)
        for r in keep:
            writer.writerow([r[k] for k in header])
        log_csv.flush()
        log_csv = csv_path.open("a", newline="")
        writer = csv.writer(log_csv)
    else:
        log_csv = csv_path.open("w", newline="")
        writer = csv.writer(log_csv)
        writer.writerow(header)

    n_params = model.num_parameters()
    print(f"[run:{tag}] params={n_params/1e6:.2f}M layers={cfg.n_layer} d={cfg.n_embd} "
          f"ctx={cfg.block_size} pos={cfg.pos_emb} norm={cfg.norm_placement}/{cfg.norm_type} "
          f"steps={steps}")

    t_start = time.perf_counter() - t_train   # 续跑时 elapsed_s 继续累计（不含重启停机空档）
    diverged = False
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()   # 显存峰值按「本次训练」计

    for step in range(start_step, steps):
        lr = lr_at(step, cfg_dict["lr"], cfg_dict["warmup"], steps)
        x, y = data_lib.get_batch(ds["train"], cfg_dict["batch_size"], cfg.block_size, rng)
        x, y = x.to(device), y.to(device)
        t0 = time.perf_counter()
        with torch.autocast(device_type=("cuda" if device.startswith("cuda") else "cpu"),
                            dtype=amp_dtype, enabled=amp_dtype is not None):
            _, loss, _ = model(x, y)
        opt.zero_grad()
        loss.backward()
        gnorm = clip_grad_norm_(model.parameters(), cfg_dict["grad_clip"])
        opt.step(lr)
        t_train += time.perf_counter() - t0
        tokens += x.numel()

        loss_val = float(loss.detach())
        loss_history.append(loss_val)
        if not math.isfinite(loss_val):
            diverged = True
            print(f"[run:{tag}] step {step}: loss 非有限值，判定发散")
            break

        val_loss = ""
        if (step + 1) % cfg_dict["eval_every"] == 0 or step == steps - 1:
            v = evaluate(model, val_batches, device)
            val_loss = f"{v:.4f}"
            val_history.append((step + 1, v))

        elapsed = time.perf_counter() - t_start
        writer.writerow([step + 1, f"{lr:.6f}", f"{loss_val:.4f}", f"{gnorm:.3f}",
                         val_loss, f"{elapsed:.1f}", f"{tokens / max(t_train, 1e-9):.0f}"])
        if val_loss != "":
            log_csv.flush()   # 周期性刷盘，防止长训练中途崩溃丢日志
        elif (step + 1) % 5 == 0:
            log_csv.flush()   # 短程探测也要能边跑边看
        if ckpt_every and (step + 1) % ckpt_every == 0:
            _save_ckpt(ckpt_path, model, opt, step + 1, steps, cfg_dict, seed, device,
                       rng, tokens, t_train, loss_history, val_history)
        if (step + 1) % 100 == 0 or step == steps - 1:
            print(f"[run:{tag}] step {step+1}/{steps} loss={loss_val:.4f} "
                  f"val={val_loss or '-'} lr={lr:.2e} tok/s={tokens/max(t_train,1e-9):,.0f} "
                  f"elapsed={elapsed:.0f}s")
    log_csv.close()
    if ckpt_path.exists():
        ckpt_path.unlink()   # 正常跑完即删断点，只保留最终 .pt（避免每格多占 GB 级磁盘）

    wall = time.perf_counter() - t_start
    # 训练段（不含 eval/采样）的显存峰值：判断这一档在当前卡上能不能装下就看它
    peak_vram = (torch.cuda.max_memory_allocated() / 2 ** 20) if device.startswith("cuda") else None
    final_train = float(np.mean(loss_history[-50:]))
    final_val = val_history[-1][1] if val_history else float("nan")
    train_eval = evaluate(model, train_eval_batches, device)

    # 收敛速度：达到自身总降幅 95% 所需步数（用验证曲线，避免训练噪声）
    conv_step = ""
    if val_history:
        start_loss = val_history[0][1]
        best = min(v for _, v in val_history)
        target = best + 0.05 * max(0.0, start_loss - best)
        for s, v in val_history:
            if v <= target:
                conv_step = s
                break

    samples = sample_texts(model, ds["tokenizer"], PROMPTS, device=device)
    torch.save({"config": cfg.__dict__, "state_dict": model.state_dict()}, out_dir / f"{tag}.pt")
    plot_curves(csv_path, out_dir / f"{tag}_loss.png", f"{tag}: loss curve")

    meta = {
        "tag": tag,
        "config": cfg.__dict__,
        "preset": cfg_dict,
        "seed": seed,
        "steps_planned": steps,
        "steps_done": len(loss_history),
        "num_params": n_params,
        "num_params_non_embedding": model.num_parameters(non_embedding=True),
        "vocab_size": ds["vocab_size"],
        "corpus_chars": stats.get("num_chars"),
        "num_poems": stats.get("num_poems"),
        "train_tokens_seen": tokens,
        "final_train_loss_smoothed": final_train,
        "final_val_loss": final_val,
        "train_eval_loss": train_eval,
        "wall_clock_s": wall,
        "train_wall_clock_s": t_train,
        "tokens_per_s": tokens / max(t_train, 1e-9),
        "convergence_step_95pct": conv_step,
        "diverged": diverged,
    }
    if peak_vram is not None:
        meta["peak_vram_mib"] = round(peak_vram, 1)
    (out_dir / f"{tag}_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / f"{tag}_samples.json").write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / f"{tag}_samples.md").open("w", encoding="utf-8") as f:
        f.write(f"# 生成样例（{tag}，temperature=0.8, top_k=40）\n\n")
        for s in samples:
            f.write(f"**prompt**: `{s['prompt']}`\n\n{s['output']}\n\n---\n\n")
    print(f"[run:{tag}] 完成 wall={wall/60:.1f}min final_val={final_val:.4f} "
          f"train={final_train:.4f} params={n_params/1e6:.2f}M")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="tiny", choices=list(PRESETS))
    ap.add_argument("--steps", type=int, default=None, help="覆盖预设步数（用于试跑）")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--pos-emb", default=None, choices=["learned", "sinusoidal", "rope"])
    ap.add_argument("--norm", default=None, choices=["pre", "post"], help="归一化位置")
    ap.add_argument("--norm-type", default="layernorm", choices=["layernorm", "rmsnorm"])
    ap.add_argument("--n-layer", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--resume", action="store_true",
                    help="从 {tag}_ckpt.pt 续跑；CSV 自动截断到断点步数后追加（默认关闭）")
    ap.add_argument("--ckpt-every", type=int, default=500, help="每 N 步存断点；0 = 关闭")
    args = ap.parse_args()

    cfg_dict = dict(PRESETS[args.preset])
    # 命令行覆盖（用于单因素实验；ablate.py 也走这条路径）
    if args.n_layer is not None:
        cfg_dict["n_layer"] = args.n_layer
    if args.threads is not None:
        cfg_dict["threads"] = args.threads

    # 位置编码与归一化位置通过环境变量/参数传给模型配置，这里用简单映射
    pos_emb = args.pos_emb or "learned"
    norm = args.norm or "pre"
    tag = args.tag or f"{args.preset}_{pos_emb}_{norm}_L{cfg_dict['n_layer']}"
    out_dir = Path(args.out_dir) if args.out_dir else EXPERIMENTS / tag

    amp_dtype = None if args.dtype == "fp32" else torch.bfloat16
    run(cfg_dict, args.seed, out_dir, tag, pos_emb, norm, steps=args.steps, dropout=args.dropout,
        norm_type=args.norm_type, device=args.device, amp_dtype=amp_dtype,
        resume=args.resume, ckpt_every=args.ckpt_every)


if __name__ == "__main__":
    main()
