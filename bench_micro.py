"""受控微基准：在固定 batch 上只测「前向 + 反向」的每步耗时。

为什么需要它：消融训练期间机器上有外部负载（别人的 ollama / 训练任务），
训练日志里的 wall-clock 只能做组内相对比较。本脚本在短时间内反复交替测量
不同配置，用「轮次中位数 + 交替顺序」抵消负载漂移，支撑「每步慢 X%」的结论。

用法：
    python bench_micro.py                 # 结果写入 experiments/bench_micro.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch

from model import GPT, GPTConfig

VOCAB, CTX, BATCH = 6497, 192, 16


def timed_steps(model, x, y, n_steps: int, warmup: int = 2) -> float:
    """返回单步（前向+反向+清零梯度）的中位耗时，单位毫秒。"""
    for _ in range(warmup):
        _, loss, _ = model(x, y)
        loss.backward()
        model.zero_grad(set_to_none=True)
    t0 = time.perf_counter()
    for _ in range(n_steps):
        _, loss, _ = model(x, y)
        loss.backward()
        model.zero_grad(set_to_none=True)
    return (time.perf_counter() - t0) / n_steps * 1000.0


def build(**overrides) -> GPT:
    torch.manual_seed(1337)
    cfg = dict(vocab_size=VOCAB, block_size=CTX, n_layer=4, n_head=6, n_embd=192)
    cfg.update(overrides)
    return GPT(GPTConfig(**cfg))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3, help="交替轮数，取每轮中位数再取中位数")
    ap.add_argument("--n-steps", type=int, default=8, help="每轮内的计时步数")
    ap.add_argument("--out", default="experiments/bench_micro.json")
    ap.add_argument("--only", default=None, help="只跑名字包含该子串的用例（如 norm）")
    args = ap.parse_args()

    cases = [
        ("pos_learned", 8, dict(pos_emb="learned")),
        ("pos_rope", 8, dict(pos_emb="rope")),
        ("depth_L2", 8, dict(n_layer=2)),
        ("depth_L4", 8, dict(n_layer=4)),
        ("depth_L6", 8, dict(n_layer=6)),
        ("norm_layernorm", 8, dict()),
        ("norm_rmsnorm", 8, dict(norm_type="rmsnorm")),
        ("threads_4", 4, dict()),
        ("threads_8", 8, dict()),
        ("threads_12", 12, dict()),
        ("threads_16", 16, dict()),
    ]
    if args.only:
        cases = [c for c in cases if args.only in c[0]]
    x = torch.randint(0, VOCAB, (BATCH, CTX))
    y = torch.randint(0, VOCAB, (BATCH, CTX))
    models = {name: build(**kw) for name, _, kw in cases}
    per_case: dict[str, list[float]] = {name: [] for name, _, _ in cases}

    for r in range(args.rounds):
        order = cases if r % 2 == 0 else list(reversed(cases))  # 交替顺序抵消负载漂移
        for name, threads, _ in order:
            torch.set_num_threads(threads)
            per_case[name].append(timed_steps(models[name], x, y, args.n_steps))

    rows = []
    for name, threads, kw in cases:
        ms = statistics.median(per_case[name])
        rows.append({
            "name": name,
            "threads": threads,
            "config": kw,
            "params": models[name].num_parameters(),
            "tokens_per_step": BATCH * CTX,
            "ms_per_step": round(ms, 1),
            "tokens_per_s": round(BATCH * CTX / (ms / 1000.0), 1),
            "round_ms": [round(v, 1) for v in per_case[name]],
            "load_avg": round(os.getloadavg()[0], 2),
        })
        print(f"{name:14s} threads={threads:2d} {ms:8.1f} ms/step  "
              f"{BATCH*CTX/(ms/1000.0):7.1f} tok/s  rounds={rows[-1]['round_ms']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2) + "\n")
    print(f"written -> {out}")


if __name__ == "__main__":
    main()
