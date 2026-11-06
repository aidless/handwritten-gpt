"""KV cache 推理加速实测：同一模型、同一 prompt，对比有/无 cache 的墙钟时间与 tokens/s。

原理：无 cache 时每生成 1 个 token 都要把整段历史重新做一遍注意力（O(t) 增长）；
有 cache 时 prefill 只做一次，之后每步只算 1 个 token 的 Q 和 1×t 的注意力，单步代价基本恒定。
因此生成长度越长，累计加速比越大 —— 脚本里的 per-token 延迟曲线给出实测支撑。

用法：
    python kv_cache_bench.py --ckpt experiments/tiny/tiny.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import torch

from model import GPT, GPTConfig

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "experiments" / "kv_cache"


def load_model(ckpt_path: Path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = GPTConfig(**ck["config"])
    model = GPT(cfg)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, cfg


def timed_generate(model, prompt_ids, n_new, use_cache, repeats=3, seed=1234):
    times = []
    for _ in range(repeats):
        g = torch.Generator().manual_seed(seed)
        t0 = time.perf_counter()
        model.generate(prompt_ids.clone(), max_new_tokens=n_new, temperature=0.8,
                       top_k=40, use_cache=use_cache, generator=g)
        times.append(time.perf_counter() - t0)
    return statistics.median(times), min(times), max(times)


@torch.no_grad()
def per_token_profile(model, prompt_ids, n_new=200, seed=7):
    """逐 token 记录 decode 延迟：cache 路径 vs 无 cache 路径。"""
    g1 = torch.Generator().manual_seed(seed)
    g2 = torch.Generator().manual_seed(seed)
    idx1, idx2 = prompt_ids.clone(), prompt_ids.clone()

    # cache 路径
    logits, _, caches = model(idx1, kv_cache=[], pos_offset=0)
    pos = idx1.size(1)
    cache_lat = []
    for i in range(n_new):
        nxt = model._sample_next(logits, 0.8, 40, g1)
        idx1 = torch.cat([idx1, nxt], dim=1)
        if i == n_new - 1:
            break
        t0 = time.perf_counter()
        logits, _, caches = model(idx1[:, -1:], kv_cache=caches, pos_offset=pos)
        cache_lat.append(time.perf_counter() - t0)
        pos += 1

    # 无 cache 路径（每步重算全序列）
    nocache_lat = []
    for i in range(n_new):
        t0 = time.perf_counter()
        logits, _, _ = model(idx2, kv_cache=None, pos_offset=0)
        nocache_lat.append(time.perf_counter() - t0)
        nxt = model._sample_next(logits, 0.8, 40, g2)
        idx2 = torch.cat([idx2, nxt], dim=1)
        if i == n_new - 1:
            break
    return cache_lat, nocache_lat


def rss_mb() -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return float(line.split()[1]) / 1024.0
    return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "experiments" / "tiny" / "tiny.pt"))
    ap.add_argument("--prompt", default="春江潮水连海平")
    ap.add_argument("--lengths", default="50,100,200")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model, cfg = load_model(Path(args.ckpt))
    tok = json.loads((ROOT / "data" / "vocab.json").read_text(encoding="utf-8"))["itos"]
    stoi = {c: i for i, c in enumerate(tok)}
    prompt_ids = torch.tensor([[stoi.get(c, 0) for c in args.prompt]], dtype=torch.long)
    lengths = [int(x) for x in args.lengths.split(",")]

    # —— 正确性：贪心解码下 cache 与无 cache 必须逐 token 一致 ——
    a = model.generate(prompt_ids.clone(), 20, temperature=0, use_cache=True)
    b = model.generate(prompt_ids.clone(), 20, temperature=0, use_cache=False)
    identical = torch.equal(a, b)
    print(f"[kv] cache 正确性（贪心 20 token 一致）: {identical}")
    assert identical, "KV cache 输出与无 cache 不一致，测试失败"

    # —— prefill 耗时 ——
    with torch.no_grad():
        t0 = time.perf_counter()
        model(prompt_ids, kv_cache=[], pos_offset=0)
        prefill_s = time.perf_counter() - t0
    print(f"[kv] prefill {prompt_ids.numel()} token: {prefill_s*1000:.2f} ms")

    # —— 加速比表 ——
    rows = []
    for n in lengths:
        t_cache, cmin, cmax = timed_generate(model, prompt_ids, n, True, args.repeats)
        t_nocache, nmin, nmax = timed_generate(model, prompt_ids, n, False, args.repeats)
        row = {
            "prompt": args.prompt,
            "new_tokens": n,
            "cache_s": t_cache,
            "nocache_s": t_nocache,
            "cache_tokens_per_s": n / t_cache,
            "nocache_tokens_per_s": n / t_nocache,
            "speedup": t_nocache / t_cache,
        }
        rows.append(row)
        print(f"[kv] n={n:3d}  cache={t_cache*1000:7.1f}ms  no-cache={t_nocache*1000:7.1f}ms  "
              f"speedup={row['speedup']:.2f}x  cache_tok/s={row['cache_tokens_per_s']:.1f}  "
              f"nocache_tok/s={row['nocache_tokens_per_s']:.1f}")

    # —— cache 内存代价 & 逐 token 延迟 ——
    cache_bytes_per_token = cfg.n_layer * 2 * cfg.n_head * cfg.head_dim * 4
    cache_bytes = cache_bytes_per_token * (prompt_ids.size(1) + max(lengths))
    cache_lat, nocache_lat = per_token_profile(model, prompt_ids, n_new=min(max(lengths), cfg.block_size - prompt_ids.size(1)))
    head = cache_lat[:50]
    tail = cache_lat[-50:]
    print(f"[kv] 单层单 token K/V={cache_bytes_per_token/cfg.n_layer/2:.0f}B, "
          f"全模型 seq={prompt_ids.size(1)+max(lengths)}: {cache_bytes/1024:.0f} KiB, RSS={rss_mb():.0f} MB")
    print(f"[kv] cache decode 前50步={statistics.mean(head)*1000:.2f}ms/token, "
          f"末50步={statistics.mean(tail)*1000:.2f}ms/token")
    print(f"[kv] 无 cache 前50步={statistics.mean(nocache_lat[:50])*1000:.2f}ms/token, "
          f"末50步={statistics.mean(nocache_lat[-50:])*1000:.2f}ms/token")

    # —— 落盘 ——
    with (OUT_DIR / "kv_cache_bench.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})

    detail = {
        "checkpoint": str(Path(args.ckpt)),
        "prompt": args.prompt,
        "config": cfg.__dict__,
        "prefill_ms": prefill_s * 1000,
        "cache_correct": identical,
        "cache_bytes_full_seq": cache_bytes,
        "cache_bytes_per_layer_per_token": cache_bytes_per_token,
        "rows": rows,
        "cache_decode_ms_first50": statistics.mean(head) * 1000,
        "cache_decode_ms_last50": statistics.mean(tail) * 1000,
        "nocache_decode_ms_first50": statistics.mean(nocache_lat[:50]) * 1000,
        "nocache_decode_ms_last50": statistics.mean(nocache_lat[-50:]) * 1000,
        "rss_mb": rss_mb(),
    }
    (OUT_DIR / "kv_cache_detail.json").write_text(json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8")

    # 曲线：逐 token 延迟
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4.2))
    plt.plot(range(len(cache_lat)), [x * 1000 for x in cache_lat], label="with KV cache")
    plt.plot(range(len(nocache_lat)), [x * 1000 for x in nocache_lat], label="no cache")
    plt.xlabel("decode step")
    plt.ylabel("latency per token (ms)")
    plt.title("per-token decode latency")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "kv_cache_latency.png", dpi=140)


if __name__ == "__main__":
    main()
