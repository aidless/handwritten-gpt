"""把落盘的 json（各次训练的 meta、KV cache 结果、微基准）汇总成 experiments/summary.csv。

不手填任何数字：所有字段都从 experiments/ 下的 json 读取。
后续新增实验只要把 meta 放在 experiments/<name>/<name>_meta.json，重跑本脚本即可。

用法：
    python make_summary.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

EXP = Path("experiments")
OUT = EXP / "summary.csv"
FIELDS = ["experiment", "config", "params", "steps", "final_val_loss",
          "wall_seconds", "tokens_per_sec", "notes"]


def cfg_str(m: dict) -> str:
    c = m["config"]
    return (f"L{c['n_layer']}/d{c['n_embd']}/H{c['n_head']}/ctx{c['block_size']};"
            f"pos={c['pos_emb']}; norm={c['norm_placement']}/{c.get('norm_type', 'layernorm')}")


def add_training(rows: list[dict], path: Path, name: str, note: str = "") -> None:
    if not path.exists():
        print(f"skip (no meta): {path}")
        return
    m = json.loads(path.read_text())
    rows.append({
        "experiment": name,
        "config": cfg_str(m),
        "params": m["num_params"],
        "steps": m["steps_done"],
        "final_val_loss": f"{m['final_val_loss']:.4f}",
        "wall_seconds": f"{m.get('wall_clock_s', 0):.1f}",
        "tokens_per_sec": f"{m.get('tokens_per_s', 0):.0f}",
        "notes": note or f"最终 train={m['final_train_loss_smoothed']:.4f}; "
                         f"收敛步={m.get('convergence_step_95pct')}",
    })


def main() -> None:
    rows: list[dict] = []

    add_training(rows, EXP / "tiny/tiny_meta.json", "main_tiny",
                 "主实验；batch=12；warmup=200；3000 步 ≈3.1 epoch")
    for tag, note in [
        ("base", "消融基线；1200 步统一预算"),
        ("rope", "只换位置编码：RoPE"),
        ("depth2", "只换深度：L2"),
        ("depth6", "只换深度：L6；该轮撞上外部负载尖峰，wall 不可比"),
        ("postln", "只换归一化位置：Post-LN"),
    ]:
        add_training(rows, EXP / f"ablations/{tag}/{tag}_meta.json", f"ablate_{tag}", note)
    add_training(rows, EXP / "rmsnorm/rmsnorm_meta.json", "rmsnorm_tiny",
                 "只换归一化类型：RMSNorm；与 main_tiny 同配置同步数")

    # 主实验第二个 seed：与 main_tiny 只换 seed（数据切分仍固定 1337），
    # checkpoint/日志写在 .local_runs/（已被 .gitignore 排除）；差值由两份 meta 现算
    second = EXP.parent / ".local_runs/tiny_seed42/tiny_seed42_meta.json"
    base_meta = EXP / "tiny/tiny_meta.json"
    if second.exists():
        delta = (json.loads(second.read_text())["final_val_loss"]
                 - json.loads(base_meta.read_text())["final_val_loss"])
        conv = json.loads(second.read_text()).get("convergence_step_95pct")
        add_training(rows, second, "main_tiny_seed42",
                     f"主实验第二 seed（seed=42）；与 main_tiny 仅 seed 不同；"
                     f"val 差={delta:+.4f}；收敛步={conv}；墙钟受负载污染不可比；"
                     f"产物在 .local_runs/ 未入库")

    kv = EXP / "kv_cache/kv_cache_detail.json"
    if kv.exists():
        d = json.loads(kv.read_text())
        tiny_meta = EXP / "tiny/tiny_meta.json"
        n_params = json.loads(tiny_meta.read_text())["num_params"] if tiny_meta.exists() else ""
        for r in d["rows"]:
            n = r["new_tokens"]
            rows.append({
                "experiment": f"kv_cache_gen{n}",
                "config": f"tiny; ctx=256; prompt_tokens={len(d['prompt'])}",
                "params": n_params,
                "steps": n,
                "final_val_loss": "",
                "wall_seconds": f"{r['cache_s']:.3f}",
                "tokens_per_sec": f"{r['cache_tokens_per_s']:.1f}",
                "notes": f"有 cache；加速比 {r['speedup']:.2f}× vs 无 cache "
                         f"{r['nocache_s']:.3f}s；prefill={d['prefill_ms']:.2f}ms",
            })
            rows.append({
                "experiment": f"kv_nocache_gen{n}",
                "config": f"tiny; ctx=256; prompt_tokens={len(d['prompt'])}",
                "params": n_params,
                "steps": n,
                "final_val_loss": "",
                "wall_seconds": f"{r['nocache_s']:.3f}",
                "tokens_per_sec": f"{r['nocache_tokens_per_s']:.1f}",
                "notes": f"无 cache；与 kv_cache_gen{n} 同一 prompt/同一次运行",
            })

    for bench_path in (EXP / "bench_micro.json", EXP / "bench_norm.json"):
        if not bench_path.exists():
            continue
        for r in json.loads(bench_path.read_text())["rows"]:
            cfg = r["config"]
            cfg_text = (f"L{cfg.get('n_layer', 4)}/d192/H6/ctx192/B16; "
                        f"pos={cfg.get('pos_emb', 'learned')}; "
                        f"norm={cfg.get('norm_type', 'layernorm')}; threads={r['threads']}")
            rows.append({
                "experiment": f"bench_{r['name']}",
                "config": cfg_text,
                "params": r["params"],
                "steps": 8,
                "final_val_loss": "",
                "wall_seconds": f"{r['ms_per_step'] / 1000 * 8:.2f}",
                "tokens_per_sec": f"{r['tokens_per_s']:.1f}",
                "notes": f"受控微基准：{r['ms_per_step']} ms/step（3 轮交替中位数 "
                         f"{r['round_ms']}）；load≈{r['load_avg']}",
            })

    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"written {len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
