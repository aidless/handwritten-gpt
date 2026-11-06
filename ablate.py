"""消融实验：三组单因素对照，其余配置完全一致（同 seed、同数据、同 batch、同步数、同 lr）。

    位置编码：learned（可学习绝对位置） vs rope
    深度：    L2 / L4 / L6（L4 与其余组共享基线）
    归一化：  Pre-LN vs Post-LN

用法：
    python ablate.py                 # 跑全部（约 1.5 小时 CPU）
    python ablate.py --groups pos    # 只跑位置编码组
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import train as train_lib

ROOT = Path(__file__).resolve().parent


VARIANTS = {
    # name: (pos_emb, n_layer, norm)
    "base":     ("learned", 4, "pre"),
    "rope":     ("rope", 4, "pre"),
    "depth2":   ("learned", 2, "pre"),
    "depth6":   ("learned", 6, "pre"),
    "postln":   ("learned", 4, "post"),
}

GROUPS = {
    "pos": ["base", "rope"],
    "depth": ["depth2", "base", "depth6"],
    "norm": ["base", "postln"],
}


def load_val_curve(csv_path: Path):
    rows = list(csv.DictReader(csv_path.open()))
    return [(int(r["step"]), float(r["val_loss"])) for r in rows if r["val_loss"]]


def plot_group(group: str, names: list[str], out_dir: Path) -> None:
    plt.figure(figsize=(7, 4.2))
    for name in names:
        curve = load_val_curve(out_dir / name / f"{name}.csv")
        plt.plot([s for s, _ in curve], [v for _, v in curve], label=name, marker="o", ms=3)
    plt.xlabel("step")
    plt.ylabel("val loss")
    plt.title(f"ablation: {group}")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"ablation_{group}.png", dpi=140)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="pos,depth,norm")
    ap.add_argument("--preset", default="ablation")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out-dir", default=str(ROOT / "experiments" / "ablations"))
    args = ap.parse_args()

    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    names = []
    for g in groups:
        for n in GROUPS[g]:
            if n not in names:
                names.append(n)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_cfg = train_lib.PRESETS[args.preset]
    metas = {}

    for name in names:
        pos_emb, n_layer, norm = VARIANTS[name]
        cfg = dict(base_cfg)
        cfg["n_layer"] = n_layer
        meta = train_lib.run(
            cfg, args.seed, out_dir / name, name,
            pos_emb=pos_emb, norm=norm, steps=args.steps,
        )
        metas[name] = meta

    # 汇总表
    summary_csv = out_dir / "summary.csv"
    with summary_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "pos_emb", "n_layer", "norm", "steps_done", "num_params",
                    "final_val_loss", "final_train_loss", "train_eval_loss",
                    "wall_clock_s", "tokens_per_s", "convergence_step_95pct", "diverged"])
        for name, m in metas.items():
            w.writerow([name, m["config"]["pos_emb"], m["config"]["n_layer"],
                        m["config"]["norm_placement"], m["steps_done"], m["num_params"],
                        f"{m['final_val_loss']:.4f}", f"{m['final_train_loss_smoothed']:.4f}",
                        f"{m['train_eval_loss']:.4f}", f"{m['wall_clock_s']:.0f}",
                        f"{m['tokens_per_s']:.0f}", m["convergence_step_95pct"], m["diverged"]])

    for g in groups:
        plot_group(g, GROUPS[g], out_dir)

    print(f"[ablate] 汇总已写入 {summary_csv}")
    print(summary_csv.read_text())


if __name__ == "__main__":
    main()
