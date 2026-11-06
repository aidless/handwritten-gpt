"""语料准备 + 字符级 tokenizer + 批采样。

语料来源：全唐诗（郑州大学整理版），经 HuggingFace 镜像
  yaodi/tangdynastypoems -> Tang_Poetry.zip -> quantangshi_zhengzhou/*.txt
原始文件为 GBK 编码、每文件一首诗（组诗则多首连写）。

预处理：GBK 解码 -> 去空白 -> 每首诗之间用一个换行分隔。
切分：按“诗”为单位做 98/2 划分，避免同一首诗的前后片段同时出现在 train 与 val。
"""

from __future__ import annotations

import json
import random
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
EXTRACT_DIR = RAW_DIR / "quantangshi_zhengzhou"
CORPUS_FILE = DATA_DIR / "corpus.txt"
VOCAB_FILE = DATA_DIR / "vocab.json"
DATASET_URL = "https://hf-mirror.com/datasets/yaodi/tangdynastypoems/resolve/main/Tang_Poetry.zip"


def prepare_corpus(force: bool = False) -> dict:
    """下载（若需要）并构建 corpus.txt，返回语料统计信息。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if not EXTRACT_DIR.exists():
        zip_path = RAW_DIR / "Tang_Poetry.zip"
        if not zip_path.exists():
            print(f"[data] 下载语料 {DATASET_URL}")
            urllib.request.urlretrieve(DATASET_URL, zip_path)
        print("[data] 解压语料")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(RAW_DIR)

    if CORPUS_FILE.exists() and not force:
        return _corpus_stats()

    files = sorted(EXTRACT_DIR.glob("*.txt"))
    poems, skipped = [], 0
    for f in files:
        try:
            text = f.read_bytes().decode("gbk")
        except UnicodeDecodeError:
            skipped += 1
            continue
        # 只保留汉字与常见中文标点，去掉原始排版里的空白与不可见字符
        text = "".join(ch for ch in text if ch.strip())
        if not text:
            skipped += 1
            continue
        poems.append(text)

    CORPUS_FILE.write_text("\n".join(poems) + "\n", encoding="utf-8")
    stats = _corpus_stats()
    stats["skipped_files"] = skipped
    _write_vocab(stats)
    print(f"[data] 语料已写出：{CORPUS_FILE}")
    print(f"[data] 诗数={stats['num_poems']:,} 字符数={stats['num_chars']:,} 词表={stats['vocab_size']}")
    return stats


def _corpus_stats() -> dict:
    text = CORPUS_FILE.read_text(encoding="utf-8")
    return {
        "num_poems": text.count("\n"),
        "num_chars": sum(1 for ch in text if ch != "\n"),
        "chars": len(text),
    }


def _write_vocab(stats: dict, min_freq: int = 2) -> None:
    """字符级词表：出现次数 >= min_freq 的字符入表，其余映射到 <unk>。"""
    text = CORPUS_FILE.read_text(encoding="utf-8")
    freq: dict[str, int] = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    chars = sorted([c for c, n in freq.items() if n >= min_freq and c != "\n"])
    itos = ["<unk>"] + chars
    VOCAB_FILE.write_text(
        json.dumps({"itos": itos, "freq": {c: freq[c] for c in chars}}, ensure_ascii=False),
        encoding="utf-8",
    )
    stats["vocab_size"] = len(itos)
    stats["min_freq"] = min_freq


class CharTokenizer:
    """字符 <-> id 映射，未知字符统一为 <unk>（id=0）。"""

    def __init__(self):
        meta = json.loads(VOCAB_FILE.read_text(encoding="utf-8"))
        self.itos: list[str] = meta["itos"]
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(ch, 0) for ch in text]

    def decode(self, ids) -> str:
        return "".join(self.itos[i] if 0 <= i < len(self.itos) else "<unk>" for i in ids)


def load_poems(text: str) -> list[str]:
    return [p for p in text.split("\n") if p]


def make_datasets(block_size: int, seed: int = 1337) -> dict:
    """按诗切分 train/val，并把 token 序列保存为 numpy 数组（ione 一维长序列）。"""
    raw = CORPUS_FILE.read_text(encoding="utf-8")
    poems = load_poems(raw)
    rng = random.Random(seed)
    rng.shuffle(poems)
    n_val = max(200, int(len(poems) * 0.02))
    val_poems, train_poems = poems[:n_val], poems[n_val:]
    tok = CharTokenizer()

    train_ids = np.array(tok.encode("\n".join(train_poems)), dtype=np.int32)
    val_ids = np.array(tok.encode("\n".join(val_poems)), dtype=np.int32)
    return {
        "train": train_ids,
        "val": val_ids,
        "block_size": block_size,
        "vocab_size": len(tok),
        "tokenizer": tok,
        "num_val_poems": n_val,
    }


def get_batch(ids: np.ndarray, batch_size: int, block_size: int, rng: np.random.Generator):
    """从长序列中均匀采样 batch_size 个起点，返回 (x, y)，y 为 x 右移一位。"""
    hi = len(ids) - block_size - 1
    assert hi > 0, "语料长度不足一个 block"
    starts = rng.integers(0, hi, size=batch_size)
    x = np.stack([ids[s : s + block_size] for s in starts])
    y = np.stack([ids[s + 1 : s + block_size + 1] for s in starts])
    return torch.from_numpy(x).long(), torch.from_numpy(y).long()


def fixed_eval_batches(ids: np.ndarray, batch_size: int, block_size: int, n_batches: int = 20, seed: int = 2024):
    """固定验证集：同样的起点集合，保证不同实验的 val loss 可比。"""
    rng = np.random.default_rng(seed)
    hi = len(ids) - block_size - 1
    starts = rng.integers(0, hi, size=batch_size * n_batches)
    out = []
    for i in range(n_batches):
        s = starts[i * batch_size : (i + 1) * batch_size]
        x = np.stack([ids[j : j + block_size] for j in s])
        y = np.stack([ids[j + 1 : j + block_size + 1] for j in s])
        out.append((torch.from_numpy(x).long(), torch.from_numpy(y).long()))
    return out
