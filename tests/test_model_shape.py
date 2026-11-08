"""CPU 前向契约：真跑 model.GPT，不 mock、不 stub。

配置缩到极小（词表 64、上下文 16、2 层 4 头、4 头 × 8 维），单次前向毫秒级，
所以这个测试在 CI 里装 CPU 版 torch 是划算的：它验证的是 model.py 真正的前向逻辑
（位置编码三条分支、因果掩码、损失可算、权重共享），而不是“能 import 就算过”。
"""

import pytest
import torch

from model import GPT, GPTConfig

VOCAB = 64


def _cfg(**overrides) -> GPTConfig:
    base = dict(vocab_size=VOCAB, block_size=16, n_layer=2, n_head=4, n_embd=32, dropout=0.0)
    base.update(overrides)
    return GPTConfig(**base)


def _batch() -> torch.Tensor:
    generator = torch.Generator().manual_seed(20261108)
    return torch.randint(0, VOCAB, (2, 8), generator=generator)


@pytest.mark.parametrize("pos_emb", ["learned", "sinusoidal", "rope"])
def test_forward_shape_and_dtype(pos_emb):
    model = GPT(_cfg(pos_emb=pos_emb)).eval()
    with torch.no_grad():
        logits, loss, caches = model(_batch())
    assert logits.shape == (2, 8, VOCAB)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()
    assert loss is None
    assert caches is None


def test_forward_with_targets_yields_finite_loss():
    model = GPT(_cfg()).train()
    idx = _batch()
    targets = _batch()
    logits, loss, _ = model(idx, targets)
    assert logits.shape == (2, 8, VOCAB)
    assert loss is not None
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.requires_grad
    loss.backward()
    assert model.lm_head.weight.grad is not None


@torch.no_grad()
def test_causal_mask_hides_future_tokens():
    model = GPT(_cfg()).eval()
    idx = _batch()
    changed = idx.clone()
    changed[:, -1] = (changed[:, -1] + 1) % VOCAB

    before, _, _ = model(idx)
    after, _, _ = model(changed)
    # 只改最后一个 token：前缀位置的 logits 必须逐位相同，末位必须变化
    assert torch.equal(before[:, :-1], after[:, :-1])
    assert not torch.equal(before[:, -1], after[:, -1])


def test_lm_head_shares_embedding_weights():
    model = GPT(_cfg())
    assert model.lm_head.weight is model.tok_emb.weight
    assert model.num_parameters(non_embedding=True) > 0
