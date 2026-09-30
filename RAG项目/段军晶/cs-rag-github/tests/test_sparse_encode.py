# -*- coding: utf-8 -*-
"""稀疏编码测试

注意：
    _to_sparse_dict 是纯函数，测试不加载模型，因此运行很快。
    端到端编码测试会真实加载 BGE-M3（约 2.3GB），标记为 slow。
"""

import pytest
import torch

from backend.embedder import _SPECIAL_TOKEN_IDS, _to_sparse_dict


def test_纯函数_忽略_padding_位置():
    # 注意：torch.tensor 默认 float32，与 Python float 字面量（float64）不可用 == 直接比较，
    # 因此数值一律用 pytest.approx。
    weights = torch.tensor([0.5, 0.3, 0.0, 0.9])
    input_ids = torch.tensor([5, 6, 1, 7])
    attention_mask = torch.tensor([1, 1, 0, 0])
    result = _to_sparse_dict(weights, input_ids, attention_mask)
    assert list(result.keys()) == [5, 6]
    assert result[5] == pytest.approx(0.5)
    assert result[6] == pytest.approx(0.3)


def test_纯函数_过滤特殊_token():
    # 0=<s> 1=<pad> 2=</s> 均应被排除
    weights = torch.tensor([0.9, 0.8, 0.7, 0.4])
    input_ids = torch.tensor([0, 1, 2, 42])
    attention_mask = torch.tensor([1, 1, 1, 1])
    result = _to_sparse_dict(weights, input_ids, attention_mask)
    assert list(result.keys()) == [42]
    assert result[42] == pytest.approx(0.4)
    assert not (_SPECIAL_TOKEN_IDS & set(result.keys()))


def test_纯函数_负权重归零():
    weights = torch.tensor([-0.5, 0.3])
    input_ids = torch.tensor([11, 12])
    attention_mask = torch.tensor([1, 1])
    result = _to_sparse_dict(weights, input_ids, attention_mask)
    assert 11 not in result
    assert list(result.keys()) == [12]
    assert result[12] == pytest.approx(0.3)


def test_纯函数_同_token_多次出现取最大权重():
    weights = torch.tensor([0.2, 0.7, 0.5])
    input_ids = torch.tensor([33, 33, 34])
    attention_mask = torch.tensor([1, 1, 1])
    result = _to_sparse_dict(weights, input_ids, attention_mask)
    assert result[33] == pytest.approx(0.7)


def test_纯函数_全空输入返回空字典():
    result = _to_sparse_dict(
        torch.tensor([0.0]), torch.tensor([0]), torch.tensor([0])
    )
    assert result == {}


@pytest.mark.slow
def test_端到端_稀疏编码非零维度数合理():
    """真实加载模型，验证输出是真正的稀疏结构且过滤了特殊 token"""
    from backend.embedder import get_embedder

    sparse = get_embedder().encode_sparse(["数据中心业务连续性分为几个等级？"])
    assert len(sparse) == 1
    vec = sparse[0]
    assert 0 < len(vec) < 64, "稀疏向量维度数异常"
    assert not (_SPECIAL_TOKEN_IDS & set(vec.keys()))
    assert all(w > 0 for w in vec.values())


@pytest.mark.slow
def test_端到端_手动稠密与_sentence_transformers_一致():
    """回归保护：证明一次 forward 的双输出方案不改变稠密向量"""
    import numpy as np

    from backend.embedder import get_embedder

    emb = get_embedder()
    texts = ["功能充分性属于哪个质量特性的子特性？"]
    dense_manual, _ = emb.encode_dense_and_sparse(texts)
    dense_st = emb.encode(texts, show_progress=False)
    cos = float(np.dot(dense_manual[0], dense_st[0]))
    assert cos > 0.9999, f"稠密向量不一致，余弦={cos}"
