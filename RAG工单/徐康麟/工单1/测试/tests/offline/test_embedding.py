"""离线测试：向量化（``app/core/embedder.py``）。

测试目标（工单 9.1 / 5.3）：
1. 向量维度 > 0 且跨调用一致；
2. **确定性**：同一文本两次编码结果完全相同（索引可复现的前提）；
3. 批量编码与单条编码结果一致（接口一致性）；
4. 向量已做 L2 归一化，可直接用点积当余弦相似度；
5. 相同文本相似度为 1，不同文本相似度明显更低；
6. 降级路径（无本地模型时的哈希向量）必须给出明确原因，禁止静默降级。

说明：本机未下载 BGE 模型，实际运行会走哈希降级后端；用例对两种后端都成立。
"""

from __future__ import annotations

import numpy as np
import pytest

from app.core.embedder import Embedder, cosine_similarity, get_embedder

TEXT_A = "报告期内，公司来自军用领域的收入分别为6,464.51万元、14,414.16万元。"
TEXT_B = "公司参与制定了全军第一个视频指挥系统技术标准。"
TEXT_C = "红烧肉的家常做法需要准备五花肉、冰糖和生抽。"


@pytest.fixture(scope="module")
def embedder() -> Embedder:
    """进程级嵌入器（与运行时同一实例，保证测的就是真实链路）。"""
    return get_embedder()


# --------------------------------------------------------------------------
# 维度
# --------------------------------------------------------------------------
def test_dimension_is_positive_and_stable(embedder) -> None:
    """向量维度必须大于 0，且与实际编码结果的列数一致。"""
    assert embedder.dimension > 0, f"嵌入维度必须为正数，实际为 {embedder.dimension}"
    matrix = embedder.encode([TEXT_A, TEXT_B])
    assert matrix.ndim == 2, f"批量编码应返回二维矩阵，实际维度为 {matrix.ndim}"
    assert matrix.shape == (2, embedder.dimension), (
        f"编码结果形状 {matrix.shape} 与声明的维度 {embedder.dimension} 不一致"
    )

    single = embedder.encode_one(TEXT_A)
    assert single.shape == (embedder.dimension,), (
        f"单条编码形状 {single.shape} 与维度 {embedder.dimension} 不一致"
    )
    assert embedder.encode([TEXT_B]).shape == (1, embedder.dimension), "批量编码的维度不稳定"


def test_dimension_consistent_across_batches(embedder) -> None:
    """不同批次的向量维度必须完全一致，否则向量库无法拼接。"""
    first = embedder.encode([TEXT_A])
    second = embedder.encode([TEXT_A, TEXT_B, TEXT_C])
    assert first.shape[1] == second.shape[1], (
        f"两次编码的维度不一致：{first.shape[1]} vs {second.shape[1]}，向量库将无法写入"
    )


# --------------------------------------------------------------------------
# 确定性与一致性
# --------------------------------------------------------------------------
def test_same_text_encodes_identically(embedder) -> None:
    """同一文本两次编码必须逐位相同（哈希后端零随机性；模型后端亦应稳定）。"""
    first = embedder.encode([TEXT_A])[0]
    second = embedder.encode([TEXT_A])[0]
    assert np.array_equal(first, second), "同一文本两次编码结果不同，索引无法复现"

    one_again = embedder.encode_one(TEXT_A)
    assert np.allclose(first, one_again, atol=1e-6), "encode 与 encode_one 的结果不一致"


def test_batch_matches_single_encoding(embedder) -> None:
    """批量编码的第 i 行必须与单独编码第 i 条文本的结果一致。"""
    texts = [TEXT_A, TEXT_B, TEXT_C]
    matrix = embedder.encode(texts)
    for index, text in enumerate(texts):
        single = embedder.encode_one(text)
        assert np.allclose(matrix[index], single, atol=1e-5), (
            f"第 {index} 条文本的批量向量与单条向量不一致（最大偏差 "
            f"{float(np.max(np.abs(matrix[index] - single))):.2e}）"
        )


# --------------------------------------------------------------------------
# 归一化与相似度
# --------------------------------------------------------------------------
def test_vectors_are_l2_normalized(embedder) -> None:
    """向量必须已归一化，检索才能用点积直接当余弦相似度。"""
    matrix = embedder.encode([TEXT_A, TEXT_B, TEXT_C])
    norms = np.linalg.norm(matrix, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4), f"存在未归一化的向量，范数为 {norms.tolist()}"


def test_identical_text_similarity_is_one(embedder) -> None:
    """相同文本的余弦相似度必须为 1。"""
    similarity = cosine_similarity(embedder.encode_one(TEXT_A), embedder.encode_one(TEXT_A))
    assert abs(similarity - 1.0) < 1e-5, f"相同文本的余弦相似度应为 1，实际为 {similarity}"


def test_different_text_similarity_is_lower(embedder) -> None:
    """不同文本的相似度必须明显低于相同文本（否则检索无区分度）。"""
    same = cosine_similarity(embedder.encode_one(TEXT_A), embedder.encode_one(TEXT_A))
    different = cosine_similarity(embedder.encode_one(TEXT_A), embedder.encode_one(TEXT_C))
    assert different < same - 1e-3, (
        f"无关文本的相似度({different:.4f})未低于相同文本({same:.4f})，向量检索已失去区分能力"
    )


def test_encode_empty_input_returns_empty_matrix(embedder) -> None:
    """空输入必须返回 (0, dim) 的空矩阵，而不是抛异常。"""
    matrix = embedder.encode([])
    assert matrix.shape == (0, embedder.dimension), f"空输入应返回 (0, {embedder.dimension})，实际 {matrix.shape}"


# --------------------------------------------------------------------------
# 健康信息与降级说明
# --------------------------------------------------------------------------
def test_health_reports_backend_and_degradation(embedder) -> None:
    """health() 必须说明后端类型、维度与降级原因（禁止静默降级）。"""
    health = embedder.health()
    for key in ("backend", "dimension", "semantic", "degraded_reason", "model"):
        assert key in health, f"health() 缺少字段 {key}"

    assert isinstance(health["semantic"], bool), "health()['semantic'] 必须是布尔值"
    assert health["dimension"] == embedder.dimension, "health() 中的维度与实际维度不一致"
    assert health["backend"] == embedder.name, "health() 中的后端名与实际后端不一致"

    if not health["semantic"]:
        assert str(health["degraded_reason"]).strip(), (
            "降级为哈希向量时必须在 degraded_reason 中说明原因，禁止静默降级"
        )
