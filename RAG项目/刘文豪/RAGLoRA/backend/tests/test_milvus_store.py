# -*- coding: utf-8 -*-
"""Milvus 客户端的纯逻辑测试。

为什么不在这里做真实读写
------------------------
Milvus 是**可选后端**（需 Docker）。本机 Docker 可能未启动，把真实读写写进
默认测试套件会让整套测试随 Docker 状态时红时绿。因此：

  * 本文件只测**不依赖 Milvus 服务**的纯函数（格式转换、ID 映射）；
  * 真实端到端验证由 `scripts/migrate_to_milvus.py --verify` 与
    在线检索的 `both` 模式对比完成，见 docs/07-Milvus接入说明.md。

要跑需要 Milvus 的测试时，先 `bash tools/demo/up.sh milvus`。
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services import milvus_store  # noqa: E402


# ---------------------------------------------------------------- ID 映射
def test_to_milvus_id_handles_unsigned_overflow():
    """Qdrant ID 是 64 位无符号，Milvus INT64 有符号 —— 溢出值必须被映射。

    实测踩过：直接写会抛
    `DataNotMatchException: Value out of range: 9236543062734117978`。
    """
    overflow = 9236543062734117978          # > 2^63-1
    assert overflow > milvus_store._INT64_MAX

    mapped = milvus_store.to_milvus_id(overflow)
    assert 0 <= mapped <= milvus_store._INT64_MAX


def test_to_milvus_id_is_deterministic():
    """必须是确定性函数 —— 否则迁移无法重复执行（幂等性依赖于此）。"""
    raw = 12345678901234567890
    assert milvus_store.to_milvus_id(raw) == milvus_store.to_milvus_id(raw)


def test_to_milvus_id_preserves_values_within_range():
    """未溢出的 ID 不应被改动。"""
    small = 42
    assert milvus_store.to_milvus_id(small) == small


def test_to_milvus_id_no_collision_on_real_ids():
    """用真实量级的 ID 验证位掩码不引入碰撞。

    迁移脚本会在全量数据上实测校验，这里用一批伪 ID 做快速回归。
    """
    import hashlib
    ids = [
        int(hashlib.sha1(f"kb_legal|doc{i}.txt|第{i}条|正文{i}".encode()).hexdigest()[:16], 16)
        for i in range(3000)
    ]
    mapped = [milvus_store.to_milvus_id(i) for i in ids]
    assert len(set(mapped)) == len(ids), "位掩码引入了主键碰撞"


# ---------------------------------------------------------------- 稀疏向量转换
def test_to_milvus_sparse_from_dict():
    """在线检索路径：Qdrant 返回 dict。"""
    out = milvus_store._to_milvus_sparse({"indices": [1, 5], "values": [0.5, 0.25]})
    assert out == {1: 0.5, 5: 0.25}


def test_to_milvus_sparse_from_object():
    """迁移路径：scroll 返回 SparseVector 对象。

    实测踩过：只按 dict 写会抛 `TypeError: 'SparseVector' object is not
    subscriptable`，而在线检索看不出来 —— 所以两种形态都要覆盖。
    """
    class _SparseVector:                      # 模拟 qdrant_client 的返回类型
        indices = [2, 7]
        values = [0.75, 0.125]

    out = milvus_store._to_milvus_sparse(_SparseVector())
    assert out == {2: 0.75, 7: 0.125}


def test_to_milvus_sparse_empty():
    assert milvus_store._to_milvus_sparse({"indices": [], "values": []}) == {}


# ---------------------------------------------------------------- 行转换
def test_row_from_qdrant_maps_all_fields():
    """Qdrant payload -> Milvus 行：字段与截断都要对。"""
    payload = {
        "text": "高血压诊断标准",
        "source": "指南.pdf",
        "page": 4,
        "law_name": None,
        "article_no": None,
    }
    row = milvus_store.row_from_qdrant(payload, 2 ** 63 + 5, [0.1] * 4, {"indices": [1], "values": [0.9]})

    assert row["text"] == "高血压诊断标准"
    assert row["source"] == "指南.pdf"
    assert row["page"] == 4
    # 缺省字段要归一成空串/0，而不是 None —— Milvus 标量字段不接受 None
    assert row["law_name"] == ""
    assert row["article_no"] == ""
    assert row["summary"] == ""
    assert isinstance(row["created_at"], int)
    assert isinstance(row["updated_at"], int)
    assert 0 <= row["id"] <= milvus_store._INT64_MAX


def test_row_from_qdrant_truncates_long_text():
    """超长文本必须按 schema 的 max_length 截断，否则写入会被服务端拒绝。"""
    payload = {"text": "x" * 20000, "source": "s" * 5000}
    row = milvus_store.row_from_qdrant(payload, 1, [0.0], {"indices": [], "values": []})
    assert len(row["text"]) == milvus_store._MAX_LEN["text"]
    assert len(row["source"]) == milvus_store._MAX_LEN["source"]


# ---------------------------------------------------------------- 探活降级
def test_health_reports_failure_gracefully():
    """Milvus 不可达时应返回 ok=False 而不是抛异常 —— 健康检查不能因此崩掉。"""
    orig = milvus_store._client

    class _Boom:
        def list_collections(self):
            raise RuntimeError("连接被拒绝")

    milvus_store._client = _Boom()
    try:
        res = milvus_store.health()
        assert res["ok"] is False
        assert "连接被拒绝" in res["error"]
    finally:
        milvus_store._client = orig
