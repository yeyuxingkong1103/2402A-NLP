# 编码模块单测。夹具用真实语料（law_chunks.jsonl 的真块），但 encoder 用替身：
# 加载 2.3GB 模型做单测不可接受。真实模型只在端到端与冒烟里出场。
import json
import pathlib
import sys
import types

import numpy as np
import pytest

from app.ingest.embed import (
    DEFAULT_MODEL_PATH, ENCODE_BATCH, build_milvus_row, encode_texts, load_model,
    normalize_dense, to_sparse_dict,
)

CHUNKS = pathlib.Path(__file__).resolve().parents[2] / "data" / "parsed" / "law_chunks.jsonl"


@pytest.fixture
def real_chunks():
    with open(CHUNKS, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()][:5]


def test_model_path_points_at_base_not_finetune():
    # bge-m3-ft-v1 的 sparse_linear.pt 缺失，sparse 头是随机初始化的，
    # 用它会让稀疏路灌噪声。这条测试守的是设计文档 4.3 的结论
    assert DEFAULT_MODEL_PATH.replace("\\", "/").endswith("/bge-m3")
    assert "ft" not in pathlib.Path(DEFAULT_MODEL_PATH).name


def test_to_sparse_dict_converts_keys_and_values():
    raw = {10: 0.5, 20: 0.25}
    assert to_sparse_dict(raw) == {10: 0.5, 20: 0.25}


def test_to_sparse_dict_drops_zero_weights():
    # 零权重项在稀疏倒排索引里占 posting 又永不命中，纯污染
    raw = {10: 0.5, 20: 0.0, 30: 0.25}
    assert to_sparse_dict(raw) == {10: 0.5, 30: 0.25}


def test_to_sparse_dict_raises_when_all_zero():
    # 全零说明这条文本的稀疏路彻底失效，属于数据异常而非正常情况，必须中断
    with pytest.raises(ValueError, match="稀疏"):
        to_sparse_dict({10: 0.0, 20: 0.0})


def test_normalize_dense_makes_unit_length():
    # 技术方案 4.3：dense 做 L2 归一化，保证 COSINE 与内积一致
    vecs = np.array([[3.0, 4.0], [0.0, 2.0]], dtype="float32")
    out = normalize_dense(vecs)
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0)
    assert np.allclose(out[0], [0.6, 0.8])


def test_normalize_dense_rejects_zero_vector():
    # 全零向量归一化会除零出 nan，nan 进了 Milvus 后续很难查，不如当场拦下
    with pytest.raises(ValueError, match="全零"):
        normalize_dense(np.array([[0.0, 0.0]], dtype="float32"))


class _FakeEncoder:
    """替身 encoder：回放预设好的输出，不碰 torch 与 2.3GB 权重。

    第 i 条文本的 dense 是 one-hot 第 i 维、sparse 取 token 100+i，两者都带下标。
    这样"第 i 条的结果被写到第 j 条上"会直接表现成断言失败——错位正是本模块
    最需要拦住的静默错误，用可辨认的输出才测得出它。
    """

    def __init__(self, rows: int | None = None):
        self.rows = rows

    def encode(self, texts, **kwargs):
        # 替身只认"一次前向同时要 dense 与 sparse"这一种调用：这是本模块存在的理由，
        # 少要哪一项后续都会 KeyError，不如在这里把调用契约钉死
        assert kwargs["return_dense"] and kwargs["return_sparse"]
        assert not kwargs["return_colbert_vecs"]
        assert kwargs["batch_size"] == ENCODE_BATCH
        n = len(texts) if self.rows is None else self.rows
        return {
            # 乘以 3 让替身的行**不是**单位长度：否则 encode_texts 里的 L2 归一化
            # 删掉也测不出来（单位矩阵的行本来就是单位长度），那条断言就成了恒真断言。
            # 乘完再归一化后正好还原成 one-hot，下标错位的检测能力不受影响。
            "dense_vecs": np.eye(n, dtype="float32") * 3.0,
            # 真实 FlagEmbedding 的权重是 numpy float32，替身照着来，
            # 免得漏掉"numpy 标量没转成 python float 就写进 Milvus"这类问题
            "lexical_weights": [{100 + i: np.float32(0.5 + i)} for i in range(n)],
        }


def test_encode_texts_pairs_each_text_with_its_own_vectors():
    # 调用方按位置拼装 dense/sparse，条数与顺序都必须与输入对齐
    out = encode_texts(_FakeEncoder(), ["甲", "乙", "丙"])
    assert len(out) == 3
    for i, (dense, sparse) in enumerate(out):
        assert isinstance(dense, list)
        assert np.allclose(dense, np.eye(3)[i])
        assert np.isclose(np.linalg.norm(dense), 1.0)
        assert sparse == {100 + i: 0.5 + i}
        # Milvus 的稀疏向量要 {int: float} 的 python 标量；真实模型的权重是 numpy
        # float32，漏了 float() 转出来的 np.float32 也 == 0.5 这样的比较会照样绿
        assert all(isinstance(k, int) and isinstance(v, float)
                   for k, v in sparse.items())


def test_encode_texts_raises_when_model_returns_fewer_vectors():
    # 条数不符说明后面按位置拼装必然错位——A 条的向量会写到 B 条上且不报错，
    # 硬门槛要求在编码这一步就中断
    with pytest.raises(ValueError, match="编码条数不符"):
        encode_texts(_FakeEncoder(rows=1), ["甲", "乙"])


def test_load_model_pins_cpu_and_fp16_off(monkeypatch):
    # 硬约束：本机 torch 是 CPU 版，fp16 在 CPU 上更慢。device/use_fp16 被改错
    # 不会报错，只会让 Task 5 的编码悄悄变慢或掉精度，故在此钉死
    captured = {}
    fake_module = types.ModuleType("FlagEmbedding")

    def fake_bgem3(path, use_fp16=None, device=None):
        captured.update(path=path, use_fp16=use_fp16, device=device)
        return "替身模型"

    fake_module.BGEM3FlagModel = fake_bgem3
    # 塞进 sys.modules 而不是 import FlagEmbedding：真身会拖起 torch，
    # 单测不该为一条签名断言付这个代价
    monkeypatch.setitem(sys.modules, "FlagEmbedding", fake_module)

    assert load_model() == "替身模型"
    assert captured == {"path": DEFAULT_MODEL_PATH, "use_fp16": False, "device": "cpu"}


def test_build_milvus_row_carries_metadata_from_mysql(real_chunks):
    # 这条守的是"块级字段来自语料、条级元数据来自 MySQL"的分工
    meta = {"law_id": "minfadian", "law_version": "v1", "status": "现行有效",
            "effective_date": "2021-01-01", "source_hash": "a" * 32,
            "article_no_cn": "一"}
    chunk = real_chunks[0]
    row = build_milvus_row(chunk, meta, [0.0] * 1023 + [1.0], {1: 0.5})
    assert row["chunk_id"] == chunk["chunk_id"]
    assert row["chunk_type"] == chunk["chunk_type"]
    assert row["article_no"] == chunk["article_no"]
    assert row["paragraph_no"] == chunk["paragraph_no"]
    assert row["parent_id"] == chunk["parent_id"]
    assert row["text"] == chunk["text"]
    assert row["law_id"] == "minfadian"
    assert row["status"] == "现行有效"
    assert row["effective_date"] == "2021-01-01"
    assert len(row["dense"]) == 1024


def test_build_milvus_row_outputs_every_schema_field(real_chunks):
    # 少一个字段 Milvus 会报错，多一个会因 enable_dynamic_field=False 报错——
    # 这条把 schema 与本模块的契约钉死
    from app.db.milvus import ALL_FIELDS

    meta = {"law_id": "minfadian", "law_version": "v1", "status": "现行有效",
            "effective_date": "2021-01-01", "source_hash": "a" * 32,
            "article_no_cn": "一"}
    row = build_milvus_row(real_chunks[0], meta, [1.0] * 1024, {1: 0.5})
    assert set(row) == set(ALL_FIELDS)


def test_encode_batch_is_positive():
    # 批大小写错（0 或负数）会让 FlagEmbedding 行为异常且报错难懂。
    # 只断 >0 的话把 16 改成 1 也照样绿，等于没钉住接口约定的值，故连值一起断
    assert ENCODE_BATCH == 16
