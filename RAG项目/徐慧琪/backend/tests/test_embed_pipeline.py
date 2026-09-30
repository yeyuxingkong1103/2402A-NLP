# 端到端组装逻辑的单测。encoder 仍是替身——这里测的是"元数据拼得对不对"，
# 不是模型好不好；真实模型在 Step 5 的实跑里出场。
import json
import pathlib

import pytest

from app.ingest.embed import article_meta, build_rows, parse_item_no

CHUNKS = pathlib.Path(__file__).resolve().parents[2] / "data" / "parsed" / "law_chunks.jsonl"


@pytest.fixture
def real_chunks():
    # 前 6 块全是 father/paragraph（实测：首个 item 块在第 60 位），只取前 6 块的话
    # 下面「item_no 只给 item 块」的 item 分支永远走不到，断言就成了恒真断言——
    # 实测把 item_no 改成恒 None 仍是 7 passed，正是本项目返工多次的坑。
    # 故再带上 2 个真实 item 块（不造数据，仍取自语料）
    with open(CHUNKS, encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f if line.strip()]
    items = [c for c in chunks if c["chunk_type"] == "item"][:2]
    return chunks[:6] + items


class FakeModel:
    """替身 encoder：返回固定形状的假向量，只验证组装链路。"""

    def encode(self, texts, batch_size=None, return_dense=True,
               return_sparse=True, return_colbert_vecs=False):
        return {"dense_vecs": [[0.0] * 1023 + [1.0]] * len(texts),
                "lexical_weights": [{1: 0.5, 2: 0.25}] * len(texts)}


def test_parse_item_no_reads_fullwidth_bracket():
    # 民法典用全角括号，但语料里也可能混半角，两种都要认
    assert parse_item_no("（一）无民事行为能力人实施的民事法律行为无效。") == "一"
    assert parse_item_no("(十二)其他情形。") == "十二"


def test_parse_item_no_returns_none_for_plain_text():
    # 普通正文不能误判成项——误判会让 item_no 记录一个错误编号
    assert parse_item_no("第一条为了保护民事主体的合法权益。") is None
    assert parse_item_no("婚姻、收养、监护等有关身份关系的协议。") is None


def test_parse_item_no_ignores_parenthesised_non_numerals():
    # 「（以下称）」这类不是项号，不该被当成项
    assert parse_item_no("（以下称债务人）应当履行。") is None


@pytest.mark.skipif(not pathlib.Path(CHUNKS).exists(), reason="语料不存在")
def test_article_meta_covers_all_1260_articles():
    from app.db.mysql import connect

    try:
        conn = connect()
    except Exception:
        pytest.skip("MySQL 未在线")
    try:
        meta = article_meta(conn)
    finally:
        conn.close()
    assert len(meta) == 1260
    sample = meta[1]
    assert sample["law_id"] == "minfadian"
    assert sample["status"] == "现行有效"
    assert sample["effective_date"] == "2021-01-01"
    assert sample["article_no_cn"] == "一"
    assert len(sample["source_hash"]) == 32
    # law_version 是 6 个键里唯一此前没被断言过的——article_meta 把它漏掉/改名时，
    # 其余 5 个键的断言照样通过，直到 main() 在第一块 build_milvus_row 处才 KeyError
    assert sample["law_version"] == "v1"


def test_build_rows_uses_metadata_from_mysql(real_chunks):
    # 元数据按 article_no 关联：块的条号决定它取哪一条元数据，
    # 关联错了会把 A 条的效力/版本装到 B 条上
    meta_by_no = {}
    for c in real_chunks:
        meta_by_no.setdefault(c["article_no"], {
            "law_id": "minfadian", "law_version": "v1", "status": "现行有效",
            "effective_date": "2021-01-01", "source_hash": f"h{c['article_no']}",
            "article_no_cn": str(c["article_no"]),
        })
    rows = build_rows(real_chunks, meta_by_no, FakeModel())
    assert len(rows) == len(real_chunks)
    for row, chunk in zip(rows, real_chunks):
        assert row["chunk_id"] == chunk["chunk_id"]
        assert row["source_hash"] == f"h{chunk['article_no']}"


@pytest.mark.skipif(not pathlib.Path(CHUNKS).exists(), reason="语料不存在")
def test_build_rows_accepts_real_article_meta(real_chunks):
    # 现有 build_rows 测试喂的是手写 dict，等于绕过了 article_meta——
    # 这两个函数的键名一旦对不上，只有接缝测试能发现
    from app.db.mysql import connect
    from app.ingest.embed import article_meta
    try:
        conn = connect()
    except Exception:
        pytest.skip("MySQL 未在线")
    try:
        meta = article_meta(conn)
    finally:
        conn.close()
    rows = build_rows(real_chunks, meta, FakeModel())
    assert len(rows) == len(real_chunks)
    # 逐块核对条级元数据确实来自它自己那条法条
    for row, chunk in zip(rows, real_chunks):
        assert row["article_no"] == chunk["article_no"]
        assert row["law_version"] == "v1"


def test_build_rows_raises_when_metadata_missing(real_chunks):
    # 元数据缺失必须中断：技术方案 4.2 的字段空了，检索过滤会静默失效
    with pytest.raises(KeyError):
        build_rows(real_chunks, {}, FakeModel())


def test_build_rows_assigns_item_no_only_to_item_chunks(real_chunks):
    # item_no 只对 chunk_type == "item" 有意义，父块与款块恒为 None
    meta_by_no = {c["article_no"]: {
        "law_id": "minfadian", "law_version": "v1", "status": "现行有效",
        "effective_date": "2021-01-01", "source_hash": "h", "article_no_cn": "x",
    } for c in real_chunks}
    items = [c for c in real_chunks if c["chunk_type"] == "item"]
    rows = build_rows(real_chunks, meta_by_no, FakeModel())
    for row, chunk in zip(rows, real_chunks):
        if chunk["chunk_type"] == "item" and items:
            assert row["item_no"] == parse_item_no(chunk["text"])
        else:
            assert row["item_no"] is None
