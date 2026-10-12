# 工单编号：人工智能NLP-RAG-Query理解优化
#          （工单 PDF 原文：人工智能NLP-RAG-Query 理解优化任务）
"""F1 回归：主索引重建后必须补回组织结构图块；补不回必须醒目告警。

背景：build_index.py 每次都 `ensure_collection(recreate=True)`，会把 Task 7 入库的
30 个 orgchart 块一并删掉。若既不自动补入、也不告警，Q5（哪个销售部的销售处最多）
在检索侧会**静默**失效——不报错、不降级，只是答不出来，直到有人碰巧重跑
`python build_orgchart.py --rebuild`。

  快测（默认 `-m "not slow"` 门禁内）：
    · `_embed_and_upsert()` 的调用序必须是 recreate → upsert 主索引 → orgchart 补入；
    · 补入失败（抛异常 / 返回非 0）不得中断建库，必须给出含恢复命令的告警。
  慢测（`-m slow`，需真 embedder + 本地 Qdrant）：走同一条链路端到端确认
  30 个 orgchart 点确实回到 collection。
"""
import os

import pytest

import build_index
import build_orgchart
from src import embedder, vector_store

# 与其它测试同口径的最小正文块（vector_store._to_point 需要的字段都在）
CHUNK = {"chunk_id": "c000001", "text": "注册资本为7,360.00万元", "parent_id": "s1",
         "parent_text": "第三节 公司基本情况……注册资本为7,360.00万元……",
         "parent_page_idx": 10, "page_idx": 10, "heading_path": ["第三节"],
         "source": "招股说明书1.pdf", "block_type": "text", "seq": 0}


def _stub_deps(monkeypatch, order: list, ingest) -> None:
    """把 embedder / vector_store / build_orgchart.ingest_orgchart 换成记录器。"""
    monkeypatch.setattr(embedder, "get_dim", lambda: 4)
    monkeypatch.setattr(embedder, "embed_texts",
                        lambda texts, **kw: [[1.0, 0.0, 0.0, 0.0] for _ in texts])
    monkeypatch.setattr(vector_store, "ensure_collection",
                        lambda **kw: order.append(("recreate", kw.get("recreate"))))
    monkeypatch.setattr(vector_store, "upsert_chunks",
                        lambda chunks, vectors, **kw: (order.append(("upsert", len(chunks))),
                                                       len(chunks))[1])
    monkeypatch.setattr(build_orgchart, "ingest_orgchart", ingest)


def test_embed_and_upsert_chains_orgchart_restore(monkeypatch):
    """recreate 之后必须紧接着补入 orgchart 块（F1 的核心接线）。"""
    order: list = []
    _stub_deps(monkeypatch, order,
               lambda *a, **kw: order.append(("orgchart", kw)) or 0)

    build_index._embed_and_upsert([dict(CHUNK)])

    assert [k for k, _ in order] == ["recreate", "upsert", "orgchart"], (
        "调用序必须是 recreate → 主索引 → orgchart 补入；少了最后一步，"
        "重建一次索引 Q5 就会静默答不出来")
    assert order[0] == ("recreate", True), "recreate 建库正是会删掉 orgchart 点的原因"
    assert order[-1][1]["rebuild"] is True


def test_restore_reports_success(monkeypatch, capsys):
    order: list = []
    _stub_deps(monkeypatch, order, lambda *a, **kw: 0)

    assert build_index._restore_orgchart() is True
    assert "组织结构图块已自动补入" in capsys.readouterr().out


@pytest.mark.parametrize("failure", ["raise", "rc2"])
def test_restore_failure_warns_instead_of_breaking_build(monkeypatch, capsys, failure):
    """补入失败不得拖垮建库：主索引照常写入，stderr 给出可执行的恢复命令。"""
    def ingest(*a, **kw):
        if failure == "raise":
            raise RuntimeError("模拟 embedder 崩溃")
        return 2                       # 结构闸门不过（build_orgchart 的退出码）

    order: list = []
    _stub_deps(monkeypatch, order, ingest)

    build_index._embed_and_upsert([dict(CHUNK)])      # 不得抛异常

    assert ("upsert", 1) in order, "主索引必须照常写入"
    err = capsys.readouterr().err
    assert "build_orgchart.py --rebuild" in err, "告警必须给出恢复命令"
    assert "组织结构图" in err and "Q5" in err, "告警要说明后果，不能只报错"


# ---------------------------------------------------------------------------
# 慢测：真 embedder + 临时本地 Qdrant，端到端确认点确实回来了
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_real_build_restores_orgchart_points(tmp_path, monkeypatch):
    """recreate 清空 → 走真实的 `_embed_and_upsert()` 链路 → 30 个 orgchart 点回来。

    只重放「主索引 1 点 + orgchart 30 点」的量级（不是全量语料），但用的是
    build_index.py 真实的调用序列与真 bge-m3 向量、真实 PDF 解析。
    """
    from src import config

    monkeypatch.setattr(config, "QDRANT_LOCAL_PATH", str(tmp_path))
    monkeypatch.setattr(config, "QDRANT_MODE", "local")
    monkeypatch.setattr(config, "TABLE_DIR", str(tmp_path / "tables"))
    os.makedirs(config.TABLE_DIR, exist_ok=True)
    vector_store.reset_client_for_tests()
    try:
        build_index._embed_and_upsert([dict(CHUNK)])
        assert vector_store.count() == 31, "1 主索引点 + 30 orgchart 点"
        org = [r for r in vector_store.scroll_all()
               if r.get("block_type") == "orgchart"]
        assert len(org) == 30
        assert {r.get("org_node") for r in org} >= {"大客户销售部", "销售部"}
        assert any(str(r.get("org_path", "")).endswith("销售部/大客户销售部")
                   for r in org), "节点路径要经 Qdrant 往返而不丢"
    finally:
        vector_store.reset_client_for_tests()
