# 工单编号：人工智能NLP-RAG 项目-LightRAG 优化
"""t4 用例：路由分流 / 双层检索 / RRF 融合 / QA 后端 / 评估产物 / 截图（离线）。

说明：**嵌入向量用桩替代**（`embed_query` 返回定长零向量），检索链路本身真实
—— 真块表（两份去水印文档）、真图谱（291 节点 / 167 边）、真 RRF 融合，
因此本文件不触发任何 Ollama 调用，整套用例秒级完成。
对应验收点：设计\\验收标准.md 2.2/2.3/2.4/2.5/2.7 + S2（截图中文命名）+ S3（真实数据）。
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pytest
from PIL import Image

from lightrag.double_retrieval import (DEFAULT_DOC_FILTER, expand_graph,
                                       retrieve_lightrag, retrieve_rag, rrf_fuse)
from lightrag.graph_store import build_edges, merge_entities
from lightrag.qa import QAEngine
from lightrag.routing import Route, build_entity_dict, route_question

REPO_ROOT = Path(__file__).resolve().parents[1]
GRAPH_DIR = REPO_ROOT / "研发" / "data" / "graph"
SHOT_DIR = REPO_ROOT / "测试" / "测试截图"
EVAL_JSONL = REPO_ROOT / "研发" / "评估" / "逐题结果.jsonl"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
EXPECTED_IDS = {1, 2, 3, 4, 5, 6, 33, 34, 95, 207, 260, 531, 543, 793, 795, 957}
# 命名规范文档里举的**反面例子**不是真实引用
BANNED_EXAMPLES = {"1.png", "2.png", "3.png", "img.png", "screenshot.png", "result.png",
                   "chart1.png", "fig_01.png", "output.png", "pic.png"}


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


class _FakeVectorIndex:
    """假向量索引：按 ids 顺序给分（真索引只影响排序，不影响结构断言）。"""

    def __init__(self, ids: list[str]) -> None:
        self.ids = list(ids)

    def search(self, vec, top_k: int):
        return [(cid, 1.0 - i * 1e-4) for i, cid in enumerate(self.ids[: int(top_k)])]


class _FakeBM25:
    """假 BM25：结构同真接口（search / size）。"""

    def __init__(self, ids: list[str]) -> None:
        self.ids = list(ids)
        self.size = len(self.ids)

    def search(self, query, top_k: int):
        return [(cid, 0.5) for cid in self.ids[: int(top_k)]]


class _StubClient:
    """桩客户端：只提供 embed_query，确保用例不调 Ollama。"""

    def embed_query(self, question: str):
        return [0.0] * 8


# ---------------------------------------------------------------- 路由分流 --
def test_路由_局部与全局两类关键词都能分出来(cfg, graph_nodes):
    entity_dict = build_entity_dict(graph_nodes,
                                    min_len=int(cfg.routing.min_entity_len),
                                    max_len=int(cfg.routing.max_entity_len))
    assert entity_dict, "图谱实体词典为空"

    # ① 真实 16 题：每题都能分出至少一类关键词（不许「分不出」）
    questions = _read_jsonl(REPO_ROOT / "研发" / "评估" / "测试问题集.jsonl")
    routes = [route_question(q["question"], entity_dict, cfg) for q in questions]
    assert len(routes) == 16
    for r in routes:
        assert r.route in ("local", "global", "both"), f"route 取值非法：{r.route}"
        assert r.local_keywords or r.global_keywords, f"分不出任何关键词：{r.question}"
    assert any(r.local_keywords for r in routes), "应有题命中局部关键词"
    assert any(r.global_keywords for r in routes), "应有题命中全局关键词"

    # ② 具体实体 + 数值型问法 → 局部
    concrete = "发行人" if "发行人" in entity_dict else sorted(entity_dict)[0]
    r1 = route_question(f"{concrete}的注册资本是多少？", entity_dict, cfg)
    assert concrete in r1.matched_entities, "图谱实体未被命中"
    assert r1.local_keywords, "具体实体问题应产出局部关键词"
    assert r1.route in ("local", "both")

    # ③ 抽象概念型问法 → 全局
    r2 = route_question("公司所处行业的竞争格局与发展前景如何？", entity_dict, cfg)
    assert r2.global_keywords, "抽象问题应产出全局关键词"
    assert set(r2.global_keywords) >= {"竞争格局", "发展前景"}
    assert not r2.local_keywords
    assert r2.route == "global"


def test_路由_未命中关键词时按配置兜底(cfg):
    r = route_question("你好", {}, cfg)
    assert r.local_keywords == [] and r.global_keywords == []
    assert r.route == str(cfg.routing.route_fallback)


# ------------------------------------------------------------- RRF 融合 --
def test_RRF融合_名次与通道数单调(cfg):
    k = int(cfg.retrieval.rrf_k)
    one = [("x", 9.9), ("y", 8.8), ("z", 7.7)]
    fused1 = rrf_fuse([one], [1.0], k=k)
    assert fused1["x"] > fused1["y"] > fused1["z"] > 0, "同名次通道内分数应随名次单调递减"

    # 权重加倍 → 分数加倍（线性）
    fused_w = rrf_fuse([one], [2.0], k=k)
    assert fused_w["x"] == pytest.approx(fused1["x"] * 2, rel=1e-9)

    # k 越大 → 单人份分数越小（单调）
    assert rrf_fuse([one], [1.0], k=k + 60)["x"] < fused1["x"]

    # 被两条通道同时命中的文档，分数高于只被一条命中的同档文档
    fused2 = rrf_fuse([one, [("y", 1.0)]], [1.0, 1.0], k=k)
    assert fused2["y"] > fused2["x"] and fused2["y"] > fused1["y"]


# --------------------------------------------------------- 双层检索结构 --
def test_双层检索_RAG与LightRAG返回结构含页码与来源(cfg, chunks_by_id, graph_nodes, graph_edges):
    ids = list(chunks_by_id)
    vindex, bm25, embed = _FakeVectorIndex(ids), _FakeBM25(ids), _StubClient().embed_query
    question = "武汉力源信息技术股份有限公司本次发行股数是多少？"

    rag = retrieve_rag(question, cfg=cfg, vindex=vindex, bm25=bm25, embed_query=embed,
                       chunks_by_id=chunks_by_id, doc_filter=DEFAULT_DOC_FILTER)
    assert rag.backend == "rag" and rag.hits
    assert rag.trace["fused_top"] and rag.trace["doc_filter"] == list(DEFAULT_DOC_FILTER)

    route = Route(question=question, route="both", local_keywords=["发行人"],
                  global_keywords=["所属行业"], matched_entities=["发行人"])
    lr = retrieve_lightrag(question, route, cfg=cfg, vindex=vindex, bm25=bm25, embed_query=embed,
                           chunks_by_id=chunks_by_id, nodes=graph_nodes, edges=graph_edges,
                           edge_mode="all", doc_filter=DEFAULT_DOC_FILTER)
    assert lr.backend == "lightrag" and lr.hits
    assert lr.trace["route"] == "both" and lr.trace["edge_mode"] == "all"

    for res in (rag, lr):
        for hit in res.hits:
            row = hit.to_dict()
            assert row["chunk_id"] in chunks_by_id, "命中块必须能在块表中追溯"
            assert row["page"] > 0, "命中必须带页码"
            assert row["source"] in ("rag_vector", "rag_bm25", "rag_fused",
                                     "lightrag_local", "lightrag_global", "lightrag_fused"), \
                f"来源通道取值非法：{row['source']}"
            assert row["doc_key"] in DEFAULT_DOC_FILTER, "命中不得来自带水印重复件"


def test_双层检索_两组边口径都能读且相容组更窄(cfg, graph_nodes, graph_edges):
    route = Route(question="发行人股权结构", route="global",
                  global_keywords=["发行人"], matched_entities=["发行人"])
    rank_all, trace_all = expand_graph(route, nodes=graph_nodes, edges=graph_edges,
                                       cfg=cfg, edge_mode="all")
    rank_strict, trace_strict = expand_graph(route, nodes=graph_nodes, edges=graph_edges,
                                             cfg=cfg, edge_mode="constraint_ok")
    assert rank_all and rank_strict, "两种边口径都应能抽出关联块"
    assert set(c for c, _ in rank_strict) <= set(c for c, _ in rank_all), \
        "仅相容边口径的候选块应是全量边口径的子集（只做减法）"
    assert trace_strict["edges_touched"] <= trace_all["edges_touched"]

    # 与图谱侧统计口径一致：46 相容 / 121 标记
    stats = json.loads((GRAPH_DIR / "graph_stats.json").read_text(encoding="utf-8"))
    assert sum(1 for e in graph_edges if e.get("constraint_ok")) == stats["edges_constraint_ok"]
    assert sum(1 for e in graph_edges if not e.get("constraint_ok")) == stats["edges_constraint_flagged"]


# ------------------------------------------------------------ QA 后端 --
def test_QA后端_rag与lightrag切换不报错(cfg, chunks_by_id, graph_nodes, monkeypatch):
    import lightrag.vector_index as vi

    ids = list(chunks_by_id)
    monkeypatch.setattr(vi, "load_index",
                        lambda cfg, logger=None: (_FakeVectorIndex(ids), _FakeBM25(ids)))
    engine = QAEngine(cfg, client=_StubClient())          # 桩客户端 → 不调 Ollama
    info = engine.load()                                   # 真实读块表与图谱
    assert info["graph_nodes"] == len(graph_nodes)
    # QAEngine 读全量 3013 块（检索侧再按 doc_filter 过滤），故 ≥ 夹具里的去水印子集
    assert info["chunks_loaded"] >= len(chunks_by_id)
    assert info["index_chunks"] > 0 and info["bm25_docs"] > 0

    question = "本次募集资金拟投资哪些项目？"
    results = [engine.answer(question, backend=b, generate=False)   # generate=False → 不调 LLM
               for b in ("rag", "lightrag", "lightrag_strict")]
    assert [r.backend for r in results] == ["rag", "lightrag", "lightrag_strict"]
    for res in results:
        assert res.error == "", f"{res.backend} 报错：{res.error}"
        assert res.hits, f"{res.backend} 无命中"
        assert all(h["page"] > 0 and h["source"] for h in res.hits)
        assert res.sources and all(s["chunk_id"] and s["page"] > 0 for s in res.sources)
        assert res.retrieve_s >= 0

    with pytest.raises(ValueError):
        engine.retrieve(question, backend="不存在后端")


# ------------------------------------------------------------ 评估产物 --
def test_RAGAS指标_json字段完整且口径透明():
    data = json.loads((REPO_ROOT / "优化" / "评估结果" / "RAGAS指标.json").read_text(encoding="utf-8"))
    assert data["work_order"] == "人工智能NLP-RAG 项目-LightRAG 优化"
    assert data["官方ragas可用"] is False, "官方 ragas 缺失必须如实标注"
    assert "ModuleNotFoundError" in data["官方ragas探测结果"]
    assert "自实现" in data["口径差异说明"], "降级口径必须写明与官方的差异"
    assert len(data["逐题"]) == 48, "逐题明细应为 16 题 × 3 后端"
    for backend in ("rag", "lightrag", "lightrag_strict"):
        row = data["汇总"][backend]
        assert row["题数"] == 16
        for metric in ("faithfulness_kw", "answer_relevancy_kw",
                       "context_precision_kw", "context_recall_kw"):
            assert metric in row, f"{backend} 缺指标 {metric}"
            assert 0.0 <= float(row[metric]) <= 1.0


def test_对比汇总_json三后端齐全且含对比元信息():
    data = json.loads((REPO_ROOT / "优化" / "评估结果" / "对比汇总.json").read_text(encoding="utf-8"))
    for backend in ("rag", "lightrag", "lightrag_strict"):
        row = data[backend]
        assert row["题数"] == 16
        for field in ("答对数", "答对率", "平均答案关键词命中率", "页命中数",
                      "平均检索耗时_s", "平均总耗时_s"):
            assert field in row, f"{backend} 缺字段 {field}"
    meta = data["_meta"]
    assert meta["题数"] == 16 and meta["生成模型"] and meta["top_k"] == 8
    assert meta["页号容差"] == 1, "±1 页容差必须写在元信息里"


def test_逐题结果_48行且命中可追溯():
    rows = _read_jsonl(EVAL_JSONL)
    assert len(rows) == 48, f"逐题结果应为 16 题 × 3 后端 = 48 行，实际 {len(rows)}"
    assert Counter(r["backend"] for r in rows) == {"rag": 16, "lightrag": 16, "lightrag_strict": 16}
    assert {int(r["id"]) for r in rows} == EXPECTED_IDS
    for row in rows:
        for field in ("question", "answer", "hits", "sources", "score",
                      "retrieve_s", "generate_s", "error"):
            assert field in row, f"id={row['id']} 缺字段 {field}"
        assert row["hits"], f"id={row['id']} {row['backend']} 无命中"
        assert all(h["page"] > 0 and h["source"] for h in row["hits"])
        assert row["error"] == "", f"id={row['id']} 生成报错：{row['error']}"


def test_派生视图_逐题结果按后端拆分且与源文件一致():
    """`设计\\验收标准.md` 2.3 的证据路径是 优化\\评估结果\\逐题结果_<后端>.jsonl，
    这里校验它们是 48 行源文件按 backend 拆出的派生视图（口径逐行一致，非另起炉灶）。"""
    src = _read_jsonl(EVAL_JSONL)
    for backend, name in (("rag", "逐题结果_rag.jsonl"), ("lightrag", "逐题结果_lightrag.jsonl")):
        rows = _read_jsonl(REPO_ROOT / "优化" / "评估结果" / name)
        expect = [r for r in src if r["backend"] == backend]
        assert len(rows) == 16, f"{name} 应为 16 行，实际 {len(rows)}"
        assert all(r["backend"] == backend for r in rows)
        assert [r["id"] for r in rows] == [r["id"] for r in expect], f"{name} 题序与源文件不一致"
        for got, want in zip(rows, expect):
            assert got["question"] == want["question"]
            assert len(got["hits"]) == len(want["hits"])


# ------------------------------------------------------------ 截图与联动 --
def test_截图_中文命名且非空图():
    shots = sorted(SHOT_DIR.glob("*.png"))
    assert len(shots) >= 3, f"测试截图应 ≥3 张，实际 {len(shots)} 张"
    for path in shots:
        assert re.search(r"[\u4e00-\u9fff]", path.stem), f"截图名必须用中文按测试内容命名：{path.name}"
        assert not re.fullmatch(r"[\d\-_]+", path.stem), f"禁止纯数字/符号名：{path.name}"
        data = path.read_bytes()
        assert data[:8] == PNG_MAGIC, f"不是合法 PNG：{path.name}"
        assert len(data) > 5000, f"文件过小疑似空图：{path.name}（{len(data)} B）"
        with Image.open(path) as im:
            assert im.width >= 200 and im.height >= 200, f"分辨率过小：{path.name}"
            low, high = im.convert("L").getextrema()
            assert high > low, f"纯色空图：{path.name}"


def test_联动_文档引用的截图文件名与磁盘完全一致():
    on_disk = {p.name for p in SHOT_DIR.glob("*.png")}
    missing: list[str] = []
    for sub in ("测试", "优化", "部署", "设计"):
        for md in sorted((REPO_ROOT / sub).glob("*.md")):
            for m in re.finditer(r"[\u4e00-\u9fff\w\-]+\.png", md.read_text(encoding="utf-8")):
                name = m.group(0)
                if name in BANNED_EXAMPLES:
                    continue
                if name not in on_disk:
                    missing.append(f"{sub}/{md.name} → {name}")
    assert not missing, "文档引用了磁盘上不存在的截图：" + "；".join(missing)


def test_图谱_别名归并与关系去重(cfg):
    records = [
        {"chunk_id": "c1", "page": 1, "doc_key": "d",
         "entities": [{"name": "本公司", "type": "公司", "type_exact": 1},
                      {"name": "发行人", "type": "公司", "type_exact": 1}],
         "relations": [{"head": "本公司", "rel": "投资", "tail": "甲公司",
                        "tail_type": "公司", "confidence": 0.9}]},
        {"chunk_id": "c2", "page": 2, "doc_key": "d",
         "entities": [{"name": "发行人", "type": "公司", "type_exact": 1}],
         "relations": [{"head": "发行人", "rel": "投资", "tail": "甲公司",
                        "tail_type": "公司", "confidence": 0.9}]},
    ]
    nodes, bag = merge_entities(records, cfg)
    assert {n["name"] for n in nodes} == {"发行人"}, "「本公司」应归并到规范名「发行人」"
    node = nodes[0]
    assert node["freq"] == 3, "三次实体提及应聚合到同一节点"
    assert set(node["raw_names"]) >= {"发行人", "本公司"}, "别名原文应保留可追溯"
    assert node["chunk_ids"] == ["c1", "c2"] and node["pages"] == [1, 2]
    assert bag["meta"]["nodes"] == 1 and bag["meta"]["type_conflicts"] == 0

    edges, estats = build_edges(records, nodes, bag["alias_map"], cfg)
    assert len(edges) == 1, "同一三元组重复出现应去重为 1 条"
    edge = edges[0]
    assert edge["rel"] == "投资" and edge["head"] == "发行人" and edge["tail"] == "甲公司"
    assert edge["weight"] == 2 and set(edge["source_chunks"]) == {"c1", "c2"}
    assert edge["constraint_ok"] is True, "公司-投资->公司 属相容组合"
    assert estats["relations_raw"] == 2 and estats["deduped"] == 1
    assert estats["endpoint_added"] == 1, "仅出现在关系上的端点应被补全为节点"
