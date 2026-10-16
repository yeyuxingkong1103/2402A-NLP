# -*- coding: utf-8 -*-
"""工单9 pytest 用例：别名归并 / 融合权重 / 指标计算自检 / 容错 / 多语言。

工单编号：人工智能NLP-RAG-Graph RAG 优化任务

设计原则（**测试必须快、无副作用、不依赖 Ollama**）：
    * 绝大部分用例是**纯函数**测试（配置、归并、融合、指标、生成层护栏），毫秒级；
    * 需要索引/图谱的用例用 ``pytest.mark.skipif`` 守卫（缺失时 skip 而不是 fail），
      并把「加载一次」缓存在 module 级 fixture（避免每题重复加载 8847 块）。

运行：
    cd E:\\gao6gongdan\\工单9
    $env:PYTHONDONTWRITEBYTECODE='1'
    pwsh -NoProfile -File run_py.ps1 -m pytest 测试 -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "研发"), str(ROOT / "研发" / "评估"), str(ROOT / "测试")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

WORK_ORDER = "人工智能NLP-RAG-Graph RAG 优化任务"


# ===========================================================================
# 一、配置（阈值全部配置化，不硬编码）
# ===========================================================================
def test_配置默认值齐全() -> None:
    from app.graphrag_opt.配置 import DEFAULTS

    for section in ("extract", "retrieval", "generation", "eval"):
        assert section in DEFAULTS, f"缺少配置段 {section}"
    assert DEFAULTS["eval"]["context_precision_target"] == 0.80
    assert DEFAULTS["eval"]["context_recall_target"] == 0.90
    assert DEFAULTS["eval"]["sla_ms"] == 3000.0
    # 资源纪律：keep_alive 必须是 0s
    assert DEFAULTS["generation"]["keep_alive"] == "0s"


def test_配置环境变量覆盖(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.graphrag_opt.配置 import OptConfig

    monkeypatch.setenv("GRAPHRAGOPT_RETRIEVAL__TOP_K", "7")
    monkeypatch.setenv("GRAPHRAGOPT_EVAL__SLA_MS", "1234")
    monkeypatch.setenv("GRAPHRAGOPT_RETRIEVAL__不存在的键", "x")
    cfg = OptConfig.from_env()
    assert cfg.retrieval["top_k"] == 7
    assert cfg.eval["sla_ms"] == 1234.0
    assert any("不存在的键" in k for k in cfg.unknown_keys), "未识别的键必须记入 unknown_keys"


def test_配置指纹随阈值变化() -> None:
    from app.graphrag_opt.配置 import OptConfig

    a = OptConfig.from_env()
    b = OptConfig.from_env()
    assert a.fingerprint() == b.fingerprint()
    b.retrieval = dict(b.retrieval, top_k=int(b.retrieval["top_k"]) + 1)
    assert a.fingerprint() != b.fingerprint(), "阈值变了指纹必须变（缓存键依赖它）"


# ===========================================================================
# 二、实体别名归并（第一层优化）
# ===========================================================================
def _fake_entities() -> list[dict]:
    return [
        {"id": "平安银行", "name": "平安银行", "type": "company", "confidence": 0.9,
         "count": 50, "chunks": [{"chunk_id": "c1", "page": 5, "file_name": "a.pdf", "type": "text"}]},
        {"id": "平安银行股份有限公司", "name": "平安银行股份有限公司", "type": "company",
         "confidence": 0.95, "count": 10,
         "chunks": [{"chunk_id": "c2", "page": 5, "file_name": "a.pdf", "type": "text"}]},
        {"id": "何肖", "name": "何肖", "type": "person", "confidence": 0.88, "count": 1,
         "chunks": [{"chunk_id": "c3", "page": 89, "file_name": "b.pdf", "type": "text"}]},
        {"id": "何肖锋", "name": "何肖锋", "type": "person", "confidence": 0.88, "count": 4,
         "chunks": [{"chunk_id": "c4", "page": 89, "file_name": "b.pdf", "type": "text"}]},
        {"id": "2020年度", "name": "2020年度", "type": "date", "confidence": 0.85, "count": 593,
         "chunks": [{"chunk_id": "c5", "page": 2, "file_name": "a.pdf", "type": "text"}]},
    ]


def test_别名归并_公司后缀与人名前缀() -> None:
    from app.graphrag_opt.抽取层 import apply_alias, merge_aliases
    from app.graphrag_opt.配置 import DEFAULTS

    out = merge_aliases(_fake_entities(), DEFAULTS["extract"])
    alias_of = out["alias_of"]
    # R1：公司全称并入提及更多的简称
    assert apply_alias(alias_of, "平安银行股份有限公司") == "平安银行"
    # R2：人名前缀补全（何肖 → 何肖锋）
    assert apply_alias(alias_of, "何肖") == "何肖锋"
    assert out["stats"]["归并组数"] >= 2


def test_别名归并_合并后提及取并集且去重() -> None:
    from app.graphrag_opt.抽取层 import merge_aliases, merged_entities
    from app.graphrag_opt.配置 import DEFAULTS

    ents = _fake_entities()
    out = merge_aliases(ents, DEFAULTS["extract"])
    nodes = merged_entities(ents, out["alias_of"], mentions_per_entity=4, min_confidence=0.6)
    by_name = {n["name"]: n for n in nodes}
    pay = by_name["平安银行"]
    assert pay["count"] == 60, "合并后 count 应求和"
    assert "平安银行股份有限公司" in pay["aliases"], "被并入的写法要进 aliases 供实体链接回溯"
    assert {c["chunk_id"] for c in pay["chunks"]} == {"c1", "c2"}


def test_噪声剪枝_通用刻度与碎片() -> None:
    from app.graphrag_opt.抽取层 import prune_noise
    from app.graphrag_opt.配置 import DEFAULTS

    nodes = [
        {"id": "2020年度", "name": "2020年度", "type": "date", "count": 593, "degree": 62},
        {"id": "5%", "name": "5%", "type": "ratio", "count": 30, "degree": 3},
        {"id": "是指本集团", "name": "是指本集团", "type": "company", "count": 10, "degree": 10},
        {"id": "审计与风险管理委员会", "name": "审计与风险管理委员会", "type": "institution",
         "count": 5, "degree": 5},
        {"id": "平安银行", "name": "平安银行", "type": "company", "count": 60, "degree": 20},
    ]
    edges = [{"head": "平安银行", "relation": "披露", "tail": "2020年度"},
             {"head": "平安银行", "relation": "披露", "tail": "审计与风险管理委员会"}]
    out = prune_noise(nodes, edges, DEFAULTS["extract"], chunks_scanned=7430)
    kept = {n["name"] for n in out["nodes"]}
    assert "2020年度" not in kept, "通用刻度日期应被剪除"
    assert "5%" not in kept, "低值整数比例应被剪除"
    assert "是指本集团" not in kept, "抽取碎片应被剪除"
    assert "审计与风险管理委员会" in kept, "真实机构不能被误剪（并列连词陷阱）"
    assert "平安银行" in kept


def test_关系规范化_方向反转与自环去重() -> None:
    from app.graphrag_opt.抽取层 import normalize_relations
    from app.graphrag_opt.配置 import DEFAULTS

    relations = [
        # 方向反了：任职应当是 person → company
        {"head": "某公司", "relation": "任职", "tail": "张三", "confidence": 0.9},
        {"head": "张三", "relation": "任职", "tail": "某公司", "confidence": 0.9},
        {"head": "张三", "relation": "任职", "tail": "张三", "confidence": 0.9},          # 自环
        {"head": "某公司", "relation": "不存在的关系统", "tail": "某指标", "confidence": 0.9},  # 非法
    ]
    node_type = {"某公司": "company", "张三": "person", "某指标": "metric"}
    out = normalize_relations(relations, node_type=node_type, alias_of={}, cfg=DEFAULTS["extract"])
    assert out["stats"]["方向反转数"] >= 1, "方向反了应当自动反转并计数"
    assert out["stats"]["自环剪除数"] == 1
    assert out["stats"]["非法关系剪除数"] == 1
    assert out["stats"]["规范关系覆盖率"] == 1.0
    assert len(out["edges"]) == 1, "反转后与正方向边重复，应被去重合并"


# ===========================================================================
# 三、检索层：融合权重与重排
# ===========================================================================
def test_融合权重_归一化后加权() -> None:
    from app.graphrag_opt.检索层 import fuse_scores
    from app.graphrag_opt.配置 import DEFAULTS

    cfg = DEFAULTS["retrieval"]
    rows = [
        {"chunk_id": "a", "base_score": 4.0, "graph_score": 0.0, "feature": 0.0},
        {"chunk_id": "b", "base_score": 1.0, "graph_score": 0.06, "feature": 1.0},
    ]
    out = fuse_scores(rows, cfg=cfg)
    by_id = {r["chunk_id"]: r for r in out}
    # min-max：base 归一后 a=1.0 / b=0.0；graph 归一后 a=0.0 / b=1.0
    assert by_id["a"]["norm_base"] == 1.0 and by_id["b"]["norm_base"] == 0.0
    assert by_id["a"]["norm_graph"] == 0.0 and by_id["b"]["norm_graph"] == 1.0
    w = cfg["fusion_weights"]
    expected_b = w["graph"] + w["feature"]
    assert abs(by_id["b"]["fusion_score"] - expected_b) < 1e-6
    assert abs(by_id["a"]["fusion_score"] - w["base"]) < 1e-6


def test_融合权重_权重和归一() -> None:
    from app.graphrag_opt.检索层 import fuse_scores

    cfg = {"fusion_weights": {"base": 2.0, "graph": 1.0, "feature": 1.0}}   # 和为 4
    rows = [{"chunk_id": "a", "base_score": 1.0, "graph_score": 1.0, "feature": 1.0}]
    out = fuse_scores(rows, cfg=cfg)
    assert abs(out[0]["fusion_score"] - 1.0) < 1e-6, "权重应被归一化到和为 1"


def test_图特征_归一区间() -> None:
    from app.graphrag_opt.检索层 import graph_features
    from app.graphrag_opt.配置 import DEFAULTS

    f0 = graph_features(distance=0, entity_count=3, node_degree=0, cfg=DEFAULTS["retrieval"])
    f2 = graph_features(distance=2, entity_count=0, node_degree=500, cfg=DEFAULTS["retrieval"])
    assert 0.0 <= f0["feature"] <= 1.0 and 0.0 <= f2["feature"] <= 1.0
    assert f0["feature"] > f2["feature"], "近跳 + 多实体命中的特征分应更高"


def test_自适应跳数_锁定语料降为一跳() -> None:
    from app.graphrag_opt.检索层 import adaptive_hop
    from app.graphrag_opt.配置 import DEFAULTS

    cfg = DEFAULTS["retrieval"]
    locked = adaptive_hop(corpus_hint="20200214", linked={"seeds": ["平安银行"]}, cfg=cfg)
    free = adaptive_hop(corpus_hint=None, linked={"seeds": []}, cfg=cfg)
    assert locked["hop"] == 1 and free["hop"] == 2


def test_重排裁剪_词命中率加权() -> None:
    from app.graphrag_opt.检索层 import rerank_trim
    from app.graphrag_opt.配置 import DEFAULTS

    cfg = dict(DEFAULTS["retrieval"], rerank_trim=5, rerank_lexical_weight=1.0)
    rows = [
        {"chunk_id": "a", "content": "无关内容", "fusion_score": 0.9},
        {"chunk_id": "b", "content": "平安银行 营业收入 净利润", "fusion_score": 0.1},
    ]
    out = rerank_trim(rows, terms=["平安银行", "营业收入"], top_k=2, cfg=cfg)
    assert out["hits"][0]["chunk_id"] == "b", "词命中率高者应当排前（词权重=1.0 时）"


def test_实体链接_长名优先() -> None:
    from app.graphrag_opt.检索层 import link_entities

    class _G:
        ready = True
        name_index = {"平安银行": "平安银行", "平安": "平安", "银行": "银行"}
        _nodes = {"平安银行": {"name": "平安银行", "type": "company", "count": 50},
                  "平安": {"name": "平安", "type": "company", "count": 5},
                  "银行": {"name": "银行", "type": "institution", "count": 900}}

        def node(self, nid):  # noqa: ANN001
            return self._nodes.get(nid)

        def name_of(self, nid):  # noqa: ANN001
            return str(nid)

    out = link_entities("平安银行2019年营业收入", _G(), cfg={"entity_linking": True,
                                                              "link_min_len": 2, "link_limit": 12})
    assert out["seeds"][0] == "平安银行", "长名必须优先于碎片名"


def test_查询改写_英中术语映射() -> None:
    from app.graphrag_opt.检索层 import rewrite_query
    from app.graphrag_opt.配置 import DEFAULTS

    cfg = DEFAULTS["retrieval"]
    out = rewrite_query("What was the bank's net profit in 2019?", cfg=cfg, language="en")
    assert "净利润" in out["added_terms"], "net profit 必须映射到 净利润"
    assert out["use_rewrite_for_base"] is True
    out2 = rewrite_query("In Ping An Bank's 2019 Annual Report, what was revenue?", cfg=cfg,
                         language="en")
    assert "平安银行" in out2["added_companies"], "英文公司名必须映射到中文公司名"


def test_对比题判定() -> None:
    from app.graphrag_opt.检索层 import is_comparison_question

    assert is_comparison_question("对比平安银行与招商银行2019年营业收入")
    assert is_comparison_question("分析这些银行的共同策略与差异化策略")
    assert not is_comparison_question("平安银行2019年营业收入是多少")


def test_证据去重_同一chunk不重复占名额() -> None:
    from app.graphrag_opt.检索层 import select_evidence
    from app.graphrag_opt.配置 import DEFAULTS

    cfg = dict(DEFAULTS["retrieval"], evidence_k=3)
    hits = [
        {"chunk_id": "x", "content": "平安银行 营业收入", "file_name": "a.pdf", "page": 5,
         "retrieval_channel": "base"},
        {"chunk_id": "x", "content": "平安银行 营业收入", "file_name": "a.pdf", "page": 5,
         "retrieval_channel": "field_probe"},
        {"chunk_id": "y", "content": "净利润", "file_name": "a.pdf", "page": 6,
         "retrieval_channel": "base"},
    ]
    out = select_evidence(hits, terms=["平安银行"], cfg=cfg)
    ids = [h["chunk_id"] for h in out]
    assert ids.count("x") == 1, "重复 chunk 必须去重（否则浪费上下文名额）"


# ===========================================================================
# 四、生成层：三元组注入 / 引用约束 / 拒答护栏
# ===========================================================================
def test_三元组注入_格式与截断() -> None:
    from app.graphrag_opt.生成层 import build_triples_block
    from app.graphrag_opt.配置 import DEFAULTS

    graph = {"seed_names": ["平安银行"], "edges": [
        {"head": "平安银行", "relation": "披露", "tail": "营业收入", "confidence": 0.92,
         "chunk_id": "c1"},
        {"head": "营业收入", "relation": "指标取值", "tail": "1,379.58亿元", "confidence": 0.9,
         "chunk_id": "c1"},
    ]}
    out = build_triples_block(graph, cfg=dict(DEFAULTS["generation"], max_triples=1))
    assert out["count"] == 1, "max_triples 应生效"
    assert "平安银行 --披露--> 营业收入" in out["text"]


def test_三元组注入_关闭开关() -> None:
    from app.graphrag_opt.生成层 import build_triples_block
    from app.graphrag_opt.配置 import DEFAULTS

    out = build_triples_block({"seed_names": [], "edges": [{"head": "a", "relation": "披露",
                                                            "tail": "b", "confidence": 1.0}]},
                              cfg=dict(DEFAULTS["generation"], inject_triples=False))
    assert out["count"] == 0 and out["text"] == ""


def test_拒答护栏_无证据与无重叠() -> None:
    from app.graphrag_opt.生成层 import refusal_guard, refusal_text
    from app.graphrag_opt.配置 import DEFAULTS

    cfg = DEFAULTS["generation"]
    empty = refusal_guard("平安银行营业收入", [], terms=["平安银行"], cfg=cfg)
    assert empty["refuse"] is True
    unrelated = refusal_guard("完全无关的问题xyz", [{"content": "招商银行2019年年度报告"}],
                              terms=["完全无关的问题xyz"], cfg=cfg)
    assert unrelated["refuse"] is True
    ok = refusal_guard("平安银行营业收入", [{"content": "平安银行营业收入1,379.58亿元"}],
                       terms=["平安银行", "营业收入"], cfg=cfg)
    assert ok["refuse"] is False
    assert "不足以回答" in refusal_text("zh") and "Insufficient" in refusal_text("en")


def test_模板生成_含引用与关键数据() -> None:
    from app.graphrag_opt.生成层 import compose_answer
    from app.graphrag_opt.配置 import DEFAULTS

    evidence = [{"chunk_id": "20200214-p19-t1-0", "page": 19, "type": "table",
                 "file_name": "2020-02-14__平安银行股份有限公司__000001__平安银行__2019年__年度报告.pdf",
                 "content": "营业收入 | 137,958 | 116,716 | 18.2%\n净利润 | 28,195 | 24,818 | 13.6%"}]
    out = compose_answer("平安银行2019年营业收入", evidence,
                         draft="全行实现营业收入1,379.58亿元，同比增长18.2%",
                         triples={"triples": [{"head": "平安银行", "relation": "披露",
                                               "tail": "营业收入"}]},
                         language="zh", cfg=DEFAULTS["generation"],
                         terms=["营业收入", "净利润"])
    assert out["used_template"] is True
    assert "【引用】" in out["text"] and "平安银行 第19页" in out["text"]
    assert "【图谱依据】" in out["text"], "三元组必须进答案（图谱结构化注入）"
    assert "13.6%" in out["text"] or "18.2%" in out["text"], "表格行不能被切成碎片丢掉"


def test_模板生成_英文分支() -> None:
    from app.graphrag_opt.生成层 import compose_answer
    from app.graphrag_opt.配置 import DEFAULTS

    evidence = [{"chunk_id": "c1", "page": 18, "type": "text", "file_name": "x__邮储银行__x.pdf",
                 "content": "实现净利润610.36亿元，增长16.52%"}]
    out = compose_answer("net profit?", evidence, draft="净利润610.36亿元",
                         triples={"triples": []}, language="en",
                         cfg=DEFAULTS["generation"], terms=["净利润"])
    assert out["text"].startswith("Conclusion:")
    assert "Sources:" in out["text"]


def test_快路径置信度判定() -> None:
    from app.graphrag_opt.生成层 import template_confidence
    from app.graphrag_opt.配置 import DEFAULTS

    cfg = DEFAULTS["generation"]
    good = template_confidence(
        "平安银行营业收入", [],
        draft="平安银行2019年实现营业收入1,379.58亿元，同比增长18.2%，净利润281.95亿元。",
        cfg=cfg)
    assert good["ok"] is True, f"长度/数值/实词三项都应通过：{good}"
    short = template_confidence("平安银行营业收入", [], draft="平安", cfg=cfg)
    assert short["ok"] is False and short["chars_ok"] is False


# ===========================================================================
# 五、指标计算自检（RAGAS 近似器）
# ===========================================================================
def test_指标自检_内置用例全通过() -> None:
    import ragas近似 as RA

    assert RA._selftest() == 0, "RAGAS 近似器的内置自检必须全通过"


def test_指标_数字边界不误命中() -> None:
    import ragas近似 as RA

    ctx = [{"chunk_id": "a", "page": 5, "file_name": "f.pdf", "text": "不良贷款率11.65%"}]
    m = RA.context_metrics(ctx, keywords=["1.65"], expected_pages_hint="", source_file=None)
    assert m["context_recall"] == 0.0, "11.65 不能命中 1.65"


def test_指标_跨文档题按下标配对() -> None:
    import ragas近似 as RA

    ctx = [{"chunk_id": "1", "page": 30, "file_name": "B.pdf", "text": "a"},
           {"chunk_id": "2", "page": 10, "file_name": "A.pdf", "text": "b"}]
    ok = RA.context_metrics(ctx, keywords=["a"], expected_pages_hint="5-20;25-40",
                            source_file="A.pdf;B.pdf")
    assert ok["context_precision"] == 1.0
    bad = RA.context_metrics([{"chunk_id": "1", "page": 30, "file_name": "A.pdf", "text": "zz"}],
                             keywords=["a"], expected_pages_hint="5-20;25-40",
                             source_file="A.pdf;B.pdf")
    assert bad["context_precision"] == 0.0, "文件与页区间必须下标配对，不能只看页码"


def test_指标_rank加权对排序敏感() -> None:
    import ragas近似 as RA

    kw = ["k1"]
    # 中间那块既不在参考页、也不含关键词 ⇒ 只有它是「不相关块」
    ordered = [{"chunk_id": "1", "page": 5, "file_name": "f.pdf", "text": "k1"},
               {"chunk_id": "2", "page": 99, "file_name": "f.pdf", "text": "无"},
               {"chunk_id": "3", "page": 5, "file_name": "f.pdf", "text": "k1"}]
    m = RA.context_metrics(ordered, keywords=kw, expected_pages_hint="5", source_file="f.pdf")
    # 指标一律四舍五入到 4 位小数（与产物 jsonl 一致）⇒ 这里比 0.6667 而不是 2/3
    assert m["context_precision"] == 0.6667
    assert m["context_precision_rank"] == 0.8333, "无关块在中间应被 rank 口径惩罚"


def test_指标_汇总达标判定() -> None:
    import ragas近似 as RA

    rows = [{"context_precision": 0.9, "context_recall": 0.95, "context_precision_rank": 0.9,
             "topk_context_precision": 0.9, "topk_context_recall": 0.9, "topk_context_precision_rank": 0.9,
             "latency_total_ms": 1000, "faithfulness_detail": {"faithfulness": 1.0},
             "answer_relevancy_detail": {"answer_relevancy": 1.0}}]
    agg = RA.aggregate(rows, override={"context_precision_target": 0.8, "context_recall_target": 0.9,
                                       "sla_ms": 3000})
    assert agg["达标"]["context_precision达标"] is True
    assert agg["达标"]["context_recall达标"] is True
    assert agg["达标"]["响应时间达标率"] == 1.0


# ===========================================================================
# 六、容错 / 降级（纯函数级；端到端场景见 研发/评估/稳定性压测.py）
# ===========================================================================
def test_容错_空与异常输入不崩() -> None:
    import ragas近似 as RA

    for bad in ("", None, "   ", "\x00\x01", "🎉🐱", "？"):
        m = RA.context_metrics([], keywords=[], expected_pages_hint=str(bad), source_file=bad)
        assert m["context_precision"] == 0.0 and m["context_recall"] == 0.0
        assert isinstance(RA.answer_relevancy(bad, keywords=[]), dict)


def test_容错_孤立节点剪枝字段缺失不崩() -> None:
    from app.graphrag_opt.抽取层 import prune_noise
    from app.graphrag_opt.配置 import DEFAULTS

    out = prune_noise([], [], DEFAULTS["extract"], chunks_scanned=0)
    assert out["nodes"] == [] and out["edges"] == []


def test_容错_配置损坏不崩() -> None:
    """配置损坏时必须退回默认值并留痕，**不得抛异常**。

    ⚠️ 用**工作区内**的临时目录（不能用 pytest 的 ``tmp_path``）：
    本机沙箱是 workspace-write，工作区 = ``E:\\gao6gongdan``，
    写系统临时目录会被拒绝（实测 ``PermissionError: [WinError 5]``）。
    """
    from app.graphrag_opt.配置 import OptConfig

    tmp_dir = ROOT / "测试" / "测试数据" / "_临时"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    bad = tmp_dir / "坏配置.json"
    bad.write_text("{ 这不是合法 JSON", encoding="utf-8")
    try:
        cfg = OptConfig.from_env(config_file=bad)          # 不得抛异常
        assert cfg.retrieval["top_k"] == 10, "配置损坏时应退回默认值并留痕"
        assert any("坏配置" in k for k in cfg.unknown_keys)
    finally:
        bad.unlink(missing_ok=True)


# ===========================================================================
# 七、多语言（不依赖 Ollama 的部分）
# ===========================================================================
def test_多语言_英中术语与公司映射表完整性() -> None:
    from app.graphrag_opt.检索层 import EN_COMPANY_MAP, EN_TERM_MAP

    assert EN_TERM_MAP["net profit"] == ("净利润",)
    assert EN_COMPANY_MAP["ping an bank"] == "平安银行"
    assert len(EN_COMPANY_MAP) >= 8, "9 份年报的主要公司都应能映射"


def test_多语言_探针词识别() -> None:
    from app.graphrag_opt.检索层 import probe_terms_of

    zh = probe_terms_of("平安银行的风险管理体系如何体现其应对宏观经济周期波动的能力")
    assert "风险管理" in zh
    en = probe_terms_of("What was the bank's net profit and non-performing loan ratio?")
    assert "净利润" in en and "不良贷款率" in en


@pytest.mark.skipif(not (ROOT / "研发" / "data" / "图谱优化后" / "entities.jsonl").is_file(),
                    reason="优化后图谱未生成（先跑 研发/scripts/构建优化图谱.py）")
def test_抽取层_优化后图谱统计自洽() -> None:
    stats = json.loads((ROOT / "研发" / "data" / "图谱优化后" / "图谱优化统计.json")
                       .read_text(encoding="utf-8"))
    assert stats["输出"]["节点"] < stats["输入"]["节点"], "节点应当减少（合并 + 剪枝）"
    assert stats["输出"]["平均度数"] > stats["输入"]["平均度数"], "平均度数应当上升（连通性提升）"
    assert stats["别名归并"]["归并组数"] > 0
    assert stats["关系规范化"]["规范关系覆盖率"] == 1.0


@pytest.mark.skipif(not (ROOT / "优化" / "评估结果" / "优化前后对比.json").is_file(),
                    reason="对比产物未生成（先跑 研发/评估/run_eval_opt.py）")
def test_对比产物_口径与达标字段齐全() -> None:
    data = json.loads((ROOT / "优化" / "评估结果" / "优化前后对比.json").read_text(encoding="utf-8"))
    assert "核心指标对比" in data and "响应时间对比" in data and "验收达标判定" in data
    assert data["口径"]["use_cache"] is False, "响应时间口径必须是非缓存"
    for key in ("context_precision", "context_recall"):
        row = data["核心指标对比"][key]
        assert "优化前" in row and "优化后" in row and "变化" in row
