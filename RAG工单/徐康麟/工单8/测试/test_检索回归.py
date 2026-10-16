# -*- coding: utf-8 -*-
"""工单8 测试：Graph RAG 检索回归（中文检索 / 图谱扩展 / 返回块字段 / 与工单7 可比性）。

工单编号：人工智能NLP-RAG-基于Graph RAG 实现金融问答

回归口径：
    * 用 工单7 测试集的**原题**做回归，断言「检索结果必须包含期望页的关键证据」；
    * 判分仍用 工单7 的纯函数判分模块（``研发/评估/判分口径.py`` 转发），保证口径一致；
    * 断言项都取**宽松但可证伪**的判据（页命中 / recall / 字段完整性），不写死易漂移的近似值。

运行：``pwsh -NoProfile -File run_py.ps1 -m pytest 测试/test_检索回归.py -q``
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "研发"))
sys.path.insert(0, str(_ROOT / "研发" / "评估"))

import 判分口径 as J  # noqa: E402  —— 工单7 同源判分口径

#: 工单编号（源码头部注释要求）：人工智能NLP-RAG-基于Graph RAG 实现金融问答

#: 检索回归用例（原题 + 期望证据页 + 必须出现的证据片段）
CASES = [
    {
        "id": "Q01",
        "question": "平安银行2019年年度报告的董事长致辞中，提到该行盈利增长的关键因素有哪些？请结合业务结构、盈利趋势与风险状况说明。",
        "expected_pages_hint": "5-51",
        "source_file": "2020-02-14__平安银行股份有限公司__000001__平安银行__2019年__年度报告.pdf",
        "must_contain": "1,379.58",
    },
    {
        "id": "Q07",
        "question": "平安银行2019年年度报告中，该行2019年实现营业收入和净利润分别是多少？同比增速分别是多少？年末不良贷款率和拨备覆盖率分别是多少？",
        "expected_pages_hint": "5-67",
        "source_file": "2020-02-14__平安银行股份有限公司__000001__平安银行__2019年__年度报告.pdf",
        "must_contain": "137,958",
    },
    {
        "id": "Q08",
        "question": "招商银行2019年年度报告中，该行全年营业收入、归属于本行股东的净利润、年末总资产和每股现金分红分别是多少？",
        "expected_pages_hint": "3-251",
        "source_file": "2020-03-21__招商银行股份有限公司__600036__招商银行__2019年__年度报告.pdf",
        "must_contain": "2,697.03",
    },
]


def _hits_text(hits: list[dict]) -> str:
    return "".join(str(h.get("text") or "") for h in hits)


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_中文检索命中期望证据(engine, case) -> None:
    """中文问句的 Top-10 必须命中期望页，且证据里出现该题的关键数值。"""
    result = engine.retrieve(case["question"], top_k=10)
    hits = list(result.get("hits") or [])
    assert hits, f"{case['id']}：检索返回空"
    page = J.page_hit(hits, case["expected_pages_hint"], source_filter=case["source_file"])
    assert page.get("hit"), f"{case['id']}：Top-10 未命中期望页 {case['expected_pages_hint']}"
    assert case["must_contain"] in _hits_text(hits), (
        f"{case['id']}：Top-10 证据里未出现关键数值 {case['must_contain']}")


def test_返回块字段完整(engine) -> None:
    """「返回答案所在文本块」的硬要求：每块必须有 chunk_id / 页码 / 文件名 / 文本。"""
    result = engine.retrieve(CASES[1]["question"], top_k=10, use_graph=False)
    hits = list(result.get("hits") or [])
    assert hits, "检索返回空"
    for hit in hits:
        assert hit.get("chunk_id"), f"chunk_id 缺失：{hit}"
        assert int(hit.get("page") or 0) > 0, f"页码缺失：{hit.get('chunk_id')}"
        assert hit.get("file_name"), f"文件名缺失：{hit.get('chunk_id')}"
        assert str(hit.get("text") or "").strip(), f"块文本为空：{hit.get('chunk_id')}"
        assert hit.get("page_basis") == "1-based-physical", "页码口径必须 1-based 物理页"


def test_图谱扩展不影响基础排名(engine) -> None:
    """图谱块只追加不占名额：开启图谱后的**前 10 名**必须与关闭图谱一致。"""
    question = CASES[1]["question"]
    with_graph = engine.retrieve(question, top_k=10, use_graph=True)
    without = engine.retrieve(question, top_k=10, use_graph=False)
    top_with = [h["chunk_id"] for h in (with_graph.get("hits") or [])[:10]]
    top_without = [h["chunk_id"] for h in (without.get("hits") or [])[:10]]
    assert top_with == top_without, (
        "开启图谱改变了基础 Top-10 排名（配置 graph_slots 应为空，图谱块只追加）")


def test_图谱通道确实生效(engine) -> None:
    """图谱通道必须真的产出候选：命中实体、邻域节点、图谱独有块三者都应非空。"""
    result = engine.retrieve(CASES[0]["question"], top_k=10, use_graph=True)
    diag = result.get("diagnostics") or {}
    assert result.get("graph_used"), "问句命中了图谱实体，graph_used 却为 False"
    assert int(diag.get("graph_nodes") or 0) > 0, "图谱邻域节点数为 0"
    assert int(diag.get("graph_hits") or 0) > 0, "图谱通道未产出候选块"
    assert (result.get("graph") or {}).get("seed_names"), "图谱结构缩略缺少命中实体"


def test_图谱独有块被追加(engine) -> None:
    """图谱独有块（base 未召回）必须被追加在 Top-K 之后，可供生成与展示使用。"""
    result = engine.retrieve(CASES[1]["question"], top_k=10, use_graph=True)
    channels = [h.get("retrieval_channel") for h in (result.get("hits") or [])]
    assert "graph_only" in channels, f"未出现图谱独有块（通道分布：{set(channels)}）"


def test_诊断字段可追溯(engine) -> None:
    """诊断字段必须齐全（报告与截图都靠它给数字）。"""
    result = engine.retrieve(CASES[0]["question"], top_k=10)
    diag = result.get("diagnostics") or {}
    for key in ("base_hits", "graph_hits", "graph_nodes", "graph_seed_names", "hop",
                "graph_ms", "total_ms", "graph_rejected_irrelevant"):
        assert key in diag, f"诊断字段缺失：{key}"
    assert float(diag["total_ms"]) > 0, "检索耗时异常"
    assert float(diag["graph_ms"]) >= 0, "图谱耗时异常"


def test_与工单7判分口径可用() -> None:
    """判分口径必须是 工单7 的原模块（否则「比对 07 工单」失去意义）。"""
    assert J.available(), f"判分模块不可用：{J.import_error()}"
    score = J.score_answer("实现营业收入1,379.58亿元，同比增长18.2%", ["1,379.58", "18.2%"], False)
    assert score["keyword_hit_count"] == 2, "判分模块对已知命中判断错误"
    assert score["is_correct"] is True, "判分模块严格判对逻辑异常"
    # 数字边界口径：11.65 不应命中 1.65（工单7 的关键口径）
    assert J.score_answer("不良贷款率11.65%", ["1.65"], False)["keyword_hit_count"] == 0, \
        "数字边界口径被破坏（11.65 误命中 1.65）"


def test_关闭图谱时退化为纯基础检索(engine) -> None:
    """``use_graph=False`` 时不得出现图谱通道标记（消融口径必须干净）。"""
    result = engine.retrieve(CASES[2]["question"], top_k=10, use_graph=False)
    channels = {h.get("retrieval_channel") for h in (result.get("hits") or [])}
    assert channels <= {"base"}, f"关闭图谱后仍出现图谱通道：{channels}"
    assert not (result.get("diagnostics") or {}).get("graph_hits"), "关闭图谱后图谱候选应为 0"
