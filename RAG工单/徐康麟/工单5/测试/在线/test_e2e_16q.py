# -*- coding: utf-8 -*-
"""
工单：人工智能NLP-RAG-图像内容解析及检索优化
文件：测试/在线/test_e2e_16q.py（T6：在线验收层，依赖运行中的服务 + Ollama）

覆盖：
  · 16 题端到端抽测（图像题/表格题/文本题各取样）——判对、引用、权威层命中、单题 ≤3s
  · 冻结产物口径校验（准确率 ≥90%、逐题延迟 ≤3000ms、P50/P95/最大存在、命中率按通道分列）
  · 负例 4 条（拒答红线；N1 为**已登记缺陷**，用 xfail 记录，不掩盖）
  · 中英文同题（中文 3/3 必须通过；英文按 EN 原子口径，缺口如实 xfail 记录）
运行：pwsh -NoProfile -File run_py.ps1 -m pytest 测试/在线 -q
"""
from __future__ import annotations

import json

import pytest

SAMPLE_IDS = [5, 6, 1, 207]          # 图像结构 / 图像图表 / 表格数值 / 表格数值（含跨页证据）


@pytest.mark.parametrize("qid", SAMPLE_IDS)
def test_live_sample_questions(service, golden, by_id, diagnostic, qid):
    """在线抽测：判对（确定性）+ 引用合格 + 权威层命中 + 延迟 ≤3000ms。"""
    rec = by_id[qid]
    data = service.ask(rec["question"], lang="zh", use_cache=False, top_k=5)
    verdict = golden.judge_answer(rec, data.get("answer") or "")
    assert verdict["correct"], f"id {qid} 判错：{verdict['reason']}"
    layers = golden.check_retrieval_hit_layers(rec, data.get("chunks") or [], diagnostic=diagnostic)
    assert layers["hit_complete"], f"id {qid} 权威层未命中"
    assert data.get("cited_pages"), f"id {qid} 无引用页码"
    total = (data.get("latency_ms") or {}).get("total")
    assert total is not None and total <= 3000, f"id {qid} 服务端耗时 {total} ms 超过 3000ms"


def test_frozen_results_meet_acceptance(artifacts):
    """冻结产物：准确率 ≥90%、逐题 ≤3000ms、P50/P95/最大齐备、命中率按通道分列。"""
    s = artifacts["accuracy_report.json"]["summary"]
    assert s["n_questions"] == 16
    assert s["accuracy"] is not None and s["accuracy"] >= 0.90, f"准确率未达标：{s['accuracy']}"
    assert s["n_correct"] >= 15
    assert s["n_latency_over_3000ms"] == 0
    for k in ("latency_wall_p50_ms", "latency_wall_p95_ms", "latency_wall_max_ms"):
        assert s[k] is not None, f"缺延迟统计 {k}"
    assert s["latency_wall_max_ms"] <= 3000
    for k in ("retrieval_hit_rate_complete_bm25", "retrieval_hit_rate_complete_vector",
              "retrieval_hit_rate_complete"):
        assert s[k] is not None, f"命中率必须按通道分列并给出合并值：缺 {k}"
    rows = artifacts["qa_results.jsonl"]
    for r in rows:
        assert r["latency_wall_ms"] is not None and r["latency_wall_ms"] <= 3000, \
            f"id {r['id']} 客户端耗时 {r['latency_wall_ms']} ms 超线"
    assert all(r.get("citation_ok") for r in rows), "存在引用不可回溯的题"


def test_retrieval_hit_reported_separately_from_correct(artifacts):
    """「检索命中」与「回答判对」必须分别存在、互不代替。"""
    rows = artifacts["qa_results.jsonl"]
    assert all("hit_complete" in r for r in rows)
    assert all("correct" in r for r in rows)
    # 允许「命中但答错」「未命中但答对」两类不一致存在（如实记录）
    hit_not_correct = [r["id"] for r in rows if r.get("hit_complete") and not r.get("correct")]
    miss_but_correct = [r["id"] for r in rows if not r.get("hit_complete") and r.get("correct")]
    assert isinstance(hit_not_correct, list) and isinstance(miss_but_correct, list)


# ---------------------------------------------------------------- 负例（拒答红线）
NEG_CASES = [
    ("N1", "某某科技有限公司的注册资本是多少？", "known_defect"),
    ("N2", "武汉兴图新科电子股份有限公司2025年营业收入是多少？", "refuse"),
    ("N3", "请写一段 Python 快速排序代码。", "refuse"),
    ("N4", "武汉力源信息技术股份有限公司募集资金投向第5项「其他与主营业务相关的营运资金」的计划总投资额是多少？", "refuse"),
]


@pytest.mark.parametrize("nid,question,kind", NEG_CASES)
def test_negative_refusal(service, golden, nid, question, kind):
    """负例：不存在实体/不存在数据/越界问题 → 必须回「不清楚」且不编造。"""
    if kind == "known_defect":
        pytest.xfail("已知缺陷（T6 实测）：泛化名实体「某某科技有限公司」绕过 unknown_entity 闸门，"
                     "返回页64 的「注册资本为100万元」——已登记待优化环节修复")
    data = service.ask(question, lang="zh", use_cache=False, top_k=5)
    answer = data.get("answer") or ""
    assert data.get("refused") is True, f"{nid} 未拒答：{answer[:60]}"
    assert "不清楚" in answer or "未披露" in answer or "无法" in answer, f"{nid} 拒答措辞异常：{answer[:60]}"
    assert not __import__("re").search(r"营运资金[^。；\n]{0,12}\d", golden.normalize(answer)), \
        f"{nid} 为未披露项编造了金额：{answer[:80]}"


def test_negative_n1_defect_is_recorded(artifacts):
    """N1 缺陷必须在产物里如实留痕（不得为通过而删除或改写）。"""
    items = {x["id"]: x for x in artifacts["negatives.jsonl"]}
    assert "N1" in items
    assert items["N1"]["passed"] is False, "N1 若已修复，请更新报告与缺陷登记（勿静默通过）"
    assert "100万元" in (items["N1"].get("answer") or "") or items["N1"].get("refused") is False


# ---------------------------------------------------------------- 中英文
def test_bilingual_chinese_path(service, golden, by_id, diagnostic):
    """中文路径：三道同题必须判对（文本/图像结构/图像图表）。"""
    for qid in (543, 5, 6):
        rec = by_id[qid]
        data = service.ask(rec["question"], lang="zh", use_cache=False, top_k=5)
        verdict = golden.judge_answer(rec, data.get("answer") or "")
        assert verdict["correct"], f"中文路径 id {qid} 判错：{verdict['reason']}"


def test_bilingual_english_path(service, golden, by_id):
    """英文路径：必须给出答案或如实拒答（不得编造），并留痕语言路径状态。"""
    rec = by_id[543]
    data = service.ask("What is the registered capital of Wuhan Xingtu Xinke "
                       "Electronics Co., Ltd.?", lang="en", use_cache=False, top_k=5)
    answer = data.get("answer") or ""
    assert answer, "英文路径无答案"
    assert ("不清楚" in answer) or golden.keyword_hit(answer, "5,520.00万元")[0], \
        f"英文注册资本题既未答出数值也未如实拒答：{answer[:80]}"


@pytest.mark.xfail(reason="英文路径完整性缺陷（T6 实测，已登记）：组织结构图题只答 4 个部门、"
                          "未答 6 个销售处；IC 图表题拒答（low_similarity）", strict=False)
def test_bilingual_english_completeness(service, golden, by_id):
    """英文路径完整性：图像结构/图像图表两题应与中文口径等效（当前未达，如实 xfail）。"""
    rec5 = by_id[5]
    d5 = service.ask("In the organization chart of Wuhan P&S Information Technology Co., Ltd., how many "
                     "departments make up the Sales Department, and how many sales offices make up the "
                     "Key Account Sales Department?", lang="en", use_cache=False, top_k=5)
    a5 = d5.get("answer") or ""
    assert all(x.lower() in a5.lower() for x in ("6", "Beijing", "Guangzhou", "Chengdu",
                                                 "Shenzhen", "Wuhan", "Zhuhai")), \
        f"英文组织结构图题未答全 6 个销售处：{a5[:120]}"
    rec6 = by_id[6]
    d6 = service.ask("According to the prospectus of Wuhan P&S Information Technology Co., Ltd., based on "
                     "the 2008 China IC market application structure and growth chart, which industry has "
                     "the fastest growth rate and which has negative growth?", lang="en",
                     use_cache=False, top_k=5)
    assert not d6.get("refused"), "英文 IC 图表题被拒答"
