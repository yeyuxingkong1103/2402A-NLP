# 工单编号：人工智能NLP-RAG-Query理解优化
#          （工单 PDF 原文：人工智能NLP-RAG-Query 理解优化任务）
"""5 轮验收序列（工单5 的核心验收项）。标记 slow：需要 Ollama + 索引。

两个实测约束决定了本文件的写法，都不是猜测：

1. **大栈线程**：本机 Windows 主线程栈约 1MB，加载 bge-m3 / reranker 会触发
   导入期栈溢出（进程 exit 139，Python 层捕获不到，见 `src/bootstrap.py`）。
   故模型加载与整条问答链都包在 `bootstrap.run_with_large_stack` 里跑，
   与 `run_acceptance.py` 同一约束。

2. **计时口径**：`timings["total_ms"]` **已含**多轮改写延迟（Task5 fix wave F4），
   不再另外加 `rewrite["latency_ms"]`——否则每个代词轮虚增 ~200ms。

冷启动 vs 稳态：首轮不做任何预热时要付 ~25s 的模型加载（实测 Q1 = 27471.6ms），
那是进程冷启动成本，不是"单轮响应时间"。验收口径是"系统起来之后每轮 ≤3s"，
故这里用 session 级 warmup 夹具把冷启动剔除，逐轮计时测的是稳态时延。
"""
import json
import os

import pytest

# bootstrap 必须最先导入（离线模式 + 大栈），不要与下面的导入合并或调换顺序
from src import bootstrap
from src import config, dialogue, rag


DATA = os.path.join(config.ROOT, "data", "acceptance_dialogue.json")
pytestmark = pytest.mark.slow


@pytest.fixture(scope="session")
def warm():
    """会话级预热：embedding / reranker / BM25 / Ollama 全部就位。

    必须在 `results` 之前跑（后者依赖本夹具）。不预热时首轮要付 ~25s 冷启动，
    计时断言会把它算成"单轮耗时"，那是错误口径。
    """
    info = bootstrap.run_with_large_stack(rag.warmup)
    print(f"\n[acceptance warmup] {info}")
    return info


@pytest.fixture(scope="module")
def spec():
    with open(DATA, encoding="utf-8") as fh:
        return json.load(fh)


def _run_sequence(spec):
    """真实跑完 5 轮（在大栈线程里执行，见模块 docstring 约束 1）。"""
    dialogue.reset_all()
    st = dialogue.get_state("acceptance")
    out = []
    for item in spec["turns"]:
        r, t = rag.ask_turn(item["question"], st)
        out.append((item, r, t))
    return out


@pytest.fixture(scope="module")
def results(spec, warm):
    return bootstrap.run_with_large_stack(_run_sequence, spec)


def test_all_turns_return_answers(results):
    for item, r, t in results:
        assert r.answer and r.mode != "error", f"第{item['turn']}轮无答案：{r.answer[:80]}"


def test_rewrite_produces_expected_company(results):
    """Q2~Q5 的改写必须收敛到正确的公司。"""
    for item, r, t in results:
        if item["turn"] == 1:
            continue
        assert item["expect_company"] in t.resolved_question, \
            f"第{item['turn']}轮改写未含期望公司：{t.resolved_question}"


def test_rewrite_modes_are_not_unresolved(results):
    for item, r, t in results:
        assert t.rewrite_mode != "unresolved", f"第{item['turn']}轮指代未能解析"


def test_required_facts_present(results):
    """逐轮关键事实命中（工单给的标准答案）。"""
    misses = []
    for item, r, t in results:
        for f in item["expected_facts"]:
            if f not in r.answer:
                misses.append((item["turn"], f))
    assert not misses, f"未命中的关键事实：{misses}"


def test_each_turn_under_3s(results):
    """硬性要求：每轮 ≤3 秒（用总耗时，含改写）。"""
    slow = [(item["turn"], r.timings.get("total_ms"))
            for item, r, t in results
            if (r.timings.get("total_ms") or 0) > 3000]
    assert not slow, f"超过 3 秒的轮次：{slow}"


def test_every_turn_has_citation(results):
    for item, r, t in results:
        assert r.citations, f"第{item['turn']}轮没有引用来源"
        assert any(c.get("source") == item["source"] for c in r.citations), \
            f"第{item['turn']}轮引用未指向 {item['source']}"


def test_rows_carry_retrieval_contexts(results):
    """验收结果必须带每轮**真实喂给 LLM 的检索上下文**（Task11 的 RAGAS 要用）。

    直接复用 `run_acceptance.build_row()`——落盘 JSON 与这里断言的是同一个函数，
    避免"脚本另写一套拼装逻辑、测试测的是影子实现"。
    """
    import run_acceptance

    for item, r, t in results:
        row = run_acceptance.build_row(item, r, t)
        assert row["context_text"].strip(), f"第{item['turn']}轮无检索上下文"
        assert row["contexts"], f"第{item['turn']}轮上下文片段为空"
        assert all(c["text"].strip() for c in row["contexts"]), \
            f"第{item['turn']}轮有空的上下文片段"
        assert any(c["source"] == item["source"] for c in row["contexts"]), \
            f"第{item['turn']}轮上下文未含 {item['source']}"
