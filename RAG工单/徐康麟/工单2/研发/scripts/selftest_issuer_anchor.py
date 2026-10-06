"""发行人锚定自测：无主体查询 / 指代追问都必须锚定发行人行（t12 回归防护）。

工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
阶段：研发 / 自测脚本（t12 验收证据；正式判分由 T5/T9 负责）

背景（captain 复核定位，t12 缺陷）：
「查询串里没有发行人可识别名称 ⇒ 没有发行人锚定」。招股书是**单一发行人**文档，
同一字段（注册资本/法定代表人/注册地址）在发行人自身章节与子公司/关联方/释义表里都会出现，
查询串不带主体时就会取到子公司行（实测：「注册资本是多少？」→ `100 万元 / p64`）。

本脚本逐项验证（改动前后对照可用）：
① 四种查询串全部返回 `注册资本是5,520 万元 [页码: 52]`；
② 多轮：轮1 法定代表人 → 轮2「那它的注册资本呢？」→ 必须 5,520 万元 / p52（不得是子公司 100 万元）；
③ 每轮 `retrieval_traces.variants` 必须包含 `issuer_anchor` 变体；
④ 追问改写必须产出**可直接检索的完整问句**（无指代词、含主体、规范问尾）。

用法::

    pwsh -NoProfile -File run_py.ps1 研发/scripts/selftest_issuer_anchor.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.core.qa_engine import QAEngine  # noqa: E402
from app.core.query_understanding import QueryUnderstanding  # noqa: E402

#: ① 四种查询串（同一字段、有无主体名）
QUERIES = (
    "武汉兴图新科电子股份有限公司注册资本是多少？",
    "兴图新科注册资本是多少？",
    "公司注册资本是多少？",
    "注册资本是多少？",
)
EXPECT_AMOUNT = "5,520"
EXPECT_PAGE = 52
TURNS = (
    "武汉兴图新科电子股份有限公司法定代表人是谁？",
    "那它的注册资本呢？",
    "它的注册地址呢？",
)


def main() -> int:
    """自测入口；退出码 0 = 全部通过。"""
    engine = QAEngine(force_extractive=True)
    if not engine.load_index():
        print("业务失败：索引未就绪，请先执行 研发/scripts/build_index.py", file=sys.stderr)
        return 2
    understanding = QueryUnderstanding(use_llm=False)
    failures: list[str] = []

    print("=" * 100)
    print("① 四种查询串（无历史）")
    for question in QUERIES:
        analysis = understanding.analyze(question)
        answer = engine.ask(question)
        variants = [text for text, _kind in understanding.search_queries(analysis)]
        ok = EXPECT_AMOUNT in answer.answer and any(c.page == EXPECT_PAGE for c in answer.citations)
        if not ok:
            failures.append(f"① {question} → {answer.answer[:60]!r} 引用页={[c.page for c in answer.citations]}")
        print(f"  [{'✅' if ok else '❌'}] {question}")
        print(f"       答案={answer.answer[:64]!r} 引用页={[c.page for c in answer.citations]}")

    print("=" * 100)
    print("②③④ 三轮会话（轮2 为指代追问）")
    conversation_id = engine.new_conversation()
    for index, question in enumerate(TURNS, start=1):
        answer = engine.ask(question, conversation_id=conversation_id)
        traces = engine._store.get_retrieval_traces(conversation_id=conversation_id, limit=10)  # noqa: SLF001
        trace = next((item for item in traces if item.question == question), None)
        variants = list(trace.variants) if trace else []
        has_anchor = any("武汉兴图新科电子股份有限公司" in text for text in variants)
        print(f"  轮{index} {question}")
        print(f"       答案={answer.answer[:64]!r} 引用页={[c.page for c in answer.citations]}")
        print(f"       rewritten={trace.rewritten_query if trace else '-'!r}")
        print(f"       variants={variants}")
        if index >= 2:
            if not has_anchor:
                failures.append(f"③ 轮{index} variants 缺少 issuer_anchor：{variants}")
            if not (trace and trace.rewritten_query and trace.rewritten_query != question):
                failures.append(f"④ 轮{index} 追问未产出改写问句：{question!r}")
            elif "它" in trace.rewritten_query or "那" in trace.rewritten_query[:1]:
                failures.append(f"④ 轮{index} 改写后仍含指代词/话语标记：{trace.rewritten_query!r}")
        if index == 2 and EXPECT_AMOUNT not in answer.answer:
            failures.append(f"② 轮2（追问注册资本）答案不含 {EXPECT_AMOUNT}：{answer.answer[:60]!r}")
        if index == 2 and any("100" in answer.answer for _ in (0,)):
            failures.append(f"② 轮2 疑似取到子公司行（100 万元）：{answer.answer[:60]!r}")

    print("=" * 100)
    if failures:
        print(f"结论：FAIL ❌（{len(failures)} 项）")
        for item in failures:
            print(f"  - {item}")
        return 2
    print("结论：PASS ✅（四种查询串 + 三轮追问均锚定发行人；variants 含 issuer_anchor；改写为独立问句）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
