# -*- coding: utf-8 -*-
"""人工智能NLP-RAG-功能测试及评估 —— 多轮对话回归测试（指代消解 / 槽位继承）。

三组真实运行（全部串行，**独立 session**，互不污染）：

* **A 组 同 session 追问（``QAEngine`` + ``session_id``）** — ``TC-M02`` 同族
  1. 平安银行2019年营业收入是多少？  → 2. 那净利润呢？ → 3. 招商银行呢？
* **B 组 指代词「它 / 该行」（``QAEngine``）** — ``TC-M01`` / ``TC-M06`` 同族
  1. ``TC-Q07`` 题面（含「该行」，焦点=平安银行）→ 2. 它的拨备覆盖率是多少？ → 3. 该行的不良贷款率是多少？
* **C 组 对照（``MultiTurnEngine``，含改写策略留痕）** — 忠实复现 ``TC-M01`` + ``TC-M02``
  1. ``TC-Q07`` 题面 → 2. 它的拨备覆盖率是多少？ → 3. 那招商银行呢？

逐轮断言：``未抛异常`` / ``答案非空`` / ``有引用页`` / ``引用页合法`` / ``引用语料同族(预期公司)`` / ``耗时被记录``。
其中「引用语料同族」直接对应历史缺陷（跨语料泄漏：曾引用非点名公司年报片段），是本组最关键的回归断言。

C 组额外把 ``设计/测试用例设计.md`` 里 ``TC-M01`` / ``TC-M02`` 的**原始判据**逐条核对，
结果写入 ``设计判据`` 字段（供报告回填，不改变本脚本自身的 PASS/FAIL 语义）。

运行：
    python 测试\\回归测试\\test_多轮对话.py
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _公共 as K  # noqa: E402

GROUP = "多轮对话"

#: 公司名 -> 语料文件名关键字（用于「同族语料」判定；取自索引清单 corpus_keys）
CORPUS_KEY = {
    "平安银行": "平安银行",
    "招商银行": "招商银行",
    "中国人寿": "中国人寿",
    "中国太保": "中国太保",
    "中信证券": "中信证券",
}

TURNS_A = [
    ("reg-mt-a-1", "A1 同 session 首问", "平安银行2019年营业收入是多少？", "平安银行"),
    ("reg-mt-a-2", "A2 追问（省略主体）", "那净利润呢？", "平安银行"),
    ("reg-mt-a-3", "A3 实体切换", "招商银行呢？", "招商银行"),
]
TURNS_B = [
    ("reg-mt-b-1", "B1 首问（题面含「该行」）", None, "平安银行"),          # None = 取题库 TC-Q07 题面
    ("reg-mt-b-2", "B2 指代「它」", "它的拨备覆盖率是多少？", "平安银行"),
    ("reg-mt-b-3", "B3 指代「该行」", "该行的不良贷款率是多少？", "平安银行"),
]
TURNS_C = [
    ("reg-mt-c-1", "C1 首问（题库 TC-Q07 题面）", None, "平安银行"),
    ("reg-mt-c-2", "C2 指代「它」（TC-M01）", "它的拨备覆盖率是多少？", "平安银行"),
    ("reg-mt-c-3", "C3 实体切换（TC-M02）", "那招商银行呢？", "招商银行"),
]


def make_engines() -> tuple[Any, Any, dict[str, Any]]:
    """构造 ``QAEngine`` 与共用同一底层引擎的 ``MultiTurnEngine``（**只预热一次**）。"""
    from app.core.config import get_config
    from app.core.qa_engine import QAEngine
    from app.multiturn_engine import MultiTurnEngine

    cfg = get_config()
    base = QAEngine(cfg=cfg)
    mt = MultiTurnEngine(cfg=cfg, engine=base, persist_dir=None)
    # 走 MultiTurnEngine.warmup：内部先预热底层（索引/嵌入/全文倒排/生成），再建改写器。
    # 刻意**不**先单独预热 base，否则底层会被预热两次，白白抢占 CPU 单实例的 Ollama。
    mt_warm = mt.warmup(with_generate=True)
    warm = {"multiturn": mt_warm, "base": mt_warm.get("base")}
    return base, mt, warm


def _corpus_ok(citations: list[dict[str, Any]], company: str | None) -> tuple[bool, str]:
    """引用语料是否属于点名公司（返回 ``(是否同族, 实测说明)``）。"""
    if company is None:
        return True, "未指定公司"
    key = CORPUS_KEY.get(company, company)
    corpora = sorted({str(c.get("corpus") or "") for c in citations if c.get("corpus")})
    if not corpora:
        return False, "无引用语料"
    foreign = [c for c in corpora if key not in c]
    return (not foreign), ("; ".join(c[:40] for c in corpora))[:200]


def _finish_turn(case: dict[str, Any], cited: list[int], citations: list[dict[str, Any]],
                 limits: dict[str, int], wall_ms: float, latency: dict[str, Any],
                 expect_company: str | None, text: str) -> None:
    """逐轮公共断言（未抛异常 / 答案非空 / 有引用页 / 引用页合法 / 同族 / 耗时）。"""
    K.add_assert(case, "未抛异常", True, "ok")
    K.add_assert(case, "答案非空", len(text.strip()) > 0, f"{len(text.strip())} 字")
    K.add_assert(case, "有引用页", len(cited) > 0, f"{cited}")
    bad: list[str] = []
    for cite in citations:
        corpus = str(cite.get("corpus") or "")
        page = cite.get("page")
        limit_pages = limits.get(corpus)
        if not isinstance(page, int):
            bad.append(f"{corpus}: 页码非整数({page!r})")
        elif page < 1 or (limit_pages is not None and page > limit_pages):
            bad.append(f"{corpus}: p{page} 越界(1..{limit_pages})")
    K.add_assert(case, "引用页合法(1..page_count)", not bad,
                 "全部合法" if not bad else "; ".join(bad)[:160])
    same, detail = _corpus_ok(citations, expect_company)
    K.add_assert(case, f"引用语料同族(预期:{expect_company})", same, detail)
    total_ms = float(latency.get("total") or 0.0)
    K.add_assert(case, "耗时被记录", total_ms > 0 and wall_ms > 0,
                 f"引擎total={total_ms}ms / 墙钟={wall_ms}ms")


def _run_qa_turn(engine: Any, *, sid: str, case_id: str, name: str, question: str,
                 expect_company: str | None, limits: dict[str, int],
                 verbose: bool) -> dict[str, Any]:
    case = K.new_case(case_id, name, group=GROUP, question=question,
                      expect=f"引用语料同族={expect_company}；答案非空；有引用页")
    t0 = time.perf_counter()
    answer = None
    try:
        answer = engine.answer(question, lang="auto", top_k=10, session_id=sid,
                               return_evidence=True, use_cache=False)
    except Exception as exc:  # noqa: BLE001
        case["异常"] = f"{type(exc).__name__}: {exc}"
        case["异常堆栈"] = traceback.format_exc()[-2000:]
    wall_ms = round((time.perf_counter() - t0) * 1000, 2)
    case["耗时ms"] = wall_ms
    case["session_id"] = sid

    if answer is None:
        for label in ("未抛异常", "答案非空", "有引用页", "引用页合法(1..page_count)",
                      f"引用语料同族(预期:{expect_company})", "耗时被记录"):
            K.add_assert(case, label, False, case.get("异常", "无返回"))
        K.finalize_case(case)
        if verbose:
            print(f"    !! {case_id} {case['异常']}")
        return case

    text = str(getattr(answer, "answer", "") or "")
    cited = [int(p) for p in (getattr(answer, "cited_pages", None) or [])]
    citations = list(getattr(answer, "citations", None) or [])
    latency = dict(getattr(answer, "latency_ms", None) or {})
    analysis = dict(getattr(answer, "analysis", None) or {})
    resolved = str(analysis.get("question") or "")
    case["答案"] = text[:600]
    case["答案全文"] = text
    case["答案字数"] = len(text.strip())
    case["拒答"] = bool(getattr(answer, "refused", False))
    case["拒答原因"] = getattr(answer, "refusal_reason", None)
    case["引用页"] = cited
    case["引用语料"] = sorted({str(c.get("corpus") or "") for c in citations if c.get("corpus")})
    case["生成方式"] = getattr(answer, "generation_method", None)
    case["引擎耗时ms"] = latency
    case["实际送入检索的问句"] = resolved
    case["指代已消解"] = bool(resolved and resolved != question)
    _finish_turn(case, cited, citations, limits, wall_ms, latency, expect_company, text)
    K.finalize_case(case)
    if verbose:
        flag = "PASS" if case["通过"] else "FAIL"
        print(f"    {flag} {case_id} | 引用语料 {case['引用语料'][:1]} | {wall_ms:.0f}ms | "
              f"消解={'是' if case['指代已消解'] else '否'} | 送入检索={resolved[:60]}")
    return case


def _run_mt_turn(mt: Any, *, sid: str, case_id: str, name: str, question: str,
                 expect_company: str | None, limits: dict[str, int],
                 verbose: bool) -> dict[str, Any]:
    case = K.new_case(case_id, name, group=GROUP, question=question,
                      expect=f"MultiTurnEngine：引用语料同族={expect_company}；改写留痕")
    t0 = time.perf_counter()
    answer = None
    try:
        answer = mt.ask(question, session_id=sid, lang="auto", use_cache=False)
    except Exception as exc:  # noqa: BLE001
        case["异常"] = f"{type(exc).__name__}: {exc}"
        case["异常堆栈"] = traceback.format_exc()[-2000:]
    wall_ms = round((time.perf_counter() - t0) * 1000, 2)
    case["耗时ms"] = wall_ms
    case["session_id"] = sid

    if answer is None:
        for label in ("未抛异常", "答案非空", "有引用页", "引用页合法(1..page_count)",
                      f"引用语料同族(预期:{expect_company})", "耗时被记录"):
            K.add_assert(case, label, False, case.get("异常", "无返回"))
        K.finalize_case(case)
        if verbose:
            print(f"    !! {case_id} {case['异常']}")
        return case

    text = str(getattr(answer, "answer", "") or "")
    cited = [int(p) for p in (getattr(answer, "cited_pages", None) or [])]
    citations = list(getattr(answer, "citations", None) or [])
    latency = dict(getattr(answer, "latency_ms", None) or {})
    case["答案"] = text[:600]
    case["答案全文"] = text
    case["答案字数"] = len(text.strip())
    case["拒答"] = bool(getattr(answer, "refused", False))
    case["引用页"] = cited
    case["引用语料"] = sorted({str(c.get("corpus") or "") for c in citations if c.get("corpus")})
    case["生成方式"] = (getattr(answer, "base", None) or {}).get("generation_method")
    case["引擎耗时ms"] = latency
    case["改写策略"] = getattr(answer, "strategy", None)
    case["焦点实体"] = getattr(answer, "focus_entity", None)
    case["改写后问句"] = getattr(answer, "rewritten_query", "")
    case["轮次"] = getattr(answer, "turn_index", None)
    _finish_turn(case, cited, citations, limits, wall_ms, latency, expect_company, text)
    K.finalize_case(case)

    # --- 把 TC-M01 / TC-M02 的**原始设计判据**逐条核对（结果供报告回填）
    if case_id == "TC-M01":
        case["设计判据"] = [
            {"判据": "strategy == pronoun", "实测": case["改写策略"],
             "结果": case["改写策略"] == "pronoun"},
            {"判据": "rewritten_query 含「平安银行」", "实测": case["改写后问句"][:80],
             "结果": "平安银行" in str(case["改写后问句"])},
            {"判据": "第 2 轮答案含 183.12", "实测": K.contains(text, "183.12"),
             "结果": bool(K.contains(text, "183.12"))},
        ]
    if case_id == "TC-M02":
        case["设计判据"] = [
            {"判据": "strategy == ellipsis_entity_switch", "实测": case["改写策略"],
             "结果": case["改写策略"] == "ellipsis_entity_switch"},
            {"判据": "focus_entity 含「招商银行」", "实测": case["焦点实体"],
             "结果": "招商银行" in str(case["焦点实体"])},
            {"判据": "答案含 2,697.03 且 8.51%",
             "实测": f"2697.03={bool(K.contains(text, '2,697.03'))}, 8.51%={bool(K.contains(text, '8.51%'))}",
             "结果": bool(K.contains(text, "2,697.03") and K.contains(text, "8.51%"))},
            {"判据": "不得出现平安银行营收 1,379.58", "实测": bool(K.contains(text, "1,379.58")),
             "结果": not bool(K.contains(text, "1,379.58"))},
        ]
    if verbose:
        flag = "PASS" if case["通过"] else "FAIL"
        print(f"    {flag} {case_id} | 策略={case['改写策略']} 焦点={case['焦点实体']} | "
              f"改写={str(case['改写后问句'])[:52]} | {wall_ms:.0f}ms")
        if case.get("设计判据"):
            for item in case["设计判据"]:
                print(f"        [{'OK ' if item['结果'] else 'NG '}] {item['判据']} ⇒ {item['实测']}")
    return case


def run(engine: Any = None, mt: Any = None, *, verbose: bool = True, limit: int | None = None,
        warm: dict[str, Any] | None = None) -> dict[str, Any]:
    """执行三组多轮回归。``engine``/``mt`` 为 ``None`` 时自建（含预热）。

    ``limit``：每组只跑前 N 轮（小批量用，避免长时间占用 CPU 单实例 Ollama）。
    """
    K.bootstrap_env()
    questions = K.load_question_set()
    limits = K.page_limits()
    q07 = str((questions.get("Q07") or {}).get("question") or "")

    if engine is None or mt is None:
        engine, mt, warm = make_engines()

    started = time.perf_counter()
    cases: list[dict[str, Any]] = []

    print("\n[A 组] 同 session 追问（QAEngine + session_id）")
    engine.conversations.clear("reg-mt-a")
    for sid, name, question, company in (TURNS_A[:limit] if limit else TURNS_A):
        case = _run_qa_turn(engine, sid="reg-mt-a", case_id=f"TC-M-A{sid[-1]}", name=name,
                            question=question, expect_company=company, limits=limits,
                            verbose=verbose)
        cases.append(case)

    print("\n[B 组] 指代词「它 / 该行」（QAEngine）")
    engine.conversations.clear("reg-mt-b")
    for sid, name, question, company in (TURNS_B[:limit] if limit else TURNS_B):
        real_q = question if question is not None else q07
        case = _run_qa_turn(engine, sid="reg-mt-b", case_id=f"TC-M-B{sid[-1]}", name=name,
                            question=real_q, expect_company=company, limits=limits,
                            verbose=verbose)
        cases.append(case)

    print("\n[C 组] 对照：MultiTurnEngine（TC-M01 / TC-M02 判据核对）")
    mt.reset("reg-mt-c")
    for sid, name, question, company in (TURNS_C[:limit] if limit else TURNS_C):
        real_q = question if question is not None else q07
        case = _run_mt_turn(mt, sid="reg-mt-c", case_id=f"TC-M-C{sid[-1]}", name=name,
                            question=real_q, expect_company=company, limits=limits,
                            verbose=verbose)
        cases.append(case)

    elapsed = (time.perf_counter() - started) * 1000
    result = K.summarize(GROUP, cases, elapsed)
    result["预热"] = warm
    if verbose:
        K.print_case_table(cases, f"多轮对话回归（{len(cases)} 轮：A/B/C 三组）")
        print(f"通过 {result['通过']} / {result['用例数']}，失败 {result['失败']}，"
              f"本组耗时 {elapsed / 1000:.1f}s")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="工单7 多轮对话回归测试")
    parser.add_argument("--limit", type=int, default=None, help="每组只跑前 N 轮（小批量）")
    parser.add_argument("--out", default=str(K.TEST_DATA_DIR / "回归_多轮对话.json"))
    args = parser.parse_args(argv)

    facts = K.env_facts()
    ready, index_info = K.index_ready()
    print(f"== {K.WORK_ORDER} / 多轮对话回归 ==")
    print(f"解释器: {facts['解释器']} ({facts['解释器版本']})｜索引: {index_info}")

    result = run(limit=args.limit)
    payload = {"工单": K.WORK_ORDER, "生成时间": K.now_str(), "环境": facts,
               "索引": index_info, "组": result}
    K.save_json(Path(args.out), payload)
    print(f"\n逐轮结果已落盘：{args.out}")
    return 0 if result["失败"] == 0 and result["用例数"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
