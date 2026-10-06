"""可答性闸门逐题诊断（t13→t1 修复期工具，不参与交付判分）。

工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
阶段：研发 / 诊断脚本（可答性闸门修复期的定位工具）

用途：把「可答性判定」拆开逐题打印，定位**是哪一条闸门在放行/拦截**：
- 旧口径（``is_confident`` + ``is_answerable`` 实义词覆盖）逐条判定 → 复现修复前行为；
- 新闸门（``is_topic_covered``：rarest-first 主题词 + 意图↔取值类型）逐条判定；
- 主题词 IDF 明细、证据命中情况（用于判断"文档里到底有没有这个词"）。

用法::

    pwsh -NoProfile -File run_py.ps1 研发/scripts/diag_answerability_gate.py          # 只做判定，不生成
    pwsh -NoProfile -File run_py.ps1 研发/scripts/diag_answerability_gate.py --ask    # 附带真实 ask 结果
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.core.answerability import (  # noqa: E402
    detect_non_document_request,
    evidence_has_kind,
    expected_value_kinds,
    join_evidence,
    question_topic_tokens,
)
from app.core.qa_engine import get_qa_engine  # noqa: E402
from app.models.schemas import GoldenQA  # noqa: E402

TEST_DATA = Path(__file__).resolve().parents[2] / "测试" / "测试数据"
UNKNOWN_FILE = TEST_DATA / "unknown_questions.jsonl"
GOLDEN_FILE = TEST_DATA / "golden_qa.jsonl"
EN_QUESTIONS = (
    "What is the registered capital of Wuhan Xingtu Xinke Electronics Co., Ltd.?",
    "Who is the legal representative of the company?",
    "How much of the raised funds will be used to supplement working capital?",
    "Which technical standard did the company participate in formulating?",
    "What percentage of main business revenue came from the military sector during the reporting period?",
)

#: 多轮追问（t1 回归点）：第 3 轮曾被闸门以 ``rare_term_missing:哪里`` 误拒。
#: 为什么必须放进正常题集：只覆盖 10 中文 + 5 英文时，**追问/指代类问句**从来没被校验过，
#: 于是"闸门把合法追问判死"这类回归可以一路漏到在线套件才暴露（工单2 t1 实际发生过）。
MULTI_TURNS: tuple[tuple[str, str], ...] = (
    ("武汉兴图新科电子股份有限公司法定代表人是谁？", "第1轮·直问"),
    ("那它的注册资本呢？", "第2轮·指代追问"),
    ("它的注册地址在哪里？", "第3轮·继续追问"),
)

#: 扩展无关集（同类泛化）：与官方 12 条**同类但不同题**，用于证明修复是"规则"而不是"逐题特判"。
#: 其中前 3 条来自 captain 独立复核（首条曾在闸门 v1 上误答成合同叙述，页码 57）。
EXTRA_UNKNOWN: tuple[str, ...] = (
    "帮我写一首关于股票的诗",
    "帮我写一段公司简介",
    "预测一下公司明年利润",
    "帮我编一个关于公司的笑话",
    "你觉得公司股票值得买吗",
    "写一首关于春天的诗",
)

#: 「非文档型任务」分类器的正/负例（负例是**必须不能误判**的合法问法）。
REQUEST_CLASSIFICATION_CASES: tuple[tuple[str, str], ...] = (
    ("帮我写一首关于股票的诗", "creative_task"),
    ("帮我写一段公司简介", "creative_task"),
    ("写一首关于春天的诗", "creative_task"),
    ("预测一下公司明年利润", "prediction_task"),
    ("你觉得公司股票值得买吗", "subjective_task"),
    ("招股说明书写了公司的简介吗？", ""),
    ("报告期内，公司来自军用领域的收入分别是多少？", ""),
    ("公司参与制定了哪个技术标准？", ""),
    ("帮我总结一下公司的募资用途", ""),
    ("公司未来三年的发展战略是什么？", ""),
)


def _probe(engine, question: str, with_ask: bool) -> dict:
    """对单题做检索 + 两条口径判定 + （可选）真实 ask。"""
    analysis = engine._understanding.analyze(question)  # noqa: SLF001
    variants = engine._understanding.search_queries(analysis)  # noqa: SLF001
    contexts = engine._retriever.retrieve_multi(variants, analysis=analysis, top_k=5)  # noqa: SLF001
    query = analysis.search_query or analysis.original or question
    idf_lookup = engine._retriever._idf_lookup  # noqa: SLF001

    tokens = question_topic_tokens(query)
    idf = {token: round(idf_lookup(token), 3) for token in tokens}
    ranked = sorted(tokens, key=lambda word: idf_lookup(word), reverse=True)
    evidence = join_evidence(contexts)
    kinds = expected_value_kinds(analysis.intent)

    confident = engine._retriever.is_confident(contexts)  # noqa: SLF001
    legacy_ok = engine._retriever.is_answerable(query, contexts)  # noqa: SLF001
    gate_ok, gate_reason = engine._retriever.is_topic_covered(query, contexts, intent=analysis.intent)  # noqa: SLF001

    record = {
        "question": question,
        "language": analysis.language,
        "intent": analysis.intent,
        "query": query,
        "tokens": tokens,
        "idf": idf,
        "rarest": ranked[0] if ranked else "",
        "rarest_in_evidence": bool(ranked) and ranked[0] in evidence,
        "covered": [token for token in tokens if token in evidence],
        "kinds": kinds,
        "kind_present": evidence_has_kind(evidence, kinds) if kinds else None,
        "top_pages": [item.page for item in contexts[:3]],
        "top_cosine": round(max((item.vector_score for item in contexts), default=0.0), 4),
        "confident": confident,
        "legacy_answerable": legacy_ok,
        "legacy_decision": "ANSWER" if (confident and legacy_ok) else "UNKNOWN",
        "gate_ok": gate_ok,
        "gate_reason": gate_reason,
        "final_decision": "ANSWER" if (confident and legacy_ok and gate_ok) else "UNKNOWN",
        "top_snippet": (contexts[0].content[:70].replace("\n", " ") if contexts else ""),
    }
    if with_ask:
        answer = engine.ask(question)
        record["ask_is_unknown"] = bool(answer.is_unknown)
        record["ask_text"] = answer.answer[:60]
        record["ask_reason"] = answer.unknown_reason
        record["ask_mode"] = answer.mode
        record["ask_pages"] = [c.page for c in answer.citations]
    return record


def _print(record: dict, with_ask: bool) -> None:
    """按题打印一行摘要 + 关键明细。"""
    mark = "✅拒答" if record["final_decision"] == "UNKNOWN" else "❌误答"
    print(f"\n{mark} | {record['question'][:48]}")
    print(
        f"    intent={record['intent']!r} lang={record['language']} "
        f"confident={record['confident']} legacy={record['legacy_decision']} "
        f"gate={record['gate_ok']} ({record['gate_reason']})"
    )
    print(f"    主题词 IDF: {record['idf']}")
    print(
        f"    最罕见词={record['rarest']!r} 在证据中={record['rarest_in_evidence']} "
        f"命中={record['covered']}"
    )
    print(
        f"    期望取值类型={record['kinds']} 证据含该类型={record['kind_present']} "
        f"top页={record['top_pages']} cos={record['top_cosine']}"
    )
    print(f"    top片段: {record['top_snippet']}")
    if with_ask:
        print(
            f"    ask → is_unknown={record.get('ask_is_unknown')} "
            f"mode={record.get('ask_mode')} reason={record.get('ask_reason')!r} "
            f"pages={record.get('ask_pages')} text={record.get('ask_text')!r}"
        )


def _signal_demo(engine) -> bool:
    """人工构造证据，分别验证闸门的两个信号确实在工作（不依赖真实检索结果）。

    信号①：rarest-first 主题词必须出现在证据里；
    信号②：意图 ↔ 取值类型一致性（**只在主题词覆盖通过时才轮到它**，故必须单独构造）。
    """
    from app.core.answerability import check  # noqa: PLC0415
    from app.models.schemas import Chunk, RetrievedChunk  # noqa: PLC0415

    def fake(content: str) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                chunk=Chunk(chunk_id="c_demo", doc_id="demo", page=52, content=content),
                score=0.9,
                vector_score=0.9,
            )
        ]

    idf_lookup = engine._retriever._idf_lookup  # noqa: SLF001
    cases = (
        (
            "信号② 触发：问注册资本，证据覆盖该公司但**无金额**",
            "武汉兴图新科电子股份有限公司注册资本是多少？",
            "注册资本",
            "第五节 发行人基本情况 公司名称：武汉兴图新科电子股份有限公司 注册资本：待定 法定代表人：程家明",
            (False, "value_kind_missing:amount"),
        ),
        (
            "信号② 放行：同一问题 + 证据含金额",
            "武汉兴图新科电子股份有限公司注册资本是多少？",
            "注册资本",
            "第五节 发行人基本情况 公司名称：武汉兴图新科电子股份有限公司 注册资本：5,520 万元",
            (True, "topic_covered"),
        ),
        (
            "信号① 触发：含公司名但问的是文档没记的事（食堂）",
            "武汉兴图新科电子股份有限公司的食堂今天中午吃什么？",
            "其他",
            "第五节 发行人基本情况 公司名称：武汉兴图新科电子股份有限公司 注册资本：5,520 万元",
            (False, "rare_term_missing:食堂"),
        ),
    )
    print("\n" + "=" * 100)
    print("③ 闸门信号功能验证（人工构造证据：证明两个信号各自可独立拦截）")
    print("=" * 100)
    for label, question, intent, evidence, expected in cases:
        got = check(question, intent, fake(evidence), idf_lookup)
        mark = "✅" if got == expected else "❌"
        print(f"  {mark} {label}")
        print(f"      实际={got}  期望={expected}")
    return all(
        check(question, intent, fake(evidence), idf_lookup) == expected
        for _label, question, intent, evidence, expected in cases
    )


def main() -> int:
    """诊断入口；返回 0 表示"无关集全部拦截 且 正常集全部放行"。"""
    with_ask = "--ask" in sys.argv
    engine = get_qa_engine()
    if not engine.load_index():
        print("业务失败：索引未就绪", file=sys.stderr)
        return 2

    unknown = [
        json.loads(line) for line in UNKNOWN_FILE.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    golden = [
        GoldenQA(**json.loads(line))
        for line in GOLDEN_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    print("\n" + "=" * 100)
    print(f"① 无关问题集（官方 {len(unknown)} 条 + 扩展同类 {len(EXTRA_UNKNOWN)} 条）：期望「最终判定 = UNKNOWN」")
    print("=" * 100)
    legacy_answered = 0
    gate_answered = 0
    for index, item in enumerate(unknown, start=1):
        record = _probe(engine, item["question"], with_ask=with_ask)
        record["index"] = index
        _print(record, with_ask)
        legacy_answered += record["legacy_decision"] == "ANSWER"
        gate_answered += record["final_decision"] == "ANSWER"
    print("\n---- 扩展同类（证明是规则泛化，不是逐题特判）----")
    extra_answered = 0
    for index, question in enumerate(EXTRA_UNKNOWN, start=1):
        record = _probe(engine, question, with_ask=with_ask)
        record["index"] = f"E{index}"
        _print(record, with_ask)
        extra_answered += record["final_decision"] == "ANSWER"

    print("\n" + "=" * 100)
    print(f"② 正常问题集（中文 {len(golden)} 条 + 英文 {len(EN_QUESTIONS)} 条）：期望「最终判定 = ANSWER」")
    print("=" * 100)
    rejected_zh: list[str] = []
    for item in golden:
        record = _probe(engine, item.question, with_ask=with_ask)
        if record["final_decision"] != "ANSWER":
            rejected_zh.append(f"Q{item.id}:{record['gate_reason']}")
        _print(record, with_ask)
    rejected_en: list[str] = []
    for question in EN_QUESTIONS:
        record = _probe(engine, question, with_ask=with_ask)
        if record["final_decision"] != "ANSWER":
            rejected_en.append(f"{question[:28]}:{record['gate_reason']}")
        _print(record, with_ask)

    signal_ok = _signal_demo(engine)

    print("\n" + "=" * 100)
    print("③b 非文档型任务分类器正/负例（负例必须**不**被误判）")
    print("=" * 100)
    classify_ok = True
    for question, expected in REQUEST_CLASSIFICATION_CASES:
        got = detect_non_document_request(question)
        ok = got == expected
        classify_ok = classify_ok and ok
        print(f"  [{'✅' if ok else '❌'}] {question[:34]:<36} 实际={got!r} 期望={expected!r}")

    print("\n" + "=" * 100)
    print("④ 多轮追问（同会话顺序提问）：期望三轮**均作答**（t1 回归点）")
    print("=" * 100)
    rejected_turns: list[str] = []
    conversation_id = engine.new_conversation("diag-多轮")
    for question, label in MULTI_TURNS:
        answer = engine.ask(question, conversation_id=conversation_id)
        text = (answer.answer or "").strip()
        ok = not answer.is_unknown and bool(text)
        if not ok:
            reason = ""
            analysis = answer.query_analysis
            probe_query = (analysis.search_query or analysis.original) if analysis else question
            gate_ok, gate_reason = engine._retriever.is_topic_covered(  # noqa: SLF001
                probe_query, answer.retrieved or [], intent=analysis.intent if analysis else ""
            )
            reason = f"{answer.unknown_reason}|{probe_query}|gate={gate_reason}" if not gate_ok else answer.unknown_reason
            rejected_turns.append(f"{label}:{reason}")
        print(f"  [{'✅作答' if ok else '❌误拒'}] {label} {question!r} → {text[:44]!r} ({answer.mode})")

    print("\n" + "=" * 100)
    print("汇总")
    print(f"  官方无关集 {len(unknown)} 条：旧口径放行 {legacy_answered} 条 → 新闸门放行 {gate_answered} 条（目标 0）")
    print(f"  扩展同类 {len(EXTRA_UNKNOWN)} 条：新闸门放行 {extra_answered} 条（目标 0）")
    print(f"  中文正常题误拒：{len(rejected_zh)}/{len(golden)} {rejected_zh}")
    print(f"  英文正常题误拒：{len(rejected_en)}/{len(EN_QUESTIONS)} {rejected_en}")
    print(f"  多轮追问误拒：{len(rejected_turns)}/{len(MULTI_TURNS)} {rejected_turns}")
    print(f"  闸门信号功能验证：{'全部符合预期' if signal_ok else '存在不符'}")
    print(f"  非文档型任务分类器：{'全部符合预期' if classify_ok else '存在不符'}")
    ok = (
        gate_answered == 0
        and extra_answered == 0
        and not rejected_zh
        and not rejected_en
        and not rejected_turns
        and signal_ok
        and classify_ok
    )
    print(f"  结论：{'PASS ✅' if ok else 'FAIL ❌'}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
