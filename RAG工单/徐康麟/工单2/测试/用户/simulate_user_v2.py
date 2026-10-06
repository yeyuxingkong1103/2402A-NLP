"""用户测试：模拟真实用户操作序列（本机无 Streamlit，走 core 端到端引擎）。

工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
阶段：测试 / 用户（T5 产出物 2/3）

模拟的用户旅程（每一步都记录真实耗时与结果，失败不中断）：
  ① 启动服务      —— 构建 QAEngine + 预热（嵌入/LLM/索引）
  ② 加载 PDF      —— 校验索引与文档元数据就绪（548 页 / 分块数）
  ③ 提问 10 个工单问题 —— 用**确定性 check_answer** 逐题判分（口径不得放宽）
  ④ 多轮追问      —— 含指代追问，校验改写、不串题、历史落库
  ⑤ 英文提问      —— 5 题，校验英文作答与跨语言引用
  ⑥ 无关问题      —— 12 题，应回「不清楚」
  ⑦ 异常演示      —— 空问题 / 超长输入，校验友好提示

产出：
  - stdout：分步耗时与结论汇总（可直接贴进报告）
  - ``优化/评估结果/用户模拟报告.md`` + ``用户模拟结果.json``
  - 日志自动写入 ``部署/日志/``（app.log / error.log / rag_trace.jsonl）

退出码语义（2026-10-03 修正，见 ``exit_code_for``）：
  ``0`` = **全部步骤通过**；``1`` = **任一步骤未通过**；``2`` = 启动失败 / 未产生任何步骤。
  ⚠️ 修正前 ``_finalize`` **恒返回 0**，故「6/7 步通过」时退出码也是 0，
  「退出码 0」不能作为通过依据（该缺陷由 tester 与 verifier 各自独立发现）。

运行命令（工作目录 = E:\\gao6gongdan\\工单2）::

    pwsh -NoProfile -File run_py.ps1 测试/用户/simulate_user_v2.py
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# 路径与编码
# --------------------------------------------------------------------------
TESTS_DIR = Path(__file__).resolve().parent.parent          # 测试/
PROJECT_ROOT = TESTS_DIR.parent                              # 工单2
SOURCE_ROOT = PROJECT_ROOT / "研发"
TEST_DATA = TESTS_DIR / "测试数据"
EVAL_RESULTS = PROJECT_ROOT / "优化" / "评估结果"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:  # pragma: no cover
    pass

PDF_PAGE_MAX = 548
FIRST_TOKEN_BUDGET_MS = 3000.0

EN_QUESTIONS: tuple[tuple[int, str], ...] = (
    (543, "What is the registered capital of Wuhan Xingtu Xinke Electronics Co., Ltd.?"),
    (531, "Who is the legal representative of the company?"),
    (207, "How much of the raised funds will be used to supplement working capital?"),
    (95, "Which technical standard did the company participate in formulating?"),
    (33, "What percentage of main business revenue came from the military sector during the reporting period?"),
)

MULTI_TURNS = (
    # (问题, 标签, 期望内容关键词组——任一命中即算该轮答对)
    ("武汉兴图新科电子股份有限公司法定代表人是谁？", "第1轮·直问", ("程家明",)),
    ("那它的注册资本呢？", "第2轮·指代追问", ("5,520", "5520")),
    ("它的注册地址在哪里？", "第3轮·继续追问", ("关山大道", "注册地址")),
)


@dataclass
class Step:
    """一步用户操作的结果。"""

    name: str
    elapsed_ms: float
    ok: bool
    detail: str = ""
    data: dict = field(default_factory=dict)


class Journey:
    """收集全过程步骤与统计。"""

    def __init__(self) -> None:
        self.steps: list[Step] = []
        self.rows: list[dict] = []

    def add(self, name: str, elapsed_ms: float, ok: bool, detail: str = "", **data) -> Step:
        step = Step(name, round(elapsed_ms, 2), ok, detail, data)
        self.steps.append(step)
        mark = "✅" if ok else "❌"
        print(f"{mark} {name}（{step.elapsed_ms:.0f} ms）{detail}")
        return step


def load_golden() -> list[dict]:
    """读取 10 个工单问题。"""
    path = TEST_DATA / "golden_qa.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_unknown() -> list[dict]:
    """读取自造无关问题集。"""
    path = TEST_DATA / "unknown_questions.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    journey = Journey()
    print("=" * 78)
    print("用户测试·模拟真实操作序列（工单2：人工智能NLP-RAG-基于PDF文档的问答系统优化）")
    print("=" * 78)

    # ---------------------------------------------------------------- ① 启动
    t0 = time.perf_counter()
    try:
        from app.core.qa_engine import QAEngine

        engine = QAEngine()
        started = time.perf_counter() - t0
        journey.add("① 启动服务（构建 QAEngine）", started * 1000, True,
                    f"engine={type(engine).__name__}")
    except Exception as exc:  # noqa: BLE001
        journey.add("① 启动服务（构建 QAEngine）", (time.perf_counter() - t0) * 1000, False,
                    f"{type(exc).__name__}: {exc}")
        return _finalize(journey, fatal=True)

    # ---------------------------------------------------------------- ② 加载 PDF / 索引
    t0 = time.perf_counter()
    try:
        loaded = engine.load_index()
        warm = engine.warmup()
        health = engine.health()
        idx = health.get("index") or {}
        t = (time.perf_counter() - t0) * 1000
        journey.add("② 加载 PDF / 索引 + 预热", t, bool(loaded and idx.get("ready")),
                    f"index_ready={idx.get('ready')} 分块={idx.get('count')} "
                    f"dim={idx.get('dimension')} warmup_keys={sorted(warm)[:4]}",
                    index=idx, warmup=warm)
    except Exception as exc:  # noqa: BLE001
        journey.add("② 加载 PDF / 索引 + 预热", (time.perf_counter() - t0) * 1000, False,
                    f"{type(exc).__name__}: {exc}")

    # ---------------------------------------------------------------- ③ 10 个工单问题
    from app.core.evaluator import get_evaluator

    evaluator = get_evaluator()
    golden = load_golden()
    correct = 0
    ft_values: list[float] = []
    cite_total = cite_valid = 0
    print("\n--- ③ 提问 10 个工单问题（确定性 check_answer 判分）---")
    for item in golden:
        t0 = time.perf_counter()
        try:
            answer = engine.ask(item["question"])
            ok, note = evaluator.check_answer(answer.answer, item["answer"])
            is_unknown = bool(answer.is_unknown)
        except Exception as exc:  # noqa: BLE001
            answer, ok, note, is_unknown = None, False, f"{type(exc).__name__}: {exc}", False
        elapsed = (time.perf_counter() - t0) * 1000
        correct += bool(ok)
        if answer is not None:
            ft_values.append(float(answer.first_token_ms))
            cits = answer.citations
            cite_total += len(cits)
            cite_valid += sum(1 for c in cits if c.chunk_id and 1 <= c.page <= PDF_PAGE_MAX)
        journey.rows.append({
            "id": item["id"], "question": item["question"],
            "answer": getattr(answer, "answer", ""), "mode": getattr(answer, "mode", ""),
            "is_correct": bool(ok), "note": note, "is_unknown": is_unknown,
            "first_token_ms": getattr(answer, "first_token_ms", 0.0),
            "total_ms": getattr(answer, "total_ms", 0.0),
            "pages": [c.page for c in getattr(answer, "citations", [])],
            "elapsed_ms": round(elapsed, 2),
        })
        mark = "✅" if ok else "❌"
        print(f"   {mark} Q{item['id']:<4} 首字={getattr(answer, 'first_token_ms', 0):>7.0f}ms "
              f"端到端={getattr(answer, 'total_ms', 0):>8.0f}ms "
              f"引用页={[c.page for c in getattr(answer, 'citations', [])][:3]} | "
              f"{getattr(answer, 'answer', '')[:52]!r}")
    accuracy = correct / len(golden) if golden else 0.0
    ft_max = max(ft_values) if ft_values else 0.0
    ft_avg = sum(ft_values) / len(ft_values) if ft_values else 0.0
    print(f"   → 判对 {correct}/{len(golden)} = {accuracy:.0%}；"
          f"首字 均值 {ft_avg:.0f}ms / 最大 {ft_max:.0f}ms（预算 {FIRST_TOKEN_BUDGET_MS:.0f}ms）")
    print(f"   → 引用 {cite_valid}/{cite_total} 条合法")
    journey.add("③ 10 个工单问题", 0.0, accuracy >= 0.90,
                f"准确率={accuracy:.0%}（{correct}/{len(golden)}）首字≤3s={'是' if ft_max <= FIRST_TOKEN_BUDGET_MS else '否'}"
                f" 引用通过率={cite_valid / cite_total if cite_total else 0:.0%}",
                accuracy=accuracy, correct=correct, first_token_avg=ft_avg, first_token_max=ft_max,
                citation_total=cite_total, citation_valid=cite_valid)

    # ---------------------------------------------------------------- ④ 多轮追问
    print("\n--- ④ 多轮追问（含指代）---")
    t0 = time.perf_counter()
    conv_id = None
    turns: list[dict] = []
    try:
        conv_id = engine.new_conversation("用户模拟会话")
        for question, label, expect in MULTI_TURNS:
            t1 = time.perf_counter()
            answer = engine.ask(question, conversation_id=conv_id)
            text = answer.answer or ""
            content_ok = any(token in text for token in expect) and not answer.is_unknown
            turns.append({
                "label": label, "question": question, "answer": text,
                "pages": [c.page for c in answer.citations],
                "elapsed_ms": round((time.perf_counter() - t1) * 1000, 2),
                "is_unknown": answer.is_unknown, "content_ok": content_ok,
            })
            print(f"   {'✅' if content_ok else '❌'} {label}：{question!r}\n"
                  f"        → {text[:76]!r} 引用页={[c.page for c in answer.citations][:3]}")
        # 每轮都答了 != 每轮都答对：追问轮必须答对**该轮的字段**，否则不算通过（验收 5）
        answered = all(t["answer"].strip() and not t["is_unknown"] for t in turns)
        ok = answered and all(t["content_ok"] for t in turns)
        bad = [t["label"] for t in turns if not t["content_ok"]]
        detail = (f"{len(turns)} 轮均有答案" if answered else "存在空答案/误拒答") + \
                 (f"；但内容不符的轮次：{bad}" if bad else "；各轮内容均命中该轮字段")
        journey.add("④ 多轮追问（3 轮含指代）", (time.perf_counter() - t0) * 1000, ok,
                    detail + f"；会话={conv_id}", turns=turns)
    except Exception as exc:  # noqa: BLE001
        journey.add("④ 多轮追问（3 轮含指代）", (time.perf_counter() - t0) * 1000, False,
                    f"{type(exc).__name__}: {exc}")

    # ---------------------------------------------------------------- ⑤ 英文提问
    print("\n--- ⑤ 英文提问 ---")
    t0 = time.perf_counter()
    en_rows: list[dict] = []
    en_ok = 0
    try:
        for qid, question in EN_QUESTIONS:
            t1 = time.perf_counter()
            answer = engine.ask(question)
            text = answer.answer or ""
            cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
            ratio = cjk / max(1, len(text))
            good = (not answer.is_unknown and answer.language == "en"
                    and answer.citations and ratio <= 0.20)
            en_ok += good
            en_rows.append({"id": qid, "question": question, "answer": text,
                            "language": answer.language, "is_unknown": answer.is_unknown,
                            "cjk_ratio": round(ratio, 3),
                            "pages": [c.page for c in answer.citations],
                            "elapsed_ms": round((time.perf_counter() - t1) * 1000, 2),
                            "ok": good})
            print(f"   {'✅' if good else '❌'} Q{qid:<4} lang={answer.language} "
                  f"中文占比={ratio:.0%} 引用页={[c.page for c in answer.citations][:3]} | {text[:52]!r}")
        journey.add("⑤ 英文提问（5 题）", (time.perf_counter() - t0) * 1000, en_ok == len(EN_QUESTIONS),
                    f"合格 {en_ok}/{len(EN_QUESTIONS)}", rows=en_rows)
    except Exception as exc:  # noqa: BLE001
        journey.add("⑤ 英文提问（5 题）", (time.perf_counter() - t0) * 1000, False,
                    f"{type(exc).__name__}: {exc}")

    # ---------------------------------------------------------------- ⑥ 无关问题
    print("\n--- ⑥ 无关问题（应回复「不清楚」）---")
    t0 = time.perf_counter()
    unknown_rows: list[dict] = []
    unknown_ok = 0
    try:
        for item in load_unknown():
            answer = engine.ask(item["question"])
            good = bool(answer.is_unknown) and answer.answer.strip() == "不清楚"
            unknown_ok += good
            unknown_rows.append({"question": item["question"], "answer": answer.answer,
                                 "is_unknown": answer.is_unknown, "ok": good})
            if not good:
                print(f"   ❌ {item['question'][:34]!r} → {answer.answer[:46]!r}")
        journey.add("⑥ 无关问题（应回「不清楚」）", (time.perf_counter() - t0) * 1000,
                    unknown_ok == len(unknown_rows),
                    f"正确拒答 {unknown_ok}/{len(unknown_rows)}", rows=unknown_rows)
    except Exception as exc:  # noqa: BLE001
        journey.add("⑥ 无关问题（应回「不清楚」）", (time.perf_counter() - t0) * 1000, False,
                    f"{type(exc).__name__}: {exc}")

    # ---------------------------------------------------------------- ⑦ 异常演示
    print("\n--- ⑦ 异常输入演示 ---")
    t0 = time.perf_counter()
    fault_rows: list[dict] = []
    try:
        for bad, label in (("", "空问题"), ("   ", "空白串"), ("啊" * 3000, "超长输入")):
            try:
                answer = engine.ask(bad)
                text = answer.answer or ""
                friendly = bool(text.strip()) and "Traceback" not in text
                fault_rows.append({"label": label, "answer": text[:60], "friendly": friendly})
                print(f"   {'✅' if friendly else '❌'} {label} → {text[:56]!r}")
            except Exception as exc:  # noqa: BLE001
                fault_rows.append({"label": label, "answer": f"{type(exc).__name__}: {exc}",
                                   "friendly": False})
                print(f"   ❌ {label} 抛出 {type(exc).__name__}: {exc}")
        journey.add("⑦ 异常输入（空/空白/超长）", (time.perf_counter() - t0) * 1000,
                    all(r["friendly"] for r in fault_rows),
                    f"{len(fault_rows)} 类均给友好文案" if all(r["friendly"] for r in fault_rows)
                    else "存在非友好响应", rows=fault_rows)
    except Exception as exc:  # noqa: BLE001
        journey.add("⑦ 异常输入（空/空白/超长）", (time.perf_counter() - t0) * 1000, False,
                    f"{type(exc).__name__}: {exc}")

    return _finalize(journey)


def exit_code_for(journey: Journey, fatal: bool = False) -> int:
    """退出码语义（2026-10-03 修正）。

    - ``0``：**全部步骤通过**；
    - ``1``：**任一步骤未通过**（含非致命步骤，例如第 ⑥ 步无关问题拒答不达标）；
    - ``2``：启动失败 / 未产生任何步骤（无证据可判）。

    修正原因：修正前 ``_finalize`` **恒返回 0**，实测「6/7 步通过」时退出码仍为 0，
    使「退出码 0」对用户脚本成为**弱证据**（会被误读为通过）。
    """
    if fatal or not journey.steps:
        return 2
    return 0 if all(step.ok for step in journey.steps) else 1


def self_test_exit_code() -> int:
    """**实测**退出码语义（不提问、不写报告）：6/7→1、启动失败→2、全通过→0。

    为什么需要独立入口：退出码语义是"退出码 0 能否作为通过依据"的关键，
    必须能被第三方（verifier）**直接复跑**，而不是只写在报告里。
    本模式构造三种合成旅程，调用**真实**的 ``exit_code_for`` 并断言其返回值；
    自身退出码 0 = 语义全部正确，1 = 有语义不符。**不落盘**任何产物（避免覆盖正式报告）。
    """
    cases: list[tuple[str, Journey, bool, int]] = []

    # ① 6/7：7 步中 1 步失败（对应"无关问题拒答 11/12"这类非致命失败）
    six_of_seven = Journey()
    for index in range(1, 8):
        six_of_seven.add(f"自测步骤{index}", 1.0, index != 7,
                         "合成旅程（非真实提问）" + ("，本步失败" if index == 7 else ""))
    cases.append(("6/7 步通过（第 7 步失败）", six_of_seven, False, 1))

    # ② 全部通过
    all_ok = Journey()
    for index in range(1, 8):
        all_ok.add(f"自测步骤{index}", 1.0, True, "合成旅程（非真实提问）")
    cases.append(("7/7 步全部通过", all_ok, False, 0))

    # ③ 启动失败（fatal）：无任何步骤
    cases.append(("启动失败（fatal，无步骤）", Journey(), True, 2))

    print("=" * 78)
    print("退出码语义自测（调用真实 exit_code_for；不提问、不写报告）")
    print("=" * 78)
    ok_all = True
    for label, journey, fatal, expected in cases:
        actual = exit_code_for(journey, fatal)
        good = actual == expected
        ok_all &= good
        print(f"{'✅' if good else '❌'} {label} → exit_code={actual}（期望 {expected}）")
    print("=" * 78)
    print(f"语义自测结论：{'全部符合 ✅' if ok_all else '存在不符 ❌'}；本进程退出码 = {0 if ok_all else 1}")
    return 0 if ok_all else 1


def _finalize(journey: Journey, fatal: bool = False, write_report: bool = True) -> int:
    """写报告并打印汇总；返回**真实退出码**（见 ``exit_code_for``）。

    ``write_report=False`` 供 ``self_test_exit_code`` 等自测路径使用（不得覆盖正式报告）。
    """
    passed = sum(1 for s in journey.steps if s.ok)
    total = len(journey.steps)
    code = exit_code_for(journey, fatal)
    print("\n" + "=" * 78)
    print(f"用户旅程汇总：{passed}/{total} 步通过" + ("（**启动失败，旅程中断**）" if fatal else ""))
    for step in journey.steps:
        print(f"   {'✅' if step.ok else '❌'} {step.name}（{step.elapsed_ms:.0f} ms）")
    print(f"退出码 = {code}（0=全部步骤通过；1=存在未通过步骤；2=启动失败/无步骤）")
    print("=" * 78)

    if not write_report:
        print("[自测模式] 跳过报告落盘（不覆盖 优化/评估结果/ 下的正式产物）")
        return code

    EVAL_RESULTS.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "steps": [{"name": s.name, "elapsed_ms": s.elapsed_ms, "ok": s.ok, "detail": s.detail}
                  for s in journey.steps],
        "summary": {"passed": passed, "total": total, "fatal": fatal},
        "questions": journey.rows,
    }
    json_path = EVAL_RESULTS / "用户模拟结果.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 用户模拟测试报告",
        "",
        "> 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化",
        "> 阶段：测试 / 用户（T5 产出物 2/3）",
        "> 运行命令：`pwsh -NoProfile -File run_py.ps1 测试/用户/simulate_user_v2.py`",
        f"> 生成时间：{payload['generated_at']}",
        "",
        "## 1. 用户旅程分步结果",
        "",
        "| 步骤 | 通过 | 耗时(ms) | 说明 |",
        "| --- | --- | --- | --- |",
    ]
    for s in journey.steps:
        lines.append(f"| {s.name} | {'✅' if s.ok else '❌'} | {s.elapsed_ms:.0f} | {s.detail} |")
    verdict = "通过" if code == 0 else ("启动失败（无证据）" if code == 2 else "不通过")
    lines += ["", f"**汇总：{passed}/{total} 步通过；整体判定：{verdict}（退出码 {code}）。**", ""]
    if journey.rows:
        lines += [
            "## 2. 10 个工单问题逐题结果（确定性 `check_answer` 判分）",
            "",
            "| 题号 | 判对 | 首字(ms) | 端到端(ms) | 引用页 | 回答 | 判分说明 |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for r in journey.rows:
            lines.append(f"| {r['id']} | {'✅' if r['is_correct'] else '❌'} | "
                         f"{r['first_token_ms']:.0f} | {r['total_ms']:.0f} | {r['pages'][:3]} | "
                         f"{r['answer'][:60]} | {r['note'][:50]} |")
        lines.append("")
    lines += [
        "## 3. 口径声明",
        "",
        "- 判分为工单1 的**确定性** `check_answer`（不依赖 LLM 裁判），`FUZZY_THRESHOLD=0.62` **未修改**，",
        "  故与「优化前 0.50」同尺可比。",
        "- 引用合法性按「页码 ∈ [1,548] 且带真实 chunk_id」判定，**不**以 `golden_qa.jsonl.evidence_pages`",
        "  为唯一真值（该字段存在错标，见 `环境事实.md` §5.1）。",
        "- 日志由引擎自动写入 `部署/日志/{app.log,error.log,rag_trace.jsonl}`。",
        "- **退出码语义（2026-10-03 修正）**：`0` = 全部步骤通过；`1` = 任一步骤未通过；"
        "`2` = 启动失败/无步骤。修正前本脚本恒返回 0，故「退出码 0」曾被误当作通过依据。",
        "",
    ]
    md_path = EVAL_RESULTS / "用户模拟报告.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n报告已写入：{md_path}")
    print(f"结果 JSON：{json_path}")
    return code


if __name__ == "__main__":
    if "--self-test-exit-code" in sys.argv[1:]:
        raise SystemExit(self_test_exit_code())
    raise SystemExit(main())
