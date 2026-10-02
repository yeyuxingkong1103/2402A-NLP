"""用户测试：模拟真实用户操作（工单 9.3）。

模拟流程：

1. 打开系统（装载索引，进入首页）；
2. 依次提出工单第 7 节固定的 10 个问题；
3. 多轮对话：先问一个完整问题，再在同一会话中做一次省略式追问（"那注册资本呢？"）；
4. 提一个与《招股说明书1.pdf》无关的问题（"今天天气怎么样？"），观察兜底行为；
5. 对最后一条回答点赞（验证反馈功能）；
6. 汇总统计并把完整过程写入 ``logs/user_simulation.log``。

运行方式::

    $env:PYTHONPATH="E:\\gao6gongdan\\工单1"
    cd E:\\gao6gongdan\\工单1
    & "E:\\gao6gongdan\\工单1\\.gao6gongdan-src\\python.exe" tests/user/simulate_user.py

说明：
- 使用 ``force_extractive=True``：本机没有 LLM 服务，抽取式路径同样基于 PDF 原文作答；
- 该脚本作为"用户操作"，会把对话与反馈写入真实索引库（``data/index/rag.sqlite3``），
  这正是用户测试要验证的落库行为；pytest 用例则使用临时库副本，二者互不影响；
- 退出码：0 表示 10 题都拿到非空答案；1 表示有题目未拿到答案；3 表示索引未就绪。
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

# --------------------------------------------------------------------------
# 路径与编码准备
#
# 目录结构：<root>/测试/tests/user/simulate_user.py，源码在 <root>/研发/app/
#   parents[0]=user parents[1]=tests parents[2]=测试 parents[3]=项目根
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = PROJECT_ROOT / "研发"
for _candidate in (SOURCE_ROOT, PROJECT_ROOT):
    if str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

try:  # 让 Windows 控制台也能正确显示中文
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - 非 Windows 或流不可重配时忽略
    pass

from app.core.config import get_settings  # noqa: E402
from app.core.evaluator import Evaluator  # noqa: E402
from app.core.logging_conf import setup_logging  # noqa: E402
from app.core.qa_engine import QAEngine  # noqa: E402
from app.models.schemas import GoldenQA  # noqa: E402

UNRELATED_QUESTION = "今天天气怎么样？"
FOLLOWUP_BASE_QUESTION = "武汉兴图新科电子股份有限公司法定代表人是谁？"
FOLLOWUP_QUESTION = "那注册资本呢？"

_LOG_LINES: list[str] = []


def emit(message: str = "") -> None:
    """同时输出到控制台与日志缓冲。"""
    print(message)
    _LOG_LINES.append(message)


def write_log(path: Path) -> None:
    """把完整过程写入日志文件（UTF-8）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(_LOG_LINES) + "\n", encoding="utf-8")


def load_golden(path: Path) -> list[GoldenQA]:
    """读取工单固定的 10 道标准问答。"""
    if not path.exists():
        return []
    return [
        GoldenQA(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def format_answer(answer, limit: int = 160) -> str:
    """把答案格式化成一行便于阅读的文本。"""
    text = answer.answer.replace("\n", " ")
    if len(text) > limit:
        text = text[:limit] + "…"
    return text


def report_answer(step: str, question: str, answer, golden: GoldenQA | None, evaluator: Evaluator) -> bool:
    """输出一次问答的完整结果，返回该题是否拿到了非空答案。"""
    emit(f"\n[{step}] 提问：{question}")
    if answer.is_unknown:
        emit(f"      回答：不清楚（原因：{answer.unknown_reason}）")
    else:
        emit(f"      回答：{format_answer(answer)}")

    pages = "、".join(str(page) for page in sorted({c.page for c in answer.citations})) or "（无）"
    emit(f"      引用页码：{pages}")
    if answer.citations:
        emit(f"      引用片段：{', '.join(c.chunk_id for c in answer.citations[:3])}")
    emit(
        f"      检索片段数：{answer.retrieved_count}；首字耗时：{answer.first_token_ms:.1f}ms；"
        f"总耗时：{answer.total_ms:.1f}ms"
    )

    if golden is not None:
        is_correct, note = evaluator.check_answer(answer.answer, golden.answer)
        emit(f"      与标准答案比对：{'✅ 一致' if is_correct else '❌ 不一致'}（{note}）")
        emit(f"      标准答案：{golden.answer[:100]}")

    return bool(answer.answer.strip())


def main() -> int:
    """执行一次完整的用户模拟，返回进程退出码。"""
    setup_logging()
    settings = get_settings()
    evaluator = Evaluator()
    started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_path = settings.paths.logs / "user_simulation.log"

    emit("=" * 78)
    emit("RAG 问答系统 · 用户操作模拟")
    emit(f"开始时间：{started_at}")
    emit(f"语料：{settings.paths.default_pdf}")
    emit(f"索引库：{settings.paths.sqlite_path}")
    emit("=" * 78)

    # ---- 步骤 1：打开系统 ----
    emit("\n【步骤 1】打开系统（装载索引）…")
    open_started = time.perf_counter()
    engine = QAEngine(force_extractive=True)
    stats = engine.stats()
    emit(f"      装载耗时：{(time.perf_counter() - open_started) * 1000:.0f}ms")
    emit(
        f"      文档：{stats['doc_title']}；页数：{stats['pages']}；分块：{stats['chunks']}；"
        f"向量：{stats['vector_count']}（{stats['vector_backend']}，{stats['embedder']}）"
    )
    if not stats["index_ready"]:
        emit("\n[错误] 索引未就绪，请先运行：python scripts/build_index.py")
        write_log(log_path)
        emit(f"\n日志已写入：{log_path}")
        return 3

    conversation_id = engine.new_conversation(title="用户模拟会话")
    emit(f"      会话 ID：{conversation_id}")

    golden_items = load_golden(settings.paths.data_eval / "golden_qa.jsonl")
    golden_by_question = {item.question: item for item in golden_items}

    # ---- 步骤 2：工单 10 题 ----
    emit("\n【步骤 2】依次提出工单固定的 10 个问题")
    answered = 0
    first_tokens: list[float] = []
    for index, item in enumerate(golden_items, start=1):
        answer = engine.ask(item.question, conversation_id=conversation_id)
        if report_answer(f"2.{index}", item.question, answer, golden_by_question.get(item.question), evaluator):
            answered += 1
        first_tokens.append(answer.first_token_ms)

    # ---- 步骤 3：多轮对话 + 省略式追问 ----
    emit("\n【步骤 3】多轮对话：先问一个完整问题，再用省略式追问（验证上下文改写）")
    base = engine.ask(FOLLOWUP_BASE_QUESTION, conversation_id=conversation_id)
    report_answer("3.1", FOLLOWUP_BASE_QUESTION, base, golden_by_question.get(FOLLOWUP_BASE_QUESTION), evaluator)

    followup = engine.ask(FOLLOWUP_QUESTION, conversation_id=conversation_id)
    report_answer(
        "3.2",
        FOLLOWUP_QUESTION,
        followup,
        golden_by_question.get("武汉兴图新科电子股份有限公司注册资本是多少？"),
        evaluator,
    )
    analysis = followup.query_analysis
    if analysis is not None:
        emit(f"      追问识别：is_followup={analysis.is_followup}；改写后：{analysis.rewritten[:80]}")

    history = engine.get_messages(conversation_id)
    emit(f"      会话历史消息数：{len(history)}（角色序列：{','.join(m.role for m in history[:6])}…）")

    # ---- 步骤 4：无关问题（兜底观察）----
    emit("\n【步骤 4】提出一个与语料无关的问题（验证『不清楚』兜底）")
    unrelated = engine.ask(UNRELATED_QUESTION, conversation_id=conversation_id)
    report_answer("4", UNRELATED_QUESTION, unrelated, None, evaluator)

    # ---- 步骤 5：点赞反馈 ----
    emit("\n【步骤 5】对最后一条回答点赞（验证反馈功能）")
    last_assistant = next((message for message in reversed(history) if message.role == "assistant"), None)
    if last_assistant is not None:
        feedback_id = engine.submit_feedback(
            conversation_id, last_assistant.message_id, "up", "追问结果符合预期", FOLLOWUP_QUESTION
        )
        emit(f"      反馈已提交：feedback_id={feedback_id}；统计={engine.store.feedback_stats()}")
    else:
        emit("      [警告] 会话中没有助手消息，无法提交反馈")

    # ---- 步骤 6：汇总 ----
    emit("\n【步骤 6】本次模拟汇总")
    emit(f"      工单问题：{len(golden_items)} 题，拿到非空答案：{answered} 题")
    if first_tokens:
        emit(
            f"      首字耗时：平均 {sum(first_tokens) / len(first_tokens):.1f}ms，"
            f"最大 {max(first_tokens):.1f}ms（预算 3000ms）"
        )
    emit(f"      无关问题是否兜底：{'是' if unrelated.is_unknown else '否（见汇报中的已知缺陷）'}")
    emit(f"      追问是否为上下文改写：{'是' if analysis and analysis.is_followup else '否'}")
    emit(f"      会话消息总数：{len(engine.get_messages(conversation_id))}")
    emit(f"结束时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    emit("=" * 78)

    write_log(log_path)
    emit(f"日志已写入：{log_path}")

    return 0 if answered == len(golden_items) and golden_items else 1


if __name__ == "__main__":
    raise SystemExit(main())
