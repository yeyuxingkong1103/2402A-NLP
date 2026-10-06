# -*- coding: utf-8 -*-
"""在线级：故障容错（模型异常 / 非法过滤）。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

本文件只测「坏了要说话」：
    * 模型异常：注入一个必然抛错的 LLM 桩 → 生成侧必须**要么**抛带 code 的 ``RagError``（log + 传播），
      **要么**显式降级并给出非空答案，**两种都必须留下结构化日志**；禁止静默返回空答案；
    * 非法 ``file_names``：核心层允许「抛 RagError」或「回不清楚」，但不得静默返回不相关块；
      HTTP 层的 400 约定由 ``test_t7_online_http.py`` 覆盖。
"""

from __future__ import annotations

from typing import Any

import pytest

from common import assertions, paths
from common.reports import trace_events, write_report

pytestmark = [pytest.mark.online]


class _BoomLLM:
    """故障注入桩：任何生成调用都抛 ``RagError``（模拟模型/服务异常）。"""

    class backend:  # 形状对齐 ``llm_client.LLMBackendInfo``
        """后端信息桩。"""

        name = "ollama"
        model = "t8-boom-model"
        base_url = "http://127.0.0.1:11434"

    def generate_full(self, *args: Any, **kwargs: Any) -> Any:
        """直接抛错（这就是被测的故障）。"""
        from app.core.errors import RagError  # noqa: PLC0415

        raise RagError("t8 故障注入：模型不可用", code="RAG-5299", stage="generate")

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        """流式接口同样抛错。"""
        from app.core.errors import RagError  # noqa: PLC0415

        raise RagError("t8 故障注入：模型流式不可用", code="RAG-5299", stage="generate")

    def warmup(self, *args: Any, **kwargs: Any) -> bool:
        """预热恒为 False（表示不可用）。"""
        return False


def test_model_exception_is_not_silent(app_config: Any, retriever: Any, golden_index: dict[int, Any]) -> None:
    """模型异常：必须 log + 传播 或 log + 显式降级（禁止静默空答案）。"""
    paths.ensure_dev_on_path()
    from app.core.errors import RagError  # noqa: PLC0415
    from app.core.generator import AnswerGenerator  # noqa: PLC0415

    item = golden_index[543]
    retrieval = retriever.retrieve(item.question, top_k=5)
    assert retrieval.chunks, "检索为空，无法构造故障注入场景"

    generator = AnswerGenerator(cfg=app_config, llm=_BoomLLM())
    before = {name: len(trace_events(name)) for name in ("app.log", "error.log")}
    outcome = ""
    answer = None
    try:
        answer = generator.answer(item.question, retrieval, trace_id="t8-fault-llm")
        outcome = "degrade"
    except RagError as exc:
        outcome = "raise"
        assert str(exc.code).startswith("RAG-"), f"抛错必须带错误码：{exc}"

    after = {name: trace_events(name) for name in ("app.log", "error.log")}
    fresh = {name: rows[before[name]:] for name, rows in after.items()}
    errorish = [row for rows in fresh.values() for row in rows
                if str(row.get("level", "")).upper() in ("ERROR", "WARNING")
                or "error" in str(row.get("event", "")).lower()
                or "degrade" in str(row.get("event", "")).lower()]
    write_report("online_fault_llm", "模型异常故障注入", [
        {"outcome": outcome,
         "answer_text": str(getattr(answer, "text", "") or ""),
         "answer_chars": len(str(getattr(answer, "text", "") or "")),
         "fresh_log_records": {name: len(rows) for name, rows in fresh.items()},
         "errorish_events": [row.get("event") for row in errorish][:8],
         "note": "两种处置都允许：抛 RagError（log+传播）或显式降级并给出非空答案；两者都必须留痕"}
    ])
    assert errorish, ("模型异常**没有留下任何 ERROR/WARNING/*.degrade 日志** → 静默失败："
                      f"outcome={outcome}")
    if outcome == "degrade":
        assert str(answer.text).strip(), "声称显式降级却返回空答案（静默失败）"


def test_invalid_file_filter_is_explicit(engine: Any, golden_index: dict[int, Any]) -> None:
    """非法 ``file_names``：必须显式（抛带码的 RagError 或回「不清楚」），不得返回无关块。"""
    paths.ensure_dev_on_path()
    from app.core.errors import RagError  # noqa: PLC0415

    item = golden_index[543]
    outcome = ""
    answer = None
    try:
        answer = engine.ask(item.question, session_id="t8-fault-filter",
                            file_names=["不存在的文档.pdf"], stream=False)
        outcome = "answered"
    except RagError as exc:
        outcome = "raise"
        assert str(exc.code).startswith("RAG-"), f"抛错必须带错误码：{exc}"

    rows = [{"outcome": outcome,
             "is_unknown": bool(getattr(answer, "is_unknown", False)),
             "unknown_reason": getattr(answer, "unknown_reason", None),
             "text": str(getattr(answer, "text", "") or "")[:80],
             "citations": [getattr(c, "file_name", "") for c in (getattr(answer, "citations", []) or [])]}]
    write_report("online_fault_filter", "非法 file_names 过滤", rows)
    if outcome == "answered":
        assert bool(answer.is_unknown) is True, f"非法过滤却给出了答案：{answer.text[:60]!r}"
        assert not list(answer.citations), "非法过滤不得给出引用"
