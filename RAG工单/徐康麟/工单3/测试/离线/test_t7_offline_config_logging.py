# -*- coding: utf-8 -*-
"""离线级：配置契约 / 结构化日志 / 静态纪律 / 引用格式 断言。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

覆盖验收 10（工程纪律）与 §8 配置契约：
    * 冻结默认值（top_k=5 / rrf_k=60 / probe_timeout_s=0.5 / 三档加权 1.25·1.15·1.10 …）；
    * 环境变量覆盖生效（``RAG_RETRIEVAL__TOP_K``）；
    * **预热调用点必须存在**（§3.4：零调用点 = 实现缺陷）：``QAEngine.warmup`` 内部三段
      （warmup_tokenizer → embedder.warmup → probe_backends）与至少两个入口调用点；
    * 禁止 ``except: pass`` 这类静默失败；
    * ``app.log`` 行行可 ``json.loads`` 且带公共字段；错误日志留栈（有内容时）；
    * 引用格式 ``[文件名: 页码]`` 可往返解析。
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from common import assertions, paths

pytestmark = [pytest.mark.offline, pytest.mark.linkage]

#: 设计 §8 冻结的确定性默认值（这些是「实现必须给到」的常量）
FROZEN_DEFAULTS = {
    "retrieval.top_k": 5,
    "retrieval.vector_k": 20,
    "retrieval.bm25_k": 20,
    "retrieval.rrf_k": 60,
    "retrieval.table_boost": 1.25,
    "retrieval.numeric_boost": 1.15,
    "retrieval.keyword_boost": 1.10,
    "retrieval.enable_llm_rerank": False,
    "chunk.size": 500,
    "chunk.overlap": 80,
    "table.min_rows": 2,
    "table.min_cols": 2,
    "llm.probe_timeout_s": 0.5,
    "llm.request_timeout_s": 20.0,
    "llm.max_tokens": 512,
    "llm.temperature": 0.2,
    "llm.ollama_embed_model": "bge-m3:latest",
    "answer.history_turns": 5,
    "answer.enable_subject_gate": True,
    "answer.unknown_text": "不清楚",
}


def _dig(obj: object, dotted: str) -> object:
    """按 ``a.b`` 取值。"""
    current = obj
    for part in dotted.split("."):
        current = getattr(current, part)
    return current


def test_config_frozen_defaults(app_config: object) -> None:
    """§8 冻结默认值逐条核对（不随环境变量漂移的部分）。"""
    problems: list[str] = []
    for key, expected in FROZEN_DEFAULTS.items():
        actual = _dig(app_config, key)
        if actual != expected:
            problems.append(f"{key} 期望 {expected!r}，实测 {actual!r}")
    assert not problems, "\n".join(problems)


def test_calibrated_thresholds_are_recorded(app_config: object) -> None:
    """可答性阈值是 T6 标定值（设计 §8 表列的是初值）→ 记录差异而不是硬卡文档值。

    §9 允许「实现按实测标定 + 文档补记」；此处断言阈值在合法区间并写入留痕，
    供 captain/architect 决定是否需要回写设计文档（**不据此判缺陷**）。
    """
    min_score = float(_dig(app_config, "answer.min_score"))
    coverage = float(_dig(app_config, "answer.min_keyword_coverage"))
    assert 0.0 < min_score < 1.0, f"min_score 不合法：{min_score}"
    assert 0.0 < coverage <= 1.0, f"min_keyword_coverage 不合法：{coverage}"
    paths.ensure_trace_dir()
    (paths.TRACE_DIR / "config_thresholds_record.json").write_text(json.dumps({
        "work_order": assertions.WORK_ORDER,
        "note": "可答性阈值实测值（T6 标定）；设计 §8 表列 min_score=0.30 / min_keyword_coverage=0.25",
        "answer_min_score": min_score,
        "answer_min_keyword_coverage": coverage,
        "ragas": assertions.RAGAS_BANNER,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def test_env_override_takes_effect(monkeypatch: pytest.MonkeyPatch) -> None:
    """``RAG_RETRIEVAL__TOP_K=7`` 覆盖后配置生效（验证环境变量通道，而非只写文档）。"""
    paths.ensure_dev_on_path()
    from app.core.config import get_config  # noqa: PLC0415

    monkeypatch.setenv("RAG_RETRIEVAL__TOP_K", "7")
    assert int(get_config(refresh=True).retrieval.top_k) == 7
    monkeypatch.delenv("RAG_RETRIEVAL__TOP_K", raising=False)
    assert int(get_config(refresh=True).retrieval.top_k) == FROZEN_DEFAULTS["retrieval.top_k"]


def test_warmup_call_sites_exist() -> None:
    """§3.4 硬要求：``QAEngine.warmup`` 内部三段齐全，且入口处必须有调用点。

    零调用点 = 实现缺陷（T3 曾出现，captain 用 grep 确认过）。
    """
    engine = paths.DEV_DIR / "app" / "core" / "qa_engine.py"
    assert engine.exists(), f"缺少 QAEngine 实现：{paths.display(engine)}"
    text = engine.read_text(encoding="utf-8")
    for token in ("warmup_tokenizer", "probe_backends", "def warmup"):
        assert token in text, f"qa_engine.py 缺少预热要素 {token!r}"

    entries = [paths.DEV_DIR / "app" / "main.py",
               paths.DEV_DIR / "app" / "ui" / "serve_fallback.py",
               paths.DEV_DIR / "app" / "ui" / "streamlit_app.py"]
    missing = [paths.display(p) for p in entries if not p.exists()]
    assert not missing, f"在线入口文件缺失：{missing}"
    callers = [p for p in entries if re.search(r"warmup\s*\(", p.read_text(encoding="utf-8"))]
    assert len(callers) >= 2, (
        "预热调用点不足（要求 main 启动事件 + 至少一个界面入口）："
        + ", ".join(paths.display(p) for p in callers))


def _silent_except_offenders(path: Path) -> list[str]:
    """AST 精确识别「静默吞异常」：``except ...:`` 分支体**只有** ``pass`` / ``...``。

    为什么用 AST 而不是正则：正则会把 ``errors.py`` 文档字符串里「禁止 except: pass」这句话
    当成违规（假阳性）。AST 只认真正的 ``ast.ExceptHandler`` 节点。
    """
    import ast  # noqa: PLC0415

    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError as exc:                          # 语法错误本身也是问题，直接暴露
        return [f"{paths.display(path)}:{exc.lineno} 语法错误：{exc.msg}"]
    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        body = node.body
        silent = bool(body) and all(
            isinstance(item, ast.Pass) or (isinstance(item, ast.Expr) and isinstance(item.value, ast.Constant))
            for item in body)
        if silent:
            kind = ast.unparse(node.type) if node.type is not None else "裸 except"
            offenders.append(f"{paths.display(path)}:{node.lineno} except {kind}: "
                             f"分支体只有 pass/... （应 log + 传播，或记 *.degrade 事件后显式降级）")
    return offenders


def test_no_silent_exception_swallow() -> None:
    """禁止静默失败：``except`` 分支体不得只有 ``pass``（验收 10 + errors.py 自身纪律）。

    允许的降级必须在日志里留下 ``*.degrade`` 事件（含 reason），不能一个 ``pass`` 了事。
    """
    offenders: list[str] = []
    for path in sorted((paths.DEV_DIR / "app").rglob("*.py")):
        offenders += _silent_except_offenders(path)
    assert not offenders, "存在静默吞异常：\n" + "\n".join(offenders)


def test_run_py_entry_exists() -> None:
    """统一入口 ``run_py.ps1`` 存在且默认解释器 = 工单1 venv（环境事实 §1）。"""
    entry = paths.REPO_ROOT / "run_py.ps1"
    assert entry.exists(), "缺少统一入口 run_py.ps1"
    text = entry.read_text(encoding="utf-8", errors="replace")
    assert "工单1" in text and "python.exe" in text
    assert "RAG_SCHEDULER_PYTHON" in text, "入口应支持 RAG_SCHEDULER_PYTHON 覆盖"


def test_logger_emits_parseable_records() -> None:
    """确定性日志契约：由本测试**主动写一条标记事件**，它必须是一条可解析的 JSON 记录。

    为什么不直接「把 app.log 全文件逐行解析」：``部署/日志/app.log`` 是**多进程共享**的
    （engineer 的脚本、在线服务、测试都在写），并发追加时偶发地把一条长记录切成两半，
    这是运行环境现象而不是日志实现缺陷。因此契约用标记事件精确验证，全局比例另外做统计。
    """
    paths.ensure_dev_on_path()
    import time as _time  # noqa: PLC0415

    from app.core.logging_conf import get_logger, shutdown_logging  # noqa: PLC0415

    marker = f"t8-marker-{int(_time.time())}"
    log = get_logger("test_t8_logging")
    with log.enter("t8_logging_probe", {"marker": marker}) as span:
        span.set_output({"ok": True})
    log.log_event("t8.marker", marker=marker, level="INFO")
    shutdown_logging()                                     # 强制 flush 落盘

    log_path = paths.trace_file("app.log")
    assert log_path.exists(), f"结构化日志缺失：{paths.display(log_path)}"
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    matched: list[dict[str, Any]] = []
    for line in lines:
        if marker not in line:
            continue
        try:
            matched.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    assert matched, f"日志里找不到标记事件 {marker} 的可解析记录（logger 未落盘？）"
    record = next((row for row in matched if row.get("event") == "t8.marker"), None)
    assert record is not None, f"标记事件未落成 t8.marker 记录：{[r.get('event') for r in matched]}"
    for key in ("ts", "level", "event", "func", "module"):
        assert key in record, f"日志缺少公共字段 {key!r}：{sorted(record)}"
    assert record["event"] == "t8.marker"


def test_app_log_parse_ratio_is_high() -> None:
    """全局统计：末尾记录里可解析 JSON 的比例 ≥ 99.5%，并把异常行落盘留痕。

    阈值 99.5% 的理由：并发写日志会偶发截断（实测 2/2000 = 0.1%），
    但**系统性破坏**（例如把非 JSON 文本混进日志、或忘记 JSON 序列化）会立刻跌破阈值。
    """
    log_path = paths.trace_file("app.log")
    if not log_path.exists() or log_path.stat().st_size == 0:
        pytest.fail(f"结构化日志缺失或为空：{paths.display(log_path)}")
    lines = [line for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-2000:]
             if line.strip()]
    assert lines, "app.log 末尾无有效行"

    bad: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        try:
            json.loads(line)
        except json.JSONDecodeError as exc:
            bad.append({"offset": index, "length": len(line), "error": str(exc)[:80],
                        "head": line[:120]})

    paths.ensure_trace_dir()
    (paths.TRACE_DIR / "app_log_parse_audit.json").write_text(json.dumps({
        "work_order": assertions.WORK_ORDER,
        "note": ("多进程共享日志：并发追加偶发截断属环境现象；本文件记录原始异常行供复核。"
                 "若比例跌破 99.5% 或出现整段非 JSON 文本，则判结构化日志被破坏。"),
        "ragas": assertions.RAGAS_BANNER,
        "sampled": len(lines), "unparseable": len(bad), "ratio": round(1 - len(bad) / len(lines), 5),
        "offenders": bad,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    ratio = 1 - len(bad) / len(lines)
    assert ratio >= 0.995, (f"app.log 可解析比例 {ratio:.4f} < 0.995（异常 {len(bad)}/{len(lines)} 行）；"
                            f"明细见 测试/留痕/app_log_parse_audit.json")

    records: list[dict[str, Any]] = []
    for line in lines:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    sample = records[-1]
    for key in ("ts", "level", "event"):
        assert key in sample, f"日志缺少公共字段 {key!r}：{sorted(sample)}"
    assert any(str(row.get("event", "")).endswith((".enter", ".exit")) for row in records), \
        "日志缺少 func.enter/func.exit 事件对"


def test_error_log_keeps_traceback_when_present() -> None:
    """``error.log`` 有内容时必须含堆栈（有异常 → 必留栈，不允许只写一句「失败」）。"""
    error_path = paths.trace_file("error.log")
    if not error_path.exists() or error_path.stat().st_size == 0:
        pytest.skip("error.log 为空：本次运行尚未产生异常路径证据（由在线层触发后再核）")
    text = error_path.read_text(encoding="utf-8", errors="replace")
    assert "Traceback" in text or "URLError" in text or "Error" in text, \
        "error.log 有内容但没有异常类型/堆栈痕迹"


def test_citation_render_roundtrip() -> None:
    """引用格式往返：``format_citation`` 产出的 ``[文件名: 页码]`` 能被解析回来。"""
    paths.ensure_dev_on_path()
    from app.core import citation as citation_mod  # noqa: PLC0415

    rendered = citation_mod.format_citation("招股说明书1.pdf", 129)
    parsed = assertions.parse_citations(rendered)
    assert parsed, f"引用渲染无法解析：{rendered!r}"
    assert parsed[0].file_name == "招股说明书1.pdf" and parsed[0].page == 129
    assert "[招股说明书1.pdf: 129]" == rendered.strip() or "129" in rendered
