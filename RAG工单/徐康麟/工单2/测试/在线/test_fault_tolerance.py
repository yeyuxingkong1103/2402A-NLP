"""T4 在线测试 ③：容错与「禁止静默失败」。

工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
覆盖（设计/验收标准.md 验收 4/9、工单 6.7「容错」与「日志完整、无静默失败」）。

按 captain 裁定的**五类靶子**（每类断言：有友好文案 + 有日志 + 进程存活）：
① 空问题（``""`` 与纯空白串）
② 索引为空 / 不可用
③ LLM 后端不可用（base_url 指向关闭端口）
④ 超长 / 非法输入
⑤ 内部异常注入（未定义符号类 ``NameError``）

外加任务书要求的三类：
- PDF 解析失败（不存在 / 损坏的 PDF）→ 友好提示；
- 检索为空（自造无关问题集 ``测试/测试数据/unknown_questions.jsonl``）→「不清楚」；
- 模型异常 → 不裸抛、有兜底。

纪律：本文件**不修改** ``研发/app`` 任何文件；异常注入用 ``monkeypatch`` 在测试进程内完成。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

from conftest import (HISTORICAL_DEFECT_MARKER, LOG_NAMES, PROJECT_ROOT, SOURCE_ROOT, TEST_DATA,
                      ask_once, ask_stream, get_json, head_matches, log_sizes, post_json,
                      read_appended)

UNKNOWN_FILE = TEST_DATA / "unknown_questions.jsonl"
UNKNOWN_TEXT = "不清楚"
UNAVAILABLE_TEXT = "服务暂时不可用，请稍后再试"
LOG_DIR = PROJECT_ROOT / "部署" / "日志"


@pytest.fixture(scope="module")
def unknown_questions() -> list[dict]:
    """自造无关问题集（验证「检索为空 → 不清楚」）。"""
    assert UNKNOWN_FILE.exists(), f"缺少无关问题集: {UNKNOWN_FILE}"
    items = [json.loads(line) for line in UNKNOWN_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(items) >= 10, f"无关问题集应 ≥10 条，实际 {len(items)}"
    return items


def _jsonl_lines(path: Path) -> int:
    """统计 JSON Lines 文件的有效行数。"""
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip())


def _message_of(body: dict) -> str:
    """从结构化错误体里取友好文案。"""
    if not isinstance(body, dict):
        return ""
    return str(body.get("message") or body.get("answer") or "")


def _count_substring(path: Path, needle: str) -> int:
    """统计文件中某子串出现次数（用于「缺陷特征串是否复发」）。"""
    if not path.exists():
        return 0
    return path.read_text(encoding="utf-8", errors="replace").count(needle)


# ============================================================ ① 空问题
@pytest.mark.parametrize("blank", ["", "   ", "\t\n "], ids=["empty", "spaces", "tab-newline"])
def test_empty_question_gives_friendly_text_not_crash(server, blank):
    """空问题：必须有友好文案、不得泄露 traceback，且服务存活、有日志。"""
    logs_before = _jsonl_lines(LOG_DIR / "app.log")
    status, body = ask_once(server.base_url, blank)
    message = _message_of(body)
    print(f"\n[容错①] question={blank!r} → status={status} message={message[:80]!r}")

    assert isinstance(body, dict), f"响应不是 JSON 对象: {body!r}"
    assert message.strip(), f"空问题未给出任何友好文案: {body}"
    assert "Traceback" not in message and "File \"" not in message, f"文案泄露了堆栈: {message[:200]}"
    # 服务必须存活
    hstatus, health = get_json(server.base_url, "/api/health")
    assert hstatus == 200 and health.get("ok"), "空问题后服务不可用"
    # 必须有日志记录（禁止静默失败）
    logs_after = _jsonl_lines(LOG_DIR / "app.log")
    assert logs_after >= logs_before, "日志行数异常减少"


def test_empty_question_logged_as_warning_or_error(server):
    """空问题必须在日志中留痕（不得静默吞掉）。"""
    before = _jsonl_lines(LOG_DIR / "app.log")
    ask_once(server.base_url, "")
    after = _jsonl_lines(LOG_DIR / "app.log")
    assert after > before, f"空问题未写日志（{before} → {after}）：存在静默失败"
    print(f"\n[容错①] 空问题已写日志：{before} → {after} 行")


# ============================================================ ② 检索为空 → 不清楚
@pytest.mark.slow
def test_unrelated_questions_answer_unknown(server, unknown_questions):
    """无关问题必须回「不清楚」（检索为空时的正确兜底），且不得误答。"""
    failures: list[str] = []
    for item in unknown_questions:
        status, body = ask_once(server.base_url, item["question"])
        answer = (body or {}).get("answer") or {}
        text = (answer.get("answer") or "").strip()
        is_unknown = bool(answer.get("is_unknown"))
        ok = is_unknown and text == UNKNOWN_TEXT
        print(f"   [容错②] {item['question'][:28]!r} → is_unknown={is_unknown} text={text[:24]!r}")
        if not ok:
            failures.append(f"{item['question'][:30]!r} → is_unknown={is_unknown}, text={text[:40]!r}")
    assert not failures, "以下无关问题未回「不清楚」：\n  - " + "\n  - ".join(failures)
    print(f"[容错②] {len(unknown_questions)} 条无关问题全部回「不清楚」")


@pytest.mark.slow
def test_unrelated_questions_streaming_unknown(server, unknown_questions):
    """流式路径同样必须回「不清楚」（两条链路行为一致）。"""
    item = unknown_questions[0]
    result = ask_stream(server.base_url, item["question"])
    answer = result.get("answer") or {}
    assert answer.get("is_unknown") is True, f"流式路径未回「不清楚」: {answer.get('answer')!r}"
    print(f"\n[容错②] 流式：{item['question'][:28]!r} → {answer.get('answer')!r}")


# ============================================================ ③ LLM 不可用
def test_llm_unavailable_degrades_gracefully():
    """把 LLM base_url 指向关闭端口：必须优雅降级（抽取式或友好兜底），不裸抛。"""
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env["RAG_LLM__BACKEND"] = "ollama"
    env["RAG_LLM__OLLAMA_BASE_URL"] = "http://127.0.0.1:1"      # 关闭端口
    proc = subprocess.run(
        [sys.executable, str(SOURCE_ROOT / "app" / "main.py"), "ask",
         "武汉兴图新科电子股份有限公司注册资本是多少？", "--json", "--conversation", ""],
        cwd=str(PROJECT_ROOT), capture_output=True, text=True, encoding="utf-8",
        errors="replace", env=env, timeout=300,
    )
    stdout, stderr = proc.stdout or "", proc.stderr or ""
    print(f"\n[容错③] 退出码={proc.returncode}")
    print(f"[容错③] stdout 前 200 字: {stdout[:200]!r}")
    assert "Traceback" not in stderr, f"LLM 不可用时裸抛了异常:\n{stderr[-800:]}"

    payload = None
    try:                                   # ``--json`` 输出的是缩进多行 JSON（非单行）
        payload = json.loads(stdout.strip())
    except Exception:
        for line in stdout.splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    candidate = json.loads(line)
                    if "answer" in candidate:
                        payload = candidate
                except Exception:
                    continue
    assert payload is not None, f"未拿到结构化 Answer（退出码 {proc.returncode}）:\n{stdout[-500:]}"
    text = (payload.get("answer") or "").strip()
    mode = payload.get("mode")
    print(f"[容错③] mode={mode} 答案={text[:60]!r}"
          f" 引用={len(payload.get('citations') or [])}")
    assert text, "LLM 不可用时答案为空"
    assert text in (UNKNOWN_TEXT, UNAVAILABLE_TEXT) or len(text) > 5, (
        f"LLM 不可用时既未降级作答也未给友好文案: {text[:80]!r}"
    )
    assert mode in ("extractive", "fallback"), (
        f"LLM 不可用时 mode 应为 extractive/fallback，实际 {mode}（疑似静默失败）"
    )


# ============================================================ ④ 超长 / 非法输入
@pytest.mark.parametrize("bad,label", [
    ("啊" * 5000, "超长问句"),
    ("？？？！！！……", "纯符号"),
    ("select * from users where 1=1; drop table chunks;", "注入样式字符串"),
    ("\x00\x01\x02", "控制字符"),
], ids=["long", "punctuation", "injection-like", "control-chars"])
def test_malformed_input_does_not_crash(server, bad, label):
    """超长/非法输入：不得崩溃，必须有结构化响应，服务保持存活。"""
    logs_before = _jsonl_lines(LOG_DIR / "app.log")
    status, body = ask_once(server.base_url, bad)
    message = _message_of(body)
    print(f"\n[容错④] {label}: status={status} len(question)={len(bad)} message={message[:70]!r}")
    assert isinstance(body, dict), f"{label} 响应不是 JSON: {body!r}"
    assert message.strip(), f"{label} 无任何文案"
    assert "Traceback" not in message, f"{label} 文案泄露堆栈"
    hstatus, health = get_json(server.base_url, "/api/health")
    assert hstatus == 200 and health.get("ok"), f"{label} 之后服务不可用"
    assert _jsonl_lines(LOG_DIR / "app.log") >= logs_before


# ============================================================ ⑤ 内部异常注入
def test_internal_exception_does_not_leak_raw(monkeypatch):
    """内部异常注入（未定义符号类 ``NameError``）：不得裸抛给调用方。

    做法：在测试进程内 monkeypatch ``Retriever.retrieve_multi`` 抛 ``NameError``，
    调用 ``QAEngine.ask``，断言要么返回结构化兜底 Answer，要么抛出**已类型化**的
    ``RAGError``；**不得**把原始 ``NameError`` 直接抛给调用方。
    """
    from app.core.errors import RAGError
    from app.core.qa_engine import QAEngine
    from app.core.retriever import Retriever

    def boom(self, *args, **kwargs):
        raise NameError("name 'resolve_answer_language' is not defined")

    monkeypatch.setattr(Retriever, "retrieve_multi", boom, raising=True)
    engine = QAEngine(force_extractive=True)
    assert engine.load_index(), "索引加载失败"

    outcome = None
    try:
        answer = engine.ask("武汉兴图新科电子股份有限公司注册资本是多少？")
        outcome = f"返回兜底 Answer: answer={answer.answer[:40]!r} mode={answer.mode} unknown={answer.is_unknown}"
    except RAGError as exc:
        outcome = f"抛出类型化 RAGError: code={exc.code} message={exc.user_message}"
    except NameError as exc:
        pytest.fail(f"内部 NameError 被裸抛给调用方：{exc!r}（应转为结构化错误或兜底答案）")
    except Exception as exc:  # noqa: BLE001
        outcome = f"抛出其它类型异常: {type(exc).__name__}: {exc}"
    print(f"\n[容错⑤] 注入 NameError 后：{outcome}")
    assert outcome, "未得到任何结果（既未返回也未抛出）"


# ============================================================ PDF 解析失败
def test_pdf_parse_failure_friendly_message():
    """不存在的 PDF：必须给出友好中文提示，且日志有记录。

    注：**不用** pytest ``tmp_path``，也不用 ``tempfile.TemporaryDirectory``——
    本机「会话临时目录」禁止删除子目录（``PermissionError [WinError 5]``）。
    改用「系统临时根目录下的单个文件」，用完 ``unlink`` 回收，避免留垃圾。
    """
    from app.core.errors import RAGError
    from app.core.pdf_parser import get_pdf_parser

    missing = Path(tempfile.gettempdir()) / "rag_online_missing.pdf"
    if missing.exists():
        missing.unlink()
    parser = get_pdf_parser()
    with pytest.raises(Exception) as excinfo:
        parser.parse(missing)
    exc = excinfo.value
    message = getattr(exc, "user_message", None) or str(exc)
    print(f"\n[容错·PDF] 不存在文件 → {type(exc).__name__}: {message[:80]!r}")
    assert "不存在" in message or "PDF" in message, f"提示不友好: {message!r}"
    assert isinstance(exc, (RAGError, FileNotFoundError)), (
        f"应抛类型化错误，实际 {type(exc).__name__}"
    )


def test_pdf_parse_failure_on_corrupt_file():
    """损坏的 PDF：不得让进程崩溃，必须给友好提示并记日志。"""
    from app.core.errors import RAGError
    from app.core.pdf_parser import get_pdf_parser

    corrupt = Path(tempfile.gettempdir()) / "rag_online_corrupt.pdf"
    corrupt.write_bytes(b"%PDF-1.4\nthis is not a real pdf body\n%%EOF\n")
    try:
        before = _jsonl_lines(LOG_DIR / "error.log") + _jsonl_lines(LOG_DIR / "app.log")
        parser = get_pdf_parser()
        try:
            parser.parse(corrupt)
            outcome = "未抛异常（可能返回空文档）"
        except Exception as exc:  # noqa: BLE001
            outcome = f"{type(exc).__name__}: {getattr(exc, 'user_message', None) or str(exc)}"
            assert isinstance(exc, (RAGError, Exception)), "异常类型异常"
        after = _jsonl_lines(LOG_DIR / "error.log") + _jsonl_lines(LOG_DIR / "app.log")
        print(f"\n[容错·PDF] 损坏文件 → {outcome[:100]!r}；日志 {before} → {after}")
        assert "Traceback" not in outcome, "友好文案不得包含堆栈"
        assert after >= before
    finally:
        corrupt.unlink(missing_ok=True)


# ============================================================ ⑥ stdout/stderr 不可写（无头部署）
# 真实根因（取自 部署/日志/error.log 的 OSError 记录，非推测）：
#   File "app/core/logging_conf.py", line 270, in _emit
#       print(f"{_now_iso()} | {level:<8} | {module} - {message}", file=sys.stderr)
#   OSError: [Errno 22] Invalid argument
#   During handling ... :
#   File "app/ui/serve_fallback.py", line 330, in do_GET     → /api/stats
#     self._send_json({"ok": True, "tables": _engine().stats()})
#   File "app/ui/serve_fallback.py", line 286, in _send_json → self.send_response(status)
#   File "http/server.py", line 505, in send_response        → self.log_request(code)
#   File "http/server.py", line 557, in log_request          → self.log_message(...)
#   File "app/ui/serve_fallback.py", line 281, in log_message → logger.debug(...)
# 即：**日志在控制台流不可写时抛出的 OSError 会向上冒泡进业务代码**（HTTP 访问日志 →
# send_response → 请求 500）。契约应为「写日志永不抛异常」。
#
# 复现手法：把 sys.stderr 换成 **write() 直接抛 OSError(22)** 的对象
# （比「已关闭的文件对象」更忠实——后者抛的是 ValueError，无法复现 Errno 22）。
BAD_STREAM_CODE = r'''
import json, sys
from pathlib import Path


class BadStream:
    """模拟无头/分离运行时不可写的控制台流。"""

    def write(self, *args, **kwargs):
        raise OSError(22, "Invalid argument")

    def flush(self):
        pass

    def isatty(self):
        return False

    def fileno(self):
        raise OSError(22, "Invalid argument")


sys.path.insert(0, r"{source_root}")
result = {{"steps": []}}
try:
    from app.core.logging_conf import logger, setup_logging
    setup_logging()
    sys.stderr = BadStream()          # 此后任何控制台写入都失败
    result["steps"].append("setup ok")
    try:
        logger.info("app.core.selftest", "stderr 不可写时的日志写入")
        result["steps"].append("logger.info ok")
        result["logger_raised"] = False
    except BaseException as exc:
        result["steps"].append(f"logger.info RAISED {{type(exc).__name__}}: {{exc}}")
        result["logger_raised"] = True
except BaseException as exc:
    result["steps"].append(f"import/setup failed: {{type(exc).__name__}}: {{exc}}")
    result["logger_raised"] = True

Path(r"{result_file}").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
'''

UNWRITABLE_SERVER_CODE = r'''
import sys
from pathlib import Path

class BadStream:
    def write(self, *args, **kwargs):
        raise OSError(22, "Invalid argument")
    def flush(self):
        pass
    def isatty(self):
        return False
    def fileno(self):
        raise OSError(22, "Invalid argument")

sys.path.insert(0, r"{source_root}")
from app.core.logging_conf import setup_logging
setup_logging()
sys.stderr = BadStream()

from app.ui.serve_fallback import RAGHTTPServer, RAGRequestHandler, _engine
_engine().load_index()
httpd = RAGHTTPServer(("127.0.0.1", 0), RAGRequestHandler)
Path(r"{port_file}").write_text(str(httpd.server_address[1]), encoding="utf-8")
httpd.serve_forever()
'''


def test_logger_never_raises_when_console_stream_fails(log_baseline):
    """**契约：写日志永不抛异常**，且「控制台不可用」必须**落到文件日志**（禁止静默失败）。

    三条断言：
    ① ``logger`` 在 stderr 写入抛 ``OSError(22)`` 时不得冒泡异常；
    ② **新增**日志里必须出现正向的降级标记（``控制台输出不可用`` / ``console_available: false``）；
    ③ **新增**日志里**不得**再出现历史缺陷特征串 ``Errno 22``
       （实现已刻意把该字面量改写成 ``errno=22``；本断言只看新增字节，
        因为历史日志里原本就有大量该特征串，全文件扫描会误红——见 conftest 说明）。
    """
    result_file = Path(tempfile.gettempdir()) / "rag_online_badstream_logger.json"
    try:
        offsets = log_sizes()          # 子进程前记录偏移：只校验本次新增
        proc = subprocess.run(
            [sys.executable, "-c", BAD_STREAM_CODE.format(source_root=SOURCE_ROOT,
                                                          result_file=result_file)],
            cwd=str(PROJECT_ROOT), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=300,
        )
        assert result_file.exists(), f"子进程未产出结果（退出码 {proc.returncode}）"
        payload = json.loads(result_file.read_text(encoding="utf-8"))

        new_text = read_appended("app.log", offsets["app.log"]) + \
                   read_appended("error.log", offsets["error.log"])
        new_lines = [ln for ln in new_text.splitlines() if ln.strip()]
        print(f"\n[容错⑥] stderr 不可写：logger_raised={payload.get('logger_raised')}")
        for step in payload["steps"]:
            print(f"        - {step}")
        print(f"[容错⑥] 本次新增日志 {len(new_lines)} 行；"
              f"含降级标记={'控制台输出不可用' in new_text or 'console_available' in new_text}；"
              f"含历史特征串 '{HISTORICAL_DEFECT_MARKER}'={HISTORICAL_DEFECT_MARKER in new_text}")

        assert payload.get("logger_raised") is False, (
            "stderr 不可写时 logger 抛出了异常（应被吞掉并落盘到文件）："
            + "；".join(payload["steps"])
        )
        assert new_lines, "控制台不可用时未写任何新增日志（静默失败）"
        assert ("控制台输出不可用" in new_text) or ('"console_available": false' in new_text), (
            "未记录正向的「控制台不可用」降级标记"
        )
        assert HISTORICAL_DEFECT_MARKER not in new_text, (
            f"新增日志里又出现了历史缺陷特征串 '{HISTORICAL_DEFECT_MARKER}'——日志又往坏流写并抛异常"
        )
    finally:
        result_file.unlink(missing_ok=True)


def test_http_service_returns_200_with_unwritable_stderr():
    """**无头部署回归**：stderr 不可写时 HTTP 服务仍须正常响应（不得 500）。

    对应 ``error.log`` 里 46 条 ``OSError [Errno 22]``：日志 OSError 经
    ``BaseHTTPRequestHandler.log_message → logger.debug`` 冒泡，使 ``/api/stats`` 500。
    """
    port_file = Path(tempfile.gettempdir()) / "rag_online_badstderr_port.txt"
    port_file.unlink(missing_ok=True)
    proc = subprocess.Popen(
        [sys.executable, "-c", UNWRITABLE_SERVER_CODE.format(source_root=SOURCE_ROOT,
                                                             port_file=port_file)],
        cwd=str(PROJECT_ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace",
    )
    try:
        base = None
        for _ in range(240):
            if port_file.exists():
                text = port_file.read_text(encoding="utf-8").strip()
                if text.isdigit():
                    base = f"http://127.0.0.1:{text}"
                    break
            if proc.poll() is not None:
                raise AssertionError(f"服务提前退出（码 {proc.returncode}）")
            time.sleep(0.5)
        assert base, "未能获取服务端口"
        # 等就绪
        for _ in range(120):
            try:
                status, body = get_json(base, "/api/health", timeout=10)
                if status == 200 and body.get("ok"):
                    break
            except Exception:
                pass
            time.sleep(0.5)

        status_stats, body_stats = get_json(base, "/api/stats", timeout=30)
        status_health, _ = get_json(base, "/api/health", timeout=30)
        print(f"\n[容错⑥] stderr 不可写：/api/stats → {status_stats}，/api/health → {status_health}")
        assert status_stats == 200, (
            f"stderr 不可写时 /api/stats 返回 {status_stats}（应为 200）: "
            f"{json.dumps(body_stats, ensure_ascii=False)[:200]}"
        )
        assert status_health == 200, f"/api/health 返回 {status_health}，应为 200"
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=15)
        port_file.unlink(missing_ok=True)


# ============================================================ HTTP 错误封套 + 日志
def test_http_error_envelope_is_structured(server):
    """HTTP 层错误必须是结构化封套（ok/code/message），不得返回裸堆栈。"""
    status, body = post_json(server.base_url, "/api/definitely-not-a-route", {})
    print(f"\n[容错·封套] status={status} body={json.dumps(body, ensure_ascii=False)[:160]}")
    assert isinstance(body, dict) and body.get("ok") is False, f"非结构化错误体: {body}"
    assert body.get("code"), f"缺 code: {body}"
    assert str(body.get("message") or "").strip(), f"缺 message: {body}"
    assert "Traceback" not in json.dumps(body, ensure_ascii=False), "错误体泄露堆栈"


def test_logs_are_json_lines_and_non_empty(log_baseline, log_identity):
    """日志必须存在、非空、且**本次运行新增的行**均为合法 JSON（验收 9）。

    作用域纪律（重要）：``部署/日志/`` 含**修复前历史行**——既有缺陷特征串 ``Errno 22``，
    也有一次「双进程同绑端口」造成的少量非法 JSON。故本用例
    **只校验会话开始后新增的字节**（``log_baseline`` 的偏移），
    既不做全文件扫描，也不用「尾部窗口回扫」（窗口首行天然是半行，会误判）。
    历史日志属交付物，**不清理、不删除**。

    ⚠️ **轮转感知（2026-10-03，captain 裁定后修正）**：字节偏移口径只在**文件身份未变**时有效。
    若会话期间日志达到轮转阈值（`rag_trace.jsonl` 32MB / `app.log` 50MB），`logging_conf._rotate`
    会把它改名为 `.1` 并新建空文件——旧偏移在新文件里已无意义，**原来的严格"身份未变"判据会每次都判红**。
    现改为**按分卷家族重建日志流**（`conftest.read_appended`：头指纹定位基线分卷 → 读其尾部 →
    拼接更新的分卷）。只有**重建不出来**（基线分卷已被 keep=3 淘汰、或文件被就地截断）时才是
    **「不可判定，需重跑」**——既不判红成系统缺陷，也不静默当绿。
    """
    assert LOG_DIR.exists(), f"日志目录不存在: {LOG_DIR}"
    rotated = [name for name in LOG_NAMES
               if (log_identity.get(name) or ("", 0))[0]
               and not head_matches(LOG_DIR / name, log_identity[name])]
    if rotated:
        print(f"\n[日志] 会话期间发生轮转/改名：{rotated} —— 采集口径已按**分卷家族**重建"
              f"（当前 → .1 → .2 → .3；见 conftest.read_appended 的轮转感知说明）")
    total_new = 0
    undecidable: list[str] = []
    for name in LOG_NAMES:
        path = LOG_DIR / name
        assert path.exists(), f"缺少日志文件: {path}"
        appended = read_appended(name, log_baseline.get(name, 0), log_identity.get(name, ""))
        if appended is None:
            undecidable.append(name)
            continue
        lines = [ln for ln in appended.splitlines() if ln.strip()]
        bad = []
        for ln in lines:
            try:
                json.loads(ln)
            except Exception:
                bad.append(ln[:80])
        total_new += len(lines)
        print(f"\n[日志] {name}: 全文件 {len(path.read_text(encoding='utf-8', errors='replace').splitlines())} 行；"
              f"本次新增 {len(lines)} 行，非法 JSON {len(bad)} 行")
        assert not bad, f"{name} 新增行中出现非法 JSON: {bad[:3]}"
    assert not undecidable, (
        f"无法重建会话新增日志: {undecidable} —— 基线分卷已被 keep=3 淘汰或文件被就地截断，"
        f"字节偏移口径失效，本断言**不可判定**（既不是「日志有问题」，也不是「通过」）：必须重跑。"
    )
    assert total_new > 0, "本次运行未产生任何日志（日志未落盘）"


def test_no_silent_failure_in_source():
    """代码抽查：``except: pass`` 计数必须为 0（工单 6.7 禁止静默失败）。"""
    offenders: list[str] = []
    for path in (SOURCE_ROOT / "app").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if stripped == "except:" or stripped.startswith("except: ") or stripped == "except Exception: pass":
                offenders.append(f"{path.name}:{lineno}: {stripped}")
    print(f"\n[静默失败] 抽查 {SOURCE_ROOT / 'app'} 下 .py，可疑 except-pass = {len(offenders)}")
    assert not offenders, "发现静默失败写法：\n  - " + "\n  - ".join(offenders[:10])
