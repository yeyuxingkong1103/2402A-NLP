"""日志健壮性自测：stdout/stderr 不可写时日志不崩、HTTP 请求不 500。

工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
阶段：研发 / 自测脚本（对应缺陷修复：``OSError [Errno 22]`` 污染 error.log 并让请求 500）

背景（captain 从 ``部署/日志/error.log`` 的 45 条同名异常定位）：
无头 / 分离进程 / 重定向到已关闭流时，`print(..., file=sys.stderr)` 抛
``OSError: [Errno 22] Invalid argument``；原兜底 ``except`` 又往**同一坏流**写，
覆盖真实异常并让正常请求（如 ``/api/stats``）变成 500。

本脚本构造"控制台不可写"场景，逐项验证：
① 日志系统不抛异常（无逃逸）；
② HTTP 端点仍返回 200（不 500）；
③ **本次新增**的日志行全部为合法 JSON Lines，且不再出现历史缺陷特征串 ``Errno 22``；
④ 日志中留下一条明确的"控制台输出不可用（degraded）"标记（结构化 ``errno`` 完整保留）。

历史遗留说明：``部署/日志/*`` 中存在少量早期损坏行（例如双进程同绑端口期间的交错写入），
属修复前的历史产物；本脚本只校验**本次新增**的行，避免把历史污染混入当前结论。

用法::

    pwsh -NoProfile -File run_py.ps1 研发/scripts/selftest_log_robustness.py
"""

from __future__ import annotations

import io
import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

REAL_OUT, REAL_ERR = sys.stdout, sys.stderr
if hasattr(REAL_OUT, "reconfigure"):
    REAL_OUT.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LOG_DIR = Path(r"E:\gao6gongdan\工单2\部署\日志")
LOG_FILES = ("app.log", "error.log", "rag_trace.jsonl")


class BrokenStream(io.TextIOBase):
    """模拟不可写的控制台流（无头/分离进程/重定向到已关闭流 → Errno 22）。"""

    def write(self, _text: str) -> int:  # noqa: D102
        raise OSError(22, "Invalid argument")

    def flush(self) -> None:  # noqa: D102
        raise OSError(22, "Invalid argument")

    def isatty(self) -> bool:  # noqa: D102
        return False


def offsets() -> dict[str, int]:
    """记录各日志文件当前字节长度（用于只校验"本次新增"的行）。"""
    result: dict[str, int] = {}
    for name in LOG_FILES:
        path = LOG_DIR / name
        result[name] = path.stat().st_size if path.exists() else 0
    return result


def appended(before: dict[str, int]) -> dict[str, list[str]]:
    """读取本次新增的整行内容。"""
    fresh: dict[str, list[str]] = {}
    for name in LOG_FILES:
        path = LOG_DIR / name
        if not path.exists():
            continue
        start = before.get(name, 0)
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            handle.seek(start)
            fresh[name] = [line for line in handle.read().splitlines() if line.strip()]
    return fresh


def main() -> int:
    before = offsets()
    escaped: list[str] = []

    sys.stdout = BrokenStream()
    sys.stderr = BrokenStream()
    try:
        from app.core.logging_conf import flush_logs, logger, log_stage, setup_logging

        setup_logging()
        logger.info("app.test.headless", "无头场景日志冒烟", case="broken_stream")
        logger.warning("app.test.headless", "警告也应安全")
        try:
            raise ValueError("模拟业务异常")
        except ValueError:
            logger.exception("app.test.headless", "异常日志应写文件且不抛")
        log_stage("headless", "阶段日志")
        flush_logs()
        stage1 = "OK（无异常逃逸）"
    except BaseException as exc:  # noqa: BLE001
        stage1 = f"FAIL：{type(exc).__name__}: {exc}"
        escaped.append(stage1)

    responses: dict[str, str] = {}
    try:
        from app.ui.serve_fallback import RAGHTTPServer, RAGRequestHandler

        httpd = RAGHTTPServer(("127.0.0.1", 0), RAGRequestHandler)
        port = httpd.server_address[1]
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        for path in ("/api/health", "/api/stats", "/"):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=60) as response:
                    responses[path] = f"{response.status} ({len(response.read())}B)"
            except urllib.error.HTTPError as exc:
                responses[path] = f"HTTPError {exc.code}"
            except Exception as exc:  # noqa: BLE001
                responses[path] = f"{type(exc).__name__}: {exc}"
        httpd.shutdown()
        httpd.server_close()
    except BaseException as exc:  # noqa: BLE001
        responses["server"] = f"FAIL：{type(exc).__name__}: {exc}"
        escaped.append(responses["server"])

    fresh = appended(before)
    sys.stdout, sys.stderr = REAL_OUT, REAL_ERR

    total_lines = invalid_lines = errno_hits = 0
    console_marks = 0
    for name, lines in fresh.items():
        for line in lines:
            total_lines += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                invalid_lines += 1
                print(f"  [非法 JSON] {name}: {line[:110]}")
                continue
            text = json.dumps(record, ensure_ascii=False)
            if "OSError: [Errno 22]" in text or "Errno 22" in text:
                errno_hits += 1
                print(f"  [含 Errno 22] {name}: {line[:110]}")
            if "控制台输出不可用" in text:
                console_marks += 1

    print("=" * 92)
    print(f"阶段1 日志系统（坏流）：{stage1}")
    print(f"阶段2 HTTP 响应：{responses}")
    print(f"阶段3 本次新增日志行：{total_lines} 行（合法 {total_lines - invalid_lines} / 非法 {invalid_lines}）")
    print(f"阶段4 新增行中含 'Errno 22'：{errno_hits} 行；含'控制台输出不可用'标记：{console_marks} 行")
    print(f"逃逸异常：{escaped if escaped else '无'}")
    ok = (
        not escaped
        and responses.get("/api/stats", "").startswith("200")
        and responses.get("/api/health", "").startswith("200")
        and invalid_lines == 0
        and errno_hits == 0
        and console_marks >= 1
    )
    print("结论：" + ("PASS ✅" if ok else "FAIL ❌"))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
