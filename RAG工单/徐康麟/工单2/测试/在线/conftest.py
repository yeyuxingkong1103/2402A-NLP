"""在线测试公共夹具：真实启动 ``serve_fallback`` HTTP 服务（随机空闲端口）。

工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
阶段：测试 / 在线（四件套公共层）

设计要点：
1. **真起服务**：以子进程运行 ``研发/app/ui/serve_fallback.py --port 0``（随机未占用端口），
   从 stdout 解析真实绑定端口，再以 ``GET /api/health`` 轮询直至就绪；会话级复用，避免重复预热。
2. **真走 HTTP**：所有断言都通过 ``urllib`` 走真实请求，不直接调用 core（那是 T3 离线层的做法）。
3. **首字计时口径**：以**客户端墙钟**从「发出请求」到「收到 ``first_token`` SSE 事件」计时
   （即用户可感知的首字时间），同时记录服务端自报的 ``first_token_ms`` 以便交叉核对。
4. 进程退出：会话结束强制终止并回收，避免残留端口。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent.parent      # 测试/
PROJECT_ROOT = TESTS_DIR.parent                          # 工单2
SOURCE_ROOT = PROJECT_ROOT / "研发"
TEST_DATA = TESTS_DIR / "测试数据"
SERVE_SCRIPT = SOURCE_ROOT / "app" / "ui" / "serve_fallback.py"

ADDR_RE = re.compile(r"http://(?P<host>\d+\.\d+\.\d+\.\d+):(?P<port>\d+)/")
START_TIMEOUT_S = 180.0

#: 日志目录（交付物；**只读**，绝不清理——历史行含已修复缺陷的痕迹）
LOG_DIR = PROJECT_ROOT / "部署" / "日志"
LOG_NAMES = ("app.log", "error.log", "rag_trace.jsonl")

#: 历史日志污染特征（captain 实测计数）：
#: - 修复前的行含特征串 ``Errno 22``（app.log 217 / error.log 222 / rag_trace.jsonl 13 处）
#: - 另有一次「双进程同绑端口」造成的非法 JSON 行（app.log 1 / rag_trace.jsonl 17 行）
#: 因此**任何日志断言都必须限定在「本次运行新增的字节」上**，
#: 既不能全文件扫描，也不能用「尾部窗口回扫」（窗口首行天然是半行，会被误判为非法 JSON）。
HISTORICAL_DEFECT_MARKER = "Errno 22"


#: 轮转分卷后缀（与 ``logging_conf._rotate`` 的 ``keep=3`` 一致：当前 / .1 / .2 / .3）
ROTATED_SUFFIXES = ("", ".1", ".2", ".3")


def log_family(name: str) -> list[Path]:
    """日志文件及其**轮转分卷**，按「**从新到旧**」排序（当前 → ``.1`` → ``.2`` → ``.3``）。"""
    return [path for path in (LOG_DIR / f"{name}{suffix}" for suffix in ROTATED_SUFFIXES) if path.exists()]


def _read_from(path: Path, offset: int) -> str:
    """从文件 ``offset`` 处往后读，并按「前一字节是否换行」决定是否丢弃半行。"""
    with path.open("rb") as fh:
        prev = b""
        if offset > 0:
            fh.seek(offset - 1)
            prev = fh.read(1)
        fh.seek(offset)
        data = fh.read()
    text = data.decode("utf-8", errors="replace")
    if offset > 0 and prev != b"\n":
        newline = text.find("\n")
        text = text[newline + 1:] if newline != -1 else ""
    return text


def read_appended(name: str, offset: int, head: str = "") -> str | None:
    """只读取自 ``offset`` 字节之后**新增**的日志内容（**轮转感知**）。

    ``offset`` 恰好落在行首时（写入以 ``\\n`` 结束），新增内容的第一行是完整行，**必须保留**；
    只有落在行中间才丢弃那半行（曾因无脑丢弃首行而漏掉仅有的 1 行降级标记）。

    ## 轮转感知（2026-10-03 补，captain 裁定）

    ``logging_conf._rotate`` 在文件达到阈值（``rag_trace.jsonl`` 32MB / ``app.log`` 50MB）时
    把当前文件改名为 ``.1`` 并新建空文件。若仍按「当前文件 + 旧偏移」读取，会有两种坏结果：
    旧偏移落到新文件里 → 语义错乱；或新文件更短 → **读出空串，断言静默退化成假绿**。
    本函数按**日志流**读取：
    1. 在分卷家族里按**头指纹 ``head``** 定位「会话开始时那个文件」（可能已被改名）；
    2. 从 ``offset`` 读它的尾部，再拼接**比它更新的分卷**（家族中排在它前面的成员）；
    3. 行序不做保证——本函数的消费者只做「每行是否为合法 JSON」的判定，与顺序无关。

    Returns:
        ``str``：可重建的新增内容；``None``：**无法重建**（基线分卷已被 keep=3 淘汰，或文件被就地截断）
        —— 此时调用方必须判「**不可判定，需重跑**」，不得当绿、也不得当红成缺陷。
    """
    family = log_family(name)
    if not family:
        return "" if not head else None
    anchor_index = 0
    if head and head[0]:
        anchor_index = -1
        for index, path in enumerate(family):
            if head_matches(path, head):
                anchor_index = index
                break
        if anchor_index < 0:
            return None                      # 基线文件已不在家族中（被 keep=3 淘汰）
    anchor = family[anchor_index]
    if offset > anchor.stat().st_size:
        return None                          # 就地截断：偏移已越界
    text = _read_from(anchor, offset)
    for newer in family[:anchor_index]:      # 比锚点更新的分卷（整份都是会话新增）
        text += newer.read_bytes().decode("utf-8", errors="replace")
    return text


def log_sizes() -> dict[str, int]:
    """记录三个日志文件的当前字节大小（作为「新增」基线）。"""
    return {name: (LOG_DIR / name).stat().st_size if (LOG_DIR / name).exists() else 0
            for name in LOG_NAMES}


#: 头指纹快照的默认前缀长度（字节）。**关键**：只哈希「快照时实际存在的前缀」，
#: 否则文件在长大到该长度之前，同一前缀的哈希会随文件增长而变 → 误判为轮转、且锚点定位失败。
HEAD_PREFIX_BYTES = 4096


def log_head_fingerprint(name: str, head_bytes: int = HEAD_PREFIX_BYTES) -> tuple[str, int]:
    """日志文件的**头指纹** ``(sha1, 参与哈希的字节数)``；空文件/不存在 → ``("", 0)``。

    为什么需要（2026-10-03 补）：``read_appended`` 按**字节偏移**取"新增内容"，
    一旦文件在会话期间**被轮转/改名**（``logging_conf._rotate``），旧偏移在新文件里已无意义 ——
    断言不会红，而是**静默退化成空断言（假绿）**。头指纹同时充当「分卷锚点」：
    无论文件是否被改名，只要前缀字节一致就能把它认出来。

    注意：返回值带**字节数**。若文件比 ``head_bytes`` 短（例如刚轮转后的空/小文件），
    只哈希当时存在的那些字节；后续追加不会改变这个前缀，从而保持可比。
    """
    path = LOG_DIR / name
    if not path.exists():
        return ("", 0)
    with path.open("rb") as fh:
        data = fh.read(head_bytes)
    return (hashlib.sha1(data).hexdigest(), len(data))


def head_matches(path: Path, head: tuple[str, int] | str | None) -> bool:
    """``path`` 的前缀是否等于 ``head`` 记录的前缀（用于分卷锚点定位/轮转检测）。

    容忍旧式「只给 sha1 字符串」的调用（按 :data:`HEAD_PREFIX_BYTES` 前缀比较），
    以免形态不一致时抛 ``TypeError`` 把「不可判定」变成无意义的红。
    """
    if not head:
        return False
    if isinstance(head, str):
        sha, size = head, HEAD_PREFIX_BYTES
    else:
        sha, size = head
    if not sha or size <= 0:
        return False
    try:
        with path.open("rb") as fh:
            data = fh.read(size)
    except OSError:
        return False
    return hashlib.sha1(data).hexdigest() == sha


def log_fingerprints() -> dict[str, tuple[str, int]]:
    """三个日志文件的头指纹（会话开始时的身份基线）。"""
    return {name: log_head_fingerprint(name) for name in LOG_NAMES}


#: **会话级日志基线**：在 conftest 导入时（早于任何用例、任何服务子进程）快照。
#: 为什么不用"首次请求夹具时"快照：``log_baseline`` 原先在**首次被请求**时才取偏移，
#: 而选择性子集运行（``-k``）里服务夹具可能先于它启动并写完日志，
#: 结果"本次运行新增 0 行"→ ``assert total_new > 0`` 假红（2026-10-03 实测）。
#: 口径改为导入时快照后，"本次运行新增的字节"才真的覆盖整个会话。
_SESSION_LOG_SIZES: dict[str, int] = log_sizes()
_SESSION_LOG_IDENTITY: dict[str, tuple[str, int]] = log_fingerprints()


@pytest.fixture(scope="session")
def log_baseline() -> dict[str, int]:
    """**会话开始时的日志字节偏移**。

    为什么需要：``部署/日志/`` 里保留着修复前的历史行（含缺陷特征串与少量非法 JSON）。
    以会话开始时的大小为基线，后续断言只看新增字节，才能既抓住「现在的缺陷」
    又不把历史算到当前实现头上（也不允许为了好看去删交付物里的历史日志）。
    基线取自 conftest 导入时的会话级快照 ``_SESSION_LOG_SIZES``（见其说明）。
    """
    return dict(_SESSION_LOG_SIZES)


@pytest.fixture(scope="session")
def log_identity() -> dict[str, tuple[str, int]]:
    """**会话开始时的日志文件身份（头指纹）**。

    与 ``log_baseline``（字节偏移）配对：偏移决定「看哪一段」，身份决定「这段偏移是否仍然有效」。
    窗口内发生轮转/截断 → 偏移口径失效，相关断言**不可判定**，必须重跑而不是当绿。
    基线取自 conftest 导入时的会话级快照 ``_SESSION_LOG_IDENTITY``（见其说明）。
    """
    return dict(_SESSION_LOG_IDENTITY)


# --------------------------------------------------------------------------
# HTTP 小工具（标准库，避免额外依赖）
# --------------------------------------------------------------------------
def post_json(base_url: str, path: str, payload: dict, timeout: float = 180.0):
    """POST JSON，返回 ``(status, 解析后的 JSON)``。"""
    req = urllib.request.Request(
        base_url + path,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:      # 4xx/5xx 也要能读到结构化错误体
        body = exc.read().decode("utf-8", errors="replace")
        try:
            return exc.code, json.loads(body)
        except Exception:
            return exc.code, {"ok": False, "raw": body}


def get_json(base_url: str, path: str, timeout: float = 60.0):
    """GET JSON，返回 ``(status, 解析后的 JSON)``。"""
    try:
        with urllib.request.urlopen(base_url + path, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8", errors="replace") or "{}")


def ask_stream(base_url: str, question: str, conversation_id: str | None = None,
               timeout: float = 180.0) -> dict:
    """走 SSE 流式提问，返回首字计时与全部事件。

    Returns:
        ``{"client_first_token_ms": float|None, "server_first_token_ms": float|None,
           "events": [事件名...], "answer": dict|None, "elapsed_ms": float}``
    """
    payload: dict = {"question": question, "stream": True}
    if conversation_id:
        payload["conversation_id"] = conversation_id
    req = urllib.request.Request(
        base_url + "/api/ask",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    t0 = time.perf_counter()
    client_first_ms: float | None = None
    server_first_ms: float | None = None
    events: list[str] = []
    answer: dict | None = None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            text = raw.decode("utf-8", errors="replace").strip()
            if not text.startswith("data:"):
                continue
            try:
                obj = json.loads(text[5:].strip())
            except Exception:
                continue
            event = obj.get("event")
            data = obj.get("data") or {}
            if event == "first_token" and client_first_ms is None:
                client_first_ms = (time.perf_counter() - t0) * 1000
                server_first_ms = float(data.get("first_token_ms") or 0.0)
            events.append(str(event))
            if event == "done":
                answer = data.get("answer")
            elif event == "error":
                answer = {"error": data}
    return {
        "client_first_token_ms": client_first_ms,
        "server_first_token_ms": server_first_ms,
        "events": events,
        "answer": answer,
        "elapsed_ms": (time.perf_counter() - t0) * 1000,
    }


def ask_once(base_url: str, question: str, conversation_id: str | None = None,
             timeout: float = 180.0):
    """非流式提问，返回 ``(status, body)``。"""
    payload: dict = {"question": question, "stream": False}
    if conversation_id:
        payload["conversation_id"] = conversation_id
    return post_json(base_url, "/api/ask", payload, timeout=timeout)


# --------------------------------------------------------------------------
# 服务夹具
# --------------------------------------------------------------------------
class ServerHandle:
    """已启动的服务句柄。"""

    def __init__(self, proc: subprocess.Popen, base_url: str, port: int, log_lines: list[str]) -> None:
        self.proc = proc
        self.base_url = base_url
        self.port = port
        self.log_lines = log_lines
        #: 预热提问的测量结果（首字冷启动成本，供报告；不参与稳态预算断言）
        self.warmup_ask: dict | None = None

    def stop(self) -> None:
        """终止服务进程（先 terminate，超时再 kill）。"""
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=15)


@pytest.fixture(scope="session")
def server() -> ServerHandle:
    """启动真实备用界面服务（随机端口），会话级复用。"""
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        [sys.executable, str(SERVE_SCRIPT), "--host", "127.0.0.1", "--port", "0"],
        cwd=str(PROJECT_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", env=env,
    )
    lines: list[str] = []
    lock = threading.Lock()

    def reader() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            with lock:
                lines.append(line.rstrip("\n"))

    threading.Thread(target=reader, daemon=True).start()

    base_url = None
    deadline = time.time() + START_TIMEOUT_S
    while time.time() < deadline:
        with lock:
            for line in lines:
                match = ADDR_RE.search(line)
                if match:
                    base_url = f"http://{match.group('host')}:{match.group('port')}"
                    break
        if base_url:
            break
        if proc.poll() is not None:
            raise AssertionError("服务进程提前退出：\n" + "\n".join(lines[-20:]))
        time.sleep(0.5)

    assert base_url, ("未能在超时内解析到服务地址：\n" + "\n".join(lines[-20:]))
    handle = ServerHandle(proc, base_url, int(base_url.rsplit(":", 1)[1]), lines)

    # 轮询健康检查直至就绪
    ready = False
    while time.time() < deadline:
        try:
            status, body = get_json(base_url, "/api/health", timeout=15)
            if status == 200 and body.get("ok"):
                ready = True
                break
        except Exception:
            pass
        time.sleep(0.5)
    assert ready, f"服务未就绪（{base_url}）：\n" + "\n".join(lines[-20:])

    # 预热一次真实提问：把「服务启动后的首次提问」冷启动成本与稳态首字**分离测量**。
    # 为什么必须分离：实测首次提问的客户端首字为 2.7~3.1 s（LLM qwen2.5:3b 首次加载），
    # 而服务端自报 first_token_ms 仅 ~0.28 s（未计入该加载）。若不预热，该一次性成本会
    # 落进某一题的稳态统计，使 3 s 预算断言变成随机结果；预热后稳态测量才是「每题的
    # 首字能力」。冷启动数值不隐藏——它被记录在 ``handle.warmup_ask`` 并由用例打印上报。
    try:
        handle.warmup_ask = ask_stream(base_url, "武汉兴图新科电子股份有限公司注册资本是多少？")
    except Exception as exc:  # pragma: no cover - 预热失败不应掩盖后续断言
        handle.warmup_ask = {"error": f"{type(exc).__name__}: {exc}"}

    try:
        yield handle
    finally:
        handle.stop()
