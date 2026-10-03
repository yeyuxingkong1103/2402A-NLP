"""验收 4 在线稳定性自测：对**真实 HTTP 服务**连跑 3 次无关问题集，逐题留痕。

工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
阶段：研发 / 自测脚本（可答性闸门修复报告 §5「三次运行逐题结果」的可重跑执行体）
来源：原 `.tmp_review/online_3runs.py`（2026-10-03 t11 复审 F10 转正，内容未改动逻辑，仅补文件头与用法）。

## 为什么必须保留为交付树内的脚本（而不是留在临时目录）

`研发/报告/可答性闸门修复报告.md` §5 的「12 条无关问题 × 3 连跑、逐题结果」如果只有一个脚本
留在 `.tmp_review/`，一旦清理临时目录，该节结论就**只剩产物、没有可重跑的执行体**——
证据链会断裂。故本脚本转正为交付物，报告 §5 引用本路径。

## 用法（cwd = `E:\\gao6gongdan\\工单2`，服务需已启动）

```powershell
# 1) 启动服务（示例端口 8123）
pwsh -NoProfile -File run_py.ps1 研发/app/main.py serve --port 8123
# 2) 连跑 3 次（默认 3 轮、12 条无关问题）
pwsh -NoProfile -File run_py.ps1 研发/scripts/selftest_answerability_online.py http://127.0.0.1:8123
```

## 判定

``is_unknown=True`` **且**正文 == ``不清楚`` 记为拒答合格；**三次全部 12/12** 才返回 0，
否则返回 2（非 0 即未通过，便于脚本化断言）。本脚本**不写日志文件**（只发 HTTP 请求），
日志由服务端正常写入 `部署/日志/`。

## 退出码

- ``0`` 三次均 12/12；``2`` 有题目未被拒答；
- ``3`` **服务未就绪**（连接被拒 / 超时 / 健康检查未通过）：给出**友好提示**并显式以非 0 退出，
  不再抛原始 Traceback（第 2 轮复审 finding：服务未起时原文案不可读）。

为什么退出码要保持非 0：脚本化断言依赖它区分"服务没起"与"闸门放行"——
前者是**输入不可用**（不是闸门缺陷），后者才是真回归。
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(r"E:\gao6gongdan\工单2")
UNKNOWN_FILE = ROOT / "测试" / "测试数据" / "unknown_questions.jsonl"
BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8123"
ROUNDS = 3
UNKNOWN_TEXT = "不清楚"
HEALTH_PATH = "/api/health"
EXIT_SERVICE_UNAVAILABLE = 3
STARTUP_HINT = (
    "请先启动服务，例如：\n"
    "  pwsh -NoProfile -File run_py.ps1 研发/app/ui/serve_fallback.py --host 127.0.0.1 --port 8123\n"
    "  （或 部署/脚本/run_app.ps1；端口需与命令行给出的 base_url 一致）"
)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
#: stderr 同样显式 UTF-8（仓库内其它脚本的一致做法）：否则友好提示在 GBK 控制台/管道里会变乱码，
#: 而"服务未就绪"这条文案正是给**人**看的，必须可读（第 2 轮复审 finding 的同一诉求）。
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


class ServiceUnavailable(RuntimeError):
    """服务未就绪 / 连接失败（携带**友好文案**，避免向用户抛原始 Traceback）。"""


def _friendly(exc: BaseException) -> str:
    """把连接类异常转成一句可读提示（保留原始类型与原因，不隐藏信息）。"""
    reason = getattr(exc, "reason", None) or exc
    return f"[selftest] 服务未就绪：{BASE} 无响应（{type(reason).__name__}: {reason}）。\n{STARTUP_HINT}"


def check_service(timeout: float = 15.0) -> None:
    """预检 ``/api/health``；不可用则抛 :class:`ServiceUnavailable`（友好文案）。"""
    try:
        with urllib.request.urlopen(BASE + HEALTH_PATH, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except Exception as exc:  # URLError / TimeoutError / OSError / JSON 解码失败
        raise ServiceUnavailable(_friendly(exc)) from exc
    if not isinstance(body, dict) or not body.get("ok"):
        raise ServiceUnavailable(f"[selftest] 服务在 {BASE} 有响应但健康检查未通过：{body}")


def ask(question: str, timeout: float = 180.0) -> dict:
    """非流式 HTTP 提问（与在线测试同一路径）；连接类失败转友好异常。"""
    payload = json.dumps({"question": question, "stream": False}, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        BASE + "/api/ask",
        data=payload,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except ServiceUnavailable:
        raise
    except Exception as exc:  # 运行中途掉线/超时同样给友好文案
        raise ServiceUnavailable(_friendly(exc)) from exc


def main() -> int:
    """三次连跑并打印逐题结果。"""
    try:
        check_service()
    except ServiceUnavailable as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_SERVICE_UNAVAILABLE
    items = [
        json.loads(line)
        for line in UNKNOWN_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"服务：{BASE}；样本：{len(items)} 条；轮次：{ROUNDS}")
    ok_all = True
    for run in range(1, ROUNDS + 1):
        print("=" * 96)
        print(f"第 {run}/{ROUNDS} 次运行")
        passed = 0
        for item in items:
            started = time.perf_counter()
            try:
                body = ask(item["question"])
            except ServiceUnavailable as exc:
                print(f"\n{exc}", file=sys.stderr)
                print(
                    f"[selftest] 第 {run} 次运行中途失败（已完成 {passed}/{len(items)} 题）"
                    f"—— 结果不可判定，退出码非 0。",
                    file=sys.stderr,
                )
                return EXIT_SERVICE_UNAVAILABLE
            elapsed = (time.perf_counter() - started) * 1000
            answer = (body or {}).get("answer") or {}
            text = (answer.get("answer") or "").strip()
            is_unknown = bool(answer.get("is_unknown"))
            ok = is_unknown and text == UNKNOWN_TEXT
            passed += ok
            print(
                f"  [{'✅' if ok else '❌'}] #{item['id']:>2} {item['question'][:38]:<40} "
                f"is_unknown={is_unknown} text={text[:24]!r} {elapsed:>6.0f}ms"
            )
        ok_all = ok_all and passed == len(items)
        print(f"  —— 第 {run} 次：拒答 {passed}/{len(items)}")
    print("=" * 96)
    print(f"三次合计：{'全部 12/12 ✅' if ok_all else '存在未拒答 ❌'}")
    return 0 if ok_all else 2


if __name__ == "__main__":
    raise SystemExit(main())
