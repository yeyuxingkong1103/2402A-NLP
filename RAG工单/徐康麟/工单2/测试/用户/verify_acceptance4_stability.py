"""验收 4 稳定性三连跑：12 条无关问题 ×3 次独立运行，产出**正式可引用产物**。

工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
阶段：测试 / 用户（关单复验证据产物；captain 2026-10-03 指派给 tester）

## 判据（本脚本自带，不依赖 pytest 进程）

每条无关问题的回答须满足：**``is_unknown is True`` 且 ``answer.answer.strip() == "不清楚"``**；
**三次独立运行**（每次**新起一个 ``serve_fallback`` 服务进程**、随机端口、各自预热）**每次均须 12/12**。

为什么必须三次：t13/t1 期间实测同一批问题在两次运行中**失败题不同**（8/12 与 10/12），
故"一次通过"不构成稳定性证据；判据与 `测试/在线/test_fault_tolerance.py` 的
``test_unrelated_questions_answer_unknown`` 完全一致（同一条 HTTP 链路、同一批数据文件）。

## 产物

默认写 ``优化/评估结果/验收4_稳定性_三连跑.md``（正式命名，**不是** ``_v3_*``/``_tmp*`` 临时件），
含：三轮逐题结果（题号/是否 is_unknown/答案文本/拒答原因）、每次运行时间戳、
被测代码 mtime 对照（窗口内一动即标注"本次复验作废"）、判定与断言方式说明。
``--output`` 可改写到临时位置（供不带冻结窗口的干跑使用）。

## 用法

```
pwsh -NoProfile -File run_py.ps1 测试/用户/verify_acceptance4_stability.py
pwsh -NoProfile -File run_py.ps1 测试/用户/verify_acceptance4_stability.py --rounds 1 --output .tmp_review/x.md
```

退出码：``0`` = 三次均 12/12 且窗口内实现文件未移动；``1`` = 存在拒答失败；
``2`` = 启动失败或窗口内实现文件发生移动（复验作废）。
"""

from __future__ import annotations

import argparse
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

TESTS_DIR = Path(__file__).resolve().parent.parent          # 测试/
PROJECT_ROOT = TESTS_DIR.parent                              # 工单2
SOURCE_ROOT = PROJECT_ROOT / "研发"
TEST_DATA = TESTS_DIR / "测试数据"
SERVE_SCRIPT = SOURCE_ROOT / "app" / "ui" / "serve_fallback.py"
LOG_DIR = PROJECT_ROOT / "部署" / "日志"
DEFAULT_OUT = PROJECT_ROOT / "优化" / "评估结果" / "验收4_稳定性_三连跑.md"

UNKNOWN_FILE = TEST_DATA / "unknown_questions.jsonl"
UNKNOWN_TEXT = "不清楚"
ADDR_RE = re.compile(r"http://(?P<host>\d+\.\d+\.\d+\.\d+):(?P<port>\d+)/")
START_TIMEOUT_S = 180.0

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# --------------------------------------------------------------------------
# HTTP 与进程小工具（镜像 测试/在线/conftest.py 的口径，保持独立、可单跑）
# --------------------------------------------------------------------------
def post_json(base_url: str, path: str, payload: dict, timeout: float = 180.0):
    """POST JSON，返回 ``(status, 解析后的 JSON)``（4xx/5xx 也读结构化错误体）。"""
    req = urllib.request.Request(
        base_url + path,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            return exc.code, json.loads(body)
        except Exception:
            return exc.code, {"ok": False, "raw": body}


def get_json(base_url: str, path: str, timeout: float = 60.0):
    try:
        with urllib.request.urlopen(base_url + path, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8", errors="replace") or "{}")


class Service:
    """一个真实运行的 ``serve_fallback`` 服务进程（随机端口）。"""

    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None
        self.base_url = ""
        self.lines: list[str] = []

    def start(self) -> None:
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        self.proc = subprocess.Popen(
            [sys.executable, str(SERVE_SCRIPT), "--host", "127.0.0.1", "--port", "0"],
            cwd=str(PROJECT_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", env=env,
        )
        lock = threading.Lock()

        def reader() -> None:
            assert self.proc is not None and self.proc.stdout is not None
            for line in self.proc.stdout:
                with lock:
                    self.lines.append(line.rstrip("\n"))

        threading.Thread(target=reader, daemon=True).start()
        deadline = time.time() + START_TIMEOUT_S
        while time.time() < deadline:
            with lock:
                for line in self.lines:
                    match = ADDR_RE.search(line)
                    if match:
                        self.base_url = f"http://{match.group('host')}:{match.group('port')}"
                        break
            if self.base_url:
                break
            if self.proc.poll() is not None:
                raise RuntimeError("服务进程提前退出：\n" + "\n".join(self.lines[-15:]))
            time.sleep(0.5)
        if not self.base_url:
            raise RuntimeError("未能在超时内解析到服务地址：\n" + "\n".join(self.lines[-15:]))
        while time.time() < deadline:
            try:
                status, body = get_json(self.base_url, "/api/health", timeout=15)
                if status == 200 and body.get("ok"):
                    return
            except Exception:
                pass
            time.sleep(0.5)
        raise RuntimeError(f"服务未就绪（{self.base_url}）")

    def stop(self) -> None:
        if self.proc is None or self.proc.poll() is not None:
            return
        self.proc.terminate()
        try:
            self.proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=15)


# --------------------------------------------------------------------------
# 日志：只读「本轮新增字节」，提取拒答原因（trace_id 关联）
# --------------------------------------------------------------------------
def _log_size(name: str) -> int:
    path = LOG_DIR / name
    return path.stat().st_size if path.exists() else 0


def _read_new_bytes(name: str, offset: int) -> str:
    path = LOG_DIR / name
    if not path.exists():
        return ""
    with path.open("rb") as fh:
        fh.seek(offset)
        return fh.read().decode("utf-8", errors="replace")


def _reason_map(appended: str) -> tuple[dict[str, str], int]:
    """从 app.log 新增字节里抽出 ``trace_id -> 拒答原因``；返回 (映射, 非法 JSON 行数)。"""
    mapping: dict[str, str] = {}
    invalid = 0
    for line in appended.splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except Exception:
            invalid += 1
            continue
        rec = record.get("record") or {}
        extra = rec.get("extra") or {}
        trace_id = str(extra.get("trace_id") or "")
        text = str(record.get("text") or "")
        if not trace_id:
            continue
        reason = str(extra.get("reason") or "")
        if reason:
            mapping[trace_id] = f"{text}: {reason}"
        elif "可答性校验不通过" in text:
            mapping[trace_id] = f"{text}(hits={extra.get('hits')}/{extra.get('terms')})"
        elif "置信度不足" in text:
            mapping[trace_id] = (f"{text}(top_cosine={extra.get('top_cosine')}"
                                 f"<{extra.get('min_cosine')})")
    return mapping, invalid


# --------------------------------------------------------------------------
# mtime 快照与对照
# --------------------------------------------------------------------------
def snapshot_mtimes() -> list[dict]:
    """`研发/` 下全部 ``.py`` 的 mtime 与体积（排除 __pycache__）。"""
    rows = []
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        stat = path.stat()
        rows.append({
            "path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)),
            "size": stat.st_size,
        })
    return rows


def diff_mtimes(before: list[dict], after: list[dict]) -> list[str]:
    """返回发生移动/改动的文件描述（空列表 = 窗口内实现文件未动）。"""
    index_before = {row["path"]: row for row in before}
    index_after = {row["path"]: row for row in after}
    changed: list[str] = []
    for path, row in index_after.items():
        old = index_before.get(path)
        if old is None:
            changed.append(f"{path}: 新增（窗口前不存在）")
        elif (old["mtime"], old["size"]) != (row["mtime"], row["size"]):
            changed.append(f"{path}: {old['mtime']}({old['size']}B) → {row['mtime']}({row['size']}B)")
    for path in index_before:
        if path not in index_after:
            changed.append(f"{path}: 窗口内被删除")
    return changed


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def load_unknown() -> list[dict]:
    return [json.loads(line) for line in UNKNOWN_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_round(round_index: int, questions: list[dict]) -> dict:
    """跑一轮：新起服务 → 预热 → 12 题逐条提问 → 收服务 → 关联拒答原因。"""
    log_offsets = {name: _log_size(name) for name in ("app.log", "rag_trace.jsonl")}
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    service = Service()
    rows: list[dict] = []
    cold_first_ms: float | None = None
    try:
        service.start()
        t0 = time.perf_counter()
        status, warm_body = post_json(service.base_url, "/api/ask", {"question": "今天天气怎么样？", "stream": False})
        cold_first_ms = round((time.perf_counter() - t0) * 1000, 1)
        print(f"  [预热] 首次提问（冷启动）{cold_first_ms} ms，status={status}")
        for item in questions:
            t1 = time.perf_counter()
            status, body = post_json(service.base_url, "/api/ask",
                                     {"question": item["question"], "stream": False})
            elapsed = round((time.perf_counter() - t1) * 1000, 1)
            answer = (body or {}).get("answer") or {}
            text = str(answer.get("answer") or "").strip()
            is_unknown = bool(answer.get("is_unknown"))
            ok = is_unknown and text == UNKNOWN_TEXT
            rows.append({
                "id": item.get("id"),
                "category": item.get("category", ""),
                "question": item["question"],
                "status": status,
                "is_unknown": is_unknown,
                "answer": text,
                "ok": ok,
                "api_unknown_reason": str(answer.get("unknown_reason") or ""),
                "trace_id": str(answer.get("trace_id") or ""),
                "elapsed_ms": elapsed,
            })
            print(f"   [{'✅' if ok else '❌'}] #{item.get('id')} {item['question'][:30]!r} → "
                  f"is_unknown={is_unknown} text={text[:16]!r}")
    finally:
        service.stop()
    mapping, invalid_lines = _reason_map(_read_new_bytes("app.log", log_offsets["app.log"]))
    trace_new_lines = len([ln for ln in _read_new_bytes("rag_trace.jsonl", log_offsets["rag_trace.jsonl"]).splitlines() if ln.strip()])
    for row in rows:
        row["refusal_reason"] = mapping.get(row["trace_id"], row["api_unknown_reason"] or "(日志未给出原因)")
    passed = sum(1 for row in rows if row["ok"])
    print(f"  —— 第 {round_index} 次：拒答 {passed}/{len(rows)}")
    return {
        "round": round_index,
        "started_at": started,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cold_first_ms": cold_first_ms,
        "rows": rows,
        "passed": passed,
        "total": len(rows),
        "app_log_new_lines": len([ln for ln in _read_new_bytes("app.log", log_offsets["app.log"]).splitlines() if ln.strip()]),
        "app_log_invalid_json_lines": invalid_lines,
        "trace_new_lines": trace_new_lines,
        "service_tail": service.lines[-5:],
    }


def build_markdown(rounds: list[dict], mtime_before: list[dict], mtime_after: list[dict],
                   changed: list[str], out_path: Path) -> str:
    """拼装正式产物（Markdown）。"""
    all_ok = all(r["passed"] == r["total"] and r["total"] > 0 for r in rounds)
    stable = all_ok and not changed
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        shown_path = out_path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:                      # 产物被写到工作区之外/相对路径临时位置
        shown_path = str(out_path)
    lines: list[str] = [
        "# 验收 4 稳定性三连跑（不知道时回复「不清楚」）",
        "",
        "> 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化",
        "> 阶段：测试 / 用户 —— 关单复验正式证据产物（captain 2026-10-03 指派给 tester）",
        "> 生成命令：`pwsh -NoProfile -File run_py.ps1 测试/用户/verify_acceptance4_stability.py`",
        f"> 生成时间：{now}",
        f"> 本文件路径：`{shown_path}`",
        "",
        "## 1. 判定结论",
        "",
    ]
    counts = " / ".join(f'{r["passed"]}/{r["total"]}' for r in rounds)
    lines += [
        f"- **三次运行拒答数**：{counts}（判据：`is_unknown is True` 且答案文本 `strip() == \"不清楚\"`）",
        f"- **稳定性**：{'三次均 12/12 ✅ 通过' if all_ok else '存在未拒答项 ❌ 不通过'}",
        f"- **取证窗口内被测代码是否移动**：{'未移动 ✅' if not changed else f'发生移动 ❌（本次复验作废）: {changed}'}",
        f"- **整体判定**：{'**通过**（验收 4 达标）' if (all_ok and not changed) else '**不通过 / 不可判定**'}",
        "",
        "## 2. 被测对象与取证窗口",
        "",
        f"- 实现文件快照数：开窗前 {len(mtime_before)} 个、收窗后 {len(mtime_after)} 个（`研发/**/*.py`，排除 `__pycache__`）",
        f"- 窗口内移动/改动：**{len(changed)} 个**" + ("（无）" if not changed else ""),
        "",
    ]
    if changed:
        lines += ["| 发生移动的文件 |", "| --- |"] + [f"| {item} |" for item in changed] + [""]
    else:
        lines += ["| 文件 | mtime（开窗前 = 收窗后） | 体积(B) |", "| --- | --- | --- |"]
        lines += [f"| `{row['path']}` | {row['mtime']} | {row['size']} |"
                  for row in mtime_after if row["path"].startswith("研发/app/core/")]
        lines += [""]
    lines += [
        "> 窗口取值方式：`--mtime-before` / `--mtime-after` 两个快照（JSON）由复验人分别在"
        "**开窗时**与**收窗时**采集；本产物直接嵌入其对照结果，窗口内任一实现文件移动即判「复验作废」。",
        "",
    ]
    for r in rounds:
        lines += [
            f"## 3.{r['round']} 第 {r['round']} 次运行（{r['started_at']} ~ {r['finished_at']}，拒答 {r['passed']}/{r['total']}）",
            "",
            f"- 冷启动首次提问耗时：{r['cold_first_ms']} ms（预热用；不计入稳态预算）",
            f"- 本轮新增日志：`app.log` {r['app_log_new_lines']} 行（非法 JSON {r['app_log_invalid_json_lines']} 行）、"
            f"`rag_trace.jsonl` {r['trace_new_lines']} 行",
            "",
            "| 题号 | 分类 | 问题 | is_unknown | 答案文本 | 拒答原因 | 耗时(ms) | 判定 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for row in r["rows"]:
            reason = row["refusal_reason"].replace("|", "\\|")[:60]
            lines.append(
                f"| {row['id']} | {row['category']} | {row['question'][:34]} | {row['is_unknown']} | "
                f"{row['answer'][:12]} | {reason} | {row['elapsed_ms']} | {'✅' if row['ok'] else '❌'} |"
            )
        lines += [""]
    lines += ["## 4. 三次逐题对照", "",
              "| 题号 | 问题 | 第1次 | 第2次 | 第3次 |", "| --- | --- | --- | --- | --- |"]
    for index in range(len(rounds[0]["rows"])):
        cells = []
        for r in rounds:
            row = r["rows"][index]
            cells.append("✅" if row["ok"] else f"❌({row['answer'][:8] or '空'})")
        base = rounds[0]["rows"][index]
        lines.append(f"| {base['id']} | {base['question'][:34]} | " + " | ".join(cells) + " |")
    lines += [""]
    lines += [
        "## 5. 断言方式与纪律说明",
        "",
        "- **判据**：每条无关问题的回答满足 `answer.is_unknown is True` **且** `answer.answer.strip() == \"不清楚\"`；",
        "  该判据与 `测试/在线/test_fault_tolerance.py::test_unrelated_questions_answer_unknown` 完全一致。",
        "- **独立运行**：三次运行各自**新起**一个 `serve_fallback` 服务进程（随机端口、各自预热），",
        "  互不复用会话；因此三次失败题若不同，即暴露非确定性，不能用「一次通过」代替。",
        "- **未放宽任何断言**：本脚本只读 HTTP 响应与新增日志字节，未修改任何 core/测试断言，",
        "  未使用 xfail、未删用例、未修改 `check_answer` 与 `FUZZY_THRESHOLD`。",
        "- **拒答原因来源**：优先取 HTTP 响应字段 `unknown_reason`；为空时从**本轮新增** `app.log` 字节中",
        "  按 `trace_id` 关联闸门日志（`_topic_gate` 的 `reason` / `可答性校验不通过` / `置信度不足`）。",
        "- **日志只读窗口**：只解析本轮新增字节，历史行（含修复前 `Errno 22` 痕迹）不参与判定、不清理。",
        "",
        "## 6. 原始证据位置",
        "",
        "- HTTP：`POST /api/ask`（`stream=false`）逐题调用；服务为 `研发/app/ui/serve_fallback.py`。",
        "- 日志：`部署/日志/app.log`、`部署/日志/rag_trace.jsonl`（均只读、追加）。",
        "- mtime 快照 JSON：由复验人在窗口两端采集（`--mtime-before` / `--mtime-after`）。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="验收 4 稳定性三连跑（产出正式产物）")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--mtime-before", type=Path, default=None)
    parser.add_argument("--mtime-after", type=Path, default=None)
    parser.add_argument("--unknown-file", type=Path, default=UNKNOWN_FILE)
    args = parser.parse_args()

    questions = [json.loads(line) for line in args.unknown_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    print("=" * 78)
    print(f"验收 4 稳定性三连跑：{len(questions)} 条无关问题 × {args.rounds} 次独立运行")
    print("=" * 78)

    if args.mtime_before and args.mtime_before.exists():
        mtime_before = json.loads(args.mtime_before.read_text(encoding="utf-8"))
    else:
        mtime_before = snapshot_mtimes()
    rounds: list[dict] = []
    for index in range(1, args.rounds + 1):
        print(f"\n--- 第 {index}/{args.rounds} 次运行（新起服务）---")
        try:
            rounds.append(run_round(index, questions))
        except Exception as exc:  # noqa: BLE001
            print(f"第 {index} 次运行失败：{type(exc).__name__}: {exc}")
            rounds.append({"round": index, "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                           "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"), "cold_first_ms": None,
                           "rows": [], "passed": 0, "total": len(questions),
                           "app_log_new_lines": 0, "app_log_invalid_json_lines": 0,
                           "trace_new_lines": 0, "service_tail": [f"{type(exc).__name__}: {exc}"]})
    if args.mtime_after and args.mtime_after.exists():
        mtime_after = json.loads(args.mtime_after.read_text(encoding="utf-8"))
    else:
        mtime_after = snapshot_mtimes()
    changed = diff_mtimes(mtime_before, mtime_after)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_markdown(rounds, mtime_before, mtime_after, changed, args.output),
                        encoding="utf-8")
    all_ok = all(r["passed"] == r["total"] and r["total"] > 0 for r in rounds)
    counts = " / ".join(f'{r["passed"]}/{r["total"]}' for r in rounds)
    print("\n" + "=" * 78)
    print(f"三次拒答：{counts}；窗口内实现文件移动：{len(changed)} 个")
    print(f"产物：{args.output}")
    print(f"整体判定：{'通过 ✅' if (all_ok and not changed) else '不通过/不可判定 ❌'}")
    print("=" * 78)
    if changed:
        return 2
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
