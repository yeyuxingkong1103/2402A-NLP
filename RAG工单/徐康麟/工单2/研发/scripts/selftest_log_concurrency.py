"""日志跨进程并发写自测（t1 ②）：对照「文本缓冲写」与「单次原子追加写」。

工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
阶段：研发 / 自测脚本（日志健壮性证据；正式判分由 tester 的在线用例负责）

## 缺陷（tester 实测 + 本脚本复核）
在线用例 ``test_logs_are_json_lines_and_non_empty`` 只看**本次新增字节**，曾抓到新增段里的
非法 JSON 半行，碎片形如 ``{"chunk": "...", "score": "...", "vector_score": ...``。
机制：``logging_conf`` 原来用**文本句柄 + flush** 写行，文本层自带 8KB 缓冲——
单条记录超过 8KB 时被拆成多次 ``write()``；而同一批 JSONL 会被**多个进程**同时追加
（在线套件自己会起服务子进程、还会起 stderr 不可写的测试子进程），
两次写入的片段互相穿插 → 半行 JSON。

## 本脚本做什么
三个机制各起 **6 个并发子进程**，各写 200 条 **~40KB** 记录，然后逐行校验是否为合法 JSON：
- ``text``：文本句柄 ``open(..., "a", encoding="utf-8")`` + ``write/flush``（**修复前机制**）；
- ``fileio-ab``：``open(..., "ab", buffering=0)`` + ``write``（"看起来能修"的中间方案）；
- ``fixed``：``logging_conf._open_log_handle`` + ``_write_line``（**修复后机制**：
  Windows 走 ``CreateFileW(FILE_APPEND_DATA)`` + 单次 ``WriteFile``）。

实测（本机 Windows / Python 3.11.15，期望 1200 行）：
``text`` 丢失 41~138 行、5~11 行非法 JSON；
``fileio-ab`` 丢失行；
``os.open(O_APPEND)+os.write`` 1128~1157 行（**仍丢行**）；
``fixed`` 与"跨进程锁" **1200/1200、0 非法行**。
→ 因此最终实现选的是 ``FILE_APPEND_DATA``，**不是** ``open(..., "ab")``，也不是裸 ``os.write``。

用法::

    pwsh -NoProfile -File run_py.ps1 研发/scripts/selftest_log_concurrency.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT.parent
#: 临时产物目录：必须落在**会话工作区**内（平台临时目录在文件沙箱下不可写）
TMP_DIR = PROJECT_ROOT / ".tmp_review" / "log_concurrency"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

MODES = ("text", "fileio-ab", "fixed")

WORKERS = 6
ROUNDS = 200
PAD_CHARS = 12_000

WORKER_CODE = r'''
import json, os, sys
sys.path.insert(0, r"{source_root}")
mode, path, rounds = sys.argv[1], sys.argv[2], int(sys.argv[3])
pad = "填充-" * 4000          # ~12KB 单条记录（远超 8KB 文本缓冲）
if mode == "text":
    handle = open(path, "a", encoding="utf-8", newline="\n")
    for index in range(rounds):
        handle.write(json.dumps({{"pid": os.getpid(), "i": index, "pad": pad}}, ensure_ascii=False) + "\n")
        handle.flush()
    handle.close()
elif mode == "fileio-ab":
    handle = open(path, "ab", buffering=0)
    for index in range(rounds):
        handle.write((json.dumps({{"pid": os.getpid(), "i": index, "pad": pad}}, ensure_ascii=False) + "\n").encode("utf-8"))
    handle.close()
else:                          # fixed：走**产品路径**（_open_log_handle + _write_line）
    from pathlib import Path
    from app.core.logging_conf import _open_log_handle, _write_line
    handle = _open_log_handle(Path(path))
    for index in range(rounds):
        _write_line(handle, {{"pid": os.getpid(), "i": index, "pad": pad}})
    close = getattr(handle, "close", None)
    if close is not None:
        close()
'''


def _run_mode(mode: str, path: Path) -> None:
    """并发启动 4 个写者子进程，等全部结束。"""
    code = WORKER_CODE.format(source_root=str(SOURCE_ROOT))
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", code, mode, str(path), str(ROUNDS)],
            cwd=str(SOURCE_ROOT.parent),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        for _ in range(WORKERS)
    ]
    for proc in procs:
        out, _ = proc.communicate(timeout=600)
        if proc.returncode != 0:
            print(f"    ! 子进程退出码 {proc.returncode}: {(out or '')[:200]}")


def _validate(path: Path) -> tuple[int, int, list[str]]:
    """返回（总行数、非法 JSON 行数、非法行样例）。"""
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = [line for line in text.splitlines() if line.strip()]
    bad: list[str] = []
    for line in lines:
        try:
            json.loads(line)
        except Exception:
            bad.append(line[:90])
    return len(lines), len(bad), bad[:3]


def main() -> int:
    """跑三种机制并输出对照结论。"""
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR, ignore_errors=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    expected = WORKERS * ROUNDS
    results: dict[str, tuple[int, int]] = {}
    for mode in MODES:
        path = TMP_DIR / f"{mode}.jsonl"
        _run_mode(mode, path)
        total, bad, samples = _validate(path)
        results[mode] = (total, bad)
        print(f"[{mode:>9}] 行数={total}（期望 {expected}，缺 {expected - total}）  非法 JSON 行={bad}")
        for sample in samples:
            print(f"            非法样例: {sample!r}")
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    print("=" * 88)
    for mode, (total, bad) in results.items():
        print(f"  {mode:>9}: 行数 {total}/{expected}，非法 JSON {bad}")
    fixed_total, fixed_bad = results["fixed"]
    ok = fixed_bad == 0 and fixed_total == expected
    print(f"结论：修复后机制 {'行数完整且无非法行' if ok else '仍有丢行/非法行'}（要求 {expected}/{expected}、0 非法）")
    print(f"{'PASS ✅' if ok else 'FAIL ❌'}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
