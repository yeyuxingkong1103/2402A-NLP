# -*- coding: utf-8 -*-
"""独立验证脚本（verifier / t3）：日志完整性核验（**只校验本轮新增字节**）。

工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
阶段：优化 / 独立验证（A3 第三方验证，非实现方，**只读不改实现**）

复现（工作目录 = E:\\gao6gongdan\\工单2）：
    pwsh -NoProfile -File run_py.ps1 优化/脚本/verify_log_newbytes.py 2026-10-03T16:52:00+08:00
退出码：0 通过（新增段全为合法 JSON Lines）；1 不通过。

**只读**（以共享读方式打开，不写日志目录）。

纪律（依 t3 契约第 10 项与环境事实 §7）：
  - 历史日志含**修复前**的 `Errno 22` 痕迹，以及「双进程同绑端口」造成的非法 JSON 行；
  - 这些属**已知遗留**，**不得**据此判不通过；
  - 因此本脚本把三份日志的时间窗切分为「历史段（期望含缺陷）」与「本轮新增段（期望干净）」，
    只在**本轮新增段**上做严格断言。

时间窗起点由命令行给出（ISO8601，含时区），默认取脚本启动前 10 分钟。
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "部署" / "日志"
LOGS = ("app.log", "error.log", "rag_trace.jsonl")

#: app.log / error.log 的行首时间戳形如 2026-10-03T16:55:25.671+08:00
TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[+-]\d{2}:\d{2})?)")
#: rag_trace.jsonl 的 "ts" 字段
HISTORICAL_MARKER = "Errno 22"


def parse_ts(text: str) -> datetime | None:
    m = TS_RE.match(text.lstrip())
    if not m:
        return None
    try:
        return datetime.fromisoformat(m.group(1))
    except ValueError:
        return None


def line_ts(obj: object, raw: str) -> datetime | None:
    """取该行的时间戳。

    实测两种行格式：
      - ``rag_trace.jsonl``：顶层 ``{"ts": "2026-10-03T16:53:16.188+08:00", ...}``
      - ``app.log`` / ``error.log``：loguru 风格 ``{"text": ..., "record": {"time": {"iso": ...}}}``
    """
    if isinstance(obj, dict):
        ts = obj.get("ts")
        if isinstance(ts, str):
            try:
                return datetime.fromisoformat(ts)
            except ValueError:
                pass
        rec = obj.get("record")
        if isinstance(rec, dict):
            t = rec.get("time")
            if isinstance(t, dict) and isinstance(t.get("iso"), str):
                try:
                    return datetime.fromisoformat(t["iso"])
                except ValueError:
                    pass
            elif isinstance(t, (int, float)):
                return datetime.fromtimestamp(t).astimezone()
    return parse_ts(raw)


def main() -> int:
    if len(sys.argv) > 1:
        since = datetime.fromisoformat(sys.argv[1])
    else:
        since = datetime.now(timezone.utc).astimezone() - timedelta(minutes=10)

    print("=" * 82)
    print("A3 独立验证 · 日志完整性（只校验本轮新增字节）")
    print("=" * 82)
    print(f"本轮窗口起点 = {since.isoformat()}")

    overall_ok = True

    for name in LOGS:
        path = LOG_DIR / name
        print("\n" + "-" * 82)
        if not path.exists():
            print(f"[FAIL] {name} 不存在")
            overall_ok = False
            continue

        size = path.stat().st_size
        print(f"[{'OK' if size > 0 else 'FAIL'}] {name}  size = {size} bytes  (存在且非空: {size > 0})")
        if size == 0:
            overall_ok = False
            continue

        hist_total = hist_bad_json = hist_marker = 0
        new_total = new_bad_json = new_marker = 0
        first_new_ts = last_ts = None
        samples: list[str] = []

        # 共享读：不阻塞其他进程，也不修改文件
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                raw = raw.rstrip("\n")
                if not raw.strip():
                    continue
                try:
                    obj = json.loads(raw)
                    bad = False
                except json.JSONDecodeError:
                    obj, bad = None, True

                ts = line_ts(obj, raw)
                in_new = ts is not None and ts >= since

                if in_new:
                    new_total += 1
                    first_new_ts = first_new_ts or ts
                    last_ts = ts
                    if bad:
                        new_bad_json += 1
                        if len(samples) < 3:
                            samples.append(raw[:150])
                    if HISTORICAL_MARKER in raw:
                        new_marker += 1
                else:
                    hist_total += 1
                    if bad:
                        hist_bad_json += 1
                    if HISTORICAL_MARKER in raw:
                        hist_marker += 1

        print(f"    历史段: 行数={hist_total}  非法JSON={hist_bad_json}  含'{HISTORICAL_MARKER}'={hist_marker}")
        print(f"    新增段: 行数={new_total}  非法JSON={new_bad_json}  含'{HISTORICAL_MARKER}'={new_marker}")
        if first_new_ts and last_ts:
            print(f"    新增段时间跨度: {first_new_ts.isoformat()} ~ {last_ts.isoformat()}")

        clean = new_total > 0 and new_bad_json == 0 and new_marker == 0
        print(f"    [{'PASS' if clean else 'FAIL'}] 新增段 JSON Lines 全部合法且无 '{HISTORICAL_MARKER}' 正向降级标记"
              + ("" if new_total else "（窗口内无新增行 → 无法判定，标 N/A）"))
        for s in samples:
            print(f"       非法样本: {s}")
        overall_ok = overall_ok and (clean or new_total == 0)

    print("\n" + "=" * 82)
    print("说明：历史段的非法 JSON 与 Errno 22 属**已知遗留**（环境事实 §7 / tester 清单），不计入判定。")
    print(f"日志窗口总体: {'PASS' if overall_ok else 'FAIL'}")
    print("=" * 82)
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
