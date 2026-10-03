# -*- coding: utf-8 -*-
"""校验 部署/日志/ 三个日志文件：JSON Lines 可解析性 + 字段是否符合《接口设计》§7 schema。

工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
阶段：部署 / 日志验证

用法（工作目录 = E:\\gao6gongdan\\工单2）：
    pwsh -NoProfile -File run_py.ps1 -B 部署/脚本/verify_logs.py

退出码：0 全部合规；1 存在不合规项（详情打印）。
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = REPO_ROOT / "部署" / "日志"

# 《设计/接口设计.md》§7 要求的事件专属字段
EVENT_REQUIRED = {
    "enter": ("args", "kwargs"),
    "exit": ("elapsed_ms", "result"),
    "error": ("error_type", "error", "traceback"),
    "retrieval": ("query", "vector_hits", "bm25_hits", "pages", "rerank_mode"),
    "generation": ("mode", "context_pages", "first_token_ms"),
    "llm_io": ("backend", "prompt", "output"),
}
COMMON_REQUIRED = ("ts", "event", "module", "function")


def check_applog(path: Path, *, require_exception: bool) -> dict:
    total = parsed = 0
    missing = Counter()
    exceptions = 0
    levels = Counter()
    ts_min = ts_max = None
    for line in path.open(encoding="utf-8", errors="replace"):
        if not line.strip():
            continue
        total += 1
        try:
            obj = json.loads(line)
        except Exception:
            continue
        parsed += 1
        rec = obj.get("record")
        if not isinstance(rec, dict):
            missing["record"] += 1
            continue
        for key in ("text", "record.time", "record.level", "record.module", "record.function", "record.extra", "record.exception"):
            cur = obj
            ok = True
            for part in key.split("."):
                if isinstance(cur, dict) and part in cur:
                    cur = cur[part]
                else:
                    ok = False
                    break
            if not ok:
                missing[key] += 1
        lvl = ((rec.get("level") or {}).get("name") or "")
        levels[lvl] += 1
        exc = rec.get("exception")
        if isinstance(exc, dict):
            exceptions += 1
            if require_exception and not exc.get("traceback"):
                missing["record.exception.traceback"] += 1
        iso = (rec.get("time") or {}).get("iso")
        if iso:
            ts_min = iso if ts_min is None or iso < ts_min else ts_min
            ts_max = iso if ts_max is None or iso > ts_max else ts_max

    return {
        "file": path.name,
        "total_lines": total,
        "json_parsed": parsed,
        "levels": dict(levels),
        "exceptions": exceptions,
        "missing_fields": dict(missing),
        "ts_range": [ts_min, ts_max],
        "size_bytes": path.stat().st_size,
    }


def check_trace(path: Path) -> dict:
    total = parsed = 0
    events = Counter()
    missing = Counter()
    per_event_missing = defaultdict(Counter)
    ts_min = ts_max = None
    selfcheck: list[dict] = []
    for i, line in enumerate(path.open(encoding="utf-8", errors="replace"), 1):
        if not line.strip():
            continue
        total += 1
        try:
            obj = json.loads(line)
        except Exception:
            continue
        parsed += 1
        event = str(obj.get("event"))
        events[event] += 1
        miss = [key for key in COMMON_REQUIRED if key not in obj]
        miss += [key for key in EVENT_REQUIRED.get(event, ()) if key not in obj]
        # 日志系统自检记录（module=app.test）字段刻意最小，不属于业务事件；
        # 不隐藏：单独列出，便于复核者核对。
        if str(obj.get("module", "")).startswith("app.test"):
            selfcheck.append({"line": i, "event": event, "missing": miss, "ts": obj.get("ts")})
            continue
        for key in miss:
            missing[key] += 1
            per_event_missing[event][key] += 1
        ts = obj.get("ts")
        if ts:
            ts_min = ts if ts_min is None or ts < ts_min else ts_min
            ts_max = ts if ts_max is None or ts > ts_max else ts_max

    return {
        "file": path.name,
        "total_lines": total,
        "json_parsed": parsed,
        "events": dict(events),
        "missing_fields": dict(missing),
        "per_event_missing": {k: dict(v) for k, v in per_event_missing.items()},
        "selfcheck_records": selfcheck,
        "ts_range": [ts_min, ts_max],
        "size_bytes": path.stat().st_size,
    }


def main() -> int:
    print(f"日志目录: {LOG_DIR}\n")
    report = {"app.log": None, "error.log": None, "rag_trace.jsonl": None}
    problems: list[str] = []

    for name, require_exc in (("app.log", False), ("error.log", True)):
        path = LOG_DIR / name
        if not path.exists():
            problems.append(f"{name} 不存在")
            continue
        info = check_applog(path, require_exception=require_exc)
        report[name] = info
        print(f"--- {name} ---")
        print(f"  行数 {info['total_lines']}（JSON 可解析 {info['json_parsed']}）  大小 {info['size_bytes']/1024/1024:.2f} MB")
        print(f"  级别分布 {info['levels']}  含异常记录 {info['exceptions']}")
        print(f"  时间范围 {info['ts_range'][0]} → {info['ts_range'][1]}")
        print(f"  缺失字段 {info['missing_fields'] or '无'}")
        if info["json_parsed"] != info["total_lines"]:
            problems.append(f"{name}: 有 {info['total_lines'] - info['json_parsed']} 行不是合法 JSON")
        if info["missing_fields"]:
            problems.append(f"{name}: 字段缺失 {info['missing_fields']}")

    trace_path = LOG_DIR / "rag_trace.jsonl"
    if trace_path.exists():
        info = check_trace(trace_path)
        report["rag_trace.jsonl"] = info
        print("\n--- rag_trace.jsonl ---")
        print(f"  行数 {info['total_lines']}（JSON 可解析 {info['json_parsed']}）  大小 {info['size_bytes']/1024/1024:.2f} MB")
        print(f"  事件分布 {info['events']}")
        print(f"  时间范围 {info['ts_range'][0]} → {info['ts_range'][1]}")
        print(f"  缺失字段 {info['missing_fields'] or '无'}")
        if info["per_event_missing"]:
            print(f"  按事件细分 {info['per_event_missing']}")
        print(f"  自检记录（module=app.test，字段刻意最小，不计入业务事件）: {len(info['selfcheck_records'])} 条")
        for rec in info["selfcheck_records"]:
            print(f"     L{rec['line']} {rec['ts']} event={rec['event']} 缺 {rec['missing']}")
        if info["json_parsed"] != info["total_lines"]:
            problems.append("rag_trace.jsonl: 存在非法 JSON 行")
        if info["missing_fields"]:
            problems.append(f"rag_trace.jsonl: 业务事件字段缺失 {info['missing_fields']}")

    print("\n--- 结论 ---")
    if problems:
        for p in problems:
            print(f"  ❌ {p}")
        print("  总体: ❌ 不通过")
        return 1
    print("  ✅ 三个日志文件均为合法 JSON Lines，且必填字段齐全")
    print("  总体: ✅ 通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
