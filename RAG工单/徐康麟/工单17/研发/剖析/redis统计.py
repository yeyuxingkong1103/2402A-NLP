# -*- coding: utf-8 -*-
"""工单编号：人工智能NLP-RAG-解决API 服务并发瓶颈与资源泄漏工单

Redis 队列/会话观测：``研发/剖析/redis统计.py``。

工单原文任务二-2-②：「检查 ``redis-cli`` 的 ``INFO stats`` 命令输出，观察任务队列堆积情况」。

本机 **未安装 redis-cli**（``Get-Command redis-cli`` 无结果，见 ``设计/环境事实.md`` §2），
复用 :mod:`service.session_store` 里工单10 已实现的标准库 RESP 客户端 ``MiniRedis``
（零依赖，实测 ``PING`` → ``+PONG``），直接取 ``INFO stats`` / ``INFO memory`` /
``DBSIZE`` / 队列长度。

用法::

    pwsh -NoProfile -File run_py.ps1 研发/剖析/redis统计.py --tag 优化后
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

WORK_ORDER = "人工智能NLP-RAG-解决API 服务并发瓶颈与资源泄漏工单"

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "研发"))

#: 关注的 INFO 字段（工单原文点名 INFO stats；这里连同 memory/keyspace 一起取）
WATCH = ("total_connections_received", "total_commands_processed", "instantaneous_ops_per_sec",
         "rejected_connections", "expired_keys", "evicted_keys", "keyspace_hits", "keyspace_misses",
         "total_error_replies", "used_memory_human", "used_memory_peak_human", "connected_clients")


def parse_info(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in str(text).splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, _sep, value = line.partition(":")
        out[key.strip()] = value.strip()
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=f"{WORK_ORDER} —— Redis 观测（redis-cli 替代）")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=6379)
    ap.add_argument("--db", type=int, default=15, help="会话库（工单17 本机等价部署用 db15）")
    ap.add_argument("--prefix", default="rag:finqa:")
    ap.add_argument("--tag", default="优化后")
    ap.add_argument("--outdir", default=str(_ROOT / "测试" / "测试数据"))
    args = ap.parse_args(argv)

    from service.session_store import MiniRedis

    out: dict[str, Any] = {"work_order": WORK_ORDER, "host": args.host, "port": args.port,
                           "db": args.db, "tag": args.tag}
    client = MiniRedis(args.host, args.port, db=args.db, timeout_s=3.0)
    try:
        out["ping"] = client.ping()
        stats = parse_info(client.do("INFO", "stats"))
        memory = parse_info(client.do("INFO", "memory"))
        clients = parse_info(client.do("INFO", "clients"))
        out["stats"] = {k: stats.get(k) for k in WATCH if k in stats}
        out["memory"] = {k: memory.get(k) for k in WATCH if k in memory}
        out["clients"] = {k: clients.get(k) for k in WATCH if k in clients}
        out["dbsize"] = int(client.do("DBSIZE") or 0)
        # 会话键数量（跨容器共享证据）
        keys: list[str] = []
        cursor = "0"
        while True:
            reply = client.do("SCAN", cursor, "MATCH", f"{args.prefix}session:*", "COUNT", 200)
            cursor, batch = str(reply[0]), list(reply[1] or [])
            keys.extend(batch)
            if cursor == "0" or len(keys) >= 5000:
                break
        out["session_keys"] = len(keys)
        out["session_key_sample"] = keys[:5]
        # 「任务队列堆积」口径：本工单的解析队列在进程内（PERF_PARSE_BACKEND=memory），
        # Redis 里存在的任何 rag:*:queue 键都是历史遗留，如实报告。
        qkeys: list[str] = []
        cursor = "0"
        while True:
            reply = client.do("SCAN", cursor, "MATCH", "rag:*queue*", "COUNT", 200)
            cursor, batch = str(reply[0]), list(reply[1] or [])
            qkeys.extend(batch)
            if cursor == "0" or len(qkeys) >= 1000:
                break
        out["queue_keys"] = qkeys
        out["queue_depths"] = {k: int(client.do("LLEN", k) or 0) for k in qkeys[:20]}
    finally:
        client.close()

    target = Path(args.outdir)
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"redis统计_{args.tag}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"PING                : {'PONG' if out.get('ping') else '失败'}")
    for section in ("stats", "memory", "clients"):
        for key, value in (out.get(section) or {}).items():
            print(f"  {section}.{key:<32} = {value}")
    print(f"  dbsize                        = {out['dbsize']}")
    print(f"  会话键（{args.prefix}session:*） = {out['session_keys']}")
    print(f"  队列键                        = {out['queue_keys'] or '（无）'}")
    print(f"落盘：{path}")
    return 0 if out.get("ping") else 1


if __name__ == "__main__":
    sys.exit(main())
