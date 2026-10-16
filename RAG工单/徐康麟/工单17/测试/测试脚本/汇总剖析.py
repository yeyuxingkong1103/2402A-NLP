# -*- coding: utf-8 -*-
"""工单编号：人工智能NLP-RAG-解决API 服务并发瓶颈与资源泄漏工单

把 `研发/剖析/*` 的产物汇总成一张前后对比表：``测试/测试脚本/汇总剖析.py``。

    pwsh -NoProfile -File run_py.ps1 测试/测试脚本/汇总剖析.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

WORK_ORDER = "人工智能NLP-RAG-解决API 服务并发瓶颈与资源泄漏工单"
_ROOT = Path(__file__).resolve().parents[2]
_DATA = _ROOT / "测试" / "测试数据"


def load(name: str) -> dict[str, Any] | None:
    path = _DATA / name
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def main(argv: list[str] | None = None) -> int:
    rows: list[dict[str, Any]] = []
    print("=" * 96)
    print("进程内剖析前后对比（urlopen 计数 / 分段耗时 / 内存）")
    print("=" * 96)
    for tag in ("优化前", "优化后"):
        prof = load(f"剖析问答链路_{tag}.json")
        mem = load(f"tracemalloc_{tag}.json")
        row: dict[str, Any] = {"tag": tag}
        if prof:
            u = prof.get("urlopen_calls", {})
            s = prof.get("stage_ms", {})
            row.update({
                "perf_enabled": prof.get("perf_enabled"),
                "urlopen_total": u.get("total"),
                "urlopen_per_request": u.get("per_request_total"),
                "api_tags": u.get("api_tags"),
                "api_tags_per_request": u.get("per_request_api_tags"),
                "api_embed": u.get("api_embed"),
                "api_generate": u.get("api_generate"),
                "by_url": u.get("by_url"),
                "retrieve_mean_ms": s.get("retrieve_mean"),
                "generate_mean_ms": s.get("generate_mean"),
                "total_mean_ms": s.get("total_mean"),
                "latency_mean_ms": (prof.get("latency_ms") or {}).get("mean"),
                "rss": prof.get("rss_mb"),
                "memory_cache": (prof.get("engine_stats") or {}).get("memory_cache"),
                "storage": (prof.get("engine_stats") or {}).get("storage"),
                "cprofile_top": (prof.get("cprofile_top") or [])[:8],
            })
        if mem:
            row.update({
                "tracemalloc_rss": mem.get("rss_mb"),
                "tracemalloc_objects": mem.get("objects"),
                "tracemalloc_top": (mem.get("tracemalloc_top") or [])[:10],
                "tracemalloc_curve_last": (mem.get("curve") or [])[-1] if mem.get("curve") else None,
            })
        rows.append(row)

        print(f"\n----- {tag}（PERF_ENABLED={row.get('perf_enabled')}）-----")
        if prof:
            print(f"  urlopen 总次数 / 每请求 : {row['urlopen_total']} / {row['urlopen_per_request']}")
            print(f"  /api/tags 探测 / 每请求 : {row['api_tags']} / {row['api_tags_per_request']}"
                  f"   ← 修复点 F2")
            print(f"  /api/embed / /api/generate: {row['api_embed']} / {row['api_generate']}")
            print(f"  按 URL 分桶            : {json.dumps(row['by_url'], ensure_ascii=False)}")
            print(f"  分段耗时 ms            : retrieve {row['retrieve_mean_ms']} / "
                  f"generate {row['generate_mean_ms']} / total {row['total_mean_ms']}")
            print(f"  RSS                    : {row['rss']}")
            print(f"  答案缓存条目           : {row['memory_cache']}")
            print(f"  落盘统计               : {json.dumps(row['storage'], ensure_ascii=False)}")
        else:
            print("  （缺少 剖析问答链路_*.json）")
        if mem:
            print(f"  tracemalloc RSS        : {json.dumps(row['tracemalloc_rss'], ensure_ascii=False)}")
            print(f"  tracemalloc 对象数     : {json.dumps(row['tracemalloc_objects'], ensure_ascii=False)}")
            print(f"  tracemalloc Top5       :")
            for item in (row.get("tracemalloc_top") or [])[:5]:
                print(f"      {item.get('size_diff_kb'):+10.1f} KB  {item.get('count_diff'):+6d}  "
                      f"{item.get('trace')}")
        else:
            print("  （缺少 tracemalloc_*.json）")
    print("\n" + "=" * 96)

    out = _DATA / "剖析对比汇总.json"
    out.write_text(json.dumps({"work_order": WORK_ORDER, "rows": rows},
                              ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"落盘：{out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
