# -*- coding: utf-8 -*-
"""工单9 脚本：构建「优化后图谱」（抽取层优化）。

工单编号：人工智能NLP-RAG-Graph RAG 优化任务

流程：读 工单8 图谱（只读）→ 别名归并 → 关系方向/类型规范化 → 噪声剪枝 + 置信度过滤
      → 写 研发/data/图谱优化后/{entities,relations}.jsonl + 图谱优化统计.json

资源纪律：单进程、流式读写，实测峰值 < 200 MB（远低于 2 GB 上限）。

用法：
    pwsh -NoProfile -File run_py.ps1 研发/scripts/构建优化图谱.py
    pwsh -NoProfile -File run_py.ps1 研发/scripts/构建优化图谱.py --base 工单8图谱目录 --out 输出目录
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

WORK_ORDER = "人工智能NLP-RAG-Graph RAG 优化任务"

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from app.graphrag_opt.配置 import ensure_dirs, get_config  # noqa: E402
from app.graphrag_opt.抽取层 import refine_graph  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"{WORK_ORDER} —— 构建优化后图谱")
    parser.add_argument("--base", default=None, help="工单8 图谱目录（只读，默认取配置）")
    parser.add_argument("--out", default=None, help="输出目录（默认 研发/data/图谱优化后）")
    parser.add_argument("--json", action="store_true", help="打印完整统计 JSON")
    args = parser.parse_args(argv)

    cfg = get_config()
    ensure_dirs(cfg)
    result = refine_graph(cfg, base_graph_dir=Path(args.base) if args.base else None,
                          out_dir=Path(args.out) if args.out else None)
    stats = result["stats"]
    if args.json:
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return 0

    print("=" * 74)
    print(f"{WORK_ORDER} —— 实体/关系构建层优化结果")
    print("=" * 74)
    print(f"输入图谱：{stats['输入']['graph_dir']}")
    print(f"  节点 {stats['输入']['节点']}  边 {stats['输入']['边']}  平均度数 {stats['输入']['平均度数']}")
    print(f"输出图谱：{stats['输出']['graph_dir']}")
    print(f"  节点 {stats['输出']['节点']}  边 {stats['输出']['边']}  平均度数 {stats['输出']['平均度数']}")
    print("-" * 74)
    print("① 别名归并：", json.dumps(stats["别名归并"], ensure_ascii=False))
    print("② 关系规范化：", json.dumps(stats["关系规范化"], ensure_ascii=False))
    print("③ 噪声剪枝：", json.dumps(stats["噪声剪枝"], ensure_ascii=False))
    print(f"   度数阈值 = {stats['度数阈值']}")
    print("-" * 74)
    print("归并组样例（前 8 组）：")
    for group in stats["归并组样例"][:8]:
        print(f"   {group['canonical']}  ←  {group['aliases']}")
    print("剪除样例（前 8 条）：")
    for item in stats["剪除样例"][:8]:
        print(f"   [{item['type']}] {item['name']}  度={item['degree']} 提及={item['mentions']}  原因={item['reason']}")
    print("=" * 74)
    print(f"统计落盘：{result['stats_path']}（耗时 {stats['elapsed_ms']} ms）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
