# -*- coding: utf-8 -*-
"""全量入库：医疗指南 → kb_medical，176 部法律 → kb_legal。

⚠️ 需在**后端服务停止**时运行（Qdrant 嵌入式模式持有独占文件锁）。
   服务运行时请改用 POST /api/kb/ingest。

用法：
    cd backend
    D:\\anaconda3\\envs\\rag_env\\python.exe scripts\\ingest_all.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core import config
from app.core.db import create_all
from app.services import ingest

DATASETS = config.DATASETS_DIR

JOBS = [
    # (路径, collection, 策略)
    (DATASETS / "medical", config.COLLECTION_MEDICAL, "auto"),
    (DATASETS / "legal" / "chinese_law", config.COLLECTION_LEGAL, "auto"),
]


def main() -> int:
    create_all()
    print("=" * 74)
    print("RAGLoRA 全量入库")
    print("=" * 74)

    grand_chunks = 0
    grand_files = 0

    for path, collection, strategy in JOBS:
        if not path.exists():
            print(f"\n[跳过] 路径不存在: {path}")
            continue

        files = ingest.collect_files(path)
        print(f"\n▶ {collection}")
        print(f"  路径: {path}")
        print(f"  文件: {len(files)} 个")
        print("-" * 74)

        t0 = time.time()
        result = ingest.ingest_path(str(path), collection, strategy)
        dt = time.time() - t0

        if not result.get("ok"):
            print(f"  !! 失败: {result.get('error')}")
            continue

        grand_chunks += result["total_chunks"]
        grand_files += result["succeeded"]

        failed = [r for r in result["results"] if r["status"] != "ready"]
        print(f"  成功 {result['succeeded']}/{result['files']} 文件"
              f" | {result['total_chunks']} chunks | 耗时 {dt:.1f}s")
        if failed:
            print(f"  失败 {len(failed)} 个：")
            for f in failed[:5]:
                print(f"     {f['file']}: {f.get('error', '')[:80]}")

    print()
    print("=" * 74)
    print("集合统计")
    print("=" * 74)
    for s in ingest.collection_stats():
        print(f"  {s['name']:<14} {s['points']:>7} 个向量")

    print()
    print("=" * 74)
    print(f"完成：{grand_files} 个文件，共 {grand_chunks} 个 chunk")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
