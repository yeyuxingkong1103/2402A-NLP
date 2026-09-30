# -*- coding: utf-8 -*-
"""把 Qdrant 里的存量向量迁移到 Milvus（**不重新编码**）。

为什么直接搬运向量而不是重新编码
--------------------------------
1. 重新编码 14955 条要跑约 4 分钟 GPU，且结果与现有向量等价，没有必要；
2. 直接搬运能保证两个库**逐条一致**，便于迁移后做一致性核对。

前提：**必须先停后端**
----------------------
Qdrant 嵌入式模式对 `qdrant_storage` 持有独占文件锁，后端运行时本脚本无法读取
（会抛 `RuntimeError: Storage folder ... already accessed`）。因此：

    bash backend/shutdown.sh
    D:/anaconda3/envs/rag_env/python.exe scripts/migrate_to_milvus.py
    bash backend/run.sh --bg

用法
----
    python scripts/migrate_to_milvus.py              # 迁移 kb_medical + kb_legal
    python scripts/migrate_to_milvus.py --verify     # 只做一致性核对，不写入
    python scripts/migrate_to_milvus.py --drop       # 先删 Milvus 侧集合再重建
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core import config                      # noqa: E402
from app.core.logging import get_logger          # noqa: E402
from app.services import milvus_store            # noqa: E402
from app.services.qdrant_store import get_client as qdrant_client  # noqa: E402

log = get_logger("migrate")

TARGETS = [config.COLLECTION_MEDICAL, config.COLLECTION_LEGAL]
SCROLL_BATCH = 256


def _iter_qdrant(collection: str, batch: int = SCROLL_BATCH):
    """滚动读取 Qdrant 的全部点（含 dense + sparse 向量与 payload）。"""
    client = qdrant_client()
    if not client.collection_exists(collection):
        log.warning("Qdrant 集合不存在，跳过: %s", collection)
        return

    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=batch,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        for p in points:
            vec = p.vector or {}
            dense = vec.get("dense")
            sparse = vec.get("sparse")
            if dense is None or sparse is None:
                # 缺向量说明该点异常，跳过而不是写一条残缺记录
                log.warning("点 %s 缺向量，跳过 (dense=%s sparse=%s)",
                            p.id, dense is not None, sparse is not None)
                continue
            yield milvus_store.row_from_qdrant(p.payload or {}, p.id, dense, sparse)

        if offset is None:
            break


def migrate(drop_first: bool = False) -> dict:
    """执行迁移。返回 {collection: 写入条数}。"""
    client = milvus_store.get_client()
    result: dict[str, int] = {}

    for coll in TARGETS:
        if drop_first and client.has_collection(coll):
            log.info("删除 Milvus 侧旧集合: %s", coll)
            client.drop_collection(coll)

        log.info("开始迁移 %s ...", coll)
        t0 = time.time()
        buf: list[dict] = []
        written = 0
        seen_ids: set[int] = set()     # 校验 to_milvus_id 的位掩码没有引入碰撞
        collisions = 0

        for row in _iter_qdrant(coll):
            rid = row["id"]
            if rid in seen_ids:
                collisions += 1
                log.warning("主键碰撞！id=%s（掩码后重复）", rid)
            seen_ids.add(rid)

            buf.append(row)
            if len(buf) >= 500:
                written += milvus_store.upsert_rows(coll, buf)
                buf.clear()
                log.info("  %s 已写入 %d 条 ...", coll, written)
        if buf:
            written += milvus_store.upsert_rows(coll, buf)

        if collisions:
            log.error("%s 出现 %d 次主键碰撞 —— 位掩码方案在此数据量下不可用",
                      coll, collisions)
        else:
            log.info("%s 主键校验通过：%d 条唯一 ID，无碰撞", coll, len(seen_ids))

        milvus_store.flush(coll)
        result[coll] = written
        log.info("%s 迁移完成：%d 条，耗时 %.1fs",
                 coll, written, time.time() - t0)

    return result


def verify() -> bool:
    """一致性核对：比对两库的条数。"""
    client = milvus_store.get_client()
    qc = qdrant_client()
    ok = True

    print(f"{'collection':<14} {'Qdrant':>10} {'Milvus':>10}  结果")
    print("-" * 46)
    for coll in TARGETS:
        q_cnt = -1
        if qc.collection_exists(coll):
            q_cnt = qc.count(collection_name=coll, exact=True).count

        # ⚠️ 必须用 count_rows（count(*) 聚合），不能用 get_collection_stats ——
        #    后者删过数据后是陈旧值，会让核对误报不一致。见 milvus_store.count_rows。
        m_cnt = milvus_store.count_rows(coll)

        same = q_cnt == m_cnt
        ok = ok and same
        print(f"{coll:<14} {q_cnt:>10} {m_cnt:>10}  {'✓ 一致' if same else '✗ 不一致'}")

    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description="Qdrant -> Milvus 向量迁移")
    ap.add_argument("--verify", action="store_true", help="只核对条数，不写入")
    ap.add_argument("--drop", action="store_true",
                    help="先删除 Milvus 侧同名集合再重建")
    args = ap.parse_args()

    if not milvus_store.health()["ok"]:
        print("✗ Milvus 不可用，请先启动：bash tools/demo/up.sh milvus")
        sys.exit(1)

    if args.verify:
        sys.exit(0 if verify() else 2)

    counts = migrate(drop_first=args.drop)
    print()
    for coll, n in counts.items():
        print(f"  {coll}: {n} 条")
    print()
    print("=== 一致性核对 ===")
    sys.exit(0 if verify() else 2)


if __name__ == "__main__":
    main()
