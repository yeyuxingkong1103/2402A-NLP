# scripts/knowledge_base.py
"""知识库运维：构建、清理、统计。

用法:
    python -m scripts.knowledge_base                 # 全量重建（删集合后重灌）
    python -m scripts.knowledge_base lawyer          # 只重建指定角色
    python -m scripts.knowledge_base --keep          # 保留旧数据，追加写入
    python -m scripts.knowledge_base --clean         # 只清理遗留集合
    python -m scripts.knowledge_base --stats         # 只看统计
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings
from app.core.ingest_service import get_ingest_service
from app.db.milvus_conn import get_milvus
from app.config.logging_conf import setup_logging
import logging

logger = logging.getLogger(__name__)

setup_logging()

KEEP = {settings.MILVUS_COLLECTION, settings.MILVUS_MEMORY_COLLECTION}


# ---------------- 清理 ----------------
def clean(assume_yes: bool) -> None:
    """删除项目迭代过程中产生的实验性集合，只保留配置中在用的。"""
    milvus = get_milvus()
    existing = milvus.list_collections()

    logger.info("当前集合 %s 个:", len(existing))
    for name in existing:
        try:
            n = milvus.count(name)
        except Exception:
            n = -1
        logger.info("  [%s] %-28s %8s 条",
                    "保留" if name in KEEP else "删除", name, n)

    doomed = [c for c in existing if c not in KEEP]
    if not doomed:
        logger.info("没有需要清理的集合")
        return
    if not assume_yes:
        if input("\n确认删除以上 %d 个集合？输入 yes 继续: " % len(doomed)).strip().lower() != "yes":
            logger.info("已取消")
            return
    for name in doomed:
        milvus.drop_collection(name)
    logger.info("清理完成，剩余集合: %s", milvus.list_collections())


# ---------------- 统计 ----------------
def stats() -> None:
    milvus = get_milvus()
    collection = settings.MILVUS_COLLECTION
    total = milvus.count(collection)
    logger.info("集合 %s 共 %s 条", collection, total)

    try:
        rows = milvus.query_all(collection, 'role_id != ""',
                                output_fields=["role_id", "source"])
        per_role, per_src = {}, {}
        for r in rows:
            per_role[r.get("role_id", "?")] = per_role.get(r.get("role_id", "?"), 0) + 1
            key = (r.get("role_id", "?"), r.get("source", "?"))
            per_src[key] = per_src.get(key, 0) + 1
        logger.info("-" * 46)
        for role, n in sorted(per_role.items()):
            logger.info("  %-20s %6s 条", role, n)
            for (r, s), c in sorted(per_src.items()):
                if r == role:
                    logger.info("      %-34s %6s", s, c)
    except Exception as e:
        logger.warning("分角色统计失败: %s", e)


# ---------------- 构建 ----------------
def build(roles, keep: bool) -> None:
    milvus = get_milvus()
    collection = settings.MILVUS_COLLECTION
    ingest = get_ingest_service()

    logger.info("Milvus: %s | 集合: %s", settings.MILVUS_URI, collection)

    if not keep:
        logger.warning("重建模式：删除集合 %s 及其全部数据", collection)
        milvus.drop_collection(collection)
    milvus.ensure_knowledge_collection(collection)

    t0 = time.time()
    summary = []
    for role in roles:
        role_dir = os.path.join(settings.DATA_DIR, role)
        if not os.path.isdir(role_dir):
            logger.warning("跳过 %s：目录不存在 %s", role, role_dir)
            continue
        logger.info("-" * 50)
        logger.info("处理角色 [%s]", role)
        try:
            result = ingest.ingest_role_dir(role, role_dir)
            summary.append(result)
            for d in result["details"]:
                logger.info("  %-34s %6s 条 -> %6s 切片",
                            d["source"], d["records"], d["chunks"])
        except Exception as e:
            logger.exception("角色 %s 入库失败: %s", role, e)

    logger.info("=" * 60)
    total = 0
    for s in summary:
        logger.info("[%s] %s 个文件, %s 条记录, %s 个切片",
                    s["role_id"], s["files"], s["records"], s["chunks"])
        total += s["chunks"]
    logger.info("集合 %s 当前共 %s 条", collection, milvus.count(collection))
    logger.info("合计写入 %s 个切片，耗时 %.1fs", total, time.time() - t0)


def main():
    argv = sys.argv[1:]
    if "--clean" in argv:
        clean(assume_yes="-y" in argv or "--yes" in argv)
        return
    if "--stats" in argv:
        stats()
        return
    roles = [a for a in argv if not a.startswith("-")] or settings.SUPPORTED_ROLES
    build(roles, keep="--keep" in argv)
    stats()


if __name__ == "__main__":
    main()
