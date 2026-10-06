#!/usr/bin/env python
"""一键灌入演示数据：角色预设 + 各角色知识库 + 示例用户。

用法：
    python scripts/seed.py            # 灌入全部预设角色、知识库与示例用户（幂等，可重复执行）
    python scripts/seed.py --users    # 仅灌入示例用户
    python scripts/seed.py --reset    # 先清空角色/文档/用户登记与 Milvus 集合，再重新灌入
    python scripts/seed.py --dry-run  # 只打印将要执行的动作，不写入

说明：
    知识库按「角色 -> 目录/文件」清单逐个入库；重复执行时按 source 先删后插，保证幂等。
    短期记忆为进程内内存，无法持久化，故「会话数据」这里以关系库 users 表（示例用户）落地。

前置条件：
    Milvus、关系库、Ollama（向量化）都要可达；**跑之前先停掉 web 服务**，Milvus Lite
    是单进程独占，服务开着会 DataDirLockedError。

副作用：
    默认只增不删，可以反复跑。但 --reset 是**破坏性且不可逆**的：drop_collection 删掉
    整个 Milvus 集合（不是只删演示文档），再清空 roles/documents/users 三张表——
    只该在演示/开发库上跑。

    --dry-run 只跳过写数据（不 upsert 角色/文档/用户、不入库），**不是零写入**：
    连库那一步照跑，SQLStore.connect 仍会 create_all/跑迁移（DDL）。所以它可用来
    确认动作清单，但不能当成"绝对不碰数据库"的保险。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import settings  # noqa: E402
from app.core.document import Chunker, DocumentParser  # noqa: E402
from app.core.document.ingest import build_chunks, store_chunks  # noqa: E402
from app.core.embedding import EmbeddingClient  # noqa: E402
from app.core.logging_config import get_logger, setup_logging  # noqa: E402
from app.core.prompt.role_presets import ROLE_PRESETS  # noqa: E402
from app.core.store.milvus_store import MilvusStore  # noqa: E402
from app.core.store.sql_store import Document, Role, SQLStore, User  # noqa: E402

log = get_logger("seed")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 角色 -> 语料目录清单（相对项目根）。目录会递归扫描，文件直接入库。
# 用相对路径而非绝对路径：source 是按路径登记的，绝对路径一换机器/换盘符就变成
# 另一份文档，重跑 seed 会重复灌而不是覆盖。main() 里的 os.chdir 就是为它兜底。
#
# 指向 data/corpus/ 而不是手写的 data/sample/：语料来自公开数据集，由
# scripts/fetch_datasets.py 下载转换而来（来源、许可、体积见 data/corpus/manifest.json）。
# **跑 seed 之前要先跑 fetch_datasets.py**，否则这里两个目录都是空的——seed 只打一条
# 「跳过不存在的路径」的 warning 就过去了，角色最后是零召回但启动日志看不出异常。
SEED_KNOWLEDGE: dict[str, list[str]] = {
    "psychologist": ["data/corpus/psychologist"],
    "lawyer": ["data/corpus/lawyer"],
}

# 多个示例用户而不是一个：验证记忆按「会话+角色」隔离时，需要不同用户/会话并排跑，
# 单用户下"串了上下文"和"没串"看起来一样。
SAMPLE_USERS = ["alice", "bob", "carol", "demo_user"]


def _dedup_document(sql: SQLStore, role_id: str, source: str) -> None:
    """删除关系库里同 (role_id, source) 的旧登记（实现已挪到 SQLStore.dedup_document，
    那样 scripts/ingest.py 的 --re-ingest 也能用同一份）。"""
    sql.dedup_document(role_id, source)


def reset_all(sql: SQLStore, milvus: MilvusStore) -> None:
    """清空演示数据：Milvus 集合 + 角色/文档/用户登记（供 --reset 重建）。

    破坏性且不可逆，没有二次确认也没有备份。三张表一次事务提交：中途失败就是
    「Milvus 已空、表还有」的半清状态，重跑 --reset 即可收敛（幂等）。
    """
    milvus.drop_collection()
    with sql.Session() as s:
        s.query(Document).delete()
        s.query(Role).delete()
        s.query(User).delete()
        s.commit()
    log.info("已清空 Milvus 集合与角色/文档/用户登记")


def ingest_path(
    parser: DocumentParser,
    chunker: Chunker,
    embedding: EmbeddingClient,
    milvus: MilvusStore,
    sql: SQLStore,
    role_id: str,
    path: Path,
) -> int:
    """把单个文件或目录（递归）里的文档入库到指定角色，返回 chunk 总数。

    与 scripts/ingest.py 的区别：这里**无条件**先按 source 删旧向量（不提供「已入库
    就跳过」），因为 seed 的语义是「把演示数据刷成清单里的样子」，跑几次结果都一样；
    也不做摘要与低质量过滤（见 build_chunks 调用处的 dedup=False 说明）。
    """
    if path.is_file():
        files = [path]
    else:
        files = sorted(p for p in path.rglob("*") if p.suffix.lower() in parser.SUPPORTED_EXTS)

    total = 0
    for f in files:
        try:
            doc = parser.parse_file(f)
        except ValueError as exc:
            log.warning("跳过 %s: %s", f, exc)
            continue
        texts = chunker.chunk(doc["text"])
        if not texts:
            log.warning("跳过 %s: 无有效文本", f)
            continue

        chunks = build_chunks(
            texts,
            source=doc["source"],
            title=doc["title"],
            embedding=embedding,
            dedup=False,  # seed 用的是固定演示文档，同批内没有重复 chunk（保持既有行为）
        )

        deleted = milvus.delete_by_source(role_id, doc["source"])
        if deleted:
            log.info("重复入库，先删除旧数据 %d 条（source=%s）", deleted, doc["source"])
        # 关系库里的同 source 旧登记也要清掉。必须在 store_chunks 之前：那里面会
        # register_document，放在之后就把刚登记的记录自己删了。
        _dedup_document(sql, role_id, doc["source"])

        count = store_chunks(
            chunks,
            role_id=role_id,
            source=doc["source"],
            title=doc["title"],
            milvus=milvus,
            sql=sql,
        )
        total += count
        log.info("入库 role=%s source=%s chunks=%d", role_id, f.name, count)

    return total


def main() -> None:
    # 切到项目根目录，保证 source 路径与 ingest.py 一致（相对路径），便于重复入库去重
    os.chdir(PROJECT_ROOT)
    setup_logging(settings)
    ap = argparse.ArgumentParser(description="RAG 演示数据一键灌入")
    ap.add_argument("--users", action="store_true", help="仅灌入示例用户")
    ap.add_argument("--reset", action="store_true", help="先清空角色/文档/用户登记与 Milvus 集合，再重新灌入")
    ap.add_argument("--dry-run", action="store_true", help="只打印动作，不实际写入")
    args = ap.parse_args()

    sql = SQLStore(settings)
    sql.connect()
    milvus = MilvusStore(settings)
    parser = DocumentParser()
    # 固定 500/50（= scripts/ingest.py 与 /knowledge/upload 的默认值）：演示数据的
    # 分块要可复现，跟着 flag 或 .env 变会让两次 seed 的 chunk 数对不上
    chunker = Chunker(chunk_size=500, overlap=50)
    embedding = EmbeddingClient(settings)

    # 0) 可选：清空后重建
    if args.reset:
        if args.dry_run:
            log.info("dry-run 将清空 Milvus 集合 + 角色/文档/用户登记")
        else:
            reset_all(sql, milvus)

    # ① 角色预设
    for role_id, preset in ROLE_PRESETS.items():
        if args.users:
            continue
        action = "upsert 角色"
        if not args.dry_run:
            sql.upsert_role(
                role_id,
                preset["name"],
                preset["description"],
                preset["system_prompt"],
                preset.get("avatar", ""),
            )
        log.info("%s: %s（%s）", action, preset["name"], role_id)

    # ② 各角色知识库
    if not args.users:
        grand_total = 0
        for role_id, rel_paths in SEED_KNOWLEDGE.items():
            if role_id not in ROLE_PRESETS:
                log.warning("跳过未知角色 %s（未在 role_presets 中定义）", role_id)
                continue
            for rel in rel_paths:
                path = Path(rel)
                if not path.exists():
                    log.warning("跳过不存在的路径: %s", path)
                    continue
                if args.dry_run:
                    log.info("dry-run 将入库 role=%s path=%s", role_id, path)
                    continue
                grand_total += ingest_path(parser, chunker, embedding, milvus, sql, role_id, path)

        log.info("知识库入库完成，共 %d chunks", grand_total)

    # ③ 示例用户
    for username in SAMPLE_USERS:
        action = "get_or_create 用户"
        if not args.dry_run:
            sql.get_or_create_user(username)
        log.info("%s: %s", action, username)

    log.info("灌入完成")


if __name__ == "__main__":
    main()
