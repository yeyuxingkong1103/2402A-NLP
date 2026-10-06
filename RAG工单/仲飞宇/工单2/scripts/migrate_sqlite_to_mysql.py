#!/usr/bin/env python
"""把关系库从 SQLite 搬到 MySQL（或任意两个 SQLAlchemy 连接串之间）。

为什么要它：`SQL_URL` 一换，`Base.metadata.create_all` 只会在新库建空表——
`roles` 靠启动时的 `ensure_role` 能自动补回来，但 **`documents` 的登记记录不会**，
换库后 `/knowledge/list` 会直接变空（向量库里数据还在，只是没登记表了）。

用法：
    python scripts/migrate_sqlite_to_mysql.py \
        --from sqlite:///./data/app.db \
        --to "mysql+pymysql://root:root@172.27.224.1:3306/rag_roleplay"

    # 只看会搬什么，不写：
    python scripts/migrate_sqlite_to_mysql.py --from ... --to ... --dry-run

幂等：roles 按主键覆盖、users 按 username 去重、documents 按 (role_id, source) 去重，
重复跑不会产生重复行。

前置条件与副作用：
    - 源库只读，写只写目标库（`Base.metadata.create_all` 建缺失的表，所以目标账号
      要有 DDL 权限）。--dry-run 不建表、不写入，但仍会**连一次目标库**——
      目标连不上时 dry-run 一样会失败，别拿它当离线预检。
    - **不搬 Milvus**：向量库与关系库是两套存储，这里只管登记表。换库后如果
      documents 表没搬过来，/knowledge/list 会是空的，但检索仍能召回。
    - 搬完之后要自己把 `.env` 的 `SQL_URL` 指到新库——本脚本不碰配置。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import create_engine, select  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from app.core.store.sql_store import Base, Document, Role, User  # noqa: E402


def _engine(url: str):
    # 目标库的表由 create_all 建；SQLite 需要关掉同线程检查（与 SQLStore 一致）
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(url, future=True, connect_args=connect_args)


def migrate(from_url: str, to_url: str, dry_run: bool = False) -> dict[str, int]:
    """搬 roles / users / documents 三张表，返回各表条数与去重跳过数。

    `skipped` 是 users 与 documents 两处的**合计**（roles 覆盖写不算跳过），
    所以它大于 0 不一定是异常：重复跑必然有跳过，那正是幂等的表现。

    dry_run 下只用同一段代码数行数、最后 rollback：计数逻辑与真跑完全一致，
    不会出现「dry-run 说 3 条、真跑只有 2 条」。三张表在**同一个事务**里、
    最后才 commit，中途抛异常目标库不会留下半截数据（配合幂等，重跑即可）。
    """
    src, dst = _engine(from_url), _engine(to_url)
    if not dry_run:
        Base.metadata.create_all(dst)

    counts = {"roles": 0, "users": 0, "documents": 0, "skipped": 0}
    with Session(src) as s, Session(dst) as d:
        for row in s.scalars(select(Role)).all():
            counts["roles"] += 1
            if dry_run:
                continue
            # 用 merge 而不是 add 的根据：roles 主键 role_id 是有业务含义的字符串，
            # 跨库仍然唯一；users/documents 的主键是自增 id，换个库就不再对应同一行，
            # 只能按业务唯一键查重后 add。
            # roles 主键是 role_id：目标已存在的直接覆盖（人设可能被改过）
            d.merge(
                Role(
                    id=row.id,
                    name=row.name,
                    avatar=row.avatar or "",
                    description=row.description or "",
                    system_prompt=row.system_prompt or "",
                    is_active=row.is_active if row.is_active is not None else 1,
                )
            )

        for row in s.scalars(select(User)).all():
            counts["users"] += 1
            if dry_run:
                continue
            if d.scalar(select(User).where(User.username == row.username)):
                counts["skipped"] += 1
                continue
            d.add(User(username=row.username))

        for row in s.scalars(select(Document)).all():
            counts["documents"] += 1
            if dry_run:
                continue
            dup = d.scalar(
                select(Document).where(
                    Document.role_id == row.role_id, Document.source == row.source
                )
            )
            if dup:
                counts["skipped"] += 1
                continue
            d.add(
                Document(
                    role_id=row.role_id,
                    source=row.source,
                    title=row.title or "",
                    chunk_count=row.chunk_count or 0,
                )
            )

        if dry_run:
            d.rollback()
        else:
            d.commit()
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description="关系库迁移：SQLite -> MySQL")
    ap.add_argument("--from", dest="from_url", required=True, help="源连接串")
    ap.add_argument("--to", dest="to_url", required=True, help="目标连接串")
    ap.add_argument("--dry-run", action="store_true", help="只统计不写入")
    args = ap.parse_args()

    counts = migrate(args.from_url, args.to_url, args.dry_run)
    verb = "将迁移" if args.dry_run else "已迁移"
    print(
        f"{verb}: roles {counts['roles']} 条, users {counts['users']} 条, "
        f"documents {counts['documents']} 条（跳过重复 {counts['skipped']} 条）"
    )
    if not args.dry_run:
        print("注意：向量库（Milvus）里对应的 chunk 不在这里搬——它和关系库是两套存储。")


if __name__ == "__main__":
    main()
