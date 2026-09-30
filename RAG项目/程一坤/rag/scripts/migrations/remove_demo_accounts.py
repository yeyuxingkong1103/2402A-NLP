"""幂等清理指定演示账号及其关联会话、短期记忆和长期记忆。

默认只执行 dry-run，必须显式传入 --execute 才会删除数据。
账号匹配使用完整邮箱等值比较，禁止通配符，避免误删其他用户。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from sqlalchemy import delete, func, select

# 让脚本从项目根目录执行时可以导入 backend/app 配置与模型。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIRECTORY = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIRECTORY))

from app.auth.session_store import SessionStore  # noqa: E402
from app.core.config import settings  # noqa: E402
from app.db.chat_models import ChatMessage, ChatSession  # noqa: E402
from app.db.engine import create_database_engine, create_session_factory  # noqa: E402
from app.db.sql_models import User  # noqa: E402
from app.db.redis_client import create_redis_client  # noqa: E402

TARGET_EMAILS = (
    "legaladmin@qq.com",
    "legaluser@qq.com",
    "928421739@qq.com",
)
LONG_TERM_MEMORY_COLLECTION = settings.milvus_long_term_collection_name


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数，默认 dry-run 保护生产数据。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只统计并打印待删除数量（默认行为，显式写出便于部署审计）",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="确认执行删除；不传时只统计并打印待删除数量",
    )
    return parser


def count_redis_keys(redis_client: Any, user_ids: Iterable[str]) -> int:
    """统计目标用户的会话集合和短期记忆 key 数量，不读取其他用户会话。"""
    total = 0
    for user_id in user_ids:
        total += int(redis_client.exists(f"user_sessions:{user_id}"))
        total += sum(
            1
            for _ in redis_client.scan_iter(match=f"short_memory:{user_id}:*")
        )
    return total


def collect_milvus_rows(user_ids: list[str]) -> int:
    """统计长期记忆 collection 中属于目标用户的记录数；不可达时明确失败。"""
    if not user_ids:
        return 0
    from pymilvus import MilvusClient

    client = MilvusClient(
        uri=f"http://{settings.milvus_host}:{settings.milvus_port}"
    )
    if not client.has_collection(LONG_TERM_MEMORY_COLLECTION):
        return 0
    expression = "user_id in [" + ",".join(json.dumps(value) for value in user_ids) + "]"
    rows = client.query(
        collection_name=LONG_TERM_MEMORY_COLLECTION,
        filter=expression,
        output_fields=["memory_id"],
        limit=16384,
    )
    return len(rows)


def delete_milvus_rows(user_ids: list[str]) -> int:
    """删除长期记忆 collection 中属于目标用户的记录。"""
    if not user_ids:
        return 0
    from pymilvus import MilvusClient

    client = MilvusClient(
        uri=f"http://{settings.milvus_host}:{settings.milvus_port}"
    )
    if not client.has_collection(LONG_TERM_MEMORY_COLLECTION):
        return 0
    expression = "user_id in [" + ",".join(json.dumps(value) for value in user_ids) + "]"
    result = client.delete(
        collection_name=LONG_TERM_MEMORY_COLLECTION,
        filter=expression,
    )
    return int(result.get("delete_count", 0)) if isinstance(result, dict) else 0


def main() -> int:
    """按用户、MySQL 会话、Redis 记忆和 Milvus 记忆顺序清理。"""
    arguments = build_parser().parse_args()
    engine = create_database_engine(
        f"mysql+pymysql://{settings.mysql_user}:{settings.mysql_password}"
        f"@{settings.mysql_host}:{settings.mysql_port}/{settings.mysql_database}"
    )
    session_factory = create_session_factory(engine)
    redis_client = create_redis_client(settings.redis_url)

    with session_factory() as session:
        users = list(
            session.scalars(
                select(User).where(User.email.in_(TARGET_EMAILS))
            ).all()
        )
        user_ids = [user.user_key for user in users]
        session_count = int(
            session.scalar(
                select(func.count(ChatSession.id)).where(ChatSession.user_id.in_(user_ids))
            )
            or 0
        ) if user_ids else 0
        message_count = int(
            session.scalar(
                select(func.count(ChatMessage.id)).join(
                    ChatSession, ChatMessage.session_id == ChatSession.id
                ).where(ChatSession.user_id.in_(user_ids))
            )
            or 0
        ) if user_ids else 0

    redis_count = count_redis_keys(redis_client, user_ids)
    milvus_count = collect_milvus_rows(user_ids)
    summary = {
        "matched_users": len(users),
        "mysql_sessions": session_count,
        "mysql_messages": message_count,
        "redis_keys": redis_count,
        "milvus_memories": milvus_count,
        "emails": list(TARGET_EMAILS),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if not arguments.execute:
        print("dry-run：未删除任何数据。需要执行删除时追加 --execute。")
        return 0

    session_store = SessionStore(redis_client, settings.session_ttl_seconds)
    for user_id in user_ids:
        session_store.revoke_all_sessions(user_id)
        for key in redis_client.scan_iter(match=f"short_memory:{user_id}:*"):
            redis_client.delete(key)

    deleted_milvus = delete_milvus_rows(user_ids)
    with session_factory() as session:
        session.execute(
            delete(ChatMessage).where(
                ChatMessage.session_id.in_(
                    select(ChatSession.id).where(ChatSession.user_id.in_(user_ids))
                )
            )
        )
        session.execute(delete(ChatSession).where(ChatSession.user_id.in_(user_ids)))
        session.execute(delete(User).where(User.email.in_(TARGET_EMAILS)))
        session.commit()

    print(json.dumps({"deleted_milvus_memories": deleted_milvus}, ensure_ascii=False))
    print("execute：指定演示账号及关联数据已清理。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
