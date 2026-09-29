"""批次 37：创建 chunk_summaries 表（幂等，只建表不动数据）。

背景：批次 37 给 Milvus collection 增加 summary 字段（条文一句话摘要），
摘要的 MySQL 单一事实来源是新表 chunk_summaries（app/db/document_models.py:ChunkSummary）。
vector_search._fetch_chunks 已外连接该表；表不存在时真库链路（如
test_retrieval_factory_smoke）会报 Table 'legal_rag.chunk_summaries' doesn't exist。

做法：直接用 ORM 模型元数据 create（checkfirst=True → 幂等），DDL 与模型保证一致。
不做任何数据迁移（新表初始为空，由 app/cli/summarize_chunks.py 批量生成摘要）。

用法（项目根）：
    python scripts/migrations/migrate_chunk_summaries.py            # 预览
    python scripts/migrations/migrate_chunk_summaries.py --apply    # 实际建表

风险：仅 CREATE TABLE（若不存在），不 ALTER、不 DROP、不写数据。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# scripts/migrations → scripts → rag
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

# 沙箱代理会劫持本机连接，剔除
for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
    os.environ.pop(key, None)

# 读项目根 .env（连接参数零口令进源码；与 backfill_law_dates.py 同口径）
env_file = PROJECT_ROOT / ".env"
if env_file.exists():
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())

from sqlalchemy import create_engine, inspect  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.db.document_models import ChunkSummary  # noqa: E402


def build_database_url() -> str:
    """与 chat_runtime.build_default_session_factory 同口径构造连接串（含密码转义）。"""
    from urllib.parse import quote_plus

    return (
        f"mysql+pymysql://{quote_plus(settings.mysql_user)}:{quote_plus(settings.mysql_password)}"
        f"@{settings.mysql_host}:{settings.mysql_port}/{settings.mysql_database}?charset=utf8mb4"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="实际建表（默认只预览）")
    args = parser.parse_args()

    engine = create_engine(build_database_url())
    inspector = inspect(engine)
    exists = inspector.has_table(ChunkSummary.__tablename__)

    print(f"目标库：{engine.url.database}")
    print(f"表 {ChunkSummary.__tablename__} 已存在：{exists}")
    if exists:
        print("无需执行（幂等）。")
        return 0

    print("将执行：CREATE TABLE chunk_summaries（列：chunk_key PK / summary / model / created_at / updated_at）")
    if not args.apply:
        print("预览模式，未写库。加 --apply 实际建表。")
        return 0

    # checkfirst=True：并发/重复执行安全
    ChunkSummary.__table__.create(engine, checkfirst=True)
    print("建表完成。核对：", inspect(engine).has_table(ChunkSummary.__tablename__))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
