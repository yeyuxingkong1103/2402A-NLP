# -*- coding: utf-8 -*-
"""构建 MySQL 法条结构化索引，支撑「元数据路召回」。

    .venv/bin/python -m scripts.build_law_index

索引表 `law_index` 存 (法律名, 归一化条文号) -> 来源文件。
用户问「刑法第一百三十三条」时，先在这里精确命中，再去 Milvus 取原文，
不必指望向量相似度把对的那一条排上来。

法条数据更新后重新执行本脚本即可。
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings
from app.core.metadata_service import MetadataService, normalize_article
from app.db import mysql_conn
from app.config.logging_conf import setup_logging
import logging

logger = logging.getLogger(__name__)

setup_logging()


def load_statute_records():
    """读入法条数据集（带 law / article 元数据的那些记录）。"""
    role_dir = os.path.join(settings.DATA_DIR, "lawyer")
    records = []
    for fn in sorted(os.listdir(role_dir)):
        if not fn.endswith(".jsonl"):
            continue
        path = os.path.join(role_dir, fn)
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (rec.get("meta") or {}).get("law"):
                    records.append(rec)
    return records


def self_check(db):
    """抽几条验证索引可用。"""
    from sqlalchemy import select
    from app.models.tables import LawIndex

    print("\n归一化自检:")
    for raw in ["第133条", "第一百三十三条", "第1079条", "第一千零七十九条",
                "第82条", "第一百三十三条之一"]:
        print("  %-16s -> %s" % (raw, normalize_article(raw)))

    print("\n索引抽样:")
    rows = db.execute(select(LawIndex).limit(5)).scalars().all()
    for r in rows:
        print("  %-28s %-16s %s" % (r.law_name[:28], r.article, r.chapter[:24]))

    total = db.query(LawIndex).count()
    laws = db.execute(
        select(LawIndex.law_name).distinct()).scalars().all()
    print("\n索引合计 %d 条，覆盖 %d 部法规" % (total, len(laws)))


def main():
    logger.info("目标库: %s:%s/%s", settings.MYSQL_HOST, settings.MYSQL_PORT,
                settings.MYSQL_DATABASE)

    # 先保证表存在（law_index 是本项目新增的表）
    mysql_conn.init_database()

    records = load_statute_records()
    if not records:
        logger.error("未找到带 law/article 元数据的法条记录，请先运行 "
                     "scripts/fetch_statutes.py")
        return 1
    logger.info("读入法条记录 %d 条", len(records))

    with mysql_conn.session_scope() as db:
        n = MetadataService.rebuild_index(db, records)
        logger.info("写入索引 %d 条", n)
        self_check(db)

    return 0


if __name__ == "__main__":
    sys.exit(main())
