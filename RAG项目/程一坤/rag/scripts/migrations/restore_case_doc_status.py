"""批次 38：document id=1（最高法劳动争议典型案例）清洗重切后的数据修复。

背景：该文档的原始页面是整页抓取（页头面包屑 + 页脚版权/ICP/校验串混入正文），
批次 38 修复了 cleaner/parser 并单独 replace 重导。重导走的是例行导入路径，
会带来两处**非预期的副作用**，需要本脚本显式修复（都属于"人工成果/审核结论"，
不能让例行操作吞掉）：

1. `document_versions.version_status` 被重置为 pending_review —— 该文档此前
   已人工审核通过；重导只是清洗重切（正文实质未变），沿用原审核结论，恢复 approved
   并写入 reviewed_by/reviewed_at/review_note 留痕。
2. `law_versions.status` 被重建为 NULL —— 该文档是案例材料，其 status='不适用'
   属人工核验成果（批次 26-B 回填、证据见 reports/metadata_verification_6laws.md）。
   重导后 law_versions 行重建（replace 会删除旧行），抽取器对本页抽不到 status，
   故须显式回填。

本脚本严格只写这两处，不碰正文/分块/向量。

用法：
    python restore_case_doc_status.py            # 预览（不写库）
    python restore_case_doc_status.py --apply    # 实际写库
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
    os.environ.pop(key, None)

from urllib.parse import quote_plus  # noqa: E402

from sqlalchemy import create_engine, text  # noqa: E402

from app.core.config import settings  # noqa: E402

CASE_DOCUMENT_ID = 1
CASE_STATUS = "不适用"
REVIEW_NOTE = "批次38 清洗重切重导（去除页脚版权/ICP样板），沿用原审核结论"


def build_url() -> str:
    return (
        f"mysql+pymysql://{quote_plus(settings.mysql_user)}:"
        f"{quote_plus(settings.mysql_password)}@{settings.mysql_host}:"
        f"{settings.mysql_port}/{settings.mysql_database}?charset=utf8mb4"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="批次38：案例文档审核状态与效力状态修复")
    parser.add_argument("--apply", action="store_true", help="实际写库（默认只预览）")
    args = parser.parse_args(argv)

    engine = create_engine(build_url())
    with engine.begin() as connection:
        version = connection.execute(
            text(
                "SELECT current_version_id FROM documents WHERE id=:d"
            ),
            {"d": CASE_DOCUMENT_ID},
        ).scalar()
        if version is None:
            print("未找到目标文档，退出")
            return 1

        before = connection.execute(
            text(
                "SELECT version_status, processing_status FROM document_versions WHERE id=:v"
            ),
            {"v": version},
        ).first()
        law_before = connection.execute(
            text("SELECT id, status FROM law_versions WHERE document_version_id=:v"),
            {"v": version},
        ).all()
        print(f"版本 {version} 修复前：{before}；law_versions={law_before}")

        if not args.apply:
            print("预览模式，未写库（加 --apply 实际执行）")
            return 0

        now = datetime.now()
        connection.execute(
            text(
                """
                UPDATE document_versions
                   SET version_status='approved', reviewed_by=:by,
                       reviewed_at=:at, review_note=:note
                 WHERE id=:v
                """
            ),
            {"by": "batch38-fix", "at": now, "note": REVIEW_NOTE, "v": version},
        )
        connection.execute(
            text(
                """
                UPDATE law_versions
                   SET status=:status
                 WHERE document_version_id=:v AND (status IS NULL OR status='')
                """
            ),
            {"status": CASE_STATUS, "v": version},
        )

    with engine.connect() as connection:
        after = connection.execute(
            text(
                """
                SELECT dv.version_status, dv.reviewed_by, lv.status
                  FROM document_versions dv
                  LEFT JOIN law_versions lv ON lv.document_version_id=dv.id
                 WHERE dv.id=:v
                """
            ),
            {"v": version},
        ).first()
    print(f"修复后：version_status={after[0]}, reviewed_by={after[1]}, law status={after[2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
