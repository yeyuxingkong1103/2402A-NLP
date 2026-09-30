"""批次 39：4 篇页面清洗重切重导后的审核/效力状态修复。

背景：doc2 / doc7 / doc8 / doc11 的原始页面带站点页脚样板（责任编辑 / 总机 /
版权所有 / 京ICP备 / 【打印】/ 主办单位…），批次 39 修好清洗规则后单独
`import_mysql --replace-existing` 重导。重导走例行导入路径，会带来两处
**非预期副作用**（都属于"人工成果/审核结论"，不能让例行操作吞掉）：

1. `document_versions.version_status` 被重置为 pending_review —— 这 4 篇此前
   已按「切块器 v3 重建存量版本」口径人工批准；重导只是清洗重切（条文数与
   正文实质未变，见 reports/b39_site_footer_cleanup.md），沿用原审核结论。
2. `law_versions` 行被重建 —— 效力状态与公布/施行日期是核验成果
   （批次 6/26 回填），抽取器对这些页面抽不到，故须显式回填。

快照来源：2026-09-23 重导前的库内状态（本文件 `SNAPSHOT` 常量逐字段记录，
便于审计）。脚本只写这些字段，不碰正文/分块/向量。

用法：
    python restore_reingest_status.py            # 预览（不写库）
    python restore_reingest_status.py --apply    # 实际写库
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
    os.environ.pop(key, None)

from urllib.parse import quote_plus  # noqa: E402

from sqlalchemy import create_engine, text  # noqa: E402

from app.core.config import settings  # noqa: E402

# 重导前快照：document_id → {版本审核结论, law_versions 效力核验}
SNAPSHOT: dict[int, dict] = {
    2: {
        "version_key": "ver-906e4d174cb1f81c2ac91f68",
        "laws": [{
            "law_id": 76, "status": "现行有效",
            "effective_date": date(2025, 9, 1),
            "promulgation_date": date(2025, 7, 31),
            "expiration_date": None,
        }],
    },
    7: {
        "version_key": "ver-2439d8d497047b57eab5ead0",
        "laws": [{
            "law_id": 74, "status": "现行有效",
            "effective_date": date(2021, 1, 1),
            "promulgation_date": date(2020, 12, 30),
            "expiration_date": None,
        }],
    },
    8: {
        "version_key": "ver-2cd4c05d144c77e195afc0d6",
        "laws": [{
            "law_id": 77, "status": "现行有效",
            "effective_date": date(2008, 5, 1),
            "promulgation_date": date(2007, 12, 29),
            "expiration_date": None,
        }],
    },
    11: {
        "version_key": "ver-2f8798396670c9e94fea5681",
        "laws": [{
            "law_id": 80, "status": "现行有效",
            "effective_date": date(1995, 1, 1),
            "promulgation_date": date(1994, 12, 6),
            "expiration_date": None,
        }],
    },
}

REVIEWED_BY = "chunker-v3-重建批准"
REVIEWED_AT = datetime(2026, 9, 20, 10, 56, 6)
REVIEW_NOTE = (
    "切块器v3重建存量版本：与阶段6迁移同一批准口径（11部法规已审定在用），"
    "非静默批量改状态；批次39 清洗重切重导后沿用"
)


def build_url() -> str:
    return (
        f"mysql+pymysql://{quote_plus(settings.mysql_user)}:"
        f"{quote_plus(settings.mysql_password)}@{settings.mysql_host}:"
        f"{settings.mysql_port}/{settings.mysql_database}?charset=utf8mb4"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="批次39：重导文档的审核与效力状态回填")
    parser.add_argument("--apply", action="store_true", help="实际写库（默认只预览）")
    args = parser.parse_args(argv)

    engine = create_engine(build_url())
    with engine.begin() as connection:
        for doc_id, snap in SNAPSHOT.items():
            version = connection.execute(
                text("SELECT current_version_id FROM documents WHERE id=:d"),
                {"d": doc_id},
            ).scalar()
            if version is None:
                print(f"doc{doc_id}: 未找到当前版本，跳过")
                continue
            current_key = connection.execute(
                text("SELECT version_key, version_status FROM document_versions WHERE id=:v"),
                {"v": version},
            ).first()
            laws = connection.execute(
                text(
                    "SELECT law_id, status, effective_date, promulgation_date "
                    "FROM law_versions WHERE document_version_id=:v"
                ),
                {"v": version},
            ).all()
            print(
                f"doc{doc_id}: version={version} key={current_key[0]} "
                f"version_status={current_key[1]} law_versions={laws}"
            )
            if current_key[0] != snap["version_key"]:
                print(f"  ⚠️ version_key 与快照不一致（快照 {snap['version_key']}）")

            if not args.apply:
                continue

            connection.execute(
                text(
                    """
                    UPDATE document_versions
                       SET version_status='approved', reviewed_by=:by,
                           reviewed_at=:at, review_note=:note
                     WHERE id=:v
                    """
                ),
                {
                    "by": REVIEWED_BY,
                    "at": REVIEWED_AT,
                    "note": REVIEW_NOTE,
                    "v": version,
                },
            )
            for law in snap["laws"]:
                connection.execute(
                    text(
                        """
                        UPDATE law_versions
                           SET status=:status, effective_date=:eff,
                               promulgation_date=:prom, expiration_date=:exp
                         WHERE document_version_id=:v AND law_id=:law
                        """
                    ),
                    {
                        "status": law["status"],
                        "eff": law["effective_date"],
                        "prom": law["promulgation_date"],
                        "exp": law["expiration_date"],
                        "v": version,
                        "law": law["law_id"],
                    },
                )

    if not args.apply:
        print("预览模式，未写库（加 --apply 实际执行）")
        return 0

    with engine.connect() as connection:
        print("\n=== 修复后 ===")
        for doc_id in SNAPSHOT:
            row = connection.execute(
                text(
                    """
                    SELECT dv.version_status, dv.reviewed_by, lv.law_id, lv.status,
                           lv.effective_date, lv.promulgation_date
                      FROM document_versions dv
                      LEFT JOIN law_versions lv ON lv.document_version_id=dv.id
                     WHERE dv.document_id=:d
                    """
                ),
                {"d": doc_id},
            ).all()
            for r in row:
                print(f"  doc{doc_id}: {r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
