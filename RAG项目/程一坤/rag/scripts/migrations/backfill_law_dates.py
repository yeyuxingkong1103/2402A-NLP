"""批次 10 任务 1：法规时效字段回填脚本（一次性，项目根目录，不进 backend/）。

背景：legal_metadata_writer 只在 law_versions 首次创建时写日期，
且导入增量判定在分块未变化时短路，导致抽取规则增强后新抽到的
生效日期无法进入库。本脚本复用同一个抽取器（严禁编造，只写页面
实际抽到的值），把 laws / law_versions 的时效字段刷新一遍。

用法：
    python backfill_law_dates.py            # 预览（不写库）
    python backfill_law_dates.py --apply    # 实际写库

产物：reports/batch10_law_dates.md（日期表 + 待人工补录清单）
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# 本脚本已从项目根移入 scripts/migrations/，故向上三级回到项目根（rag/）
# （父目录链：rag/scripts/migrations → rag/scripts → rag）
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

# 沙箱代理会劫持本机请求，剔除（本脚本不访问网络，稳妥起见仍清掉）
for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
    os.environ.pop(key, None)

# 读 .env
env_file = PROJECT_ROOT / ".env"
if env_file.exists():
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())

from sqlalchemy import create_engine, select  # noqa: E402
from sqlalchemy.orm import Session, sessionmaker  # noqa: E402

from app.db.sql_models import DocumentVersion, Law, LawVersion  # noqa: E402
from app.ingest.law_metadata_extractor import extract_legal_metadata  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="实际写库（默认只预览）")
    args = parser.parse_args()

    database_url = os.environ["DATABASE_URL"]
    engine = create_engine(database_url)
    SessionLocal = sessionmaker(bind=engine)

    raw_root = PROJECT_ROOT / "data" / "labor_law_raw"

    rows: list[dict] = []
    with SessionLocal() as session:  # type: Session
        pairs = (
            session.execute(
                select(LawVersion, Law, DocumentVersion)
                .join(Law, Law.id == LawVersion.law_id)
                .join(DocumentVersion, DocumentVersion.id == LawVersion.document_version_id)
                .order_by(Law.id)
            )
            .all()
        )
        for law_version, law, doc_version in pairs:
            filename = Path(doc_version.raw_file_path).name
            raw_file = raw_root / filename
            if not raw_file.exists():
                rows.append({"law": law.name, "error": f"raw 文件缺失: {filename}"})
                continue
            metadata = extract_legal_metadata(
                law_name=law.name,
                source_url=law.source_url or "",
                html_file_path=raw_file,
            )
            changes: dict[str, tuple] = {}
            for field in ("promulgation_date", "effective_date", "expiration_date", "status"):
                old = getattr(law_version, field)
                new = getattr(metadata, field)
                if old != new:
                    changes[field] = (old, new)
            if args.apply and changes:
                for field, (_old, new) in changes.items():
                    setattr(law_version, field, new)
            rows.append({
                "law": law.name,
                "law_key": law.law_key,
                "source_url": law.source_url or "",
                "changes": changes,
                "missing": metadata.missing_fields(),
            })
        if args.apply:
            session.commit()

    # ---- 控制台输出 ----
    print(f"mode={'APPLY' if args.apply else 'DRY-RUN'}  laws={len(rows)}")
    for row in rows:
        if "error" in row:
            print(f"[错误] {row['law']}: {row['error']}")
            continue
        print(f"\n{row['law']}")
        if not row["changes"]:
            print("  （无变化）")
        for field, (old, new) in row["changes"].items():
            print(f"  {field}: {old} -> {new}")
        if row["missing"]:
            print(f"  仍缺: {'、'.join(row['missing'])}")

    # ---- 报告 ----
    report_dir = PROJECT_ROOT / "reports"
    report_dir.mkdir(exist_ok=True)
    lines = [
        "# 批次 10 任务 1：法规时效字段回填报告",
        "",
        f"- 运行时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 模式：{'写入' if args.apply else '预览'}",
        f"- 抽取原则：只从页面实际内容抽取（与 app/ingest/law_metadata_extractor.py 同一实现），抽不到即留空，严禁编造",
        "",
        "## 一、11 部法规日期表（回填后）",
        "",
        "| 法规 | 公布日期 | 生效日期 | 失效日期 | 效力状态 | 本次变化 |",
        "|---|---|---|---|---|---|",
    ]

    def fmt(value):
        return str(value) if value not in (None, "") else "（待补录）"

    for row in rows:
        if "error" in row:
            lines.append(f"| {row['law']} | 错误：{row['error']} | - | - | - | - |")
            continue
        law_version_fields = {}
        # 回填后的终值 = changes 里的新值，否则为库内原值（无变化）
        for field in ("promulgation_date", "effective_date", "expiration_date", "status"):
            if field in row["changes"]:
                law_version_fields[field] = row["changes"][field][1]
        change_desc = "、".join(
            f"{k} {v[0]}→{v[1]}" for k, v in row["changes"].items()
        ) or "无"
        lines.append(
            f"| {row['law']} | {fmt(law_version_fields.get('promulgation_date', '未变'))} "
            f"| {fmt(law_version_fields.get('effective_date', '未变'))} "
            f"| {fmt(law_version_fields.get('expiration_date', '未变'))} "
            f"| {fmt(law_version_fields.get('status', '未变'))} | {change_desc} |"
        )

    lines += [
        "",
        "## 二、待人工补录清单",
        "",
    ]
    pending = [row for row in rows if "error" not in row and row["missing"]]
    if not pending:
        lines.append("（无）")
    for row in pending:
        lines.append(f"### {row['law']}")
        lines.append(f"- 缺什么：{'、'.join(row['missing'])}")
        lines.append(f"- 页面链接：{row['source_url']}")
        lines.append(
            "- 说明：页面实际内容中没有对应字段的明确文字，脚本拒绝编造，需要人工核实后补录"
        )
        lines.append("")

    (report_dir / "batch10_law_dates.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n报告已写入 reports/batch10_law_dates.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
