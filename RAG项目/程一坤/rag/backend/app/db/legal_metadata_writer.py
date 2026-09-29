"""三层法规元数据表（laws / law_versions / articles）写入。

原 import_service.py 并入部分：负责从已入库数据包提取法规级元数据，
并幂等写入 laws、law_versions、articles 三层表。
"""

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.sql_models import Article, Law, LawVersion
from app.ingest.law_metadata_extractor import extract_legal_metadata
from app.pipeline.package_validator import ValidatedPackage


def _add_legal_metadata(session: Session, package: ValidatedPackage, version_id: int) -> None:
    """写入三层表（laws / law_versions / articles）。

    Args:
        session: 数据库会话
        package: 已校验的数据包
        version_id: 刚创建的 DocumentVersion 的 ID
    """
    from pathlib import Path

    # 1. 提取元数据
    # raw_file_path 格式: "raw/HASH.html"，需要从项目根目录的 labor_law_raw 找到实际文件
    raw_relative_path = package.data.version.raw_file_path
    filename = Path(raw_relative_path).name  # 提取文件名: "HASH.html"

    # 从项目根目录推导 raw 文件的绝对路径
    project_root = Path(__file__).resolve().parents[3]  # backend/app/db/import_service.py -> rag/
    raw_file_path = project_root / "data" / "labor_law_raw" / filename

    if not raw_file_path.exists():
        raise FileNotFoundError(f"Raw file not found: {raw_file_path}")

    metadata = extract_legal_metadata(
        law_name=package.data.document.title,
        source_url=package.data.document.source_url,
        html_file_path=raw_file_path,
    )

    # 2. 生成 law_key（法规唯一标识）
    law_key = _generate_law_key(metadata.law_name)

    # 3. 创建或获取 Law 记录（幂等：按 law_key 去重）
    law = session.scalar(select(Law).where(Law.law_key == law_key))
    if law is None:
        law = Law(
            law_key=law_key,
            name=metadata.law_name,
            # 抽不到就留空（表示待人工补录），不用"待人工确认"这类占位字符串，
            # 也不填 99 这种假等级 —— 占位值会被下游当成真实数据参与排序与过滤
            document_type=metadata.document_type,
            authority_level=metadata.authority_level,
            issuing_authority=metadata.issuing_authority,
            jurisdiction=metadata.jurisdiction,
            source_url=metadata.source_url,
        )
        session.add(law)
        session.flush()

    # 4. 生成 version_key（版本唯一标识）
    version_key = f"{law_key}_v1"  # 当前阶段每部法规只有一个版本

    # 5. 创建 LawVersion 记录（幂等：按 version_key 去重）
    law_version = session.scalar(select(LawVersion).where(LawVersion.version_key == version_key))
    if law_version is None:
        law_version = LawVersion(
            version_key=version_key,
            law_id=law.id,
            version_number="v1",
            promulgation_date=metadata.promulgation_date,  # 留空表示待补录
            effective_date=metadata.effective_date,  # 留空表示待补录
            expiration_date=metadata.expiration_date,
            status=metadata.status,  # 留空表示待补录
            document_version_id=version_id,
        )
        session.add(law_version)
        session.flush()
    else:
        # 已存在：用本次从页面重新抽取的时效字段刷新（批次 10）。
        # 抽取规则增强后，旧记录里留空（待补录）的日期/状态可能已能抽到，
        # 所以重导入要能把这些字段补上。
        #
        # 批次 26-C5：**抽不到（None）不得清空已有非空值**。
        # 时效字段（公布/生效/失效日期、效力状态）是人工补录与人工核对的成果
        # （见《人工确认清单》、scripts/migrations/backfill_law_dates.py、
        # scripts/migrations/migrate_law_status_vocabulary.py），而重导入是例行操作。
        # 旧实现是"取值不同就写"，于是 None 也会被当作新值写回去——
        # 只要页面改版、抽取规则退化、或原始 HTML 缺了该字段，
        # 一次重导入就会静默抹掉人工成果，且不留痕迹。
        # 新规则：**新值非空才写**（新值为 None 时保留原值；原值也为空时本就无值可写，
        # 因此"原值为空才写"这一半被这条规则完全覆盖）。
        # 副作用说明：这是有意的——重复导入不再能把某字段"改回空"，
        # 需要清空请显式改库（人工可追溯），不要让例行操作承担数据丢失的风险。
        refreshed = {
            "promulgation_date": metadata.promulgation_date,
            "effective_date": metadata.effective_date,
            "expiration_date": metadata.expiration_date,
            "status": metadata.status,
        }
        for field, value in refreshed.items():
            if value is not None and getattr(law_version, field) != value:
                setattr(law_version, field, value)

    # 6. 从 chunks 创建 Article 记录（只处理有 article_number 的父块）
    from app.ingest.article_number_rules import build_article_path
    from app.ingest.chinese_number import to_arabic_number

    parent_chunks = [
        c for c in package.data.chunks
        if c.chunk_type == "parent" and c.article_number is not None
    ]

    skipped_articles = []  # 记录无法识别的条文

    for chunk in parent_chunks:
        try:
            # 先将中文条号转为阿拉伯数字
            arabic_article = to_arabic_number(chunk.article_number)
            arabic_paragraph = to_arabic_number(chunk.paragraph_number)
            arabic_item = to_arabic_number(chunk.item_number)

            # 生成 article_path
            article_path = build_article_path(
                article_number=arabic_article,
                paragraph_number=arabic_paragraph,
                item_number=arabic_item,
            )
            article_key = f"{version_key}_art{article_path}"

            # 幂等：按 article_key 去重
            existing = session.scalar(select(Article).where(Article.article_key == article_key))
            if existing is None:
                session.add(
                    Article(
                        article_key=article_key,
                        law_version_id=law_version.id,
                        article_number=arabic_article,
                        paragraph_number=arabic_paragraph,
                        item_number=arabic_item,
                        article_path=article_path,
                        content=chunk.content,
                        sequence=chunk.sequence,
                    )
                )
        except (ValueError, TypeError) as e:
            # 单条识别失败不拖垮整篇文档
            skipped_articles.append({
                "article_number": chunk.article_number,
                "sequence": chunk.sequence,
                "error": str(e)[:50]
            })

    # 如果有跳过的条文，记录到日志（但不中断导入）
    if skipped_articles:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(
            f"Skipped {len(skipped_articles)} articles in {metadata.law_name}: "
            f"{skipped_articles[:3]}"  # 只记录前3条样本
        )


def _generate_law_key(law_name: str) -> str:
    """从法规名称生成稳定的 law_key。

    规则：
    - 提取"中华人民共和国XXX法"中的"XXX"
    - 转为拼音首字母或规范化标识符
    - 当前简化版：直接用法规名的哈希前16位（保证唯一性）
    """
    import hashlib

    # 简化版：使用哈希保证唯一性（生产环境应改为拼音或手工映射）
    name_hash = hashlib.sha256(law_name.encode('utf-8')).hexdigest()[:16]
    return f"law_{name_hash}"
