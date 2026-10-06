"""法规时效元数据刷新规则：重导入不得冲掉人工补录/人工核对的取值（批次 26-C5）。

背景：`legal_metadata_writer._add_legal_metadata` 在 law_versions 已存在时会用
"本次从页面重新抽取"的时效字段刷新旧记录（批次 10 加的能力，目的是让抽取规则增强后
新抽到的日期能进库）。但旧实现是"取值不同就写"，**None 也算新值**——
于是页面改版、抽取规则退化、或原始 HTML 缺了该字段时，
一次例行的重导入就会把人工补录的公布/生效/失效日期与效力状态**静默清空**，
而且不留任何痕迹。批次 26 往 status 写 `不适用`（页面根本不会写这个词，抽取器永远抽不到）
正是最容易踩这个坑的取值。

新规则：**新值非空才写**；抽不到（None）保留原值。

为什么走真实重导入路径：数据包内容没变时 `import_package` 在增量判定处就短路了
（unchanged 分支根本不碰 law_versions），只有内容哈希变化（updated 分支）才会刷新。
所以这里造两个 content_hash 不同的包，模拟"官网改版后重抓重导"。
"""
from datetime import date

from app.db.import_service import import_package
from app.db.law_status import NOT_APPLICABLE
from app.db.legal_metadata_writer import _add_legal_metadata
from app.db.sql_models import LawVersion
from app.pipeline.package_models import (
    ChunkEntry,
    CrawlEntry,
    DocumentEntry,
    PackageData,
    VersionEntry,
)
from app.pipeline.package_validator import PackageManifest, SCHEMA_VERSION, ValidatedPackage
from app.pipeline.package_writer import PIPELINE_VERSION
from sqlalchemy import select

from conftest import create_test_session

LAW_TITLE = "中华人民共和国测试法"
# 页面正文：带"法律"判定所需的国名 + 发布机关；status_text 决定页面写了什么效力状态
PAGE_TEMPLATE = (
    "<html><body><h1>{title}</h1><p>全国人民代表大会常务委员会</p>"
    "<p>第一条 测试正文。</p>{status}</body></html>"
)


def build_page(status_text: str = "") -> str:
    """生成一页法规原文；status_text 为空表示页面**没有**写效力状态。"""
    return PAGE_TEMPLATE.format(title=LAW_TITLE, status=status_text)


def build_package(package_id: str, version_id: str, content_hash: str) -> ValidatedPackage:
    """构造一个最小可导入的数据包（同文档、同 raw 文件路径，仅内容哈希不同）。"""
    data = PackageData(
        document=DocumentEntry(
            document_id="doc-c5",
            source_url="https://example.com/c5",
            title=LAW_TITLE,
        ),
        version=VersionEntry(
            version_id=version_id,
            document_id="doc-c5",
            content_hash=content_hash,
            media_type="text/html",
            cleaned_content="第一条 测试正文。",
            raw_file_path="raw/a.html",
        ),
        chunks=(
            ChunkEntry(
                chunk_id=f"{version_id}-parent",
                version_id=version_id,
                parent_chunk_id=None,
                chunk_type="parent",
                article_number="第一条",
                sequence=1,
                content="第一条 测试正文。",
                retrieval_text=f"{LAW_TITLE} 第一条 测试正文。",
            ),
        ),
        crawl=CrawlEntry(
            crawl_id=f"crawl-{content_hash}",
            source_id="source-c5",
            source_url="https://example.com/c5",
            http_status=200,
            content_hash=content_hash,
            raw_file_path="raw/a.html",
            collected_at="2026-09-21T00:00:00+00:00",
        ),
    )
    manifest = PackageManifest(
        schema_version=SCHEMA_VERSION,
        pipeline_version=PIPELINE_VERSION,
        package_id=package_id,
        created_at="2026-09-21T00:00:00+00:00",
        source_id="source-c5",
        source_url="https://example.com/c5",
        record_counts={},
        file_hashes={},
    )
    return ValidatedPackage(manifest=manifest, data=data)


def redirect_raw_file_lookup(monkeypatch, tmp_path, page_html: str) -> None:
    """把"原始 HTML 目录"指到临时目录。

    `_add_legal_metadata` 用 `Path(__file__).resolve().parents[3]` 定位项目根，
    再去 data/labor_law_raw/ 找原始 HTML。这里改的是模块的 __file__，
    让它算出的"项目根"落在 tmp_path —— 不改生产代码、也不往项目 data/ 里写测试文件。
    文件名必须是 a.html（与 build_package 的 raw_file_path 一致）。
    """
    fake_module_file = tmp_path / "backend" / "app" / "db" / "legal_metadata_writer.py"
    raw_dir = tmp_path / "data" / "labor_law_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "a.html").write_text(page_html, encoding="utf-8")
    import app.db.legal_metadata_writer as writer

    monkeypatch.setattr(writer, "__file__", str(fake_module_file))


def _single_law_version(session) -> LawVersion:
    return session.scalar(select(LawVersion))


def test_reimport_keeps_manually_filled_status_and_dates(monkeypatch, tmp_path) -> None:
    """核心用例：人工填了 status 与生效日期 → 走一次真实重导入 → 取值全部保留。"""
    session = create_test_session()
    try:
        # 官网页面没有写效力状态（这正是"人工补录"会发生的场景）
        redirect_raw_file_lookup(monkeypatch, tmp_path, build_page())
        first = import_package(session, build_package("pkg-c5-1", "ver-c5-1", "hash-c5-1"))
        assert first.status == "imported", first

        version = _single_law_version(session)
        assert version.status is None, "页面没写状态，首导应留空待人工补录"
        print(f"\n  [首导后]   status={version.status!r}  effective_date={version.effective_date}")

        # 人工核对后填库（模拟《人工确认清单》回填；页面永远抽不到"不适用"）
        version.status = NOT_APPLICABLE
        version.effective_date = date(2008, 1, 1)
        session.flush()
        print(f"  [人工填写] status={version.status!r}  effective_date={version.effective_date}")

        # 官网改版后重抓重导：内容哈希变化 → updated 分支 → 触发时效字段刷新
        second = import_package(session, build_package("pkg-c5-2", "ver-c5-2", "hash-c5-2"))
        assert second.status == "imported", second

        after = _single_law_version(session)
        print(f"  [重导入后] status={after.status!r}  effective_date={after.effective_date}")
        assert after.status == NOT_APPLICABLE, "重导入把人工填的效力状态清空了"
        assert after.effective_date == date(2008, 1, 1), "重导入把人工补的生效日期清空了"
    finally:
        session.close()


def test_reimport_still_fills_empty_fields_from_page(monkeypatch, tmp_path) -> None:
    """反面：抽取规则增强后新抽到的值仍要能进库（批次 10 的能力不得退化）。"""
    session = create_test_session()
    try:
        redirect_raw_file_lookup(monkeypatch, tmp_path, build_page())
        import_package(session, build_package("pkg-c5-3", "ver-c5-3", "hash-c5-3"))
        assert _single_law_version(session).status is None

        # 页面这次写了效力状态 → 新值非空 → 应当写进库
        redirect_raw_file_lookup(monkeypatch, tmp_path, build_page("<p>现行有效</p>"))
        result = import_package(session, build_package("pkg-c5-4", "ver-c5-4", "hash-c5-4"))
        assert result.status == "imported", result
        assert _single_law_version(session).status == "现行有效"
    finally:
        session.close()


def test_reimport_does_not_clear_expiration_date(monkeypatch, tmp_path) -> None:
    """失效日期页面从不写（抽取器恒返回 None），绝不能被重导入抹掉。"""
    session = create_test_session()
    try:
        redirect_raw_file_lookup(monkeypatch, tmp_path, build_page())
        import_package(session, build_package("pkg-c5-5", "ver-c5-5", "hash-c5-5"))
        version = _single_law_version(session)
        version.expiration_date = date(2024, 12, 31)
        session.flush()

        import_package(session, build_package("pkg-c5-6", "ver-c5-6", "hash-c5-6"))
        assert _single_law_version(session).expiration_date == date(2024, 12, 31)
    finally:
        session.close()


def test_refresh_rule_only_writes_non_null_values(monkeypatch, tmp_path) -> None:
    """规则本身写在函数里的边界说明：新值非空才写（页面值优先于人工值）。

    这是本轮**有意**保留的边界，不是漏改：
    - 抽不到（None）→ 保留原值（防"静默清空"，本批次修的就是这个）
    - 抽到了且与库里不同 → 按页面刷新（批次 10 的既定口径：时效以页面实际内容为准）
    若将来要求"人工值优先"，需要在库里加"人工确认过"的标记，而不是在这里反向推断。
    """
    session = create_test_session()
    try:
        redirect_raw_file_lookup(monkeypatch, tmp_path, build_page("<p>已废止</p>"))
        import_package(session, build_package("pkg-c5-7", "ver-c5-7", "hash-c5-7"))
        assert _single_law_version(session).status == "已废止"

        redirect_raw_file_lookup(monkeypatch, tmp_path, build_page("<p>现行有效</p>"))
        import_package(session, build_package("pkg-c5-8", "ver-c5-8", "hash-c5-8"))
        assert _single_law_version(session).status == "现行有效"
    finally:
        session.close()


def test_add_legal_metadata_is_idempotent_on_same_page(monkeypatch, tmp_path) -> None:
    """同一页重复写入不清空、不重复建行（直接调写入函数，跑两遍）。"""
    session = create_test_session()
    try:
        redirect_raw_file_lookup(monkeypatch, tmp_path, build_page("<p>现行有效</p>"))
        package = build_package("pkg-c5-9", "ver-c5-9", "hash-c5-9")
        from app.db.import_service import _import_package_rows

        _import_package_rows(session, package)
        version = _single_law_version(session)
        version_id = version.id
        version.expiration_date = date(2030, 1, 1)
        session.flush()

        # 再写一遍（同页、同版本）→ 应更新为同一值，且不新建行、不清掉失效日期
        _add_legal_metadata(session, package, version.document_version_id)
        versions = session.scalars(select(LawVersion)).all()
        assert len(versions) == 1
        assert versions[0].id == version_id
        assert versions[0].status == "现行有效"
        assert versions[0].expiration_date == date(2030, 1, 1)
    finally:
        session.close()
