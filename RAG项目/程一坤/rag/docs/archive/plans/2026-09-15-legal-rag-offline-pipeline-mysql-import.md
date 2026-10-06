# 法律 RAG 离线 Pipeline 与 MySQL 入库实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 subagent-driven-development（推荐）或 executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 建立一个从授权官方来源爬取、解析、清洗、父子分块并生成标准数据包的离线 Pipeline，再通过独立脚本幂等、事务化导入 MySQL。

**架构：** 离线 Pipeline 不连接数据库，只生成经过完整性校验且原子发布的不可变数据包。MySQL 入库脚本读取并复验数据包，通过 SQLAlchemy Repository 写入文档、版本、采集记录、父子 chunk 和导入记录；Milvus、Redis、Embedding 和检索 Pipeline 不进入本计划。

**技术栈：** Python 3.11+、Python 标准库、SQLAlchemy 2.x、PyMySQL、pytest、SQLite 内存测试、MySQL 8.x。

**规格：** `docs/superpowers/specs/2026-09-15-legal-rag-mysql-offline-pipeline-design.md`

## 全局约束

- 正式法律数据只能通过授权官方来源采集。
- 测试假数据只写入 pytest 临时目录或 SQLite 内存数据库，不污染项目 `data` 和 `artifacts` 目录。
- 每个代码文件不超过 300 行。
- API Key、数据库密码和完整连接 URL 不得硬编码、打印或写入数据包。
- 所有新增注释和用户可见错误使用简体中文，技术标识保持英文。
- 离线 Pipeline 不连接 MySQL、Redis 或 Milvus。
- MySQL 入库脚本不重新爬取、解析、清洗或分块。
- MySQL 不保存 1024 维向量。
- 所有功能遵循 TDD：先确认测试失败，再编写最少实现并确认通过。
- 当前工作区不是 Git 仓库，不执行 commit、branch 或 worktree 操作。

## 文件结构

### 创建

- `backend/pyproject.toml`：声明 Python、SQLAlchemy、PyMySQL 和 pytest 依赖及测试配置。
- `backend/.env.example`：只提供非敏感配置名和示例占位值。
- `backend/app/pipeline/package_models.py`：数据包内存模型及 JSON 序列化边界。
- `backend/app/pipeline/package_validator.py`：Schema、哈希、记录数、引用和路径安全校验。
- `backend/app/pipeline/package_writer.py`：临时目录写入、manifest 生成及原子发布。
- `backend/app/pipeline/offline_pipeline.py`：串联现有 crawler、parser、chunker 并生成数据包。
- `backend/app/pipeline/__init__.py`：导出离线 Pipeline 公共接口。
- `backend/app/config.py`：安全读取和校验 `DATABASE_URL`。
- `backend/app/database/base.py`：SQLAlchemy `DeclarativeBase`。
- `backend/app/database/session.py`：Engine 和 Session 工厂。
- `backend/app/database/models.py`：文档、版本、采集记录、chunk 和导入记录 ORM。
- `backend/app/database/repository.py`：数据包级幂等与事务写入。
- `backend/app/database/__init__.py`：导出数据库公共接口。
- `backend/app/importers/mysql_importer.py`：数据包加载、复验和 MySQL 导入服务。
- `backend/app/importers/__init__.py`：导出导入服务。
- `backend/app/cli/build_ingestion_package.py`：离线 Pipeline 命令入口。
- `backend/app/cli/import_mysql.py`：MySQL 入库命令入口。
- `backend/app/cli/__init__.py`：CLI 包标识。
- `backend/tests/unit/test_package_models.py`：数据包模型与序列化测试。
- `backend/tests/unit/test_package_validator.py`：完整性与安全校验测试。
- `backend/tests/unit/test_package_writer.py`：原子发布测试。
- `backend/tests/unit/test_offline_pipeline.py`：完整离线处理测试。
- `backend/tests/unit/test_config.py`：数据库配置脱敏测试。
- `backend/tests/unit/test_database_repository.py`：SQLite 内存事务和幂等测试。
- `backend/tests/unit/test_mysql_importer.py`：导入服务测试。
- `backend/tests/unit/test_cli.py`：两个 CLI 的参数和退出码测试。
- `backend/tests/integration/test_mysql_import.py`：显式启用的真实 MySQL 集成测试。

### 保持稳定，仅在测试证明必要时修改

- `backend/app/crawler/runner.py:26`：现有采集和原始文件原子保存入口。
- `backend/app/ingest/parser.py:92`：现有统一解析与清洗入口。
- `backend/app/ingest/chunker.py:49`：现有父子分块入口。

---

### 任务 1：建立数据包模型与稳定标识

**文件：**
- 创建：`backend/app/pipeline/package_models.py`
- 创建：`backend/app/pipeline/__init__.py`
- 测试：`backend/tests/unit/test_package_models.py`

- [ ] **步骤 1：编写数据包往返序列化失败测试**

```python
from pathlib import Path

from app.pipeline.package_models import (
    ChunkEntry,
    CrawlEntry,
    DocumentEntry,
    PackageData,
    VersionEntry,
    stable_document_id,
    stable_version_id,
)


def test_package_data_round_trips_json_lines(tmp_path: Path) -> None:
    document_id = stable_document_id("https://example.gov.cn/law/1")
    version_id = stable_version_id(document_id, "a" * 64)
    package = PackageData(
        document=DocumentEntry(
            document_id=document_id,
            source_url="https://example.gov.cn/law/1",
            title="测试法律",
        ),
        version=VersionEntry(
            version_id=version_id,
            document_id=document_id,
            content_hash="a" * 64,
            media_type="html",
            cleaned_content="第一条 测试内容",
            raw_file_path="raw/" + "a" * 64 + ".html",
        ),
        chunks=(
            ChunkEntry(
                chunk_id="law-a1",
                version_id=version_id,
                parent_chunk_id=None,
                chunk_type="parent",
                article_number="第一条",
                sequence=1,
                content="第一条 测试内容",
                retrieval_text="测试法律\n第一条\n第一条 测试内容",
            ),
        ),
        crawl=CrawlEntry(
            crawl_id="crawl-1",
            source_id="official-test",
            source_url="https://example.gov.cn/law/1",
            http_status=200,
            content_hash="a" * 64,
            raw_file_path="raw/" + "a" * 64 + ".html",
            collected_at="2026-09-15T00:00:00+00:00",
        ),
    )

    package.write_records(tmp_path)
    loaded = PackageData.read_records(tmp_path)

    assert loaded == package
```

- [ ] **步骤 2：运行测试并确认因模块不存在而失败**

运行：`cd backend && python -m pytest tests/unit/test_package_models.py -v`

预期：FAIL，包含 `ModuleNotFoundError: No module named 'app.pipeline'`。

- [ ] **步骤 3：实现不可变数据模型和 JSONL 序列化**

在 `package_models.py` 中定义 `frozen=True` dataclass：

```python
@dataclass(frozen=True)
class DocumentEntry:
    document_id: str
    source_url: str
    title: str


@dataclass(frozen=True)
class VersionEntry:
    version_id: str
    document_id: str
    content_hash: str
    media_type: str
    cleaned_content: str
    raw_file_path: str


@dataclass(frozen=True)
class ChunkEntry:
    chunk_id: str
    version_id: str
    parent_chunk_id: str | None
    chunk_type: str
    article_number: str | None
    sequence: int
    content: str
    retrieval_text: str


@dataclass(frozen=True)
class CrawlEntry:
    crawl_id: str
    source_id: str
    source_url: str
    http_status: int
    content_hash: str
    raw_file_path: str
    collected_at: str
```

`PackageData.write_records()` 分别写入四个 UTF-8 JSONL 文件；`read_records()` 严格按字段恢复对象。稳定标识使用完整输入 SHA-256：

```python
def stable_document_id(source_url: str) -> str:
    return "doc-" + sha256(source_url.strip().encode("utf-8")).hexdigest()[:24]


def stable_version_id(document_id: str, content_hash: str) -> str:
    value = f"{document_id}:{content_hash}"
    return "ver-" + sha256(value.encode("utf-8")).hexdigest()[:24]
```

- [ ] **步骤 4：补充稳定性和字段缺失测试**

覆盖：相同 URL 标识稳定、不同 URL 不冲突、版本标识随哈希变化、缺少必要字段时抛出 `PackageFormatError`，错误消息不包含记录正文。

- [ ] **步骤 5：运行数据包模型测试**

运行：`cd backend && python -m pytest tests/unit/test_package_models.py -v`

预期：全部 PASS。

### 任务 2：实现数据包完整性与路径安全校验

**文件：**
- 创建：`backend/app/pipeline/package_validator.py`
- 测试：`backend/tests/unit/test_package_validator.py`

- [ ] **步骤 1：编写被篡改文件和逃逸路径失败测试**

```python
def test_validate_package_rejects_tampered_file(valid_package_dir: Path) -> None:
    chunks_path = valid_package_dir / "document_chunks.jsonl"
    chunks_path.write_text("{}\n", encoding="utf-8")

    try:
        validate_package(valid_package_dir)
    except PackageValidationError as error:
        assert "文件哈希不一致" in str(error)
    else:
        raise AssertionError("篡改后的数据包必须被拒绝")


def test_validate_package_rejects_escaping_raw_path(valid_package_dir: Path) -> None:
    replace_raw_path_and_refresh_manifest(valid_package_dir, "../secret.html")

    try:
        validate_package(valid_package_dir)
    except PackageValidationError as error:
        assert "原始文件路径非法" in str(error)
    else:
        raise AssertionError("逃逸数据包目录的路径必须被拒绝")
```

- [ ] **步骤 2：运行测试并确认校验器尚不存在**

运行：`cd backend && python -m pytest tests/unit/test_package_validator.py -v`

预期：FAIL，缺少 `validate_package`。

- [ ] **步骤 3：实现 manifest 和校验结果类型**

定义：

```python
@dataclass(frozen=True)
class PackageManifest:
    schema_version: str
    pipeline_version: str
    package_id: str
    created_at: str
    source_url: str
    record_counts: dict[str, int]
    file_hashes: dict[str, str]


@dataclass(frozen=True)
class ValidatedPackage:
    manifest: PackageManifest
    data: PackageData
    package_root: Path
```

`validate_package(package_root)` 必须检查：必需文件、支持的 `schema_version="1.0"`、哈希、记录数、`document_id/version_id` 关联、父子 chunk 关联、chunk ID 唯一、相对路径位于 package root 且指向普通文件。

- [ ] **步骤 4：补充引用关系与重复 ID 测试**

覆盖：子块缺少父块、chunk 引用错误版本、重复 `chunk_id`、manifest 数量错误、缺少 raw 文件和不支持的 Schema 版本。

- [ ] **步骤 5：运行校验器测试**

运行：`cd backend && python -m pytest tests/unit/test_package_validator.py -v`

预期：全部 PASS。

### 任务 3：实现数据包写入与原子发布

**文件：**
- 创建：`backend/app/pipeline/package_writer.py`
- 测试：`backend/tests/unit/test_package_writer.py`

- [ ] **步骤 1：编写成功发布与失败无半成品测试**

```python
def test_publish_package_writes_manifest_and_renames_atomically(
    tmp_path: Path,
    package_data: PackageData,
    raw_source: Path,
) -> None:
    package_dir = publish_package(
        package_data=package_data,
        raw_source=raw_source,
        output_root=tmp_path,
        package_id="pkg-001",
        created_at="2026-09-15T00:00:00+00:00",
        pipeline_version="1.0.0",
    )

    assert package_dir == tmp_path / "pkg-001"
    assert (package_dir / "manifest.json").is_file()
    assert not (tmp_path / ".pkg-001.tmp").exists()
    assert validate_package(package_dir).manifest.package_id == "pkg-001"
```

失败用例注入一个抛出异常的 validator，断言最终目录不存在、既有目录不被覆盖。

- [ ] **步骤 2：运行测试确认发布器尚不存在**

运行：`cd backend && python -m pytest tests/unit/test_package_writer.py -v`

预期：FAIL，缺少 `publish_package`。

- [ ] **步骤 3：实现临时写入、manifest 生成和原子重命名**

`publish_package()`：

1. 拒绝已存在的最终目录；
2. 创建同级 `.<package_id>.tmp`；
3. 复制原始文件到 `raw/<content_hash>.<extension>`；
4. 写四个 JSONL；
5. 计算每个受管文件 SHA-256；
6. 写 `manifest.json`；
7. 调用 `validate_package()`；
8. 使用 `Path.replace()` 发布最终目录；
9. 失败时只清理本次创建的临时目录。

- [ ] **步骤 4：补充不可覆盖与哈希确定性测试**

覆盖：目标目录已存在时失败、同样输入的受管文件哈希一致、manifest 不包含绝对本地路径和敏感字段。

- [ ] **步骤 5：运行发布器及前置测试**

运行：`cd backend && python -m pytest tests/unit/test_package_models.py tests/unit/test_package_validator.py tests/unit/test_package_writer.py -v`

预期：全部 PASS。

### 任务 4：贯通离线 Pipeline

**文件：**
- 创建：`backend/app/pipeline/offline_pipeline.py`
- 修改：`backend/app/pipeline/__init__.py`
- 测试：`backend/tests/unit/test_offline_pipeline.py`

- [ ] **步骤 1：编写爬取到数据包的失败测试**

```python
def test_offline_pipeline_builds_valid_package(
    tmp_path: Path,
    successful_crawl_result: SavedCrawlResult,
) -> None:
    crawler = FakeCrawler(successful_crawl_result)
    pipeline = OfflinePipeline(
        crawler=crawler,
        output_root=tmp_path / "artifacts",
        parse=parse_document,
        chunk=chunk_document,
        now=lambda: datetime(2026, 9, 15, tzinfo=timezone.utc),
        id_factory=lambda: "pkg-001",
    )

    result = pipeline.run(
        crawl_id="crawl-001",
        source_id="official-test",
        source_url="https://example.gov.cn/law/1",
        document_title="测试法律",
    )

    validated = validate_package(result.package_path)
    assert result.parent_chunk_count == 2
    assert result.child_chunk_count == 2
    assert validated.data.document.source_url.endswith("/law/1")
```

Fake crawler 返回位于 `tmp_path` 的 HTML，其中包含两个条文和正文引用条号。

- [ ] **步骤 2：运行测试确认 Pipeline 尚不存在**

运行：`cd backend && python -m pytest tests/unit/test_offline_pipeline.py -v`

预期：FAIL，缺少 `OfflinePipeline`。

- [ ] **步骤 3：实现 Pipeline 结果和依赖协议**

定义：

```python
@dataclass(frozen=True)
class OfflinePipelineResult:
    package_id: str
    package_path: Path
    document_id: str
    version_id: str
    parent_chunk_count: int
    child_chunk_count: int


class OfflinePipeline:
    def run(
        self,
        crawl_id: str,
        source_id: str,
        source_url: str,
        document_title: str,
    ) -> OfflinePipelineResult:
        ...
```

执行顺序固定为 crawler → `parse_document(file_path)` → `chunk_document(document, document_title, document_id)` → 映射 PackageData → `publish_package()`。

- [ ] **步骤 4：实现阶段化错误**

定义 `OfflinePipelineError(stage: str, message: str)`，阶段限定为 `crawl`、`parse`、`chunk`、`validate`、`publish`。错误只包含截断后的异常文本，不包含页面正文。采集失败、无文件、空分块或发布失败均不得生成最终数据包。

- [ ] **步骤 5：补充失败与隔离测试**

覆盖：采集失败、解析失败、空正文无 chunk、发布失败、测试输出仅位于 `tmp_path`、Pipeline 不导入或调用数据库模块。

- [ ] **步骤 6：运行离线 Pipeline 测试和现有采集/解析/分块回归测试**

运行：

```bash
cd backend && python -m pytest \
  tests/unit/test_offline_pipeline.py \
  tests/unit/test_runner.py \
  tests/unit/test_parser.py \
  tests/unit/test_cleaner.py \
  tests/unit/test_chunker.py -v
```

预期：全部 PASS。

### 任务 5：建立 SQLAlchemy 配置、模型和会话工厂

**文件：**
- 创建：`backend/pyproject.toml`
- 创建：`backend/.env.example`
- 创建：`backend/app/config.py`
- 创建：`backend/app/database/base.py`
- 创建：`backend/app/database/session.py`
- 创建：`backend/app/database/models.py`
- 创建：`backend/app/database/__init__.py`
- 测试：`backend/tests/unit/test_config.py`
- 测试：`backend/tests/unit/test_database_repository.py`

- [ ] **步骤 1：声明最小依赖**

`pyproject.toml` 使用 `setuptools`，声明：

```toml
[project]
name = "legal-rag-backend"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "SQLAlchemy>=2.0,<3.0",
  "PyMySQL>=1.1,<2.0",
]

[project.optional-dependencies]
test = ["pytest>=8.0,<9.0"]
```

保留现有 `pytest.ini`，不重复配置 pytest。

- [ ] **步骤 2：编写配置脱敏失败测试**

```python
def test_database_settings_safe_summary_hides_credentials() -> None:
    settings = DatabaseSettings.from_url(
        "mysql+pymysql://legal_user:secret@example.invalid:3306/legal_rag"
    )

    summary = settings.safe_summary()

    assert "secret" not in summary
    assert "legal_user" not in summary
    assert "example.invalid" in summary
    assert "legal_rag" in summary
```

同时测试空 URL 和非 `mysql+pymysql`/`sqlite` URL 被拒绝。

- [ ] **步骤 3：运行配置测试确认失败**

运行：`cd backend && python -m pytest tests/unit/test_config.py -v`

预期：FAIL，缺少 `DatabaseSettings`。

- [ ] **步骤 4：实现安全配置和会话工厂**

`DatabaseSettings.from_env()` 只读取 `DATABASE_URL`；`from_url()` 使用 SQLAlchemy URL 解析器校验。`create_engine_from_settings()` 对 SQLite 测试配置启用 `StaticPool` 和 `check_same_thread=False`，`create_session_factory()` 返回 `sessionmaker[Session]`。

- [ ] **步骤 5：编写 ORM Schema 失败测试**

在 SQLite 内存库调用 `Base.metadata.create_all(engine)`，断言存在：

```python
{
    "documents",
    "document_versions",
    "crawl_records",
    "document_chunks",
    "import_records",
}
```

同时断言 `documents.source_url`、`document_versions.version_key`、`document_chunks.chunk_id`、`import_records.package_id` 唯一。

- [ ] **步骤 6：实现 SQLAlchemy 2.x typed declarative 模型**

使用 `Mapped[...]` 与 `mapped_column()`；长正文使用 `Text`，时间使用带时区 `DateTime(timezone=True)`，关联使用明确 `ForeignKey`。通过表级唯一约束保证 `document_id + content_hash` 唯一。模型不保存向量字段。

- [ ] **步骤 7：运行配置和 Schema 测试**

运行：`cd backend && python -m pytest tests/unit/test_config.py tests/unit/test_database_repository.py -v`

预期：当前已实现测试 PASS；Repository 行为测试尚未加入。

### 任务 6：实现数据包事务与幂等 Repository

**文件：**
- 创建：`backend/app/database/repository.py`
- 修改：`backend/app/database/__init__.py`
- 修改：`backend/tests/unit/test_database_repository.py`

- [ ] **步骤 1：编写首次导入、unchanged 和 updated 失败测试**

```python
def test_import_package_creates_new_then_updated_versions(session: Session) -> None:
    repository = PackageRepository(session)

    first = repository.import_package(make_validated_package(content_hash="a" * 64))
    repeated = repository.import_package(make_validated_package(content_hash="a" * 64, package_id="pkg-2"))
    updated = repository.import_package(make_validated_package(content_hash="b" * 64, package_id="pkg-3"))

    assert first.status == "new"
    assert repeated.status == "unchanged"
    assert updated.status == "updated"
    assert session.scalar(select(func.count(DocumentVersion.id))) == 2
```

- [ ] **步骤 2：运行测试确认 Repository 尚不存在**

运行：`cd backend && python -m pytest tests/unit/test_database_repository.py -v`

预期：FAIL，缺少 `PackageRepository`。

- [ ] **步骤 3：实现导入结果和事务写入**

定义：

```python
@dataclass(frozen=True)
class ImportResult:
    package_id: str
    status: str
    document_id: int
    document_version_id: int | None
    chunk_count: int


class PackageRepository:
    def import_package(self, package: ValidatedPackage) -> ImportResult:
        ...
```

Repository 只执行数据库映射：按 `source_url` 获取/创建 Document；按 `document_id + content_hash` 判定版本；按父块后子块顺序写入；成功后设置 `current_version_id`；新版本 `processing_status="awaiting_embedding"`。

- [ ] **步骤 4：实现数据包和记录级幂等**

相同成功 `package_id` 返回 `already_imported`；相同内容的新 package 返回 `unchanged` 并写审计记录但不新增版本和 chunk；`chunk_id` 冲突且属于其他版本时抛出 `PackagePersistenceError`。

- [ ] **步骤 5：编写事务回滚测试**

注入包含冲突 chunk 的数据包，调用方使用 `with session.begin():`，断言异常后 DocumentVersion、DocumentChunk 和成功 ImportRecord 数量均未增加。

- [ ] **步骤 6：运行 Repository 测试**

运行：`cd backend && python -m pytest tests/unit/test_database_repository.py -v`

预期：全部 PASS。

### 任务 7：实现 MySQL 导入服务

**文件：**
- 创建：`backend/app/importers/mysql_importer.py`
- 创建：`backend/app/importers/__init__.py`
- 测试：`backend/tests/unit/test_mysql_importer.py`

- [ ] **步骤 1：编写校验先于数据库事务的失败测试**

```python
def test_importer_rejects_invalid_package_before_opening_session(
    tampered_package_dir: Path,
) -> None:
    sessions = CountingSessionFactory()
    importer = MysqlPackageImporter(session_factory=sessions)

    try:
        importer.import_path(tampered_package_dir)
    except MysqlImportError as error:
        assert error.stage == "validate"
    else:
        raise AssertionError("非法数据包必须导入失败")

    assert sessions.call_count == 0
```

- [ ] **步骤 2：运行测试确认 Importer 尚不存在**

运行：`cd backend && python -m pytest tests/unit/test_mysql_importer.py -v`

预期：FAIL，缺少 `MysqlPackageImporter`。

- [ ] **步骤 3：实现 load、validate、persist 三阶段服务**

`import_path(package_path)` 先调用 `validate_package()`，通过后才创建 Session，并在 `with session.begin():` 中调用 Repository。返回 `ImportResult`；错误包装为带固定阶段的 `MysqlImportError`，错误摘要最多 500 字符并脱敏。

- [ ] **步骤 4：补充成功、重复和回滚测试**

覆盖首次 `new`、新 package 相同内容 `unchanged`、相同 package `already_imported`、新哈希 `updated` 和持久化异常回滚。

- [ ] **步骤 5：运行 Importer 与 Repository 测试**

运行：`cd backend && python -m pytest tests/unit/test_database_repository.py tests/unit/test_mysql_importer.py -v`

预期：全部 PASS。

### 任务 8：实现两个命令行入口

**文件：**
- 创建：`backend/app/cli/__init__.py`
- 创建：`backend/app/cli/build_ingestion_package.py`
- 创建：`backend/app/cli/import_mysql.py`
- 测试：`backend/tests/unit/test_cli.py`

- [ ] **步骤 1：编写参数解析和退出码失败测试**

```python
def test_import_mysql_main_returns_zero_for_success(tmp_path: Path) -> None:
    importer = FakeImporter(status="new")

    exit_code = import_mysql_main(
        ["--package", str(tmp_path / "pkg-001")],
        importer_factory=lambda: importer,
    )

    assert exit_code == 0
    assert importer.paths == [tmp_path / "pkg-001"]
```

同时测试缺失参数由 argparse 返回非零、业务失败返回 1、输出不包含连接 URL 和密码。

- [ ] **步骤 2：运行测试确认 CLI 尚不存在**

运行：`cd backend && python -m pytest tests/unit/test_cli.py -v`

预期：FAIL，缺少 CLI 模块。

- [ ] **步骤 3：实现离线 Pipeline CLI**

命令：

```bash
python -m app.cli.build_ingestion_package \
  --crawl-id <id> \
  --source-id <whitelist-id> \
  --source-url <authorized-url> \
  --title <document-title> \
  --output <artifact-root>
```

`main(argv=None, pipeline_factory=None) -> int`，便于单测注入。成功输出 package ID、路径和父子块计数；失败只输出阶段和脱敏摘要。

- [ ] **步骤 4：实现 MySQL 入库 CLI**

命令：

```bash
python -m app.cli.import_mysql --package <package-path>
```

从 `DatabaseSettings.from_env()` 创建 Engine、Session factory 和 Importer。成功输出 `new`、`updated`、`unchanged` 或 `already_imported` 及记录计数，不打印完整 URL。

- [ ] **步骤 5：补充构造依赖测试**

验证 build CLI 使用现有官方白名单和 requester 构造 crawler；所有输出目录必须由 `--output` 显式指定；测试中只使用 fake，不访问网络。

- [ ] **步骤 6：运行 CLI 测试**

运行：`cd backend && python -m pytest tests/unit/test_cli.py -v`

预期：全部 PASS。

### 任务 9：增加真实 MySQL 集成测试入口

**文件：**
- 创建：`backend/tests/integration/test_mysql_import.py`
- 修改：`backend/pytest.ini`

- [ ] **步骤 1：注册 integration 标记**

在 `pytest.ini` 增加：

```ini
markers =
    integration: 需要显式外部服务的集成测试
```

- [ ] **步骤 2：编写默认跳过的 MySQL 集成测试**

```python
@pytest.mark.integration
def test_mysql_import_round_trip(mysql_database_url: str, valid_package_dir: Path) -> None:
    settings = DatabaseSettings.from_url(mysql_database_url)
    engine = create_engine_from_settings(settings)
    Base.metadata.create_all(engine)
    importer = MysqlPackageImporter(create_session_factory(engine))

    first = importer.import_path(valid_package_dir)
    second = importer.import_path(valid_package_dir)

    assert first.status == "new"
    assert second.status == "already_imported"
```

fixture 仅在设置 `RUN_MYSQL_INTEGRATION=1` 且存在 `DATABASE_URL` 时启用，否则 `pytest.skip()`；不得输出 URL。

- [ ] **步骤 3：运行默认测试并确认安全跳过**

运行：`cd backend && python -m pytest tests/integration/test_mysql_import.py -v`

预期：SKIPPED，不尝试连接数据库。

- [ ] **步骤 4：在用户 MySQL 配置可用时显式运行**

运行：`cd backend && RUN_MYSQL_INTEGRATION=1 python -m pytest tests/integration/test_mysql_import.py -v`

预期：PASS。若连接失败，如实记录错误，不修改用户数据库配置，不输出凭据。

### 任务 10：完整验证并受控爬取一份授权文档

**文件：**
- 修改：`backend/.env.example`
- 按验证发现仅修复本计划直接引入的问题。

- [ ] **步骤 1：安装项目开发依赖**

运行：`cd backend && python -m pip install -e ".[test]"`

预期：SQLAlchemy、PyMySQL 和项目测试依赖安装成功。若网络或权限受限，取得用户许可后重试，不改用未经确认的镜像。

- [ ] **步骤 2：运行定向测试**

运行：

```bash
cd backend && python -m pytest \
  tests/unit/test_package_models.py \
  tests/unit/test_package_validator.py \
  tests/unit/test_package_writer.py \
  tests/unit/test_offline_pipeline.py \
  tests/unit/test_config.py \
  tests/unit/test_database_repository.py \
  tests/unit/test_mysql_importer.py \
  tests/unit/test_cli.py -v
```

预期：全部 PASS。

- [ ] **步骤 3：运行完整测试和编译检查**

运行：

```bash
cd backend && python -m pytest -q
cd backend && python -m compileall -q app tests
```

预期：全部测试通过，编译命令退出码为 0。

- [ ] **步骤 4：运行敏感信息和文件长度检查**

检查 `app`、`tests`、`pyproject.toml` 和 `.env.example` 中不存在真实 API Key、密码或完整生产连接 URL；检查所有新增 Python 文件不超过 300 行。不要读取或输出现有 `.env` 全文。

- [ ] **步骤 5：确认授权来源后执行真实离线 Pipeline**

使用现有官方白名单中的一个明确授权法律页面，输出到用户指定的受控目录：

```bash
cd backend && python -m app.cli.build_ingestion_package \
  --crawl-id legal-smoke-001 \
  --source-id <existing-whitelist-id> \
  --source-url <authorized-official-url> \
  --title <official-document-title> \
  --output <controlled-artifact-root>
```

预期：生成一个通过 `validate_package()` 的最终数据包。执行前不得猜测来源 URL；必须从当前白名单配置中选择，并确认页面属于授权官方来源。

- [ ] **步骤 6：检查真实数据包质量**

仅输出统计和非敏感元数据：来源域名、内容哈希前缀、正文字符数、父块数、子块数、识别条文数。人工抽查开头、中部和末尾若干条文，确认标题、正文、条号和父子关系没有明显错位。

- [ ] **步骤 7：执行 MySQL 入库脚本**

在用户配置的 MySQL 可连接且目标库明确后运行：

```bash
cd backend && python -m app.cli.import_mysql --package <generated-package-path>
```

预期：首次为 `new` 或 `updated`，数据库中版本状态为 `awaiting_embedding`；再次执行同一 package 返回 `already_imported`。不得自行创建、删除或清空用户未明确授权的数据库。

- [ ] **步骤 8：运行最终独立代码审查**

调用 `requesting-code-review`，重点核查：数据包路径安全、事务边界、幂等竞争、凭据泄漏、父子引用完整性和现有采集规则是否被绕过。修复确认的 Critical/Important 后重新执行步骤 2 至步骤 4。

- [ ] **步骤 9：按证据报告结果**

明确报告：测试总数、跳过的集成测试、编译结果、真实爬取来源、数据包路径、MySQL 导入状态，以及任何因服务不可用而未执行的步骤。没有实际运行的步骤不得声称完成。
