# 劳动法真实 HTML 离线导入实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 subagent-driven-development（推荐）或 executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 将 `data/labor_law_raw/` 中已成功采集的真实 HTML 批量解析、清洗、条文分块，并输出可由 `PackageData` 重新读取的离线 JSONL 数据包。

**架构：** 新增一个薄的批处理入口负责遍历原始 HTML、提取来源元数据、调用既有 `parse_document()` 和 `chunk_document()`，再把每个页面组装成独立的 `PackageData` 输出目录。单个文件失败时记录错误并继续处理，其余页面仍生成结果；不在本计划中接入 Embedding、MySQL/Milvus、SPA/WAF 绕过或恢复全量测试体系。

**技术栈：** Python 3.11+、`pathlib`、`json`、既有 `app.ingest.parser`、`app.ingest.chunker`、`app.pipeline.package_models`、pytest。

**规格：** `docs/技术栈与架构文档.md`、`docs/需求文档.md`、`法律RAG-真实数据采集方案.md` 第 129-162、166-196、222-232 行。

## 全局约束

- 仅处理 `data/labor_law_raw/*.html` 和 `*.htm`，不猜测未知格式。
- 每个原始页面必须经过 `parse_document()`、`clean_text()`（由解析器调用）和 `chunk_document()`。
- 输出必须包含 `documents.jsonl`、`document_versions.jsonl`、`document_chunks.jsonl`、`crawl_records.jsonl` 四类 JSONL 文件。
- 输出数据必须可由 `PackageData.read_records()` 重新读取，且每个数据包至少包含一个 chunk。
- 单个文件解析失败或无法生成 chunk 时记录错误并继续，不伪造成功记录。
- 使用原始文件内容 SHA-256 作为 `content_hash`，使用稳定的 `source_url` 生成 `document_id`，使用文档 ID 与内容哈希生成 `version_id`。
- 不绕过登录、验证码、WAF 或反爬机制，不新增远程服务依赖。
- 现有仓库已删除 `backend/tests/` 下原测试文件；本计划只新增覆盖本功能的最小测试，不恢复无关测试。
- 当前目录不是 Git 仓库，因此不能执行 commit；每个任务以测试结果和工作区变更记录作为检查点。

---

### 任务 1：定义离线导入入口的行为契约

**文件：**
- 创建：`backend/tests/test_offline_ingest.py`
- 创建：`backend/app/ingest/offline_ingest.py`

- [ ] **步骤 1：编写失败的测试**

在 `backend/tests/test_offline_ingest.py` 中覆盖以下契约：

```python
from pathlib import Path

from app.ingest.offline_ingest import build_package, collect_raw_files
from app.pipeline.package_models import PackageData


def test_collect_raw_files_returns_sorted_html_files(tmp_path: Path) -> None:
    (tmp_path / "b.html").write_text("<p>第二条</p>", encoding="utf-8")
    (tmp_path / "a.htm").write_text("<p>第一条</p>", encoding="utf-8")
    (tmp_path / "ignore.txt").write_text("ignore", encoding="utf-8")

    assert collect_raw_files(tmp_path) == [tmp_path / "a.htm", tmp_path / "b.html"]


def test_build_package_contains_parse_and_chunk_results(tmp_path: Path) -> None:
    source = tmp_path / "law.html"
    source.write_text("<html><body><h1>劳动条例</h1><p>第一条 适用范围。</p></body></html>", encoding="utf-8")

    package = build_package(source, source_url="https://example.test/law.html", source_id="example")

    assert package.document.source_url == "https://example.test/law.html"
    assert package.version.raw_file_path == str(source)
    assert package.version.cleaned_content
    assert package.chunks
    assert package.crawl.http_status == 200
    PackageData(package.document, package.version, package.chunks, package.crawl).write_records(tmp_path / "out")
    assert PackageData.read_records(tmp_path / "out").chunks
```

- [ ] **步骤 2：运行测试验证失败**

运行：`cd backend && python -m pytest tests/test_offline_ingest.py -q`

预期：FAIL，报错 `ModuleNotFoundError` 或 `ImportError`，因为 `app.ingest.offline_ingest` 尚未创建。

- [ ] **步骤 3：编写最少实现代码**

在 `backend/app/ingest/offline_ingest.py` 中定义：

```python
from pathlib import Path


def collect_raw_files(raw_root: Path) -> list[Path]:
    return sorted(
        path for path in raw_root.iterdir()
        if path.is_file() and path.suffix.lower() in {".html", ".htm"}
    )


def build_package(source_path: Path, source_url: str, source_id: str):
    ...
```

实现 `build_package()` 时读取原始字节计算 SHA-256，调用 `parse_document(source_path)` 与 `chunk_document()`，将 `DocumentEntry`、`VersionEntry`、`ChunkEntry`、`CrawlEntry` 组装成 `PackageData`。初始 `http_status` 固定为 200，仅用于表示这些文件来自已验证的成功采集结果；不得把函数用于未验证 URL。

- [ ] **步骤 4：运行测试验证通过**

运行：`cd backend && python -m pytest tests/test_offline_ingest.py -q`

预期：PASS，所有契约测试通过。

- [ ] **步骤 5：工作区检查**

运行：`git status --short`；由于当前目录不是 Git 仓库，预期输出类似 `fatal: not a git repository`。改用 `git diff --no-index NUL NUL` 不适用 Windows 文件夹；直接记录新增文件路径和测试结果，不执行 commit。

---

### 任务 2：增加批处理命令和错误隔离

**文件：**
- 修改：`backend/app/ingest/offline_ingest.py`
- 创建：`backend/app/ingest/run_offline_ingest.py`
- 修改：`backend/tests/test_offline_ingest.py`

- [ ] **步骤 1：编写失败的测试**

追加批处理行为测试：

```python
from app.ingest.offline_ingest import process_raw_directory


def test_process_raw_directory_writes_packages_and_continues_after_failure(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    output_root = tmp_path / "processed"
    raw_root.mkdir()
    (raw_root / "good.html").write_text("<p>第一条 有效内容。</p>", encoding="utf-8")
    (raw_root / "bad.html").write_bytes(b"\\xff\\xfe")

    result = process_raw_directory(raw_root, output_root, {"good.html": "https://example.test/good"})

    assert result.success_count == 1
    assert result.failure_count == 1
    assert (output_root / "good" / "documents.jsonl").is_file()
    assert result.errors[0].filename == "bad.html"
```

- [ ] **步骤 2：运行测试验证失败**

运行：`cd backend && python -m pytest tests/test_offline_ingest.py::test_process_raw_directory_writes_packages_and_continues_after_failure -q`

预期：FAIL，因为 `process_raw_directory()`、批处理结果类型和错误记录尚未实现。

- [ ] **步骤 3：编写最少实现代码**

在 `offline_ingest.py` 中新增不可变结果类型，至少包含 `success_count`、`failure_count`、`errors`；实现 `process_raw_directory(raw_root, output_root, source_urls)`：

- 使用 `collect_raw_files()` 获取稳定排序文件；
- 从 `source_urls[filename]` 获取已验证 URL，缺失时记录失败；
- 调用 `build_package()`；
- 以安全的稳定目录名写入对应包；
- 使用 `PackageData.write_records()` 落盘；
- 捕获 `OSError`、`UnicodeError`、`ValueError` 和 `PackageFormatError`，转换为结构化错误并继续下一个文件。

在 `run_offline_ingest.py` 中提供 `python -m app.ingest.run_offline_ingest` 入口，默认读取项目根目录下的 `data/labor_law_raw`，输出到 `data/labor_law_processed`，并从明确的已验证 URL 映射生成结果。命令输出成功数、失败数和失败文件名；存在失败时返回非零退出码。

- [ ] **步骤 4：运行测试验证通过**

运行：`cd backend && python -m pytest tests/test_offline_ingest.py -q`

预期：PASS。

- [ ] **步骤 5：检查 CLI 帮助和错误退出**

运行：`cd backend && python -m app.ingest.run_offline_ingest --help`

预期：显示输入目录、输出目录和 URL 清单参数；不访问远程站点。

---

### 任务 3：用 3 个真实采集文件执行离线导入

**文件：**
- 修改：`法律RAG-真实数据采集方案.md`
- 创建：`data/labor_law_processed/` 下由命令生成的 JSONL 数据包
- 不修改：`data/labor_law_raw/` 中现有原始文件

- [ ] **步骤 1：确认输入文件和 URL 映射**

运行：`python -c "from pathlib import Path; print('\\n'.join(str(p) for p in Path('data/labor_law_raw').glob('*.html')))"`

预期：至少看到当前已记录的 3 个真实 HTML 文件；若文件数多于 3 个，只处理方案文档中明确列出的已验证来源，其他文件记录为未映射失败，不删除任何文件。

- [ ] **步骤 2：执行离线导入命令**

运行：`cd backend && python -m app.ingest.run_offline_ingest`

预期：3 个已验证页面成功生成数据包；未映射或损坏文件只进入失败报告，不伪造成功。

- [ ] **步骤 3：重新读取所有成功数据包**

运行：`cd backend && python -c "from pathlib import Path; from app.pipeline.package_models import PackageData; roots=Path('../data/labor_law_processed').iterdir(); packages=[PackageData.read_records(p) for p in roots if p.is_dir()]; print(len(packages), sum(len(p.chunks) for p in packages))"`

预期：所有成功目录均能被 `PackageData.read_records()` 读取，chunk 总数大于 0。

- [ ] **步骤 4：抽查真实正文和条文识别**

运行：`cd backend && python -c "from pathlib import Path; from app.pipeline.package_models import PackageData; [print(p.name, len(PackageData.read_records(p).chunks), PackageData.read_records(p).chunks[0].article_number) for p in Path('../data/labor_law_processed').iterdir() if p.is_dir()]"`

预期：输出每个来源的 chunk 数；司法部法规页面应出现可识别条文或文档级 chunk，法院案例页面不得因没有标准条文号而被当作失败。

- [ ] **步骤 5：更新采集方案记录**

在“真实采集结果”后增加离线导入结果：处理命令、成功包数量、失败文件数量、人工抽查结论；明确原始 HTML 未删除，旧哈希仍保留。

---

### 任务 4：执行完整验证并形成检查点

**文件：**
- 修改：`docs/测试文档.md`

- [ ] **步骤 1：运行单元测试**

运行：`cd backend && python -m pytest tests/test_offline_ingest.py -q`

预期：全部 PASS。

- [ ] **步骤 2：运行模块编译检查**

运行：`python -m compileall backend/app/ingest backend/app/pipeline`

预期：退出码 0，无语法错误。

- [ ] **步骤 3：验证 JSONL 格式**

运行：`cd backend && python -c "from pathlib import Path; from app.pipeline.package_models import PackageData; roots=[p for p in Path('../data/labor_law_processed').iterdir() if p.is_dir()]; assert roots; [PackageData.read_records(p) for p in roots]; print(f'validated {len(roots)} packages')"`

预期：打印已验证的数据包数量并退出码 0。

- [ ] **步骤 4：更新测试文档**

记录新增命令、单元测试结果、真实数据包验证结果和已知限制：当前只验证 3 个成功来源；未接入 Embedding、MySQL/Milvus；未处理 SPA/WAF 与反爬来源。

- [ ] **步骤 5：最终工作区检查**

运行：`git status --short`。

预期：确认新增和修改文件清单；当前目录不是 Git 仓库，因此不创建 commit、不声称存在提交记录。

## 计划自检

- **规格覆盖度：** 解析、清洗、条文分块、JSONL 四文件输出、单文件失败隔离、真实数据验证、人工抽查记录分别由任务 1-4 覆盖。
- **占位符扫描：** 未使用 `TODO`、`待定`、`后续实现` 或“适当处理”等无具体动作描述。
- **类型一致性：** 任务 1 定义 `build_package()` 与 `collect_raw_files()`；任务 2 定义 `process_raw_directory()` 与结果字段；任务 3、4 只调用前述接口及既有 `PackageData`。
- **范围检查：** 本计划只覆盖离线导入闭环，不包含 Embedding、数据库入库、自动链接发现和受保护站点访问。
