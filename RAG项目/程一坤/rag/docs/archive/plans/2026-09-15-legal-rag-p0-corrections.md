# 法律 RAG P0 修复实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 subagent-driven-development（推荐）或 executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 修复测试无法完整收集、首次采集被限速拦截、HTML 丢失条文边界、法律条号误切和跨文档 `chunk_id` 冲突五类 P0/P1 基础问题。

**架构：** 保持现有 `crawler`、`ingest` 和单元测试分层，仅通过测试配置、共享测试替身、请求器 robots 限速边界、HTML 文本提取边界和分块 ID 生成接口进行最小修改。当前阶段不引入 MySQL、Redis、Milvus 或新的第三方依赖；后续数据库文档 ID 可通过显式参数接入分块器。

**技术栈：** Python、pytest、`urllib`、`html.parser.HTMLParser`、正则表达式、现有 dataclass 和依赖注入接口。

**规格：** `docs/superpowers/specs/2026-09-15-legal-rag-p0-corrections-design.md`

## 全局约束

- 在实现代码前先编写能复现缺陷的测试，并确认旧实现失败。
- 测试数据只保存在内存或 pytest 临时目录，不写入正式 `data` 目录。
- 不访问真实官方站点；采集测试使用注入的响应替身。
- 不新增 MySQL、Redis、Milvus 或第三方 HTTP 依赖。
- 保留官方来源白名单、robots 校验和来源限速约束。
- 所有用户可见说明、测试注释和新增代码注释使用简体中文；技术标识符保留原文。
- 每个代码文件保持不超过 300 行；删除复述式注释时只处理本次修改涉及的文件。
- 不输出或硬编码 API Key。

---

### 任务 1：修复 pytest 收集和共享测试替身

**文件：**
- 创建：`backend/pytest.ini`
- 创建：`backend/tests/unit/conftest.py`
- 修改：`backend/tests/unit/test_requester.py`
- 修改：`backend/tests/unit/test_runner.py`

- [ ] **步骤 1：编写收集回归基线**

在 `test_runner.py` 中移除对 `test_requester.py` 的设计依赖前，先保留现状并运行完整收集，记录当前 collection error；新增的共享替身测试应验证 `FakeResponse` 可从 `conftest.py` 注入使用。

- [ ] **步骤 2：运行测试确认旧问题**

运行：

```bash
cd C:\Users\92842\Desktop\rag\backend
python -m pytest tests/unit --collect-only -q
```

预期：当前实现因 `tests.unit.test_requester` 导入链或 pytest 路径配置产生 collection error。

- [ ] **步骤 3：创建 pytest 配置和共享替身**

创建 `backend/pytest.ini`：

```ini
[pytest]
pythonpath = .
testpaths = tests
```

将 `FakeResponse` 放入 `tests/unit/conftest.py`，保留 `__enter__`、`__exit__`、`getcode()`、`read()`、`headers` 和可配置状态码。将白名单构造辅助也移入 `conftest.py`，通过 pytest fixture 提供 `official_whitelist`。

- [ ] **步骤 4：移除测试文件互相导入**

修改 `test_requester.py` 和 `test_runner.py`，删除从 `test_requester.py` 导入 `FakeResponse`、白名单辅助的语句，改为使用 `conftest.py` 提供的 fixture 或本文件独立构造；不得新增测试文件间 import。

- [ ] **步骤 5：运行收集和单元测试**

运行：

```bash
python -m pytest tests/unit --collect-only -q
python -m pytest tests/unit -q
```

预期：不再出现 collection error；若行为测试暂时失败，记录具体失败并在对应任务中修复，不修改无关模块。

- [ ] **步骤 6：提交本任务**

```bash
git add backend/pytest.ini backend/tests/unit/conftest.py backend/tests/unit/test_requester.py backend/tests/unit/test_runner.py
git commit -m "test: fix unit test collection setup"
```

---

### 任务 2：修复 robots 请求占用页面限速窗口

**文件：**
- 修改：`backend/app/crawler/requester.py:59-161`
- 修改：`backend/tests/unit/test_requester.py`

- [ ] **步骤 1：新增首次采集失败回归测试**

新增测试使用 `minimum_interval_seconds=3.0`、可变模拟时钟和两条内存响应，断言一次 `fetch()` 依次请求 robots 和页面，并返回：

```python
assert result.content == "temporary official page"
assert result.record.crawl_status == "success"
assert requested_urls == [
    "https://flk.npc.gov.cn/robots.txt",
    "https://flk.npc.gov.cn/detail?id=document",
]
```

- [ ] **步骤 2：运行回归测试确认失败**

运行：

```bash
python -m pytest tests/unit/test_requester.py -k first_fetch -v
```

预期：旧实现失败，错误为来源访问间隔尚未达到，页面 URL 不会被请求。

- [ ] **步骤 3：修改 robots 限速边界**

在 `_get_robots_policy()` 中保留 robots 请求本身的网络访问和缓存，但删除对同一个页面限速器的 `can_request()` 检查与 `record_request()` 调用。保留 `fetch()` 中页面请求前的 `can_request()` 和实际页面请求前的 `record_request()`，确保页面请求仍按来源间隔限制。

更新原有 `test_fetch_reuses_cached_robots_after_rate_limit_window`：首次调用应成功；第二次调用在模拟时钟未推进时应返回限速失败；推进到 `3.0` 后再次调用才成功，并断言 robots 只请求一次。

- [ ] **步骤 4：运行采集相关测试**

运行：

```bash
python -m pytest tests/unit/test_requester.py tests/unit/test_runner.py -q
```

预期：首次正间隔采集成功，页面连续访问仍受限，runner 测试通过。

- [ ] **步骤 5：提交本任务**

```bash
git add backend/app/crawler/requester.py backend/tests/unit/test_requester.py
 git commit -m "fix: keep robots requests outside page rate limit"
```

---

### 任务 3：修复法律条文边界并稳定生成跨文档块 ID

**文件：**
- 修改：`backend/app/ingest/chunker.py:46-159`
- 修改：`backend/tests/unit/test_chunker.py`

- [ ] **步骤 1：新增条号引用误切测试**

新增 `ParsedDocument` 测试文本：

```text
第四十七条 用人单位应当依照第五十条规定办理手续。
第四十八条 违反本法第八十七条规定的，依法承担责任。
第四十九条 本法自公布之日起施行。
```

调用 `chunk_document()` 后筛选父块，断言 `article_no` 依次为：

```python
["第四十七条", "第四十八条", "第四十九条"]
```

- [ ] **步骤 2：新增跨文档 ID 稳定性测试**

使用两个不同 `source_path` 的 `ParsedDocument`，断言两次结果的 `chunk_id` 集合没有交集；对同一文档重复调用，断言 ID 集合完全一致。断言父块和子块分别符合 `{document_key}-a1` 与 `{document_key}-a1-c1` 的结构，不再使用无文档前缀的 `parent-1`。

- [ ] **步骤 3：运行测试确认旧实现失败**

运行：

```bash
python -m pytest tests/unit/test_chunker.py -k "article_reference or chunk_id" -v
```

预期：条号引用测试会多生成父块，跨文档测试会发现 ID 集合冲突。

- [ ] **步骤 4：实现行首条文匹配**

将 `_split_articles()` 的正则改为多行模式，仅匹配行首并要求条号后有普通空白或全角空格：

```python
re.compile(
    r"(?m)^(?=\s*(第[一二三四五六七八九十百千万零〇两\d]+条(?:之[一二三四五六七八九十\d]+)?)[\s　])"
)
```

保留当前无条文号时的文档级回退逻辑。

- [ ] **步骤 5：实现文档标识前缀**

为 `chunk_document()` 增加可选 `document_id: str | None = None` 参数；传入时直接使用该稳定标识，未传入时对规范化 `source_path` 计算短的确定性哈希前缀。父块使用 `{document_key}-a{article_index}`，子块使用 `{document_key}-a{article_index}-c{paragraph_index}`。不得使用随机 UUID，以保证重复分块稳定。

- [ ] **步骤 6：运行分块测试**

运行：

```bash
python -m pytest tests/unit/test_chunker.py -q
```

预期：条号引用、单换行段落、父子关联、无条文号回退和跨文档 ID 测试全部通过。

- [ ] **步骤 7：提交本任务**

```bash
git add backend/app/ingest/chunker.py backend/tests/unit/test_chunker.py
git commit -m "fix: preserve article boundaries and unique chunk ids"
```

---

### 任务 4：修复 HTML 块级换行并接入条文回归

**文件：**
- 修改：`backend/app/ingest/parser.py:20-100`
- 修改：`backend/tests/unit/test_parser.py`

- [ ] **步骤 1：新增 HTML 换行失败测试**

新增测试将以下内容交给 HTML 解析器：

```html
<div><p>第四十七条 第一项义务。</p><p>第四十八条 第二项义务。</p></div>
```

断言解析结果包含换行，且每个条文位于独立行；禁止只断言两个条文字符串都存在，因为那无法复现边界丢失。

- [ ] **步骤 2：运行测试确认旧实现失败**

运行：

```bash
python -m pytest tests/unit/test_parser.py -k html -v
```

预期：旧实现将两个段落拼成一行，换行断言失败。

- [ ] **步骤 3：实现块级标签换行**

在 `_HtmlTextExtractor` 中维护块级标签集合：

```python
BLOCK_TAGS = {
    "p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6",
    "tr", "table", "section", "article",
}
```

在块级标签开始或结束时追加换行；行内标签不追加额外换行。`get_text()` 使用 `"".join(self._text_parts)`，把连续换行压缩为一个并去除首尾空白。避免改变 TXT、Markdown 和 PDF 的清洗契约。

- [ ] **步骤 4：增加 HTML 到条文切分的联测**

在解析测试中将解析结果构造为 `ParsedDocument` 并交给 `chunk_document()`，断言父块条号为 `第四十七条`、`第四十八条`，确保换行修复真正服务于法律分块，而不是只满足字符串测试。

- [ ] **步骤 5：运行解析和分块测试**

运行：

```bash
python -m pytest tests/unit/test_parser.py tests/unit/test_chunker.py -q
```

预期：HTML、TXT、Markdown、PDF 现有测试以及条文联测全部通过。

- [ ] **步骤 6：提交本任务**

```bash
git add backend/app/ingest/parser.py backend/tests/unit/test_parser.py
 git commit -m "fix: preserve HTML block boundaries"
```

---

### 任务 5：完成全量验证并检查变更范围

**文件：**
- 检查：`backend/app/crawler/`
- 检查：`backend/app/ingest/`
- 检查：`backend/tests/unit/`
- 检查：`backend/pytest.ini`

- [ ] **步骤 1：运行完整单元测试**

运行：

```bash
cd C:\Users\92842\Desktop\rag\backend
python -m pytest tests/unit -q
```

预期：collection 无错误，全部测试通过。

- [ ] **步骤 2：检查代码文件大小和敏感信息**

运行：

```bash
powershell.exe -NoProfile -Command 'Get-ChildItem app,tests -Recurse -Filter *.py | Where-Object { $_.Length -gt 300KB } | Select-Object FullName,Length'
rg -n "sk-[A-Za-z0-9]|API_KEY\s*=\s*['\"]" app tests
```

预期：没有新增超过项目约束的代码文件，没有 API Key 硬编码。

- [ ] **步骤 3：检查变更范围和差异**

运行：

```bash
git status --short
git diff --check HEAD~4..HEAD
```

预期：变更只涉及本计划列出的测试配置、crawler requester、chunker、HTML parser 及对应测试；无临时法律数据和无关重构。

- [ ] **步骤 4：记录真实验证结果**

保存 pytest 实际通过数量和任何非阻断警告；若测试失败，不得声称完成，先按失败输出修复或明确报告阻塞原因。

---
