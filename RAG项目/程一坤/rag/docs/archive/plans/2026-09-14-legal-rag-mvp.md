# 法律 RAG MVP 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 subagent-driven-development（推荐）或 executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 构建一个可在 Ubuntu 服务器通过 Docker Compose 运行的劳动法法律知识助手 MVP，支持 QQ 邮箱认证、官方法规采集审核、文档索引、混合检索、引用校验和 Web 流式问答。

**架构：** 使用 FastAPI 模块化单体承载认证、管理员、知识库、检索和聊天 API；使用 Celery Worker 执行采集、解析、分块、Embedding 和索引任务。Next.js Web 前端通过 Nginx 访问 API 和 SSE，MySQL 保存业务元数据，Redis 保存验证码、任务状态和短期会话，Milvus 保存法律知识向量。

**技术栈：** Python 3.11+、FastAPI、Pydantic、SQLAlchemy、Alembic、MySQL、Redis、Celery、Milvus、PyMuPDF、pdfplumber、MinerU、PaddleOCR/PaddleOCR-VL 适配器、Next.js、TypeScript、Docker Compose、Nginx、pytest。

**规格：** `docs/需求文档.md`、`docs/技术栈与架构文档.md`、`docs/接口文档.md`、`docs/测试文档.md`、`docs/部署文档.md`

## 全局约束

- 首期只提供一个预设「法律知识助手」。
- 法域限定为中国大陆全国层面，专题限定为劳动法。
- 首期法源为全国性法律法规和司法解释，不纳入案例材料、合同模板和普通裁判文书。
- 官方站点采集必须使用白名单、遵守 `robots.txt` 和访问频率限制，不得绕过登录或验证码。
- 采集结果必须经过管理员人工审核，审核通过且索引成功后才能参与检索。
- 仅允许 `@qq.com` 和 `@foxmail.com` 注册，验证码通过 QQ SMTP 发送。
- 普通用户只能使用预设角色并管理自己的会话；管理员负责采集、审核、发布和索引。
- 首期只启用 Redis 短期记忆，不自动写入 Milvus 长期记忆。
- LLM、Embedding 和 Reranker 使用统一适配器；供应商暂不锁定，自动化测试使用 Mock。
- 首期只要求在 Ubuntu 服务器使用 Docker Compose 成功部署。
- 原始文件、解析结果、日志和备份使用 Ubuntu 服务器本地持久化目录。
- 目标规模为约 20 个并发问答用户、1 万份以内法律文档、单文件不超过 50 MB。
- 问答首 Token `p95` 不超过 3 秒，完整回答 `p95` 不超过 10 秒，检索阶段 `p95` 不超过 2 秒。
- 所有受保护接口鉴权；日志不得记录密码、Token、API Key 和完整敏感内容。
- 用户输入、采集文档和检索内容均视为不可信数据，不能覆盖系统安全规则。

---

### 任务 1：建立后端项目骨架与可测试配置

**文件：**
- 创建：`backend/app/main.py`、`backend/app/core/config.py`、`backend/app/core/logging.py`
- 创建：`backend/pyproject.toml`、`backend/requirements.txt`、`backend/.env.example`
- 创建：`backend/tests/test_health.py`、`backend/tests/conftest.py`

- [ ] **步骤 1：编写失败的健康检查测试**

```python
from fastapi.testclient import TestClient
from app.main import app


def test_live_health_check_reports_process_alive():
    response = TestClient(app).get("/health/live")
    assert response.status_code == 200
    assert response.json()["code"] == 0
    assert response.json()["data"]["status"] == "alive"
```

- [ ] **步骤 2：运行测试验证失败**

运行：`cd backend && pytest tests/test_health.py -v`
预期：FAIL，报错 `ModuleNotFoundError` 或 `/health/live` 路由不存在。

- [ ] **步骤 3：编写最少实现代码**

实现 `FastAPI` 应用、统一响应模型、`GET /health/live` 和环境变量配置；日志使用结构化字段输出 `request_id`，不输出敏感配置。

- [ ] **步骤 4：运行测试验证通过**

运行：`cd backend && pytest tests/test_health.py -v`
预期：PASS。

- [ ] **步骤 5：检查基础格式**

运行：`cd backend && python -m compileall app && python -m pytest -q`
预期：无语法错误，已有测试全部通过。

### 任务 2：实现 MySQL 数据模型、迁移与权限依赖

**文件：**
- 创建：`backend/app/db/session.py`、`backend/app/db/models.py`、`backend/app/db/migrations/`
- 创建：`backend/app/api/dependencies.py`
- 创建：`backend/tests/test_models.py`、`backend/tests/test_permissions.py`

- [ ] **步骤 1：编写失败的模型和越权测试**

```python
def test_session_query_is_scoped_to_authenticated_user(session_factory):
    user_session = create_session(user_id="user-1", character_id="legal-assistant")
    assert find_session_for_user(session_factory, user_session.id, "user-2") is None


def test_regular_user_cannot_review_document(authenticated_user):
    assert can_review_document(authenticated_user) is False
```

- [ ] **步骤 2：运行测试验证失败**

运行：`cd backend && pytest tests/test_models.py tests/test_permissions.py -v`
预期：FAIL，模型、查询函数和权限函数尚未定义。

- [ ] **步骤 3：编写最少实现代码**

建立用户、验证码、角色、知识库、法律文档、文档版本、采集任务、索引任务、会话、消息和反馈模型；所有资源查询必须同时带资源 ID 与当前用户或管理员权限条件。添加 Alembic 初始迁移。

- [ ] **步骤 4：运行测试验证通过**

运行：`cd backend && pytest tests/test_models.py tests/test_permissions.py -v`
预期：PASS。

- [ ] **步骤 5：验证迁移**

运行：`cd backend && alembic upgrade head`
预期：迁移成功创建首期业务表。

### 任务 3：实现 QQ 邮箱注册、登录和密码重置

**文件：**
- 创建：`backend/app/auth/service.py`、`backend/app/auth/router.py`、`backend/app/auth/schemas.py`、`backend/app/auth/mailer.py`
- 修改：`backend/app/main.py`、`backend/app/core/config.py`
- 创建：`backend/tests/test_auth.py`、`backend/tests/test_mailer.py`

- [ ] **步骤 1：编写失败的认证测试**

```python
def test_register_rejects_non_qq_email(client):
    response = client.post("/api/v1/auth/register/code", json={"email": "user@example.com"})
    assert response.status_code == 400
    assert response.json()["code"] == 40000


def test_register_code_can_be_used_once(client, fake_mailer):
    client.post("/api/v1/auth/register/code", json={"email": "user@qq.com"})
    code = fake_mailer.last_code
    first = client.post("/api/v1/auth/register", json={"email": "user@qq.com", "code": code, "password": "StrongPass123!"})
    second = client.post("/api/v1/auth/register", json={"email": "user@qq.com", "code": code, "password": "StrongPass123!"})
    assert first.status_code == 200
    assert second.status_code == 400
```

- [ ] **步骤 2：运行测试验证失败**

运行：`cd backend && pytest tests/test_auth.py tests/test_mailer.py -v`
预期：FAIL，认证路由和 QQ 邮箱校验尚未实现。

- [ ] **步骤 3：编写最少实现代码**

实现邮箱域名校验、验证码生成与 Redis TTL、QQ SMTP 适配器、密码哈希、JWT access token、登录、注册和密码重置接口。验证码只能使用一次，发送和校验均限流；错误响应不泄露账号是否存在。

- [ ] **步骤 4：运行测试验证通过**

运行：`cd backend && pytest tests/test_auth.py tests/test_mailer.py -v`
预期：PASS。

- [ ] **步骤 5：验证认证安全行为**

运行：`cd backend && pytest -q -k 'auth or permission'`
预期：PASS，密码、验证码和 Token 不出现在日志测试输出中。

### 任务 4：实现法律角色、知识库与管理员审核流程

**文件：**
- 创建：`backend/app/characters/`、`backend/app/knowledge/`、`backend/app/admin/`
- 创建：`backend/tests/test_character_scope.py`、`backend/tests/test_review_workflow.py`
- 修改：`backend/app/main.py`、`backend/app/db/models.py`

- [ ] **步骤 1：编写失败的角色和审核测试**

```python
def test_regular_user_can_only_see_published_default_character(client, user_token):
    response = client.get("/api/v1/characters", headers=auth(user_token))
    assert [item["type"] for item in response.json()["data"]] == ["legal_assistant"]


def test_unreviewed_document_is_not_searchable(search_service, pending_document):
    assert search_service.search("解除劳动合同", [pending_document.knowledge_base_id]) == []
```

- [ ] **步骤 2：运行测试验证失败**

运行：`cd backend && pytest tests/test_character_scope.py tests/test_review_workflow.py -v`
预期：FAIL，预设角色、审核状态和过滤逻辑尚未实现。

- [ ] **步骤 3：编写最少实现代码**

初始化唯一预设法律知识助手；实现知识库创建、文档元数据、管理员查询待审核文档、审核通过/拒绝和版本发布接口。文档状态至少支持 `collected`、`pending_review`、`approved`、`rejected`、`indexed` 和 `failed`；检索只读取已审核且已生效版本。

- [ ] **步骤 4：运行测试验证通过**

运行：`cd backend && pytest tests/test_character_scope.py tests/test_review_workflow.py -v`
预期：PASS。

- [ ] **步骤 5：验证资源隔离**

运行：`cd backend && pytest -q -k 'scope or review or permission'`
预期：PASS，普通用户无法调用管理员审核接口。

### 任务 5：实现官方站点白名单采集和异步任务

**文件：**
- 创建：`backend/app/crawler/allowlist.py`、`backend/app/crawler/client.py`、`backend/app/crawler/parser.py`、`backend/app/crawler/tasks.py`
- 创建：`backend/tests/test_crawler_safety.py`、`backend/tests/test_crawler_tasks.py`
- 修改：`backend/app/knowledge/`

- [ ] **步骤 1：编写失败的采集合规测试**

```python
def test_crawler_rejects_url_outside_allowlist(crawler):
    with pytest.raises(UrlNotAllowed):
        crawler.fetch("https://unknown.example/legal/1")


def test_crawler_deduplicates_same_content(crawler, page_factory):
    first = crawler.ingest(page_factory(content="law text"))
    second = crawler.ingest(page_factory(content="law text"))
    assert first.document_id == second.document_id
    assert second.is_new_version is False
```

- [ ] **步骤 2：运行测试验证失败**

运行：`cd backend && pytest tests/test_crawler_safety.py tests/test_crawler_tasks.py -v`
预期：FAIL，白名单、robots、限速和哈希去重尚未实现。

- [ ] **步骤 3：编写最少实现代码**

实现可配置官方来源白名单、URL 规范化、`robots.txt` 检查、每来源访问频率限制、请求超时、内容哈希、增量版本识别和 Celery 任务状态。采集任务只产生 `pending_review` 文档，不自动发布；不支持登录绕过、验证码绕过或未授权来源。

- [ ] **步骤 4：运行测试验证通过**

运行：`cd backend && pytest tests/test_crawler_safety.py tests/test_crawler_tasks.py -v`
预期：PASS。

- [ ] **步骤 5：验证任务幂等性**

运行：`cd backend && pytest -q -k 'crawler or task'`
预期：PASS，重复触发相同 URL 和内容不会生成重复有效文档。

### 任务 6：实现 PDF/OCR/表格解析、分块和 Milvus 索引

**文件：**
- 创建：`backend/app/documents/parsers/`、`backend/app/documents/chunking.py`、`backend/app/documents/indexing.py`、`backend/app/vectorstore/milvus.py`
- 创建：`backend/tests/test_parsers.py`、`backend/tests/test_chunking.py`、`backend/tests/test_indexing.py`

- [ ] **步骤 1：编写失败的解析和分块测试**

```python
def test_pdf_parser_preserves_page_and_title_path(text_pdf):
    pages = parse_document(text_pdf)
    assert pages[0].page == 1
    assert pages[0].content


def test_title_chunk_keeps_article_number(article_text):
    chunks = chunk_document(article_text, strategy="title")
    assert any(chunk.article_no == "第八十七条" for chunk in chunks)
```

- [ ] **步骤 2：运行测试验证失败**

运行：`cd backend && pytest tests/test_parsers.py tests/test_chunking.py tests/test_indexing.py -v`
预期：FAIL，解析器、统一文档结构和分块器尚未实现。

- [ ] **步骤 3：编写最少实现代码**

实现普通 PDF 使用 PyMuPDF、表格使用 `pdfplumber`、复杂版式使用 MinerU、扫描 PDF 和图片使用 PaddleOCR/PaddleOCR-VL 适配器；统一输出文档 ID、页码、标题路径、内容、表格、来源、解析器和版本。实现固定长度、句子、段落、标题和语义策略，首期默认标题分块；保留法规名称、条文号、法域、生效日期、失效日期和来源。

- [ ] **步骤 4：运行测试验证通过**

运行：`cd backend && pytest tests/test_parsers.py tests/test_chunking.py tests/test_indexing.py -v`
预期：PASS。

- [ ] **步骤 5：验证索引失败不发布**

运行：`cd backend && pytest -q -k 'parser or chunk or index'`
预期：PASS，Embedding 或 Milvus 写入失败时文档保持不可检索状态。

### 任务 7：实现法律混合检索、引用校验和 SSE 问答

**文件：**
- 创建：`backend/app/rag/query.py`、`backend/app/rag/retrieval.py`、`backend/app/rag/prompt.py`、`backend/app/rag/postprocess.py`、`backend/app/chat/router.py`
- 创建：`backend/app/models/adapters.py`
- 创建：`backend/tests/test_retrieval.py`、`backend/tests/test_citations.py`、`backend/tests/test_chat_stream.py`

- [ ] **步骤 1：编写失败的检索和引用测试**

```python
def test_retrieval_filters_expired_law(retriever, expired_chunk, active_chunk):
    results = retriever.search("解除劳动合同", as_of_date=date(2026, 9, 14))
    assert expired_chunk not in results
    assert active_chunk in results


def test_citation_validator_rejects_unknown_chunk():
    with pytest.raises(InvalidCitation):
        validate_citations("依据[chunk-unknown]", {"chunk-001"})
```

- [ ] **步骤 2：运行测试验证失败**

运行：`cd backend && pytest tests/test_retrieval.py tests/test_citations.py tests/test_chat_stream.py -v`
预期：FAIL，检索、适用时间过滤、引用校验和 SSE 尚未实现。

- [ ] **步骤 3：编写最少实现代码**

实现问题标准化、Query 改写接口、BM25 与 Milvus 向量结果合并去重、权限/法域/文书类型/生效状态过滤、Reranker 适配器、上下文 token 限制和法规引用校验。实现 Mock LLM 流式适配器、短期记忆 Redis 读写、法律风险提示、无依据拒答和 `message_start`、`token`、`citation`、`message_end`、`error` SSE 事件。

- [ ] **步骤 4：运行测试验证通过**

运行：`cd backend && pytest tests/test_retrieval.py tests/test_citations.py tests/test_chat_stream.py -v`
预期：PASS。

- [ ] **步骤 5：验证安全边界**

运行：`cd backend && pytest -q -k 'retrieval or citation or chat'`
预期：PASS，过期法规、未授权块、未知引用和低相关性问题不会生成确定性法律结论。

### 任务 8：实现 Next.js Web 前端

**文件：**
- 创建：`frontend/app/`、`frontend/components/`、`frontend/lib/api.ts`、`frontend/package.json`、`frontend/next.config.js`
- 创建：`frontend/tests/`
- 修改：`README.md`

- [ ] **步骤 1：编写失败的页面测试**

```typescript
import { render, screen } from "@testing-library/react";
import LoginPage from "@/app/login/page";

it("renders QQ email login fields", () => {
  render(<LoginPage />);
  expect(screen.getByLabelText("QQ 邮箱")).toBeInTheDocument();
  expect(screen.getByLabelText("密码")).toBeInTheDocument();
});
```

- [ ] **步骤 2：运行测试验证失败**

运行：`cd frontend && npm test -- --runInBand`
预期：FAIL，Next.js 页面和测试配置尚未创建。

- [ ] **步骤 3：编写最少实现代码**

实现登录、注册验证码、注册、密码重置、角色主页、会话列表、法律问答页、SSE token 拼接、引用展示和管理员采集/审核页面。前端只展示预设法律知识助手；API 地址由环境变量配置。`next.config.js` 设置 `output: "standalone"`，并为流式请求关闭代理缓冲所需的响应头。

- [ ] **步骤 4：运行测试验证通过**

运行：`cd frontend && npm test -- --runInBand && npm run build`
预期：组件测试和 Next.js 生产构建均通过。

- [ ] **步骤 5：验证前后端接口契约**

运行：`cd frontend && npm run lint`
预期：无 lint 错误，登录、聊天和管理员页面使用接口文档定义的路径及字段。

### 任务 9：完成 Docker Compose、Nginx 和 Ubuntu 部署脚本

**文件：**
- 创建：`docker-compose.yml`、`backend/Dockerfile`、`frontend/Dockerfile`、`nginx/nginx.conf`
- 创建：`scripts/install.sh`、`scripts/deploy.sh`、`scripts/run.sh`、`scripts/shutdown.sh`、`scripts/health-check.sh`、`scripts/backup.sh`
- 修改：`docs/部署文档.md`、`.env.example`
- 创建：`tests/test_deployment_files.py`

- [ ] **步骤 1：编写失败的部署文件测试**

```python
def test_compose_defines_required_services():
    services = load_compose()["services"]
    assert {"frontend", "api", "worker", "mysql", "redis", "milvus", "nginx"} <= services.keys()


def test_required_scripts_exist():
    for name in ("install.sh", "deploy.sh", "run.sh", "shutdown.sh", "health-check.sh", "backup.sh"):
        assert Path("scripts", name).exists()
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/test_deployment_files.py -v`
预期：FAIL，Compose 和脚本尚未创建。

- [ ] **步骤 3：编写最少实现代码**

编写多阶段后端和 Next.js standalone 镜像；Compose 配置服务依赖、健康检查、数据卷和内部网络；Nginx 代理 Web、API 和 SSE，并关闭 SSE 缓冲。脚本实现 Ubuntu Docker 检查、`.env` 初始化、构建启动、幂等停止、健康检查和 MySQL/法律文件/解析结果备份。管理员邮箱、初始化密码、QQ SMTP 配置和本地文件目录从环境变量读取。

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest tests/test_deployment_files.py -v && docker compose config`
预期：部署文件测试通过，Compose 配置解析成功。

- [ ] **步骤 5：验证 Ubuntu 容器启动**

运行：`./scripts/install.sh && ./scripts/deploy.sh && ./scripts/health-check.sh`
预期：API、Worker、Next.js、MySQL、Redis、Milvus 和 Nginx 均达到预期健康状态。

### 任务 10：完成端到端验收、性能和安全验证

**文件：**
- 创建：`backend/tests/integration/`、`frontend/tests/e2e/`、`tests/ragas/qa.json`
- 创建：`tests/postman/legal-rag.postman_collection.json`、`tests/jmeter/20-users.jmx`
- 修改：`docs/测试文档.md`、`README.md`

- [ ] **步骤 1：编写失败的端到端验收测试**

```python
def test_document_to_answer_flow(api_client, official_fixture):
    document = api_client.collect_and_review(official_fixture)
    api_client.publish(document["document_id"])
    answer = api_client.ask("解除劳动合同需要核对哪些事实？")
    assert answer.citations
    assert answer.risk_notice
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest backend/tests/integration frontend/tests/e2e -v`
预期：在完整链路尚未打通或依赖未启动时失败，并明确显示缺失服务或断言。

- [ ] **步骤 3：补齐测试夹具和评测数据**

准备文本 PDF、扫描 PDF、表格 PDF、重复版本、过期法规、无答案问题、Prompt Injection、越权和伪造引用样本；配置 RAGAS 数据集覆盖法域、适用时间、条文号、无答案和安全问题；补充 Postman、JMeter 和 Ubuntu 部署验收脚本。

- [ ] **步骤 4：运行完整验证**

运行：`docker compose up -d --build && pytest -q && newman run tests/postman/legal-rag.postman_collection.json`
预期：单元、集成和接口测试通过；SSE 事件顺序、权限隔离、引用真实性和错误码符合文档。

- [ ] **步骤 5：运行性能和发布门禁检查**

运行：`jmeter -n -t tests/jmeter/20-users.jmx -l tests/jmeter/results.jtl`
预期：记录约 20 并发下的 `p50`、`p95`、`p99`、QPS、错误率和资源使用；首 Token、完整回答、检索和健康检查达到规格目标，P0/P1 缺陷为 0。

- [ ] **步骤 6：完成文档和工作区检查**

运行：`git diff --check`（如果目录初始化 Git）或 `python -m compileall backend/app`
预期：无空白错误、语法错误和未记录的首期范围冲突；在 Ubuntu 部署文档中记录实际版本、命令和结果。
