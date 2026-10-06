# LAW-RAG 测试与验收文档

本文档按照当前代码和工作区状态整理，用于项目收尾、系统验收和最终报告编写。

## 1. 当前测试状态

当前工作区的 `tests/` 目录没有保留可执行测试文件，因此不能继续沿用旧版本的“通过多少条测试”结论，也不能把旧版本 pytest 结果作为当前版本结果。

当前可用的验证资源主要包括：

- `tools/check_services.py`：检查 MySQL、Redis、Milvus、模型和工作区服务；
- `tools/evaluate.py`：执行离线检索、答案、引用和延迟评测，也可以调用在线问答接口；
- `evaluation/datasets/`：离线冒烟集和公共法律库 Golden 数据集；
- `evaluation/reports/`：历史离线评测报告，可作为参考材料，但不应直接标记为本次代码版本的新增测试结果。

## 2. 测试目标

测试重点覆盖以下五类质量属性：

1. 功能正确性：接口、认证、上传、检索、记忆和解决方案功能可用；
2. 数据隔离：不同用户无法读取彼此的会话、文件和私有向量；
3. 检索质量：召回结果、重排结果、答案关键词和引用具有基本可靠性；
4. 稳定性：依赖服务异常、模型异常和后台任务异常时能够返回可理解的降级结果；
5. 性能和可观测性：限流、缓存、流式输出、结构化日志和健康检查正常。

## 3. 环境和依赖

### 3.1 必需组件

| 组件 | 用途 |
|---|---|
| Python 3.11+ | 运行 FastAPI、数据处理和评测脚本 |
| MySQL | 用户、文件、会话历史和长期记忆持久化 |
| Redis | 认证会话、短期记忆、缓存和限流辅助数据 |
| Milvus | 公共法律库和私有文档向量检索 |
| LLM 服务 | 查询理解、答案生成和方案生成 |
| Embedding 服务 | 文档和问题向量化 |
| Reranker 服务 | 检索结果重排 |

PDF、DOCX、XLSX、图片 OCR 和多媒体测试还需要对应 Python 包及系统级 OCR/媒体依赖。

### 3.2 启动服务

```powershell
python run.py
```

默认服务地址为 `http://127.0.0.1:7294`。启动脚本会尝试启动 Celery worker；Celery 不可用时回退到本地线程池。

## 4. 静态和基础检查

### 4.1 检查 Python 语法

```powershell
python -m compileall backend data_pipeline tools evaluation
```

预期结果：命令退出码为 `0`，没有 SyntaxError 或 ImportError。

### 4.2 检查测试目录

```powershell
python -m pytest
```

当前状态下，由于 `tests/` 目录没有测试文件，该命令不会产生有效的业务测试结果。执行记录中应明确标注“当前版本无可执行 pytest 用例”，不能写成“全部测试通过”。

### 4.3 检查路由注册

启动应用后访问：

```text
GET /openapi.json
GET /docs
```

检查认证、用户、工作区、记忆、RAG 和系统路由均已出现在 OpenAPI 文档中。

## 5. 依赖服务检查

```powershell
python tools/check_services.py
```

脚本首先检查 MySQL、Redis、Milvus 的 TCP 可达性，然后创建应用服务并调用各服务的 `health()` 方法。输出为 JSON，核心字段为：

```json
{
  "ok": true,
  "services": [
    {"name": "mysql_tcp", "ok": true},
    {"name": "redis_tcp", "ok": true},
    {"name": "milvus_tcp", "ok": true},
    {"name": "mysql", "ok": true},
    {"name": "redis", "ok": true},
    {"name": "milvus", "ok": true},
    {"name": "model", "ok": true},
    {"name": "workspace", "ok": true}
  ]
}
```

依赖未启动时，脚本会返回 `ok=false` 和具体错误原因。该结果应记录为环境问题，不应误判成业务逻辑缺陷。

## 6. 接口功能验收

### 6.1 认证和用户

| 编号 | 用例 | 预期结果 |
|---|---|---|
| AUTH-01 | 使用合法用户名、邮箱和密码注册 | 返回成功，创建会话并写入 Cookie |
| AUTH-02 | 重复用户名注册 | 返回 `409` |
| AUTH-03 | 重复邮箱注册 | 返回 `409` |
| AUTH-04 | 使用正确账号密码登录 | 返回用户信息和会话 Cookie |
| AUTH-05 | 使用错误密码登录 | 返回 `401`，不泄露账号是否存在 |
| AUTH-06 | 连续登录失败 | 达到阈值后返回 `429` |
| AUTH-07 | 调用 `/auth/me` | 返回公开用户、设置、画像和注销状态 |
| AUTH-08 | 登出后继续访问受保护接口 | 返回 `401` |
| AUTH-09 | 修改设置和用户画像 | 返回保存后的数据 |
| AUTH-10 | 发起、查询和取消注销申请 | 状态变化符合预期 |

### 6.2 RAG 问答

| 编号 | 用例 | 预期结果 |
|---|---|---|
| RAG-01 | 提交正常法律问题 | 返回 `answer`、`sources` 和 `meta` |
| RAG-02 | 使用 `query` 兼容字段 | 能正常进入问答流程 |
| RAG-03 | 提交空问题 | 返回 `422` |
| RAG-04 | 匿名问答 | 可以回答，但不读取用户会话和私有材料 |
| RAG-05 | 登录用户携带 `session_id` 问答 | 可读取当前会话上下文和允许的工作区材料 |
| RAG-06 | 开启 `include_web` | 在配置和网络允许时增加 Web 检索通道 |
| RAG-07 | 请求 `/ask/stream` | 收到 `thinking`、`status`、`step`、`chunk`、`complete` 或 `error` 事件 |
| RAG-08 | 连续超过匿名限流阈值 | 返回 `429` |
| RAG-09 | 生成法律解决方案 | 返回方案内容和 `cache_key` |
| RAG-10 | `force_refresh=true` | 跳过旧缓存并刷新 Redis 缓存 |
| RAG-11 | 导出 PDF | 返回 `application/pdf` 附件 |

### 6.3 工作区文件

| 编号 | 用例 | 预期结果 |
|---|---|---|
| FILE-01 | 上传 TXT/Markdown/JSON | 保存成功并生成可检索文本 |
| FILE-02 | 上传 PDF/DOCX/XLSX | 返回抽取结果和处理状态 |
| FILE-03 | 上传图片 | 按配置执行 OCR |
| FILE-04 | 批量上传多个文件 | 返回逐文件结果和数量统计 |
| FILE-05 | 上传不支持的扩展名 | 返回失败原因，不写入非法索引 |
| FILE-06 | 超过单文件大小限制 | 拒绝上传 |
| FILE-07 | 查询当前用户文件列表 | 只能看到当前用户文件 |
| FILE-08 | 按 `session_id` 查询文件 | 只返回该会话材料 |
| FILE-09 | 删除文件 | 同时清理数据库记录、原文件、快照和私有向量 |
| FILE-10 | 使用其他会话删除文件 | 返回 `session_mismatch`，文件不被删除 |
| FILE-11 | 音频或视频异步处理 | 先返回 `processing`，后台完成后变为 `ready`、`stored` 或 `failed` |

### 6.4 记忆和会话

| 编号 | 用例 | 预期结果 |
|---|---|---|
| MEM-01 | 查询和更新长期记忆开关 | 返回当前开关状态 |
| MEM-02 | 长期记忆关闭时问答 | 不读取长期记忆 |
| MEM-03 | 长期记忆开启后问答 | 按用户范围检索相关事实 |
| MEM-04 | 查询长期记忆列表 | 只返回当前用户记录 |
| MEM-05 | 查看单条记忆解释 | 返回来源和关联信息 |
| MEM-06 | 处理记忆冲突 | 支持 `accept_new`、`keep_old`、`discard` |
| MEM-07 | 删除单条或全部长期记忆 | 仅在功能开启时执行删除 |
| MEM-08 | 删除会话 | 清理短期记忆和会话派生状态，保留历史、长期记忆和工作区材料 |

## 7. 数据处理和离线评测

### 7.1 公共数据构建

公共数据处理流程为加载、解析、清洗、质量校验、切分、向量化和 Milvus 入库。建议至少检查：

- 公共集合名称与代码定义一致；
- 每条记录存在可检索文本和稳定标识；
- 向量维度等于 `EMBEDDING_DIM`；
- 空记录、占位记录和重复记录被清理；
- `quality_report.json` 能够生成并反映异常数量。

### 7.2 离线 Golden 评测

只评估已有返回结果：

```powershell
python tools/evaluate.py evaluation/datasets/public_law_sample8.json --format text
```

评估当前在线接口：

```powershell
python tools/evaluate.py `
  evaluation/datasets/public_law_sample8.json `
  --ask-url http://127.0.0.1:7294/api/v1/legal/ask `
  --format text `
  --output evaluation/reports/current-rag-eval.json
```

可评估指标包括：

- `recall@k`：期望来源是否被召回；
- `precision@k`：召回来源中有效来源比例；
- `mrr@k`：第一条相关来源的排序质量；
- `keyword_coverage`：回答覆盖期望关键词的程度；
- `citation.groundedness`：回答引用与来源的对应程度；
- `latency.total_ms.p95_ms`：总体延迟 P95。

可以通过 `--fail-under` 和 `--max-latency-ms` 设置验收阈值，例如：

```powershell
python tools/evaluate.py `
  evaluation/datasets/public_law_sample8.json `
  --ask-url http://127.0.0.1:7294/api/v1/legal/ask `
  --fail-under retrieval.recall@5=0.8 `
  --fail-under answer.keyword_coverage=0.8 `
  --max-latency-ms 15000 `
  --format text
```

## 8. 安全测试

应至少执行以下检查：

- 未登录访问用户、文件、记忆和 PDF 接口，均返回 `401`；
- 用户 A 不能读取用户 B 的文件列表、会话、长期记忆和私有检索结果；
- 使用 `../`、绝对路径或非法文档 ID 上传/删除文件时被拒绝；
- 文件删除不会影响其他用户同名文件；
- 响应体不包含密码哈希、认证令牌或内部连接信息；
- CORS 只允许配置的来源；
- Cookie 在生产环境启用 `Secure`；
- 登录失败限流、RAG 限流和 PDF 生成限流均生效；
- 日志完整性接口不能被普通用户越权调用；
- 日志中不记录明文密码和完整认证令牌。

## 9. 异常和降级测试

| 场景 | 预期行为 |
|---|---|
| Redis 不可用 | 健康检查标记异常，非关键功能按实现降级，不应泄露异常堆栈 |
| MySQL 不可用 | 认证、历史和用户状态接口返回明确的服务不可用信息 |
| Milvus 集合未加载 | 检索跳过异常集合或进入降级路径，服务仍返回可解释结果 |
| 模型服务超时 | 问答接口返回错误事件或业务错误，不阻塞其他请求 |
| 后台任务队列不可用 | 文件处理回退到本地线程池 |
| 文件解析失败 | 保留原文件，状态为 `stored` 或 `failed`，返回失败原因 |
| 向量入库失败 | 不伪造 `ready` 状态，处理快照和状态保持一致 |
| 长期记忆检索失败 | 当前问答继续使用可用的短期记忆和检索证据 |

## 10. 测试记录模板

项目报告中建议使用以下格式记录实际执行结果：

| 日期 | 环境 | 用例范围 | 结果 | 失败原因/备注 |
|---|---|---|---|---|
| YYYY-MM-DD | 本地/测试环境 | 认证、RAG、文件、记忆 | 通过/失败/阻塞 | 依赖服务或具体现象 |

最终报告应分别列出：已实际执行的用例、因依赖未启动而阻塞的用例、当前尚未自动化的用例。当前版本最重要的测试结论是：测试目录没有保留可执行 pytest 用例，因此功能验收应以接口冒烟、依赖检查和离线/在线评测报告为主要证据。
