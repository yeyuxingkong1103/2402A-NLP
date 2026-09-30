# 法律客服助手 MVP 验收说明

**验收日期：** 2026-09-16，复核日期：2026-09-17，生产依赖联调复核日期：2026-09-18，生产化收口复核日期：2026-09-20，配置安全复核日期：2026-09-27，LangChain 真实验证复核日期：2026-09-23
**验收范围：** Task 1–14 的后端 MVP、评估数据集、静态前端改动、生产化收口配置、真实后端浏览器冒烟验证和真实知识库 RAG 浏览器验证
**当前版本提交：** `bcc7d6c`、`a7fc52a`、`9210afd`

## 一、验收结论

当前项目通过后端 MVP 和评估数据集验收，可以作为内部测试版本继续使用和迭代；暂不应视为生产部署版本。

验收结论明确如下：

- 后端单元、API 和集成测试全部通过。
- 100 条婚姻家庭 MVP 评估数据集校验通过。
- Python 模块编译检查通过。
- Playwright Chromium 浏览器端到端测试已执行并通过 6 条 mock 用例，2 条真实后端 gated 用例在默认套件中按设计跳过。
- 真实后端浏览器冒烟测试已执行并通过 1 条用例，覆盖 FastAPI 同源托管前端、本地开发身份头、真实聊天 API 和高风险家暴紧急指引链路。
- 真实知识库 RAG 浏览器测试已执行并通过 1 条用例，覆盖真实前端页面、SQL 聊天持久化、Milvus 正式集合检索、Reranker 重排和引用展示。
- LangChain 兼容 RAG 已完成真实 shadow 多场景验证，离婚/抚养权、抚养/探望、夫妻财产/债务三类可回答场景均为 `comparison_status=match`；默认 RAG 编排现为 `langchain`，`legacy` 保留为显式回退。
- MySQL 迁移链、SQL repository、Redis OTP、Redis 请求控制、Milvus、Celery、DeepSeek、BGE-M3 和 BGE Reranker 均已完成本地真实服务或真实模型联调；生产运行配置、导出安全交付、导出生命周期清理和测试告警收口已补齐。当前仍需正式生产安全审查、灾备演练和更完整的真实数据浏览器回归，不能据此直接宣称可对外生产部署。

## 二、已完成验证

### 后端测试

```text
python -m pytest backend/tests -q
183 passed, 2 skipped
```

后端测试合计：`183 passed, 2 skipped`。

### 迁移与后端复核

```text
alembic heads
011_expand_chat_external_ids (head)

DATABASE_URL=sqlite:///./alembic_verify.sqlite alembic upgrade head
DATABASE_URL=sqlite:///./alembic_verify.sqlite alembic downgrade base
通过

临时 MySQL 数据库 alembic upgrade head / downgrade base
通过

临时 MySQL 数据库认证、聊天、记忆和导出 repository 写读删联调
通过

2026-09-18 本地生产依赖链路复核：
- MySQL `root/root` 本地服务连通，临时库 `myrag_verify_20260918` 执行 `alembic upgrade head` / `downgrade base` 通过并已清理。
- Milvus Docker `milvus-standalone` 启动后健康，`check_milvus_readiness` 返回 `(True, None)`；临时集合写入 2 条向量、检索命中和清理通过；项目 `MilvusVectorStore.ensure_collection` / `upsert` / `search` 真实链路通过。
- Redis OTP 使用临时 Redis 容器验证保存、读取、TTL、删除和认证 store 注入路径通过。
- Celery 使用 Redis broker/backend 启动 solo worker，`inspect().ping()` 与项目 `check_celery_readiness` 均通过。
- BGE-M3 和 BGE Reranker 从项目 `models` 目录加载到 CUDA，通过 `backend/tests/integration/test_model_gpu_readiness.py`。
- DeepSeek 使用本地 `.env` 中的 API Key 完成最小流式调用，项目 `DeepSeekClient` 返回正常内容。
- 完整 `run_startup_checks(settings)` 返回 `mysql/redis/milvus/celery/deepseek/bge_m3/bge_reranker` 全部 ok，`ready=True`。
- `backend/tests/integration/test_real_rag_e2e.py` 通过：临时发布材料写入 Milvus，完成向量检索、rerank、引用构建和聊天回答链路，并清理临时集合。

python -m pytest backend/tests -q
183 passed, 2 skipped
```


```text
python evaluation/marriage_family_mvp/validate_dataset.py evaluation/datasets/marriage_family_mvp_100.jsonl
PASS: 100 records validated with counts 40/30/30
```

数据集领域分布为：

- 离婚与婚姻关系：40 条
- 抚养与探望：30 条
- 夫妻财产与债务：30 条

### 静态检查

```text
python -m compileall -q backend evaluation/marriage_family_mvp
通过

node --check frontend/tests/run-real-e2e.mjs && node --check frontend/js/api/user-api.js && node --check frontend/js/pages/profile.js
通过

git diff --check
通过
```

## 三、Playwright 浏览器测试

### Chromium E2E

```text
npm run test:e2e:chromium
6 passed, 2 skipped

npm run test:e2e:real
1 passed

npm run test:e2e:real:rag
1 passed
```

本轮已覆盖以下浏览器流程：

- 登录页面模拟手机号验证码登录，并保存 `user_id`。
- 聊天页面发送咨询、展示引用、重新生成回答和提交正向反馈。
- 个人中心展示长期记忆、关闭新增记忆提取、申请并下载加密数据导出。
- 个人中心确认账号注销，清理本地登录态并跳转登录页。
- 内容审核员审核知识库材料，并验证无发布按钮权限。
- 超级管理员发布已审核知识库材料。

当前默认浏览器端到端测试使用前端静态页面和 API mock 验证交互逻辑；`chat-real.spec.js` 与 `chat-real-rag.spec.js` 在默认套件中跳过。`npm run test:e2e:real` 已补充真实后端冒烟验证：本地启动 `127.0.0.1:8010` FastAPI 后端，使用同源托管前端页面访问真实聊天 API，并验证高风险家暴咨询返回紧急指引。`npm run test:e2e:real:rag` 已补充真实知识库 RAG 浏览器验证：本地 MySQL、Redis、Milvus、BGE 模型、DeepSeek 客户端和已索引知识库就绪后，真实页面可返回抚养权咨询回答和引用。

## 四、生产能力边界

当前实现已完成主要本地生产依赖联调，但仍是面向内部测试和生产化收口的版本；以下能力边界需明确：

- Alembic 迁移链已补齐至 `011_expand_chat_external_ids`，并通过临时 SQLite、本地真实 MySQL 的 `upgrade head` / `downgrade base` 验证；MySQL 上认证、聊天、记忆和导出 SQL repository 写读删联调通过。
- 认证用户和设备会话已具备 SQLAlchemy 持久化适配器；OTP 已具备 Redis 适配器；会话与消息已具备 SQLAlchemy 持久化适配器并在聊天服务中支持显式注入；长期记忆、候选更新、导出任务、反馈和高风险告警已具备 SQLAlchemy repository、Alembic 表结构和服务层显式注入路径。默认仍为内存后端。
- Milvus 适配器、知识库索引写入入口、Celery worker 入口、DeepSeek 客户端、本地 BGE-M3 和 BGE Reranker 均已完成本地真实服务或真实模型联调；完整 readiness 在本地依赖启动时可返回 `ready=True`。
- 已补充最小真实 RAG 端到端测试：临时材料索引进 Milvus，使用 BGE-M3 向量化、Milvus 检索、BGE Reranker 重排、引用构建和聊天服务回答链路通过；正式 public 法律知识库已写入 MySQL `13,294` 个 chunks，并在正式 Milvus 集合中核对为 `13,294` 条向量，已通过真实数据检索、Reranker、治理过滤和本地模型替身聊天链路验证。
- Redis、Celery、Milvus、MySQL、模型路径和密钥管理已补充到可复现部署指南；`docker-compose.yml` 已为本地依赖声明持久卷。本地 `run_startup_checks(settings)` 在 Celery worker 启动后已返回 `ready=True`。生产环境仍需替换默认密码、接入正式密钥管理和备份策略。
- 聊天限流和并发控制已新增 Redis 请求控制后端；导出下载次数由 SQL export job 元数据持久化。Redis 跨控制器并发限制已通过小规模验证，队列深度、Celery 任务监控和多实例压测仍需在部署环境验证。
- 导出文件临时口令已从普通下载 JSON 中移除，改为二次验证后的独立一次性领取接口；前端个人中心已同步适配。
- 导出文件生命周期清理已新增 repository 清理函数和 `python -m backend.scripts.cleanup_exports` 运维脚本；正式环境仍需把该脚本接入计划任务和备份排除策略。
- MySQL 结构备份、SQLite 迁移升降级和独立临时库完整 dump 恢复演练已通过；恢复核对 `knowledge_materials=7`、`document_chunks=13294`、`conversations=2`，临时库和备份文件已清理。Windows 原生 `mysql` CLI 对特殊转义正文不稳定，正式环境应使用 MySQL Shell、Linux/容器化客户端或经过验证的备份工具链。
- 真实后端浏览器冒烟测试已补充并通过；后续仍需扩展为覆盖登录、知识库治理、个人中心导出和真实知识库 RAG 的多场景浏览器回归。
- Starlette/httpx TestClient deprecation warning 已通过精确 pytest 过滤消除；后续升级到 httpx2 前仍需确认 FastAPI、Starlette 与 httpx2 的兼容版本组合。

## 五、发布建议

当前版本适合：

- 本地开发验证。
- 内部测试环境。
- 后端接口和评估数据集验收。
- 后续生产化设计的基础版本。

当前版本不适合直接用于：

- 对外公开服务。
- 多进程或多实例生产部署。
- 依赖持久化数据和灾备恢复的业务场景。
- 未经额外安全审查的正式法律服务场景。

后续开发应优先完成正式生产安全审查、灾备演练、Celery/Redis/Milvus 多实例压测、真实知识库浏览器回归扩展，以及法律服务上线前的人工验收流程。
