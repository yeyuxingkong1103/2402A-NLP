# RAGAS 评测报告

## 1. 评测概览

- 评测时间：2026-09-27 06:54:32 UTC
- 评测数据：`data/evaluation/ragas_dataset.json`
- 样本数：5
- 检索后端：Milvus `knowledge_chunks`
- 检索数量：每个问题 5 个上下文
- Embedding：`BAAI/bge-m3`
- Reranker：`BAAI/bge-reranker-v2-m3`
- 生成模型：项目配置中的 DeepSeek OpenAI-compatible API
- 评测框架：RAGAS 0.4.3
- 原始逐题结果：`data/evaluation/ragas_latest.json`
- 混合检索逐题结果：`data/evaluation/ragas_optimized.json`

评测脚本使用真实 Milvus 检索结果和真实模型回答。危机问题沿用项目的固定安全响应，不绕过安全规则。RAGAS 的 `context_relevancy` 在 0.4.3 中内部名称为 `nv_context_relevance`，脚本已统一映射为需求文档使用的名称。

## 2. 平均指标

| 指标 | 优化前 | 混合检索后 | 变化 |
| --- | ---: | ---: | ---: |
| faithfulness | 0.6428 | 0.6020 | -0.0408 |
| answer_relevancy | 0.4752 | 0.6565 | +0.1813 |
| context_precision | 0.4633 | 0.5433 | +0.0800 |
| context_recall | 0.2000 | 0.3000 | +0.1000 |
| context_relevancy | 0.6500 | 0.9500 | +0.3000 |

混合检索版提升了 `answer_relevancy`、`context_precision`、`context_recall` 和 `context_relevancy`，但 `faithfulness` 下降约 0.0408。五项指标均为有效数值，没有将 `NaN` 当作评测结果。由于 `faithfulness`、`answer_relevancy` 和 `context_relevancy` 含有 LLM-as-a-judge 成分，重复运行时会有一定波动，比较不同版本时应使用相同模型、数据集和参数。

## 3. 逐题结果（混合检索后）

| 问题 | faithfulness | answer_relevancy | context_precision | context_recall | context_relevancy |
| --- | ---: | ---: | ---: | ---: | ---: |
| 压力大时怎么缓解？ | 1.0000 | 0.9740 | 0.8667 | 1.0000 | 1.0000 |
| 睡眠不好怎么办？ | 0.6512 | 0.8469 | 0.0000 | 0.0000 | 1.0000 |
| 焦虑时可以做什么？ | 0.5333 | 0.8803 | 0.7000 | 0.5000 | 1.0000 |
| 我是不是抑郁症？ | 0.5946 | 0.0000 | 0.7000 | 0.0000 | 0.7500 |
| 如果有人说不想活了怎么办？ | 0.2308 | 0.5813 | 0.4500 | 0.0000 | 1.0000 |

与基线相比，压力、焦虑和诊断问题的上下文排序整体改善；睡眠问题的上下文相关性和生成回答相关性提高，但召回内容仍没有覆盖参考答案中的睡眠卫生措施，因此 `context_precision` 和 `context_recall` 仍为零。

## 4. 结果分析

### 4.1 已验证的改进

- 增加中文词面与主题匹配后，平均 `context_precision` 从 0.4633 提升到 0.5433，`context_recall` 从 0.2000 提升到 0.3000。
- 平均 `answer_relevancy` 从 0.4752 提升到 0.6565，`context_relevancy` 从 0.6500 提升到 0.9500，说明候选上下文与问题的主题匹配更稳定。
- 结果增加了 chunk 文本去重、同文档和同页数量限制，减少单一来源占满 top-k 的情况；当候选不足时才回填被限制的结果，避免返回过少上下文。
- “压力大时怎么缓解？”仍是表现最稳定的样本；“焦虑时可以做什么？”的上下文精确率和召回率均有提升。

### 4.2 仍需优先改进的问题

- `faithfulness` 从 0.6428 降至 0.6020。该指标含有 LLM-as-a-judge 成分，且本次生成内容会随上下文排序变化；应继续减少泛化资料混入，并对回答长度和引用范围做约束，不能把这次下降忽略为全面提升。
- 睡眠问题的 `context_precision=0`、`context_recall=0` 仍说明知识库缺少睡眠卫生、睡前习惯和失眠就医指引。当前优化改善了主题识别，但没有凭空补足知识；应补充真实睡眠主题资料后重新解析、分块、向量化。
- 诊断问题的 `answer_relevancy=0` 仍受评测参考答案与生成回答侧重点不一致影响。应增加安全边界专项样本，并用人工检查确认“不诊断、建议专业评估”的行为。
- 危机问题的 `faithfulness=0.2308` 和 `context_recall=0` 不能简单解释为安全响应错误：固定响应优先于普通 RAG 生成。危机安全应继续使用独立规则集、人工安全验收和专门的危机测试，不把普通 RAGAS 忠实度作为唯一判据。

## 5. 根据需求文档的完成情况

需求依据：`docs/requirements/requirements_spec.md` 第 3、6、9、10、11、13、16、17 节。

### 已完成或可演示

- FastAPI、Vue 3、MySQL、Redis、Milvus 基础链路已启动并通过健康检查。
- 用户注册、登录、退出、当前用户查询和用户隔离已实现。
- 会话持久化、会话重命名、软删除和历史查询已实现。
- 多个心理健康助手角色及管理员角色配置接口已实现。
- 普通聊天、SSE 流式聊天、Milvus 检索、BM25、中文词面融合、BGE rerank、DeepSeek 生成和危机固定响应已接入。
- 当前 10 个 PDF 已形成可检索知识库，RAGAS 测试集和五项指标脚本已实现。
- 文档上传、文档列表、解析、向量化和任务状态 API 已实现，管理员前端可操作并查看失败原因。
- RAGAS 后端任务、历史查询、MySQL 持久化和管理员评测页面已实现。
- Redis 已支持按用户/角色隔离的历史 key、查询缓存和文档状态缓存，并兼容旧 session key。
- API Key、数据库密码和用户密码没有写入评测代码；密码使用哈希保存。

### 部分完成

- **PDF 入库链路**：HTTP 管理接口、管理员页面和任务状态已完成；后台任务当前使用进程内线程，生产环境仍建议替换为 Celery/RQ 等可恢复队列。
- **知识库检索**：已完成 Milvus + BM25 + CrossEncoder rerank + 词面分数融合；MySQL 结构化召回和长期记忆召回仍未接入在线链路。
- **日志**：部分 ingestion、Redis 和服务模块使用 Python logging，但尚未按 API、RAG、解析、LLM、错误、评测分类并统一输出，也未形成完整脱敏验收。
- **测试**：已有编译、OpenAPI、健康检查、Redis、BM25、数据库和前端构建验证；PDF/OCR/Embedding/Milvus/prompt/后处理单元测试、接口报告和压测尚不完整。

### 尚未实现的需求

#### 数据库和向量集合

- `long_term_memories` 长期记忆表及对应 Milvus collection。
- 需求建议的 `roles` 统一表结构仍需与当前 `ai_roles` 方案做正式迁移和文档对齐。

#### 检索和记忆能力

- MySQL 结构化数据多路召回。
- 长期记忆提取、用户确认、删除和语义检索。
- Query 改写/扩写和统一回答后处理安全校验。

#### 前端和管理员功能

- 管理员角色管理页面。
- 需求中的游客模式和用户偏好管理界面。

#### 运维、文档和质量保障

- Ubuntu/云服务器的完整 Nginx/systemd 配置模板。
- 完整 API 测试报告和 JMeter 压测报告。
- 统一分类日志、敏感字段脱敏和评测日志规范仍需专项完善。

## 6. 建议实施顺序

1. 先补充睡眠、焦虑和危机干预的高质量知识库内容，并重新解析、分块、向量化。
2. 增加 BM25 与向量检索融合，再调 rerank、阈值和 top-k；每次变更固定同一评测集对比五项指标。
3. 实现文档上传、解析、向量化接口和管理员文档页。
4. 增加 `rag_evaluation_runs` 与 `POST /api/v1/evaluations/ragas`，让评测可从网页启动并保存历史结果。
5. 统一 Redis key，补充查询缓存和文档状态缓存。
6. 补齐日志分类、单元/集成/API/压测和部署文档，再进行一期验收。
