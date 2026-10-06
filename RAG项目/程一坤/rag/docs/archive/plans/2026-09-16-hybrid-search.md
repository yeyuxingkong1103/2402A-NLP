# 关键词与向量混合检索实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 `subagent-driven-development`（推荐）或 `executing-plans` 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 实现 MySQL 关键词检索 + Milvus 向量检索的法律知识混合召回，并与用户记忆结果分离融合。

**架构：** MySQL 是法律文档权威存储并承担关键词检索，Milvus `legal_documents` 保存法律 chunk dense embedding 并承担向量检索。在线查询同时执行关键词召回和向量召回，按 `chunk_id` 合并去重、过滤、融合排序，再从 MySQL 回查权威正文、父块和来源。

**技术栈：** MySQL FULLTEXT、SQLAlchemy、Milvus dense vector、Embedding 客户端、RRF 融合、Reranker、pytest。

**规格：** `docs/superpowers/specs/2026-09-15-legal-rag-mysql-offline-pipeline-design.md`

## 全局约束

- 混合检索定义为 MySQL 关键词检索 + Milvus dense 向量检索。
- Milvus 不承担法律文档 sparse 关键词检索。
- `legal_documents` 和 `user_memories` collection 必须分离。
- Milvus 返回 `chunk_id` 后，正文、父块和来源以 MySQL 为准。
- 未完成 Milvus 索引前不得把版本状态伪造成 `indexed`。
- 召回、融合和重排错误必须脱敏，不泄露其他用户内容。

---

### 任务 1：实现 MySQL 关键词检索 Repository

**文件：**
- 创建：`backend/app/search/keyword_search.py`
- 修改：`backend/app/db/sql_models.py`
- 测试：`backend/tests/test_keyword_search.py`

- [ ] **步骤 1：编写失败测试**

使用 SQLite fallback 或测试替身覆盖：按条文号、法规名称、专有名词命中 chunk；只返回已索引或允许检索的版本；结果包含 `chunk_id`、关键词分数、命中字段和可回查的版本标识。

- [ ] **步骤 2：运行测试验证失败**

运行：`python -m pytest tests/test_keyword_search.py -q`
预期：因关键词检索 Repository 不存在而失败。

- [ ] **步骤 3：实现最少代码**

为 MySQL 生产路径使用 `MATCH(content, retrieval_text) AGAINST (...)`；为 SQLite 单元测试提供等价的 `LIKE` fallback，确保测试语义稳定但不把 fallback 当成生产方案；所有查询都过滤版本状态。

- [ ] **步骤 4：运行测试验证通过**

运行：`python -m pytest tests/test_keyword_search.py -q`
预期：全部通过。

---

### 任务 2：实现法律文档 Milvus 向量索引

**文件：**
- 创建：`backend/app/search/vector_index.py`
- 创建：`backend/app/search/milvus_schema.py`
- 创建：`backend/app/cli/index_legal_documents.py`
- 测试：`backend/tests/test_legal_vector_index.py`

- [ ] **步骤 1：编写失败测试**

使用 fake Milvus client 和 fake embedding client 覆盖：从 MySQL 读取 `awaiting_embedding` chunk、生成 dense embedding、写入 `legal_documents`、upsert 字段包含 `chunk_id` 和版本过滤字段、成功后回写状态、失败时保持 `awaiting_embedding`。

- [ ] **步骤 2：运行测试验证失败**

运行：`python -m pytest tests/test_legal_vector_index.py -q`
预期：因索引服务不存在而失败。

- [ ] **步骤 3：实现最少代码**

定义 `legal_documents` schema，只包含 dense vector 与必要过滤字段；实现批量索引服务和 CLI `python -m app.cli.index_legal_documents --limit <n>`；维度由 embedding 客户端配置提供，不硬编码伪造成功。

- [ ] **步骤 4：运行测试验证通过**

运行：`python -m pytest tests/test_legal_vector_index.py -q`
预期：全部通过。

---

### 任务 3：实现混合召回与 RRF 融合

**文件：**
- 创建：`backend/app/search/hybrid_search.py`
- 创建：`backend/app/search/result_models.py`
- 测试：`backend/tests/test_hybrid_search.py`

- [ ] **步骤 1：编写失败测试**

构造关键词结果和向量结果：相同 `chunk_id` 应去重；只出现在一路的结果应保留；RRF 排序不依赖两个分数字段同量纲；条文号查询时关键词高排名结果应进入最终 top-k。

- [ ] **步骤 2：运行测试验证失败**

运行：`python -m pytest tests/test_hybrid_search.py -q`
预期：因混合检索服务不存在而失败。

- [ ] **步骤 3：实现最少代码**

定义统一结果模型 `SearchHit(chunk_id, source, rank, score, metadata)`；并行调用关键词检索与向量检索；使用 RRF `1 / (k + rank)` 融合；合并来源标签，保留最高解释信息；输出 top-k `chunk_id` 列表。

- [ ] **步骤 4：运行测试验证通过**

运行：`python -m pytest tests/test_hybrid_search.py -q`
预期：全部通过。

---

### 任务 4：实现检索上下文组装

**文件：**
- 创建：`backend/app/search/context_builder.py`
- 测试：`backend/tests/test_context_builder.py`

- [ ] **步骤 1：编写失败测试**

覆盖根据 `chunk_id` 从 MySQL 回查正文、父块、标题、source_url、版本状态；长期记忆结果必须标记为 `memory` 来源，法律结果必须标记为 `legal_document` 来源；无效版本或已删除记忆不得进入上下文。

- [ ] **步骤 2：运行测试验证失败**

运行：`python -m pytest tests/test_context_builder.py -q`
预期：因上下文组装器不存在而失败。

- [ ] **步骤 3：实现最少代码**

实现 `build_context(user_id, legal_chunk_ids, memory_hits)`；法律正文以 MySQL 权威数据为准；父块优先进入上下文，子块作为命中证据；用户记忆只接受已按 `user_id` 过滤的结果，并在输出中明确来源。

- [ ] **步骤 4：运行测试验证通过**

运行：`python -m pytest tests/test_context_builder.py -q`
预期：全部通过。

---

### 任务 5：端到端检索服务薄封装

**文件：**
- 创建：`backend/app/search/service.py`
- 测试：`backend/tests/test_search_service.py`

- [ ] **步骤 1：编写失败测试**

使用 fake keyword/vector/memory/reranker/context 组件覆盖：认证用户问题同时触发关键词和向量召回；Redis 短期记忆参与上下文；Milvus 长期记忆按用户过滤；最终结果包含引用和来源类型。

- [ ] **步骤 2：运行测试验证失败**

运行：`python -m pytest tests/test_search_service.py -q`
预期：因检索服务不存在而失败。

- [ ] **步骤 3：实现最少代码**

编排查询清洗、关键词检索、向量检索、长期记忆召回、短期记忆读取、RRF 融合、rerank 和上下文组装；服务层不直接访问请求体中的 `user_id`，只接收认证上下文。

- [ ] **步骤 4：运行测试验证通过**

运行：`python -m pytest tests/test_search_service.py -q`
预期：全部通过。
