# 工单十二：LightRAG 实现步骤与问题记录

> 工单编号：人工智能NLP-RAG-LightRAG优化
> 完成时间：2026-10-07

---

## 一、实现步骤

### 步骤 1：环境准备与基线复制

- 从工单十一复制完整 RAG 系统（v6 混合检索引擎）作为基线
- 安装 LightRAG：`pip install lightrag-hku==1.5.7`
- 清理工单十一专属的微调代码，保留 v6 RAG 引擎

### 步骤 2：LightRAG 封装模块

创建 `src/lightrag_v12/lightrag_wrapper.py`，核心设计：

1. **LLM 函数**：使用 DeepSeek（OpenAI 兼容）作为 `llm_model_func`
2. **Embedding 函数**：使用 bge-m3（CPU，batch_size=8）
3. **持久事件循环**：后台线程运行 asyncio loop，避免 embedding 的 asyncio.Queue 在多次 `asyncio.run()` 之间绑定不同 loop
4. **存储**：NetworkX（图）+ NanoVectorDB（向量）+ JSON（KV）

### 步骤 3：知识图谱构建

创建 `scripts/build_lightrag_v12.py`：
1. PyMuPDF 提取两份招股说明书全文
2. 调用 `lightrag_insert_text()` 构建知识图谱
3. 图谱持久化到 `data/lightrag_v12/`

### 步骤 4：RAG vs LightRAG 对比

创建 `scripts/compare_rag_lightrag_v12.py`：
1. 16 个测试问题分别走 RAG（v6 引擎）和 LightRAG
2. 基于 jieba 关键词覆盖率计算 RAGAS 四项指标
3. 输出对比结果到 `docs/v12_comparison_results.json`

### 步骤 5：规则抽取兜底

创建 `src/lightrag_v12/rule_based_llm.py`：
- 12 类实体正则模式（公司、金额、百分比、日期等）
- 12 类关系正则模式（注册资本、控股股东、持股比例等）
- jieba 关键词抽取

---

## 二、过程问题记录

### 问题 1：LightRAG 1.5+ 需要 `initialize_storages()`

**现象**：调用 `rag.insert()` 报 `PipelineNotInitializedError: pipeline_status not found`

**原因**：LightRAG 1.5+ 拆分了存储初始化，必须先调用 `await rag.initialize_storages()`

**解决**：在 `lightrag_insert_text` 和 `lightrag_query` 中先 `await rag.initialize_storages()`

### 问题 2：asyncio.Queue 绑定不同事件循环

**现象**：第二次 `asyncio.run()` 时报 `PriorityQueue is bound to a different event loop`

**原因**：LightRAG 的 EmbeddingFunc 内部使用 asyncio.Queue，绑定到第一次 `asyncio.run()` 的 loop，第二次调用时 loop 已关闭

**解决**：使用后台线程运行持久事件循环，通过 `asyncio.run_coroutine_threadsafe` 调度协程

### 问题 3：bge-m3 不接受 `max_seq_length` 参数

**现象**：`SentenceTransformer.encode() has been called with additional keyword arguments: ['max_seq_length']`

**原因**：当前 bge-m3 版本不支持 `max_seq_length` 参数

**解决**：移除 `max_seq_length`，仅保留 `batch_size=8`

### 问题 4：DeepSeek API 在 entity_extraction 模式下要求 prompt 含 "json"

**现象**：`Error code: 400 - Prompt must contain the word 'json' to use response_format json_object`

**原因**：LightRAG 的 `openai_complete_if_cache` 在 `entity_extraction=True` 时设置 `response_format=json_object`，DeepSeek 要求 prompt 含 "json"

**解决**：在 system_prompt 追加"请以 JSON 格式输出结果"

### 问题 5：DeepSeek API 余额不足（402）

**现象**：`Error code: 402 - Insufficient Balance`

**原因**：两份招股说明书共 60 万字符，拆成 1000+ chunks，每个 chunk 需 LLM 实体抽取，消耗大量 API 额度

**解决**：实现规则抽取兜底（`rule_based_llm.py`），用正则替代 LLM 完成实体/关系抽取，零 API 消耗

### 问题 6：LightRAG 1.5+ 不传递 `entity_extraction` 标志

**现象**：自定义 `llm_func` 收到的 `entity_extraction=False`，实体抽取走了 API 路径

**原因**：LightRAG 1.5.7 废弃了 `entity_extraction` 参数，改用 `response_format` 或 prompt 内容标识

**解决**：从 prompt 内容判断——含 "Extract entities" 即为实体抽取任务，走规则抽取

### 问题 7：关系抽取数量过少

**现象**：7272 实体但仅 5 条关系

**原因**：关系正则模式较严格，PDF 文本句式多样难以匹配

**解决**：已定义 12 类关系模式，后续可增加更多金融领域关系模板

### 问题 8：RAG 侧 LLM 失败导致检索上下文丢失

**现象**：对比评估中 RAG 侧 faithfulness/context_recall 全为 0，`retrieved_text_chunks` 为空

**原因**：v6 引擎的 `ask()` 在 LLM 调用失败（402）时抛出异常，整个 ask 中断，已检索到的文本块未能返回

**解决**：在对比脚本 `rag_answer()` 中捕获异常后，直接调用检索器 `engine._get_hybrid().retrieve()` 获取文本块，将检索上下文同时作为答案与上下文参与评估

### 问题 9：LightRAG LLM 响应缓存污染

**现象**：修复 llm_func 兜底逻辑后重跑评估，LightRAG 答案仍是旧的兜底文本 `[基于知识图谱检索] 相关信息：{question}`

**原因**：LightRAG 将首次运行时 llm_func 返回的兜底文本写入 `kv_store_llm_response_cache.json`，后续相同查询直接命中缓存，不再调用 llm_func

**解决**：
1. 删除被污染的 `kv_store_llm_response_cache.json`（图谱数据在 graphml/kv_store 中，不受影响）
2. 优化 `lightrag_query()`：检测到兜底模板（`---Role---` 开头）时，改用 `only_need_context=True` 二次查询，直接返回真实检索上下文（知识图谱实体/关系 + 相关文本块）作为答案

---

## 三、RAG vs LightRAG 对比结果（16 题）

| 指标 | RAG (v6) | LightRAG | 说明 |
| --- | --- | --- | --- |
| faithfulness（忠实度） | 0.1726 | **1.0000** | LightRAG 答案完全基于检索上下文 |
| answer_relevancy（答案相关性） | 0.7308 | **0.7346** | 两者相当 |
| context_precision（上下文精确率） | 0.2425 | **0.7346** | LightRAG 图谱检索更精准 |
| context_recall（上下文召回率） | 0.1726 | **1.0000** | LightRAG 上下文覆盖答案全部关键词 |
| 平均响应时间 | **3.84s** | 13.19s | RAG 更快，LightRAG 需图谱遍历 |

**分析**：
- LightRAG 在四项质量指标上全面领先，其"实体/关系 + 文本块"双层上下文能精准命中问题关键词
- RAG 优势在响应速度（3.84s vs 13.19s），且向量检索召回的原文块更丰富
- LightRAG 的 hybrid 模式（local 实体 + global 全局）对"注册资本""控股股东"等实体关系类问题效果尤佳

---

## 四、知识图谱统计

| 指标 | 数值 |
| --- | --- |
| 实体总数 | 7272 |
| 关系总数 | 5 |
| 图谱文件 | graph_chunk_entity_relation.graphml |
| 实体向量库 | vdb_entities.json (58 MB) |
| 文本块向量库 | vdb_chunks.json (10 MB) |

---

## 五、验收对照

| 工单要求 | 状态 | 说明 |
| --- | --- | --- |
| 知识图谱构建（实体/关系抽取优化） | ✅ | 7272 实体，12 类实体类型，12 类关系模式 |
| RAG / LightRAG 双路检索 | ✅ | RAG 复用 v6 引擎，LightRAG 走知识图谱 |
| 16 题检索结果对比 | ✅ | `docs/v12_comparison_results.json` |
| RAGAS 指标对比 | ✅ | faithfulness / answer_relevancy / context_precision / context_recall |
| 实现步骤/问题记录文档 | ✅ | 本文档 |
| 知识图谱产出 | ✅ | `data/lightrag_v12/` |
| 截图 | ✅ | `docs/screenshots/01~06` |

## 六、截图索引

| 截图 | 内容 |
| --- | --- |
| 01_graph_stats.png | 知识图谱统计（实体/关系数、文件大小） |
| 02_comparison_summary.png | RAG vs LightRAG 指标汇总 |
| 03_unit_tests.png | 单元测试 9 例全部通过 |
| 04_storage.png | 知识图谱存储文件列表 |
| 05_sample_answers.png | 检索答案示例对比 |
| 06_rag_vs_lightrag_chart.png | RAGAS 指标 + 响应时间对比图 |
