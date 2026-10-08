# 工单17 问题排查分析报告

**工单编号**：人工智能 NLP-RAG-解决 API 服务并发瓶颈与资源泄漏
**分析时间**：2026年10月8日
**分析对象**：RAGFlow 源码 + 自有 RAG 系统

---

## 1. 故障现象

| 现象 | 描述 |
|------|------|
| 响应延迟飙升 | 10+并发时，问答接口 P95 从 2s 增至 10s+ |
| 内存泄漏与崩溃 | 容器内存持续增长，数小时后 OOM |
| GPU 资源不释放 | 纯文本问答也大量占用 GPU，请求结束后不释放 |

---

## 2. 根因定位（5个确认 + 2个排除）

### 根因1：模型重复加载（主因，RAGFlow）

位置：api/db/services/llm_service.py 的 model_instance 方法

问题：每次调用都 new 一个模型实例（EmbeddingModel / ChatModel / RerankModel 等），
调用点经 grep 统计有 30+ 处（LLMBundle + TenantLLMService.model_instance）。

影响：20并发时 20 次重复加载模型，叠加延迟 + GPU 显存累积。

### 根因2：模型重复加载（主因，自有系统）

位置：~/autodl-tmp/rerankers.py 的 build_reranker 函数

问题：CrossEncoderReranker.__init__ 里每次 new CrossEncoder(model_name)，
build_reranker 每次调用都返回新实例。

影响：并发时反复加载 BAAI/bge-reranker-base 模型。

### 根因3：GPU 释放不完整（主因）

RAGFlow：
- rag/llm/rerank_model.py:93 有 torch_empty_cache 方法
- rag/llm/embedding_model.py 无 torch_empty_cache 方法

自有系统：
- rerankers.py 的 CrossEncoderReranker.rerank 后无 empty_cache 调用

影响：embedding/rerank 推理后的张量不释放，GPU 显存持续占用。

### 根因4：WSGI 线程无上限（次因，RAGFlow）

位置：api/ragflow_server.py:121

问题：run_simple(..., threaded=True, ...)，werkzeug threaded 模式每请求 new 线程，无池化上限。

### 根因5：DeepDoc Parser 重复 new（次因，RAGFlow）

位置：rag/app/*.py 共 11 处 PlainParser() / ExcelParser() 调用。

### 排除1：DB 连接池（安全）

位置：api/db/db_models.py:275-290

RAGFlow 已使用 PooledMySQLDatabase，配置 max_connections=100, stale_timeout=30。
结论：DB 连接池无泄漏。

### 排除2：ES/Redis 连接（安全）

位置：
- rag/utils/es_conn.py:40 使用 @singleton 装饰器
- rag/utils/redis_conn.py:51 使用 @singleton 装饰器

结论：ES/Redis 已单例，无泄漏。

---

## 3. 根因汇总表

| # | 根因 | 文件 | 状态 |
|---|------|------|------|
| 1 | 模型重复加载（RAGFlow） | llm_service.py:model_instance | 确认 |
| 2 | 模型重复加载（自有系统） | rerankers.py:build_reranker | 确认 |
| 3 | GPU 释放不完整 | embedding_model.py + rerankers.py | 确认 |
| 4 | WSGI 线程无上限 | ragflow_server.py:121 | 确认 |
| 5 | Parser 重复 new | rag/app/*.py | 确认 |
| 6 | DB 连接池 | db_models.py:275 | 安全 |
| 7 | ES/Redis 连接 | es_conn.py:40 / redis_conn.py:51 | 安全 |

---

## 4. 分析方法

| 方法 | 工具 | 用途 |
|------|------|------|
| 静态分析 | grep -rn | 定位 model_instance 调用点 |
| 代码审查 | sed -n | 查看函数实现 |
| 语法验证 | ast.parse | 确认补丁正确 |
| 运行时验证 | .venv/bin/python -c | 确认导入 OK |
| 压测 | Locust | 并发验证 |
| 资源监控 | ps + nvidia-smi | 内存/GPU 曲线 |

---

## 5. 结论

P95 飙升主因 = 模型重复加载（RAGFlow 的 model_instance + 自有系统的 build_reranker）

GPU 显存持续增长主因 = 模型实例不释放 + embedding/rerank 无 empty_cache

排除项 = DB/ES/Redis 连接池已安全，非泄漏点
