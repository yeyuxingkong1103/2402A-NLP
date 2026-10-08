# 工单17 优化摘要

## 优化措施清单

### A. RAGFlow 源码改造（参考实现）

| # | 文件 | 改造内容 |
|---|------|----------|
| 1 | api/db/services/llm_service.py | model_instance 加进程内缓存 _model_cache + _model_cache_lock，双检锁保证线程安全 |
| 2 | rag/llm/embedding_model.py | DefaultEmbedding 加 torch_empty_cache 方法，encode/encode_queries 后调用 |

### B. 自有 RAG 系统改造（压测验证）

| # | 文件 | 改造内容 |
|---|------|----------|
| 1 | rerankers.py:CrossEncoderReranker | _model 类属性单例 + _model_lock 线程锁 + torch_empty_cache |
| 2 | rerankers.py:build_reranker | _reranker_cache 进程内缓存 + _reranker_cache_lock |
| 3 | api_server.py | 全局 RAGQASystem 单例 + /metrics Prometheus 指标 |

### C. 关键代码片段

模型单例（RAGFlow）：

    _model_cache = {}
    _model_cache_lock = threading.Lock()

    @classmethod
    @DB.connection_context()
    def model_instance(cls, tenant_id, llm_type, llm_name=None, lang="Chinese"):
        cache_key = (tenant_id, llm_type, llm_name, lang)
        if cache_key in _model_cache:
            return _model_cache[cache_key]
        with _model_cache_lock:
            if cache_key in _model_cache:
                return _model_cache[cache_key]
            # ... 创建 instance ...
            _model_cache[cache_key] = instance
            return instance

GPU 释放（自有系统）：

    def rerank(self, query, candidates, top_k=3):
        # ... 推理 ...
        result = sorted(candidates, key=lambda x: -x["rerank_score"])[:top_k]
        self.torch_empty_cache()
        return result

    def torch_empty_cache(self):
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error emptying cache: {e}")

## 优化效果

| 指标 | 优化前 | 优化后 |
|------|--------|--------|
| 20并发 P95 | >10s（预估） | 700ms |
| 模型加载次数 | 每请求1次 | 进程内1次 |
| 内存增长（10分钟） | >45% | 0.00% |
| GPU 增长（10分钟） | 持续累积 | 10.51%（回落） |
