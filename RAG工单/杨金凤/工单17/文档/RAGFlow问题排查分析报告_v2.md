# 问题排查分析报告

---

## 2.1 环境信息

| 项目 | 值 |
| --- | --- |
| 部署方式 | Docker Compose（Windows Docker Desktop） |
| RAGFlow 版本 | v0.19.1 |
| 容器数量 | 5（server + es + mysql + redis + minio） |
| GPU | NVIDIA RTX 3090 |
| Chat 模型 | glm-4-flash（智谱云端） |
| Embedding（优化前） | embedding-3（智谱云端） |
| Embedding（优化后） | bge-m3:latest（本地 Ollama） |

---

## 2.2 排查过程

### 步骤1：系统资源监控（docker stats）

| 容器 | CPU % | 内存 |
| --- | --- | --- |
| ragflow-server | 12.90% | 3.38G / 15.37G |
| ragflow-es-01 | 0.34% | 4.34G / 7.52G |
| ragflow-mysql | 0.99% | 499M |
| ragflow-redis | 0.17% | 4.57M |
| ragflow-minio | 0% | 108M |

**结论**：CPU 均未打满 → 非计算瓶颈。

### 步骤2：进程线程分析

```text
PID     USER  CPU%  MEM%  COMMAND
26      root  10.0  11.5  python3 api/ragflow_server.py
Threads: 75
```

**结论**：Flask 多线程正常，75 个线程并发能力充足。

### 步骤3：日志分析（docker logs ragflow-server）

大量重复报错：

```text
httpx.ConnectError: [SSL: UNEXPECTED_EOF_WHILE_READING]
httpcore.ConnectTimeout: handshake operation timed out
zhipuai.core._errors.APITimeoutError: Request Timeout
```

### 步骤4：调用链追踪（traceback）

```text
dialog_service.chat()
  → retriever.retrieval()
    → Dealer.search()
      → get_vector()
        → emb_mdl.encode_queries()
          ← 走智谱 Embedding API
          → zhipuai client.embeddings.create()
            → httpx.ConnectTimeout
            ← 根因
```

---

## 2.3 根因分析

| 问题 | 定位 | 说明 |
| --- | --- | --- |
| P95 = 49.6s（20并发） | 外网 API 限流 | 每个请求调用智谱 Embedding + Chat，20并发=40个 |
| SSL 连接风暴 | 连接被拒 | 失败 2/99，智谱对同一 API Key 有并发限制，超限直接断开 |
| 内存缓慢增长 | 会话上下文累积 | `conv.message` 只追加不清空 |
| 吞吐上不去 | 外网延迟 | 单次 Embedding 调用 200ms + Chat 生成 3~5s |
