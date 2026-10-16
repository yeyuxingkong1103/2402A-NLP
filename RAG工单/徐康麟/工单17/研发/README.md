# 研发目录说明（工单17）

**工单编号**：`人工智能NLP-RAG-解决API 服务并发瓶颈与资源泄漏工单`

本目录是**可运行源码**，分为四块：

```
研发/
├── app/                 RAG 业务内核（来自工单7，工单17 打了 5 处性能补丁）
│   ├── core/
│   │   ├── perf_hook.py     ★新增：性能开关钩子（app 层接入点，不强依赖 service 包）
│   │   ├── qa_engine.py     ✎F5：答案缓存加锁 + TTL + 上限 + 条目瘦身
│   │   ├── llm_client.py    ✎F1/F7：连接池 + keep_alive 可配 + unload
│   │   ├── embedder.py      ✎F1/F2/F7：连接池 + 探测结果缓存 + keep_alive
│   │   └── …（其余为工单7 原样复用）
│   └── storage/sqlite_manager.py  ✎F6：WAL + 批量提交 + 缓存表上限
├── service/             部署层 FastAPI 服务
│   ├── api.py               ✎F3/F4/F9/F10 + 4 个新端点
│   ├── perf_config.py   ★新增：全部 PERF_* 阈值
│   ├── http_pool.py     ★新增：F1 标准库 HTTP keep-alive 连接池
│   ├── singletons.py    ★新增：F2 线程安全单例 + TTL 探测缓存
│   ├── limiter.py       ★新增：F3 有界并发闸门（429 快速失败）
│   ├── task_queue.py    ★新增：F9 解析队列 + 工作线程池
│   └── session_store.py    ✎F8：会话文件上限 + TTL 物理清理
├── 压测/                 自研 jmeter 替代
│   ├── load_runner.py      并发/时长/思考时间/分位/吞吐/失败率/熔断
│   ├── 场景配置.json        场景与验收阈值（配置化）
│   └── 问题集.jsonl         6 个简单问句（工单2 风格）
├── 剖析/                 自研 py-spy / memory-profiler / objgraph 替代
│   ├── 剖析问答链路.py       cProfile + urlopen 计数 + 分段耗时
│   ├── 内存剖析.py          tracemalloc 快照 diff + gc 对象分类
│   ├── redis统计.py         MiniRedis 读 INFO stats（redis-cli 替代）
│   └── 探GPU.ps1           nvidia-smi + /api/ps 连续采样
├── scripts/
│   └── 汇总对比.py          前后对比 → 优化/评估结果/{对比.csv,对比.md,...}
├── 基座快照/             **优化前**（工单10）的 app/ 与 service/ 全量副本（回滚 + 复现基准用）
├── 第三方库/             空（本工单**零新增依赖**）
├── requirements.txt
└── README.md            本文件
```

---

## 运行姿势

所有脚本统一走工单根目录的 `run_py.ps1`（解释器默认 `E:\gao6gongdan\工单1\.venv\Scripts\python.exe`）：

```powershell
cd E:\gao6gongdan\工单17

# 起服务（优化后）
pwsh -NoProfile -File 部署/脚本/起服务.ps1 -PerfEnabled 1

# 压测
pwsh -NoProfile -File run_py.ps1 研发/压测/load_runner.py --scenario A --concurrency 20 --duration 180 --think-ms 0 --tag 优化后

# 剖析（**先停服务**，脚本自建引擎约 2.4 GB）
pwsh -NoProfile -File run_py.ps1 研发/剖析/剖析问答链路.py --tag 优化后 --n 8 --profile
pwsh -NoProfile -File run_py.ps1 研发/剖析/内存剖析.py --tag 优化后 --rounds 6 --per-round 12

# 单元测试
pwsh -NoProfile -File run_py.ps1 -m pytest 测试 -q
```

---

## 新增端点（工单17）

| 端点 | 说明 |
| --- | --- |
| `POST /api/v1/uploads` | 上传 PDF。优化后立即 `202 + task_id`；队列满 `429`。优化前同步解析后 `200` |
| `GET /api/v1/tasks` | 任务列表 + 队列深度 + 各状态计数 |
| `GET /api/v1/tasks/{id}` | 单任务状态（`pending/running/done/failed` + 页数/字符数/耗时/worker） |
| `POST /api/v1/admin/unload` | 主动卸载 Ollama 模型（`keep_alive="0s"`），仅回环可调 |
| `GET /api/v1/perf` | 性能自观测快照（限流/缓存/连接池/单例/队列/Ollama） |

---

## 十项修复（F1~F10）速查

| 编号 | 一句话 | 开关 |
| --- | --- | --- |
| F1 | 外部 HTTP 走 keep-alive 连接池，不再每请求新建 TCP | `PERF_HTTP_POOL_ENABLED` |
| F2 | 嵌入后端探测结果缓存，不再每请求打一次 `/api/tags` | `PERF_PROBE_CACHE_TTL_S` |
| F3 | 入口有界限流，超出立即 429 而不是无限堆积 | `PERF_MAX_INFLIGHT` / `PERF_QUEUE_MAX` |
| F4 | 问答路由 `async def` + `run_in_threadpool`，事件循环不被阻塞 | `PERF_ASYNC_ROUTES` |
| F5 | 答案内存缓存加锁 + TTL + 上限 + 条目瘦身 | `PERF_ANSWER_CACHE_MAX/TTL_S` / `PERF_CACHE_CHUNK_CHARS` |
| F6 | sqlite WAL + 批量提交 + `answer_cache` 表条数上限 | `PERF_SQLITE_*` / `PERF_ANSWER_CACHE_TABLE_MAX` |
| F7 | Ollama `keep_alive` 可配 + 主动卸载 | `PERF_OLLAMA_KEEP_ALIVE` |
| F8 | 会话文件卷条数上限 + TTL 物理清理 | `PERF_SESSION_FILE_MAX` / `PERF_SESSION_SWEEP_S` |
| F9 | 文档解析有界队列 + 工作线程池（对应 RAGFlow 的 Task Executor） | `PERF_PARSE_*` |
| F10 | `/health` 结果 TTL 缓存，探针不打重活 | `PERF_HEALTH_CACHE_S` |

**总开关**：`PERF_ENABLED=0` 时全部关闭，行为与工单10 基座**逐位一致**（用于复现基准与回滚）。

---

## 纪律

* **零新增依赖**：全部新增能力用标准库（`http.client` / `threading` / `queue` /
  `concurrent.futures` / `ctypes` / `tracemalloc` / `cProfile`）。
* **禁止多进程**：只用 `ThreadPoolExecutor`，压测并发硬上限 20。
  自检（大小写敏感，避免误报 `threadpool(`）：
  ```powershell
  Get-ChildItem 研发 -Recurse -Include *.py |
    Select-String -CaseSensitive -Pattern 'multiprocessing|ProcessPoolExecutor|Pool\(|joblib|n_jobs'
  # → 0 命中
  ```
* **阈值配置化**：所有判断阈值在 `service/perf_config.py` / `service/config.py` /
  `app/core/config.py` 三处收敛，代码里不写死。
* **降级不裸崩**：连接池/队列/缓存/Redis/sqlite 任一环失败都写 WARNING 后走原路径。
* **索引只读**：`RAG_DATA__INDEX_DIR` 指向工单7 真实索引，本目录**不含 data/**。
