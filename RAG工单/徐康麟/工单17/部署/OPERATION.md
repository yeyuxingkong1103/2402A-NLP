# OPERATION.md —— 运维文档

**工单编号**：`人工智能NLP-RAG-解决API 服务并发瓶颈与资源泄漏工单`
**适用读者**：日常监控、排障、容量规划的值班运维

---

## 1. 日常监控（每天 3 分钟）

### 1.1 三档探针（用途不同，别混用）

| 端点 | 用途 | 期望 | 特点 |
| --- | --- | --- | --- |
| `GET /healthz` | **存活**（容器 HEALTHCHECK） | 200 | 完全不碰索引，毫秒级 |
| `GET /readyz` | **就绪**（编排摘流量依据） | 200；索引不可用 → 503 | 会真读索引元信息 |
| `GET /health` | **体检**（人看降级原因） | 始终 200，看 `checks.*.ok` | 结果带 5 s TTL 缓存（`PERF_HEALTH_CACHE_S`） |

```powershell
$h = (Invoke-WebRequest http://127.0.0.1:8200/health -UseBasicParsing).Content | ConvertFrom-Json
$h.status; $h.ready; $h.uptime_s
$h.checks.PSObject.Properties | ForEach-Object { "{0,-16} ok={1}" -f $_.Name, $_.Value.ok }
```

**每日必看**：`checks.engine.ok`（索引+Ollama）、`checks.session_store.ok`（Redis 或文件卷）。
`status="degraded"` 时逐个看 `reason` 字段，不要猜。

### 1.2 性能与容量（`GET /api/v1/perf`）

```powershell
$p = (Invoke-WebRequest http://127.0.0.1:8200/api/v1/perf -UseBasicParsing).Content | ConvertFrom-Json
$p.limiter          # inflight / queued / peaked / accepted / rejected / timeouts
$p.answer_cache     # 条目数 / 上限 / TTL / 淘汰数 / 过期数
$p.http_pool        # created / reused / idle / reuse_ratio
$p.task_queue       # queue_depth / done / failed / rejected
$p.ollama           # keep_alive / resident（应 <=1 个模型）
$p.storage          # sqlite 表计数 / 清理数 / 提交数
```

**健康画像（可直接当告警阈值）**

| 指标 | 正常 | 告警条件 | 含义与处置 |
| --- | --- | --- | --- |
| `limiter.rejected` | 缓慢增长 | 1 分钟内 > 100 次 | 过载。看 §3.1 |
| `limiter.inflight` | < `max_inflight` | 长期贴住上限 | 后端能力不足或 `max_inflight` 偏大 |
| `limiter.queued` | 0~少量 | 长期 > 0 且 `rejected` 上升 | 队列在堆积 → 扩容或降 `max_inflight` |
| `http_pool.reuse_ratio` | > 0.8 | < 0.5 | 连接没复用上（配置被关或对端关连接） |
| `answer_cache` 条目 | ≤ 上限，有淘汰 | 长期贴住上限且淘汰频繁 | 缓存不够用（调大 `PERF_ANSWER_CACHE_MAX`）或问句太散 |
| `task_queue.queue_depth` | 尽快归零 | 持续 > 0 | 解析能力不足（加 `PERF_PARSE_WORKERS`） |
| `ollama.resident` | **1 个** | ≥ 2 个 | **立即** `POST /api/v1/admin/unload`，见 §3.4 |
| 进程私有内存 | 2.4 GB 上下小幅波动 | 单调上涨 | 见 §3.2 泄漏排查 |

### 1.3 Prometheus 指标（`GET /metrics`）

工单17 新增指标：

```
rag_api_inflight / rag_api_queued / rag_api_inflight_peak / rag_api_max_inflight
rag_api_rejected_total / rag_api_gate_timeout_total
rag_api_uploads_total / rag_api_upload_rejected_total
rag_http_pool_created_total / rag_http_pool_reused_total / rag_http_pool_idle
rag_answer_cache_entries / _evictions_total / _expired_total
rag_parse_queue_depth / rag_parse_tasks_total{status="done|failed|rejected"}
rag_singleton_probe_hits_total / _misses_total{cache="backend_probe"}
```

**建议告警规则**

```yaml
- alert: RAG过载拒绝
  expr: rate(rag_api_rejected_total[5m]) > 0.5      # >0.5 次/秒
  for: 5m
  annotations: {summary: "问答入口持续 429，检查 PERF_MAX_INFLIGHT 与后端能力"}
- alert: RAG队列堆积
  expr: rag_parse_queue_depth > 10
  for: 5m
- alert: 连接未复用
  expr: rate(rag_http_pool_created_total[5m]) > rate(rag_http_pool_reused_total[5m])
  for: 10m
- alert: 答案缓存打满
  expr: rag_answer_cache_entries >= 0.95 * 256
  for: 30m
```

### 1.4 GPU / Ollama 驻留

```powershell
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv
(Invoke-RestMethod http://127.0.0.1:11434/api/ps) | ConvertTo-Json -Depth 4
pwsh -NoProfile -File 研发/剖析/探GPU.ps1 -Watch 60 5   # 连续采样落盘
```

**基线**：RTX 2060（6144 MiB）。空载约 **700~950 MiB**；`qwen2.5:3b` 驻留时 +2.01 GiB；
`bge-m3` 驻留时 +0.63 GiB。**两个同时驻留 ≈ 2.7 GB（占 44% 显存）—— 不允许**。

---

## 2. 备份与清理

| 数据 | 位置（本机等价部署） | 位置（容器） | 保留策略 |
| --- | --- | --- | --- |
| 业务日志 | `部署/卷/logs/*.log`、`rag_trace.jsonl` | 卷 `rag-log` → `/app/logs` | 建议 14 天 |
| 会话 | `部署/卷/sessions/*.json` | 卷 `rag-session` → `/app/sessions` | 上限 `PERF_SESSION_FILE_MAX`（默认 2000），超 TTL 自动物理清理 |
| 问答历史/答案缓存 | `部署/卷/runtime/qa_runtime.sqlite3` | 卷 `rag-runtime` → `/app/data/runtime` | 缓存表条数上限 `PERF_ANSWER_CACHE_TABLE_MAX`（默认 2000） |
| 上传文件 | `部署/卷/uploads/` | 卷 `rag-upload` → `/app/data/uploads` | 按需清理 |
| 解析任务留痕 | `部署/日志/parse_tasks.jsonl` | `/app/logs/parse_tasks.jsonl` | 建议 30 天 |
| 索引 | `E:\gao6gongdan\工单7\研发\data\index` | 卷 `rag-index`（`:ro`） | **只读，勿在运行期改** |

```powershell
# 备份（卷目录直接拷走即可；sqlite 建议先停服务或用 .backup）
Copy-Item 部署\卷\runtime\qa_runtime.sqlite3 备份\qa_runtime-$(Get-Date -f yyyyMMdd).sqlite3
# 只读打开历史（不影响服务）
python -c "import sqlite3;c=sqlite3.connect('file:部署/卷/runtime/qa_runtime.sqlite3?mode=ro',uri=True);print(c.execute('select count(*) from qa_history').fetchone())"
```

---

## 3. 排障手册（症状 → 定位 → 处置）

### 3.1 响应变慢 / 大量 429

**症状**：客户端报 429，或 `rag_api_rejected_total` 快速上升，`limiter.queued` 长期 > 0。

```powershell
$p = (Invoke-WebRequest http://127.0.0.1:8200/api/v1/perf -UseBasicParsing).Content | ConvertFrom-Json
$p.limiter
(Invoke-WebRequest http://127.0.0.1:8200/metrics -UseBasicParsing).Content -split "`n" | Select-String "inflight|queued|rejected"
```

**判断顺序**：

1. `ollama.resident` 是否 ≥ 2 个模型 → 是则先卸载（§3.4），这通常直接解决。
2. Ollama 是否健康：`Invoke-RestMethod http://127.0.0.1:11434/api/ps`；不通用 `ollama serve` 拉起。
3. 是「能力不足」还是「配置太紧」：
   - `inflight` 长期贴住 `max_inflight` 且后端 GPU 利用率已满 → **能力不足**（加机器 / 换更小模型）。
   - `inflight` 很低但 `queued` 高 → 有慢请求卡住（看 §3.3）。
4. 应急降级：调大 `PERF_MAX_INFLIGHT` 只会在后端前面排更长的队，**不会提升吞吐**；
   真正的应急手段是**降低单请求成本**（`RAG_RETRIEVAL__TOP_K` 调小、`PERF_OLLAMA_KEEP_ALIVE` 调短、
   或临时切 `RAG_LLM__BACKEND=extractive` 走确定性合成器，不调 LLM）。

### 3.2 内存持续上涨

**症状**：进程私有内存单调上升；容器被 OOM Killer 杀掉重启。

```powershell
# ① 看进程内存曲线（每 5 s 落盘）
pwsh -NoProfile -File 研发/剖析/探GPU.ps1 -Watch 300 5
# ② 三分钟定位泄漏对象（**先停服务**，脚本自建引擎约 2.4 GB，避免叠加）
pwsh -NoProfile -File 部署/脚本/本地停止.ps1
$env:PERF_ENABLED='1'
pwsh -NoProfile -File run_py.ps1 研发/剖析/内存剖析.py --tag 巡检 --rounds 6 --per-round 12
```

**看什么**：`tracemalloc_巡检.json` 的 `tracemalloc_top` ——
`size_diff_kb` **为正且随轮次持续增长**的调用点才是泄漏；一次性大分配不算。

**已知会涨的三处（本工单均已加上界）**：

| 位置 | 机制 | 上界开关 |
| --- | --- | --- |
| `app/core/qa_engine.py` 答案内存缓存 | 每条缓存含 `chunks/evidence` 全文 | `PERF_ANSWER_CACHE_MAX` / `PERF_ANSWER_CACHE_TTL_S` / `PERF_CACHE_CHUNK_CHARS` |
| `app/storage/sqlite_manager.py` `answer_cache` 表 | 只增不减 | `PERF_ANSWER_CACHE_TABLE_MAX` |
| `service/session_store.py` 会话文件 | 无物理删除 | `PERF_SESSION_FILE_MAX` / `PERF_SESSION_SWEEP_S` |

**如果上限都被调成 0 了** → 那就是基座行为，泄漏会复现。用 `/api/v1/perf` 核对生效值。

### 3.3 单个请求卡很久

```powershell
# 看服务端每请求一行 JSON 日志里的分段耗时
Get-Content 部署\卷\logs\app.log -Tail 200 | Select-String "api.ask" |
  ForEach-Object { $_.Line | ConvertFrom-Json } |
  Sort-Object elapsed_ms -Descending | Select-Object -First 10 trace_id,elapsed_ms,retrieve_ms,generate_ms
```

- `retrieve_ms` 大 → 检索侧（索引冷、BM25 首建、FAISS 争用）。
- `generate_ms` 大 → Ollama；查 `/api/ps` 是否在换模型（换模型会有一次冷加载）。
- 两者都小但 `elapsed_ms` 大 → 在闸门排队（看 `limiter.queued`）或落盘慢（看 `storage.commits_total` 增速）。

### 3.4 GPU 显存被占住 / 模型不释放

```powershell
nvidia-smi --query-gpu=memory.used --format=csv,noheader      # 基线约 700~950 MiB
(Invoke-RestMethod http://127.0.0.1:11434/api/ps).models | Select-Object name,size_vram,expires_at
Invoke-WebRequest -Method Post http://127.0.0.1:8200/api/v1/admin/unload -UseBasicParsing | Select-Object -Expand Content
nvidia-smi --query-gpu=memory.used --format=csv,noheader      # 卸载后应回落到基线
```

**实测证据**（本工单）：卸载前 **3102 MiB**（qwen2.5:3b + bge-m3 双驻留）→
卸载后 **947 MiB**。

**根因说明**：基座 `app/core/llm_client.py:196,235` 把 `keep_alive` **写死成 `"30m"`**，
且**没有任何卸载路径** —— 最后一次请求之后显存还要被占 30 分钟。
本工单改为读 `PERF_OLLAMA_KEEP_ALIVE`（默认 `5m`）并提供 `POST /api/v1/admin/unload`。

### 3.5 上传解析不完成 / 429

```powershell
(Invoke-WebRequest http://127.0.0.1:8200/api/v1/tasks -UseBasicParsing).Content | ConvertFrom-Json
Get-Content 部署\日志\parse_tasks.jsonl -Tail 20
```

- 队列满返回 429 + `Retry-After`：说明解析速度跟不上上传速度 → 加 `PERF_PARSE_WORKERS`。
- 任务 `failed`：`error` 字段有原因（加密 PDF / 损坏 / 超时）；**服务不会因此崩**。
- `PERF_ENABLED=0` 时没有队列（`/api/v1/tasks` 返回 `mode="sync-in-request"`），
  上传是**同步解析**，会拖慢问答 —— 生产必须开 `PERF_ENABLED=1`。

### 3.6 Redis 挂了

`/health` 的 `checks.session_store.redis.ok=false`、`degraded_reason` 有原因。
**行为**：自动降级到文件卷（双写机制保证数据不丢），问答不受影响。
Redis 恢复后 30 秒内自动半开重试（熔断器 `_BREAKER_COOLDOWN_S`）。

---

## 4. 常规操作手册

| 操作 | 命令 |
| --- | --- |
| 起服务（生产口径） | `pwsh -NoProfile -File 部署/脚本/起服务.ps1 -PerfEnabled 1` |
| 停服务 | `pwsh -NoProfile -File 部署/脚本/本地停止.ps1` |
| 健康巡检 | `pwsh -NoProfile -File 部署/脚本/health.ps1` |
| 看日志 | `pwsh -NoProfile -File 部署/脚本/logs.ps1` |
| 起压测 | `pwsh -NoProfile -File run_py.ps1 研发/压测/load_runner.py --scenario A --concurrency 20 --duration 180` |
| 主动卸模型 | `Invoke-WebRequest -Method Post http://127.0.0.1:8200/api/v1/admin/unload -UseBasicParsing` |
| 跑单元测试 | `pwsh -NoProfile -File run_py.ps1 -m pytest 测试 -q` |
| 生成截图 | `pwsh -NoProfile -File run_py.ps1 测试/测试脚本/生成截图.py` |
| 重算对比 | `pwsh -NoProfile -File run_py.ps1 研发/scripts/汇总对比.py` |

---

## 5. 容量规划参考（实测数字）

| 场景 | 并发 | 实测 RPS | P95 | 说明 |
| --- | --- | --- | --- | --- |
| 场景A 固定会话（缓存友好） | 20 | 98 | 229 ms | 优化后 |
| 场景A 多用户（每请求新会话） | 20 | 98 | 395 ms | 优化后（优化前 9.9 RPS / 3608 ms） |
| 场景B 混合（5 问答 + 5 上传解析） | 10 | 10.8 | 问答 604 ms | 优化后（优化前 4.9 RPS / 8824 ms） |
| soak | 8 | — | — | 15 分钟无增长趋势 |

**结论**：单实例（1×RTX 2060 + 8 核）在**缓存命中为主**时约可承载 **~100 RPS**；
**冷路径（每次真检索+生成）** 的吞吐受 Ollama 串行推理限制，实测约 **1.5~2 QPS/并发**。
要提吞吐只能：① 加大缓存命中率；② 换更快的推理后端；③ 多实例 + 共享 Redis/索引。

---

## 6. 值班检查清单（打印版）

```
□ /health  → status=ok，checks.*.ok 全 true
□ /api/v1/perf → limiter.rejected 增速正常（<0.5/s）
□ /api/v1/perf → ollama.resident <= 1
□ /api/v1/perf → http_pool.reuse_ratio > 0.8
□ /api/v1/perf → answer_cache 未长期打满
□ /api/v1/perf → task_queue.queue_depth 归零
□ nvidia-smi  → 显存 < 2000 MiB（空载/单模型）
□ 进程私有内存 → 2.4 GB 上下波动，非单调上涨
□ 日志无 ERROR / traceback
□ 磁盘：部署/卷 与 部署/日志 增长在预期内
```
