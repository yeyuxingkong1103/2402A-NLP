# 工单8 部署说明（README）

> 工单编号：**人工智能NLP-RAG-基于Graph RAG 实现金融问答**
> 目录：`E:\gao6gongdan\工单8\部署\`

---

## 1. 一键部署/跑通

```powershell
cd E:\gao6gongdan\工单8

# 全流程：建图谱 → 可视化 → 17 题评估 → 与工单7 对比 → 优化对比 → 截图 → pytest → 核验
pwsh -NoProfile -File 部署\脚本\一键跑通.ps1

# 常用变体
pwsh -NoProfile -File 部署\脚本\一键跑通.ps1 -SkipEval        # 跳过 17 题评估（省约 25 分钟）
pwsh -NoProfile -File 部署\脚本\一键跑通.ps1 -SkipBuild       # 复用已有图谱
pwsh -NoProfile -File 部署\脚本\一键跑通.ps1 -LlmSample 400   # 额外做 Ollama 关系抽样补抽（≤400 块）

# 启动问答界面（端口 8124）
pwsh -NoProfile -File 部署\脚本\启动界面.ps1
```

**前置条件**（已满足，列出以便迁移到别的机器时核对）：

| 条件 | 值 |
| --- | --- |
| Python 解释器 | `E:\gao6gongdan\工单1\.venv\Scripts\python.exe`（3.11.15） |
| 工单7 索引（只读） | `E:\gao6gongdan\工单7\研发\data\index\bge-m3_1024`（8847 块） |
| 工单7 冻结评估结果（只读） | `E:\gao6gongdan\工单7\测试\测试数据\评估汇总.json` / `评估结果.jsonl` |
| Ollama | `127.0.0.1:11434`，需有 `qwen2.5:3b` 与 `bge-m3:latest` |
| 可用内存 | ≥ 4000 MB（脚本会自检，不足直接拒绝启动） |

## 2. 目录说明

```
部署/
├── 脚本/
│   ├── 一键跑通.ps1        全流程（含内存自检与日志重定向）
│   ├── 启动界面.ps1        启动 Web 问答界面（8124）
│   └── 核验交付物.py        交付物清单/工单编号注释/禁用写法/截图/真实数据自检
├── 配置/
│   └── GraphRAG配置快照.json  阈值与路径快照（真源：研发/app/graphrag/config.py）
├── 日志/                   所有运行日志（重定向落这里，不刷终端）
├── README.md               本文件
└── 交付物核验.md            核验报告（自动生成）
```

## 3. 日志清单

| 日志 | 内容 |
| --- | --- |
| `build_graph_stdout.log` | 图谱构建：扫描块数、剪枝计数、产物路径 |
| `render_viz.log` | 可视化：节点/边数、布局耗时、外链检查 |
| `run_eval_stdout.log` | 17 题评估：逐题覆盖率/召回/耗时 + 汇总 |
| `build_comparison.log` | 与工单7 并排对比结果 |
| `opt_compare.log` | 优化前后对比 |
| `shots.log` | 5 张截图的生成结果 |
| `pytest.log` | 单元测试输出 |
| `verify.log` | 交付物核验输出 |
| `app.log` / `error.log` / `rag_trace.jsonl` | 引擎结构化日志（复用 工单7 的 logging_conf） |

> 日志均为 JSON 行格式（`ts/level/event/...`），便于 `Select-String` 追溯「数字来自哪一步」。

## 4. 资源配置与纪律（`_共享约定.md` §5.5）

| 项 | 本工单实测 | 措施 |
| --- | --- | --- |
| 进程模型 | **单进程** | 全链路无 `multiprocessing` / 进程池 / `joblib`（核验脚本会扫） |
| 图谱构建峰值内存 | < 400 MB | 流式读分块、边跑边落盘 |
| 索引加载 | 8847 块 / 4.3 s | 复用 工单7 索引；FAISS 走 memmap 反序列化（零拷贝） |
| Ollama 模型驻留 | 单模型 | `keep_alive="120s"`（会话内驻留、闲置 2 分钟自动卸载）；评估收尾显式卸载；**禁止** qwen2.5:3b 与 bge-m3 同时常驻 |
| LLM 抽取规模 | ≤ 400 块（默认 0 = 关闭） | 规则抽取为主，抽样补抽可选 |
| 长日志 | 全部重定向到 `部署\日志\` | 一键脚本已内置 |

## 5. 常见问题

| 现象 | 原因 | 处置 |
| --- | --- | --- |
| `IndexMissingError` | 工单7 索引路径不可达 | 确认 `E:\gao6gongdan\工单7\研发\data\index\bge-m3_1024` 存在；或用 `GRAPHRAG_INDEX_DIR` 指定 |
| 图谱不可用（`ready=false`） | 未构建图谱 | 跑 `研发/scripts/build_graph.py` |
| 答案变短、出现「抽取式兜底」 | Ollama 不可用或模型名不符 | `ollama list` 确认有 `qwen2.5:3b`；`GET /health` 查看 `checks.llm` |
| 可用内存不足 4000 MB | 有并行重活在跑 | 等其它任务结束，或加 `-SkipMemoryCheck`（自担风险） |
| 端口被占用 | 8124 已被别的程序占用 | `启动界面.ps1 -Port 8125` |
| 首问较慢（数秒） | 索引加载 + 生成模型冷启动 | 界面默认预热（`warmup(with_llm=True)`）；可用 `-NoWarmup` 跳过 |
