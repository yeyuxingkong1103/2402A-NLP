# 工单9 部署与运行说明

> 工单编号：**人工智能NLP-RAG-Graph RAG 优化任务**

---

## 1. 目录

```
部署/
├── 脚本/
│   ├── 一键跑通.ps1      全流水线（探内存 → 参考答案集 → 抽取层优化 → pytest → 对比评估 → 稳定性 → 多语言 → 截图）
│   ├── 启动界面.ps1      Web 界面（端口 8125，支持 -Background / -OpenBrowser）
│   └── 核验交付物.py     交付物自检（目录/文档/截图/资源纪律/只读边界）
├── 配置/
│   ├── 优化配置.json        ★ 可编辑的优化开关与阈值（改完立即生效）
│   └── 优化配置快照.json    运行时落盘的「生效配置」（可追溯每个数字）
├── 日志/                  所有运行日志（结构化 JSON 日志 + 脚本输出重定向）
├── README.md（本文件）
└── 交付物核验.md
```

---

## 2. 环境要求

| 项 | 值 |
| --- | --- |
| 解释器 | `E:\gao6gongdan\工单1\.venv\Scripts\python.exe`（Python 3.11.15） |
| 运行入口 | 工单根目录的 `run_py.ps1`（已写好解释器默认值） |
| Ollama | `127.0.0.1:11434` 需在运行；模型 `qwen2.5:3b`（生成）+ `bge-m3:latest`（嵌入，工单7 索引已用） |
| 索引 | 工单7 `研发/data/index/bge-m3_1024`（8847 块）**只读引用**，不需要重建 |
| 图谱 | `研发/data/图谱优化后`（6717 节点 / 3807 边）；缺失时引擎会回退到 工单8 原始图谱 |
| 磁盘 | 建议 ≥2 GB 空闲（本工单不复制索引/模型） |

---

## 3. 一键跑通

```powershell
cd E:\gao6gongdan\工单9
pwsh -NoProfile -File 部署/脚本/一键跑通.ps1
# 只跑检索与模板快路径（秒级，用于快速验收）
pwsh -NoProfile -File 部署/脚本/一键跑通.ps1 -SkipEval -Repeat 10
```

脚本会：
1. **探内存**（可用 <2000 MB 或提交比 >80% 直接中止 —— 资源纪律，本机曾被搞崩两次）；
2. 构建参考答案集 → 抽取层优化 → pytest → 全量对比评估 → 稳定性/容错 → 多语言 → 截图；
3. 收尾：清理 `__pycache__` / `.pytest_cache` 并请求卸载 Ollama 模型。

---

## 4. 启动 Web 界面（端口 8125）

```powershell
# 前台（Ctrl+C 停止）
pwsh -NoProfile -File 部署/脚本/启动界面.ps1

# 后台 + 自动打开浏览器
pwsh -NoProfile -File 部署/脚本/启动界面.ps1 -Background -OpenBrowser

# 直接跑（等价）
pwsh -NoProfile -File run_py.ps1 研发/app/ui/serve_fallback.py --port 8125
```

页面地址：**http://127.0.0.1:8125/**

**页面展示四要素**（工单要求）：

| 区块 | 内容 |
| --- | --- |
| ① 问题 | 输入框 + 6 个中英文示例问题快捷按钮（含跨文档对比题） |
| ② 答案 | 答案全文（含 `【结论】/【关键数据】/【图谱依据】/【引用】`）；拒答时显示护栏文案；下方一排徽章显示生成方式与分段耗时 |
| ③ 引用 | 每条：`[块N] 公司 第N页 · chunk_id` |
| ④ 检索到的上下文 | **【知识图谱三元组】**区块（`head --关系--> tail（置信度；来源块）`）+ 每条上下文（名次 / 通道 / 页码 / 类型 / 融合分 / 图谱实体 / 正文） |
| ⑤ 图谱结构 | 问句命中的局部子图（命中实体 + 节点/边数 + 边表） |

**HTTP 接口**：`GET /`、`GET /healthz`、`GET /api/stats`、`GET /api/demos`、`POST /api/ask`
（契约见 `设计/接口设计.md` §2）。任何异常都返回**结构化 JSON 错误**，不 500 裸崩。

---

## 5. 配置（阈值全部配置化）

`部署/配置/优化配置.json` —— 改完**立即生效**，不需要动代码。覆盖优先级：

```
app/graphrag_opt/配置.py::DEFAULTS   →   部署/配置/优化配置.json   →   环境变量 GRAPHRAGOPT_<段>__<键>
```

常用开关：

| 键 | 默认 | 作用 |
| --- | --- | --- |
| `extract.enabled` | true | 抽取层优化总开关（false = 用工单8 原始图谱） |
| `retrieval.fusion_weights` | `{base:0.62, graph:0.23, feature:0.15}` | 图特征 × 向量/BM25 融合权重 |
| `retrieval.hop` / `hop_when_linked` | 2 / 1 | 自适应跳数（锁定语料时降到 1 跳） |
| `retrieval.field_probe` | true | 字段锚点召回（上下文召回的主要来源） |
| `retrieval.evidence_k` | 14 | 进生成阶段的证据块条数 |
| `retrieval.evidence_quota_focus` / `_probe` | 7 / 5 | 焦点页 / 字段锚点的名额配额 |
| `retrieval.probe_expand` | **false** | 锚点页兄弟块扩展（实测净收益为负，默认关闭） |
| `generation.template_first` | true | 模板化快路径（SLA 的关键；false 则每题都调 LLM） |
| `generation.keep_alive` | `"0s"` | Ollama 用完立刻卸载（资源纪律硬性要求） |
| `eval.context_precision_target` / `context_recall_target` / `sla_ms` | 0.80 / 0.90 / 3000 | 达标判定阈值 |

**单次实验覆盖示例**：

```powershell
$env:GRAPHRAGOPT_RETRIEVAL__FIELD_PROBE='false'       # 关掉字段锚点召回（消融）
$env:GRAPHRAGOPT_GENERATION__TEMPLATE_FIRST='false'   # 回退到全 LLM 生成
pwsh -NoProfile -File run_py.ps1 研发/评估/run_eval_opt.py --profiles after --no-llm --tag _消融
```

---

## 6. 日志

| 文件 | 内容 |
| --- | --- |
| `部署/日志/app.log` | 结构化 JSON 日志（检索/生成/图谱扩展/配置告警） |
| `部署/日志/run_eval_opt.log` | 对比评估全过程（逐题指标 + 汇总） |
| `部署/日志/稳定性压测.log`、`容错.log` | 稳定性与容错测试输出 |
| `部署/日志/多语言实测.log` | 中英配对实测 |
| `部署/日志/pytest.log` | 单元测试 |
| `部署/日志/构建优化图谱.log` | 抽取层优化（归并/规范化/剪枝计数） |
| `部署/日志/serve_8125.log` | Web 界面启动与预热 |

---

## 7. 运维注意（本机实测踩坑）

1. **端口 8125 只允许一个进程**。重启前先确认没有残留监听：
   ```powershell
   Get-NetTCPConnection -LocalPort 8125 -State Listen -ErrorAction SilentlyContinue
   ```
   `job_kill` 只杀 pwsh 包装进程，**python 子进程可能仍占着端口** ——
   此时新服务绑定失败、旧服务继续应答（实测踩过：改了代码却拿到旧答案）。
   处置：`Get-Process python` 找到启动时间较新的、`Stop-Process -Id <pid> -Force`。
2. **`app/core` 是从 工单7 复制的**，其 `config.py::REPO_ROOT` 按文件位置计算 ⇒
   必须显式设 `RAG_LOG__DIR` 指向本工单，否则日志会写进**只读的工单7 目录**。
   本工单已在 `OptimizedEngine.__init__` 与 `run_eval_opt._logger` 两处强制覆盖。
3. **Ollama 单模型**：所有请求带 `keep_alive="0s"`；同时驻留两个模型曾把提交内存推到 96% 触发 OOM。
4. **收工清理**：
   ```powershell
   Get-ChildItem -Recurse -Directory -Include '__pycache__','.pytest_cache' | Remove-Item -Recurse -Force
   Get-Process python    # 确认没有本工单遗留的进程
   ```
