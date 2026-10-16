# 部署说明（工单12 / LightRAG 优化）

> **工单编号：人工智能NLP-RAG 项目-LightRAG 优化**
> 工作目录：`E:\gao6gongdan\工单12`｜解释器：`E:\gao6gongdan\工单1\.venv\Scripts\python.exe`（`run_py.ps1` 已内置该默认值）

## 1. 怎么跑（照抄即可，全部实测过）

```powershell
cd E:\gao6gongdan\工单12
$env:PYTHONDONTWRITEBYTECODE='1'

# ① 语料块（复用已解析产物；PDF 首次解析见 研发/scripts/01_parse_pdf.py）
& .\run_py.ps1 研发/scripts/10_build_corpus.py

# ② 知识图谱：小样本抽样抽取（默认 200 块，串行、单模型；绝不跑全量 3013）
& .\run_py.ps1 研发/scripts/20_build_graph.py

# ③ 图谱统计 + 可视化（落盘 研发/data/graph/ + 测试/测试截图/知识图谱可视化.png）
& .\run_py.ps1 研发/scripts/21_graph_stats.py

# ④ 16 题 × 三后端（rag / lightrag / lightrag_strict）检索 + 生成；实测 83 秒
& .\run_py.ps1 研发/scripts/30_compare_rag_vs_lightrag.py --no-resume

# ⑤ RAGAS 四项（自实现 _kw 降级口径）+ 中文命名截图 + 对比报告
& .\run_py.ps1 研发/scripts/31_ragas_eval.py
& .\run_py.ps1 研发/scripts/32_make_screenshots.py

# ⑥ 测试（离线用例，秒级）
& .\run_py.ps1 -PyArgs '-m','pytest','测试','-q'
```

## 1.1 两段最常用的跑法

**怎么跑 16 题对比**（RAG vs LightRAG，唯一变量是检索通道；两后端共用同一提示词/模型/生成参数）

```powershell
cd E:\gao6gongdan\工单12
# 全量 16 题（--no-resume 忽略已有逐题结果重跑；约 83 秒）
& .\run_py.ps1 研发/scripts/30_compare_rag_vs_lightrag.py --no-resume
# 冒烟：只跑前 3 题、只检索不生成（不调 LLM，秒级）
& .\run_py.ps1 研发/scripts/30_compare_rag_vs_lightrag.py --limit 3 --no-generate
```
产物：`研发\评估\逐题结果.jsonl`（48 行 = 16 题 × 3 后端）、`优化\评估结果\对比汇总.json` / `.csv` / `对比报告.md`。

**怎么跑 RAGAS**（官方包缺失 → 自实现 `_kw` 四项，口径差异见结果文件）

```powershell
& .\run_py.ps1 研发/scripts/31_ragas_eval.py          # 依赖上一步的逐题结果
& .\run_py.ps1 -PyArgs '-m','pytest','测试','-q'      # 26 用例，含 RAGAS 字段完整性校验
```
产物：`优化\评估结果\RAGAS指标.json`（含 3 后端汇总 + 48 条逐题明细 + 口径差异说明）与 `研发\评估\RAGAS指标.json`。

> ⚠️ `run_py.ps1` 用 `ValueFromRemainingArguments` 收集参数，`-m` 这类带横线的参数会被
> PowerShell 当成脚本自己的参数而报错，**必须**写成 `-PyArgs '-m','pytest',...`；传脚本路径时可直接位置传参。

## 2. 依赖清单

| 依赖 | 用途 | 是否需新装 |
| --- | --- | --- |
| `pymupdf`(fitz) | PDF 文本/表格/图片抽取 | 已装（共享解释器自带） |
| `numpy` | 向量与统计计算 | 已装 |
| `Pillow` | 图谱可视化 PNG 渲染 | 已装 |
| Ollama `qwen2.5:3b` | 实体关系抽取 + 答案生成 | 本机 `127.0.0.1:11434` 已运行 |
| Ollama `bge-m3:latest` | 1024 维嵌入（复用，不微调） | 同上 |

**未执行任何 `pip install`**（`_共享约定.md` §2 pip 落点红线）；向量索引与嵌入模型复用 `工单7` 产物。

## 3. 小样本口径声明（验收标准 1.6 要求三处声明之一）

> 以下一段与 `研发\data\graph\graph_meta.json` 的 `抽样口径` 字段**逐字一致**（t5 联动核验用），
> 由 `研发\lightrag\graph_store.sampling_caliber_text()` 统一生成：

小样本建图抽样口径：实际建图 88 块 / 全量 3013 块（2.9%），抽样计划 200 块（实际因时间预算先行收束）｜随机种子 20250826｜信息密度下限 150 字符｜table+image 加权｜每块 1 次 LLM 调用串行（batch_chunks=1）｜按 doc_key 分层（招股说明书1无水=59、招股说明书2pd=61）｜只保留 招股说明书1无水 + 招股说明书2pd，剔除带水印重复件 招股说明书1pd。

- **为何是 88 而非 200**：试点实测单块 ≈4.7~9.1 秒，400 块预估 ≈35 分钟，超出「约 10 分钟量级」目标；
  设计口径先降到 200 块，实跑在 1200 秒时间预算处自然收束于 **88 块**（已抽块全部保留，无浪费）。
- 另两处声明见 `优化\评估结果\对比报告.md` 与 `设计\验收标准.md`。

## 4. 产物落点

| 产物 | 路径 |
| --- | --- |
| 生效配置（阈值唯一来源） | `部署\配置\lightrag.json` |
| 本次跑数生效配置快照 | `部署\日志\effective_config.json` |
| 图谱（实体/关系/GraphML/统计/口径） | `研发\data\graph\` |
| 图谱可视化 PNG | `测试\测试截图\知识图谱可视化.png` |
| 16 题对比与 RAGAS 指标 | `优化\评估结果\` |
| 运行日志 | `部署\日志\`、`E:\gao6gongdan\_共享工具\运行日志\` |
| 交付物核验（脚本自动生成） | `部署\交付物核验.md` |
