# 工单9 研发说明（可运行源代码）

> 工单编号：**人工智能NLP-RAG-Graph RAG 优化任务**

---

## 1. 目录结构

```
研发/
├── app/
│   ├── graphrag_opt/          ★ 本工单新增：三层面优化
│   │   ├── 配置.py            全部优化开关与阈值（唯一真源）
│   │   ├── 抽取层.py           实体/关系构建层优化（别名归并/规范化/剪枝）
│   │   ├── 检索层.py           检索层优化（融合/裁剪/扩展/锚点）
│   │   ├── 生成层.py           生成层优化（三元组/引用/拒答/模板）
│   │   └── 引擎.py             端到端编排（profile=before|after）
│   ├── graphrag/              ← 复用 工单8（图谱存储 / schema / 确定性草稿 / LLM 客户端）
│   ├── core/                  ← 复用 工单7（检索 / 索引 / 语言 / 日志）
│   ├── ui/serve_fallback.py   ★ 本工单改造：Web 界面（端口 8125）
│   ├── prompts/               提示词模板
│   └── storage/               SQLite 会话存储
├── 评估/
│   ├── ragas近似.py           ★ 本工单新增：自研 RAGAS 近似器（4 指标 + 内置自检）
│   ├── run_eval_opt.py        ★ 本工单新增：优化前后对比实跑
│   ├── 稳定性压测.py           ★ 本工单新增：连续查询 + 容错场景
│   ├── 多语言实测.py           ★ 本工单新增：中英配对实测
│   ├── 判分口径.py            ← 复用 工单8（导入 工单7 判分.py，保证同口径）
│   ├── 参考答案集.jsonl        ★ 17 题（含 PDF 核对标记）
│   ├── PDF参考答案原文对照.jsonl ★ PDF 抽出 4 条原文
│   ├── 参考答案集核对报告.json  ★ 核对结论
│   └── 参考答案原文.txt        ★ PDF 逐页原文（人工复核用）
├── scripts/
│   ├── 构建参考答案集.py       ★ PDF → jsonl + 核对
│   └── 构建优化图谱.py         ★ 抽取层优化入库
├── data/
│   ├── 图谱优化后/            ★ 6717 节点 / 3807 边 + 图谱优化统计.json
│   ├── 测试问题集.jsonl        ← 复用 工单8
│   └── graph → 工单8 图谱（Junction，只读）
├── requirements.txt
└── README.md（本文件）
```

> ★ = 本工单新增或改造；← = 从 工单7/工单8 复用（**只读逻辑，不修改上游文件**）。

---

## 2. 快速使用

```python
import sys
sys.path.insert(0, r"E:\gao6gongdan\工单9\研发")

from app.graphrag_opt.配置 import get_config
from app.graphrag_opt.引擎 import OptimizedEngine

cfg = get_config()
eng = OptimizedEngine(cfg=cfg, profile="after")      # "before" = 工单8 基线
print(eng.warmup())                                   # 索引 8847 块 / 图谱 6717 节点

ans = eng.answer("平安银行2019年年度报告中，该行2019年实现营业收入和净利润分别是多少？")
print(ans.answer)                 # 答案（含 [块N] 引用与图谱依据）
print(ans.latency_ms)             # {'total': …, 'retrieve': …, 'graph': …, 'generate': …}
print(len(ans.evidence_chunks))   # 进生成阶段的证据块（上下文主口径）
print(ans.diagnostics["field_probe_fields"])   # 命中的字段锚点
```

---

## 3. 依赖

见 `requirements.txt`。**本工单没有新增任何第三方依赖** ——
全部能力由「已装包 + 标准库 + 复用 工单7/8 的代码」实现
（本机没有 ragas / matplotlib / pandas / gradio / streamlit，且纪律禁止裸 pip install）。

---

## 4. 纪律

* **只读上游**：`工单7`、`工单8` 的目录一律只读；本工单只写 `E:\gao6gongdan\工单9\`。
* **资源纪律**（`_共享约定.md` §5.5）：单进程、**无进程池**、Ollama 单模型驻留
  （`keep_alive="0s"`）、流式处理、日志重定向到 `部署/日志/`。
  自检：`Select-String -Path 研发\**\*.py -Pattern 'multiprocessing|ProcessPoolExecutor|Pool\(|joblib|n_jobs'` → **0 命中**。
* **工单编号注释**：所有源码头部都含 `人工智能NLP-RAG-Graph RAG 优化任务`。
* **阈值配置化**：所有阈值在 `app/graphrag_opt/配置.py` 的 `DEFAULTS`
  + `部署/配置/优化配置.json`，可用环境变量 `GRAPHRAGOPT_<段>__<键>` 覆盖。
