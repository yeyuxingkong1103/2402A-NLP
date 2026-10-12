# experiments · 实验过程与溯源证据

本目录保存工单 14 优化过程中的原始实验材料，用于支撑 `deliverables/D2-多轮测试结果与原因分析.md`
中「哪项改动改善了哪几题」的可追溯性。

> 本目录是**过程留痕**，不是交付物本体。正式交付物见 `../deliverables/`。
> 从原始工作目录 `.ragflow_tmp/` 精选而来：保留了各轮报告、脚本、提示词版本、补丁与真值，
> 剔除了重复 PDF、临时 API 报文、以及 53–58 KB/轮的原始 chunks 全量转储（其分析结论已并入各轮报告）。

---

## 一、实验设计（单变量控制）

严格执行「每轮只改一个变量」，否则无法归因。

| 轮次 | VLM 模型 | VLM 提示词 | 解析后处理代码 | 变更点 | 报告 |
|------|----------|-----------|---------------|--------|------|
| **A** | 旧模型 | 容器内置旧版 | 无 | *基线* | `reports/AB_REPORT.md` |
| **B** | `qwen3-vl-plus` | 容器内置旧版 | 无 | **只换模型** | `reports/AB_REPORT.md` |
| **C** | `qwen3-vl-plus` | 三元组 v1 | 无 | **只换提示词**（首次强制方位/流向） | `reports/BC_REPORT.md` |
| **D** | `qwen3-vl-plus` | 三元组 v2 | 无 | **只改提示词**（+5 条收紧规则） | `reports/D_REPORT.md` |
| **E** | `qwen3-vl-plus` | 三元组 v3 | 无 | **只改提示词**（+3 条排他规则） | `reports/E_REPORT.md` |
| **F** | `qwen3-vl-plus` | 三元组 v3 | **+ dedup 后处理器** | **只加代码** | `reports/F_REPORT.md` |
| **E2E** | 同 F 配置 | 三元组 v3 | 同 F | 端到端问答判分 | `reports/E2E_REPORT.md` |

六轮其余参数完全一致（已逐项核对）：

```
布局识别器 = DeepDOC      chunk_token_num = 512      delimiter = "\n"
image_context_size = 0    table_context_size = 0     raptor = on   graphrag = on
embedding = BAAI/bge-m3@SILICONFLOW
```

### 关键发现：运行时提示词 ≠ 本地源码树

| 文件 | 本地源码树 | 容器内实际生效 | 一致 |
|------|-----------|--------------|------|
| `rag/prompts/vision_llm_figure_describe_prompt.md` | 2749 B（`ROLE/GOAL/DECISION RULE/MODE 1/MODE 2`） | **1552 B**（旧版「visual data analyst / bar chart」） | ❌ |
| `deepdoc/parser/figure_parser.py` | 255 行 | **178 行** | ❌ |
| `docker-compose.yml` 卷列表 | — | 不含 `deepdoc/`、`rag/` | ❌ |

A、B 两轮用的旧提示词**只问「趋势/模式」，从不询问部件方位或流动顺序** ——
这从根上解释了为何 A、B 两轮 Q3/Q5 信息为空：**不是模型不行，是提示词没问**。

---

## 二、目录说明

### `baseline/` — 改动前的 v0.23.1 原版

从**部署容器内**取出的真实原始文件，是本次改动的对照基线。

| 文件 | 说明 |
|------|------|
| `figure_parser.py` | 镜像内烘焙的原版，**178 行**，无 `dedup_component_triples` |
| `naive.py` | 镜像内原版 `rag/app/naive.py`，用于核对分块器行为 |
| `vision_llm_figure_describe_prompt.md` | 原版提示词，2749 B，「visual data analyst / Trends-Insights」 |

### `diff/` — 交付改动的 unified diff

| 文件 | 说明 |
|------|------|
| `figure_parser.diff` | `baseline/figure_parser.py` → 交付版（272 行 diff） |
| `vision_llm_figure_describe_prompt.diff` | `baseline/` 原版提示词 → 交付版（138 行 diff） |

文件头已改写为 `a/` `b/` 形式，可在 `ragflow-main/` 根目录下用 `git apply` 或 `patch -p1` 应用。

### `reports/` — 各轮分析报告

| 文件 | 内容 |
|------|------|
| `AB_REPORT.md` | A（基线）vs B（换模型）——证明换模型不足以解决问题 |
| `BC_REPORT.md` | B vs C（首次换用三元组提示词）——方位/流向首次入库 |
| `D_REPORT.md` | C vs D（+5 条收紧规则）——抑制模型自述与臆造 |
| `E_REPORT.md` | D vs E（+3 条排他规则）——消除歧义候选 |
| `F_REPORT.md` | E vs F（+dedup 后处理器）——去重后噪声下降 |
| `E2E_REPORT.md` | F 配置的端到端问答结果：两轮 **4/6 与 5/6**，含逐题原始回答 |

> 原始工作目录中另有与 `.md` 内容重复的控制台输出 `.txt` 转储，未纳入本目录。

### `prompts/` — VLM 提示词各版本

| 文件 | 说明 |
|------|------|
| `C_orig_prompt_container.md` | 从容器内提取的**原版**提示词（1552 B，即 A/B 两轮实际生效者） |
| `C_prompt.md` | 三元组 v1 —— 首次强制输出方位 / 流向 |
| `D_prompt.md` | 三元组 v2 —— 追加 5 条收紧规则 |
| `E_prompt.md` | 三元组 v3 —— 追加 3 条排他规则（**最终交付版**，见 `../ragflow-main/rag/prompts/`） |

### `scripts/` — 分析 / 评估脚本

| 文件 | 说明 |
|------|------|
| `analyze.py` | 顶层切片提取与统计 |
| `ab_analyze.py` / `bc_analyze.py` / `d_analyze.py` / `e_analyze.py` / `f_analyze.py` | 各轮切片对比分析 |
| `e2e_eval.py` / `e2e_eval2.py` | 端到端问答评估（建助手 → 建会话 → 逐题问答 → 选项字母判分） |

### `chunks/` — 各轮切片证据

每轮抽取**第 17、18 号切片**（对应第 7 页图 1）的原文，用于横向对比图纸信息的抽取质量：
`A_chunk17/18.txt` … `F_chunk17/18.txt`。

> 各轮的 chunks 全量 JSON 转储（53–58 KB/轮）未纳入，其结论已并入各轮报告。

### `results/` — 端到端原始作答

| 文件 | 说明 |
|------|------|
| `e2e_results.json` | 第 1 轮 6 题完整作答与检索命中 |
| `e2e_results2.json` | 第 2 轮 6 题完整作答与检索命中 |

两轮结果差异（Q2 由 ✘ 转 ✔）即 D2 中「生成波动」结论的原始依据。

### `groundtruth/` — 真值与视图

| 文件 | 说明 |
|------|------|
| `pdf_groundtruth.md` | `CN100342976C.pdf` 8 页的人工转写真值 |
| `chunks_view.md` | 优化前切片可读视图（基线对照） |
| `patent_chunks_view.md` | 优化后切片可读视图 |

### `patch/` — 中间版本产物

| 文件 | 说明 |
|------|------|
| `figure_parser_patched.py` | 282 行的中间补丁版（早于最终交付版 364 行） |
| `test_dedup.py` | `dedup_component_triples` 单元测试 |

---

## 三、最终交付版本

源码树中的最终版本：

| 文件 | 行数 / 大小 |
|------|-----------|
| `../ragflow-main/deepdoc/parser/figure_parser.py` | 364 行 |
| `../ragflow-main/rag/prompts/vision_llm_figure_describe_prompt.md` | 4799 B |
| `../ragflow-main/docker/docker-compose.yml` | 含源码挂载 |

已校验源码树与容器内文件 md5 一致（见 `deliverables/D2-多轮测试结果与原因分析.md` §8.5）。
