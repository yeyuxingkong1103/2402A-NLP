# 工单十五 · 优化技术图纸与文本的跨模态检索流程 —— 运行说明

工单编号：人工智能NLP-RAG-15-优化技术图纸与文本的跨模态检索流程
目标文档：`CN100347506C.pdf`（说明书附图「图3」位于第 11 页）

## 零、实现功能与验收标准对照

### 产出物对照

| 工单产出物 | 实现情况 | 对应文件 |
|---|---|---|
| 一、技术分析文档：原有检索流程、识别出的瓶颈、优化方案设计 | 已完成 241 行技术分析文档 | `技术分析文档_跨模态检索优化.md` |
| 二、优化后的代码：查询理解优化、混合检索策略、Prompt 模板修改 | 已完成，13 个模块均在 300 行以内 | `query_understanding.py`、`cross_modal_retriever.py`、`reranker.py`、`generator.py` 等 |
| 三、测试与优化报告：多轮测试结果（含优化前基准）、逐条失败案例原因分析与解决措施、准确率验证日志 | 已完成，报告含「多轮测试记录 / 总体对比 / 优化前后逐题明细 / 失败案例的原因分析与解决措施」五节 | `evaluation/测试与优化报告.md`、`evaluation/results.json` |
| 四、演示视频：展示优化后的系统界面 | 界面已完成，可逐题提问并展示检索到的关键图文块与答案；录屏步骤见第四节 | `app.py`、`start_app.py` |

### 验收标准对照

| 验收项 | 要求 | 实测 |
|---|---|---|
| 问答准确性（核心） | 6 个测试问题准确率 **必须 100%** | **100.0%（6/6）** |
| 检索相关性 | 后 4 个图文题，Top-1 或 Top-3 必须包含对「第 11 页图3」的有效描述或引用 | 图纸题 **Top-3 命中率 100%**，且 4 题中图3 图纸块均排在第 **1** 名 |
| 响应时间 | 优化后端到端不显著劣化，保持 3～5 秒内 | 检索平均 **2.5 秒**，端到端稳态 **3.7～4.2 秒**（首题含重排模型加载约 5 秒） |
| 系统稳定性 | 不引入新的崩溃或异常，功能回归通过 | 6 题完整跑通，无异常抛出 |

> 说明：优化后的端到端耗时高于优化前（约 4 秒对 1～2 秒），代价来自新增的 bge-reranker-base
> 跨模态重排（约 2.2 秒）。这是换取准确率从 83.3% 提升到 100% 的必要代价，仍落在验收要求的
> 3～5 秒区间内。

## 一、环境

- Python：`C:\Users\dang5\anaconda3\envs\nlp2\python.exe`（Python 3.8.10）
- PowerShell 5 下命令分隔符用 `;`，不能用 `&&`
- 依赖：`sentence_transformers 3.2.1`、`transformers 4.46.3`、`torch 2.4.1+cpu`、`PyMuPDF 1.24.11`、`rapidocr_onnxruntime`、`jieba`、`openai`、`numpy`、`streamlit`
- 本地模型（全部离线，不联网）：
  - 嵌入：`C:\Users\dang5\Desktop\zongwenjianjia\zhuangao4\m3e-base`
  - 重排：`C:\Users\dang5\Desktop\zongwenjianjia\zhuangao4\bge-reranker-base`
  - 视觉语言模型：`C:\Users\dang5\Desktop\zongwenjianjia\zhuangao4\Qwen2-VL-2B-Instruct`（无此模型时自动退化为规则化图纸描述）

## 二、前置条件

1. 附件已解压：`C:\Users\dang5\Desktop\新工单\14-17附件\original_problems\original_problems\`
   其中 `documents\CN100347506C.pdf` 与 `questions.jsonl` 必须存在。
2. 首次运行需要联网调用 DeepSeek 生成答案（`config.py` 中的 `DEEPSEEK_API_KEY`）。

## 三、运行方式

```powershell
cd C:\Users\dang5\Desktop\工单十五

# 1) 构建索引（解析 PDF → OCR → 图纸描述 → 分块 → 向量化 → BM25），结果缓存到 data\
python indexer.py

# 2) 跑优化前后对比评估，报告写入 evaluation\
python evaluate.py
```

可选参数：

| 命令 | 说明 |
|---|---|
| `python indexer.py` | 重建索引；命中 `data\` 缓存时自动跳过，不重复计算 |
| `python evaluate.py` | 优化前 / 优化后各跑 6 题，输出 `evaluation\results.json` 与 `evaluation\测试与优化报告.md` |
| `python evaluate.py --refresh` | 忽略缓存，强制重新解析与向量化 |
| `python evaluate.py --no-vlm` | 不使用视觉语言模型，图纸描述退化为规则化方案 |

## 四、演示界面（对应工单产出物「四、演示视频」的界面部分）

```powershell
python start_app.py
```

启动脚本会用 socket 检测空闲端口（默认从 8501 起探测），端口被占用时自动切换，并打印实际访问地址。

- 访问地址：`http://localhost:<端口>`（启动日志中会打印，默认 `http://localhost:8501`）
- 界面能力：
  1. 左侧「检索流程」可切换 **优化前（文本向量+BM25）** / **优化后（跨模态多路检索）**，直接对比同一问题的检索结果差异；
  2. 展示查询理解结果（是否识别到"图3""第11页"等视觉引用、识别出的图号、图像增强查询是否启用、各路召回规模）；
  3. 高亮展示检索到的知识片段（图纸块 / 著录项块 / 正文块，含图号、页码、综合分）；
  4. 展示 RAG 回答，并可勾选同时展示"纯 LLM 回答（不检索）"作对照；
  5. 页面底部展示优化前后的评估结果表。

按「1）… 6）」预置问题依次提问，即可复现工单要求的"系统能够准确检索到包含图3 的关键图文块并给出正确答案"的完整过程。

## 五、目录说明

```
工单十五\
├─ config.py                  全局配置（路径、模型、解析参数、跨模态检索超参、6 个验收问题）
├─ pdf_parser.py              修复型 PDF 解析（扫描页 OCR、图纸页识别与导出）
├─ chunker.py                 分块：正文块 / 著录项块 / 图形块
├─ figure_describer.py        图纸描述生成（Qwen2-VL 优先，规则化兜底）
├─ bm25.py                    中文 BM25 全文检索
├─ indexer.py                 编排解析→分块→向量化→建索引，结果缓存 data\
├─ query_understanding.py     查询理解：视觉引用识别、图纸定位、图像增强查询构造
├─ reranker.py                跨模态重排（bge-reranker-base Cross-Encoder）
├─ cross_modal_retriever.py   四路召回 + 加权融合 + 重排 + 图号加成
├─ generator.py               Prompt 模板与 DeepSeek 调用
├─ evaluate.py                优化前后对比评估与报告生成
├─ app.py / start_app.py      Streamlit 演示界面与自动端口启动脚本
├─ logger.py                  统一日志
├─ data\                      解析与向量化缓存（chunks.json / embeddings.npy / bm25.pkl / figures\）
├─ evaluation\                评估报告（results.json / 测试与优化报告.md）
└─ 技术分析文档_跨模态检索优化.md   RAGFlow 检索架构剖析与优化方案设计
```

## 六、注意事项

1. **首次运行较慢**：需要解析 11 页扫描件并做 OCR（约 2 分钟），加载 m3e-base 嵌入模型（约 10 秒）、bge-reranker-base 重排模型（约 5 秒）；若启用了视觉语言模型，图纸描述推理在 CPU 上每张图约需 1～3 分钟，结果会缓存到 `data\figure_descriptions.json`，再次运行不重复推理。
2. 所有路径均基于 `os.path.dirname(os.path.abspath(__file__))` 或 `config.py` 中的绝对路径，可在任意目录下用绝对路径调用。
3. 关键参数（融合权重 `ROUTE_WEIGHTS`、重排开关 `RERANK_ENABLED`、重排/融合权重 `RERANK_WEIGHT`/`FUSION_WEIGHT`、图号加成 `FIGURE_LABEL_BOOST`、各 Top-K）均在 `config.py` 中集中提供，可直接外部调整而无需改代码。
