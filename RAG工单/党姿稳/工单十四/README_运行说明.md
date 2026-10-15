# 工单十四 · 修复低质量工业 PDF 的解析与信息丢失 · 运行说明

## 一、实现功能对照（逐条）

| 工单要求 | 实现情况 | 对应文件 |
|---|---|---|
| 任务一：梳理 RAGFlow 的文件解析流程、`do_handle_task` 逻辑、DeepDoc 解析器体系与 PDF 解析技术 | 已完成，输出 597 行技术总结，含 RAGFlow 调用链、Redis Stream 队列触发机制、`paper`/`table`/`one`/`knowledge_graph` 四种分块策略、DeepDoc 内置解析器清单与 PDF 解析技术栈 | `技术总结_RAGFlow_DeepDoc解析方案.md` |
| 任务二：修复低可提取文本 / 信息丢失问题 | 文档类型自动分诊（文字型/扫描型/混合型）；扫描页按 200 DPI 渲染后补 RapidOCR；图纸页识别并裁剪导出 PNG；图纸块把图号、图内标注、说明书对应段落汇聚为可检索文本 | `pdf_parser.py`、`chunker.py`、`figure_describer.py` |
| 任务二：多轮测试结果、原因分析、优化方案及效果 | 已完成，报告含「多轮测试结果 / 失败案例分析 / 优化方案及效果 / 逐题明细」四节；轮次一 83.3% → 轮次二 100.0% | `evaluation/测试报告.md` |
| 任务二：调整后的实现代码 | 已完成，14 个模块均在 300 行以内 | 本目录全部 `.py` |
| 演示：针对 6 个指定问题检索并显示答案与检索精确度 | 已完成，Streamlit 界面支持逐题提问、显示检索片段与相似度分数、显示识别出的图纸页图像，并提供「RAG 回答 / 纯 LLM 回答」对比 | `app.py`、`start_app.py` |

### 验收标准对照

| 验收项 | 要求 | 实测 |
|---|---|---|
| 文档验收 | 三个问题给出清晰准确的技术实现方案总结 | 已完成（`技术总结_RAGFlow_DeepDoc解析方案.md`） |
| 问答准确性 | 准确率 100% | **100.0%（6/6）** |
| 响应时间 | 不超过 3 秒 | 检索平均 **0.1 秒**，端到端稳态 **0.9～1.4 秒**（首题含模型加载约 33 秒） |
| 资源消耗 | 计算资源合理、高并发稳定 | 仅 CPU 推理，无 GPU 依赖；单进程内存峰值 < 4 GB |

## 二、运行方式

在 `C:\Users\dang5\Desktop\工单十四\` 目录下执行（Windows PowerShell，用 `;` 而非 `&&`）：

```powershell
# 1) 完成一次解析与索引（首次约 20 分钟：需要 OCR 11 页 +
#    为 3 张图纸生成描述，结果全部缓存到 data\，之后不再重算）
python indexer.py

# 2) 跑 6 个验收问题，生成评测报告
python evaluate.py

# 3a) 启动演示界面（推荐）
python start_app.py

# 3b) 或直接指定端口启动
streamlit run app.py --server.port 8501
```

## 三、访问地址

- 演示界面：`http://localhost:8501`（端口被占用时 `start_app.py` 会自动向后寻找空闲端口并打印实际地址）
- 评测报告：`evaluation\测试报告.md`
- 评测原始数据：`evaluation\results.json`

## 四、注意事项

1. **首次运行前请先解压附件**：`C:\Users\dang5\Desktop\新工单\14-17附件\` 需已解压，
   确保 `original_problems\original_problems\documents\CN100342976C.pdf` 存在。
2. **首次运行耗时较长**：会加载嵌入模型 `m3e-base` 并对 11 页扫描件做 OCR；
   解析结果写入 `data\parse_*.json`，索引写入 `data\chunks.json` / `embeddings.npy` / `bm25.pkl`，
   再次运行直接命中缓存（秒级）。
3. **需要联网**：生成答案调用 DeepSeek API（`config.py` 中已配置密钥与地址）。
   断网时检索仍可用，但无法生成答案。
4. **模型路径**：嵌入模型指向 `C:\Users\dang5\Desktop\zongwenjianjia\zhuangao4\m3e-base`，
   如换机器需同步修改 `config.py`。
5. **演示视频**：按工单要求需录屏展示功能与操作流程，建议用 `start_app.py` 启动界面后，
   依次提问 6 个验收问题并录屏（本机可用桌面上的 EV 录屏工具）。

## 五、目录说明

```
工单十四/
├── 技术总结_RAGFlow_DeepDoc解析方案.md   # 任务一技术总结（文档验收物）
├── README_运行说明.md                     # 本文件
├── config.py                 # 路径、解析参数、检索参数、6 个测试问题
├── pdf_parser.py             # 文档分诊 + OCR 补文本 + 图纸页识别导出
├── chunker.py                # 正文块 / 著录项块 / 图纸块
├── figure_describer.py       # 图纸描述生成（VLM 优先，缺失时规则化兜底）
├── indexer.py                # 索引构建与缓存
├── bm25.py                   # BM25 全文索引（jieba 分词）
├── retriever.py              # BM25 + 向量 + RRF 混合检索
├── generator.py              # Prompt 组装与 DeepSeek 调用
├── evaluate.py               # 6 题批量评测与报告生成
├── app.py / start_app.py     # Streamlit 演示界面与端口自适应启动
└── data/ evaluation/         # 解析与索引缓存、评测报告
```
