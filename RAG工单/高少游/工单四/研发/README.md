# 招股说明书图像内容解析问答系统

> 工单编号：**人工智能 NLP-RAG-图像内容解析及检索优化**
> 项目：RAG 项目 · 图像内容解析及检索优化（八维文化与产业研究院）

基于 01/02/03 工单构建的 PDF 问答系统，新增对《招股说明书2.pdf》
（武汉力源信息技术股份有限公司）的**图像内容解析**与**检索优化**。

---

## 一、能力概览

| 能力 | 说明 |
|---|---|
| 图像内容抽取 | 嵌入位图 + 矢量绘图簇 → 图形区域（含水印/表格/版式线剔除） |
| 图像语义解析 | 组织结构图层级还原、统计图数值配对与结论派生 |
| 多模态模型 | Chinese-CLIP 图文跨模态编码 + 零样本图形类型分类 |
| 以文搜图 | 查询文本编码后与图像向量余弦匹配，定位答案所在图形 |
| 混合检索 | 向量 + BM25 + RRF + 多信号重排（含图形块/CLIP 加成） |
| 答案合成 | 表格/图形/正文三类抽取式合成，零幻觉、可溯源 |
| 多文档消歧 | 公司别名 → 源文件名路由，避免两份招股说明书串味 |
| 中英双语界面 | Streamlit 演示界面 |

---

## 二、快速开始

```powershell
conda activate langchain2
cd C:\Users\30274\Desktop\工单代码\工单四\研发

python build_kb.py          # 1. 构建知识库（文本+表格+图像语义块 + CLIP 图像向量）
python evaluate.py          # 2. 评测 16 个验收问题
streamlit run app.py        # 3. 启动演示界面 http://localhost:8501
```

---

## 三、目录说明

```
研发/
├── src/                    # 核心模块（见下表）
├── data/                   # 两份招股说明书 + questions.json
├── figures/                # 抽取出的图形素材（PNG）
├── vector_db/
│   ├── optimized/          # 优化后 FAISS 文本向量库
│   ├── baseline/           # 优化前基线向量库
│   └── image/              # CLIP 图像向量库（figures.npy + figures.jsonl）
├── output/                 # 评估结果 / 报告 / 图表 / 截图
├── build_kb.py             # 知识库构建入口
├── evaluate.py             # 评测脚本
├── app.py                  # Streamlit 演示界面
└── .env                    # 本地路径与模型配置
```

### 核心模块

| 模块 | 职责 |
|---|---|
| `config.py` | 全局参数、路径、权重、阈值 |
| `pdf_parser.py` | PDF 文本/图形/表格对象提取 |
| `table_parser.py` | 表格结构化 + 键值对行 |
| `chunking.py` | 结构感知切片 + 父子块 |
| `figure_extractor.py` | 图形区域定位与渲染 |
| `figure_semantics.py` | 组织结构还原 + 图表数值配对 |
| `image_encoder.py` | Chinese-CLIP 图文编码 |
| `image_index.py` | 图像语义块 + CLIP 向量索引 |
| `knowledge_base.py` | FAISS 向量化与持久化 |
| `query_understanding.py` | 查询理解、文档路由、题型判定 |
| `retriever.py` | 向量 + BM25 + RRF + 以文搜图 |
| `reranker.py` | 加权线性多信号重排 |
| `answer_builder.py` | 三类抽取式答案合成 |
| `qa_engine.py` | 优化/基线/消融四链路 |

---

## 四、验收对照

| 验收项 | 标准 | 实现 |
|---|---|---|
| 问答准确性 | ≥ 90% | 抽取式合成 + 图形/表格感知检索（见 `测试/测试报告.md`） |
| 响应时间 | ≤ 3s | 离线预计算 + 确定性 Query 理解（见 `测试/性能测试报告.md`） |
| 交互友好性 | 清晰简洁 | Streamlit 界面（`app.py`） |
| 多语言支持 | 中英 | 界面中英文切换 |
| 图像语义解析 | 多模态模型 | Chinese-CLIP（`image_encoder.py`） |
| 代码注释 | 含工单编号 | 各模块文件头均标注 |

---

## 五、设计文档索引

- 系统设计：`../设计/01_系统设计文档.md`
- 部署文档：`../部署/部署文档.md`
- 测试报告：`../测试/测试报告.md`
- 优化报告：`../优化/优化报告.md`
- 研发文档：`../研发/研发文档.md`