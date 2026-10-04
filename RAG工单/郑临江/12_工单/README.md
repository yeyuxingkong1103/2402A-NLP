# 工单 12：LightRAG 优化

**工单编号**：人工智能NLP-RAG项目-LightRAG优化

参考项目：https://github.com/HKUDS/LightRAG

## 一、项目简介

在传统 RAG（基于向量扁平化表示）基础上，使用 LightRAG 对《招股说明书1.pdf》、
《招股说明书2.pdf》构建知识图谱与检索流程。LightRAG 通过图结构融入文本索引，
采用**双层检索机制**，增强复杂实体依赖关系理解能力，并支持增量更新。

## 二、实现步骤

1. **PDF 解析与切块**（`pdf_parser.py`）：PyMuPDF 提取中文文本并切块；
2. **实体/关系抽取**（`extractor.py`）：根据招股书内容优化实体类型、关系类型；
3. **知识图谱构建**（`graph_builder.py`）：抽取、去重、构建图谱，支持增量更新；
4. **传统 RAG 基线**（`rag.py`）：TF-IDF 向量检索 + LLM 生成；
5. **LightRAG**（`lightrag.py`）：局部（实体/属性）+ 全局（概念/主题）双层检索；
6. **检索结果对比 + RAGAS 指标对比**（`main.py` / `evaluation.py`）。

## 三、优化后的实体类型与关系类型

| 实体类型 | 说明 | 关系类型 | 说明 |
|----------|------|----------|------|
| 公司 | 股份有限公司/集团等 | 持股 | 持有股权 |
| 人物 | 法定代表人/董事等 | 控股 | 实际控制 |
| 金额 | 万元/亿元/元 | 发行 | 公开发行股数 |
| 日期 | 报告期/年份 | 投资 | 募集资金投向 |
| 比例 | 百分比 | 参与制定 | 技术标准 |
| 产品项目 | 项目/工程/系统 | 属于 | 上下游行业 |
| 技术标准 | 标准/规范 | 主营 | 主营业务 |
| 行业 | 行业/产业/领域 | 关联方 | 关联关系 |

## 四、双层检索机制

- **低层次（局部）检索**：提取具体实体（公司名、人名、金额等）→ 向量库相似匹配；
- **高层次（全局）检索**：提取概念关键词（行业/领域/标准等）→ 知识图谱沿关系扩展。

## 五、目录结构

```
12_工单/
├── config.py        # 配置（路径、实体/关系类型、16 个测试问题）
├── pdf_parser.py    # PDF 解析与切块
├── extractor.py     # 实体/关系抽取
├── graph_builder.py # 知识图谱构建（含增量更新）
├── llm.py           # LLM 调用封装
├── rag.py           # 传统 RAG 基线
├── lightrag.py      # LightRAG（双层检索）
├── evaluation.py    # RAGAS 指标
├── main.py          # 对比主程序
├── requirements.txt
└── README.md
```

## 六、运行

```bash
pip install -r requirements.txt
python main.py
```

输出：
- `knowledge_graph.json` — 知识图谱（实体 + 关系）
- `results.json` — 16 个问题的 RAG / LightRAG 检索结果与 RAGAS 指标
- 控制台打印 RAGAS 平均指标对比

## 七、过程问题记录

| 问题 | 原因 | 处理 |
|------|------|------|
| PDF 中文乱码 | 嵌入字体缺少 ToUnicode 映射 | 改用 PyMuPDF 提取 |
| 实体类型混杂 | 未区分公司/金额/日期 | 定义 8 类实体正则规则 |
| 关系抽取为空 | 仅靠关键词无头尾实体 | 头实体取句内公司，尾实体取其他公司 |
| 全局检索缺失 | 传统 RAG 仅向量匹配 | 概念词 + 图谱邻居扩展 |

## 八、验收对照

- 根据 PDF 内容优化实体/关系类型：8 类实体、8 类关系（见第三节）；
- RAG / LightRAG 检索结果对比：`results.json`；
- RAGAS 指标对比：faithfulness、answer_relevance、context_precision、context_recall；
- 代码注释含工单编号：人工智能NLP-RAG项目-LightRAG优化。
