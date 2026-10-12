# 工单 12 — LightRAG 优化

> 工单编号：人工智能NLP-RAG 项目-LightRAG 优化

在 RAG 系统基础上，使用 LightRAG（轻量级图增强检索）对《招股说明书1.pdf》《招股说明书2.pdf》构建知识图谱并检索，对比 RAG 与 LightRAG 的检索结果及 RAGAS 指标。

## LightRAG 核心

- **图结构索引**：文本切块 → 抽取实体/关系 → 存知识图谱 + 向量
- **双层检索**：低层次（具体实体/属性，沿图关系扩展）+ 高层次（主题/概念，向量相似度）
- **增量更新**：新数据只更新相关部分，无需重建全库（`lightrag.insert()`）

## 文件

```
12/
├── main.py      # RAG vs LightRAG 对比 + RAGAS 指标
├── lightrag.py  # LightRAG 实现（图谱 + 双层检索 + 增量更新）
├── questions.py # 16 个测试问题
├── rag_core.py
└── README.md
```

## 运行

```bash
pip install -r requirements.txt
cd 12
python main.py                          # 对比 RAG / LightRAG + RAGAS 指标
python main.py --query "武汉力源信息技术股份有限公司本次发行股数是多少？"
```

## 说明

- 默认用规则抽取实体类型（公司/人物）与关系类型（控股/参股/担任等），离线可运行。
- 配置 LLM 接口后可扩展为基于大模型的实体类型/关系类型抽取（见 `lightrag.py` 的 `_extract`）。
- 若需使用官方 `HKUDS/LightRAG` 包，可 `pip install lightrag-hku` 后在其 `LightRAG` 类上替换本实现的抽取/检索。

## 验收要点

- 根据 PDF 内容优化实体、关系，抽取准确的实体类型和关系类型
- 给出 RAG 与 LightRAG 检索结果对比及 RAGAS 指标对比
