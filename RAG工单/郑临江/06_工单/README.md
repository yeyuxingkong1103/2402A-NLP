# 工单 06：混合检索

**工单编号**：人工智能NLP-RAG-混合检索任务

## 一、项目简介

在招股说明书问答系统中优化检索策略，提供 **向量检索（召回+重排）**、
**全文检索** 以及二者同时执行的 **混合检索**，并支持检索策略的配置。

## 二、功能

### 1. 向量检索（召回 + 重排）
- 向量嵌入（支持 bge / m3e，默认 TF-IDF）与余弦相似度召回；
- 提供 3 种重排算法：
  - `TfidfReranker`：TF-IDF 相似度重排；
  - `LLMReranker`：基于 LLM 的相关性重排；
  - `AdaptiveFeedbackReranker`：基于用户反馈的自适应重排。

### 2. 全文检索
- 倒排索引；
- 布尔查询、短语匹配、模糊匹配；
- 多字段检索（标题、正文、摘要）。

### 3. 混合检索
- 向量/全文权重可配置（`config.py`）；
- 融合算法：加权平均、倒数排名融合（RRF）、投票机制。

## 三、目录结构

```
06_工单/
├── app.py               # 主程序（三种策略演示）
├── config.py            # 配置（权重、嵌入模型）
├── embedder.py          # 嵌入（TF-IDF / bge / m3e）
├── vector_retriever.py  # 向量召回 + 3 种重排
├── fulltext_retriever.py# 倒排索引全文检索
├── hybrid.py            # 融合与混合检索
├── llm.py / pdf_parser.py
├── requirements.txt
└── README.md
```

## 四、运行

```bash
pip install -r requirements.txt
python app.py
```

## 五、验收对照

- 支持多种嵌入模型（bge、m3e）与至少 3 种重排算法；
- 倒排索引全文检索，支持多字段；
- 混合检索支持权重调整与多种融合算法；
- 准确率 ≥ 90%、召回率 ≥ 95%；
- 代码注释含工单编号：人工智能NLP-RAG-混合检索任务。
