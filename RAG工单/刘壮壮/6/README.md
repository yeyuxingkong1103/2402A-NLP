# 工单 06 — 混合检索

> 工单编号：人工智能NLP-RAG-混合检索任务

在招股说明书问答系统中优化检索策略，提供三种可配置的检索方式：向量检索（召回 + 重排）、全文检索、混合检索。

## 三种检索策略

| 策略 | 技术 | 说明 |
| --- | --- | --- |
| 向量检索 | 向量嵌入 + 余弦相似度召回 + 重排 | 语义召回 |
| 全文检索 | BM25 倒排索引，关键词/短语/模糊匹配，多字段 | 关键词精确定位 |
| 混合检索 | 向量 + 全文 RRF 加权融合 | 兼顾语义与关键词 |

## 三种重排算法（`rerankers.py`）

1. **LLM 重排器**（`LLMReranker`）：大模型对候选片段相关性打分
2. **TF-IDF 重排器**（`TfidfReranker`）：词面重合度重排
3. **自适应重排器**（`AdaptiveReranker`）：基于用户反馈累积加权

## 多嵌入模型支持

通过 `EMBEDDING_MODEL` 配置：`BAAI/bge-small-zh-v1.5`、`moka-ai/m3e-base` 等（需 sentence-transformers）；留空则使用离线 TF-IDF。

## 运行

```bash
pip install -r requirements.txt
cd 6
python main.py                                            # 演示三种策略对比
python main.py --query "武汉兴图新科电子股份有限公司注册资本是多少？" --strategy hybrid
python main.py --strategy vector --rerank llm             # 向量检索 + LLM 重排
```

参数：
- `--strategy`：`vector` / `fulltext` / `hybrid`（默认 hybrid）
- `--rerank`：`tfidf` / `llm` / `adaptive`
- `--alpha`：混合检索中向量检索权重（0~1，默认 0.5）

## 验收要点

- 准确率 ≥ 90%，召回率 ≥ 95%
- 响应时间 ≤ 3 秒
- 提供语义检索、全文检索分别使用的检索技术说明
