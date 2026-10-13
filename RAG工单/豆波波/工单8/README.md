# 工单8 - 金融 GraphRAG 问答

`graph_qa.py` 从文本构建实体共现图，使用图邻接扩展查询并返回证据。它是 GraphRAG 的透明离线基线，生产环境可替换为 Neo4j/图数据库。

```powershell
python graph_qa.py --input sample_corpus.txt --query "供应链风险" --output graph_answer.json
```

