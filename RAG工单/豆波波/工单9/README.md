# 工单9 - GraphRAG 优化

`optimize_graphrag.py` 对 lexical、hybrid、graph 三种上下文进行 precision/recall/F1 对比，并列出每条问题的证据覆盖情况。

```powershell
python optimize_graphrag.py --corpus sample_corpus.txt --questions questions.jsonl --output graphrag_metrics.json
```

