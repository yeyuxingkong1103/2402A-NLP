# 工单9 - Graph+RAG 优化

对比普通 RAG 和两跳 GraphRAG，报告上下文召回与精度代理指标；验收目标写入报告。

```powershell
python .\工单9\main.py --docs "附件\ccf_competition\txt"
```

产出：`outputs/graphrag_evaluation.json`。

真实 RAGAS 复核（需要 LLM Key）：

```powershell
python -m pip install -r .\工单9\requirements-ragas.txt
$env:OPENAI_API_KEY="你的密钥"
python .\工单9\evaluate_ragas.py
```
