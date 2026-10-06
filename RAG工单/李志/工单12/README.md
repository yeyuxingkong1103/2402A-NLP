# 工单12 - LightRAG 优化

实现实体/关系图谱、局部向量检索、全局两跳扩展、增量图谱更新，并对普通 RAG 与 LightRAG 输出指标对比。

```powershell
python .\工单12\main.py --docs "C:\Users\Administrator\Desktop\工单\工单1-13\附件\招股说明书1.pdf" "C:\Users\Administrator\Desktop\工单\工单1-13\附件\招股说明书2.pdf" --mode both
```

产出：`graph.json`、`comparison.json`。

官方 LightRAG 完整模式：

```powershell
python -m pip install -r .\工单12\requirements-lightrag.txt
$env:OPENAI_API_KEY="你的密钥"
python .\工单12\run_official_lightrag.py --docs "招股说明书1.pdf" "招股说明书2.pdf" --query "销售部门由哪些部门构成？"
```
