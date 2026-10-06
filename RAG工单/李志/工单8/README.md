# 工单8 - Graph+RAG 金融问答

从金融文本抽取实体与共现关系，输出可视化所需 `nodes/edges`，并用两跳图扩展增强检索。

```powershell
python .\工单8\main.py --docs "附件\ccf_competition\txt" --query "主营业务与风险因素有什么关系？"
```

产出：`outputs/graph.json`、`outputs/answer.json`、可用浏览器打开的 `outputs/graph.html`。
