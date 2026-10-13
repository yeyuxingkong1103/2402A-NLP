# 工单5 - 查询理解优化

`query_understanding.py` 提供规范化、同义词扩展、问题拆分、多轮指代拼接和检索路由。输出结构可直接作为 RAG 编排器的中间协议。

```powershell
python query_understanding.py --query "它的维护周期是多少？" --history "泵的型号是 P-101"
```

