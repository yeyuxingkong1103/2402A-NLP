# 工单2 - PDF 问答优化

对比纯向量基线与 BM25+向量混合检索，输出准确度代理指标及耗时。

```powershell
python .\工单2\main.py --docs "附件PDF路径" --query "募集资金用于哪些项目？"
```

产出：`outputs/comparison.json`。
