# 工单13 - RAG 性能瓶颈识别与优化

对检索、答案组织和格式化进行结构化计时，执行多轮基准测试，输出均值/P50/P95、瓶颈阶段和 3 秒验收结论，同时保存 cProfile 报告。

```powershell
python .\工单13\main.py --docs "附件\ccf_competition\txt" --runs 30
```

产出：`profile.txt`、`performance_report.json`。
