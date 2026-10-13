# 工单2 - PDF Q&A 优化

`optimize.py` 对 chunk size、overlap、混合权重做小型网格搜索，以标注问题的命中率和 MRR 选择配置，避免凭经验固定参数。

```powershell
python optimize.py --corpus sample_corpus.txt --questions questions.jsonl --output optimization.json
```

生产环境还应按文档类型分别调参，并用线上反馈持续更新问题集。


