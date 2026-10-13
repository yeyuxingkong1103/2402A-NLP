# 工单7 - 功能测试与评估

`evaluate.py` 读取 JSONL 问题集，计算 hit@k、MRR、平均延迟和错误问题列表。`questions.jsonl` 是 10 条可替换的最小评估集。

```powershell
python evaluate.py --corpus sample_corpus.txt --questions questions.jsonl --output evaluation.json
```

