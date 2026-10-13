# 工单13 - RAG 性能瓶颈定位与优化

`profile.py` 对读取、分块、检索阶段计时，计算 p50/p95，并给出最大阶段作为优化建议。可把同一命令放入 CI 或压测脚本做回归。

```powershell
python profile.py --input sample_corpus.txt --query "设备" --iterations 20 --output profile.json
```

