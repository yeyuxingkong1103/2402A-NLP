# 工单11 - Embedding 模型微调

`dataset.jsonl` 是正负样本协议，`finetune_config.yaml` 是 Sentence-Transformers/对比学习配置，`evaluate_embeddings.py` 可在没有模型包时用词法代理跑通数据校验和基线。

```powershell
python evaluate_embeddings.py --data dataset.jsonl --output embedding_eval.json
```

GPU 微调命令和显存参数需按实际模型与算力调整，不能在本地离线环境代替训练。

