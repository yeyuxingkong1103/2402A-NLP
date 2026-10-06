# 工单11 - Embeddings 模型微调

完整演示数据集生成、三元组训练、模型权重保存和微调前后 MRR 评估。默认实现轻量且离线可跑；数据格式可直接迁移到 `sentence-transformers` 的 TripletLoss/MNRL。

```powershell
python .\工单11\main.py --docs "附件\ccf_competition\txt" --epochs 8
```

产出：`training_dataset.json`、`model_weights.json`、`evaluation.json`。

## 指定 BGE 模型的完整微调

```powershell
python -m pip install -r .\工单11\requirements-full.txt
python .\工单11\main.py --docs "附件目录"
python .\工单11\train_bge.py --epochs 1
```

完整模式使用 `BAAI/bge-base-en-v1.5`、Triplet Loss 和微调前后 TripletEvaluator。基础模型缓存统一位于 `D:\工单models\huggingface`，微调模型保存到 `D:\工单models\bge-domain-model`。首次运行会从未完成位置续传；CPU 可运行但较慢，建议使用 CUDA GPU。
