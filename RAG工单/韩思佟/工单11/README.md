# 工单11 · Embedding 模型微调

工单编号：人工智能NLP-RAG-Embedding 模型微调

## 一、工单目标

数据集生成 → 问答对生成 → 模型加载 → 损失函数 → 训练参数 → 评估器 → 微调前后评估；微调后检索效果优于微调前。

## 二、怎么跑

```bash
python scripts/w11_finetune_embedding.py --n-pairs 300 --epochs 1
```

## 三、产出物
- `outputs/w11/工单11_微调报告.md`
- `outputs/w11/训练配置.json`
- `outputs/w11/微调前评估.json`
- `outputs/w11/微调后评估.json`
- `cache/finetune/bge-small-zh-v1.5-ft`

## 五、需要截图的地方

见同目录 `截图清单.md`。