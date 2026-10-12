# 工单 11 — Embedding 模型微调

> 工单编号：人工智能NLP-RAG 项目-Embedding 模型微调任务

在金融招股说明书专业数据上微调嵌入模型，弥补通用模型与专业领域之间的「语义鸿沟」，提升 RAG 检索准确率。

## 原理

通用嵌入模型未充分掌握专业术语、独特实体与特定语义关联。微调通过调整嵌入空间，使专业相关的查询与文档被映射到更接近的向量，从而提升检索效果。

## 完整流程（对应工单要求）

| 步骤 | 实现 |
| --- | --- |
| 数据集生成 | `dataset.py`：从 PDF 抽取文本块，生成 (anchor, positive, negative) |
| 问答对生成 | 配置 LLM 后 `--use-llm` 生成自然问题，否则用关键句启发式 |
| 数据集与模型加载 | 加载 `BAAI/bge-small-zh-v1.5` |
| 定义损失函数 | `TripletLoss`（可替换 `MultipleNegativesRankingLoss` / `MatryoshkaLoss`） |
| 定义训练参数 | epochs / batch_size / warmup |
| 创建评估器 | `InformationRetrievalEvaluator`（NDCG/MAP/MRR/Recall） |
| 微调前后评估 | 训练前后各跑一次评估，输出指标对比 |

## 运行

```bash
pip install -r requirements.txt
cd 11
python finetune.py --epochs 3 --batch-size 8
python finetune.py --use-llm          # 用 LLM 生成更自然的问答对（需配置 .env）
```

## 损失函数说明（`finetune.py` 可切换）

- **TripletLoss**：适用于 (anchor, positive, negative) 三元组
- **ContrastiveLoss / MultipleNegativesRankingLoss**：适用于正负例对
- **CosineSimilarityLoss**：适用于带相似度分数的句子对
- **MatryoshkaLoss**：生成可截断的分层嵌入

## 验收要点

- 生成数据集、模型训练过程、微调前后评估结果
- 微调后检索效果优于微调前，有数据指标（NDCG/Recall）支撑
