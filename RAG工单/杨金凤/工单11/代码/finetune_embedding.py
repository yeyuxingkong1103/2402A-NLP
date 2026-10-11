# ============================================================
# 工单编号：人工智能NLP-RAG-Embedding模型微调任务
# 文件：finetune_embedding.py
# 说明：用 (query, positive) 对微调 bge-small-zh-v1.5
#       损失函数：MultipleNegativesRankingLoss（对比学习）
# ============================================================
import json
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

BASE_MODEL = "/root/autodl-tmp/models/models/AI-ModelScope--bge-small-zh-v1.5/snapshots/master"
OUTPUT_DIR = "/root/autodl-tmp/projects/RAG/finetuned_bge"
TRAIN_FILE = "/root/autodl-tmp/projects/RAG/train.jsonl"

# 1. 加载训练数据
train_examples = []
with open(TRAIN_FILE, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        train_examples.append(
            InputExample(texts=[item["query"], item["positive"]])
        )
print(f"训练样本：{len(train_examples)} 条")

# 2. 加载基础模型
model = SentenceTransformer(BASE_MODEL, device="cuda")

# 3. 数据加载器
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=8)

# 4. 损失函数：多负样本排序损失
train_loss = losses.MultipleNegativesRankingLoss(model)

# 5. 训练
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=5,
    warmup_steps=5,
    output_path=OUTPUT_DIR,
    show_progress_bar=True,
)

print(f"微调完成，模型保存到：{OUTPUT_DIR}")
