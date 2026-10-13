# 工单编号：人工智能 NLP-RAG-Embedding 模型微调任务
import json
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from config_v11 import (
    DATASET_FILE, MODEL_OUTPUT_DIR, BASE_MODEL,
    EPOCHS, BATCH_SIZE, LEARNING_RATE, DEVICE, MAX_SEQ_LENGTH
)

if __name__ == "__main__":
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        triplets = json.load(f)
    print(f"加载 {len(triplets)} 个三元组")

    model = SentenceTransformer(BASE_MODEL, device=DEVICE)
    model.max_seq_length = MAX_SEQ_LENGTH

    # 清一下显存
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    train_examples = [
        InputExample(texts=[t["anchor"], t["positive"]])
        for t in triplets
    ]
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    print(f"开始微调（device={DEVICE}, batch={BATCH_SIZE}, epoch={EPOCHS}, max_len={MAX_SEQ_LENGTH}）...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=EPOCHS,
        warmup_steps=int(len(train_dataloader) * 0.1),
        optimizer_params={"lr": LEARNING_RATE},
        show_progress_bar=True
    )

    model.save(MODEL_OUTPUT_DIR)
    print(f"微调后模型已保存到 {MODEL_OUTPUT_DIR}")