# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG项目-Embedding模型微调任务
"""
Embedding 模型微调主脚本（金融年报领域）

完整流程（按工单要求）：
    1. 加载数据集（train 三元组 / eval / corpus）
    2. 加载基线模型 BAAI/bge-base-en-v1.5（本地权重）
    3. 定义损失函数：套娃损失 MatryoshkaLoss(多重负例排序损失 MNRL)
    4. 定义训练参数（epochs / batch_size / lr / warmup）
    5. 创建评估器 IREvaluator
    6. 微调前评估
    7. 执行微调（每个 epoch 回调评估）
    8. 微调后评估，输出对比指标
"""

import json
import os
import time

import torch
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.losses import MultipleNegativesRankingLoss, MatryoshkaLoss
from sentence_transformers.evaluation import SentenceEvaluator

from evaluator import IREvaluator

BASE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(BASE, "dataset")
OUT_MODEL = os.path.join(BASE, "models", "bge-base-finance-ft")

# ==================== 训练配置 ====================
class Config:
    # 基线模型（本地 snapshot）
    base_model = (
        r"D:\Projects\Models\BAAI--bge-base-en-v1.5"
        r"\models--BAAI--bge-base-en-v1.5"
        r"\snapshots\a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"
    )
    max_seq_length = 256       # 缩短序列，CPU 训练更友好
    epochs = 2
    batch_size = 16
    lr = 2e-5
    warmup_ratio = 0.1
    # 套娃损失的截断维度（768 为 bge-base 原始维度）
    matryoshka_dims = [768, 512, 256, 128, 64]


def load_jsonl(name):
    path = os.path.join(DATASET, name)
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


class IREvalCallback(SentenceEvaluator):
    """将 IREvaluator 包装为训练回调，每个 epoch 结束后输出检索指标。"""

    def __init__(self, ir_evaluator: IREvaluator, history: list):
        self.ir = ir_evaluator
        self.history = history

    def __call__(self, model, output_path=None, epoch=-1, steps=-1, *args, **kwargs):
        metrics = self.ir.evaluate(model, verbose=False)
        self.history.append({"epoch": epoch + 1, "steps": steps, **metrics})
        print(f"    [epoch {epoch + 1} 结束] Recall@3={metrics['Recall@3']:.4f} "
              f"| Recall@5={metrics['Recall@5']:.4f} | MRR@10={metrics['MRR@10']:.4f}")
        return metrics["Recall@5"]


def main():
    print("=" * 60)
    print("Embedding 模型微调（金融年报领域 / bge-base-en-v1.5）")
    print("=" * 60)

    # ---------- 1. 加载数据集 ----------
    train_raw = load_jsonl("train.jsonl")
    eval_raw = load_jsonl("eval.jsonl")
    corpus_raw = load_jsonl("corpus.jsonl")
    corpus_texts = [c["text"] for c in corpus_raw]
    print(f"数据集：train={len(train_raw)} eval={len(eval_raw)} corpus={len(corpus_texts)}")

    train_examples = [
        InputExample(texts=[r["query"], r["positive"], r["hard_negative"]])
        for r in train_raw
    ]

    # ---------- 2. 加载基线模型 ----------
    print(f"\n加载基线模型：{Config.base_model}")
    model = SentenceTransformer(Config.base_model, device="cpu")
    model.max_seq_length = Config.max_seq_length
    print(f"  模型维度：{model.get_sentence_embedding_dimension()}")

    # ---------- 3. 定义损失函数 ----------
    # 内层：多重负例排序损失（正例对 + in-batch negatives + 困难负例）
    inner_loss = MultipleNegativesRankingLoss(model)
    # 外层：套娃损失（让 768/512/256/128/64 各维度都可用）
    train_loss = MatryoshkaLoss(
        model, inner_loss, matryoshka_dims=Config.matryoshka_dims
    )
    print(f"损失函数：MatryoshkaLoss(MultipleNegativesRankingLoss) dims={Config.matryoshka_dims}")

    # ---------- 4/5. 评估器 ----------
    ir_evaluator = IREvaluator(eval_raw, corpus_texts, name="金融年报评估集")
    history = []
    eval_callback = IREvalCallback(ir_evaluator, history)

    # ---------- 6. 微调前评估 ----------
    print("\n---------- 微调前评估（基线） ----------")
    t0 = time.time()
    before_metrics = ir_evaluator.evaluate(model, verbose=True)
    print(f"耗时 {time.time()-t0:.0f}s")

    # ---------- 7. 执行微调 ----------
    print("\n---------- 开始微调 ----------")
    import math
    steps_per_epoch = math.ceil(len(train_examples) / Config.batch_size)
    total_steps = steps_per_epoch * Config.epochs
    warmup_steps = int(total_steps * Config.warmup_ratio)
    print(f"总步数 {total_steps}（{steps_per_epoch}/epoch × {Config.epochs}），warmup {warmup_steps} 步")

    t0 = time.time()
    model.fit(
        train_objectives=[(
            torch.utils.data.DataLoader(
                train_examples, shuffle=True, batch_size=Config.batch_size
            ),
            train_loss,
        )],
        evaluator=eval_callback,
        epochs=Config.epochs,
        optimizer_params={"lr": Config.lr},
        warmup_steps=warmup_steps,
        show_progress_bar=True,
        output_path=None,
    )
    print(f"微调耗时 {time.time()-t0:.0f}s")

    # ---------- 保存模型 ----------
    os.makedirs(OUT_MODEL, exist_ok=True)
    model.save(OUT_MODEL)
    print(f"微调模型已保存：{OUT_MODEL}")

    # ---------- 8. 微调后评估 ----------
    print("\n---------- 微调后评估 ----------")
    after_metrics = ir_evaluator.evaluate(model, verbose=True)

    # ---------- 对比汇总 ----------
    print("\n" + "=" * 60)
    print("微调前后对比")
    print("=" * 60)
    print(f"{'指标':12s} {'微调前':>8s} {'微调后':>8s} {'提升':>8s}")
    for k in before_metrics:
        delta = after_metrics[k] - before_metrics[k]
        arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
        print(f"{k:12s} {before_metrics[k]:8.4f} {after_metrics[k]:8.4f} {arrow}{abs(delta):7.4f}")

    # 保存完整结果
    report = {
        "base_model": Config.base_model,
        "config": {
            "max_seq_length": Config.max_seq_length,
            "epochs": Config.epochs,
            "batch_size": Config.batch_size,
            "lr": Config.lr,
            "matryoshka_dims": Config.matryoshka_dims,
        },
        "before": before_metrics,
        "after": after_metrics,
        "epoch_history": history,
        "data_stats": {"train": len(train_raw), "eval": len(eval_raw), "corpus": len(corpus_texts)},
    }
    with open(os.path.join(BASE, "finetune_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n报告已保存：{os.path.join(BASE, 'finetune_report.json')}")


if __name__ == "__main__":
    main()
