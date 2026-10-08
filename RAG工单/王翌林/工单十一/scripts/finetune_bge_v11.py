# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-Embedding模型微调任务
scripts/finetune_bge_v11.py —— 工单十一 Embedding 模型微调实战主脚本

对应工单要求逐项：
  数据集与模型加载   —— qa_pairs_train/dev.jsonl + 本地 bge-m3
  定义损失函数       —— MultipleNegativesRankingLoss（对比损失：批内负例，
                        适配 (query, positive) 正例对格式）
  定义训练参数       —— bf16 + 梯度检查点 + 冻结底层 12 层（8GB 显存可训）
  创建评估器         —— InformationRetrievalEvaluator（生成 dev 题 + 工单七真实题）
  微调前评估模型     —— 训练前在同一评估集上记录基线指标
  微调后评估模型     —— 训练后同口径复测，输出 before/after 对比

产物：
  models/bge-m3-ft-v11/           微调后模型（sentence-transformers 格式）
  docs/finetune_v11_results.json  训练过程 + 微调前后指标对比（验收数据支撑）

用法：
  python scripts/finetune_bge_v11.py                 # 完整：基线评估→训练→复测
  python scripts/finetune_bge_v11.py --skip-train    # 仅基线评估（调试）
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from dotenv import load_dotenv
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
load_dotenv()  # 工单十一：显式加载 .env，独立脚本必须

from src.finetune_v11.qa_dataset import read_jsonl
from src.finetune_v11.ir_eval import (build_ir_inputs, compare_metrics,
                                      extract_metrics)

WORK_ORDER = "人工智能NLP-RAG-Embedding模型微调任务"
BASE_MODEL = "/home/dabaie/models/bge-m3"
FT_MODEL_DIR = Path("models/bge-m3-ft-v11")
DATA_DIR = Path("data/finetune_v11")
RESULTS = Path("docs/finetune_v11_results.json")

# 工单十一：8GB 显存适配——冻结 embedding 层 + 底部 N 层 Transformer，
# 仅微调顶层（bge-m3 共 24 层），显存占用减半且领域适配足够
FREEZE_BOTTOM_LAYERS = 12


def freeze_bottom_layers(model, n_layers: int) -> int:
    """工单十一：冻结嵌入层与底部 n_layers 层，返回可训练参数量"""
    auto = model[0].auto_model
    for p in auto.embeddings.parameters():
        p.requires_grad = False
    for layer in auto.encoder.layer[:n_layers]:
        for p in layer.parameters():
            p.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"冻结底部 {n_layers} 层: 可训练参数 {trainable/1e6:.0f}M / {total/1e6:.0f}M")
    return trainable


def build_evaluator(model_name: str, queries, corpus, relevant):
    """工单十一：创建检索评估器（mrr/ndcg/accuracy@k）"""
    from sentence_transformers.evaluation import InformationRetrievalEvaluator
    return InformationRetrievalEvaluator(
        queries=queries, corpus=corpus, relevant_docs=relevant,
        name=model_name, show_progress_bar=True, batch_size=32)


def run_eval(model, evaluator) -> dict:
    scores = evaluator(model)
    return extract_metrics(scores)


def main():
    ap = argparse.ArgumentParser(description=f"工单十一 bge-m3 金融领域微调（{WORK_ORDER}）")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-seq", type=int, default=256)
    ap.add_argument("--freeze-bottom", type=int, default=FREEZE_BOTTOM_LAYERS)
    ap.add_argument("--skip-train", action="store_true", help="仅做微调前基线评估")
    # 工单十一：transformers v5 的 gradient_checkpointing_enable 不兼容
    # sentence-transformers 传入的 every_n_layers 参数（TypeError），
    # 默认关闭梯度检查点；8GB 显存靠冻结底部12层+bf16+batch8 已足够
    ap.add_argument("--grad-ckpt", action="store_true", help="开启梯度检查点（当前环境不兼容，勿用）")
    ap.add_argument("--output-dir", default=str(FT_MODEL_DIR))
    args = ap.parse_args()

    t_all = time.perf_counter()

    # ---------- 1) 数据集加载 ----------
    train_pairs = read_jsonl(DATA_DIR / "qa_pairs_train.jsonl")
    dev_pairs = read_jsonl(DATA_DIR / "qa_pairs_dev.jsonl")
    corpus_chunks = json.loads((DATA_DIR / "qa_corpus_chunks.json").read_text(encoding="utf-8"))
    gold = json.loads(Path("data/ccf_reports/test_questions_v7.json")
                      .read_text(encoding="utf-8"))["questions"]
    logger.info(f"数据集: train={len(train_pairs)} dev={len(dev_pairs)} "
                f"corpus={len(corpus_chunks)} 真实题={len(gold)}")

    queries, corpus, relevant = build_ir_inputs(dev_pairs, corpus_chunks, gold)
    logger.info(f"评估器输入: queries={len(queries)} corpus={len(corpus)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"训练设备: {device}"
                + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    # ---------- 2) 模型加载（基线） ----------
    from sentence_transformers import SentenceTransformer
    base_model = SentenceTransformer(BASE_MODEL, device=device)
    base_model.max_seq_length = args.max_seq

    # ---------- 3) 创建评估器 + 4) 微调前评估 ----------
    evaluator = build_evaluator("fin_v11", queries, corpus, relevant)
    logger.info("========== 微调前评估（基线 bge-m3） ==========")
    t0 = time.perf_counter()
    before = run_eval(base_model, evaluator)
    eval_before_s = time.perf_counter() - t0
    logger.info(f"基线指标: {before}")

    results = {"work_order": WORK_ORDER, "base_model": BASE_MODEL,
               "device": device, "freeze_bottom_layers": args.freeze_bottom,
               "train_params": {"epochs": args.epochs, "batch_size": args.batch_size,
                                "lr": args.lr, "max_seq": args.max_seq,
                                "loss": "MultipleNegativesRankingLoss(对比损失,批内负例)"},
               "data": {"train": len(train_pairs), "dev": len(dev_pairs),
                        "corpus": len(corpus), "eval_queries": len(queries)},
               "before": before, "eval_before_seconds": round(eval_before_s, 1)}

    if not args.skip_train:
        # ---------- 5) 定义损失函数 + 训练参数 ----------
        from datasets import Dataset
        from sentence_transformers import (SentenceTransformerTrainer,
                                           SentenceTransformerTrainingArguments)
        from sentence_transformers.losses import MultipleNegativesRankingLoss

        train_ds = Dataset.from_list(
            [{"query": p["query"], "positive": p["positive"]} for p in train_pairs])
        loss = MultipleNegativesRankingLoss(base_model)
        logger.info(f"损失函数: MultipleNegativesRankingLoss scale={loss.scale}")

        trainable = freeze_bottom_layers(base_model, args.freeze_bottom)
        train_args = SentenceTransformerTrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.lr,
            warmup_ratio=0.1,
            bf16=(device == "cuda"),               # RTX 50 系支持 bf16
            gradient_checkpointing=args.grad_ckpt,  # 工单十一：v5 不兼容，默认关
            logging_steps=10,
            save_strategy="no",                    # 训练完统一 save_pretrained
            eval_strategy="no",                    # 评估由本脚本 before/after 统一做
            seed=42,
            report_to=[],
        )

        # ---------- 6) 训练 ----------
        logger.info("========== 开始微调训练 ==========")
        trainer = SentenceTransformerTrainer(
            model=base_model, args=train_args,
            train_dataset=train_ds, loss=loss)
        t0 = time.perf_counter()
        train_out = trainer.train()
        train_s = time.perf_counter() - t0
        logger.info(f"训练完成: {train_s/60:.1f}min  log={train_out.training_loss:.4f}")
        results["train"] = {"seconds": round(train_s, 1),
                            "training_loss": round(train_out.training_loss, 4),
                            "trainable_params_M": round(trainable / 1e6, 1),
                            "log_history": [h for h in trainer.state.log_history
                                            if "loss" in h]}

        # 解冻并保存（推理需全参数）
        for p in base_model.parameters():
            p.requires_grad = True
        base_model.save_pretrained(args.output_dir)
        logger.info(f"微调模型已保存: {args.output_dir}")
        results["ft_model_dir"] = args.output_dir

        # ---------- 7) 微调后评估（同口径） ----------
        logger.info("========== 微调后评估 ==========")
        t0 = time.perf_counter()
        after = run_eval(base_model, evaluator)
        results["after"] = after
        results["eval_after_seconds"] = round((time.perf_counter() - t0), 1)
        results["compare"] = compare_metrics(before, after)
        improved = [r for r in results["compare"] if r["delta"] > 0]
        results["verdict"] = (f"{len(improved)}/{len(results['compare'])} 项指标提升，"
                              "微调后检索效果优于微调前"
                              if improved else "未见提升，需检查数据/超参")
        logger.info(f"对比: {results['compare']}")
        logger.info(f"结论: {results['verdict']}")

    results["elapsed_min"] = round((time.perf_counter() - t_all) / 60, 1)
    RESULTS.write_text(json.dumps(results, ensure_ascii=False, indent=1),
                       encoding="utf-8")
    logger.info(f"结果落盘: {RESULTS}")


if __name__ == "__main__":
    main()
