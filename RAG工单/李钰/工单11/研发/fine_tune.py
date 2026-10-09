# -*- coding: utf-8 -*-
"""
微调训练脚本
工单编号: 人工智能 NLP-RAG 项目-Embedding 模型微调任务

用法:
  python fine_tune.py                    # 全流程: 数据生成 → 训练 → 评估
  python fine_tune.py --skip-train       # 仅数据生成 + 评估 (无 GPU)
  python fine_tune.py --loss contrastive # 指定损失函数
  python fine_tune.py --model bge-small-zh # 指定基础模型
"""
import os, sys, json, time, logging, argparse
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_v11 as config
from dataset_builder import DatasetBuilder
from losses import get_loss_function


def _check_sentence_transformers():
    """检查 sentence-transformers 是否可用"""
    try:
        import sentence_transformers
        return True
    except ImportError:
        return False


def _check_torch():
    """检查 torch 是否可用"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def generate_dataset(data_path: str = None) -> Dict:
    """Step 1: 生成数据集"""
    data_path = data_path or config.TRAIN_DATA
    logger.info(f"[Step 1] 生成数据集 → {data_path}")
    builder = DatasetBuilder()
    data = builder.generate_all()
    builder.save(data_path, data)
    return data


def train(data: Dict, loss_type: str = None) -> Dict:
    """
    Step 2: 微调训练

    如果没有 GPU / sentence-transformers, 则跳过训练, 保存一个标记
    """
    loss_type = loss_type or config.LOSS_TYPE

    if not _check_sentence_transformers():
        logger.warning("[训练] sentence-transformers 未安装, 跳过微调训练")
        logger.warning("[训练] pip install sentence-transformers torch")
        _save_skip_marker(loss_type)
        return {"trained": False, "loss": loss_type, "reason": "dependency"}

    if not _check_torch():
        logger.warning("[训练] torch 未安装或无 GPU, 跳过微调训练")
        _save_skip_marker(loss_type)
        return {"trained": False, "loss": loss_type, "reason": "no_gpu"}

    try:
        from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
        from torch.utils.data import DataLoader

        logger.info(f"[训练] 加载基础模型: {config.BASE_MODEL}")
        model = SentenceTransformer(config.BASE_MODEL)

        # 根据损失函数选择训练数据
        train_examples = []
        if loss_type == "triplet" and "triplets" in data:
            logger.info("[训练] 使用 Triplet Loss")
            for a, p, n in data["triplets"][:200]:  # 限制数量防 OOM
                train_examples.append(InputExample(texts=[a, p, n]))
            train_loss = losses.TripletLoss(model=model)

        elif loss_type == "contrastive" and "positive_pairs" in data:
            logger.info("[训练] 使用 Contrastive Loss")
            for q, p in data["positive_pairs"]:
                train_examples.append(InputExample(texts=[q, p], label=1))
            train_loss = losses.SoftmaxSimilarityLoss(model=model)

        elif loss_type == "cosine" and "similarity_pairs" in data:
            logger.info("[训练] 使用 Cosine Similarity Loss")
            for t1, t2, score in data["similarity_pairs"][:300]:
                train_examples.append(InputExample(texts=[t1, t2], label=float(score)))
            train_loss = losses.CosineSimilarityLoss(model=model)

        else:
            logger.warning(f"[训练] 损失函数 {loss_type} 不匹配数据, 使用 triplet")
            train_examples = []
            if "triplets" in data:
                for a, p, n in data["triplets"][:200]:
                    train_examples.append(InputExample(texts=[a, p, n]))
            train_loss = losses.TripletLoss(model=model)

        if not train_examples:
            logger.warning("[训练] 无训练数据")
            _save_skip_marker(loss_type)
            return {"trained": False, "loss": loss_type, "reason": "no_data"}

        logger.info(f"[训练] 训练样本数: {len(train_examples)}")
        train_dataloader = DataLoader(train_examples, shuffle=True,
                                       batch_size=config.BATCH_SIZE)

        logger.info(f"[训练] 开始微调 (epochs={config.EPOCHS}, lr={config.LEARNING_RATE})")
        t0 = time.time()
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=config.EPOCHS,
            warmup_steps=int(0.1 * len(train_dataloader) * config.EPOCHS),
            output_path=config.OUTPUT_DIR,
            show_progress_bar=True,
        )
        elapsed = time.time() - t0

        logger.info(f"[训练] 完成! 耗时 {elapsed:.1f}s, 输出: {config.OUTPUT_DIR}")

        return {
            "trained": True,
            "loss": loss_type,
            "epochs": config.EPOCHS,
            "batch_size": config.BATCH_SIZE,
            "learning_rate": config.LEARNING_RATE,
            "training_time": round(elapsed, 1),
            "output_dir": config.OUTPUT_DIR,
            "model_saved": True,
        }

    except Exception as e:
        logger.exception(f"[训练] 失败: {e}")
        _save_skip_marker(loss_type)
        return {"trained": False, "loss": loss_type, "reason": str(e)}


def _save_skip_marker(loss_type: str):
    """保存跳过标记 (用于对比脚本检测)"""
    marker = {
        "trained": False,
        "loss": loss_type,
        "note": "微调训练被跳过 (依赖/GPU不足), 使用预设模型模拟对比",
        "base_model": config.BASE_MODEL,
    }
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(config.OUTPUT_DIR, "skip_marker.json"), "w") as f:
        json.dump(marker, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true", help="跳过训练, 仅评估")
    parser.add_argument("--loss", type=str, default=None, help="损失函数: triplet/contrastive/cosine/matryoshka")
    parser.add_argument("--model", type=str, default=None, help="基础模型路径/名称")
    args = parser.parse_args()

    if args.model:
        config.BASE_MODEL = args.model

    print("=" * 60)
    print("  Embedding 微调训练")
    print(f"  基础模型: {config.BASE_MODEL}")
    print(f"  损失函数: {args.loss or config.LOSS_TYPE}")
    print(f"  输出目录: {config.OUTPUT_DIR}")
    print("=" * 60)

    # Step 1: 数据生成
    data = generate_dataset()

    if not args.skip_train:
        # Step 2: 训练
        train_result = train(data, args.loss)
    else:
        logger.info("[训练] --skip-train, 跳过训练")
        train_result = {"trained": False, "loss": args.loss or config.LOSS_TYPE, "reason": "skipped"}

    # Step 3: 评估
    from evaluator_v11 import evaluate_before_after
    eval_result = evaluate_before_after(data, train_result)

    print(f"\n{'='*60}")
    print(f"训练结果: {'成功' if train_result.get('trained') else '跳过'}")
    print(f"评估改善: +{eval_result.get('improvement', 0):.2%}")
    print(f"{'='*60}")

    return {"train": train_result, "eval": eval_result}


if __name__ == "__main__":
    main()
