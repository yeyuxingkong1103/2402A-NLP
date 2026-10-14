from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from dataset import load_jsonl
from embedding_model import DomainEmbeddingModel
from evaluator import compare_models


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class TrainingConfig:
    dimension: int = 96
    epochs: int = 60
    learning_rate: float = 0.03
    margin: float = 0.5
    top_k: int = 3
    seed: int = 42


def run_training(
    train_records: list[dict],
    evaluation_records: list[dict],
    documents: list[dict],
    output_dir: Path,
    config: TrainingConfig = TrainingConfig(),
) -> dict:
    if not train_records or not evaluation_records:
        raise ValueError("训练集和评估集都不能为空")
    texts = [item["text"] for item in documents]
    texts += [row["query"] for row in train_records + evaluation_records]
    model = DomainEmbeddingModel(dimension=config.dimension, seed=config.seed)
    model.fit_base(texts)
    history = model.fit_adapter(
        train_records,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        margin=config.margin,
    )
    comparison = compare_models(model, evaluation_records, documents, top_k=config.top_k)
    report = {
        **comparison,
        "training": {
            "base_model": "TF-IDF字符嵌入 + TruncatedSVD",
            "adapter": "可训练线性投影层",
            "loss_name": "Triplet Loss",
            "train_samples": len(train_records),
            "evaluation_samples": len(evaluation_records),
            "document_count": len(documents),
            "actual_dimension": model.dimension,
            "config": asdict(config),
            "loss_history": history,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir / "domain_embedding.pkl")
    (output_dir / "evaluation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="训练金融领域Embedding适配器")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--dimension", type=int, default=96)
    args = parser.parse_args()
    train_records = load_jsonl(args.data_dir / "train.jsonl")
    evaluation_records = load_jsonl(args.data_dir / "evaluation.jsonl")
    chunks = json.loads((args.data_dir / "chunks.json").read_text(encoding="utf-8"))
    documents = [{"chunk_id": row["chunk_id"], "text": row["text"]} for row in chunks]
    top_k = min(3, len(documents))
    report = run_training(
        train_records,
        evaluation_records,
        documents,
        args.data_dir / "model",
        TrainingConfig(dimension=args.dimension, epochs=args.epochs, top_k=top_k),
    )
    print(json.dumps({"status": "trained", "before": report["before"], "after": report["after"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

