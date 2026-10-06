"""使用 BAAI/bge-base-en-v1.5 进行真实领域微调。首次运行会下载模型。"""
import argparse, json, os
from pathlib import Path

MODEL_ROOT = Path(r"D:\工单models")
os.environ.setdefault("HF_HOME", str(MODEL_ROOT / "huggingface"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(MODEL_ROOT / "huggingface" / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(MODEL_ROOT / "huggingface" / "transformers"))

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="outputs/training_dataset.json")
parser.add_argument("--model", default="BAAI/bge-base-en-v1.5")
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--batch-size", type=int, default=8)
args = parser.parse_args()

try:
    from sentence_transformers import InputExample, SentenceTransformer, losses
    from sentence_transformers.evaluation import TripletEvaluator
    from torch.utils.data import DataLoader
except ImportError as exc:
    raise SystemExit("请先执行：python -m pip install -r requirements-full.txt") from exc

root = Path(__file__).parent
dataset_path = root / args.dataset
rows = json.loads(dataset_path.read_text(encoding="utf-8"))
if len(rows) < 2:
    raise SystemExit("训练数据至少需要 2 条三元组。")
model = SentenceTransformer(args.model)
examples = [InputExample(texts=[row["anchor"], row["positive"], row["negative"]]) for row in rows]
loader = DataLoader(examples, shuffle=True, batch_size=args.batch_size)
loss = losses.TripletLoss(model)
evaluator = TripletEvaluator(
    anchors=[row["anchor"] for row in rows], positives=[row["positive"] for row in rows],
    negatives=[row["negative"] for row in rows], name="domain-triplets")
before = evaluator(model)
target = MODEL_ROOT / "bge-domain-model"
model.fit(train_objectives=[(loader, loss)], epochs=args.epochs, evaluator=evaluator,
          output_path=str(target), show_progress_bar=True)
after = evaluator(model)
(root / "outputs" / "bge_evaluation.json").write_text(
    json.dumps({"model": args.model, "before": before, "after": after, "output": str(target)},
               ensure_ascii=False, indent=2, default=str), encoding="utf-8")
print(f"模型已保存到 {target}")
