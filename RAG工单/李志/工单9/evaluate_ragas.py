"""对已有检索上下文执行真实 RAGAS 指标评估，需要配置 OpenAI 兼容模型。"""
import argparse, json, os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="outputs/ragas_dataset.json")
args = parser.parse_args()
if not os.getenv("OPENAI_API_KEY"):
    raise SystemExit("请先设置 OPENAI_API_KEY；RAGAS 的 LLM 裁判不能离线伪造。")
try:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import context_precision, context_recall
except ImportError as exc:
    raise SystemExit("请先执行：python -m pip install -r requirements-ragas.txt") from exc
path = Path(__file__).parent / args.dataset
rows = json.loads(path.read_text(encoding="utf-8"))
score = evaluate(Dataset.from_list(rows), metrics=[context_precision, context_recall])
result = score.to_pandas().to_dict(orient="records")
(Path(__file__).parent / "outputs/ragas_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
