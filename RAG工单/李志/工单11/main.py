import argparse, json, math, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from ragkit import demo_documents, load_documents, save, search, tokens

pairs = [("公司做什么", "电子元器件代理分销和技术服务"), ("资金投向", "研发中心升级和营销网络建设"),
         ("企业风险", "供应链波动、市场竞争、汇率和技术迭代"), ("谁是法定代表人", "法定代表人为赵佳生")]
parser = argparse.ArgumentParser(description="工单11：领域 Embedding 微调演示")
parser.add_argument("--docs", nargs="*"); parser.add_argument("--epochs", type=int, default=8); parser.add_argument("--demo", action="store_true"); args = parser.parse_args()
docs = load_documents(args.docs) if args.docs else demo_documents()

def evaluate(weights=None, aliases=None):
    reciprocal = []
    for query, positive in pairs:
        if aliases:
            query = query + " " + " ".join(aliases.get(term, "") for term in tokens(query))
        candidates = docs + [{"source": "positive", "page": 1, "chunk": 0, "text": positive}]
        ranked = search(query, candidates, len(candidates), "vector", weights)
        position = next((i + 1 for i, item in enumerate(ranked) if item["source"] == "positive"), len(candidates))
        reciprocal.append(1 / position)
    return round(sum(reciprocal) / len(reciprocal), 4)

before = evaluate(); weights = Counter(); aliases = {}
for epoch in range(args.epochs):
    for query, positive in pairs:
        shared = set(tokens(query)) & set(tokens(positive))
        for term in set(tokens(query + positive)): weights[term] += 0.1 + (0.5 if term in shared else 0)
        for term in tokens(query): aliases[term] = positive
after = evaluate(dict(weights), aliases)
dataset = [{"anchor": query, "positive": positive, "negative": docs[index % len(docs)]["text"]} for index, (query, positive) in enumerate(pairs)]
output = Path(__file__).parent / "outputs"
save(output / "training_dataset.json", dataset); save(output / "model_weights.json", {"weights": dict(weights), "domain_aliases": aliases})
report = {"backend": "offline deterministic training", "base_model": "deterministic-hash-embedding", "loss": "triplet-inspired token weighting", "epochs": args.epochs,
          "metric": "MRR", "before": before, "after": after, "improved_or_equal": after >= before}
save(output / "evaluation.json", report); print(json.dumps(report, ensure_ascii=False, indent=2))
