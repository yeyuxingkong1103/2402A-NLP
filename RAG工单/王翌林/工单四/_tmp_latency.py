# 临时：逐题延迟（人工智能NLP-RAG-图像内容解析及检索优化）
import json

d = json.load(open("docs/eval_v4_results.json", encoding="utf-8"))
print("ts:", d["ts"])
for mode in ("v4", "v3", "pure_llm"):
    rows = [r for r in d["rows"] if r["mode"] == mode]
    lats = sorted(r["latency_ms"] for r in rows)
    over3 = [r["id"] for r in rows if r["latency_ms"] > 3000]
    fails = [r["id"] for r in rows if not r["accuracy"]]
    print(f"\n{mode}: acc={sum(r['accuracy'] for r in rows)}/{len(rows)} "
          f"avg={sum(lats)/len(lats):.0f} min={lats[0]:.0f} max={lats[-1]:.0f} "
          f"median={lats[len(lats)//2]:.0f} over3s={over3} fails={fails}")
    for r in sorted(rows, key=lambda x: -x["latency_ms"])[:4]:
        print(f"   id{r['id']} [{r['qtype']}] {r['latency_ms']:.0f}ms")
