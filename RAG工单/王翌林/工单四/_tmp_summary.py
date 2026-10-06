# 临时：输出最终评估汇总（工单四）
import json

d = json.load(open("docs/eval_v4_results.json"))
print("ts:", d.get("ts"))
for mode in ("pure_llm", "v3", "v4"):
    rows = [r for r in d["rows"] if r.get("mode") == mode]
    n = len(rows)
    acc = sum(r["accuracy"] for r in rows) / n
    faith = sum(r.get("faithfulness", 0) for r in rows) / n
    rel = sum(r.get("relevance", 0) for r in rows) / n
    cp = sum(r.get("ctx_precision", 0) for r in rows) / n
    cr = sum(r.get("ctx_recall", 0) for r in rows) / n
    lat = sum(r.get("latency_ms", 0) for r in rows) / n
    tbl = [r for r in rows if r.get("qtype") == "table"]
    img = [r for r in rows if r["id"] in (105, 106)]
    ta = sum(r["accuracy"] for r in tbl) / len(tbl) if tbl else 0
    ia = sum(r["accuracy"] for r in img) / len(img)
    over3 = sum(1 for r in rows if r.get("latency_ms", 0) > 3000)
    print(f"{mode}: n={n} acc={acc:.3f} faith={faith:.3f} rel={rel:.3f} "
          f"cp={cp:.3f} cr={cr:.3f} lat={lat:.0f}ms table={ta:.3f} "
          f"img={ia:.3f} over3s={over3}")
