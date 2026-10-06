# 临时：查看评估明细（人工智能NLP-RAG-图像内容解析及检索优化）
import json

d = json.load(open("docs/eval_v4_results.json", encoding="utf-8"))
print("ts:", d.get("ts"))
for mode in ("pure_llm", "v3", "v4"):
    s = d["summary"][mode]
    print(f"\n### {mode}: all={s['all']}")
    for qt in ("text", "table", "image"):
        if qt in s:
            print(f"   {qt}: acc={s[qt]['accuracy']} lat={s[qt]['avg_latency_ms']}")

print("\n===== v4 错题 =====")
for r in d["rows"]:
    if r["mode"] == "v4" and not r["accuracy"]:
        print(f"\nid{r['id']} [{r['qtype']}] {r['question']}")
        print(f"  kws_hit: {r['kws_hit']} n_images={r['n_images']} lat={r['latency_ms']}ms")
        print(f"  answer: {r['answer'][:500]}")

print("\n===== 图像题 v3 vs v4 =====")
for r in d["rows"]:
    if r["qtype"] == "image":
        print(f"\n[{r['mode']}] id{r['id']} acc={r['accuracy']} kws={r['kws_hit']} "
              f"n_img={r['n_images']} lat={r['latency_ms']}ms")
        print(f"  Q: {r['question']}")
        print(f"  A: {r['answer'][:350]}")
