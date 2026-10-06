# 临时脚本：列出 v4 评估全部题目对错（人工智能NLP-RAG-图像内容解析及检索优化）
import json

d = json.load(open("docs/eval_v4_results.json", encoding="utf-8"))
qs = d["questions"]
print("questions 类型:", type(qs).__name__)
if isinstance(qs, list):
    for i, q in enumerate(qs):
        if isinstance(q, dict):
            print(i, q.get("id"), q.get("qtype"), str(q.get("question"))[:60])
        else:
            print(i, q)

print("\n===== v4 全部 16 题 =====")
v4 = [r for r in d["rows"] if r["mode"] == "v4"]
for r in v4:
    print(f"seq={r.get('id')} qtype={r.get('qtype'):6s} acc={r.get('accuracy')} lat={r.get('latency_ms'):.0f}ms n_img={r.get('n_images')} | {str(r.get('question'))[:70]}")

print("\n===== v4 错题详情 =====")
for r in v4:
    if float(r.get("accuracy", 0)) < 1.0:
        print(f"\n### seq={r.get('id')} qtype={r.get('qtype')} | {r.get('question')}")
        print("kws_hit:", r.get("kws_hit"))
        print("answer:", str(r.get("answer"))[:400])

print("\n===== v3 错题详情（对比） =====")
for r in [x for x in d["rows"] if x["mode"] == "v3" and float(x.get("accuracy", 0)) < 1.0]:
    print(f"\n### seq={r.get('id')} qtype={r.get('qtype')} | {r.get('question')}")
    print("answer:", str(r.get("answer"))[:300])
