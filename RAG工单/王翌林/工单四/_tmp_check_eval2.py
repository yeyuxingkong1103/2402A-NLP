# 临时：汇总评估 JSON 各模式每题准确率/延迟（工单四）
import json

d = json.load(open("docs/eval_v4_results.json"))
for mode in ("pure_llm", "v3", "v4"):
    print(f"==== {mode} ====")
    accs, lats, fails = [], [], []
    for r in d["rows"]:
        if r.get("mode") != mode:
            continue
        accs.append(r["accuracy"])
        lats.append(r.get("latency_ms", 0))
        flag = "" if r["accuracy"] >= 1.0 else "  <<< FAIL"
        print(f"id={r['id']:>3} acc={r['accuracy']} lat={r.get('latency_ms',0):.0f}ms "
              f"n_img={r.get('n_images', 0)}{flag}")
        if r["accuracy"] < 1.0:
            fails.append(r)
    if accs:
        print(f"mean acc={sum(accs)/len(accs):.3f} mean lat={sum(lats)/len(lats):.0f}ms "
              f"fails={len(fails)}")
    for r in fails:
        print("--- FAIL id", r["id"], r["question"])
        print("    ans:", (r.get("answer") or "")[:300].replace("\n", " "))
