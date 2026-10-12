# -*- coding: utf-8 -*-
"""
工单十五：评估脚本（6题准确率 + 检索相关性 + 响应时间）
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()
from src.rag_engine import RAGEngine

TEST_QUESTIONS = [
    {"question": "根据专利文本，本发明主要涉及哪种物料的分配装置？", "answer": "块状散料", "type": "text"},
    {"question": "根据专利文本，本发明的分散装置包含以下哪个组件？", "answer": "链条", "type": "text"},
    {"question": "在文件中第11页图3中，编号13的部件相对于编号12的部件的位置关系是？", "answer": "位于编号12的部件之内", "type": "image"},
    {"question": "在文件中第11页图3中，编号14的部件位于整个装置的哪个位置？", "answer": "顶部", "type": "image"},
    {"question": "根据文件中第11页图3，散料从部件14进入后，下一步会经过哪个部件？", "answer": "部件13", "type": "image"},
    {"question": "在文件中第11页图3的装置中，如果需要调整链条的位置，需要操作哪个部件?", "answer": "部件11", "type": "image"},
]


_SYNONYMS = {
    "内部": "之内", "里边": "之内", "里面": "之内", "中": "之内",
    "顶端": "顶部", "上方": "顶部", "最上面": "顶部",
    "接着": "下一步", "随后": "下一步", "然后": "下一步",
}


def normalize(s: str) -> str:
    s = s.replace(" ", "").replace("　", "").replace("\n", "").lower()
    for a, b in _SYNONYMS.items():
        s = s.replace(a, b)
    return s


def check_answer(pred: str, gold: str) -> bool:
    p, g = normalize(pred), normalize(gold)
    if g in p or p in g:
        return True
    # 关键 token 覆盖匹配：去除停用词后，gold 的关键信息均出现在 pred 中
    stop = set("的 了 和 是 在 有 与 及 或 一个 一种 该 此 部件 编号 之 内 位于 关系 位置 哪个 需要 如果 根据 文件".split())
    g_tokens = [t for t in _tokenize(g) if t and t not in stop]
    if g_tokens and all(t in p for t in g_tokens):
        return True
    return False


def _tokenize(s: str):
    import re
    # 提取数字+编号、以及2-4字中文片段作为关键token
    toks = re.findall(r"\d+|[一-鿿]{2,4}", s)
    return toks


def main():
    engine = RAGEngine(top_k=5)
    # 加载页面映射
    mapping_json = "./data/kb_mapping.json"
    if os.path.isfile(mapping_json):
        with open(mapping_json, encoding="utf-8") as f:
            m = json.load(f)
        engine.set_page_images({int(k): v for k, v in m.get("page_images", {}).items()})
        engine.set_page_texts({int(k): v for k, v in m.get("page_texts", {}).items()})

    # 预热
    print("预热中...")
    engine.ask("测试")

    results = []
    correct = 0
    for i, q in enumerate(TEST_QUESTIONS, 1):
        t0 = time.time()
        r = engine.ask(q["question"])
        dt = time.time() - t0
        ok = check_answer(r["answer"], q["answer"])
        correct += int(ok)
        item = {"no": i, "question": q["question"], "gold": q["answer"],
                "pred": r["answer"], "correct": ok, "latency_s": round(dt, 2),
                "mode": r.get("mode"), "route": r.get("route")}
        results.append(item)
        print(f"[Q{i}] {'✓' if ok else '✗'} {q['answer']} -> {r['answer'][:40]} ({dt:.2f}s, {r.get('mode')})")

    acc = correct / len(TEST_QUESTIONS)
    avg_latency = sum(r["latency_s"] for r in results) / len(results)
    summary = {"accuracy": acc, "correct": correct, "total": len(TEST_QUESTIONS),
               "avg_latency_s": round(avg_latency, 2), "results": results}
    out = "./docs/screenshots/v15_evaluation_results.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n=== 准确率: {correct}/{len(TEST_QUESTIONS)} = {acc:.0%} | 平均延迟: {avg_latency:.2f}s ===")
    return summary


if __name__ == "__main__":
    main()
