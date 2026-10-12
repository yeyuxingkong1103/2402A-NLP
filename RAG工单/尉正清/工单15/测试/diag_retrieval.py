# 工单编号：人工智能NLP-RAG-修复低质量工业PDF的解析与信息丢失
"""诊断：逐题看召回排名与分数，找出「该进的没进、不该进的占了位」的情况

用法：
    ... diag_retrieval.py            # 用助手当前配置
    ... diag_retrieval.py --vec 0.5 --thr 0.1   # 试别的权重/阈值
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from ragflow_client import RAGFlow

HERE = Path(__file__).parent
IDS = json.loads((HERE / ".ragflow_ids.json").read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vec", type=float, default=0.3, help="vector_similarity_weight")
    ap.add_argument("--thr", type=float, default=0.2, help="similarity_threshold")
    ap.add_argument("--topn", type=int, default=6)
    ap.add_argument("--no-options", action="store_true")
    args = ap.parse_args()

    qs = json.loads((HERE / "测试问题.json").read_text(encoding="utf-8"))
    c = RAGFlow()
    c.load_token()
    print(f"阈值={args.thr}  向量权重={args.vec}  top_n={args.topn}\n")

    for n, q in enumerate(qs, 1):
        text = q["question"] if args.no_options else \
            q["question"] + "\n" + "\n".join(q["options"])
        code, j = c._req("POST", "/api/v1/retrieval", headers=c._h(), json={
            "question": text, "dataset_ids": [IDS["ds_id"]],
            "similarity_threshold": args.thr,
            "vector_similarity_weight": args.vec,
            "top_k": 1024, "page_size": args.topn,
        }, timeout=120)
        ch = (j.get("data") or {}).get("chunks", [])
        print(f"第{n}题（期望 {q['answer']}）{q['question'][:30]}  召回 {len(ch)} 条")
        for i, x in enumerate(ch):
            head = x.get("content", "").replace("\n", " ")[:56]
            print(f"   {i+1}. sim={x.get('similarity', 0):.3f} "
                  f"vec={x.get('vector_similarity', 0):.3f} "
                  f"term={x.get('term_similarity', 0):.3f} | {head}")
        if not ch:
            print("   （空 —— 全部低于阈值）")
        print()


if __name__ == "__main__":
    main()
