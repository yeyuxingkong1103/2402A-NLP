# -*- coding: utf-8 -*-
"""采集评测数据：对评测集跑一遍完整问答，保存回答与检索上下文。

用项目自身的 venv 运行（需要连 Milvus / Redis / MySQL）：

    .venv/bin/python evals/collect.py --tag baseline
    .venv/bin/python evals/collect.py --tag baseline --ids law-01,law-02

RAGAS 的判分在另一个 venv 里做（见 score.py），两份环境互不污染。
支持断点续跑：已采集的题目会自动跳过，中途中断后重跑即可续上。
"""
import argparse
import json
import os
import sys
import time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from app.core.rag_service import get_rag_service      # noqa: E402
from app.db import mysql_conn                          # noqa: E402
from app.core.role_service import RoleService          # noqa: E402

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(EVAL_DIR, "dataset.jsonl")


def load_dataset():
    with open(DATASET, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_done(path):
    """已采集的结果，用于断点续跑。"""
    if not os.path.exists(path):
        return {}
    done = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                done[r["id"]] = r
            except json.JSONDecodeError:
                continue        # 中断时可能留下半行，忽略
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="结果标记，如 baseline / optimized")
    ap.add_argument("--ids", default="", help="只跑指定题号，逗号分隔")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题")
    ap.add_argument("--user-id", type=int, default=1)
    args = ap.parse_args()

    out_path = os.path.join(EVAL_DIR, "results_%s.jsonl" % args.tag)
    dataset = load_dataset()
    if args.ids:
        want = {x.strip() for x in args.ids.split(",") if x.strip()}
        dataset = [d for d in dataset if d["id"] in want]
    if args.limit:
        dataset = dataset[:args.limit]

    done = load_done(out_path)
    todo = [d for d in dataset if d["id"] not in done]

    print("评测标记: %s" % args.tag)
    print("题目总数: %d，已完成 %d，本次待跑 %d" % (len(dataset), len(done), len(todo)))
    if not todo:
        print("全部已完成")
        return

    svc = get_rag_service()
    t_start = time.time()
    written = 0

    # 以追加方式写入，每条落盘一次，中断也只丢当前这条
    with open(out_path, "a", encoding="utf-8") as fout, \
            mysql_conn.session_scope() as db:
        # 触发角色缓存，避免首题额外耗时
        RoleService.list_roles(db)

        for i, item in enumerate(todo, 1):
            t0 = time.time()
            try:
                result = svc.chat(db, user_id=args.user_id,
                                  role_key=item["role_key"],
                                  question=item["question"])
                row = {
                    "id": item["id"],
                    "role_key": item["role_key"],
                    "category": item["category"],
                    "question": item["question"],
                    "ground_truth": item["ground_truth"],
                    "answer": result["answer"],
                    # RAGAS 的 contexts：本次检索到的切片原文
                    "contexts": [d["text"] for d in result["sources"]],
                    "source_titles": [d["title"] for d in result["sources"]],
                    "search_query": result.get("search_query", ""),
                    "elapsed": round(time.time() - t0, 2),
                    "error": None,
                }
                status = "OK"
            except Exception as e:
                row = {
                    "id": item["id"], "role_key": item["role_key"],
                    "category": item["category"], "question": item["question"],
                    "ground_truth": item["ground_truth"],
                    "answer": "", "contexts": [], "source_titles": [],
                    "search_query": "", "elapsed": round(time.time() - t0, 2),
                    "error": "%s: %s" % (type(e).__name__, e),
                }
                status = "ERR"

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            written += 1

            avg = (time.time() - t_start) / i
            left = avg * (len(todo) - i)
            print("[%2d/%2d] %-8s %-42s %5.1fs  剩余约 %.0f 分钟  %s" % (
                i, len(todo), item["id"], item["question"][:40],
                row["elapsed"], left / 60, status))

    print("\n完成 %d 题，总耗时 %.1f 分钟" % (written, (time.time() - t_start) / 60))
    print("结果: %s" % out_path)


if __name__ == "__main__":
    main()
