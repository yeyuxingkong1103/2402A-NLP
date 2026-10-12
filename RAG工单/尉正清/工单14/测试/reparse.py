# 工单编号：人工智能NLP-RAG-修复低质量工业PDF的解析与信息丢失
"""改解析配置 + 重新解析文档，并盯到解析结束

RAGFlow 对「解析配置没变」的文档会直接复用旧 chunk、不入队
（rag/svr/task_service.py:467-511 的 digest 去重），所以想真重解析
就得先动配置 —— 这里也正好是第 4 轮要改的东西。
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")   # SystemExit 的报错走 stderr
from ragflow_client import RAGFlow

HERE = Path(__file__).parent
IDS = json.loads((HERE / ".ragflow_ids.json").read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-token-num", type=int, default=512)
    ap.add_argument("--layout", default="DeepDOC")
    ap.add_argument("--image-context-size", type=int, default=0)
    ap.add_argument("--no-parse", action="store_true", help="只改配置不解析")
    ap.add_argument("--force", action="store_true",
                    help="先删掉已入库的 chunk，强制走真正的重解析（配置没变时 RAGFlow 会跳过）")
    args = ap.parse_args()

    c = RAGFlow()
    c.load_token()

    if args.force:
        print("=== 删除已有 chunk ===")
        code, j = c._req("DELETE",
                         f"/api/v1/datasets/{IDS['ds_id']}/documents/{IDS['doc_id']}/chunks",
                         headers=c._h())
        print(f"  HTTP {code} {json.dumps(j, ensure_ascii=False)[:150]}")

    print("=== 改知识库解析配置 ===")
    # 只发接口 schema 认的字段：image_context_size / table_context_size 不在
    # 这个 schema 里，带上会被整体拒掉（"Extra inputs are not permitted"），
    # 配置一条都改不进去 —— 之前就白改过一次。
    c.update_dataset(IDS["ds_id"], parser_config={
        "chunk_token_num": args.chunk_token_num,
        "delimiter": "\n",
        "layout_recognize": args.layout,
        "auto_keywords": 0,
        "auto_questions": 0,
        "html4excel": False,
    })
    if args.no_parse:
        return

    print("\n=== 触发重新解析 ===")
    c.parse(IDS["ds_id"], [IDS["doc_id"]])

    print("\n=== 等解析完成 ===")
    t0 = time.time()
    last = ""
    while time.time() - t0 < 900:
        d = c.doc_status(IDS["ds_id"], IDS["doc_id"])
        prog = d.get("progress", 0)
        msg = (d.get("progress_msg") or "").strip().splitlines()
        cur = msg[-1] if msg else ""
        if cur != last:
            print(f"  [{prog:>5.1f}%] {cur[:150]}")
            last = cur
        if prog >= 1.0:
            print(f"\n  解析完成，耗时 {time.time()-t0:.0f}s，chunk 数 {d.get('chunk_count')}")
            return
        if prog < 0:
            raise SystemExit(f"  解析失败：\n{(d.get('progress_msg') or '')[-1500:]}")
        time.sleep(5)
    raise SystemExit("  超时")


if __name__ == "__main__":
    main()
