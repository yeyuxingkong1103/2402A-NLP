# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-Embedding模型微调任务
scripts/gen_qa_pairs_v11.py —— 工单十一 运行问答对生成（对应工单"运行问答对生成"）

流程：分层抽样年报 chunk → DeepSeek 逐 chunk 生成金融问题 → 解析为
(query, positive) 正例对 → 按 chunk 防泄漏切分 train/dev → 落盘：
  data/finetune_v11/qa_pairs_train.jsonl
  data/finetune_v11/qa_pairs_dev.jsonl
  data/finetune_v11/qa_corpus_chunks.json   （评估语料=抽样chunk+干扰chunk）
  data/finetune_v11/gen_report.json         （生成统计）

用法：
  python scripts/gen_qa_pairs_v11.py [--per-doc 24] [--n-questions 2] \
      [--distractors 800] [--limit N]   # --limit 仅调试小跑
"""
import argparse
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
load_dotenv()  # 工单十一：显式加载 .env（DEEPSEEK_API_KEY），独立脚本必须

from src.finetune_v11.qa_dataset import (build_qa_prompt, iter_chunks,
                                         make_pair, parse_questions,
                                         sample_chunks, split_train_dev,
                                         write_jsonl)
from src.finetune_v11.ir_eval import pick_distractor_chunks

WORK_ORDER = "人工智能NLP-RAG-Embedding模型微调任务"
OUT_DIR = Path("data/finetune_v11")


def main():
    ap = argparse.ArgumentParser(description=f"工单十一 问答对生成（{WORK_ORDER}）")
    ap.add_argument("--chunks-dir", default="data/ccf_reports/chunks")
    ap.add_argument("--per-doc", type=int, default=24, help="每份年报抽样 chunk 数")
    ap.add_argument("--n-questions", type=int, default=2, help="每 chunk 生成问题数")
    ap.add_argument("--distractors", type=int, default=800, help="评估语料干扰 chunk 数")
    ap.add_argument("--dev-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="仅处理前 N 个 chunk（调试用）")
    args = ap.parse_args()

    from src.llm_client_v4 import chat

    all_chunks = list(iter_chunks(Path(args.chunks_dir)))
    logger.info(f"语料 chunk 总数（≥200字）: {len(all_chunks)}")
    sampled = sample_chunks(all_chunks, per_doc=args.per_doc, seed=args.seed)
    if args.limit:
        sampled = sampled[:args.limit]
    logger.info(f"分层抽样 {len(sampled)} 个 chunk 用于问答对生成")

    pairs = []
    failures = 0
    t0 = time.perf_counter()
    for i, chunk in enumerate(sampled, 1):
        try:
            res = chat(messages=build_qa_prompt(chunk["text"], args.n_questions),
                       temperature=0.7, max_tokens=600)
            questions = parse_questions(res["content"])[:args.n_questions]
        except Exception as e:
            logger.warning(f"[{i}/{len(sampled)}] 生成失败({chunk['doc_id']} p{chunk['page']}): {e}")
            failures += 1
            continue
        for q in questions:
            pairs.append(make_pair(q, chunk))
        if i % 20 == 0 or i == len(sampled):
            logger.info(f"[{i}/{len(sampled)}] 已累计 {len(pairs)} 对"
                        f"（{(time.perf_counter()-t0)/60:.1f}min）")

    if not pairs:
        logger.error("未生成任何问答对，终止")
        sys.exit(1)

    train, dev = split_train_dev(pairs, dev_ratio=args.dev_ratio, seed=args.seed)

    # 评估语料：全部抽样 chunk + 干扰 chunk（排除抽样避免泄漏重叠无所谓，
    # 但为增大难度加入其他 chunk；排除 id 防重复）
    sampled_ids = {c["chunk_id"] for c in sampled}
    corpus = sampled + pick_distractor_chunks(all_chunks, sampled_ids,
                                              n=args.distractors, seed=args.seed)

    write_jsonl(train, OUT_DIR / "qa_pairs_train.jsonl")
    write_jsonl(dev, OUT_DIR / "qa_pairs_dev.jsonl")
    (OUT_DIR / "qa_corpus_chunks.json").write_text(
        json.dumps(corpus, ensure_ascii=False), encoding="utf-8")
    report = {"work_order": WORK_ORDER, "seed": args.seed,
              "per_doc": args.per_doc, "n_questions": args.n_questions,
              "sampled_chunks": len(sampled), "llm_failures": failures,
              "pairs_total": len(pairs), "train": len(train), "dev": len(dev),
              "corpus_chunks": len(corpus),
              "elapsed_min": round((time.perf_counter() - t0) / 60, 1)}
    (OUT_DIR / "gen_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    logger.info(f"完成: {report}")


if __name__ == "__main__":
    main()
