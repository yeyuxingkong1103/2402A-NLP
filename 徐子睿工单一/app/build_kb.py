# -*- coding: utf-8 -*-
# 工单编号：人工智能 NLP-RAG-基于 PDF 文档的问答系统优化
# 模块：build_kb —— 一键构建知识库
# 说明：解析(MinerU) -> 清洗 -> 分块 -> 向量化 -> 建索引 -> 落盘。
#       用法：python app/build_kb.py            # 复用已解析结果（需先跑 MinerU）
#             python app/build_kb.py --parse  # 先调用 MinerU 解析 PDF 再构建

import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

import config           # noqa: E402
import parse as parse_mod   # noqa: E402
from clean import clean_blocks  # noqa: E402
from chunk import build_chunks  # noqa: E402
import kb as kb_mod     # noqa: E402
import llm              # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parse", action="store_true", help="先调用 MinerU 解析 PDF")
    ap.add_argument("--pdf", default=config.DEFAULT_PDF)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    if args.parse:
        print("[1/5] MinerU 解析 ...")
        ok, _ = parse_mod.run_mineru(args.pdf, config.MINERU_DIR,
                                     log_path=os.path.join(config.DATA_DIR, "mineru_parse.log"))
        print("      mineru ok =", ok)

    print("[2/5] 读取解析结果 ...")
    blocks, source = parse_mod.build_blocks(pdf_path=args.pdf)
    print("      source =", source, "| blocks =", len(blocks))

    print("[3/5] 清洗 + 分块 ...")
    blocks = clean_blocks(blocks)
    chunks = build_chunks(blocks)
    print("      chunks =", len(chunks))

    print("[4/5] 向量化（bge-m3 via Ollama）...")
    t0 = time.time()
    kb = kb_mod.KB()
    kb.build(chunks, batch=args.batch,
             progress=lambda d, t: print("      embed %d/%d" % (d, t), end="\r"))
    print("\n      耗时 %.1fs | dim = %d" % (time.time() - t0, kb.emb.shape[1]))

    print("[5/5] 保存索引 ...")
    kb.save()
    print("      saved ->", config.INDEX_DIR)
    print("完成 ✅  chunks=%d" % len(chunks))


if __name__ == "__main__":
    main()
