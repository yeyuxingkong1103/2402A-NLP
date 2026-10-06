# -*- coding: utf-8 -*-
# 工单编号：人工智能 NLP-RAG-基于 PDF 文档的问答系统优化
# 关联工单：人工智能NLP-RAG-PDF文档的表格解析及检索优化 | 人工智能NLP-RAG-图像内容解析及检索优化 | 人工智能NLP-RAG-Query理解优化任务 | 人工智能NLP-RAG-混合检索任务
# 模块：build_kb —— 一键构建知识库（支持多文档）
# 说明：解析 -> 清洗 -> 分块 -> 向量化 -> 建索引 -> 落盘。
#       默认解析 data/pdfs 下的**所有 PDF**（多文档合并索引，块带 doc 字段）。
#       用法：python app/build_kb.py                 # 解析全部 PDF 并构建
#             python app/build_kb.py --parse       # 先调用 MinerU 解析再构建
#             python app/build_kb.py --pdf xxx.pdf # 只构建单本

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
    ap.add_argument("--pdf", default="", help="只构建指定 PDF（默认 data/pdfs 下全部）")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    pdfs = [args.pdf] if args.pdf else config.list_pdfs()
    if not pdfs:
        print("data/pdfs 下没有 PDF"); return

    if args.parse:
        print("[1/5] MinerU 解析 ...")
        for p in pdfs:
            ok, _ = parse_mod.run_mineru(p, config.MINERU_DIR,
                                         log_path=os.path.join(config.DATA_DIR, "mineru_parse.log"))
            print("      %s mineru ok = %s" % (os.path.basename(p), ok))

    print("[2/5] 读取/解析块 ...")
    blocks, sources = parse_mod.build_all_blocks(pdfs)
    for d, (src, n, title) in sources.items():
        print("      %-16s source=%s blocks=%d title=%s" % (d, src, n, title))

    print("[3/5] 清洗 + 分块 ...")
    blocks = clean_blocks(blocks)
    chunks = build_chunks(blocks)
    per_doc = {}
    for c in chunks:
        per_doc[c.get("doc", "?")] = per_doc.get(c.get("doc", "?"), 0) + 1
    print("      chunks =", len(chunks), "| per-doc =", per_doc)

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
