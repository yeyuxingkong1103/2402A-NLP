# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-功能测试及评估任务
# 关联工单：人工智能NLP-RAG-PDF文档的表格解析及检索优化 | 人工智能NLP-RAG-图像内容解析及检索优化 | 人工智能NLP-RAG-Query理解优化任务 | 人工智能NLP-RAG-混合检索任务 | 人工智能NLP-RAG-功能测试及评估任务
# 模块：build_kb_txt —— 金融年报语料（ccf_competition）入库
# 说明：语料为 JSONL（每行 {"page","allrow","type","inside"}），按页聚合成 blocks，再清洗/分块/向量化/建索引。
#       文档 id = 公司名（如 平安银行），与 config.DOC_ALIASES 对齐，实现按公司路由。
import os
import sys
import json
import glob
import time
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

import config                    # noqa: E402
from clean import clean_blocks   # noqa: E402
from chunk import build_chunks   # noqa: E402
import kb as kb_mod              # noqa: E402


def company_of(path):
    """从文件名 `2020-02-14__000001__平安银行__2019年年报.txt` 取公司名。"""
    base = os.path.splitext(os.path.basename(path))[0]
    parts = base.split("__")
    return parts[2] if len(parts) >= 4 else base


def load_txt_blocks():
    blocks = []
    for p in sorted(glob.glob(os.path.join(config.TXT_DIR, "*.txt"))):
        doc = company_of(p)
        n = 0
        for line in open(p, encoding="utf-8", errors="replace"):
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            t = (o.get("inside") or "").strip()
            if not t:
                continue
            blocks.append({"doc": doc, "page": int(o.get("page") or 1),
                           "type": o.get("type") or "text", "text": t,
                           "section": doc, "level": 0})
            n += 1
        print("      %-28s blocks=%d" % (doc, n))
    return blocks


def main():
    print("[1/4] 读取金融年报语料 ...")
    blocks = load_txt_blocks()
    print("      total blocks =", len(blocks))

    print("[2/4] 清洗 + 分块 ...")
    chunks = clean_blocks(blocks)
    chunks = build_chunks(chunks)
    per_doc = {}
    for c in chunks:
        per_doc[c.get("doc", "?")] = per_doc.get(c.get("doc", "?"), 0) + 1
    print("      chunks =", len(chunks), "| per-doc =", per_doc)

    print("[3/4] 向量化（bge-m3 via Ollama）...")
    t0 = time.time()
    kb = kb_mod.KB()
    kb.build(chunks, batch=16, progress=lambda d, t: print("      embed %d/%d" % (d, t), end="\r"))
    print("\n      耗时 %.1fs | dim = %d" % (time.time() - t0, kb.emb.shape[1]))

    print("[4/4] 保存索引 ...")
    kb.save()
    print("      saved ->", config.INDEX_DIR, "| chunks =", len(chunks))


if __name__ == "__main__":
    main()
