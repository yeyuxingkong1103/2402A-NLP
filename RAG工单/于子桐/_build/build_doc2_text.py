# -*- coding: utf-8 -*-
"""
补建《招股说明书2》的**纯正文**索引 (工单03 "表格解析前" 的对照基线)。

背景: run_builds.py 的 step_doc2 早期版本直接把"正文+表格"建成了 doc2_txt,
      缺少一个只有正文的版本, 工单03 就没法做"解析前 vs 解析后"的对比。
      本脚本从缓存的正文页码里重建纯正文索引, 不重复解析 PDF。

运行: python -u build_doc2_text.py
产出: 工单(3)/index/doc2_text
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rag_common as R

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PDF2 = os.path.join(ROOT, "工单(1)", "附件", "招股说明书2.pdf")
DOC2_TEXT = os.path.join(ROOT, "工单(3)", "index", "doc2_text")
CTX2 = "《招股说明书2》武汉力源信息技术股份有限公司 "


def main():
    if R.VectorStore.exists(DOC2_TEXT):
        print("已存在, 跳过:", DOC2_TEXT)
        return
    t0 = time.time()
    print("=" * 66)
    print("补建 《招股说明书2》纯正文索引 (工单03 对照基线)")
    print("=" * 66, flush=True)

    chunks = []
    for p in R.extract_pdf_pages(PDF2):
        for c in R.chunk_text(p["text"], 500, 50, page=p["page"],
                              source="招股说明书2.pdf"):
            c["text"] = CTX2 + c["text"]
            chunks.append(c)
    print(f"正文块 {len(chunks)}, 开始嵌入 ...", flush=True)
    R.build_index_from_chunks(chunks, DOC2_TEXT, shard=500)
    print(f"完成, 耗时 {(time.time()-t0)/60:.1f} 分钟")


if __name__ == "__main__":
    main()
