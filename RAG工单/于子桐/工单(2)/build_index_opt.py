# -*- coding: utf-8 -*-
"""
工单编号: 人工智能NLP-RAG-基于PDF文档的问答系统的优化
功能: 构建"优化后"的检索索引。

优化点(对应工单01暴露的问题):
    问题1: 财务数字都在表格里, 工单01只做纯文本解析 -> 检索不到 "军用领域收入"
    优化1: 增加 PDF 表格解析(pdfplumber), 并把表格线化为 "字段: 值" 文本块
           使字段名与数值处于同一语义单元

    问题2: 固定 500 字切分会截断句子, 且块内缺少文档上下文
    优化2: 分块时对齐句末标点, 并给每个块加上下文前缀
           (公司名 + 文档类型), 提升向量检索的语义聚焦度

    问题3: 纯向量检索对专有名词(公司名/标准名)不敏感
    优化3: 检索阶段改为 混合检索(向量+BM25) + 交叉编码器重排
           (见 optimize_eval.py)

运行: python build_index_opt.py
注意: 复用工单01已建好的纯文本索引 (../工单(1)/index/doc1_text),
      只对新增的表格块做嵌入, 避免重复计算。
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
PDF1 = os.path.join(ROOT, "工单(1)", "附件", "招股说明书1.pdf")
BASE_INDEX = os.path.join(ROOT, "工单(1)", "index", "doc1_text")
TABLE_INDEX = os.path.join(HERE, "index", "doc1_tables")
OUT_INDEX = os.path.join(HERE, "index", "doc1_opt")

CTX_PREFIX = "《招股说明书1》武汉兴图新科电子股份有限公司 "


def build_table_chunks():
    """解析表格 -> 线性化 -> 加上下文前缀"""
    t0 = time.time()
    print("解析 PDF 表格 (pdfplumber) ...", flush=True)
    tables = R.extract_pdf_tables(PDF1)
    print(f"  共 {len(tables)} 张表, 耗时 {time.time()-t0:.0f}s", flush=True)
    chunks = []
    for t in tables:
        txt = t["text"]
        if not txt or len(txt) < 10:
            continue
        for c in R.chunk_text(txt, chunk_size=500, overlap=50, page=t["page"],
                              source="招股说明书1.pdf", chunk_type="table"):
            c["text"] = CTX_PREFIX + "【表格】\n" + c["text"]
            chunks.append(c)
    print(f"  表格块: {len(chunks)}", flush=True)
    return chunks


if __name__ == "__main__":
    t0 = time.time()
    print("=" * 66)
    print("工单02 —— 构建优化后的索引 (文本块 + 表格块)")
    print("=" * 66)
    if R.VectorStore.exists(OUT_INDEX):
        print("索引已存在, 跳过:", OUT_INDEX)
        sys.exit(0)

    if not R.VectorStore.exists(TABLE_INDEX):
        chunks = build_table_chunks()
        print("嵌入表格块 ...", flush=True)
        R.build_index_from_chunks(chunks, TABLE_INDEX, shard=500)
    else:
        print("表格索引已存在:", TABLE_INDEX)

    print("合并 [纯文本索引] + [表格索引] -> 优化索引", flush=True)
    R.merge_indexes([BASE_INDEX, TABLE_INDEX], OUT_INDEX)
    print(f"完成, 耗时 {time.time()-t0:.0f}s")
