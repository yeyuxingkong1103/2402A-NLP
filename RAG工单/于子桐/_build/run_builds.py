# -*- coding: utf-8 -*-
"""
一次性把工单 02 / 03 / 04 / 07 需要的检索索引全部建好。

为什么要集中建:
    本机没有 GPU, bge-base-zh-v1.5 在 CPU 上约 2.6 块/秒。
    多个脚本同时跑会互相抢 CPU, 反而更慢, 所以串行在一个进程里跑完。

步骤:
    1. 招股说明书1 表格解析 -> 工单(2)/index/doc1_tables
       合并 工单(1)/index/doc1_text + doc1_tables -> 工单(2)/index/doc1_opt
    2. 招股说明书2 正文(文本) + 表格 -> 工单(3)/index/doc2_txt
    3. 招股说明书2 图像区域 OCR 解析 -> 工单(3)/index/doc2_img
    4. 合并 1+2+3 -> 工单(3)/index/all_data (全部数据, 供 05/06/12/13 用)
    5. ccf_competition 9 份年报 -> 工单(7)/index/ccf

所有 PDF 解析结果都带缓存 (cache/*.json), 中断重跑不会重复解析。
运行: python -u run_builds.py
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
W1 = os.path.join(ROOT, "工单(1)")
W2 = os.path.join(ROOT, "工单(2)")
W3 = os.path.join(ROOT, "工单(3)")
W7 = os.path.join(ROOT, "工单(7)")
ATT = os.path.join(W1, "附件")
PDF1 = os.path.join(ATT, "招股说明书1.pdf")
PDF2 = os.path.join(ATT, "招股说明书2.pdf")

DOC1_TEXT = os.path.join(W1, "index", "doc1_text")
DOC1_TABLES = os.path.join(W2, "index", "doc1_tables")
DOC1_OPT = os.path.join(W2, "index", "doc1_opt")
DOC2_TEXT = os.path.join(W3, "index", "doc2_text")
DOC2_TXT = os.path.join(W3, "index", "doc2_txt")
DOC2_IMG = os.path.join(W3, "index", "doc2_img")
DOC2 = os.path.join(W3, "index", "doc2")
ALL_DATA = os.path.join(W3, "index", "all_data")
CCF = os.path.join(W7, "index", "ccf")

CTX1 = "《招股说明书1》武汉兴图新科电子股份有限公司 "
CTX2 = "《招股说明书2》武汉力源信息技术股份有限公司 "


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def step_tables_doc1():
    """步骤1: 招股说明书1 的表格 -> 索引, 再和纯文本索引合并"""
    if R.VectorStore.exists(DOC1_OPT):
        log("步骤1 已完成, 跳过")
        return
    if not R.VectorStore.exists(DOC1_TABLES):
        log("步骤1a 解析招股说明书1 表格 (pdfplumber, 约 10 分钟) ...")
        tables = R.extract_pdf_tables(PDF1)
        log(f"       共 {len(tables)} 张表")
        chunks = []
        for t in tables:
            if not t["text"] or len(t["text"]) < 10:
                continue
            for c in R.chunk_text(t["text"], 500, 50, page=t["page"],
                                  source="招股说明书1.pdf", chunk_type="table"):
                c["text"] = CTX1 + "【表格】\n" + c["text"]
                chunks.append(c)
        log(f"       表格块 {len(chunks)}, 开始嵌入 ...")
        R.build_index_from_chunks(chunks, DOC1_TABLES, shard=500)
    log("步骤1b 合并 [纯文本] + [表格] -> doc1_opt")
    R.merge_indexes([DOC1_TEXT, DOC1_TABLES], DOC1_OPT)


def step_doc2():
    """步骤2: 招股说明书2 正文 + 表格"""
    if R.VectorStore.exists(DOC2_TEXT) and R.VectorStore.exists(DOC2_TXT):
        log("步骤2 已完成, 跳过")
        return
    chunks, text_chunks = [], []
    log("步骤2a 解析招股说明书2 正文 (pypdf) ...")
    for p in R.extract_pdf_pages(PDF2):
        for c in R.chunk_text(p["text"], 500, 50, page=p["page"],
                              source="招股说明书2.pdf"):
            c["text"] = CTX2 + c["text"]
            chunks.append(c)
            text_chunks.append(dict(c))
    log(f"       正文块 {len(chunks)}")
    # 纯正文索引: 作为工单03 "表格解析前" 的对照基线
    if not R.VectorStore.exists(DOC2_TEXT):
        log("步骤2a2 建立纯正文索引 (doc2_text, 表格解析前的基线) ...")
        R.build_index_from_chunks(text_chunks, DOC2_TEXT, shard=500)
    log("步骤2b 解析招股说明书2 表格 (pdfplumber, 约 8 分钟) ...")
    tables = R.extract_pdf_tables(PDF2)
    n0 = len(chunks)
    for t in tables:
        if not t["text"] or len(t["text"]) < 10:
            continue
        for c in R.chunk_text(t["text"], 500, 50, page=t["page"],
                              source="招股说明书2.pdf", chunk_type="table"):
            c["text"] = CTX2 + "【表格】\n" + c["text"]
            chunks.append(c)
    log(f"       {len(tables)} 张表 -> 表格块 {len(chunks)-n0}, 合计 {len(chunks)}")
    log("步骤2c 嵌入 ...")
    R.build_index_from_chunks(chunks, DOC2_TXT, shard=500)


def step_images():
    """步骤3: 招股说明书2 图像区域 OCR 解析 (工单04)"""
    if R.VectorStore.exists(DOC2_IMG):
        log("步骤3 已完成, 跳过")
        return
    log("步骤3a 找出图像区域 ...")
    regions = R.extract_pdf_chart_regions(PDF2)
    log(f"       {len(regions)} 个图像区域, 开始 OCR ...")
    chunks = []
    for n, reg in enumerate(regions, 1):
        img_path = os.path.join(HERE, "_pages",
                                f"chart_p{reg['page']}.png")
        try:
            R.render_region_image(PDF2, reg["page"], reg["bbox"], img_path, dpi=300)
        except Exception as e:
            log(f"       第{reg['page']}页渲染失败: {e}")
            continue
        try:
            from PIL import Image
            cap = R.ocr_chart_caption(Image.open(img_path))
        except Exception as e:
            log(f"       第{reg['page']}页 OCR 失败: {e}")
            cap = ""
        parts = [cap, reg.get("text", "")]
        body = "\n".join(x for x in parts if x)
        if len(body) < 10:
            continue
        for c in R.chunk_text(body, 500, 50, page=reg["page"],
                              source="招股说明书2.pdf", chunk_type="image"):
            c["text"] = (CTX2 + f"【图像】第{reg['page']}页 图表内容:\n" + c["text"])
            c["image"] = img_path
            chunks.append(c)
        if n % 10 == 0:
            log(f"       OCR 进度 {n}/{len(regions)}")
    log(f"       图像块 {len(chunks)}, 开始嵌入 ...")
    R.build_index_from_chunks(chunks, DOC2_IMG, shard=500)


def step_merge_all():
    """步骤4: 合并成一库"""
    if R.VectorStore.exists(ALL_DATA):
        log("步骤4 已完成, 跳过")
        return
    log("步骤4 合并 doc1_opt + doc2_txt + doc2_img -> all_data")
    R.merge_indexes([DOC1_OPT, DOC2_TXT, DOC2_IMG], ALL_DATA)
    # 工单03 里的名字叫 doc2 / all_tables, 这里一并生成, 保持工单脚本可直接跑
    R.merge_indexes([DOC2_TXT, DOC2_IMG], DOC2)
    R.merge_indexes([DOC1_OPT, DOC2], os.path.join(W3, "index", "all_tables"))


def step_ccf():
    """步骤5: ccf_competition 9 份年报 (工单07 语料)"""
    if R.VectorStore.exists(CCF):
        log("步骤5 已完成, 跳过")
        return
    sys.path.insert(0, W7)
    import build_index_ccf as B
    chunks = B.build_chunks()
    log(f"       {len(chunks)} 块, 开始嵌入 ...")
    R.build_index_from_chunks(chunks, CCF, shard=500)


if __name__ == "__main__":
    t0 = time.time()
    print("=" * 66)
    print("RAG 工单 02/03/04/07 索引集中构建")
    print("=" * 66, flush=True)
    for fn in (step_tables_doc1, step_doc2, step_images,
               step_merge_all, step_ccf):
        log(f"--- {fn.__name__} ---")
        try:
            fn()
        except Exception as e:
            import traceback
            traceback.print_exc()
            log(f"!! {fn.__name__} 失败: {e}")
    log(f"全部完成, 总耗时 {(time.time()-t0)/60:.1f} 分钟")
