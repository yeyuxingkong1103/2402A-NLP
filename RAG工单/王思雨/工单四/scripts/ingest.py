# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
# 文件说明：数据入库流程。PDF 解析 → ★图像语义解析 → 语义分块 → bge-m3 向量化 → Milvus 入库 → BM25 索引。
# 用法：python scripts/ingest.py [--no-mineru] [--only 招股说明书2]
import argparse, json, pickle, sys, time                                     # 参数、序列化、索引持久化、计时
from pathlib import Path                                                     # 路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))               # 让 scripts 能 import 项目根的 config 与 backend
from backend import chunking as G                                            # 语义分块
from backend import imaging as I                                             # ★图像内容解析
from backend import parse as P                                               # PDF 解析
from backend import rag as R                                                 # 复用 BM25 构建
from backend import store as V                                               # 向量库
import config as C                                                           # 集中配置


def image_blocks(images: list, doc: str) -> tuple:                            # ★图像解析批处理
    """逐张图调用多模态解析，产出图像语义块；单张失败只跳过该张，不影响整篇。"""
    out, types = [], {}                                                      # 块列表与图像类型统计
    for k, im in enumerate(images):                                          # 逐张
        b = I.extract(im["path"], doc=doc, page=im["page"], caption=im.get("caption", ""))  # CLIP + OCR + 层级还原
        if not b: continue                                                   # 过小或解析失败的图丢弃
        b["content"] = b.pop("text")                                         # 统一成下游约定的 content 字段
        b["type"] = "image"                                                  # 块类型
        b["doc"] = doc                                                       # 所属文档
        b["image_path"] = str(Path(b["image_path"]).relative_to(C.ROOT))      # 转成相对路径，便于接口回显
        out.append(b)                                                        # 收下
        types[b["image_type"]] = types.get(b["image_type"], 0) + 1            # 统计语义类型
        if (k + 1) % 20 == 0: print(f"    图像解析 {k + 1}/{len(images)}", flush=True)  # 进度
    return out, types                                                        # 返回块与类型分布


def main() -> None:                                                          # 入库主流程
    ap = argparse.ArgumentParser()                                           # 命令行
    ap.add_argument("--no-mineru", action="store_true", help="跳过 MinerU，直接用 pymupdf/pdfplumber 兜底解析")  # 快速回归用
    ap.add_argument("--only", default=None, help="只处理指定文档名（不含 .pdf）")   # 单篇重建
    ap.add_argument("--no-images", action="store_true", help="跳过图像语义解析（仅用于对照实验）")  # 对照组
    a = ap.parse_args()                                                      # 解析参数
    pdfs = [p for p in C.docs() if not p.stem.startswith("_")]                # 知识库内正式 PDF（下划线开头的是切片）
    if a.only: pdfs = [p for p in pdfs if p.stem == a.only]                   # 只留指定文档
    if not pdfs: raise SystemExit(f"未在 {C.RAW_DIR} 找到 PDF")                 # 无文档则退出
    all_blocks, report = [], []                                              # 全部块与逐篇报告
    for pdf in pdfs:                                                         # 逐篇串行（本机 GPU 只有 8G，并行会被系统杀）
        t0 = time.time()                                                     # 单篇计时
        print(f"[1/4] 解析 {pdf.stem} …", flush=True)                          # 进度
        blocks, images, parser = P.parse_pdf(pdf, use_mineru=not a.no_mineru)  # 正文 + 表格行 + 图像任务
        print(f"[2/4] 图像语义解析 {len(images)} 张 …", flush=True)             # 进度
        ib, types = ([], {}) if a.no_images else image_blocks(images, pdf.stem)  # ★多模态图像解析
        blocks += ib                                                         # 图像块并入
        all_blocks += blocks                                                 # 汇总
        report.append(dict(doc=pdf.stem, parser=parser, text_blocks=len(blocks) - len(ib),  # 逐篇报告
                           images=len(ib), image_types=types,                # 图像块数与语义类型分布
                           elapsed=round(time.time() - t0, 1)))              # 耗时
        print(f"    {pdf.stem}: 正文块 {len(blocks) - len(ib)}，图像块 {len(ib)}，耗时 {report[-1]['elapsed']}s", flush=True)  # 小结
    print("[3/4] 语义分块 …", flush=True)                                      # 进度
    chunks = G.semantic_chunks(all_blocks, V.embed)                           # 语义分块（图像块独立成块）
    print(f"    {len(all_blocks)} 个单元 → {len(chunks)} 个语义块", flush=True)  # 块规模
    print("[4/4] 向量入库 + BM25 …", flush=True)                               # 进度
    client = V.milvus_client()                                                # 连接 Milvus
    V.recreate_collection(client)                                             # 重建集合，保证状态干净
    V.insert_chunks(client, chunks)                                           # 向量入库
    bm25 = R.build_bm25(chunks)                                               # 建 BM25 索引
    stats = dict(docs=report, chunks=len(chunks),                              # 逐篇报告与总块数
                 image_chunks=sum(1 for c in chunks if c["block_type"] == "image"),  # ★图像语义块数
                 table_rows=sum(1 for c in chunks if c["block_type"] == "table_row"),  # 表格行块数
                 engine="rrf")                                                # 融合算法标记
    C.INDEX_FILE.write_text(json.dumps(dict(chunks=chunks, stats=stats), ensure_ascii=False), encoding="utf-8")  # 块与统计落盘
    pickle.dump(bm25, open(C.BM25_FILE, "wb"))                                # BM25 落盘，重启免重建
    print(json.dumps(stats, ensure_ascii=False, indent=1), flush=True)         # 输出报告
    print(f"入库完成：{len(chunks)} 块（其中图像语义块 {stats['image_chunks']}）")   # 完成提示


if __name__ == "__main__":                                                    # 入口
    main()                                                                    # 执行
