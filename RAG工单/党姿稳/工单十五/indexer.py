# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-15-优化技术图纸与文本的跨模态检索流程
# 模块职责：编排"解析 → 分块 → 向量化 → 建索引"，并把中间结果缓存到 data/，重复运行不重算。

import os                                                        # 路径
import json                                                      # 块与元信息缓存
import pickle                                                    # BM25 缓存
import numpy as np                                               # 向量矩阵
import config                                                    # 配置
from logger import get_logger                                    # 日志
from pdf_parser import parse_document                            # 修复型解析器
from chunker import build_chunks                                 # 分块
from figure_describer import build_descriptions                  # 图纸描述
from bm25 import BM25, tokenize                                  # 全文索引

logger = get_logger("rag15.indexer")

CHUNKS_FILE = os.path.join(config.DATA_DIR, "chunks.json")        # 块缓存
EMBED_FILE = os.path.join(config.DATA_DIR, "embeddings.npy")      # 向量缓存
BM25_FILE = os.path.join(config.DATA_DIR, "bm25.pkl")             # BM25 缓存
META_FILE = os.path.join(config.DATA_DIR, "parse_meta.json")      # 解析元信息缓存
FIGDESC_FILE = os.path.join(config.DATA_DIR, "figure_descriptions.json")  # 图纸描述缓存
FIG_DIR = os.path.join(config.DATA_DIR, "figures")                # 图纸图片目录

_MODEL = None                                                     # 嵌入模型单例


def load_model():
    """惰性加载本地 m3e-base 嵌入模型，限制序列长度控制内存与耗时。"""
    global _MODEL
    if _MODEL is None:
        import torch                                              # 延迟导入
        from sentence_transformers import SentenceTransformer     # 延迟导入
        torch.set_num_threads(config.EMBED_THREADS)               # 绑定线程数
        logger.info("加载嵌入模型 %s ...", os.path.basename(config.EMBED_MODEL_PATH))
        _MODEL = SentenceTransformer(config.EMBED_MODEL_PATH)     # 本地加载，无需联网
        _MODEL.max_seq_length = 256                               # 限制长度，提速
        logger.info("嵌入模型加载完成（维度 %d）", _MODEL.get_sentence_embedding_dimension())
    return _MODEL


def encode(texts, batch_size=16):
    """把文本编码为 L2 归一化向量，之后点积即余弦相似度。"""
    model = load_model()                                          # 单例模型
    vecs = model.encode(texts, batch_size=batch_size,             # 批量编码
                        normalize_embeddings=True,                # L2 归一化
                        show_progress_bar=False,
                        convert_to_numpy=True)
    return np.asarray(vecs, dtype=np.float32)                     # 统一 float32


def build_index(pdf_path=None, force=False, use_vlm=True):
    """构建（或加载）完整索引，返回 chunks / embeddings / bm25 / meta。"""
    pdf_path = pdf_path or config.TARGET_PDF                      # 默认目标文档
    if not force and all(os.path.exists(f) for f in             # 缓存齐全则直接复用
                         (CHUNKS_FILE, EMBED_FILE, BM25_FILE, META_FILE)):
        logger.info("命中索引缓存，直接加载 data/ 下的块、向量与 BM25 索引")
        return load_index()

    os.makedirs(FIG_DIR, exist_ok=True)                           # 图纸目录
    logger.info("未命中缓存，开始解析文档 %s", os.path.basename(pdf_path))
    pages, meta = parse_document(pdf_path, figure_dir=FIG_DIR)    # 1) 修复型解析
    doc_name = meta["file"]                                       # 文档名

    figure_files = {p["page_no"]: p["figure_file"]                # 2) 导出图纸页图片映射
                    for p in pages if p.get("figure_file")}
    descs = build_descriptions(pages, figure_files, FIGDESC_FILE,  # 3) 生成图纸描述（VLM/规则）
                               doc_name, use_vlm=use_vlm)
    chunks = build_chunks(pages, doc_name, figure_descs=descs)    # 4) 分块（含图形块）

    texts = [c["text"] for c in chunks]                           # 块文本
    embeddings = encode(texts)                                    # 5) 向量化
    bm25 = BM25()                                                 # 6) 全文索引
    bm25.fit(texts)                                               # 构建倒排

    _dump(chunks, embeddings, bm25, meta, pages)                  # 7) 落盘缓存
    logger.info("索引构建完成：%d 块，向量维度 %d", len(chunks), embeddings.shape[1])
    return {"chunks": chunks, "embeddings": embeddings, "bm25": bm25,
            "meta": meta, "pages": pages}


def _dump(chunks, embeddings, bm25, meta, pages):
    """把索引写入 data/ 缓存。"""
    with open(CHUNKS_FILE, "w", encoding="utf-8") as fh:          # 块
        json.dump(chunks, fh, ensure_ascii=False, indent=2)
    np.save(EMBED_FILE, embeddings)                               # 向量
    with open(BM25_FILE, "wb") as fh:                             # BM25 状态
        pickle.dump(bm25.state(), fh)
    slim = {"meta": {k: v for k, v in meta.items() if k != "page_chars"},  # 元信息
            "pages": [{k: v for k, v in p.items() if k != "ocr_boxes"} for p in pages]}
    with open(META_FILE, "w", encoding="utf-8") as fh:
        json.dump(slim, fh, ensure_ascii=False, indent=2)


def load_index():
    """从 data/ 缓存加载索引。"""
    with open(CHUNKS_FILE, "r", encoding="utf-8") as fh:          # 块
        chunks = json.load(fh)
    embeddings = np.load(EMBED_FILE)                              # 向量
    with open(BM25_FILE, "rb") as fh:                             # BM25
        bm25 = BM25().load_state(pickle.load(fh))
    with open(META_FILE, "r", encoding="utf-8") as fh:            # 元信息
        meta = json.load(fh)
    logger.info("索引缓存加载完成：%d 块（%s）", len(chunks), meta["meta"]["file"])
    return {"chunks": chunks, "embeddings": embeddings, "bm25": bm25,
            "meta": meta["meta"], "pages": meta["pages"]}


if __name__ == "__main__":                                        # 自检
    idx = build_index(force=True, use_vlm=False)                  # 先不带 VLM 跑通流程
    print("块数:", len(idx["chunks"]), "向量:", idx["embeddings"].shape)
    for c in idx["chunks"]:                                       # 打印图形块
        if c["type"] == "figure":
            print("-" * 60)
            print(c["text"][:600])
