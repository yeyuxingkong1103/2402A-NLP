# -*- coding: utf-8 -*-
"""
离线建库模块
功能：PDF解析（PyMuPDF/PaddleOCR/MinerU）、文本分块、向量入库Milvus、BM25稀疏索引构建
"""
import os
import re
import pickle
import config as C
from logger import get_logger

log = get_logger("build_kb")


# ======================== PDF解析 ========================
def load_pdf_pymupdf(path):
    """使用PyMuPDF提取PDF文本（默认方法，速度快）"""
    import fitz  # PyMuPDF
    doc = fitz.open(str(path))
    pages = []
    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        pages.append({"page": page_num + 1, "text": text})
    doc.close()
    log.info("[PyMuPDF] 解析完成，共 %d 页", len(pages))
    return pages


def load_pdf_paddleocr(path):
    """使用PaddleOCR对PDF进行OCR识别（适用于扫描件/图片型PDF）"""
    from paddleocr import PaddleOCR
    import fitz
    ocr = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False) # 启用方向分类，中文识别，不显示日志
    doc = fitz.open(str(path))
    pages = []
    for page_num, page in enumerate(doc):
        # 将PDF页面转为图片
        pix = page.get_pixmap(dpi=200)  # 把当前页渲染为 200 DPI 的位图
        img_path = str(C.PROJECT_DIR / f"_tmp_ocr_{page_num}.png")
        pix.save(img_path)
        # OCR识别
        result = ocr.ocr(img_path, cls=True)
        text = ""
        if result and result[0]:
            for line in result[0]:
                text += line[1][0] + "\n"
        pages.append({"page": page_num + 1, "text": text})
        os.remove(img_path)  # 清理临时图片
    doc.close()
    log.info("[PaddleOCR] 解析完成，共 %d 页", len(pages))
    return pages


def load_pdf_mineru(path):
    """使用MinerU(magic-pdf)进行文档解析，支持版面分析与结构化输出"""
    from magic_pdf.pipe.UNIPipe import UNIPipe  # type: ignore
    from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter  # type: ignore
    # MinerU配置：使用本地模型或API
    image_dir = str(C.PROJECT_DIR / "_mineru_images") # 设置图片输出目录
    os.makedirs(image_dir, exist_ok=True)
    image_writer = DiskReaderWriter(image_dir) # 创建磁盘写入器，指定图片目录
    pdf_bytes = open(str(path), "rb").read() # 以二进制方式读取 PDF 全部内容
    pipe = UNIPipe(pdf_bytes, {"_pdf_type": "ocr"}, image_writer)
    pipe.pipe_classify() # 执行文档分类
    pipe.pipe_analyze() # 执行版面分析
    pipe.pipe_parse() # 执行文档解析
    content = pipe.pipe_mk_uni_format(drop_mode="none")
    # 提取文本
    pages = []
    for i, page_content in enumerate(content):
        text = ""
        for block in page_content.get("blocks", []):
            if block.get("type") == "text":
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text += span.get("content", "")
                    text += "\n"
        pages.append({"page": i + 1, "text": text})
    log.info("[MinerU] 解析完成，共 %d 页", len(pages))
    return pages


def load_pdf(path, method="auto"):
    """根据method选择解析方法；auto优先PyMuPDF，失败则降级"""
    if method == "paddleocr":
        return load_pdf_paddleocr(path)
    elif method == "mineru":
        return load_pdf_mineru(path)
    else:
        try:
            return load_pdf_pymupdf(path)
        except Exception as e:
            log.warning("[PyMuPDF] 解析失败(%s)，尝试PaddleOCR降级...", e)
            try:
                return load_pdf_paddleocr(path)
            except Exception:
                return load_pdf_mineru(path)


# ======================== 文本分块 ========================
def split_text(text, chunk_size=C.CHUNK_SIZE, overlap=C.CHUNK_OVERLAP):
    """滑动窗口分块，支持中文长文本切分"""
    if not text or not text.strip():
        return []
    text = re.sub(r"\s+", " ", text).strip() # 把连续空白字符替换为单个空格，并去掉首尾空白
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0  # 滑动窗口起点为 0
    while start < len(text):
        end = start + chunk_size # 计算当前块理论终点
        chunk = text[start:end]
        # 尝试在句号/换行处截断，避免切断句子
        if end < len(text):
            for sep in ["。", "！", "？", "\n", "；", " "]:
                pos = chunk.rfind(sep)
                if pos > chunk_size * 0.5:
                    end = start + pos + len(sep)
                    chunk = text[start:end]
                    break
        chunks.append(chunk.strip())
        start = end - overlap
    return [c for c in chunks if c] # 返回所有非空块


# ======================== 中文分词（BM25用） ========================
def tokenize(text):
    """中文分词，优先jieba，降级为字符级"""
    try:
        import jieba
        return [w for w in jieba.lcut(text) if w.strip()]
    except ImportError:
        return [c for c in text if c.strip()]


# ======================== BM25稀疏索引 ========================
def build_bm25_index(chunks):
    """构建BM25索引并持久化到磁盘"""
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        log.warning("[BM25] rank-bm25未安装，跳过稀疏索引构建")
        return None
    tokenized = [tokenize(c) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    with open(C.BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks, "tokenized": tokenized}, f)
    log.info("[BM25] 索引构建完成，%d 个文档，保存至 %s", len(chunks), C.BM25_INDEX_PATH)
    return bm25


# ======================== Milvus向量存储 ========================
def init_milvus():
    """初始化Milvus客户端并创建集合"""
    from pymilvus import MilvusClient, DataType
    client = MilvusClient(C.MILVUS_DB)
    # 如果集合已存在则删除重建
    if client.has_collection(C.COLLECTION_NAME):
        client.drop_collection(C.COLLECTION_NAME)
    # 文档集合schema
    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=C.EMBED_DIM)
    schema.add_field("text", DataType.VARCHAR, max_length=2048)
    schema.add_field("source", DataType.VARCHAR, max_length=256)
    # 创建向量索引参数（pymilvus 3.x API）
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="embedding",
                           index_type="IVF_FLAT",
                           metric_type=C.METRIC_TYPE,  # 距离度量类型
                           params={"nlist": C.IVF_NLIST})
    client.create_collection(C.COLLECTION_NAME, schema=schema, index_params=index_params)
    # 多轮对话历史集合
    if client.has_collection(C.HISTORY_COLLECTION):
        client.drop_collection(C.HISTORY_COLLECTION)
    hist_schema = MilvusClient.create_schema(auto_id=True)
    hist_schema.add_field("id", DataType.INT64, is_primary=True)
    hist_schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=C.EMBED_DIM)
    hist_schema.add_field("session_id", DataType.VARCHAR, max_length=64)
    hist_schema.add_field("role", DataType.VARCHAR, max_length=16)
    hist_schema.add_field("content", DataType.VARCHAR, max_length=2048)
    hist_index = client.prepare_index_params()
    hist_index.add_index(field_name="embedding",
                         index_type="IVF_FLAT",
                         metric_type=C.METRIC_TYPE,
                         params={"nlist": 64})
    client.create_collection(C.HISTORY_COLLECTION, schema=hist_schema, index_params=hist_index)
    log.info("[Milvus] 集合初始化完成: %s, %s", C.COLLECTION_NAME, C.HISTORY_COLLECTION)
    return client


def get_embeddings(texts):
    """调用Ollama获取文本向量"""
    from langchain_ollama import OllamaEmbeddings
    embedder = OllamaEmbeddings(model=C.EMBED_MODEL, base_url=C.OLLAMA_URL)
    return embedder.embed_documents(texts)


def store_vectors(client, chunks, source_name):
    """将文本块向量化并存入Milvus"""
    batch_size = 32
    total = len(chunks)
    for i in range(0, total, batch_size):
        batch = chunks[i:i + batch_size]
        vectors = get_embeddings(batch)
        records = []
        for text, vec in zip(batch, vectors):
            records.append({
                "embedding": vec,
                "text": text[:2048],
                "source": source_name[:256],
            })
        client.insert(C.COLLECTION_NAME, records)
        log.info("[Milvus] 写入进度: %d/%d", min(i + batch_size, total), total)
    client.flush(C.COLLECTION_NAME)
    log.info("[Milvus] 全部 %d 条向量入库完成", total)


# ======================== 主流程 ========================
def main():
    log.info("=" * 60)
    log.info("  离线建库流水线启动")
    log.info("=" * 60)
    # 第1步：解析PDF
    log.info("[第1步] PDF解析 (方法: %s)", C.PARSER_METHOD)
    pages = load_pdf(C.PDF_PATH, C.PARSER_METHOD)
    full_text = "\n".join(p["text"] for p in pages)
    log.info("  全文共 %d 字符", len(full_text))

    # 第2步：文本分块
    log.info("[第2步] 文本分块 (块大小=%d, 重叠=%d)", C.CHUNK_SIZE, C.CHUNK_OVERLAP)
    chunks = split_text(full_text)
    log.info("  共生成 %d 个文本块", len(chunks))

    # 第3步：BM25稀疏索引
    log.info("[第3步] BM25稀疏索引构建")
    build_bm25_index(chunks)

    # 第4步：Milvus初始化
    log.info("[第4步] Milvus向量库初始化")
    client = init_milvus()

    # 第5步：向量化并入库
    log.info("[第5步] 向量化与入库")
    source = C.PDF_PATH.name
    store_vectors(client, chunks, source)

    log.info("=" * 60)
    log.info("  离线建库完成！")
    log.info("  Milvus数据库: %s", C.MILVUS_DB)
    log.info("  BM25索引: %s", C.BM25_INDEX_PATH)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
