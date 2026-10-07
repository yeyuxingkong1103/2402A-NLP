# -*- coding: utf-8 -*-
"""
RAG 共享核心模块 (rag_common.py)
================================
人工智能 NLP-RAG 项目 —— 工单 01~13 共用的轻量 RAG 内核。

包含:
    - PDF 解析 (文本 / 表格 / 图像)
    - 文本分块
    - 向量嵌入 (本地 bge-m3) + 向量检索
    - 全文检索 (BM25 + jieba 分词)
    - 混合检索 (向量 + 全文 加权融合)
    - 三种重排算法 (LLM重排 / TF-IDF重排 / 用户反馈自适应重排)
    - LLM 调用 (DeepSeek)
    - RAG 问答与评估指标 (RAGAS 风格)

依赖: pdfplumber, pypdf, sentence-transformers, jieba, numpy, openai
"""
import os
import re
import json
import time
import math
import hashlib
from collections import Counter, defaultdict

import numpy as np

# 屏蔽 pypdf / pdfminer 的告警刷屏 (部分 PDF 存在错误的交叉引用对象)
import logging
for _n in ("pypdf", "pdfminer", "pdfplumber", "PIL"):
    logging.getLogger(_n).setLevel(logging.ERROR)

# --------------------------------------------------------------------------
# 全局配置
# --------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))

# 附件目录 (工单原始附件, 由 01 工单目录统一存放)
ATTACH_DIR = os.environ.get(
    "RAG_ATTACH_DIR",
    r"C:\Users\Lenovo\Desktop\于子桐\工单(1)\附件",
)

# 默认嵌入模型: BAAI/bge-base-zh-v1.5 (768维, 中文优化, CPU 上约 12 块/秒)
# 备选: D:\专高三资料\bge-m3 (1024维, 多语言更强但 CPU 上慢约 12 倍)
EMBED_MODEL_PATH = os.environ.get(
    "RAG_EMBED_MODEL",
    r"D:\专高三资料\bge-base-zh-v1.5",
)
RERANKER_MODEL_PATH = os.environ.get(
    "RAG_RERANKER_MODEL",
    r"D:\专高三资料\bge-reranker-base",
)

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-chat")
# 密钥优先取环境变量 DEEPSEEK_API_KEY, 其次 LLM_API_KEY; 都没有再读本目录 .env
LLM_API_KEY = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("LLM_API_KEY", "")


def _load_api_key():
    """读取 LLM API Key: 环境变量 DEEPSEEK_API_KEY / LLM_API_KEY > 本目录 .env"""
    global LLM_API_KEY
    if LLM_API_KEY:
        return LLM_API_KEY
    p = os.path.join(ROOT, ".env")
    if os.path.exists(p):
        for line in open(p, encoding="utf-8", errors="replace"):
            line = line.strip()
            if line.startswith("LLM_API_KEY") and "=" in line:
                LLM_API_KEY = line.split("=", 1)[1].strip().strip('"').strip("'")
                return LLM_API_KEY
    raise RuntimeError("未找到 API Key: 请设置环境变量 DEEPSEEK_API_KEY, "
                       "或在 .env 中配置 LLM_API_KEY")


# --------------------------------------------------------------------------
# 1. PDF 解析
# --------------------------------------------------------------------------
def extract_pdf_pages(pdf_path, max_pages=None, verbose=False, engine="pypdf"):
    """
    解析 PDF 文本, 返回 [{'page': 页号, 'text': 文本}, ...]  (页码从 1 开始)

    engine="pypdf"       : 原生文本层快速抽取 (~0.06 s/页), 默认
    engine="pdfplumber"  : 版式还原更好但慢 30 倍, 按需使用

    工单编号: 人工智能NLP-RAG-基于PDF文档的问答系统
    """
    if engine == "pdfplumber":
        import pdfplumber
        pages = []
        with pdfplumber.open(pdf_path) as pdf:
            total = len(pdf.pages) if max_pages is None else min(max_pages, len(pdf.pages))
            for i in range(total):
                try:
                    txt = pdf.pages[i].extract_text() or ""
                except Exception as e:                   # 容错: 单页失败不影响整体
                    if verbose:
                        print(f"  [warn] 第{i+1}页解析失败: {e}")
                    txt = ""
                txt = clean_text(txt)
                if txt:
                    pages.append({"page": i + 1, "text": txt})
        return pages

    from pypdf import PdfReader
    reader = PdfReader(pdf_path)
    total = len(reader.pages) if max_pages is None else min(max_pages, len(reader.pages))
    pages = []
    for i in range(total):
        try:
            txt = reader.pages[i].extract_text() or ""
        except Exception as e:                           # 容错: 单页失败不影响整体
            if verbose:
                print(f"  [warn] 第{i+1}页解析失败: {e}")
            txt = ""
        txt = clean_text(txt)
        if txt:
            pages.append({"page": i + 1, "text": txt})
        if verbose and (i + 1) % 100 == 0:
            print(f"  已解析 {i+1}/{total} 页")
    return pages


def clean_text(txt):
    """通用文本清洗 (工单编号: 人工智能NLP-RAG-基于PDF文档的问答系统优化)"""
    if not txt:
        return ""
    txt = txt.replace("\u3000", " ").replace("\xa0", " ")
    txt = re.sub(r"-\n", "", txt)                 # 断词连字符
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    # 去掉页眉页脚式的水印重复行
    txt = re.sub(r"^.*八维文化与产业研究院.*$", "", txt, flags=re.M)
    return txt.strip()


CACHE_DIR = os.path.join(ROOT, "cache")


def _cache_file(pdf_path, tag):
    """PDF 解析结果的缓存文件路径 (pdfplumber 很慢, 同一份 PDF 只解析一次)"""
    h = hashlib.md5(os.path.abspath(pdf_path).encode("utf-8")).hexdigest()[:8]
    name = os.path.splitext(os.path.basename(pdf_path))[0]
    return os.path.join(CACHE_DIR, f"{name}.{tag}.{h}.json")


def cached_parse(pdf_path, tag, fn, max_pages=None):
    """
    通用解析缓存: 首次调用执行 fn() 并把结果落到 cache/*.json, 之后直接读缓存。
    表格/图像解析动辄十几分钟, 缓存后可被多个工单复用。
    """
    cf = _cache_file(pdf_path, tag)
    if max_pages is None and os.path.exists(cf):
        try:
            return load_json(cf)
        except Exception:
            pass
    out = fn()
    if max_pages is None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        save_json(cf, out)
    return out


def extract_pdf_tables(pdf_path, max_pages=None, use_cache=True):
    """
    解析 PDF 表格, 返回 [{'page':页号, 'table':[[...]], 'text':线性化文本}, ...]
    (工单编号: 人工智能NLP-RAG-PDF文档的表格解析及检索优化)
    """
    import pdfplumber

    def _run():
        out = []
        with pdfplumber.open(pdf_path) as pdf:
            total = (len(pdf.pages) if max_pages is None
                     else min(max_pages, len(pdf.pages)))
            for i in range(total):
                try:
                    tables = pdf.pages[i].extract_tables() or []
                except Exception:
                    continue
                for t in tables:
                    t = [[(c or "").strip() for c in row] for row in t if row]
                    if len(t) < 2:
                        continue
                    out.append({
                        "page": i + 1,
                        "table": t,
                        "text": table_to_text(t),
                    })
        return out

    return cached_parse(pdf_path, "tables", _run, max_pages) if use_cache else _run()


def table_to_text(table):
    """
    把表格线性化为可检索文本, 是表格解析检索优化的关键步骤。

    分两种版式处理:
      1) 键值型(2列)表格, 如"发行股数 | 1,670万股" -> 每行输出 "字段: 值"
      2) 矩阵型(多列)表格 -> 用首行作表头, 每行输出 "表头: 值 | 表头: 值"

    这样字段名(发行股数/持股比例/关联方)与数值在同一个语义单元里,
    向量检索与 BM25 都能同时命中, 解决"数字对了但检索不到"的问题。
    (工单编号: 人工智能NLP-RAG-PDF文档的表格解析及检索优化)
    """
    if not table:
        return ""
    rows = [[(c or "").strip() for c in r] for r in table if r]
    rows = [r for r in rows if any(r)]
    if not rows:
        return ""

    ncol = max(len(r) for r in rows)
    lines = []

    if ncol <= 2:
        # 键值型: 全部行都是 "字段: 值"
        for r in rows:
            if len(r) >= 2 and r[0] and r[1]:
                lines.append(f"{r[0]}: {r[1]}")
            elif len(r) == 1 and r[0]:
                lines.append(r[0])
        return "\n".join(lines)

    # 矩阵型: 判断首行是否为表头 (首行单元格互不相同且不含长数字)
    header = rows[0]
    is_header = (len(header) >= ncol - 1 and
                 all(header) and
                 not any(re.search(r"\d[\d,]{4,}", h) for h in header))

    body = rows[1:] if is_header else rows
    if is_header:
        lines.append(" | ".join(h for h in header if h))
    for r in body:
        parts = []
        for j, cell in enumerate(r):
            if not cell:
                continue
            h = (header[j] if (is_header and j < len(header) and header[j])
                 else f"列{j+1}")
            parts.append(f"{h}: {cell}" if is_header else cell)
        if parts:
            lines.append(" | ".join(parts))
    return "\n".join(lines).strip()


def extract_pdf_chart_regions(pdf_path, max_pages=None, min_w=80, min_h=50,
                              use_cache=True):
    """
    找出 PDF 中的"图像区域"(饼图 / 柱状图 / 组织结构图等), 供 OCR 解析。

    坑1: 不能直接用 page.images —— 招股说明书2 每页铺了 15 个平铺水印小图
         (350 页共 5920 个), 真图表混在里面。
         解决办法: 同一页里"像素尺寸完全相同且重复 >=3 次"的图就是水印, 剔除。
    坑2: 真图表常由 2 张位图拼成(如左饼图 + 右柱状图), 所以按页合并包围盒。

    返回 [{'page','bbox','width','height','text'}, ...]
    (工单编号: 人工智能NLP-RAG-图像内容解析及检索优化)
    """
    import pdfplumber

    def _run():
        out = []
        with pdfplumber.open(pdf_path) as pdf:
            total = (len(pdf.pages) if max_pages is None
                     else min(max_pages, len(pdf.pages)))
            for i in range(total):
                pg = pdf.pages[i]
                imgs = pg.images or []
                cnt = Counter((im.get("width"), im.get("height")) for im in imgs)
                box = None
                for im in imgs:
                    if cnt[(im.get("width"), im.get("height"))] >= 3:
                        continue                       # 水印, 跳过
                    try:
                        iw = abs(im["x1"] - im["x0"])
                        ih = abs(im["bottom"] - im["top"])
                    except Exception:
                        continue
                    if iw < min_w or ih < min_h:
                        continue
                    b = (im["x0"], im["top"], im["x1"], im["bottom"])
                    box = b if box is None else (
                        min(box[0], b[0]), min(box[1], b[1]),
                        max(box[2], b[2]), max(box[3], b[3]))
                if box is None:
                    continue
                bbox = (max(0, box[0] - 2), max(0, box[1] - 2),
                        min(pg.width, box[2] + 2), min(pg.height, box[3] + 2))
                try:
                    txt = clean_text(pg.crop(bbox).extract_text() or "")
                except Exception:
                    txt = ""
                out.append({
                    "page": i + 1,
                    "bbox": [round(v, 1) for v in bbox],
                    "width": round(bbox[2] - bbox[0], 1),
                    "height": round(bbox[3] - bbox[1], 1),
                    "text": txt,
                })
        return out

    return cached_parse(pdf_path, "charts", _run, max_pages) if use_cache else _run()



_OCR = None


def get_ocr():
    """懒加载 RapidOCR (PP-OCRv3, CPU, 约 13MB 模型, 随 pip 包自带)"""
    global _OCR
    if _OCR is None:
        from rapidocr_onnxruntime import RapidOCR
        _OCR = RapidOCR()
    return _OCR


def ocr_image(img):
    """
    对 PIL 图片做 OCR, 返回 [{'box','text','score'}, ...]

    坑: RapidOCR 直接传中文路径会读不到文件(返回 None),
       必须传 numpy 数组。
    (工单编号: 人工智能NLP-RAG-图像内容解析及检索优化)
    """
    import numpy as np
    if not isinstance(img, np.ndarray):
        img = np.array(img.convert("RGB"))
    res, _ = get_ocr()(img)
    out = []
    for item in (res or []):
        try:
            box, text, score = item[0], item[1], item[2]
            ys = [p[1] for p in box]
            xs = [p[0] for p in box]
            out.append({"box": box, "text": str(text).strip(),
                        "score": float(score),
                        "y": sum(ys) / 4.0, "x": min(xs)})
        except Exception:
            continue
    return out


def ocr_chart_caption(img, max_lines=40):
    """
    把图表 OCR 结果按"行"聚合, 还原成可检索的图注文本。

    图表里的文字(图例、坐标轴、数据标签)本身没有阅读顺序,
    这里按 y 坐标聚成行、行内按 x 排序, 得到近似 "标签 值" 的文本,
    再交给嵌入模型/BM25, 使"增长率最快的行业"这类问题能命中数据。
    (工单编号: 人工智能NLP-RAG-图像内容解析及检索优化)
    """
    items = [it for it in ocr_image(img) if it["score"] > 0.5 and it["text"]]
    if not items:
        return ""
    items.sort(key=lambda d: d["y"])
    lines, cur, cur_y = [], [], None
    for it in items:
        if cur_y is None or abs(it["y"] - cur_y) <= 12:
            cur.append(it)
            cur_y = it["y"] if cur_y is None else (cur_y + it["y"]) / 2.0
        else:
            lines.append(cur)
            cur, cur_y = [it], it["y"]
    if cur:
        lines.append(cur)
    out = []
    for ln in lines[:max_lines]:
        ln.sort(key=lambda d: d["x"])
        txt = " ".join(d["text"] for d in ln)
        if txt.strip():
            out.append(txt.strip())
    return "\n".join(out)


def render_region_image(pdf_path, page_no, bbox, out_path, dpi=150):
    """
    把页面上指定区域(图表)渲染成 PNG, 供 CLIP 等多模态模型编码。
    (工单编号: 人工智能NLP-RAG-图像内容解析及检索优化)
    """
    import pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        if page_no > len(pdf.pages):
            return None
        pg = pdf.pages[page_no - 1]
        im = pg.crop(tuple(bbox)).to_image(resolution=dpi)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        im.save(out_path)
    return out_path


def render_page_image(pdf_path, page_no, out_path, dpi=150):
    """把整页渲染成图片 (用于页内嵌图/矢量图, 如组织结构图)"""
    import pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        if page_no > len(pdf.pages):
            return None
        im = pdf.pages[page_no - 1].to_image(resolution=dpi)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        im.save(out_path)
    return out_path


# --------------------------------------------------------------------------
# 2. 文本分块
# --------------------------------------------------------------------------
def chunk_text(text, chunk_size=500, overlap=50, page=None, source=None,
               chunk_type="text"):
    """
    按字符数滑窗分块, 优先在句号/换行处断开, 避免切断句子。
    (工单编号: 人工智能NLP-RAG-基于PDF文档的问答系统优化)
    """
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            # 向后找一个自然断点
            window = text[start:end]
            cut = max(window.rfind("。"), window.rfind("\n"), window.rfind("；"))
            if cut > chunk_size * 0.5:
                end = start + cut + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append({
                "text": piece,
                "page": page,
                "source": source,
                "type": chunk_type,
            })
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def parse_pdf_to_chunks(pdf_path, chunk_size=500, overlap=50,
                        with_tables=True, with_images=False, image_dir=None):
    """PDF -> 分块列表 (文本块 + 表格块 [+ 图像块])"""
    src = os.path.basename(pdf_path)
    pages = extract_pdf_pages(pdf_path)
    chunks = []
    for p in pages:
        chunks.extend(chunk_text(p["text"], chunk_size, overlap,
                                 page=p["page"], source=src))
    if with_tables:
        for t in extract_pdf_tables(pdf_path):
            chunks.extend(chunk_text(t["text"], chunk_size, overlap,
                                     page=t["page"], source=src,
                                     chunk_type="table"))
    if with_images and image_dir:
        for im in extract_pdf_images(pdf_path, image_dir):
            chunks.append({"text": "", "page": im["page"], "source": src,
                           "type": "image", "image_path": im["path"]})
    for i, c in enumerate(chunks):
        c["id"] = i
    return chunks


# --------------------------------------------------------------------------
# 3. 向量嵌入
# --------------------------------------------------------------------------
class Embedder:
    """本地 bge 系列嵌入器 (CPU), 进程内共享一份模型权重。"""

    _shared = {}

    def __init__(self, model_path=EMBED_MODEL_PATH, batch_size=32):
        self.model_path = model_path
        self.batch_size = batch_size
        # 限制 torch 线程数。默认 torch 会占满所有核心, 建索引时整台机器会卡住,
        # 用 RAG_THREADS 环境变量调整 (默认 8, 兼顾速度与"机器还能用")。
        try:
            import torch
            n = int(os.environ.get("RAG_THREADS", "8"))
            if n > 0:
                torch.set_num_threads(n)
        except Exception:
            pass
        if model_path not in Embedder._shared:
            from sentence_transformers import SentenceTransformer
            Embedder._shared[model_path] = SentenceTransformer(model_path,
                                                               device="cpu")
        self.model = Embedder._shared[model_path]
        self.dim = self.model.get_embedding_dimension()

    def encode(self, texts, show_progress=False):
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return self.model.encode(texts, batch_size=self.batch_size,
                                 normalize_embeddings=True,
                                 show_progress_bar=show_progress,
                                 convert_to_numpy=True)


# --------------------------------------------------------------------------
# 4. 向量库 (numpy 内存实现, 落盘为 .npy + .jsonl)
# --------------------------------------------------------------------------
class VectorStore:
    """极简向量库: 余弦相似度检索, 支持持久化。"""

    def __init__(self, dim=None):
        self.dim = dim
        self.chunks = []
        self.emb = None

    def add(self, chunks, embeddings):
        self.chunks = list(chunks)
        self.emb = np.asarray(embeddings, dtype=np.float32)
        self.dim = self.emb.shape[1]
        if self.emb.shape[0]:
            norms = np.linalg.norm(self.emb, axis=1, keepdims=True)
            norms[norms == 0] = 1e-9
            self.emb = self.emb / norms

    def save(self, index_dir):
        os.makedirs(index_dir, exist_ok=True)
        with open(os.path.join(index_dir, "chunks.jsonl"), "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        np.save(os.path.join(index_dir, "emb.npy"), self.emb)
        with open(os.path.join(index_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({"count": len(self.chunks), "dim": self.dim}, f,
                      ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, index_dir):
        vs = cls()
        with open(os.path.join(index_dir, "chunks.jsonl"), encoding="utf-8") as f:
            vs.chunks = [json.loads(l) for l in f if l.strip()]
        vs.emb = np.load(os.path.join(index_dir, "emb.npy"))
        vs.dim = vs.emb.shape[1] if vs.emb.size else None
        return vs

    @staticmethod
    def exists(index_dir):
        return os.path.exists(os.path.join(index_dir, "chunks.jsonl")) and \
               os.path.exists(os.path.join(index_dir, "emb.npy"))

    def search(self, query_vec, top_k=10):
        """返回 [(index, score), ...]"""
        if self.emb is None or len(self.chunks) == 0:
            return []
        q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        q = q / (np.linalg.norm(q) + 1e-9)
        sims = self.emb @ q                      # 已归一化 => 余弦相似度
        top_k = min(top_k, len(sims))
        idx = np.argpartition(-sims, top_k - 1)[:top_k]
        idx = idx[np.argsort(-sims[idx])]
        return [(int(i), float(sims[i])) for i in idx]


# --------------------------------------------------------------------------
# 5. 全文检索 (BM25 + jieba)
# --------------------------------------------------------------------------
class BM25:
    """
    倒排索引 + BM25 打分, 支持布尔查询(空格=AND)、短语匹配("...")、模糊匹配(~词)。
    (工单编号: 人工智能NLP-RAG-混合检索任务)
    """

    def __init__(self, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = []            # 分词结果
        self.df = Counter()       # 文档频率
        self.avg_len = 0.0
        self.N = 0

    @staticmethod
    def tokenize(text):
        import jieba
        toks = [t.strip().lower() for t in jieba.lcut(text or "") if t.strip()]
        return [t for t in toks if not re.fullmatch(r"[\s\W_]+", t)]

    def build(self, chunks):
        self.docs = []
        self.df = Counter()
        for c in chunks:
            text = c.get("text", "")
            if c.get("type") == "image" and c.get("caption"):
                text = text + " " + c["caption"]
            toks = self.tokenize(text)
            self.docs.append(toks)
            for t in set(toks):
                self.df[t] += 1
        self.N = len(self.docs)
        self.avg_len = (sum(len(d) for d in self.docs) / self.N) if self.N else 0.0

    def _score(self, query_tokens, i):
        doc = self.docs[i]
        dl = len(doc)
        if dl == 0:
            return 0.0
        tf = Counter(doc)
        s = 0.0
        for qt in query_tokens:
            f = tf.get(qt, 0)
            if f == 0:
                continue
            idf = math.log(1 + (self.N - self.df[qt] + 0.5) / (self.df[qt] + 0.5))
            s += idf * (f * (self.k1 + 1)) / (
                f + self.k1 * (1 - self.b + self.b * dl / (self.avg_len or 1)))
        return s

    def search(self, query, top_k=10, fuzzy=True):
        """返回 [(index, score), ...]"""
        if self.N == 0:
            return []
        phrase = None
        q = query
        m = re.search(r'"([^"]+)"', q)
        if m:                                     # 短语匹配
            phrase = m.group(1)
            q = q.replace(m.group(0), " ")
        must = [t for t in self.tokenize(q)]
        if fuzzy:                                 # 模糊匹配: 词+~ 前缀召回
            q2 = re.sub(r"(\S+)~", r"\1", query)
            must = self.tokenize(q2)
        if not must and not phrase:
            return []
        scores = np.zeros(self.N, dtype=np.float32)
        for i in range(self.N):
            s = self._score(must, i)
            if phrase:                            # 短语必须完整出现
                s = s * 2.0 if phrase in "".join(self.docs[i]) or \
                    phrase.lower() in " ".join(self.docs[i]) else 0.0
            scores[i] = s
        nz = np.nonzero(scores)[0]
        if len(nz) == 0:
            return []
        top_k = min(top_k, len(nz))
        idx = nz[np.argsort(-scores[nz])[:top_k]]
        return [(int(i), float(scores[i])) for i in idx]


# --------------------------------------------------------------------------
# 6. 重排算法 (三种)
# --------------------------------------------------------------------------
def rerank_tfidf(query, candidates):
    """
    重排算法 1: 基于 TF-IDF 的重排器。
    对查询与候选文档做 TF-IDF 向量夹角余弦, 兼顾词频与逆文档频率。
    (工单编号: 人工智能NLP-RAG-混合检索任务)
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    if not candidates:
        return []
    texts = [c["chunk"].get("text", "") or c["chunk"].get("caption", "")
             for c in candidates]
    try:
        vec = TfidfVectorizer(tokenizer=BM25.tokenize, token_pattern=None,
                              lowercase=False)
        mat = vec.fit_transform(texts + [query])
        qv = mat[-1]
        sims = (mat[:-1] @ qv.T).toarray().reshape(-1)
    except Exception:
        sims = np.zeros(len(texts))
    out = []
    for c, s in zip(candidates, sims):
        d = dict(c)
        d["rerank_score"] = float(s)
        out.append(d)
    out.sort(key=lambda x: -x["rerank_score"])
    return add_rank(out)


def rerank_llm(query, candidates, top_n=5):
    """
    重排算法 2: 基于 LLM 的重排器 (DeepSeek 逐条打分 0~10)。
    (工单编号: 人工智能NLP-RAG-混合检索任务)
    """
    if not candidates:
        return []
    for c in candidates:
        chunk = c["chunk"]
        text = (chunk.get("text") or chunk.get("caption") or "")[:400]
        prompt = (
            "你是检索结果重排器。判断下面这段文档能否回答用户问题。\n"
            f"用户问题: {query}\n"
            f"文档片段: {text}\n"
            "只输出一个 0-10 的整数相关性分数(10=完全能回答, 0=完全无关), 不要解释。"
        )
        try:
            s = llm(prompt, max_tokens=8, temperature=0.0)
            m = re.search(r"\d+", s)
            score = min(10, max(0, int(m.group()))) if m else 0
        except Exception:
            score = 0
        c["rerank_score"] = score / 10.0
    out = sorted(candidates, key=lambda x: -x["rerank_score"])
    return add_rank(out)


def rerank_feedback(query, candidates, feedback_path=None):
    """
    重排算法 3: 基于用户反馈的自适应重排器。
    用历史反馈学到的"优秀词/劣质词"权重调整打分, 无反馈时退化为 BM25 分。
    (工单编号: 人工智能NLP-RAG-混合检索任务)
    """
    weights = load_feedback_weights(feedback_path)
    out = []
    for c in candidates:
        text = c["chunk"].get("text", "") or c["chunk"].get("caption", "")
        toks = set(BM25.tokenize(text))
        toks |= set(BM25.tokenize(query))
        adj = sum(weights.get(t, 0.0) for t in toks)
        d = dict(c)
        d["rerank_score"] = c.get("score", 0.0) * (1.0 + 0.3 * adj)
        out.append(d)
    out.sort(key=lambda x: -x["rerank_score"])
    return add_rank(out)


def rerank_crossencoder(query, candidates, model_path=None, top_n=None):
    """
    重排算法 4: 基于交叉编码器 (bge-reranker-base) 的重排器。
    与双塔向量检索不同, cross-encoder 对 (query, doc) 联合编码, 精度更高。
    (工单编号: 人工智能NLP-RAG-混合检索任务)
    """
    if not candidates:
        return []
    from sentence_transformers import CrossEncoder
    model_path = model_path or RERANKER_MODEL_PATH
    key = "ce:" + model_path
    if key not in Embedder._shared:
        Embedder._shared[key] = CrossEncoder(model_path, device="cpu",
                                             max_length=512)
    ce = Embedder._shared[key]
    pairs = [[query, (c["chunk"].get("text") or c["chunk"].get("caption", ""))[:512]]
             for c in candidates]
    try:
        scores = ce.predict(pairs, batch_size=16, show_progress_bar=False)
    except Exception:
        scores = [0.0] * len(candidates)
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    out = sorted(candidates, key=lambda x: -x["rerank_score"])
    return add_rank(out)


def add_rank(cands):
    for i, c in enumerate(cands):
        c["rank"] = i + 1
    return cands


def load_feedback_weights(feedback_path=None):
    """读取用户反馈权重表 (词 -> 权重)"""
    feedback_path = feedback_path or os.path.join(ROOT, "feedback.json")
    if os.path.exists(feedback_path):
        try:
            return json.load(open(feedback_path, encoding="utf-8")).get("weights", {})
        except Exception:
            return {}
    return {}


def update_feedback(query, good_doc_ids, bad_doc_ids, index_dir,
                    feedback_path=None):
    """记录用户反馈: 命中的文档加分, 未命中的减分, 落到 feedback.json"""
    feedback_path = feedback_path or os.path.join(ROOT, "feedback.json")
    vs = VectorStore.load(index_dir)
    data = {"weights": {}, "log": []}
    if os.path.exists(feedback_path):
        try:
            data = json.load(open(feedback_path, encoding="utf-8"))
        except Exception:
            pass
    w = data.setdefault("weights", {})
    for did in good_doc_ids:
        if 0 <= did < len(vs.chunks):
            for t in set(BM25.tokenize(vs.chunks[did].get("text", ""))):
                w[t] = w.get(t, 0.0) + 0.1
    for did in bad_doc_ids:
        if 0 <= did < len(vs.chunks):
            for t in set(BM25.tokenize(vs.chunks[did].get("text", ""))):
                w[t] = w.get(t, 0.0) - 0.1
    data.setdefault("log", []).append(
        {"query": query, "good": list(good_doc_ids), "bad": list(bad_doc_ids)})
    json.dump(data, open(feedback_path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    return w


# --------------------------------------------------------------------------
# 7. 检索器 (向量 / 全文 / 混合)
# --------------------------------------------------------------------------
class Retriever:
    """
    统一检索入口, 支持三种策略:
        vector   向量检索 (召回 + 重排)
        fulltext 全文检索 (BM25 倒排索引)
        hybrid   混合检索 (加权融合)
    (工单编号: 人工智能NLP-RAG-混合检索任务)
    """

    def __init__(self, index_dir, load_embedder=True):
        self.index_dir = index_dir
        self.vs = VectorStore.load(index_dir)
        self.bm25 = BM25()
        self.bm25.build(self.vs.chunks)
        self.embedder = Embedder() if load_embedder else None

    def search(self, query, top_k=5, mode="vector", candidate_k=20,
               weight_vector=0.5, weight_fulltext=0.5, rerank=None):
        cands = []
        if mode in ("vector", "hybrid") and self.embedder is not None:
            qv = self.embedder.encode([query])[0]
            for i, s in self.vs.search(qv, candidate_k):
                cands.append({"idx": i, "score": s, "vscore": s, "bscore": 0.0})
        if mode in ("fulltext", "hybrid"):
            for i, s in self.bm25.search(query, candidate_k):
                hit = next((c for c in cands if c["idx"] == i), None)
                if hit:
                    hit["bscore"] = s
                else:
                    cands.append({"idx": i, "score": s, "vscore": 0.0, "bscore": s})
        if not cands:
            return []

        # 分数归一化 + 加权融合
        vmax = max([c["vscore"] for c in cands] + [1e-9])
        bmax = max([c["bscore"] for c in cands] + [1e-9])
        for c in cands:
            if mode == "vector":
                c["score"] = c["vscore"]
            elif mode == "fulltext":
                c["score"] = c["bscore"]
            else:
                c["score"] = (weight_vector * c["vscore"] / vmax +
                              weight_fulltext * c["bscore"] / bmax)
        cands.sort(key=lambda x: -x["score"])

        results = [{"idx": c["idx"], "score": c["score"], "vscore": c["vscore"],
                    "bscore": c["bscore"], "chunk": self.vs.chunks[c["idx"]]}
                   for c in cands]
        results = add_rank(results)

        # 重排
        if rerank:
            fn = {"tfidf": rerank_tfidf, "llm": rerank_llm,
                  "feedback": rerank_feedback,
                  "cross-encoder": rerank_crossencoder,
                  "ce": rerank_crossencoder}.get(rerank)
            if fn:
                results = fn(query, results)
        return results[:top_k]


# --------------------------------------------------------------------------
# 8. LLM 调用
# --------------------------------------------------------------------------
_LLM_CLIENT = None


def get_client():
    global _LLM_CLIENT
    if _LLM_CLIENT is None:
        from openai import OpenAI
        _LLM_CLIENT = OpenAI(api_key=_load_api_key(), base_url=LLM_BASE_URL)
    return _LLM_CLIENT


def llm(prompt, system=None, max_tokens=1024, temperature=0.3, retries=3):
    """调用 DeepSeek 生成文本 (带重试)"""
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    last = None
    for a in range(retries):
        try:
            r = get_client().chat.completions.create(
                model=LLM_MODEL, messages=msgs,
                max_tokens=max_tokens, temperature=temperature)
            return r.choices[0].message.content.strip()
        except Exception as e:
            last = e
            time.sleep(1.5 * (a + 1))
    raise RuntimeError(f"LLM 调用失败: {last}")


# --------------------------------------------------------------------------
# 9. RAG 问答
# --------------------------------------------------------------------------
ANSWER_SYSTEM = (
    "你是金融招股说明书问答助手。请严格依据【参考资料】回答问题; "
    "资料中没有的信息不要编造, 回答'根据提供的资料无法确定'。"
    "回答要简洁准确, 涉及数字/比例时必须完整列出。"
)


def build_context(results, max_chars=3000):
    """把检索结果拼成上下文 (工单编号: 人工智能NLP-RAG-基于PDF文档的问答系统)"""
    parts, total = [], 0
    for i, r in enumerate(results):
        c = r["chunk"]
        txt = c.get("text") or c.get("caption") or ""
        if not txt:
            continue
        seg = f"[资料{i+1} | 来源:{c.get('source')} 第{c.get('page')}页]\n{txt}"
        if total + len(seg) > max_chars:
            break
        parts.append(seg)
        total += len(seg)
    return "\n\n".join(parts)


def rag_answer(question, retriever, top_k=5, mode="vector", rerank=None,
               temperature=0.2, **kw):
    """
    完整 RAG 流程: 检索 -> 组装上下文 -> LLM 生成, 返回答案与检索结果、耗时。
    (工单编号: 人工智能NLP-RAG-基于PDF文档的问答系统)
    """
    t0 = time.time()
    results = retriever.search(question, top_k=top_k, mode=mode,
                               rerank=rerank, **kw)
    t_ret = time.time() - t0
    ctx = build_context(results)
    prompt = f"【参考资料】\n{ctx}\n\n【问题】{question}\n\n请依据参考资料作答:"
    t1 = time.time()
    answer = llm(prompt, system=ANSWER_SYSTEM, temperature=temperature)
    t_gen = time.time() - t1
    return {
        "question": question,
        "answer": answer,
        "contexts": [r["chunk"].get("text", "") or r["chunk"].get("caption", "")
                     for r in results],
        "retrieved": [{"id": r["chunk"].get("id"), "score": r["score"],
                       "page": r["chunk"].get("page"),
                       "source": r["chunk"].get("source"),
                       "text": (r["chunk"].get("text") or "")[:200]}
                      for r in results],
        "time_retrieval": round(t_ret, 3),
        "time_generation": round(t_gen, 3),
        "time_total": round(t_ret + t_gen, 3),
    }


# --------------------------------------------------------------------------
# 10. 评估 (RAGAS 风格指标)
# --------------------------------------------------------------------------
def eval_faithfulness(answer, contexts):
    """
    忠实度: 答案中的句子有多少能被检索上下文支持 (LLM 判定)。
    (工单编号: 人工智能NLP-RAG-功能测试及评估)
    """
    if not answer or not contexts:
        return 0.0
    ctx = "\n".join(contexts)[:3000]
    prompt = (
        "判断【答案】中的每个陈述是否都能从【上下文】中得到支持。\n"
        f"【上下文】\n{ctx}\n\n【答案】\n{answer}\n\n"
        "只输出被支持的陈述比例, 一个 0 到 1 之间的小数, 不要解释。"
    )
    try:
        s = llm(prompt, max_tokens=10, temperature=0.0)
        m = re.search(r"\d+(\.\d+)?", s)
        v = float(m.group()) if m else 0.0
        return min(1.0, v)
    except Exception:
        return 0.0


def eval_answer_relevancy(question, answer):
    """答案相关性: 答案是否切题 (LLM 打分 0~1)"""
    if not answer:
        return 0.0
    prompt = (f"【问题】{question}\n【答案】{answer}\n"
              "答案与问题的相关程度是多少? 只输出 0 到 1 的小数, 不要解释。")
    try:
        s = llm(prompt, max_tokens=10, temperature=0.0)
        m = re.search(r"\d+(\.\d+)?", s)
        return min(1.0, float(m.group())) if m else 0.0
    except Exception:
        return 0.0


def eval_context_precision(question, contexts, ground_truth=None):
    """
    上下文精度: 检索到的上下文中, 与问题相关的比例。
    (工单编号: 人工智能NLP-RAG-Graph RAG 优化任务)
    """
    if not contexts:
        return 0.0
    try:
        return rerank_tfidf(question, contexts) if False else _ctx_precision_llm(
            question, contexts)
    except Exception:
        return 0.0


def _ctx_precision_llm(question, contexts):
    good = 0
    for c in contexts:
        prompt = (f"【问题】{question}\n【文档片段】{c[:400]}\n"
                  "该片段对回答这个问题有用吗? 只回答 是 或 否。")
        try:
            s = llm(prompt, max_tokens=4, temperature=0.0)
            if "是" in s or "yes" in s.lower():
                good += 1
        except Exception:
            pass
    return good / len(contexts)


def eval_context_recall(question, contexts, ground_truth):
    """
    上下文召回: 参考答案中的要点有多少能从检索上下文里找到 (LLM 判定)。
    (工单编号: 人工智能NLP-RAG-Graph RAG 优化任务)
    """
    if not contexts or not ground_truth:
        return 0.0
    ctx = "\n".join(contexts)[:4000]
    prompt = (
        "下面是【参考答案】和【检索上下文】。\n"
        f"【参考答案】\n{ground_truth[:1200]}\n\n【检索上下文】\n{ctx}\n\n"
        "参考答案中的信息有多少比例可以从检索上下文中找到? "
        "只输出 0 到 1 之间的小数, 不要解释。"
    )
    try:
        s = llm(prompt, max_tokens=10, temperature=0.0)
        m = re.search(r"\d+(\.\d+)?", s)
        return min(1.0, float(m.group())) if m else 0.0
    except Exception:
        return 0.0


def eval_answer_correctness(answer, ground_truth):
    """答案正确性: 与参考答案的一致程度 (LLM 打分 0~1)"""
    if not answer or not ground_truth:
        return 0.0
    prompt = (f"【参考答案】{ground_truth[:1200]}\n【待评估答案】{answer[:1200]}\n"
              "待评估答案与参考答案在事实层面的一致程度是多少? "
              "只输出 0 到 1 的小数, 不要解释。")
    try:
        s = llm(prompt, max_tokens=10, temperature=0.0)
        m = re.search(r"\d+(\.\d+)?", s)
        return min(1.0, float(m.group())) if m else 0.0
    except Exception:
        return 0.0


def keyword_hit(answer, keywords):
    """关键词命中率: 答案里包含多少参考答案关键词 (轻量客观指标)"""
    if not keywords:
        return 0.0
    a = re.sub(r"[\s,，、%％]", "", answer or "")
    hit = sum(1 for k in keywords if re.sub(r"[\s,，、%％]", "", k) in a)
    return hit / len(keywords)


def ragas_evaluate(question, answer, contexts, ground_truth=None):
    """RAGAS 风格整体评估 (工单编号: 人工智能NLP-RAG-功能测试及评估)"""
    res = {
        "faithfulness": eval_faithfulness(answer, contexts),
        "answer_relevancy": eval_answer_relevancy(question, answer),
    }
    if ground_truth:
        res["context_precision"] = eval_context_precision(question, contexts)
        res["context_recall"] = eval_context_recall(question, contexts,
                                                    ground_truth)
        res["answer_correctness"] = eval_answer_correctness(answer, ground_truth)
    return {k: round(v, 3) for k, v in res.items()}


# --------------------------------------------------------------------------
# 11. 索引构建入口
# --------------------------------------------------------------------------
def build_index(pdf_paths, index_dir, chunk_size=500, overlap=50,
                with_tables=True, with_images=False, verbose=True):
    """
    构建向量索引: PDF -> 分块 -> 嵌入 -> 落盘。
    (工单编号: 人工智能NLP-RAG-基于PDF文档的问答系统)
    """
    all_chunks = []
    for p in pdf_paths:
        if verbose:
            print(f"  解析 {os.path.basename(p)} ...")
        cs = parse_pdf_to_chunks(
            p, chunk_size=chunk_size, overlap=overlap,
            with_tables=with_tables, with_images=with_images,
            image_dir=os.path.join(index_dir, "images"))
        if verbose:
            print(f"    -> {len(cs)} 块")
        all_chunks.extend(cs)
    for i, c in enumerate(all_chunks):
        c["id"] = i
    if verbose:
        print(f"  嵌入 {len(all_chunks)} 块 (bge-m3, CPU) ...")
    emb = Embedder().encode([c.get("text") or c.get("caption", "")
                             for c in all_chunks], show_progress=verbose)
    vs = VectorStore()
    vs.add(all_chunks, emb)
    vs.save(index_dir)
    if verbose:
        print(f"  索引已保存: {index_dir} ({len(all_chunks)} 块)")
    return vs


def build_index_from_chunks(chunks, index_dir, shard=500, verbose=True):
    """
    分批嵌入并逐片落盘, 支持断点续跑 (中途中断后重跑只补未完成的分片)。
    大语料(如 9 份金融年报)在 CPU 上嵌入耗时长, 分片保存可避免前功尽弃。

    工单编号: 人工智能NLP-RAG-功能测试及评估
    """
    os.makedirs(index_dir, exist_ok=True)
    emb_dir = os.path.join(index_dir, "shards")
    os.makedirs(emb_dir, exist_ok=True)
    emb = Embedder()
    n = len(chunks)
    for i, c in enumerate(chunks):
        c["id"] = i
    parts = []
    for s in range(0, n, shard):
        path = os.path.join(emb_dir, f"shard_{s:06d}.npy")
        if os.path.exists(path):
            vecs = np.load(path)
            if vecs.shape[0] == min(shard, n - s):
                parts.append(vecs)
                if verbose:
                    print(f"  [resume] shard {s} 已存在, 跳过")
                continue
        texts = [(c.get("text") or c.get("caption", ""))
                 for c in chunks[s:s + shard]]
        vecs = emb.encode(texts)
        np.save(path, vecs)
        parts.append(vecs)
        if verbose:
            done = min(s + shard, n)
            print(f"  嵌入进度 {done}/{n}", flush=True)
    matrix = np.vstack(parts) if parts else np.zeros((0, emb.dim), np.float32)
    vs = VectorStore()
    vs.add(chunks, matrix)
    vs.save(index_dir)
    if verbose:
        print(f"  索引已保存: {index_dir} ({n} 块)")
    return vs


def merge_indexes(src_dirs, out_dir, verbose=True):
    """
    合并多个已建好的索引 (增量入库): 直接拼接 chunk 与向量, 避免重复嵌入。
    用于工单03/04 在 01 已有索引基础上新增《招股说明书2.pdf》。
    (工单编号: 人工智能NLP-RAG-PDF文档的表格解析及检索优化)
    """
    chunks, embs = [], []
    for d in src_dirs:
        if not VectorStore.exists(d):
            if verbose:
                print(f"  [skip] 索引不存在: {d}")
            continue
        vs = VectorStore.load(d)
        chunks.extend(vs.chunks)
        embs.append(vs.emb)
        if verbose:
            print(f"  + {os.path.basename(d)}: {len(vs.chunks)} 块")
    if not chunks:
        raise RuntimeError("没有可合并的索引")
    emb = np.vstack(embs)
    for i, c in enumerate(chunks):
        c["id"] = i
    vs = VectorStore()
    vs.add(chunks, emb)
    vs.save(out_dir)
    if verbose:
        print(f"  合并完成: {len(chunks)} 块 -> {out_dir}")
    return vs


def load_json(path, default=None):
    if os.path.exists(path):
        return json.load(open(path, encoding="utf-8"))
    return default if default is not None else []


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(obj, open(path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
