# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 本模块职责（解析与分块）：向量模型加载、PDF 解析（文字+表格）、语义分块。
# 存储层见 app/store.py，在线问答见 app/retrieval.py，HTTP 层见 app/main.py。

import os                             # 读取/传递子进程环境变量
import re                             # 正则：HTML 转文本、句子切分
import sys                            # 取当前解释器路径，用于拉起 MinerU 子进程
import json                           # 读取 MinerU 的 content_list.json
import hashlib                        # 生成文档 doc_id
import subprocess                     # 调用 MinerU 命令行
from pathlib import Path              # 路径处理

import numpy as np                    # 向量批量运算与余弦相似度计算
from . import config as C             # 全局配置（路径、参数、开关）

_EMBEDDER = None                      # 向量模型全局单例，避免重复加载占用显存
_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")   # 英文/数字 token 提取（中文交给 jieba）
_HTML_TAG = re.compile(r"<[^>]+>")        # 用于剥离表格 HTML 标签


# ============================ 向量模型 ============================
def get_embedder():                   # 懒加载 bge-m3 向量模型
    """返回 sentence-transformers 模型单例；加载失败抛出异常由上层降级。"""
    global _EMBEDDER                  # 声明使用模块级单例
    if _EMBEDDER is None:             # 首次调用才加载
        from sentence_transformers import SentenceTransformer   # 延迟导入，缩短启动时间
        _EMBEDDER = SentenceTransformer(C.EMBED_MODEL_PATH, device=C.EMBED_DEVICE)  # 本地路径加载
        _EMBEDDER.max_seq_length = C.EMBED_MAX_TOKENS   # 限制序列长度：8192 会让长表格块拖慢几十倍
    return _EMBEDDER                  # 返回单例


def embed_texts(texts: list[str]) -> np.ndarray:                    # 批量文本向量化
    """返回 L2 归一化后的向量矩阵（配合 Milvus COSINE 度量即等价余弦相似度）。

    实现要点：**先按文本长度排序再分批，最后还原顺序**。
    编码时同批内所有样本会被 padding 到该批最长长度，若长短样本混在一批，
    一个 8000 字的表格块会把整批拖到 8000 token，实测单批从 3.6s 恶化到 70s。
    排序后同批长度接近，padding 浪费最小，整体提速一个数量级。
    """
    if not texts:                                                   # 空输入直接返回空矩阵
        return np.zeros((0, C.EMBED_DIM), dtype=np.float32)          # 避免下游报错
    model = get_embedder()                                          # 取模型单例
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))   # 按长度升序的原始下标
    vecs = model.encode([texts[i] for i in order], batch_size=32,    # 用排序后的顺序编码
                        normalize_embeddings=True, show_progress_bar=False,
                        convert_to_numpy=True)                       # GPU 批量编码
    out = np.empty_like(vecs)                                        # 准备还原顺序的输出
    for pos, orig in enumerate(order):                               # 遍历排序位置与原下标
        out[orig] = vecs[pos]                                        # 放回原来的位置
    return np.asarray(out, dtype=np.float32)                         # 统一 float32，Milvus 要求


# ============================ PDF 解析 ============================
def _html_table_to_text(html: str) -> str:                          # 表格 HTML 转纯文本
    """把 MinerU 的 table_body(HTML) 还原成「行内以 | 分隔」的可读文本，保留行列结构。"""
    if not html:                                                    # 空值直接返回
        return ""                                                   # 避免后续报错
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, flags=re.S | re.I)   # 逐行提取
    lines = []                                                      # 收集每一行的文本
    for row in rows:                                                # 遍历表格行
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, flags=re.S | re.I)  # 提取单元格
        cells = [re.sub(r"\s+", " ", _HTML_TAG.sub("", c)).strip() for c in cells]  # 去标签、压空白
        if any(cells):                                              # 跳过全空行
            lines.append(" | ".join(cells))                         # 用 | 连接单元格，保持列对应
    return "\n".join(lines)                                         # 行间换行


def parse_with_mineru(pdf_path: Path, out_root: Path) -> list[dict]:    # MinerU 主解析路径
    """调用 MinerU pipeline 解析 PDF，返回统一结构的块列表。失败抛异常，由上层降级。"""
    cached = list(out_root.rglob("*_content_list.json")) if out_root.exists() else []  # 查找已有解析产物
    if cached and cached[0].stat().st_mtime >= pdf_path.stat().st_mtime:   # 产物比 PDF 新 → 直接复用
        print(f"[解析复用] 命中已有 MinerU 产物，跳过重复解析：{cached[0].name}")  # 提示复用，省时
        return _blocks_from_content_list(json.loads(cached[0].read_text(encoding="utf-8")))  # 转换并返回
    env = dict(os.environ)                                          # 复制当前环境变量
    env["PYTHONPATH"] = C.MINERU_LANGFIX_DIR + os.pathsep + env.get("PYTHONPATH", "")  # 注入中文路径补丁
    env["MINERU_MODEL_SOURCE"] = C.MINERU_MODEL_SOURCE              # 指定模型来源为 modelscope 镜像
    cmd = [sys.executable, "-c", "from mineru.cli.client import main; main()",  # 以模块入口方式调用
           "-p", str(pdf_path), "-o", str(out_root),                # 输入 PDF 与输出目录
           "-b", "pipeline", "-l", "ch", "-t", "True"]              # pipeline 后端、中文、开启表格
    subprocess.run(cmd, env=env, check=True, capture_output=True,   # 同步执行并捕获输出
                   timeout=C.MINERU_TIMEOUT)                        # 超时保护，防止卡死
    found = list(out_root.rglob("*_content_list.json"))             # 递归找解析结果文件
    if not found:                                                   # 没有产物说明解析失败
        raise RuntimeError("MinerU 未生成 content_list.json")        # 抛出以便触发降级
    return _blocks_from_content_list(json.loads(found[0].read_text(encoding="utf-8")))  # 转换并返回


def _blocks_from_content_list(raw: list[dict]) -> list[dict]:       # content_list → 统一结构块
    """把 MinerU 的 content_list.json 转成本项目统一的块结构，表格取 table_body 字段。"""
    blocks = []                                                     # 统一结构的块列表
    last_title = ""                                                 # 最近出现的章节标题，用于给表格补充语境
    for item in raw:                                                # 遍历 MinerU 的每一个块
        kind = item.get("type")                                     # 块类型：text/table/header/...
        page = int(item.get("page_idx", 0)) + 1                     # MinerU 页号从 0 开始，转 1 起
        if kind == "table":                                         # 表格块：内容在 table_body 而非 text
            body = _html_table_to_text(item.get("table_body") or "")  # HTML 表格转文本
            caption = item.get("table_caption") or ""               # 表题可能是列表或字符串
            if isinstance(caption, list):                           # 列表形式
                caption = caption[0] if caption else ""             # 取首个表题
            # 表格本身往往只有"国防领域 | 4,627.14 | 94.34%"这类裸数字，既没有"军用领域"也没有"收入"，
            # 不给语境的话 BM25 与向量都很难命中。这里把最近的章节标题一并带上，显著提升表格可检索性。
            content = (last_title + "\n" + caption + "\n" + body).strip() if last_title else \
                      (caption + "\n" + body).strip()               # 章节标题 + 表题 + 表体
            if content:                                             # 非空才保留
                blocks.append({"type": "table", "page_no": page, "content": content})  # 记录块
        elif kind == "text":                                        # 只取正文块
            # 说明：MinerU 的 header/footer/page_number 是每页重复的页眉页脚（如“招股意向书”“1-1-20”），
            #       全篇重复上千次且无检索价值，直接丢弃；text_level 非空的是章节标题，属有效结构。
            text = (item.get("text") or "").strip()                 # 取出文本
            if text:                                                # 过滤空块
                if item.get("text_level"):                          # 是章节标题
                    t, last_title = "title", text                   # 记为标题并更新最近标题
                else:                                               # 普通正文
                    t = "text"                                      # 标记为正文
                blocks.append({"type": t, "page_no": page, "content": text})  # 记录块
    if not blocks:                                                  # 一个块都没有也算失败
        raise RuntimeError("MinerU 解析结果为空")                     # 触发降级
    return blocks                                                   # 返回统一结构


def parse_with_fast(pdf_path: Path) -> list[dict]:                  # 降级解析路径：pymupdf + pdfplumber
    """不依赖 MinerU 的快速解析，文字用 pymupdf、表格用 pdfplumber，输出同一结构。"""
    import pymupdf                                                  # 延迟导入，减小启动开销
    import pdfplumber                                               # 延迟导入
    doc = pymupdf.open(str(pdf_path))                               # 打开 PDF
    blocks = []                                                     # 统一结构的块列表
    for i in range(doc.page_count):                                 # 逐页抽取文字
        text = doc[i].get_text().strip()                            # 第 i 页（0 起）文本
        for para in re.split(r"\n{2,}", text):                      # 按空行切段，避免整页一大块
            para = para.strip()                                     # 去掉首尾空白
            if len(para) >= 10:                                     # 过短的碎片丢弃（多为页眉页脚）
                blocks.append({"type": "text", "page_no": i + 1, "content": para})  # 记录段落
    doc.close()                                                     # 释放文件句柄
    with pdfplumber.open(str(pdf_path)) as pdf:                     # 用 pdfplumber 抽表格
        for i, page in enumerate(pdf.pages):                        # 逐页处理
            for table in page.extract_tables() or []:               # 每页可能有 0~N 张表
                rows = [" | ".join((c or "").replace("\n", " ").strip() for c in r) for r in table]  # 规整
                rows = [r for r in rows if r.strip(" |")]            # 丢弃空行
                if len(rows) >= 2:                                  # 至少 2 行才算有效表格
                    blocks.append({"type": "table", "page_no": i + 1, "content": "\n".join(rows)})  # 记录表
    return blocks                                                   # 返回统一结构


def parse_pdf(pdf_path: Path, use_mineru: bool = True) -> tuple[list[dict], str]:  # 解析总入口
    """按「MinerU → 降级 pymupdf」顺序解析，返回 (块列表, 实际使用的解析器名)。

    若两条路径都取不到任何文本，直接抛错而不是返回空列表。
    动机：实测把一页招股书渲染成位图（模拟扫描件）后，pymupdf 提取出 0 字符。
    原实现会把 0 个块"成功"入库，用户得到一个静默的空知识库，还以为上传成功了。
    """
    if use_mineru:                                                  # 用户/配置允许用 MinerU
        try:                                                        # 尝试主解析路径
            out_root = C.PARSED_DIR / pdf_path.stem                 # 每篇文档独立输出目录
            return parse_with_mineru(pdf_path, out_root), "mineru"  # 成功则直接返回
        except Exception as exc:                                    # 任何失败（含中文路径、超时、OOM）
            print(f"[解析降级] MinerU 失败，转用 pymupdf：{type(exc).__name__}: {exc}")  # 打印原因
    blocks = parse_with_fast(pdf_path)                              # 降级：快速解析
    if not blocks:                                                  # 两条路径都没提取到文本
        raise RuntimeError("该 PDF 未提取到任何文本，可能是扫描件（无文本层）。"
                           "请改用带文本层的 PDF，或用支持 OCR 的解析方式重新导出。")  # 明确报错
    return blocks, "fast"                                           # 返回块列表与解析器名


# ============================ 语义分块 ============================
def _split_sentences(text: str) -> list[str]:                       # 句子切分
    """按中英文句末标点切句，过短片段向后合并，保证语义单元有足够上下文。"""
    parts = [s for s in re.split(r"(?<=[。！？!?；;\n])", text) if s and s.strip()]  # 按句末标点切
    merged, buf = [], ""                                            # merged 存结果，buf 存待合并片段
    for s in parts:                                                 # 遍历切出的片段
        buf += s                                                    # 累积到缓冲区
        if len(buf) >= 20:                                          # 攒够 20 字才算一个语义单元
            merged.append(buf.strip())                              # 收进结果
            buf = ""                                                # 清空缓冲
    if buf.strip():                                                 # 处理尾部残留
        merged.append(buf.strip())                                  # 收进结果
    return merged or ([text.strip()] if text.strip() else [])       # 兜底：整块作为一句


def semantic_chunk(blocks: list[dict]) -> list[dict]:               # 语义分块主函数
    """基于相邻句向量相似度的语义分块：相似度骤降处断开，再按长度归并。表格块整块保留。"""
    sentences, meta = [], []                                        # 句子列表与对应的来源元信息
    for b in blocks:                                                # 遍历所有结构化块
        if b["type"] == "table":                                    # 表格单独成块，不参与语义切分
            continue                                                # 跳过，稍后直接追加
        for s in _split_sentences(b["content"]):                    # 正文块切句
            sentences.append(s)                                     # 收集句子
            meta.append(b["page_no"])                               # 记录来源页
    chunks = []                                                     # 最终分块结果
    if sentences:                                                   # 有正文才做语义切分
        vecs = embed_texts(sentences)                               # 批量向量化所有句子
        sims = np.sum(vecs[:-1] * vecs[1:], axis=1)                 # 归一化向量点积即余弦相似度
        thr = float(np.percentile(sims, C.SEMANTIC_BREAK_PERCENTILE)) if sims.size else 0.0  # 断点阈值
        buf, pages = "", []                                         # 当前块文本与涉及页码
        for i, s in enumerate(sentences):                           # 逐句决定是否断块
            buf += s                                                # 拼接到当前块
            pages.append(meta[i])                                   # 记录页码
            is_break = i < len(sims) and sims[i] < thr              # 相似度低于阈值 → 语义转折点
            if len(buf) >= C.CHUNK_MAX_CHARS or (is_break and len(buf) >= C.CHUNK_MIN_CHARS):  # 断块条件
                chunks.append({"content": buf.strip(), "page_no": min(pages),  # 起始页
                               "page_end": max(pages), "type": "text"})          # 结束页与类型
                buf, pages = "", []                                 # 重置缓冲，开始下一块
        if buf.strip():                                             # 尾部剩余内容
            chunks.append({"content": buf.strip(), "page_no": min(pages),        # 收尾块
                           "page_end": max(pages), "type": "text"})              # 类型为正文
    for b in blocks:                                                # 再把表格块加入
        if b["type"] == "table":                                    # 只处理表格
            for content in _split_big_table(b["content"]):           # 超大表按行切分（保留表头）
                chunks.append({"content": content, "page_no": b["page_no"],      # 每片沿用原页码
                               "page_end": b["page_no"], "type": "table"})       # 保留行列结构
    chunks.sort(key=lambda c: (c["page_no"], c["type"]))            # 按页码排序，便于人工核查
    for i, c in enumerate(chunks):                                  # 编号，供引用定位
        c["chunk_index"] = i                                        # 写入分块序号
    return chunks                                                   # 返回带序号的分块


# ============================ 工具函数 ============================
def _split_big_table(content: str, max_chars: int = 1200) -> list[str]:   # 超大表格切分
    """表格超过 max_chars 时按行切分，每片都重复首行表头，避免列名与数据分离。

    动机：bge-m3 编码时按批内最长样本 padding，单个 8000 字的表格块会把整批拖慢数十倍；
    同时过长的表格块命中后也会挤占上下文预算。切分后每片自带表头，检索与阅读都不受影响。
    """
    if len(content) <= max_chars:                                    # 未超限
        return [content]                                             # 原样返回
    lines = content.split(chr(10))                                   # 按换行符拆分（chr(10) 避免转义歧义）
    header, body = lines[0], lines[1:]                               # 首行为表头（含表题）
    out, buf, n = [], [header], len(header)                          # 结果、当前片缓冲、当前片长度
    for ln in body:                                                  # 逐行累加
        if n + len(ln) > max_chars and len(buf) > 1:                 # 超过上限且已有数据行
            out.append(chr(10).join(buf) + chr(10) + "（表续）")       # 收片并标注续接
            buf, n = [header], len(header)                           # 重置为新片，保留表头
        buf.append(ln); n += len(ln) + 1                             # 累加当前行
    if len(buf) > 1:                                                 # 处理尾部
        out.append(chr(10).join(buf))                                # 收尾片
    return out or [content]                                          # 兜底返回原内容


def tokenize(text: str) -> list[str]:                               # 中英混合分词
    """中文用 jieba 精确模式，英文/数字用正则提取，结果统一小写。"""
    import jieba                                                    # 延迟导入，避免影响启动
    toks = [w.strip() for w in jieba.lcut(text) if w.strip()]       # 中文分词
    toks += [w.lower() for w in _TOKEN_RE.findall(text)]            # 补充英文/数字 token
    return [t.lower() for t in toks if len(t) > 1 or not t.isascii()]  # 丢掉单字符英文噪声


def make_doc_id(pdf_path: Path) -> str:                             # 生成文档标识
    """按文件内容 md5 生成 doc_id：同一文件重复上传可复用、不同文件不冲突。"""
    h = hashlib.md5()                                               # 初始化 md5
    with open(pdf_path, "rb") as f:                                 # 二进制读取
        for blk in iter(lambda: f.read(1 << 20), b""):              # 每次 1MB，避免大文件吃内存
            h.update(blk)                                           # 累加摘要
    return h.hexdigest()[:16]                                       # 取前 16 位做短标识
