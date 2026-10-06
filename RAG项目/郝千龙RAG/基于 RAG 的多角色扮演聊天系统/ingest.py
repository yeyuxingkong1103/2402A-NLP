# -*- coding: utf-8 -*-  # 声明源文件编码为 UTF-8，确保中文注释/字符串不乱码
"""【文档入库 · ingest.py】离线/动态知识入库：解析 TSV/TXT/PDF（PyMuPDF 主解析 → PaddleOCR 识别扫描件 → pdfplumber 表格兜底，含去水印）→ 段落+定长分块 → BM25/数据库/Milvus 三路落库。"""  # 模块级 docstring：中文名 + 文件名 + 一句话作用
from __future__ import annotations  # 开启 PEP 563 延迟注解求值，让 list[SentencePair] 这类注解在 3.9 以下也能用

from pathlib import Path  # 引入 Path 类，统一用面向对象方式处理文件路径（跨平台、更安全）

from database import KnowledgeDoc, SessionLocal  # ORM 文档表模型与会话工厂，用于 MySQL 落库
from logger import log  # 全局日志器，输出处理过程中的告警与调试信息
from rag import SentencePair, TranslationRetriever  # 句对数据结构与 BM25 检索器，用于分块封装与索引更新

MAX_CHUNK = 400  # 单块最大字符数（分块大小，需按文档类型调参）  # 控制块大小：太小丢上下文，太大稀释向量语义且撑爆提示词


def chunk_text(text: str, source: str) -> list[SentencePair]:  # 定义分块函数：入参为原文与来源路径，返回 SentencePair 列表
    """分块：段落累积到 MAX_CHUNK；超长段落硬切。每块封装为 SentencePair。"""  # docstring：说明混合切分策略（段落累积 + 固定长度）
    paragraphs = [p.strip() for p in text.replace("\r\n", "\n").split("\n") if p.strip()]  # 统一换行符后按行切分，去首尾空白并过滤空行，得到段落列表
    chunks: list[str] = []  # 最终分块结果容器，存放切好的每段文本
    buf = ""  # 累积缓冲区：当前正在拼装的块，未达 MAX_CHUNK 前持续累积下一段
    for para in paragraphs:  # 遍历每个段落，按"能塞进当前块就累积，否则落盘并开新块"策略推进
        if len(buf) + len(para) <= MAX_CHUNK:  # 当前缓冲区加上本段后仍未超上限
            buf = f"{buf}\n{para}".strip()  # 用换行拼接到缓冲区，strip 掉首尾换行保持整洁
        else:  # 超出 MAX_CHUNK：当前缓冲区已满，需要落盘并处理本段
            if buf:  # 若缓冲区非空，先把已累积的内容作为一个块保存
                chunks.append(buf)
            if len(para) <= MAX_CHUNK:  # 本段虽超当前剩余空间，但单独不超上限
                buf = para  # 把它作为新块的开头，继续后续累积
            else:  # 段落本身超长：按 MAX_CHUNK 硬切  # 单段已超过单块容量，无法靠累积解决，必须硬切
                for i in range(0, len(para), MAX_CHUNK):  # 以 MAX_CHUNK 为步长遍历整段，逐段切片
                    chunks.append(para[i : i + MAX_CHUNK])  # 取出固定长度子串直接作为独立块
                buf = ""  # 硬切后缓冲区清空，等待下一段累积
    if buf:  # 循环结束后若缓冲区仍有残留内容，作为最后一块保存，避免漏数据
        chunks.append(buf)
    pairs = []  # 句对结果列表：把每个文本块封装成 SentencePair 供下游索引使用
    for i, chunk in enumerate(chunks):  # 遍历分块并带索引，用于生成稳定的块 ID
        summary = chunk[:80].replace("\n", " ")  # 摘要取块前 80 字符  # 取块前 80 字符作为摘要，并把换行换成空格避免单行展示断裂
        pairs.append(  # 构造并追加一个 SentencePair 对象到结果列表
            SentencePair(  # 复用句对结构承载文档块，使下游检索/落库逻辑统一
                en_id=f"doc-{i}",  # 文档块英文侧 ID 用 doc- 前缀加序号，与句对区分
                english=f"[DOC] {summary}",  # [DOC] 前缀标记这是文档块而非句对  # 在英文位填摘要并加 [DOC] 前缀，便于检索后过滤文档类条目
                zh_id=str(i),  # 中文侧 ID 直接用序号字符串，保持唯一
                chinese=chunk,  # 中文位存放完整块文本，作为检索召回后回填给 LLM 的正文
                source=source,  # 记录来源路径，便于溯源和按文档管理
                summary=summary,  # 单独字段保留摘要，供 knowledge_docs 表展示
            )
        )
    return pairs  # 返回所有句对，供 ingest_file 三路入库


def read_pdf(path: Path) -> str:  # PDF 文本提取入口：返回拼好的全篇纯文本
    """PyMuPDF 读取 PDF，并尝试去水印。"""  # docstring：说明优先用 PyMuPDF，并在提取前去水印
    try:  # 用 try 包裹，避免 fitz 缺失或文件损坏时中断整个入库流程
        import fitz  # 懒加载 fitz（PyMuPDF），避免未安装时整个模块导入失败

        doc = fitz.open(path)  # 打开 PDF 文档对象，支持流式逐页访问
        pages = []  # 收集每页文本的容器，最后拼接为整篇
        for page in doc:  # 遍历每一页，逐页处理避免一次性内存占用过大
            _remove_watermark(page)  # 逐页去水印后再取文本  # 在取文本前先调用去水印，避免水印文字混入正文
            pages.append(page.get_text())  # 提取当前页纯文本并加入列表
        text = "\n".join(pages)  # 用换行把各页文本拼成完整文档，保持页边界
        if text.strip():  # 提取到非空内容：直接返回交给分块
            return text
        log.warning("pymupdf 文本为空，可能为扫描件: %s", path)  # 文本为空多半是扫描件，告警并走兜底
    except Exception as exc:  # fitz 未安装、PDF 加密或解析异常都到这里
        log.warning("pymupdf failed: %s", exc)  # 记录告警但不抛出，保证流程继续到兜底路径
    # 扫描件兜底：先走 PaddleOCR 图像识别（适合纯图片型扫描 PDF），再退回 PDFPlumber 表格解析
    ocr_text = _read_pdf_ocr(path)  # 尝试用 PaddleOCR 把每页渲染成图片后做文字识别
    if ocr_text:  # OCR 成功提取到非空文本：直接返回
        return ocr_text
    return _read_pdf_tables(path)  # 兜底：PDFPlumber 表格解析  # OCR 也失败时退回 PDFPlumber，专攻表格型 PDF


def _remove_watermark(page) -> None:  # 页级水印清理：直接修改页面对象，无返回值
    """去除常见浅色/透明水印：删除低透明度或灰度极高的文字/图片对象。"""  # docstring：说明判定逻辑（颜色浅 + 透明度低）
    try:  # 用 try 包裹整段逻辑：水印结构千差万别，任何一步异常都不应中断 PDF 解析
        import fitz  # 延迟导入 fitz（PyMuPDF），与 read_pdf 中独立导入，确保本函数可访问
        # 文字水印：透明度低或颜色极浅（RGB 均大于 230 视为水印色）  # 保留原解释：判定阈值基于经验，浅色文字几乎只出现在水印里
        for block in page.get_text("dict").get("blocks", []):  # 取页面结构化 dict，遍历顶层 block（可能是文本或图片块）
            for line in block.get("lines", []):  # 进入文本块内部的行
                for span in line.get("spans", []):  # 进入行内的 span，每个 span 是一段同格式文字
                    color = span.get("color", 0)  # 取该 span 的颜色整数（sRGB 编码，缺失则按黑色 0 处理）
                    r, g, b = ((color >> 16) & 0xFF, (color >> 8) & 0xFF, color & 0xFF)  # 把整数颜色拆成 R/G/B 三通道，按位右移+掩码
                    if r > 230 and g > 230 and b > 230:  # 三通道都极亮：判定为水印色（接近白色，正文一般不会这么浅）
                        rect = fitz.Rect(span["bbox"])  # 用白色矩形遮盖水印区域  # 用 span 的包围盒构造矩形区域，准备做 redact
                        page.add_redact_annot(rect, fill=(1, 1, 1))  # 添加 redact 注解并指定白色填充（1,1,1 即 RGB 满值），遮盖原内容
        page.apply_redactions()  # 实际应用所有 redact 注解：擦除原像素并用填充色覆盖，水印被永久抹掉
    except Exception as exc:  # 任一环节失败（如 fitz 未导入、结构异常）都不致命
        log.debug("watermark text strip skipped: %s", exc)  # 仅 debug 级别记录：去水印是锦上添花，跳过不影响主流程


def _read_pdf_ocr(path: Path) -> str:  # 扫描版 PDF 识别：PyMuPDF 提不出文本时，把每页渲染成图片交给 PaddleOCR 识别
    """PaddleOCR 识别扫描版 PDF：每页渲染为图片 → OCR 提取文字；未安装或失败返回空串。"""  # docstring：说明本函数是扫描件的图像识别兜底
    try:  # try 包裹：paddleocr 体积大可能未安装，失败时静默降级到下一个兜底
        import io  # 内存字节流：把 PNG 字节包装成 PIL 可读的流，无需落盘临时文件
        import fitz  # 复用 PyMuPDF 做 PDF 页面渲染（把矢量页转成位图）
        import numpy as np  # OCR 模型输入要求是 numpy 数组
        from paddleocr import PaddleOCR  # 懒加载 PaddleOCR，避免未安装时整个模块导入失败
        from PIL import Image  # PIL 解码 PNG 字节流

        doc = fitz.open(path)  # 打开 PDF，准备逐页渲染
        ocr = PaddleOCR(use_textline_orientation=True, lang="ch")  # 初始化 OCR：开启文字方向纠正（自动扶正倾斜行），中文模型
        pages_text = []  # 收集每页识别出的文字
        for page in doc:  # 遍历每一页
            pix = page.get_pixmap(dpi=200)  # 把页面渲染成 200 DPI 位图（DPI 太低识别率差，太高内存爆）
            img = Image.open(io.BytesIO(pix.tobytes("png")))  # PNG 字节 → PIL 图片对象
            result = ocr.predict(np.array(img))  # 执行 OCR：PaddleOCR 3.x 用 predict，输入 numpy 数组
            for res in result:  # predict 返回 OCRResult 列表（每页一个）
                pages_text.extend(res["rec_texts"])  # rec_texts 是该页所有识别出的文字行
        text = "\n".join(pages_text)  # 每行识别结果按换行拼接
        if text.strip():  # 识别到非空内容才返回
            log.info("paddleocr extracted %s chars from %s", len(text), path)  # 记录 OCR 成功日志
            return text
        log.warning("paddleocr 未识别到文字: %s", path)  # OCR 跑了但无结果（可能是纯图片/图表页），继续降级
    except Exception as exc:  # paddleocr 未安装、模型下载失败、渲染异常都到这里
        log.warning("paddleocr failed: %s", exc)  # 记录告警，不阻断流程
    return ""  # 返回空串，调用方继续走 PDFPlumber 兜底


def _read_pdf_tables(path: Path) -> str:  # PDF 兜底解析：PyMuPDF 拿不到文本时改用 PDFPlumber 抽表格
    """PDFPlumber 解析表格，作为兜底。"""  # docstring：说明本函数是 read_pdf 失败后的退路
    try:  # try 包裹：pdfplumber 也可能未装或对加密 PDF 失败，需要进一步兜底
        import pdfplumber  # 懒加载 pdfplumber，避免未安装时模块导入失败

        rows_text = []  # 收集每页表格行与普通文本的结果容器
        with pdfplumber.open(path) as pdf:  # 用 with 保证文件句柄正确关闭
            for page in pdf.pages:  # 逐页处理，与 read_pdf 保持一致
                tables = page.extract_tables() or []  # 先抽表格（逐行拼接）  # 优先抽表格：表格型 PDF 的核心信息在表里，抽不到返回空列表兜底
                for tbl in tables:  # 遍历本页所有识别到的表格
                    for row in tbl:  # 遍历表格每一行
                        rows_text.append("\t".join((c or "").strip() for c in row))  # 单元格用制表符连接，便于后续还原成结构化文本
                text = page.extract_text() or ""  # 再抽普通文本  # 表格之外再抽普通正文，避免漏掉非表格区域
                if text:  # 若有普通文本则追加，没有就跳过
                    rows_text.append(text)
        return "\n".join(rows_text)  # 把每页结果按行拼接返回，交给上层分块
    except Exception as exc:  # pdfplumber 也失败：多半是非 PDF 或严重损坏
        log.warning("pdfplumber failed: %s", exc)  # 记录告警，准备走最后兜底
        return path.read_text(encoding="utf-8", errors="ignore")  # 直接把文件当 UTF-8 文本读取（忽略坏字节），保证流程不空手而归


def ingest_file(path: Path, retriever: TranslationRetriever) -> int:  # 入库总入口：按文件类型分发解析，返回入库块数
    """知识入库总入口：按扩展名分发解析 → 分块 → BM25/Milvus/MySQL 三路落库。"""  # docstring：说明三路落库的整体设计
    suffix = path.suffix.lower()  # 取扩展名并转小写，做分支分发，避免大小写不一致漏判
    if suffix == ".tsv":  # TSV：约定四列句对格式，走专门加载器
        from rag import load_pairs  # 懒加载 load_pairs：仅 TSV 分支需要，避免无谓依赖

        pairs = load_pairs(path)  # TSV：按四列句对加载  # 直接按既定结构读出 SentencePair 列表，无需分块
    elif suffix == ".pdf":  # PDF：先解析后分块
        pairs = chunk_text(read_pdf(path), str(path))  # PDF：去水印/表格解析后分块  # read_pdf 完成提取（含去水印/表格兜底），再交给 chunk_text 切块
    else:  # 其余扩展名按纯文本处理
        pairs = chunk_text(path.read_text(encoding="utf-8", errors="ignore"), str(path))  # 纯文本  # UTF-8 读全文并忽略坏字节，再分块
    n = retriever.add_pairs(pairs)  # ① 进 BM25 索引  # 第一路：写入内存 BM25 索引，支持关键词召回，返回实际写入条数
    with SessionLocal() as db:  # 打开数据库会话，with 保证会话结束自动关闭
        db.add(  # ② 记录到 knowledge_docs 表（来源/摘要/块数）  # 第二路：在 MySQL 的 knowledge_docs 表登记一条文档元信息
            KnowledgeDoc(  # 构造 ORM 对象：记录来源、摘要、块数，便于后续管理与溯源
                source=str(path),  # 用绝对路径字符串作为来源标识，唯一指向该文档
                summary=pairs[0].summary if pairs else "",  # 取首个块的摘要代表整篇；空列表时给空串避免报错
                chunk_count=n,  # 记录本次入库的块数，便于核查与统计
            )
        )
        db.commit()  # 提交事务，真正写入数据库
    try:  # 第三路向量入库放 try：向量服务依赖外部模型/Milvus，失败不应影响前两路
        from embeddings import embed_texts  # 懒加载向量化函数，避免无向量服务时整个模块报错
        from vector_store import upsert_texts  # 懒加载 Milvus 写入函数

        texts = [p.as_text() for p in pairs]  # 把每个 SentencePair 转成可向量化的纯文本串
        vectors = embed_texts(texts)  # 批量生成向量，依赖 embeddings 服务
        if vectors:  # 向量非空才入库，避免空列表触发 Milvus 写入异常
            upsert_texts(texts, vectors, source=str(path), summary=pairs[0].summary if pairs else "")  # ③ 向量入 Milvus  # 第三路：upsert 到 Milvus，带来源与摘要便于过滤
    except Exception as exc:  # 向量服务未启用或异常：跳过向量入库，不影响主流程
        log.warning("vector ingest skipped: %s", exc)  # 仅告警，BM25 与 MySQL 已写入，检索仍可走关键词路径
    return n  # 返回入库块数，供调用方打印/统计


if __name__ == "__main__":  # 脚本直接执行时跑批入库；被 import 时不触发
    pairs = []  # 空句对列表，add_pairs 在入库时填充
    retriever = TranslationRetriever(pairs=pairs, bm25=None)  # 构造检索器，后续 ingest_file 会通过它写入索引

    docs_dir = Path("./docs")  # 指定文档目录（相对路径，运行时的工作目录下应有 docs/）
    for f in docs_dir.iterdir():  # 遍历目录下所有条目（含子目录与各种文件）
        if f.suffix.lower() in [".pdf", ".txt", ".tsv"]:  # 仅处理三种支持的扩展名，跳过子目录与其他文件
            print(f"\n正在入库：{f.name}")  # 打印当前处理文件名，便于观察批处理进度
            cnt = ingest_file(f, retriever)  # 调用入库总入口，对该文件解析+分块+三路落库
            print(f"{f.name} -> 分块 {cnt} 条")  # 打印入库后的分块条数，确认每文件处理结果
    print("\n🎉 全部文档入库完毕")  # 全部处理完打印完成提示，方便人工确认批次结束




# =====================================================================
# 知识点说明（RAG：数据准备 / 分块）
# ---------------------------------------------------------------------
# 1. RAG 离线阶段：原始文档（PDF/TSV/TXT）→ 解析 → 清洗（去水印）→
#    分块（Chunking）→ 向量化 → 入库（BM25 + Milvus）。
# 2. 分块策略：本文件用"段落累积 + 固定长度 MAX_CHUNK=400"混合切分。
#    常见策略还有：按句子、按标题层级、语义分块（按向量相似度找断点）、
#    父子块（子块用于检索、父块用于生成，兼顾召回精度与上下文完整）。
#    块太大会稀释向量语义、撑爆提示词；块太小会丢上下文——需按文档调参。
# 3. PDF 处理：PyMuPDF(fitz) 提取文本并删除浅色水印（redact 去水印）；
#    纯扫描件回退 PDFPlumber 表格解析；更复杂版面可用 PaddleOCR /
#    minerU / 多模态大模型做版面理解与 OCR。
# 4. 数据优化：入库前去重、过滤低质量文档（知识库数据增强）——
#    "垃圾进、垃圾出"，数据质量决定 RAG 效果上限。
# =====================================================================
