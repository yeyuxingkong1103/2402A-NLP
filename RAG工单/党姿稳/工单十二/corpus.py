# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG项目-LightRAG 优化任务
# 模块职责：从两份招股说明书 PDF 抽取逐页文本，并按句子边界切分为语料块（带来源与页码元信息）

import os                                                       # 导入 os 用于路径与文件判断
import re                                                       # 导入正则用于空白清洗与句子切分
import json                                                     # 导入 json 用于缓存读写
import fitz                                                     # 导入 PyMuPDF 用于 PDF 文本抽取
import config                                                   # 导入本工单配置模块
from logger import LOGGER                                       # 导入统一日志对象

_WS = re.compile(r"[ \t 　]+")                         # 连续水平空白（含全角空格）压缩正则
_MULTI_NL = re.compile(r"\n{2,}")                               # 连续换行压缩正则
# 句末标点后切分：中英文句号、问号、叹号、分号，以及右引号/右括号
_SENT_SPLIT = re.compile(r"(?<=[。！？；!?;])(?=[^、。，])")


def _clean(text):                                               # 定义页面文本清洗函数
    """压缩多余空白与换行，去掉页眉页脚残留的孤立数字行。"""
    text = text.replace("\r", "\n")                             # 统一换行符
    text = _WS.sub(" ", text)                                   # 水平空白压缩为单个空格
    lines = [ln.strip() for ln in text.split("\n")]             # 逐行去首尾空白
    lines = [ln for ln in lines if ln]                          # 丢弃空行
    lines = [ln for ln in lines if not re.fullmatch(r"[-—\s]*\d{1,4}[-—\s]*", ln)]  # 丢弃纯页码行
    return _MULTI_NL.sub("\n", "\n".join(lines)).strip()        # 重新拼接并压缩连续换行


def build_or_load_pages(force=False):                           # 定义逐页文本抽取函数
    """抽取两份 PDF 的逐页文本；结果缓存到 data/pages.json，重复运行直接命中缓存。"""
    if not force and os.path.isfile(config.PAGE_CACHE_FILE):    # 缓存存在且非强制重建
        with open(config.PAGE_CACHE_FILE, "r", encoding="utf-8") as f:  # 以 UTF-8 打开缓存
            pages = json.load(f)                                # 反序列化页面列表
        LOGGER.info("命中页面缓存：%d 页", len(pages))           # 输出命中信息
        return pages                                            # 返回缓存的页面列表
    pages = []                                                  # 初始化页面列表
    for pdf_path in config.PDF_FILES:                          # 遍历两份招股说明书
        if not os.path.isfile(pdf_path):                        # 文件不存在时直接报错，避免静默产出空语料
            raise RuntimeError("招股说明书不存在：%s" % pdf_path)  # 抛出明确错误
        doc = fitz.open(pdf_path)                               # 用 PyMuPDF 打开 PDF
        name = os.path.splitext(os.path.basename(pdf_path))[0]  # 取文件名作为来源标识（如「招股说明书1」）
        total_pages = len(doc)                                  # 记录总页数（关闭文档后无法再取）
        for i in range(total_pages):                            # 遍历每一页
            text = _clean(doc[i].get_text())                    # 抽取并清洗该页文本
            if len(text) < config.MIN_CHUNK_LEN:                # 跳过无文本页（封面图、纯图表页）
                continue                                        # 继续下一页
            pages.append({"source": name, "page": i + 1, "text": text})  # 记录来源、页码与正文
        doc.close()                                             # 关闭文档释放句柄
        LOGGER.info("已抽取 %s：共 %d 页", name, total_pages)     # 输出该文件的抽取页数
    config.ensure_dirs()                                        # 确保数据目录存在
    with open(config.PAGE_CACHE_FILE, "w", encoding="utf-8") as f:  # 以 UTF-8 覆盖写入缓存
        json.dump(pages, f, ensure_ascii=False)                 # 序列化页面列表（保留中文）
    LOGGER.info("页面抽取完成：%d 页，写入 %s", len(pages), config.PAGE_CACHE_FILE)  # 输出完成日志
    return pages                                                # 返回页面列表


def _split_sentences(text):                                     # 定义句子切分函数
    """按句末标点切分文本，保证语料块不切断完整句子。"""
    parts = [s.strip() for s in _SENT_SPLIT.split(text)]        # 按句末标点切分并去空白
    return [p for p in parts if p]                              # 丢弃空片段


def _chunk_page(page):                                          # 定义单页切块函数
    """把一页文本按目标长度贪心切块，相邻块保留重叠，块不跨页以便标注来源页码。"""
    sents = _split_sentences(page["text"])                      # 先把页面文本切成句子
    chunks = []                                                 # 初始化该页的块列表
    buf = ""                                                    # 当前累积缓冲区
    for sent in sents:                                          # 遍历每个句子
        if len(buf) + len(sent) <= config.CHUNK_SIZE:           # 加入后不超过目标长度
            buf += sent                                         # 追加到缓冲区
            continue                                            # 继续下一个句子
        if len(buf) >= config.MIN_CHUNK_LEN:                    # 缓冲区已有足够内容才落块
            chunks.append(buf)                                  # 保存当前块
        tail = buf[-config.CHUNK_OVERLAP:] if buf else ""       # 取上一块尾部作为重叠上下文
        buf = tail + sent                                       # 新块以重叠内容开头，避免语义被切断
        while len(buf) > config.CHUNK_SIZE * 2:                 # 单句超长时（如无标点的长表格行）强制切分
            chunks.append(buf[:config.CHUNK_SIZE])              # 按目标长度截断成块
            buf = buf[config.CHUNK_SIZE - config.CHUNK_OVERLAP:]  # 剩余部分保留重叠继续累积
    if len(buf) >= config.MIN_CHUNK_LEN:                        # 处理结尾残留缓冲区
        chunks.append(buf)                                      # 保存最后一块
    return chunks                                               # 返回该页的块文本列表


def build_or_load_chunks(force=False):                          # 定义语料块构建函数
    """把逐页文本切分为语料块并编号，结果缓存到 data/chunks.json。"""
    if not force and os.path.isfile(config.CHUNK_CACHE_FILE):   # 缓存存在且非强制重建
        with open(config.CHUNK_CACHE_FILE, "r", encoding="utf-8") as f:  # 以 UTF-8 打开缓存
            chunks = json.load(f)                               # 反序列化块列表
        LOGGER.info("命中分块缓存：%d 块", len(chunks))          # 输出命中信息
        return chunks                                           # 返回缓存的块列表
    pages = build_or_load_pages()                               # 先确保页面文本已抽取
    chunks = []                                                 # 初始化全局块列表
    for page in pages:                                          # 遍历每一页
        for text in _chunk_page(page):                          # 遍历该页切出的每个块
            chunks.append({"id": len(chunks),                   # 全局唯一块编号（与向量矩阵行号对齐）
                           "source": page["source"],            # 来源说明书名称
                           "page": page["page"],                # 来源页码
                           "text": text})                       # 块正文
    config.ensure_dirs()                                        # 确保数据目录存在
    with open(config.CHUNK_CACHE_FILE, "w", encoding="utf-8") as f:  # 以 UTF-8 覆盖写入缓存
        json.dump(chunks, f, ensure_ascii=False)                # 序列化块列表（保留中文）
    chars = sum(len(c["text"]) for c in chunks)                 # 统计块正文总字数
    LOGGER.info("分块完成：%d 块，共 %d 字", len(chunks), chars)  # 输出分块统计
    return chunks                                               # 返回块列表


if __name__ == "__main__":                                      # 仅在直接运行时执行
    import sys                                                  # 导入 sys 解析命令行参数
    build_or_load_chunks(force="--force" in sys.argv)           # 支持 --force 强制重建
