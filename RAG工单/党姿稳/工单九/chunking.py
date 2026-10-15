# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-Graph RAG 优化任务
# 模块职责：把页面级文档切分为带定位信息的文本块，并负责分块结果的磁盘缓存

import os                                                       # 导入 os 用于缓存路径处理
import json                                                     # 导入 json 用于分块结果序列化
import hashlib                                                  # 导入哈希库用于生成稳定的块 ID
import config                                                   # 导入本工单配置模块
from timing import LOGGER                                       # 导入统一日志对象


def _split_sentences(text):                                     # 定义中文/英文句子切分函数
    """按中英文句末标点切分句子，保留标点，便于在句子边界处切块。"""
    parts = []                                                  # 初始化句子列表
    buf = ""                                                    # 当前句子的缓冲区
    for ch in text:                                             # 逐字符扫描
        buf += ch                                               # 把字符追加到缓冲区
        if ch in "。！？；!?;\n":                                # 遇到句末标点或换行
            if buf.strip():                                     # 缓冲区非空白时
                parts.append(buf)                               # 作为一个句子保存
            buf = ""                                            # 清空缓冲区
    if buf.strip():                                             # 处理末尾残留文本
        parts.append(buf)                                       # 追加为最后一个句子
    return parts                                                # 返回句子列表


def make_chunks(docs, chunk_size=None, overlap=None):           # 定义分块主函数
    """基于句子边界贪心累积成块，超出 chunk_size 即切分，保留 overlap 字符重叠。"""
    chunk_size = chunk_size or config.CHUNK_SIZE                # 取配置中的块大小
    overlap = overlap if overlap is not None else config.CHUNK_OVERLAP  # 取配置中的重叠长度
    chunks = []                                                 # 初始化块列表
    for doc in docs:                                            # 遍历每个页面文档
        sents = _split_sentences(doc["text"])                   # 切分为句子序列
        cur = ""                                                # 当前块缓冲区
        for sent in sents:                                      # 遍历句子
            if len(cur) + len(sent) <= chunk_size:              # 若加入后不超过块上限
                cur += sent                                     # 直接追加到当前块
                continue                                        # 处理下一个句子
            if cur.strip():                                     # 当前块非空时先落盘
                chunks.append(_mk_chunk(doc, cur, len(chunks)))  # 保存当前块
            tail = cur[-overlap:] if overlap > 0 and len(cur) > overlap else ""  # 取尾部重叠文本
            cur = tail + sent                                   # 新块以重叠文本加当前句子起始
        if cur.strip():                                         # 处理残留的最后一个块
            chunks.append(_mk_chunk(doc, cur, len(chunks)))     # 保存该块
    return chunks                                               # 返回全部文本块


def _mk_chunk(doc, text, idx):                                  # 定义单块构造函数
    """构造带元信息的块字典；ID 由来源、页码与内容共同决定，保证可复现。"""
    text = text.strip()                                         # 去除首尾空白
    raw = "%s|%d|%s" % (doc["source"], doc["page"], text[:80])   # 拼接可用于生成稳定 ID 的原始串
    cid = hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]     # 取 MD5 前 12 位作为块 ID
    return {"id": cid,                                          # 块唯一标识
            "text": text,                                       # 块正文
            "source": doc["source"],                            # 来源文件短名
            "page": doc["page"],                                # 来源页码
            "seq": idx,                                         # 全局序号（便于稳定排序）
            "n_chars": len(text)}                               # 块字符数（便于过滤与统计）


def build_or_load_chunks(docs, use_cache=True):                 # 定义带缓存的分块入口
    """优先读取 data/chunks.json 缓存；缓存缺失或强制刷新时重新分块并落盘。"""
    config.ensure_dirs()                                        # 确保数据目录存在
    path = config.CHUNK_CACHE_FILE                              # 取分块缓存路径
    if use_cache and os.path.isfile(path):                      # 缓存开关开启且文件存在
        try:                                                    # 尝试读取缓存
            with open(path, "r", encoding="utf-8") as f:        # 以 UTF-8 打开缓存文件
                chunks = json.load(f)                           # 反序列化分块列表
            LOGGER.info("命中分块缓存：%d 块", len(chunks))      # 输出缓存命中日志
            return chunks                                       # 返回缓存结果
        except Exception as e:                                  # 缓存损坏时容错
            LOGGER.warning("分块缓存损坏，将重建：%r", e)         # 输出告警
    chunks = make_chunks(docs)                                  # 重新执行分块
    chunks = [c for c in chunks if c["n_chars"] >= config.MIN_CHUNK_LEN]  # 过滤过短的噪音块
    try:                                                        # 尝试写入缓存
        with open(path, "w", encoding="utf-8") as f:            # 以 UTF-8 覆盖写入
            json.dump(chunks, f, ensure_ascii=False)            # 序列化分块结果
        LOGGER.info("分块完成并写入缓存：%d 块", len(chunks))     # 输出完成日志
    except Exception as e:                                      # 写入失败时容错
        LOGGER.warning("分块缓存写入失败：%r", e)                # 仅告警，不影响本次运行
    return chunks                                               # 返回分块结果
