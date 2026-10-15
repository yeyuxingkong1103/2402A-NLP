# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-金融问答系统部署
# 模块职责：加载金融年报语料（JSONL）并切分为文本块；结果可缓存到数据卷以加速容器重启

import os                                                       # 导入 os 用于目录遍历与路径处理
import re                                                       # 导入正则用于文本清洗
import json                                                     # 导入 json 用于解析语料与缓存
import unicodedata                                              # 导入 Unicode 归一化用于修复 PDF 部首字符
import hashlib                                                  # 导入哈希库用于生成稳定块 ID
import config                                                   # 导入本工单配置模块
from logging_setup import LOGGER                                # 导入统一日志对象

# PDF 字体提取常把汉字渲染为部首字符，NFKC 修不了的少数需要显式映射
_RADICAL_FIX = {"⻛": "风", "⻓": "长"}             # 部首补充区字符 → 标准汉字
_PUNCT_RE = re.compile(r"[ \t]+")                               # 连续空白压缩正则
_SENT_RE = re.compile(r"(?<=[。！？；!?;])")                     # 句末标点切分点


def clean_text(text):                                           # 定义文本清洗函数
    """归一化字符、压缩空白、去除控制字符。"""
    text = unicodedata.normalize("NFKC", text)                  # NFKC 归一化：修复康熙部首、统一全半角
    for k, v in _RADICAL_FIX.items():                           # 遍历需显式替换的部首字符
        text = text.replace(k, v)                               # 替换为标准汉字
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)    # 去除不可见控制字符
    text = _PUNCT_RE.sub(" ", text)                             # 压缩连续空白
    return text.strip()                                         # 去除首尾空白后返回


def _pretty_source(filename):                                   # 定义来源短名生成函数
    """由 ccf 文件名生成唯一短名，如『招商银行(600036)_2019』。"""
    base = os.path.splitext(filename)[0]                        # 去掉扩展名
    parts = base.split("__")                                    # 按双下划线切分字段
    if len(parts) >= 5:                                         # 符合『日期__公司__代码__简称__年度__类型』结构
        abbr = parts[3] if len(parts[3]) <= 8 else parts[1][:6]  # 优先用简称，过长则退回公司名前 6 字
        return "%s(%s)_%s" % (abbr, parts[2], parts[4])         # 用股票代码保证唯一性
    return base[:20]                                            # 非标准命名时截断返回


def load_corpus(corpus_dir=None, max_chars=None):               # 定义语料加载函数
    """解析语料目录下的 JSONL 文件，按页聚合文本，返回文档列表。"""
    corpus_dir = corpus_dir or config.CORPUS_DIR                # 取语料目录
    max_chars = max_chars or 600000                             # 字符预算（CPU 推理下的规模控制）
    docs = []                                                   # 文档列表：{source, page, text}
    if not os.path.isdir(corpus_dir):                           # 容错：目录不存在
        LOGGER.error("语料目录不存在：%s", corpus_dir)           # 记录错误
        return docs                                             # 返回空列表
    names = sorted(n for n in os.listdir(corpus_dir) if n.lower().endswith(".txt"))  # 列出全部 txt
    for name in names:                                          # 逐个文件解析
        path = os.path.join(corpus_dir, name)                   # 拼接完整路径
        pages = {}                                              # 页码 -> 行文本列表
        try:                                                    # 尝试读取
            with open(path, "r", encoding="utf-8", errors="ignore") as f:  # UTF-8 打开并忽略坏字节
                for line in f:                                  # 逐行解析 JSONL
                    line = line.strip()                         # 去除空白
                    if not line:                                # 跳过空行
                        continue                                # 继续下一行
                    try:                                        # 尝试解析单行
                        rec = json.loads(line)                  # 反序列化
                    except Exception:                           # 坏行容错
                        continue                                # 跳过该行
                    inner = rec.get("inside")                   # 取出正文
                    if not inner:                               # 正文为空则跳过
                        continue                                # 继续下一行
                    pages.setdefault(int(rec.get("page", 0)), []).append(inner)  # 追加到对应页
        except Exception as e:                                  # 文件级失败
            LOGGER.error("读取语料失败 %s：%r", name, e)         # 记录错误
            continue                                            # 跳过该文件
        src = _pretty_source(name)                              # 生成来源短名
        for pg in sorted(pages.keys()):                         # 按页码顺序遍历
            text = clean_text("\n".join(pages[pg]))             # 合并同页文本并清洗
            if len(text) >= 20:                                 # 过滤过短页面（封面/空白）
                docs.append({"source": src, "page": pg, "text": text})  # 加入文档列表
    docs = limit_corpus(docs, max_chars)                        # 按字符预算采样控制规模
    LOGGER.info("语料加载完成：%d 个页面，约 %d 字", len(docs),   # 输出统计日志
                sum(len(d["text"]) for d in docs))              # 输出总字符数
    return docs                                                 # 返回文档列表


def limit_corpus(docs, max_chars):                              # 定义语料采样函数
    """按字符预算采样：每文件优先保留头部叙述页，其余均匀抽样。"""
    total = sum(len(d["text"]) for d in docs)                   # 统计总字符数
    if total <= max_chars:                                      # 未超预算则原样返回
        return docs                                             # 直接返回
    groups = {}                                                 # 按来源分组
    for d in docs:                                              # 遍历文档
        groups.setdefault(d["source"], []).append(d)            # 归入对应来源
    picked = []                                                 # 采样结果
    for _src, items in groups.items():                          # 遍历每个来源
        items = sorted(items, key=lambda x: x["page"])          # 按页码升序
        quota = int(max_chars * sum(len(x["text"]) for x in items) / total)  # 按占比分配额度
        head = items[:40]                                       # 头部 40 页叙述性内容优先保留
        chosen = list(head)                                     # 已选集合先放入头部
        rest = items[40:]                                       # 其余页面
        if rest:                                                # 若存在剩余页
            per_page = sum(len(x["text"]) for x in rest) / float(len(rest))  # 剩余页平均字符数
            take = max(1, int(max(0, quota - sum(len(x["text"]) for x in head)) / max(1.0, per_page)))  # 需抽页数
            step = max(1, len(rest) // take)                    # 抽样步长
            chosen += [d for i, d in enumerate(rest) if i % step == 0]  # 均匀抽样补充
        picked.extend(chosen)                                   # 汇总该来源结果
    return sorted(picked, key=lambda x: (x["source"], x["page"]))  # 排序返回


def make_chunks(docs):                                          # 定义分块函数
    """按句边界累积成块，块大小接近 CHUNK_SIZE，相邻块保留重叠。"""
    chunks = []                                                 # 块列表
    for doc in docs:                                            # 遍历文档
        sents = [s for s in _SENT_RE.split(doc["text"]) if s.strip()]  # 按句末标点切分
        cur = ""                                                # 当前块缓冲区
        for sent in sents:                                      # 遍历句子
            if len(cur) + len(sent) <= config.CHUNK_SIZE:       # 未超块上限则继续累积
                cur += sent                                     # 追加到当前块
                continue                                        # 处理下一句
            if cur.strip():                                     # 当前块非空则落盘
                chunks.append(_mk_chunk(doc, cur))              # 保存当前块
            cur = (cur[-config.CHUNK_OVERLAP:] if len(cur) > config.CHUNK_OVERLAP else "") + sent  # 新块带重叠
        if cur.strip():                                         # 处理残留块
            chunks.append(_mk_chunk(doc, cur))                  # 保存该块
    return chunks                                               # 返回全部块


def _mk_chunk(doc, text):                                       # 定义单块构造函数
    """构造带来源定位信息的块字典，ID 由内容哈希决定以保证可复现。"""
    text = text.strip()                                         # 去除首尾空白
    cid = hashlib.md5(("%s|%d|%s" % (doc["source"], doc["page"], text[:80]))  # 拼装稳定标识源串
                      .encode("utf-8")).hexdigest()[:12]        # 取哈希前 12 位
    return {"id": cid, "text": text, "source": doc["source"],    # 返回块字典：ID、正文、来源
            "page": doc["page"], "n_chars": len(text)}          # 页码与字符数
