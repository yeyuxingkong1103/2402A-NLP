# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG项目-Embedding 模型微调任务
# 模块职责：加载金融年报语料（JSONL 行式文本）并切分为带定位信息的文本块（工单步骤 1 的数据基础）

import os                                                       # 导入 os 用于目录遍历与路径处理
import re                                                       # 导入正则用于文本清洗
import json                                                     # 导入 json 用于解析语料的 JSONL 行
import hashlib                                                  # 导入哈希库用于生成稳定的块 ID
import unicodedata                                              # 导入 Unicode 归一化库用于修复 PDF 部首字符
import config                                                   # 导入本工单配置模块

# PDF 字体提取时常把汉字渲染成康熙部首区字符，NFKC 可修复大部分，少数需显式补充映射
_RADICAL_FIX = {"⻛": "风", "⻓": "长"}                # 显式补齐 NFKC 无法处理的部首字符


def _clean_text(text):                                          # 定义文本清洗函数
    """归一化字符、压缩空白、去除不可见控制字符，保留中文标点。"""
    text = unicodedata.normalize("NFKC", text)                  # NFKC 归一化：统一全半角并修复康熙部首
    for k, v in _RADICAL_FIX.items():                           # 遍历需显式替换的部首字符
        text = text.replace(k, v)                               # 逐一替换为对应标准汉字
    text = text.replace("　", " ")                              # 将全角空格替换为半角空格
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)    # 去除各类不可见控制字符
    text = re.sub(r"[ \t]+", " ", text)                         # 将连续空格/制表符压缩为单个空格
    text = re.sub(r"\n{2,}", "\n", text)                        # 将连续空行压缩为单个换行
    return text.strip()                                         # 去除首尾空白后返回


def _pretty_source(filename):                                   # 定义文件名美化函数
    """把长文件名转成便于展示且唯一的短名，例如『招商银行(600036)_2019』。"""
    base = os.path.splitext(filename)[0]                        # 去掉扩展名
    parts = base.split("__")                                    # ccf 文件名以双下划线分隔字段
    if len(parts) >= 5:                                         # 符合『日期__公司__代码__简称__年度__类型』结构时
        code = parts[2]                                         # 股票代码，天然唯一，用于避免同名公司混淆
        abbr = parts[3] if len(parts[3]) <= 8 else parts[1][:6]  # 优先用简称，过长则退回公司名前 6 字
        return "%s(%s)_%s" % (abbr, code, parts[4])             # 返回唯一短名
    return base[:20]                                            # 非标准命名时截断返回


def limit_corpus(docs, max_chars=None, head_pages=None):        # 定义语料采样函数
    """按字符预算限制语料规模：头部页优先保留，剩余额度在各文件内均匀抽样。"""
    max_chars = max_chars or config.CORPUS_MAX_CHARS            # 取字符预算
    head_pages = head_pages if head_pages is not None else config.CORPUS_HEAD_PAGES  # 取头部页数
    total = sum(len(d["text"]) for d in docs)                   # 统计当前总字符数
    if total <= max_chars:                                      # 未超预算则原样返回
        return docs                                             # 直接返回
    groups = {}                                                 # 按来源文件分组
    for d in docs:                                              # 遍历全部文档
        groups.setdefault(d["source"], []).append(d)            # 归入对应来源的列表
    picked = []                                                 # 采样结果列表
    for _src, items in groups.items():                          # 遍历每个来源文件
        items = sorted(items, key=lambda x: x["page"])          # 按页码升序，保证阅读顺序
        quota = int(max_chars * sum(len(x["text"]) for x in items) / total)  # 按字符占比分配额度
        head = items[:head_pages]                               # 头部页：叙述性内容密集，优先保留
        used = sum(len(x["text"]) for x in head)                # 统计头部已用字符数
        chosen = list(head)                                     # 已选页面（先放入头部）
        rest = items[head_pages:]                               # 其余页面
        if rest and used < quota:                               # 仍有额度且存在剩余页
            per_page = sum(len(x["text"]) for x in rest) / float(len(rest))  # 剩余页平均字符数
            take = max(1, int((quota - used) / max(1.0, per_page)))  # 估算需要抽取的页数
            step = max(1, len(rest) // take)                    # 均匀抽样的步长
            for i, d in enumerate(rest):                        # 遍历剩余页
                if i % step == 0:                               # 按步长命中则选取
                    chosen.append(d)                            # 加入已选集合
        picked.extend(chosen)                                   # 汇总该文件的采样结果
    return sorted(picked, key=lambda x: (x["source"], x["page"]))  # 按来源与页码排序返回


def load_corpus():                                              # 定义语料加载主函数
    """遍历语料目录下所有 txt（JSONL 格式），按页聚合文本，返回文档列表。"""
    docs = []                                                   # 初始化文档列表：每项为 {source, page, text}
    if not os.path.isdir(config.CORPUS_DIR):                    # 容错：语料目录不存在时直接返回空
        print("[错误] 语料目录不存在：%s" % config.CORPUS_DIR)   # 输出错误提示
        return docs                                             # 返回空列表
    names = sorted(n for n in os.listdir(config.CORPUS_DIR)     # 列出全部 txt 并排序
                   if n.lower().endswith(".txt"))               # 仅保留 txt 文件
    for name in names:                                          # 逐个文件解析
        pages = {}                                              # 以页码为键聚合该页所有行文本
        try:                                                    # 尝试读取与解析
            with open(os.path.join(config.CORPUS_DIR, name),    # 拼接文件绝对路径
                      "r", encoding="utf-8", errors="ignore") as f:  # 以 UTF-8 打开（忽略坏字节）
                for line in f:                                  # 逐行读取 JSONL
                    line = line.strip()                         # 去除首尾空白
                    if not line:                                # 跳过空行
                        continue                                # 继续下一行
                    try:                                        # 尝试解析单行 JSON
                        rec = json.loads(line)                  # 解析为字典
                    except Exception:                           # 单行损坏时容错
                        continue                                # 跳过该行，保证整体不中断
                    inner = rec.get("inside")                   # 取出该行的正文内容
                    if not inner:                               # 内容为空则跳过
                        continue                                # 继续下一行
                    pages.setdefault(int(rec.get("page", 0)), []).append(inner)  # 按页码归类
        except Exception as e:                                  # 文件级读取失败
            print("[错误] 读取语料失败 %s：%r" % (name, e))      # 记录错误
            continue                                            # 跳过该文件继续下一个
        for pg in sorted(pages.keys()):                         # 按页码顺序遍历
            text = _clean_text("\n".join(pages[pg]))            # 合并同页文本并清洗
            if len(text) < 20:                                  # 过短的页面（封面/空白）直接丢弃
                continue                                        # 跳过该页
            docs.append({"source": _pretty_source(name),        # 可读的文件短名
                         "page": pg,                            # 页面页码
                         "text": text})                         # 页面正文
    print("[信息] 语料加载完成：%d 个文件，%d 个页面" % (len(names), len(docs)))  # 输出统计
    return limit_corpus(docs)                                   # 按资源约束采样后返回


def _split_sentences(text):                                     # 定义句子切分函数
    """按中英文句末标点切分句子，保留标点，便于在句子边界处切块。"""
    parts, buf = [], ""                                         # 初始化句子列表与缓冲区
    for ch in text:                                             # 逐字符扫描
        buf += ch                                               # 把字符追加到缓冲区
        if ch in "。！？；!?;\n":                                # 遇到句末标点或换行
            if buf.strip():                                     # 缓冲区非空白时
                parts.append(buf)                               # 作为一个句子保存
            buf = ""                                            # 清空缓冲区
    if buf.strip():                                             # 处理末尾残留文本
        parts.append(buf)                                       # 追加为最后一个句子
    return parts                                                # 返回句子列表


def _mk_chunk(doc, text, idx):                                  # 定义单块构造函数
    """构造带元信息的块字典；ID 由来源、页码与内容共同决定，保证可复现。"""
    text = text.strip()                                         # 去除首尾空白
    raw = "%s|%d|%s" % (doc["source"], doc["page"], text[:80])   # 拼接用于生成稳定 ID 的原始串
    return {"id": hashlib.md5(raw.encode("utf-8")).hexdigest()[:12],  # 取 MD5 前 12 位作为块 ID
            "text": text,                                       # 块正文
            "source": doc["source"],                            # 来源文件短名
            "page": doc["page"],                                # 来源页码
            "seq": idx,                                         # 全局序号（便于稳定排序）
            "n_chars": len(text)}                               # 块字符数（便于过滤与统计）


def make_chunks(docs, chunk_size=None, overlap=None):           # 定义分块主函数
    """基于句子边界贪心累积成块，超出 chunk_size 即切分，保留 overlap 字符重叠。"""
    chunk_size = chunk_size or config.CHUNK_SIZE                # 取配置中的块大小
    overlap = config.CHUNK_OVERLAP if overlap is None else overlap  # 取配置中的重叠长度
    chunks = []                                                 # 初始化块列表
    for doc in docs:                                            # 遍历每个页面文档
        cur = ""                                                # 当前块缓冲区
        for sent in _split_sentences(doc["text"]):              # 遍历该页切分出的句子
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


def build_or_load_chunks(use_cache=True):                       # 定义带缓存的分块入口
    """优先读取 data/chunks.json 缓存；缓存缺失时重新分块并落盘。"""
    config.ensure_dirs()                                        # 确保数据目录存在
    path = config.CHUNK_CACHE_FILE                              # 取分块缓存路径
    if use_cache and os.path.isfile(path):                      # 缓存开关开启且文件存在
        try:                                                    # 尝试读取缓存
            with open(path, "r", encoding="utf-8") as f:        # 以 UTF-8 打开缓存文件
                chunks = json.load(f)                           # 反序列化分块列表
            print("[信息] 命中分块缓存：%d 块" % len(chunks))     # 输出缓存命中提示
            return chunks                                       # 返回缓存结果
        except Exception as e:                                  # 缓存损坏时容错
            print("[警告] 分块缓存损坏，将重建：%r" % e)          # 输出告警
    chunks = make_chunks(load_corpus())                         # 加载语料并执行分块
    chunks = [c for c in chunks if c["n_chars"] >= config.MIN_CHUNK_LEN]  # 过滤过短的噪音块
    with open(path, "w", encoding="utf-8") as f:                # 以 UTF-8 覆盖写入缓存
        json.dump(chunks, f, ensure_ascii=False)                # 序列化分块结果（保留中文）
    print("[信息] 分块完成并写入缓存：%d 块" % len(chunks))       # 输出完成提示
    return chunks                                               # 返回分块结果
