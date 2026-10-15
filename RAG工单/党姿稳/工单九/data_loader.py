# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-Graph RAG 优化任务
# 模块职责：加载金融年报语料（JSONL 行式文本）与验收问题（PDF），统一为结构化数据

import os                                                       # 导入 os 用于目录遍历与路径处理
import re                                                       # 导入正则用于问题/答案切分与文本清洗
import json                                                     # 导入 json 用于解析语料的 JSONL 行
import unicodedata                                              # 导入 Unicode 归一化库用于修复 PDF 部首字符
import fitz                                                     # 导入 PyMuPDF（fitz）用于读取问题 PDF
import config                                                   # 导入本工单配置模块
from timing import LOGGER                                       # 导入统一日志对象

# PDF 字体提取时常把汉字渲染成康熙部首/部首补充区字符，导致与正常汉字无法匹配
# NFKC 归一化可修复大部分康熙部首；少数部首补充区字符无兼容映射，需显式补充
_RADICAL_FIX = {                                                # 显式补齐 NFKC 无法处理的部首字符
    "⻛": "风",                                         # ⻛（CJK 部首-简化风）→ 风
    "⻓": "长",                                         # ⻓（CJK 部首-简化长）→ 长
}


def _clean_text(text):                                          # 定义文本清洗函数
    """归一化字符、压缩空白、去除不可见控制字符，保留中文标点。"""
    text = unicodedata.normalize("NFKC", text)                  # NFKC 归一化：统一全半角并修复康熙部首
    for k, v in _RADICAL_FIX.items():                           # 遍历需显式替换的部首字符
        text = text.replace(k, v)                               # 逐一替换为对应标准汉字
    text = text.replace("　", " ")                          # 将全角空格替换为半角空格
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)    # 去除各类不可见控制字符
    text = re.sub(r"[ \t]+", " ", text)                         # 将连续空格/制表符压缩为单个空格
    text = re.sub(r"\n{2,}", "\n", text)                        # 将连续空行压缩为单个换行
    return text.strip()                                         # 去除首尾空白后返回


def load_corpus():                                              # 定义语料加载主函数
    """遍历语料目录下所有 txt（JSONL 格式），按页聚合文本，返回文档列表。"""
    docs = []                                                   # 初始化文档列表：每项为 {source, page, text}
    if not os.path.isdir(config.CORPUS_DIR):                    # 容错：语料目录不存在时直接返回空
        LOGGER.error("语料目录不存在：%s", config.CORPUS_DIR)    # 输出错误日志
        return docs                                             # 返回空列表
    names = sorted(n for n in os.listdir(config.CORPUS_DIR) if n.lower().endswith(".txt"))  # 列出全部 txt 并排序
    for name in names:                                          # 逐个文件解析
        path = os.path.join(config.CORPUS_DIR, name)            # 拼接文件绝对路径
        pages = {}                                              # 以页码为键聚合该页所有行文本
        try:                                                    # 尝试读取与解析
            with open(path, "r", encoding="utf-8", errors="ignore") as f:  # 以 UTF-8 打开（忽略坏字节）
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
                    pg = int(rec.get("page", 0))                # 取出页码（缺省为 0）
                    pages.setdefault(pg, []).append(inner)      # 将该行文本追加到对应页码的列表
        except Exception as e:                                  # 文件级读取失败
            LOGGER.error("读取语料失败 %s：%r", name, e)         # 记录错误日志
            continue                                            # 跳过该文件继续处理下一个
        for pg in sorted(pages.keys()):                         # 按页码顺序遍历
            text = _clean_text("\n".join(pages[pg]))            # 合并同页文本并清洗
            if len(text) < 20:                                  # 过短的页面（封面/空白）直接丢弃
                continue                                        # 跳过该页
            docs.append({"source": _pretty_source(name),        # 组装文档记录：可读的文件短名
                         "page": pg,                            # 页面页码
                         "text": text})                         # 页面正文
    LOGGER.info("语料加载完成：%d 个文件，%d 个页面", len(names), len(docs))  # 输出统计日志
    docs = limit_corpus(docs)                                   # 按资源约束采样，控制向量化规模
    return docs                                                 # 返回文档列表


def limit_corpus(docs, max_chars=None, head_pages=None):        # 定义语料采样函数
    """按字符预算限制语料规模：头部页优先保留，剩余额度在各文件内均匀抽样。"""
    max_chars = max_chars or config.CORPUS_MAX_CHARS            # 取字符预算
    head_pages = head_pages if head_pages is not None else config.CORPUS_HEAD_PAGES  # 取头部页数
    total = sum(len(d["text"]) for d in docs)                   # 统计当前总字符数
    if total <= max_chars:                                      # 未超预算则原样返回
        LOGGER.info("语料规模 %d 字，未超预算，无需采样", total)  # 输出提示
        return docs                                             # 直接返回
    groups = {}                                                 # 按来源文件分组
    for d in docs:                                              # 遍历全部文档
        groups.setdefault(d["source"], []).append(d)            # 归入对应来源的列表
    picked = []                                                 # 采样结果列表
    for src, items in groups.items():                           # 遍历每个来源文件
        items = sorted(items, key=lambda x: x["page"])          # 按页码升序，保证阅读顺序
        quota = int(max_chars * sum(len(x["text"]) for x in items) / total)  # 按字符占比分配额度
        head = items[:head_pages]                               # 头部页：叙述性内容密集，优先保留
        used = sum(len(x["text"]) for x in head)                # 统计头部已用字符数
        chosen = list(head)                                     # 已选页面（先放入头部）
        rest = items[head_pages:]                               # 其余页面
        if rest and used < quota:                               # 仍有额度且存在剩余页
            need = quota - used                                 # 还需补充的字符数
            per_page = sum(len(x["text"]) for x in rest) / float(len(rest))  # 剩余页平均字符数
            take = max(1, int(need / max(1.0, per_page)))       # 估算需要抽取的页数
            step = max(1, len(rest) // take)                    # 均匀抽样的步长
            for i, d in enumerate(rest):                        # 遍历剩余页
                if i % step == 0:                               # 按步长命中则选取
                    chosen.append(d)                            # 加入已选集合
        picked.extend(chosen)                                   # 汇总该文件的采样结果
    LOGGER.info("语料采样：%d 页 → %d 页，约 %d 字", len(docs), len(picked),  # 输出采样统计
                sum(len(d["text"]) for d in picked))            # 输出采样后字符数
    return sorted(picked, key=lambda x: (x["source"], x["page"]))  # 按来源与页码排序返回


def _pretty_source(filename):                                   # 定义文件名美化函数
    """把 ccf 长文件名转成便于展示且唯一的短名，例如『招商银行(600036)_2019』。"""
    base = os.path.splitext(filename)[0]                        # 去掉扩展名
    parts = base.split("__")                                    # ccf 文件名以双下划线分隔字段
    if len(parts) >= 5:                                         # 符合『日期__公司__代码__简称__年度__类型』结构时
        code = parts[2]                                         # 股票代码，天然唯一，用于避免同名公司混淆
        abbr = parts[3] if len(parts[3]) <= 8 else parts[1][:6]  # 优先用简称，过长则退回公司名前 6 字
        year = parts[4]                                         # 报告年度
        return "%s(%s)_%s" % (abbr, code, year)                 # 返回唯一短名
    return base[:20]                                            # 非标准命名时截断返回


def load_questions():                                           # 定义验收问题加载函数
    """从 sample_questions.pdf 解析出问题与参考答案列表。"""
    path = config.QUESTION_PDF                                  # 取问题 PDF 路径
    if not os.path.isfile(path):                                # 容错：文件不存在时返回内置兜底
        LOGGER.error("问题文件不存在：%s", path)                 # 记录错误日志
        return []                                               # 返回空列表
    try:                                                        # 尝试打开并提取文本
        doc = fitz.open(path)                                   # 打开 PDF
        raw = "\n".join(page.get_text() for page in doc)        # 拼接所有页面文本
        doc.close()                                             # 关闭文件句柄
    except Exception as e:                                      # 打开失败时容错
        LOGGER.error("问题 PDF 解析失败：%r", e)                 # 记录错误日志
        return []                                               # 返回空列表
    return _parse_qa(raw)                                       # 调用解析函数拆分问题与答案


def _parse_qa(raw):                                             # 定义问答切分函数
    """按『问题：』『答案：/参考答案：』标记把整段文本切分为 QA 列表。"""
    text = raw.replace(" ", " ")                           # 将不换行空格统一为普通空格
    text = re.sub(r"[ \t]+", " ", text)                         # 压缩连续空白
    marks = list(re.finditer(r"问题\s*[:：]", text))             # 定位所有『问题：』标记位置
    items = []                                                  # 初始化结果列表
    for i, m in enumerate(marks):                               # 遍历每个问题标记
        start = m.end()                                         # 问题正文起始位置
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)  # 问题正文结束位置（下一标记或文末）
        block = text[start:end]                                 # 截取该问题的完整区块
        qm = re.search(r"(参考答案|答案)\s*[:：]", block)        # 在该区块内定位答案标记
        if not qm:                                              # 若找不到答案标记
            continue                                            # 跳过该条，避免脏数据
        question = _clean_text(block[:qm.start()])              # 答案标记之前为问题
        answer = _strip_source_note(block[qm.end():])           # 答案标记之后为参考答案（并清理尾部标注）
        if len(question) > 3 and len(answer) > 10:              # 过滤过短的无效条目
            items.append({"id": len(items) + 1,                 # 分配递增编号
                          "question": question,                 # 问题文本
                          "reference": answer})                 # 参考答案文本
    LOGGER.info("验收问题加载完成：%d 条", len(items))           # 输出统计日志
    return items                                                # 返回问答列表


def _strip_source_note(text):                                   # 定义来源标注清理函数
    """去掉参考答案尾部形如『xxx__2019年__年度报告』的来源标注行。"""
    lines = [ln for ln in text.split("\n")                  # 逐行过滤
             if not re.search(r"__\s*\d{4}\s*年\s*__", ln)]  # 丢弃含双下划线年份标注的行
    return _clean_text("\n".join(lines))                        # 重新清洗并返回
