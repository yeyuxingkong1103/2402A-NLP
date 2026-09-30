# -*- coding: utf-8 -*-
"""文本清洗与语义切分模块：被 ingest.py 调用，只做纯文本处理，不涉及任何入库操作。"""

import re                                     # 导入 re，用于正则匹配页码与关键词

# 整行等于这些词时直接判定为水印（例如整页只有"标准"两个字的印章）
WATERMARK_EXACT = ("标准", "内部", "水印", "受控", "仅供", "内部资料")   # 水印整行词表

# 短行里包含这些词就判定为水印，用多字词组避免误删正文
WATERMARK_KEYWORDS = ("受控", "仅供", "内部资料", "禁止复制", "标准下载", "标准分享", "内部使用", "无水印")  # 水印关键词表

WATERMARK_MAX_LEN = 10                        # 判定为水印的行最大长度，超过则视为正文
EDGE_LINES = 3                                # 每页首尾各取 3 行视为页眉页脚区
REPEAT_RATIO = 0.5                            # 出现在一半以上页面才算重复页眉页脚
REPEAT_MIN_PAGES = 2                          # 至少出现在 2 个页面才算重复
REPEAT_MAX_LEN = 30                           # 只有短行才可能是页眉页脚，长行是正文

# 页码行的正则：纯数字、第X页、带横线的数字、罗马数字
PAGE_NUMBER_PATTERNS = (                      # 页码正则元组开始
    re.compile(r"^\d{1,4}$"),                 # 形如 12
    re.compile(r"^第\s*\d{1,4}\s*页$"),        # 形如 第 12 页
    re.compile(r"^[-—–\s]*\d{1,4}[-—–\s]*$"),  # 形如 -12- 或 — 12 —
    re.compile(r"^[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+$"),       # 形如 Ⅰ、Ⅻ 的罗马数字页码
    re.compile(r"^[IVXivx]{1,6}$"),           # 形如 IV、xii 的拉丁罗马数字页码
)                                             # 页码正则元组结束

# 句子结束符：按这些符号切句，切完把符号保留在句尾
SENTENCE_ENDINGS = "。！？；!?;"                # 中英文句子结束符


def is_page_number(line: str) -> bool:        # 判断一行是不是页码
    """判断一行文本是否为页码，用于页眉页脚区的清理。"""
    text = line.strip()                       # 去掉首尾空白
    if not text:                              # 空行不算页码
        return False                          # 直接返回否
    return any(p.match(text) for p in PAGE_NUMBER_PATTERNS)   # 命中任一页码正则即为页码


def is_watermark(line: str) -> bool:          # 判断一行是不是水印
    """判断一行文本是否为水印：整行等于水印词，或短行里包含水印关键词。"""
    text = line.strip()                       # 去掉首尾空白
    if not text:                              # 空行不算水印
        return False                          # 直接返回否
    if text in WATERMARK_EXACT:               # 整行刚好等于水印词
        return True                           # 判定为水印
    if len(text) > WATERMARK_MAX_LEN:         # 太长的行一定是正文
        return False                          # 直接返回否
    return any(kw in text for kw in WATERMARK_KEYWORDS)   # 短行里含关键词则判定为水印


def find_repeated_lines(pages: list) -> set:  # 找出跨页重复的页眉页脚行
    """统计所有页面首尾各 3 行，找出在多页重复出现的短行，作为页眉页脚。"""
    counter = {}                              # 用字典统计每行出现的页数
    for text in pages:                        # 逐页遍历文本
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]   # 取出该页所有非空行
        if not lines:                         # 该页没有文本（扫描页）
            continue                          # 跳过，不做统计
        edge = lines[:EDGE_LINES] + lines[-EDGE_LINES:]   # 只取首尾各 3 行作为候选
        for line in set(edge):                # 同一页重复出现的行只算一次
            if len(line) > REPEAT_MAX_LEN:    # 过长的行是正文，不是页眉页脚
                continue                      # 跳过
            if is_page_number(line):          # 页码单独按页码规则处理
                continue                      # 跳过，不混进重复行
            counter[line] = counter.get(line, 0) + 1   # 该行出现页数加一
    total = max(len(pages), 1)                # 总页数，至少为 1，避免除零
    threshold = max(REPEAT_MIN_PAGES, int(total * REPEAT_RATIO))   # 计算重复阈值
    return {line for line, cnt in counter.items() if cnt >= threshold}   # 返回重复行集合


def remove_watermark_and_header_footer(page_text: str, page_no: int, total_pages: int,
                                       repeated_lines: set = None) -> str:   # 清洗单页文本
    """清洗单页文本：去掉跨页重复的页眉页脚、页码行、水印短行，返回清洗后的文本。"""
    if not page_text:                         # 文本为空时
        return ""                             # 直接返回空字符串
    repeated = repeated_lines or set()        # 没有传入重复行时用空集合
    lines = page_text.splitlines()            # 按行拆开
    kept = []                                 # 保存保留下来的行
    for idx, raw in enumerate(lines):         # 逐行遍历，idx 为行号
        line = raw.strip()                    # 去掉首尾空白
        if not line:                          # 空行
            kept.append("")                   # 保留空行以维持段落结构
            continue                          # 处理下一行
        if line in repeated:                  # 命中跨页重复的页眉页脚
            continue                          # 丢弃该行
        is_edge = idx < EDGE_LINES or idx >= len(lines) - EDGE_LINES   # 是否位于首尾页眉页脚区
        if is_edge and is_page_number(line):  # 页眉页脚区里的页码行
            continue                          # 丢弃该行
        if is_watermark(line):                # 命中水印规则
            continue                          # 丢弃该行
        kept.append(line)                     # 其余内容保留
    return "\n".join(kept).strip()            # 重新拼成文本并去掉首尾空白


def _split_paragraphs(text: str) -> list:     # 内部函数：按段落切分
    """优先按空行（\\n\\n）切分段落，没有空行时按单换行切分。"""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")   # 统一换行符
    if "\n\n" in normalized:                  # 存在空行分隔
        parts = normalized.split("\n\n")      # 按空行切段
    else:                                     # 没有空行分隔（PDF 常见）
        parts = normalized.split("\n")        # 退化为按单换行切，后续靠合并还原段落
    return [p.strip() for p in parts if p.strip()]   # 去掉空白段后返回


def _split_sentences(text: str) -> list:      # 内部函数：按句子切分
    """按中文/英文句末标点切句，标点保留在句尾。"""
    sentences = []                            # 保存切出的句子
    buf = ""                                  # 当前句子缓冲区
    for ch in text:                           # 逐字符扫描
        buf += ch                             # 字符先并入缓冲区
        if ch in SENTENCE_ENDINGS:            # 遇到句末标点
            sentences.append(buf.strip())     # 把缓冲区作为一句存下
            buf = ""                          # 清空缓冲区
    if buf.strip():                           # 结尾还有没标点的残余
        sentences.append(buf.strip())         # 也作为一句存下
    return [s for s in sentences if s]        # 过滤空串后返回


def _merge_short(units: list, max_len: int, min_len: int) -> list:   # 内部函数：合并短块
    """把过短的相邻块合并到不小于 min_len，同时保证单块不超过 max_len。"""
    chunks = []                               # 保存合并后的块
    buf = ""                                  # 当前正在拼装的块
    for unit in units:                        # 逐个处理待合并单元
        if not buf:                           # 当前块还是空的
            buf = unit                        # 直接放入
            if len(buf) >= min_len:           # 已经够长
                chunks.append(buf)            # 收进结果
                buf = ""                      # 清空继续
            continue                          # 处理下一个单元
        if len(buf) + len(unit) <= max_len:   # 合并后不超长
            buf += unit                       # 继续拼接
            if len(buf) >= min_len:           # 拼够了最小长度
                chunks.append(buf)            # 收进结果
                buf = ""                      # 清空继续
        else:                                 # 合并会超长
            chunks.append(buf)                # 先把当前块收下
            buf = unit                        # 新单元作为下一块的开头
    if buf:                                   # 循环结束后还有残余
        chunks.append(buf)                    # 收进结果
    return chunks                             # 返回合并结果


def _fix_last_chunk(chunks: list, max_len: int, min_len: int) -> list:   # 内部函数：修正尾块
    """若最后一块过短，则与倒数第二块合并或对半重分，保证每块不小于 min_len。"""
    if len(chunks) < 2:                       # 只有一块时无需处理
        return chunks                         # 直接返回
    if len(chunks[-1]) >= min_len:            # 尾块长度达标
        return chunks                         # 直接返回
    merged = chunks[-2] + chunks[-1]          # 把尾块并回前一块
    if len(merged) <= max_len:                # 合并后不超长
        return chunks[:-2] + [merged]         # 直接合并返回
    if len(merged) >= 2 * min_len:            # 合并后超长但对半切仍满足最小长度
        mid = len(merged) // 2                # 取中间位置
        return chunks[:-2] + [merged[:mid], merged[mid:]]   # 对半重分成两块
    return chunks                             # 无法合并时保持原样


def semantic_chunk(text: str, max_len: int = 500, min_len: int = 80) -> list:   # 语义切分入口
    """语义切分：先按段落切，段落过长再按句子切，最后合并过短块并限制单块长度。"""
    if not text or not text.strip():          # 文本为空时
        return []                             # 直接返回空列表
    paras = _split_paragraphs(text)           # 第一步：按段落切分
    units = []                                # 保存切分后的最小单元
    for para in paras:                        # 逐个段落处理
        if len(para) <= max_len:              # 段落本身不超长
            units.append(para)                # 直接作为一个单元
        else:                                 # 段落超长
            units.extend(_split_sentences(para))   # 再用句子切分
    units = [u for u in units if u]           # 过滤掉空单元
    if not units:                             # 没有可用单元
        return []                             # 返回空列表
    chunks = _merge_short(units, max_len, min_len)        # 第二步：合并过短块
    chunks = [c[:max_len] for c in chunks]                # 第三步：硬截断超长块
    chunks = _fix_last_chunk(chunks, max_len, min_len)    # 第四步：修正过短的尾块
    return chunks                             # 返回最终切分结果
