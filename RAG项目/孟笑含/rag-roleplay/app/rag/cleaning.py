# -*- coding: utf-8 -*-
"""数据清洗：入库前清理文本（控制字符、乱码、多余空行、符号垃圾行）。"""
import re
# 解析：正则模块（匹配控制字符与垃圾行）

# 控制字符与常见乱码替换符
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f�]")
# 解析：控制字符（除换行/制表）与乱码替换符的正则——提取文本中的脏字符
# 纯符号行（无中文、字母、数字）视为垃圾
_GARBAGE_LINE = re.compile(r"^[^一-鿿A-Za-z0-9]+$")
# 解析：整行都是符号（如 ====、---）判定为垃圾行——正文行必含中文/字母/数字


def clean_text(text: str) -> str:
    """清洗文本：去控制字符/乱码 → 逐行去符号垃圾行 → 折叠多余空行。"""
    text = _CONTROL.sub("", text)
    # 解析：第一步——删除所有控制字符与乱码替换符
    lines = []
    # 解析：收集清洗后的行
    for line in text.splitlines():
        # 解析：逐行处理
        line = line.strip()
        # 解析：去掉行首尾空白
        if not line:
            # 解析：空行
            lines.append("")
            # 解析：保留空行占位（后面统一折叠）
            continue
            # 解析：继续下一行
        if _GARBAGE_LINE.match(line):
            # 解析：纯符号垃圾行
            continue
            # 解析：丢弃不保留
        lines.append(line)
        # 解析：正常行保留
    # 折叠连续空行为单个空行
    collapsed, prev_blank = [], False
    # 解析：结果列表与前一行是否空行标记
    for line in lines:
        # 解析：二次遍历折叠空行
        if line == "":
            # 解析：当前是空行
            if prev_blank:
                # 解析：上一行也是空行
                continue
                # 解析：跳过（连续空行只留一个）
            prev_blank = True
            # 解析：标记当前为空行
        else:
            # 解析：当前是内容行
            prev_blank = False
            # 解析：重置空行标记
        collapsed.append(line)
        # 解析：保留本行
    return "\n".join(collapsed).strip()
    # 解析：拼接并去掉首尾空白，返回清洗后文本


def make_summary(text: str, max_len: int = 80) -> str:
    """抽取式摘要：取开头 max_len 字符并压成单行。"""
    text = " ".join(text.split())
    # 解析：所有空白（含换行）压成单空格——摘要必须是单行
    if len(text) <= max_len:
        # 解析：文本不长于上限
        return text
        # 解析：直接返回全文
    return text[: max_len - 1] + "…"
    # 解析：截取前 max_len-1 字符并加省略号（中文用 … 而非 ...）
