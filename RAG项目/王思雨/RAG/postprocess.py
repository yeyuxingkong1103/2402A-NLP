# -*- coding: utf-8 -*-
"""后处理与校验模块：答案的文本清洗与合规校验，被 rag.py 引用。"""

import re                                     # 导入 re，用于后处理的正则替换
from logger import get_logger                 # 导入日志工具，用于记录清洗过程

logger = get_logger("postprocess")            # 创建本模块的 logger 实例

MAX_ANSWER_LEN = 8000                         # 答案最大字符数，超过则截断
TRUNCATE_SUFFIX = "（回答过长已截断）"          # 截断后追加的提示语
NO_EVIDENCE_TEXT = "标准文档中没有找到相关条款"  # 无依据的标准话术，用于校验标记
BANNED_WORDS = ("绝对安全", "保证不出问题")     # 违规词表，出现即判定不通过

# 后处理用的正则：代码块标记、连续空行、行尾空格、中文标点后空格、全角空格
RE_CODE_FENCE = re.compile(r"```[a-zA-Z]*\n?")                # markdown 代码块标记
RE_BLANK_LINES = re.compile(r"\n{3,}")                        # 3 个及以上连续换行
RE_TRAIL_SPACE = re.compile(r"[ \t]+$", re.MULTILINE)         # 行尾多余空格
RE_SPACE_AFTER_PUNCT = re.compile(r"([，。；：、！？）】”])[ \t]+")   # 中文标点后的空格
RE_FULLWIDTH_SPACE = re.compile("　")                     # 全角空格


def postprocess(text: str) -> str:
    """后处理：清理代码块标记、多余空行、行尾空格、中文标点后空格与全角空格，超长截断。"""
    if not text:                                     # 空文本
        return ""                                    # 直接返回空串
    cleaned = RE_CODE_FENCE.sub("", text)            # 去掉 markdown 代码块标记
    cleaned = RE_FULLWIDTH_SPACE.sub(" ", cleaned)   # 全角空格转成半角空格
    cleaned = RE_TRAIL_SPACE.sub("", cleaned)        # 去掉每行行尾的空格
    cleaned = RE_SPACE_AFTER_PUNCT.sub(r"\1", cleaned)   # 删掉中文标点后的多余空格
    cleaned = RE_BLANK_LINES.sub("\n\n", cleaned)    # 连续 3 个以上空行合并为 2 个
    cleaned = cleaned.strip()                        # 去掉首尾空白
    if len(cleaned) > MAX_ANSWER_LEN:                # 超过长度上限
        keep = MAX_ANSWER_LEN - len(TRUNCATE_SUFFIX)   # 预留提示语长度，保证总量不超限
        cleaned = cleaned[:keep] + TRUNCATE_SUFFIX     # 截断并追加提示语
        logger.warning("答案超过 %d 字，已截断", MAX_ANSWER_LEN)   # 记录告警
    return cleaned                                   # 返回清洗后的文本


def validate_answer(text: str) -> dict:
    """校验答案：长度、违规词、是否有依据，返回校验结果字典。"""
    content = (text or "").strip()                   # 去掉首尾空白
    is_no_evidence = NO_EVIDENCE_TEXT in content     # 是否声明了标准中没有依据
    if not content:                                  # 长度为 0
        return {"ok": False, "reason": "答案为空", "is_no_evidence": is_no_evidence}   # 不通过
    if len(content) > MAX_ANSWER_LEN:                # 超过长度上限
        return {"ok": False, "reason": f"答案超过 {MAX_ANSWER_LEN} 字",
                "is_no_evidence": is_no_evidence}    # 不通过
    for word in BANNED_WORDS:                        # 逐个检查违规词
        if word in content:                          # 命中违规词
            return {"ok": False, "reason": f"包含违规表述：{word}",
                    "is_no_evidence": is_no_evidence}   # 不通过
    return {"ok": True, "reason": "校验通过", "is_no_evidence": is_no_evidence}   # 通过
