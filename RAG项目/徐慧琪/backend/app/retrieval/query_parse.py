"""从问句里抽条款号。

存在的理由：FR-3.2 要求"输入第X条必须返回该条"，AC-2 定的门槛是 100%。
纯向量检索做不到 100%——问句里的数字对 embedding 几乎无区分度。所以这一路
必须走精确查库，而它的入口就是把条号从自然语言里抽出来。

本模块不查库、不依赖任何服务，因此它的测试不需要起 MySQL。
"""
from __future__ import annotations

import re

from app.ingest.cn_num import cn2int

# 「第X条」的 X：阿拉伯（半角/全角）或中文数字。允许中间夹空格，是从网页
# 复制来的问句常见形态。不写成 \d+ 是因为中文数字占多数，而「十」这种
# 无前缀数字的写法必须交给 cn2int 处理。
# 字符类必须含「千」——cn2int 的千分支（ingest/cn_num.py）早就实现了，是这里
# 没把「千」喂给它：民法典 1000~1260 条正文写法全是千字头（如「第一千零四十条」），
# 漏千会让这 261 条整段抽不出号、静默不置顶且不报错；口径对齐 structure.py 的 CN
ARTICLE_NO_RE = re.compile(r"第\s*([0-9０-９一二三四五六七八九十百千零两]+)\s*条")

# 全角数字到半角。这步转换对本项目跑的 CPython 3 是冗余的——int() 与 str.isdigit()
# 原生认全角（'５８４'.isdigit() → True、int('５８４') → 584），删掉这行测试也不会红，
# 之前的注释写「不转会让 int() 抛错」是错的。保留的动机不是守当前行为，而是把
# 「全角也算数字」显式写在代码里：换语言或换只认 ASCII 的解析库时，这一层不必重新发现
FULLWIDTH_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")


def extract_article_nos(text: str) -> list[int]:
    """按出现顺序返回问句里的条号，重复的只留第一次。

    重复去掉是因为下游按顺序置顶，同一个条号置顶两次会挤掉别的条。
    """
    seen: list[int] = []
    for match in ARTICLE_NO_RE.finditer(text):
        raw = match.group(1).translate(FULLWIDTH_DIGITS)
        number = int(raw) if raw.isdigit() else cn2int(raw)
        # cn2int 对无法识别的串会返回 0；条号从 1 起，0 必然是解析失败
        if number > 0 and number not in seen:
            seen.append(number)
    return seen
