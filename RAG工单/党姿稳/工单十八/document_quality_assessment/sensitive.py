# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-18-实现文档质量评估Skill并集成至智能体工作流
# 模块职责：敏感信息检测（手机号/邮箱/身份证，银行卡默认关闭）。关键点是输出“待审核列表”
#           而不是只给统计数字——每条命中都附带出现位置与上下文，供人工判断。

import re                                                          # 正则匹配

# 手机号：中国大陆 1[3-9] 开头 11 位；用前后向否定环视避免把长串数字截断后误报
RE_MOBILE = re.compile(r"(?<!\d)(1[3-9]\d{9})(?!\d)")
# 邮箱：常见写法，域名段限制长度，避免把 obfuscated 文本整段吞掉
RE_EMAIL = re.compile(r"(?<![A-Za-z0-9._%+-])([A-Za-z0-9._%+-]{1,64}"
                      r"@[A-Za-z0-9.-]{1,63}\.[A-Za-z]{2,10})(?![A-Za-z0-9.-])")
# 身份证：18 位（末位可为 X），再做校验位验证
RE_ID18 = re.compile(r"(?<!\d)(\d{17}[\dXx])(?!\d)")
# 银行卡：16~19 位连续数字，需过 Luhn 校验，默认关闭
RE_BANK = re.compile(r"(?<!\d)(\d{16,19})(?!\d)")

_ID_WEIGHT = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]  # 身份证加权因子
_ID_CHECK = "10X98765432"                                          # 校验位映射表


def valid_id18(value):
    """身份证 18 位校验位验证，用于过滤纯数字的误报。"""
    if not value or len(value) != 18:
        return False
    if not value[:17].isdigit():                                   # 前 17 位必须是数字
        return False
    total = sum(int(value[i]) * _ID_WEIGHT[i] for i in range(17))  # 加权和
    return _ID_CHECK[total % 11] == value[17].upper()              # 比对校验位


def valid_luhn(value):
    """银行卡 Luhn 校验。"""
    digits = [int(c) for c in value]                               # 逐位数字
    odd = digits[-1::-2]                                           # 从右往左奇数位
    even = digits[-2::-2]                                          # 偶数位
    total = sum(odd) + sum(sum(divmod(d * 2, 10)) for d in even)   # 偶数位加倍后拆位求和
    return total % 10 == 0


def _excluded(value, patterns):
    """命中值若匹配排除正则，则视为误报丢弃（如全零串）。"""
    for p in patterns or []:
        try:
            if re.search(p, value):
                return True
        except re.error:                                           # 配置写错的正则忽略
            continue
    return False


def _context(text, start, end, width):
    """截取命中位置前后 width 个字符作为上下文，并标注命中片段。"""
    left = max(0, start - width)                                   # 左边界
    right = min(len(text), end + width)                            # 右边界
    ctx = text[left:right].replace("\n", " ").replace("\r", " ")   # 单行化
    return re.sub(r"\s{2,}", " ", ctx).strip()                     # 合并多余空白


def detect(text, cfg):
    """检测文本中的敏感信息，返回命中列表（每条含类型、值、位置、上下文）。"""
    if not text:
        return []
    enabled = cfg.get("sensitive.enabled_types", {}) or {}         # 各类型开关
    width = cfg.get("sensitive.context_chars", 30)                 # 上下文宽度
    cap = cfg.get("sensitive.max_hits_per_type", 200)              # 单类型条数上限
    excludes = cfg.get("sensitive.exclude_patterns", [])           # 排除正则
    rules = []                                                     # 生效的规则
    if enabled.get("mobile", True):
        rules.append(("手机号", RE_MOBILE, None))                   # 手机号
    if enabled.get("email", True):
        rules.append(("邮箱", RE_EMAIL, None))                     # 邮箱
    if enabled.get("id_card", True):
        rules.append(("身份证号", RE_ID18, valid_id18))             # 身份证（需校验位通过）
    if enabled.get("bank_card", False):
        rules.append(("银行卡号", RE_BANK, valid_luhn))             # 银行卡（默认关闭）

    hits, counts = [], {}                                          # 命中与各类型计数
    for label, pattern, validator in rules:                        # 逐规则扫描
        for m in pattern.finditer(text):                           # 逐个匹配
            if counts.get(label, 0) >= cap:                        # 超过上限就停
                break
            value = m.group(1)                                     # 命中值
            if validator is not None and not validator(value):     # 校验不通过 → 误报
                continue
            if _excluded(value, excludes):                         # 排除规则命中
                continue
            counts[label] = counts.get(label, 0) + 1               # 计数
            hits.append({"type": label,                            # 类型
                         "value": value,                           # 命中值
                         "char_offset": m.start(),                 # 在文本中的字符偏移
                         "context": _context(text, m.start(), m.end(), width)})
    return hits
