"""中文数字 → 阿拉伯数字转换。

原 legal_meta.py 并入部分：把"第九十九条之一"等中文编号规范化为
"99之1"形式的阿拉伯数字编号，供入库映射与向量化服务调用。
"""

import re


def to_arabic_number(value: str | None) -> str | None:
    """将中文数字转换为阿拉伯数字。

    规则：
    - None 或空串 → 返回 None
    - "一"→"1"、"十"→"10"、"二十三"→"23"、"一百零五"→"105"
    - "第九十九条之一" → "99之1"（提取数字并保留「之」）
    - 已经是阿拉伯数字 → 原样返回（含前导零规范化）
    - 中文与阿拉伯混写、非数字内容 → 抛 ValueError

    参数：
    - value: 中文或阿拉伯数字字符串

    返回：
    - 阿拉伯数字字符串，或 None

    异常：
    - ValueError: 无法识别的格式

    示例：
    >>> to_arabic_number(None)
    >>> to_arabic_number("")
    >>> to_arabic_number("一")
    '1'
    >>> to_arabic_number("十")
    '10'
    >>> to_arabic_number("二十三")
    '23'
    >>> to_arabic_number("一百零五")
    '105'
    >>> to_arabic_number("第九十九条之一")
    '99之1'
    >>> to_arabic_number("47")
    '47'
    >>> to_arabic_number("0047")
    '47'
    >>> to_arabic_number("99之1")
    '99之1'
    """
    # None 或空串返回 None
    if value is None or not value.strip():
        return None

    value = value.strip()

    # 提取数字部分（去除"第"、"条"等修饰词）
    # 匹配模式：提取所有数字和「之」
    cleaned = re.sub(r'[第条款项]', '', value)

    # 如果已经是纯阿拉伯数字（可能含「之」），直接规范化并返回
    if re.fullmatch(r'[\d之]+', cleaned):
        # 去除前导零
        if '之' in cleaned:
            parts = cleaned.split('之')
            normalized_parts = [part.lstrip('0') or '0' for part in parts]
            return '之'.join(normalized_parts)
        else:
            return cleaned.lstrip('0') or '0'

    # 中文数字映射表
    chinese_numerals = {
        '零': 0, '〇': 0, '一': 1, '二': 2, '两': 2, '三': 3, '四': 4,
        '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
        '十': 10, '百': 100, '千': 1000, '万': 10000
    }

    # 处理「之」分隔的情况（如"第九十九条之一"）
    if '之' in cleaned:
        parts = cleaned.split('之')
        converted_parts = []
        for part in parts:
            if not part:
                raise ValueError(f"「之」前后不能为空：{value!r}")
            # 递归转换每个部分
            converted = _convert_chinese_to_arabic(part, chinese_numerals)
            if converted is None:
                raise ValueError(f"无法识别的数字格式：{value!r}")
            converted_parts.append(converted)
        return '之'.join(converted_parts)
    else:
        # 无「之」，直接转换
        result = _convert_chinese_to_arabic(cleaned, chinese_numerals)
        if result is None:
            raise ValueError(f"无法识别的数字格式：{value!r}")
        return result


def _convert_chinese_to_arabic(text: str, numerals: dict[str, int]) -> str | None:
    """转换纯中文数字为阿拉伯数字（内部辅助函数）。

    支持：一、十、二十三、一百零五、九十九
    不支持：万以上、小数、负数
    """
    if not text:
        return None

    # 检查是否全是中文数字字符
    if not all(ch in numerals for ch in text):
        return None

    # 特殊处理：单个数字
    if len(text) == 1:
        return str(numerals[text])

    result = 0
    current = 0  # 当前累积值
    prev_unit = 1  # 上一个单位（十、百、千）

    for ch in text:
        value = numerals[ch]

        if value >= 10:  # 十、百、千、万
            if current == 0:
                current = 1  # "十"表示"一十"
            current *= value
            if value > prev_unit:
                # 遇到更大的单位，累加到结果
                result += current
                current = 0
            prev_unit = value
        elif value == 0:  # 零
            continue  # 跳过
        else:  # 一~九
            current += value

    result += current
    return str(result)
