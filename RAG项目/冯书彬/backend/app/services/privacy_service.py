import re
from dataclasses import dataclass

PHONE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
ID_CARD_RE = re.compile(r"(?<!\d)\d{6}(18|19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\d{3}[0-9Xx](?!\d)")
BANK_CARD_RE = re.compile(r"(?<!\d)\d{16,19}(?!\d)")


@dataclass(frozen=True)
class RedactionResult:
    text: str
    redacted_types: list[str]


def _replace_pattern(text: str, pattern: re.Pattern[str], label: str) -> tuple[str, bool]:
    # 只替换命中的敏感片段，不保留任何原始数字。
    redacted_text, count = pattern.subn(label, text)
    # 返回是否命中，便于调用方记录脱敏类型。
    return redacted_text, count > 0


def redact_pii(text: str) -> RedactionResult:
    # 脱敏服务保持纯函数行为，不读取配置、不写日志、不产生副作用。
    redacted_text = text
    # 按顺序记录类型，保证结果稳定便于测试和审计。
    redacted_types: list[str] = []

    # 先处理身份证号，避免被银行卡号规则误替换。
    redacted_text, matched_id_card = _replace_pattern(redacted_text, ID_CARD_RE, "[身份证号]")
    if matched_id_card:
        redacted_types.append("id_card")

    # 手机号长度固定且有号段约束，独立替换为固定占位符。
    redacted_text, matched_phone = _replace_pattern(redacted_text, PHONE_RE, "[手机号]")
    if matched_phone:
        redacted_types.append("phone")

    # 银行卡号最后处理，覆盖 16 到 19 位连续数字。
    redacted_text, matched_bank_card = _replace_pattern(redacted_text, BANK_CARD_RE, "[银行卡号]")
    if matched_bank_card:
        redacted_types.append("bank_card")

    return RedactionResult(text=redacted_text, redacted_types=redacted_types)
