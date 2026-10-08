"""D1：词典白名单式的 CJK 词内空格合并（归入报告里的「转换类」）。

背景：PDF 换行会让 MinerU 在中文字词中间切出一个空格（实测：`血 压计`、
`医用电 子血压计`、`急 性并发症`），直接进检索会污染召回。

判据（把方案 C 的误合并风险压掉）：
    取空格两侧的 CJK 连续段，拼起来交给 jieba 分词；**只有当存在一个词
    横跨该空格**时才合并。这样 `办公室 国家基层…`、`中心 中国医学科学院…`
    这类合法的机构名分隔不会被误并——jieba 会在这些位置正好断词。

jieba 缺失时降级为「不合并」并显式告警，绝不静默跳过（FR-008）。
"""

from __future__ import annotations

import re

from .errors import CleanError, EXIT_DEP_MISSING

# 空格两侧的 CJK 连续段（含中文标点的边界不参与）
CJK = r"㐀-䶿一-鿿"
_SPACE_BETWEEN_CJK = re.compile(r"(?<=[%s])([ 　]+)(?=[%s])" % (CJK, CJK))
_CJK_RUN_LEFT = re.compile(r"[%s]+$" % CJK)
_CJK_RUN_RIGHT = re.compile(r"^[%s]+" % CJK)

# 交给 jieba 的上下文窗口上限（实测 10 已足够，且避免长串带来的歧义）
WINDOW = 10

_jieba = None
_jieba_warned = False


def load_jieba():
    """返回 jieba 模块；不可用时返回 None 并（仅一次）告警。"""
    global _jieba, _jieba_warned
    if _jieba is not None:
        return _jieba
    if _jieba_warned:
        return None
    try:
        import jieba  # noqa: PLC0415

        jieba.setLogLevel(60)  # 关掉 "Building prefix dict" 之类的日志
        _jieba = jieba
    except ImportError as exc:
        _jieba_warned = True
        print(
            "[告警] 未安装 jieba，D1 的「词内空格合并」已降级为不合并。\n"
            "       安装：D:/zg6_Project/9/med_rag/rag/python.exe -m pip install jieba\n"
            "       原因：%s" % exc
        )
        return None
    return _jieba


def _spans_word_boundary(jieba, left: str, right: str) -> bool:
    """拼接后是否存在一个词横跨 left/right 的接缝。"""
    junction = len(left)
    cursor = 0
    for token in jieba.cut(left + right):
        if cursor < junction < cursor + len(token):
            return True
        cursor += len(token)
    return False


def merge_cjk_spaces(text: str, opts, log) -> str:
    """合并词内空格；每次合并都通过 log 留痕。"""
    if not text or not getattr(opts, "space_merge", True):
        return text

    jieba = load_jieba()
    if jieba is None:
        return text

    def repl(match: re.Match) -> str:
        start, end = match.span(1)
        left_run = _CJK_RUN_LEFT.search(text[:start])
        right_run = _CJK_RUN_RIGHT.search(text[end:])
        if not left_run or not right_run:
            return match.group(0)

        left = left_run.group(0)[-WINDOW:]
        right = right_run.group(0)[:WINDOW]
        if not _spans_word_boundary(jieba, left, right):
            return match.group(0)

        original = text[max(0, start - len(left)) : end + len(right)]
        merged = left + right
        log(original, merged)
        return ""

    return _SPACE_BETWEEN_CJK.sub(repl, text)


# D1 验收基线：这 5 组必须稳定通过（--self-test 会跑）
BASELINE = (
    ("国家基层高血压管理办公室 国家基层高血压管理专家委员会", False),
    ("国家心血管病中心 中国医学科学院阜外医院", False),
    ("上臂式医用电子血 压计", True),
    ("经准确度验证的医用电 子血压计", True),
    ("不伴心、脑、肾急 性并发症", True),
)


def self_test() -> list[str]:
    """返回失败项描述；空列表表示全通过。"""
    jieba = load_jieba()
    if jieba is None:
        raise CleanError(EXIT_DEP_MISSING, "jieba 不可用，无法自检 D1 基线")

    failures = []
    for sample, expect_merged in BASELINE:
        merged = merge_cjk_spaces(sample, _NoOpts(), lambda a, b: None)
        changed = merged != sample
        if changed != expect_merged:
            failures.append(
                "期望 %s：%r -> %r"
                % ("合并" if expect_merged else "保持原样", sample, merged)
            )
    return failures


class _NoOpts:
    space_merge = True
