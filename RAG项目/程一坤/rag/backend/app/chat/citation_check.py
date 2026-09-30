"""引用校验模块：检查回答中的引用是否合规（结构性校验入口）。

校验项：
1. 引用编号是否在法源清单范围内
2. 整回答至少有一处引用（零引用兜底）
3. 是否有无引用的确定性结论（不确定/概括表述放行，见词表注释）
4. 是否引用了清单外的法规名称（归一化后比对；简称/全称/全半角差异不误拦，清单外法规仍拦截；
   先经 LAW_NAME_SUFFIXES 后缀白名单滤掉文书名——《…认定书》不算法规名）

职责分界（批次 32 拆分）：本模块只保留结构性校验——回答是否为空、
[n] 编号提取与越界判定、四类异常的抛出时机；"一句话/一个名称算什么"的
语言层规则（表述词表、法规名后缀白名单、归一化、逐句扫描）在
citation_language.py。语言层公共名在此转发，既有 import 路径全部不变。

变更记录：
- v2 允许跳号：要求"从 1 开始连续不跳号"会逼迫模型把不相关的法源也硬引一遍，
  或把好答案整段拦掉（真实案例：好答案引了 [1,2,3,4,5,7,8] 因跳过 [6] 被拦）。
  编号在清单范围内即可，按需引用。
- v3 逐句判定降级：该处原抛通用 CitationError，上层按"引用错乱"整段替换，
  导致"一句话漏标引用"惩罚重于"整篇零引用"（严重度倒挂）。
  现改抛 UncitedConclusionError（CitationError 子类）→ 上层保留回答 + 追加警示。
- v4 文书名排除：回答里的《道路交通事故认定书》这类**文书名**被当成"清单外法规"误报
  （处置只是加警示、不替换回答）。现先按 LAW_NAME_SUFFIXES 后缀白名单筛出法规名候选，
  只有候选才参与清单比对。真·清单外法规（《民法典》等）仍照常拦截。
- v5 按职责拆分：语言层规则移入 citation_language.py，本模块行为逐字不变。
"""
import re

from app.chat.citation_language import (  # noqa: F401 —— 公共名转发，保持既有 import 路径
    DETERMINISTIC_CONCLUSION_KEYWORDS,
    LAW_NAME_SERIAL_SUFFIX,
    LAW_NAME_SUFFIXES,
    UNCERTAIN_EXPRESSION_KEYWORDS,
    _contains_any,
    _is_deterministic_conclusion,
    _is_law_name_candidate,
    _is_uncertain_expression,
    _normalize_law_name,
    find_unknown_laws,
    scan_uncited_conclusion,
)


class CitationError(ValueError):
    """引用不合规异常。"""
    pass


class NoCitationError(CitationError):
    """回答零引用：模型没有标注任何 [n] 引用编号（批次 10 新增子类）。

    单独区分子类的原因：有候选条文时的"零引用"与"引用越界"性质不同——
    前者只是模型没标引用，回答内容本身可能来自注入的法条；
    后者是引用错乱、无法核查。上层护栏据此分级：
    零引用 → 保留回答 + 追加警示；越界/空回答 → 仍整段替换。
    """
    pass


class UnknownLawCitationError(CitationError):
    """回答提到了清单外法规名。

    单独区分子类的原因：这类回答通常整体引用是好的，只是顺带提了
    一句清单外法规——上层护栏据此"降级处理"（保留回答 + 追加警示），
    而不是和"零引用"一样整段替换。
    """
    pass


class UncitedConclusionError(CitationError):
    """回答中个别句子是无引用的确定性结论（比整篇零引用轻，只警示不替换）。

    与 NoCitationError（整篇零引用）的区别：
    - NoCitationError：回答里一个 [n] 都没有，全部内容无从核查；
    - 本类：回答整体是有引用的，只是**个别句子**写成了确定性断言却没带 [n]。
      这类回答的主体可信，最严重的处置不应该是把整篇丢掉。

    分级依据（批次 28 修正的逻辑倒挂）：旧实现抛通用 CitationError，上层按
    "引用错乱"处置 → 整段替换为兜底文案。结果是"一句话漏标引用"惩罚重于
    "整篇零引用"，严重度与处置反了。故单独成类，上层改为"保留回答 + 追加警示"。
    引用越界、回答为空等**不可核查**的硬违规仍抛 CitationError（→ 整段替换）。
    """
    pass


def check_citations(
    answer: str,
    total_sources: int,
    available_sources: list[str] | None = None,
) -> None:
    """校验回答中的引用是否合规。

    Args:
        answer: 大模型生成的回答
        total_sources: 法源清单中的总条数
        available_sources: 法源清单中的法规名称列表（可选）

    Raises:
        CitationError: 引用不合规时抛出，携带具体问题描述。
            具体子类决定上层处置力度：
            - CitationError（空回答 / 引用越界）：不可核查 → 整段替换；
            - NoCitationError（整篇零引用）：保留回答 + 追加警示；
            - UnknownLawCitationError（提到清单外法规）：保留回答 + 追加警示；
            - UncitedConclusionError（个别句无引用的确定性结论）：保留回答 + 追加警示。
    """
    # 检查空回答
    if not answer or not answer.strip():
        raise CitationError("回答为空或无内容")

    # 引用位置归一化：LLM 常把引用编号写在句末标点之后（「工资。[8]」），
    # 而下面按 。！ 切句会吃掉标点，导致编号落入下一句、本句被判"无引用"。
    # 把「标点+[n]」改写成「[n]+标点」，使两种书写习惯等价后再切句。
    answer = re.sub(r"([。！])(\[\d+\])", r"\2\1", answer)

    # 提取所有引用编号 [n]
    citation_pattern = re.compile(r'\[(\d+)\]')
    citations = citation_pattern.findall(answer)

    # 转换为整数并去重
    citation_numbers = sorted(set(int(c) for c in citations))

    # 检查是否完全没有引用
    if not citation_numbers:
        # 判断是否有实质性内容（排除纯免责声明）
        content_without_disclaimer = re.sub(
            r'内容仅供.*?法律意见。?',
            '',
            answer,
            flags=re.DOTALL
        ).strip()
        if content_without_disclaimer and len(content_without_disclaimer) > 10:
            raise NoCitationError("回答中没有任何引用编号，但包含实质性结论")
        return  # 纯免责声明，不需要引用

    # 检查引用是否越界
    for num in citation_numbers:
        if num < 1 or num > total_sources:
            raise CitationError(
                f"引用编号 [{num}] 越界：法源清单只有 {total_sources} 条，有效范围是 [1] 到 [{total_sources}]"
            )

    # 检查是否引用了清单外的法规（比对规则在 citation_language.find_unknown_laws）
    # （跳号已允许：模型按需引用相关法源即可，见模块 docstring 变更记录）
    if available_sources:
        undefined_laws = find_unknown_laws(answer, available_sources)
        if undefined_laws:
            raise UnknownLawCitationError(
                f"回答中提到了清单外的法规：{', '.join(f'《{law}》' for law in undefined_laws)}。"
                f"只能使用清单内的法规：{', '.join(f'《{law}》' for law in available_sources)}"
            )

    # 检查是否有无引用的实质性结论（逐句判定规则在 citation_language.scan_uncited_conclusion）
    uncited_sentence = scan_uncited_conclusion(answer)
    if uncited_sentence:
        raise UncitedConclusionError(
            f"发现无引用的实质性结论：「{uncited_sentence[:50]}...」。"
            f"每个实质结论后必须标注引用编号。"
        )
