"""引用校验的语言层规则：表述词表、法规名候选、归一化比对、逐句结论扫描。

与 citation_check.py 的职责分界（批次 32 拆分，行为逐字不变）：
- citation_check.py：结构性校验——回答是否为空、[n] 编号提取与越界判定，
  以及四类异常（空回答/零引用/越界/清单外/逐句无引用结论）的抛出时机；
- 本模块：判断"一句话 / 一个书名号名称算什么"——
  表述词表（确定性结论 / 不确定表述）、法规名后缀白名单、
  法规名归一化与清单外比对、逐句扫描第一条无引用的确定性结论。

术语与 docs/CONTEXT.md 一致：法源清单、法律引用、法规名。
"""
import re
import unicodedata

# ==================== 表述判定词表（唯一出处，勿在别处复写） ====================
#
# 【确定性结论词】句中出现任一 → 该句是"对法律后果的确定性断言"，
# 无 [n] 引用即拦截。选词原则：只收法律规范用语——
#   义务性（应当/必须）、禁止性（不得/禁止）、赋权（有权/可以要求/无需）、
#   违法性断言（属于违法）。
# 刻意不收的词（旧版误拦的根因，勿加回）：
#   "规定/超过/不满/期限"是主题词不是结论动词——"无法给出统一期限"
#   也含"期限"，按旧规则会被误判成实质性结论；
#   裸"可以"太宽（"你可以在页面查看进度"也含"可以"），收窄为"可以要求"。
DETERMINISTIC_CONCLUSION_KEYWORDS: tuple[str, ...] = (
    "应当",
    "必须",
    "可以要求",
    "属于违法",
    "不得",
    "禁止",
    "有权",
    "无需",
)

# 【不确定/概括表述词】句中出现任一 → 整句视为非确定性表述，放行。
# 依据：不确定表述（无法确定/取决于/视情况/需结合具体事实/建议咨询…）
# 本来就不构成可核查的确定性断言，不应强求附引用。
# 判定顺序：句内"不确定表述"优先于"确定性词"——混合句
# （如"是否属于违法取决于具体情况"）宁可放行也不误拦；
# 真·确定性结论整句无引用时仍会被逐句规则拦截，不会因词表漏放。
UNCERTAIN_EXPRESSION_KEYWORDS: tuple[str, ...] = (
    "无法确定",
    "无法给出",
    "难以确定",
    "取决于",
    "视情况",
    "视具体",
    "需结合",
    "需根据",
    "因情况而异",
    "因人而异",
    "不一定",
    "可能",
    "一般",
    "通常",
    "能否",
    "建议咨询",
    "建议联系",
)

# 【法规名后缀白名单】《…》里的名称只有以这些词结尾，才算"法规名候选"参与"清单外法规"比对。
# 为什么按后缀判定：回答里的书名号不止用于法规，还用于文书与材料名（《道路交通事故认定书》
# 《…通知书》），它们本就不会出现在法源清单里，全量比对必然把它们误报成"清单外法规"（真实误报见
# 批次 30 的 direct-013）。穷举法规全名不可行（法规无限多、还有简称），但中文规范性文件的通名是
# 封闭集合，文书名的通名也是（书/证/表/单），故用"后缀落在哪一边"区分。
# 改表前先想清楚：加词 = 以它结尾的一切《…》都进比对；删词 = 真法规静默漏判，两者都要配回归测试。
# 案例材料（如『最高法发布劳动争议典型案例』）不属于法规名，不参与清单外比对；实测 b29 轮影响 0。
LAW_NAME_SUFFIXES: tuple[str, ...] = (
    "法典",  # 《民法典》末字是"典"不是"法"，只收"法"会漏判（既有回归用例锁着）
    "法",
    "条例",
    "规定",
    "办法",
    "解释",
    "决定",
    "规则",
    "细则",
    "通则",
    "准则",
    "标准",
    "批复",
    "复函",
)

# 结尾的序号括号（《…解释（一）》）：判后缀前先剥掉，否则真法规名会被当文书名静默跳过
# ——库里就有"…适用法律问题的解释（一）"这个正式名称。
LAW_NAME_SERIAL_SUFFIX = re.compile(r"[（(][^（）()]{1,4}[）)]$")


def _contains_any(sentence: str, keywords: tuple[str, ...]) -> bool:
    """句子是否包含词表中任一词。"""
    return any(keyword in sentence for keyword in keywords)


def _is_uncertain_expression(sentence: str) -> bool:
    """句子是否属于不确定/概括表述。"""
    return _contains_any(sentence, UNCERTAIN_EXPRESSION_KEYWORDS)


def _is_deterministic_conclusion(sentence: str) -> bool:
    """句子是否属于确定性结论。"""
    return _contains_any(sentence, DETERMINISTIC_CONCLUSION_KEYWORDS)


def _is_law_name_candidate(name: str) -> bool:
    """《…》里的名称是否像"法规名"（而非文书名/材料名）。判定依据见 LAW_NAME_SUFFIXES 注释。"""
    text = LAW_NAME_SERIAL_SUFFIX.sub("", name.strip()).strip()
    return text.endswith(LAW_NAME_SUFFIXES)


def _normalize_law_name(name: str) -> str:
    """法规名归一化，仅用于"清单内外"比对，不改变向用户展示的名称。

    规则（按顺序）：
    1. NFKC 归一：全角字符（如全角空格、全角括号）转半角等价形式
    2. 去掉所有空白（含半角空格、全角空格 U+3000、制表符等）
    3. 去掉书名号与首尾空白
    4. 去掉国名前缀"中华人民共和国"（《劳动合同法》==《中华人民共和国劳动合同法》）

    刻意不做的事：
    - 不做"包含关系"匹配：《劳动合同法实施条例》包含《劳动合同法》，
      但两者是不同法规，包含匹配会把清单外的实施条例误放行
    - 不做近似名合并：《劳动法》与《劳动合同法》归一化后仍是不同名字
    """
    # 1. 全角→半角等价归一
    value = unicodedata.normalize("NFKC", name)
    # 2. 去掉所有空白字符（NFKC 已把全角空格转为普通空格）
    value = re.sub(r"\s+", "", value)
    # 3. 去掉书名号与首尾残留
    value = value.strip("《》").strip()
    # 4. 去掉国名前缀
    if value.startswith("中华人民共和国"):
        value = value[len("中华人民共和国"):]
    return value


def find_unknown_laws(answer: str, available_sources: list[str]) -> list[str]:
    """返回回答中提到、但归一化后不在法源清单里的法规名（空列表 = 全部在清单内）。

    函数体自 citation_check.check_citations 的"清单外法规"比对块原样搬移
    （批次 32 拆分）；抛 UnknownLawCitationError 的时机仍留在 citation_check.py。
    """
    # 提取回答中提到的法规名（《...》格式）
    law_name_pattern = re.compile(r'《([^》]+)》')
    # 先按后缀白名单滤掉文书名（《道路交通事故认定书》这类），
    # 它们不进"清单外法规"比对，否则会把文书名误报成法规
    mentioned_laws = {
        law for law in law_name_pattern.findall(answer) if _is_law_name_candidate(law)
    }

    # 归一化只用于比对：清单按"归一化名 → 原始名"建映射，
    # 命中判定走归一化名，报错信息仍展示原始名称（用户可读）
    normalized_available = {
        _normalize_law_name(source): source for source in available_sources
    }

    # 找出清单外的法规（归一化后仍不匹配的才报错）
    return [
        law
        for law in mentioned_laws
        if _normalize_law_name(law) not in normalized_available
    ]


def scan_uncited_conclusion(answer: str) -> str | None:
    """逐句扫描：返回第一条"无引用的确定性结论"句（无则返回 None）。

    函数体自 citation_check.check_citations 的逐句判定块原样搬移（批次 32 拆分）；
    抛 UncitedConclusionError 的时机仍留在 citation_check.py——返回第一条违规句
    与原实现在第一条违规句处 raise 等价（逐句顺序一致）。

    开头的引用位置归一化与 check_citations 对 [n] 提取前的归一化是同一行正则：
    幂等替换，重复应用结果不变，本函数因此保持自包含（不依赖调用方先归一化）。
    """
    # 引用位置归一化：LLM 常把引用编号写在句末标点之后（「工资。[8]」），
    # 而按 。！ 切句会吃掉标点，导致编号落入下一句、本句被判"无引用"。
    # 把「标点+[n]」改写成「[n]+标点」，使两种书写习惯等价后再切句。
    answer = re.sub(r"([。！])(\[\d+\])", r"\2\1", answer)
    citation_pattern = re.compile(r'\[(\d+)\]')

    # 简化检测：查找结论性语句（句号结尾）附近是否有引用
    sentences = re.split(r'[。！]', answer)
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # 跳过免责声明
        if "仅供" in sentence or "不能替代" in sentence:
            continue

        # 跳过纯提示性语句（没有实质结论）
        if len(sentence) < 10:
            continue

        # 无 [n] 的句子：不确定/概括表述放行；确定性结论才拦截
        has_citation = bool(citation_pattern.search(sentence))
        if has_citation:
            continue
        if _is_uncertain_expression(sentence):
            continue
        if _is_deterministic_conclusion(sentence):
            return sentence
    return None
