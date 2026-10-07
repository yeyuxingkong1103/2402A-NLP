# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
# 工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
# 工单编号：人工智能NLP-RAG-Query 理解优化任务
# 工单编号：人工智能NLP-RAG-功能测试及评估
# 工单01 - 基于PDF文档的问答系统
# 工单02 - 基于PDF文档的问答系统的优化
# 工单03 - PDF文档的表格解析及检索优化
"""
文档配置（DocProfile）：一份 PDF 一条。

【为什么要外置】加第二份 PDF 时才暴露出，有**五处**是与具体文档绑死的：
页眉模板、页脚模板、页码格式、查询要抽象掉的实体专名、证据页的抽取正则。
它们原先散在 pdf_parser / config / evaluator 里，全部写死成招股说明书1.pdf。
工单04/05 还要继续加文档，所以收拢到这里，一份文档一行配置。

【两份文档的差异都是实测出来的，不是猜的】
                    招股说明书1.pdf            招股说明书2.pdf
  页眉        武汉兴图新科电子股份有限公司…招股意向书   武汉力源信息技术股份有限公司…招股意向书
  页脚        1-1-21（分节页码）              21（**裸数字**）
  篇幅        548 页                        350 页
  旋转页      无                            8 页（旋转 90° 的横向表，页眉页脚不在常规分带内）
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# 页眉/页脚所在的纵向比例带（实测页高 842：页眉 y≈44 → 0.05，页脚 y≈780 → 0.93）
HEADER_BAND = 0.12
FOOTER_BAND = 0.85


@dataclass(frozen=True)
class DocProfile:
    """一份 PDF 的解析与检索配置。"""

    key: str                     # 稳定标识
    doc_name: str                # 文件名，与 Milvus 的 doc_name 字段一致
    header_re: re.Pattern        # 页眉：整行模板
    footer_re: re.Pattern        # 页脚：整行模板
    label_fmt: str               # 页码格式，如 "1-1-{n}" 或 "{n}"
    entity_names: list[str] = field(default_factory=list)
    evidence_re: re.Pattern | None = None   # 证据页字符串的抽取正则
    # ---------- 提示词用 ----------
    # 【为什么这两项也必须进配置】原本写死在 generator.SYSTEM_PROMPT 里
    # （"基于《武汉兴图新科电子股份有限公司招股意向书》…"和「（见 1-1-XX 页）」）。
    # 加了第二份文档后，这个写死的 system prompt 会造成**系统性拒答**：
    # 模型读到"这是兴图新科的文件"，于是对力源的问题判定"问题与文档不符"，
    # 直接输出"资料中未提及相关信息" —— 而答案明明就在片段里（实测 id=1）。
    # 同时它还会按 1-1-XX 编造书2 的页码（书2 的页脚是裸数字）。
    title: str = ""              # 文档全称，写进 system prompt
    citation_hint: str = ""      # 页码标注格式示例，如 "1-1-XX" / "XX"
    notes: str = ""              # 该文档特有的作答提醒（拼在规则末尾）
    # ---------- 工单07：多文档页码 ----------
    # 【为什么偏移必须按文档走】原先偏移只有一个全局值
    # （settings.page_label_offset = 0），它只对招股书1 成立。实测工单07 的
    # 9 份年报，首页偏移各不相同：中信证券 / 招商证券 / 国泰君安 / 招商银行
    # 是 +1，中国人寿是 −1，中国平安是 −3。
    page_label_offset: int = 0
    # 【为什么还要读页脚】招商银行 / 邮储银行 / 中国太保的年报在「财务报告」
    # 一节**重新从 1 开始编号**（实测招商银行 idx=140 的页脚印的是 `- 15 -`，
    # idx=60 印的是 `61`）—— 任何单一偏移都不可能同时对上两段编号。
    # 所以页码**优先从该页页脚实际印着的那串数字读**，读不到才回退到
    # `index − 偏移`。引用页码因此与读者在 PDF 里看到的完全一致。
    # 模板用 fullmatch 匹配**整块**文本，避免把正文里的裸数字误当页码。
    page_no_re: re.Pattern | None = None
    footer_zone: float = FOOTER_BAND       # 判定「页脚带」的纵向比例

    # ------------------------------------------------------------------
    def footer_label(self, page, blocks) -> str | None:
        """从页脚带里读出**该页实际印刷的页码**；读不到返回 None。

        【为什么不用 search 而是 fullmatch】页脚带里除了页码还会有正文
        （实测平安银行 p50 的页脚带里就压着一行「2 不良生成率=不良生成额…」）。
        search 会从那行里抠出一个 `2` 当作页码。fullmatch 要求整块文本
        恰好就是页码模板，误命中面小得多。

        【为什么单独一个方法】复核脚本要能分清「页脚实读成功」与
        「读出来的值恰好等于回退值」—— 后者在偏移本来就对时是常态
        （实测中信证券 370 页全部如此）。用 `printed_label() != fallback`
        去统计实读率，会把"实读成功"整片误判成"没实读"。
        """
        if self.page_no_re is None or page is None:
            return None
        h = page.rect.height
        best: tuple[float, str] | None = None
        for b in blocks or []:
            if b.get("type") != 0:
                continue
            if b["bbox"][1] < h * self.footer_zone:
                continue
            raw = "".join(s["text"] for line in b.get("lines", [])
                          for s in line.get("spans", [])).strip()
            if not raw:
                continue
            m = self.page_no_re.fullmatch(raw)
            if not m:
                continue
            # 取模板里**第一个**捕获组 —— 各文档的页码在页脚里的位置不同
            # （平安银行单独一块；中国平安/中国人寿与栏目名挤在同一行）。
            page_no = next((g for g in m.groups() if g), None)
            if page_no and (best is None or b["bbox"][1] > best[0]):
                best = (b["bbox"][1], page_no)
        return best[1] if best else None

    def printed_label(self, page, blocks, fallback: str) -> str:
        """页码：页脚实读优先，读不到才用 `index − 偏移`。

        【为什么要挡非正数】中国太保的偏移是 20（正文节从第 21 个 position 起
        才算第 1 页），于是一个「没有页脚」的前导页会算出 `18 − 20 = −2` 这种
        负数页码。少数页确实整体是图、页脚带里什么都没有（实测太保 idx18）。
        这种页给不出有意义的"印刷页码"，退回**1 基位置**（idx+1）—— 它至少是
        正的、且在 PDF 里可以直接翻到。
        """
        got = self.footer_label(page, blocks)
        if got:
            return got
        s = (fallback or "").strip()
        # 【只对**数值型**页码挡非正数】招股书的页码是 `1-1-25` 这种带前缀的
        # 非数值格式，它**不该**走这条路 —— 第一版写成 `s.isdigit() and int(s) >= 1`
        # 就正好把它判成"不可用"，于是全书页码被替换成裸序号（`1-1-26` → `26`）。
        # 是 tests/test_pipeline.py 的页标签用例当场抓出来的（引用页码全错，
        # 正是本项目最忌讳的"不报错的错"）。非数值格式一律原样返回。
        if s.isdigit() and int(s) < 1:
            # 太保 offset=20，前导页会算出 −2 这种负数页码 → 退回 1 基位置。
            # PyMuPDF 的 Page 上没有 `page_number`，0 基索引叫 `number`（1.25 实测）。
            return str(getattr(page, "number", 0) + 1)
        return s or str(getattr(page, "number", 0) + 1)

    # ------------------------------------------------------------------
    def label(self, page_no: int, offset: int = 0) -> str:
        """PDF position index → 给人看的页码。

        【为什么格式要按文档走，不统一成 1-1-N】
        招股书2 的页脚印的就是裸数字 `21`。若统一成 `1-1-21`：
          (a) 引用与读者在 PDF 里看到的页码对不上，验收时无法核对；
          (b) 两份文档都产生 `1-1-21`，页级精确率/召回率会**跨文档歧义**。
        按文档走格式，引用可核对，页级指标也不再含糊。
        """
        return self.label_fmt.format(n=page_no - offset)

    def matches_header(self, line: str) -> bool:
        return bool(self.header_re.match(line.strip()))

    def matches_footer(self, line: str) -> bool:
        return bool(self.footer_re.match(line.strip()))


# ----------------------------------------------------------------------
DOC_PROFILES: dict[str, DocProfile] = {
    "xingtu": DocProfile(
        key="xingtu",
        doc_name="招股说明书1.pdf",
        header_re=re.compile(r"^\s*武汉兴图新科电子股份有限公司\s+招股意向书\s*$"),
        footer_re=re.compile(r"^\s*1-1-\d+\s*$"),
        label_fmt="1-1-{n}",
        entity_names=[
            "武汉兴图新科电子股份有限公司",
            "武汉兴图新科电子",
            "兴图新科",
            # 英文全称（工单01 功能验收 4 要求中英双语，英文提问也得能路由）
            "Wuhan Xingtu Xinke Electronics Co.,Ltd.",
            "Wuhan Xingtu Xinke Electronics",
        ],
        # 证据页字符串形如 "1-1-128" 或 "1-1-21（发行人基本情况表）；1-1-51 亦载明"
        evidence_re=re.compile(r"1-1-\d+"),
        title="武汉兴图新科电子股份有限公司招股意向书",
        citation_hint="1-1-XX",
        notes=(
            "7. 【脱敏名称原样】原文用「某」字脱敏的名称（如《某视频技术规范1.0》、\n"
            "   「某情报、指挥、控制与通信网络一体化工程」）必须**原样照抄**，\n"
            "   不得补全、改写或意译成别的名称。"
        ),
    ),
    "liyuan": DocProfile(
        key="liyuan",
        doc_name="招股说明书2.pdf",
        header_re=re.compile(r"^\s*武汉力源信息技术股份有限公司\s+招股意向书\s*$"),
        # 页脚是**裸数字**（实测第 21 页页脚 = "21"，index 21 → 印刷页 21，偏移 0）
        footer_re=re.compile(r"^\s*\d{1,3}\s*$"),
        label_fmt="{n}",
        entity_names=[
            "武汉力源信息技术股份有限公司",
            "武汉力源信息技术",
            "武汉力源",
            "力源信息",
            # 英文全称（书2 封面印的就是这个写法）
            "Wuhan P&S Information Technology Co.,Ltd.",
            "Wuhan P&S Information Technology",
        ],
        # 【注意】这条正则比较宽，所以本文件的 evidence_page 字段必须**只写页码**
        # （如 "21" 或 "21 23"），不要混入年份之类的其它数字，否则会被误抽。
        evidence_re=re.compile(r"(?<!\d)\d{1,3}(?!\d)"),
        title="武汉力源信息技术股份有限公司招股意向书",
        citation_hint="XX",
        # 书2 大量用「[ ◆ ]」占位（如发行价格、募集资金净额）。不提醒的话，
        # 模型会把空占位当成"有内容"或者自己补一个数出来 —— 而书2 的题目
        # （发行股数、募投项目、关联方）恰好都紧邻这些占位。
        notes=(
            "7. 【占位符原样】原文中印作「[ ◆ ]」的位置是**空缺占位**，"
            "表示该处未披露；照抄「[ ◆ ]」即可，不要臆测或补填其内容。"
        ),
    ),
}


# ======================================================================
# 工单07：ccf_competition 语料 —— 9 份 A 股上市公司年度报告
# ======================================================================
# 工单 07 要求「用 01–06 实现的 RAG，对 ccf_competition.zip 里的 pdf 做功能测试
# 与检索评估」。这 9 份年报与两份招股书的排版差异极大，**逐份实测**后写入：
#
#   文档                页数  页脚形态                          首页偏移  页眉
#   平安银行2019年报     265  裸数字 `30`                          0     公司名+「财务报表附注」
#   中国平安2019年报     342  `二零一九年年报…公司47`（黏在一行）     3     仅「2019年度」
#   招商银行2019年报     351  裸数字 / `- 15 -`（财务报告节重编号）   -1    节名+公司名+报告名
#   邮储银行2019年报     421  裸数字 / `- 12 -`（附注节重编号）        0     公司名+「年度财务报表附注」
#   中信证券2020年报     370  `2 / 370`（页码/总页数）               -1    公司名+报告名（每页）
#   中国人寿2020年报     235  `公司名 | 二零二零年年报 | 财务报告116`    1    「年度财务报表附注（续）」
#   中国太保2021年报     264  形态多达 6 种（见下）                  20    无（全是节名，删了会误伤标题）
#   招商证券2021年报     278  `2 / 278`                            -1    公司名+报告名（每页）
#   国泰君安2021年报     286  `2/286`                              -1    「2021 年年度报告」（每页）
#
# 【为什么这份清单必须逐份实测】页码对不上 = 引用页码全错，属"静默答错"。
# 实测两个具体坑：
#   1. 招商银行 / 邮储银行 / 中国太保 的年报在「财务报告」一节**重新从 1 编号**
#      （招行 idx140 印 `- 15 -`，idx60 印 `61`）。**任何单一偏移都不可能同时
#      对上两段** —— 所以页码改为优先从页脚实读（见 DocProfile.printed_label）。
#   2. `2 / 370` 这类"页码/总页数"如果只写 `\d+` 会把总页数也吃进来。
# 全部由 `scripts/verify_ccf_profiles.py` 逐页复核（判据见该脚本）。
_NO_HEADER = re.compile(r"(?!)")   # 永不匹配：显式声明"这份文档不做页眉删除"

# 招商银行/邮储银行/中国太保 的公共页脚形态：裸数字 或 `- N -`
_BARE_OR_DASH_PAGE = re.compile(r"^\s*-?\s*(\d{1,3})\s*-?\s*$")
# 中信证券/招商证券/国泰君安 的公共页脚形态：`N / 总页数`（国泰君安无空格）
_PAGE_SLASH_TOTAL = re.compile(r"^\s*(\d{1,3})\s*/\s*\d{1,4}\s*$")

_CCF_PROFILES: dict[str, DocProfile] = {
    "payh": DocProfile(
        key="payh",
        doc_name="平安银行2019年报.pdf",
        header_re=re.compile(r"^\s*平安银行股份有限公司\s+财务报表附注\s*$"),
        footer_re=_BARE_OR_DASH_PAGE,
        page_no_re=_BARE_OR_DASH_PAGE,
        label_fmt="{n}",
        page_label_offset=0,
        entity_names=["平安银行股份有限公司", "平安银行", "Ping An Bank Co., Ltd."],
        evidence_re=re.compile(r"(?<!\d)\d{1,3}(?!\d)"),
        title="平安银行股份有限公司2019年年度报告",
        citation_hint="XX",
    ),
    "zgpa": DocProfile(
        key="zgpa",
        doc_name="中国平安2019年报.pdf",
        # 【为什么不做页眉删除】页眉只有孤零零一行「2019年度」——它太短了，
        # 复核实测有 **20 处**命中落在 y=0.17~0.23（正文带）而不是页眉带，
        # 无法区分"被大图挤下来的页眉"与"正文里的年度小标题"。
        # 按"宁可留噪声，也不静默删正文"（同 pdf_parser 顶部第 2 条设计），
        # 这里选择不删：留一行 `2019年度` 只是噪声，删错就是答案消失。
        header_re=_NO_HEADER,
        # 页脚把栏目名、公司名、页码**黏在同一行**，页码在最后
        footer_re=re.compile(
            r"^\s*二零一九年年报\s*中国平安保险（集团）股份有限公司\s*(\d{1,3})\s*$"),
        page_no_re=re.compile(
            r"^\s*二零一九年年报\s*中国平安保险（集团）股份有限公司\s*(\d{1,3})\s*$"),
        label_fmt="{n}",
        page_label_offset=3,
        entity_names=["中国平安保险（集团）股份有限公司", "中国平安"],
        evidence_re=re.compile(r"(?<!\d)\d{1,3}(?!\d)"),
        title="中国平安保险（集团）股份有限公司2019年年度报告",
        citation_hint="XX",
    ),
    "zsyh": DocProfile(
        key="zsyh",
        doc_name="招商银行2019年报.pdf",
        header_re=re.compile(
            r"^\s*\S.*?\s{2,}招商银行股份有限公司\s*2019\s*年度报告（A 股）\s*$"),
        footer_re=_BARE_OR_DASH_PAGE,
        page_no_re=_BARE_OR_DASH_PAGE,
        label_fmt="{n}",
        # 正文段印的是 idx+1；「财务报告」节重编号后由页脚实读兜住
        page_label_offset=-1,
        entity_names=["招商银行股份有限公司", "招商银行", "招行"],
        evidence_re=re.compile(r"(?<!\d)\d{1,3}(?!\d)"),
        title="招商银行股份有限公司2019年度报告（A 股）",
        citation_hint="XX",
    ),
    "ycyh": DocProfile(
        key="ycyh",
        doc_name="邮储银行2019年报.pdf",
        header_re=re.compile(
            r"^\s*中国邮政储蓄银行股份有限公司\s+2019\s*年度财务报表附注\s*$"),
        footer_re=_BARE_OR_DASH_PAGE,
        page_no_re=_BARE_OR_DASH_PAGE,
        label_fmt="{n}",
        page_label_offset=0,
        entity_names=["中国邮政储蓄银行股份有限公司", "中国邮政储蓄银行",
                      "邮储银行", "邮政储蓄银行"],
        evidence_re=re.compile(r"(?<!\d)\d{1,3}(?!\d)"),
        title="中国邮政储蓄银行股份有限公司2019年年度报告",
        citation_hint="XX",
    ),
    "zxzq": DocProfile(
        key="zxzq",
        doc_name="中信证券2020年报.pdf",
        header_re=re.compile(r"^\s*中信证券\s*2020\s*年年度报告\s*$"),
        footer_re=_PAGE_SLASH_TOTAL,
        page_no_re=_PAGE_SLASH_TOTAL,
        label_fmt="{n}",
        page_label_offset=-1,
        entity_names=["中信证券股份有限公司", "中信证券"],
        evidence_re=re.compile(r"(?<!\d)\d{1,3}(?!\d)"),
        title="中信证券股份有限公司2020年年度报告",
        citation_hint="XX",
    ),
    "zgrs": DocProfile(
        key="zgrs",
        doc_name="中国人寿2020年报.pdf",
        header_re=re.compile(r"^\s*2020\s*年度财务报表附注（续）\s*$"),
        # 页脚有**两种**形态：页码在前（`116中国人寿…| 财务报告`）
        # 与页码在后（`中国人寿…| 前导信息02`）。两个捕获组各管一种。
        footer_re=re.compile(
            r"^\s*(?:(\d{1,3})\s*)?中国人寿保险股份有限公司\s*\|"
            r"\s*二零二零年年报\s*\|[^\d]*(?:(\d{1,3}))?\s*$"),
        page_no_re=re.compile(
            r"^\s*(?:(\d{1,3})\s*)?中国人寿保险股份有限公司\s*\|"
            r"\s*二零二零年年报\s*\|[^\d]*(?:(\d{1,3}))?\s*$"),
        label_fmt="{n}",
        page_label_offset=1,
        entity_names=["中国人寿保险股份有限公司", "中国人寿"],
        evidence_re=re.compile(r"(?<!\d)\d{1,3}(?!\d)"),
        title="中国人寿保险股份有限公司2020年年度报告",
        citation_hint="XX",
    ),
    "zgtb": DocProfile(
        key="zgtb",
        doc_name="中国太保2021年报.pdf",
        # 【为什么不做页眉删除】这份年报的页眉就是「财务报告 / 公司治理 / 经营业绩」
        # 这类**节名**，同样的字符串在正文里就是真标题。删它 = 静默删正文标题。
        header_re=_NO_HEADER,
        # 6 种页脚形态（实测 264 页全扫）：前导节 `2021 年 年度报告前导 ｜N` /
        # `N ｜ 公司名` / 正文节 `N 公司名` / `公司名N` / `N 2021 年度报告` / 裸 `N`。
        # 注意 `412021 年度报告` 是「页码 41」与「2021 年度报告」**黏在一起**，
        # 所以 `(\d{1,3})` 写在后缀前面，靠回溯取到正确的 41 而不是 412。
        footer_re=re.compile(
            r"^\s*(?:"
            r"2021\s*年\s*年度报告前导\s*｜\s*(\d{1,3})"
            r"|(\d{1,3})\s*｜\s*中国太平洋保险（集团）股份有限公司"
            r"|(\d{1,3})\s*中国太平洋保险（集团）股份有限公司"
            r"|中国太平洋保险（集团）股份有限公司\s*(\d{1,3})"
            r"|中国太平洋保险（集团）股份有限公司"
            r"|(\d{1,3})\s*202[01]\s*年度报告"
            r"|(\d{1,3})"
            r")\s*$"),
        page_no_re=re.compile(
            r"^\s*(?:"
            r"2021\s*年\s*年度报告前导\s*｜\s*(\d{1,3})"
            r"|(\d{1,3})\s*｜\s*中国太平洋保险（集团）股份有限公司"
            r"|(\d{1,3})\s*中国太平洋保险（集团）股份有限公司"
            r"|中国太平洋保险（集团）股份有限公司\s*(\d{1,3})"
            r"|中国太平洋保险（集团）股份有限公司"
            r"|(\d{1,3})\s*202[01]\s*年度报告"
            r"|(\d{1,3})"
            r")\s*$"),
        label_fmt="{n}",
        page_label_offset=20,
        entity_names=["中国太平洋保险（集团）股份有限公司", "中国太平洋保险",
                      "中国太保", "太平洋保险"],
        evidence_re=re.compile(r"(?<!\d)\d{1,3}(?!\d)"),
        title="中国太平洋保险（集团）股份有限公司2021年年度报告",
        citation_hint="XX",
    ),
    "zszq": DocProfile(
        key="zszq",
        doc_name="招商证券2021年报.pdf",
        header_re=re.compile(r"^\s*招商证券股份有限公司\s*2021\s*年年度报告\s*$"),
        footer_re=_PAGE_SLASH_TOTAL,
        page_no_re=_PAGE_SLASH_TOTAL,
        label_fmt="{n}",
        page_label_offset=-1,
        entity_names=["招商证券股份有限公司", "招商证券"],
        evidence_re=re.compile(r"(?<!\d)\d{1,3}(?!\d)"),
        title="招商证券股份有限公司2021年年度报告",
        citation_hint="XX",
    ),
    "gtja": DocProfile(
        key="gtja",
        doc_name="国泰君安2021年报.pdf",
        header_re=re.compile(r"^\s*2021\s*年年度报告\s*$"),
        footer_re=_PAGE_SLASH_TOTAL,
        page_no_re=_PAGE_SLASH_TOTAL,
        label_fmt="{n}",
        page_label_offset=-1,
        entity_names=["国泰君安证券股份有限公司", "国泰君安证券", "国泰君安"],
        evidence_re=re.compile(r"(?<!\d)\d{1,3}(?!\d)"),
        title="国泰君安证券股份有限公司2021年年度报告",
        citation_hint="XX",
    ),
}

DOC_PROFILES.update(_CCF_PROFILES)


def doc_profiles_for_ccf() -> dict[str, DocProfile]:
    """工单07 的 9 份年报配置（按 key）。

    单独开一个取用口，是为了让「本工单新增了哪些文档」这件事在代码里**可枚举** ——
    复核脚本（verify_ccf_profiles）与入库脚本都拿它当默认范围，不必手抄 key 列表。
    """
    return dict(_CCF_PROFILES)


# 未指定文档时的默认配置（保持工单01/02 的既有行为）
DEFAULT_DOC = DOC_PROFILES["xingtu"]


def get_doc_profile(name_or_key: str | None) -> DocProfile | None:
    """按 key 或文件名取配置。取不到返回 None。"""
    if not name_or_key:
        return None
    s = name_or_key.strip()
    if s in DOC_PROFILES:
        return DOC_PROFILES[s]
    for p in DOC_PROFILES.values():
        if p.doc_name == s:
            return p
    return None


def profile_for_path(path: str | Path) -> DocProfile:
    """按 PDF 文件名取配置；取不到**直接报错**，不静默套用别的文档的规则。

    【为什么报错而不是回退】页眉/页脚/页码格式都是文档特有的 ——
    静默套用招股书1 的规则，结果就是"页眉一条都清不掉、页码格式显示错"
    （工单03 加招股书2 时实测踩过：页眉页脚各清 0 条）。
    这种失败不报错、只是结果悄悄变差，所以宁可当场失败。
    """
    name = Path(path).name
    prof = get_doc_profile(name)
    if prof is None:
        known = "、".join(p.doc_name for p in DOC_PROFILES.values())
        raise ValueError(
            f"没有为《{name}》配置解析规则。请在 app/core/doc_profiles.py 的 "
            f"DOC_PROFILES 里加一条（页眉模板 / 页脚模板 / 页码格式 / 实体专名）。"
            f"已配置的文档：{known}"
        )
    return prof


def all_entity_names() -> list[str]:
    """所有文档的实体专名（查询抽象用）。**按长度降序**，保证最长匹配优先。"""
    names: list[str] = []
    for p in DOC_PROFILES.values():
        names.extend(p.entity_names)
    return sorted(set(names), key=len, reverse=True)


def doc_ids_by_entity() -> dict[str, str]:
    """实体专名 → 文档 key。供路由用（最长匹配优先）。"""
    out: dict[str, str] = {}
    for p in DOC_PROFILES.values():
        for n in p.entity_names:
            out[n] = p.key
    return out


def canonical_entity(doc_key: str, *, english: bool = False) -> str:
    """该文档的**规范实体名**（工单05 指代消解用）。取 entity_names[0]（全称）。

    【为什么必须是全称，不能是 route().matched】route() 返回的是**命中的那个专名**，
    可能是短名（「武汉力源」/「兴图新科」）。改写时若代入短名：
      · 路由结果一样（同一个 doc_key），所以**看起来没问题**；
      · 但 `query_views()` 的「原句视角」变了 → 稠密向量变了 → 排序可能漂移。
    这是典型的「数字悄悄变了却查不出原因」的静默回归。所以统一取全称。

    【为什么要 english 分支】工单01 功能验收 4 要求中英双语。英文轮次的指代消解若代入
    中文全称，会产出「Who is 武汉兴图新科电子股份有限公司？」这种中英混句。
    entity_names 里已配了英文全称（如 "Wuhan P&S Information Technology Co.,Ltd."）。

    【注意】本函数是**纯新增**，不改 entity_names、不改 abstract_query 的行为 ——
    后者是已交付 16 题查询视角的事实来源，动它会直接威胁 16/16 的回归闸门。
    """
    prof = DOC_PROFILES[doc_key]
    if english:
        ascii_names = [n for n in prof.entity_names if n.isascii()]
        if ascii_names:
            return max(ascii_names, key=len)
    return prof.entity_names[0]
