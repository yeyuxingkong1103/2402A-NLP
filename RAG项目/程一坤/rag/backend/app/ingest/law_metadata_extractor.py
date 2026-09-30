"""法规级元数据抽取：从法规页面提取结构化元数据。

三条原则：
1. 只从页面实际内容提取；抽不到就留空并交由人工补录，绝不编造
2. 文书类型与发布机关**成套判定**（例如"国务院令" ⇒ 行政法规 + 国务院），
   而不是各自靠正则猜，避免出现"《实施条例》被判成法律""《调解仲裁法》
   的发布机关写成最高人民法院"这类张冠李戴
3. 每个字段都能回答"从页面哪一处得到的"，逻辑写在对应函数的注释里

日期识别规则（施行/公布/修订语境）与 HTML 文本预处理已抽到
app/ingest/law_date_rules.py；本文件只保留文书类型、发布机关的成套判定
与整体编排（extract_legal_metadata）。
"""

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

# 效力状态词表（唯一定义处）；本文件不得再出现状态取值的裸字面量
from app.db.law_status import PAGE_STATUS_KEYWORDS

# 日期解析与文本预处理（细则见 law_date_rules.py；页面排版变化改那边）
from app.ingest.law_date_rules import (
    _to_plain_text,
    extract_effective_date,
    extract_promulgation_date,
)

# 文书类型 → 效力等级（数值越小效力越高，用于检索排序与展示）
DOCUMENT_TYPE_AUTHORITY_LEVEL = {
    "法律": 2,
    "行政法规": 3,
    "部门规章": 4,
    "司法解释": 5,
    "案例材料": 9,
}

# 文书类型 → 可能的发布机关（按可能性排序，逐个到页面里核对）
ISSUING_AUTHORITY_CANDIDATES = {
    "法律": ["全国人民代表大会常务委员会", "全国人民代表大会"],
    "行政法规": ["国务院"],
    "部门规章": ["人力资源和社会保障部", "劳动部", "劳动和社会保障部"],
    "司法解释": ["最高人民法院", "最高人民检察院"],
    "案例材料": ["最高人民法院"],
}

# 判定文书类型用的关键词，顺序不能乱（先判具体类型，再判"法"）
JUDICIAL_INTERPRETATION_KEYWORDS = ("司法解释", "适用法律问题的解释", "若干问题的解释", "解释（一）", "解释（二）", "解释（三）")
CASE_MATERIAL_TITLE_KEYWORDS = ("发布", "案例")


@dataclass
class LegalMetadata:
    """法规元数据。"""

    law_name: str
    source_url: str
    jurisdiction: str
    document_type: str | None
    issuing_authority: str | None
    promulgation_date: date | None = None
    effective_date: date | None = None
    expiration_date: date | None = None
    status: str | None = None
    authority_level: int | None = None
    # 记录哪些字段是从页面抽到的，哪些仍需人工补录
    extracted_fields: tuple[str, ...] = ()

    def missing_fields(self) -> list[str]:
        """列出仍为空的字段名，供《待人工补录清单》使用。"""
        names = {
            "promulgation_date": "公布日期",
            "effective_date": "生效日期",
            "expiration_date": "失效日期",
            "status": "效力状态",
            "issuing_authority": "发布机关",
            "document_type": "文书类型",
        }
        return [
            label
            for attribute, label in names.items()
            if getattr(self, attribute) in (None, "")
        ]


def extract_status(plain_text: str) -> str | None:
    """抽取效力状态。

    依据：法规库页面会明确标注现行有效 / 已废止 / 已失效 等字样，直接采用。

    取值一律取自 app/db/law_status.py（唯一定义处），本函数不得返回裸字符串：
    历史上这里返回的中文与列注释里的英文是两套词表，写库后消费点按中文逐字比较，
    英文取值静默漏过"已失效"判断（批次 26 用英文写过 5 行，是真实踩过的坑）。
    抽不到返回 None（待人工补录），绝不猜。
    """
    for keyword in PAGE_STATUS_KEYWORDS:
        if keyword in plain_text:
            return keyword
    return None


def extract_document_type(title: str, source_url: str, plain_text: str) -> str | None:
    """判定文书类型。

    判定顺序很关键：必须先判司法解释与案例，再判条例/规定/办法，最后才判"法"。
    否则《劳动合同法实施条例》会因为标题里含"法"而被误判成法律。
    """
    # 1) 司法解释：标题含"解释"类措辞且来自最高法，或标题直接写明司法解释
    if any(keyword in title for keyword in JUDICIAL_INTERPRETATION_KEYWORDS):
        return "司法解释"

    # 2) 案例材料：新闻式标题（"最高法发布……典型案例"）或标题含"案例"
    if "案例" in title and any(keyword in title for keyword in CASE_MATERIAL_TITLE_KEYWORDS):
        return "案例材料"

    # 3) 条例：一律是行政法规（国务院制定）
    if "条例" in title:
        return "行政法规"

    # 4) 规定 / 办法 / 细则：页面出现"国务院令"则为行政法规，否则按部门规章
    if any(keyword in title for keyword in ("规定", "办法", "细则")):
        if re.search(r"国务院令第\s*\d+\s*号", plain_text) or "国务院令" in plain_text:
            return "行政法规"
        return "部门规章"

    # 5) 法律：标题含"法"且有国名，或页面出现全国人大/常委会字样
    if "中华人民共和国" in title and title.rstrip().endswith("法"):
        return "法律"
    if "全国人民代表大会常务委员会" in plain_text or "全国人民代表大会" in plain_text:
        return "法律"

    return None


def extract_issuing_authority(document_type: str | None, plain_text: str) -> str | None:
    """按文书类型到页面里核对发布机关。

    做法：先根据文书类型确定"可能的机关名单"，再逐个到页面里找，
    找到即用、找不到返回 None（交人工补录）。
    这样不会出现"法律被标成最高人民法院发布"这类跨类型误判。
    """
    if not document_type:
        return None
    candidates = ISSUING_AUTHORITY_CANDIDATES.get(document_type, [])
    for candidate in candidates:
        if candidate in plain_text:
            return candidate
    return None


def extract_law_name_from_page(plain_text: str, fallback_title: str | None) -> str:
    """清洗法规名：去掉站点后缀（如"工资支付暂行规定_人力资源和社会保障部"）。"""
    # 标题可能为空（数据包未提供），此时返回空串，由调用方决定是否报错
    cleaned = (fallback_title or "").strip()
    for separator in ("_", "｜", "|"):
        if separator in cleaned:
            cleaned = cleaned.split(separator, 1)[0].strip()
    return cleaned or fallback_title


def extract_legal_metadata(
    law_name: str,
    source_url: str,
    html_file_path: Path,
) -> LegalMetadata:
    """从法规名称、来源 URL 与原始 HTML 提取元数据；抽不到的字段留空。"""
    html_content = html_file_path.read_text(encoding="utf-8", errors="ignore")
    plain_text = _to_plain_text(html_content)

    document_type = extract_document_type(law_name, source_url, plain_text)
    issuing_authority = extract_issuing_authority(document_type, plain_text)
    # 先抽公布日期再抽生效日期："自公布之日起施行"需要公布日期做映射
    promulgation_date = extract_promulgation_date(plain_text)
    effective_date = extract_effective_date(plain_text, promulgation_date=promulgation_date)
    status = extract_status(plain_text)

    # 记录哪些字段是抽到的，便于报告与排查
    extracted = tuple(
        name
        for name, value in (
            ("document_type", document_type),
            ("issuing_authority", issuing_authority),
            ("promulgation_date", promulgation_date),
            ("effective_date", effective_date),
            ("status", status),
        )
        if value is not None
    )

    return LegalMetadata(
        law_name=extract_law_name_from_page(plain_text, law_name),
        source_url=source_url,
        # 本批全部来自中国大陆官方站点，法域固定
        jurisdiction="中国大陆",
        document_type=document_type,
        issuing_authority=issuing_authority,
        promulgation_date=promulgation_date,
        effective_date=effective_date,
        # 失效日期页面通常不写；由后续版本管理维护，不在这里猜
        expiration_date=None,
        status=status,
        authority_level=DOCUMENT_TYPE_AUTHORITY_LEVEL.get(document_type or ""),
        extracted_fields=extracted,
    )
