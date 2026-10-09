# 工单编号：人工智能NLP-RAG-金融问答系统部署
"""ccf_competition 语料清单

附件 ccf_competition.zip 解压后有 9 份 A 股上市公司年报（pdf/ 与 txt/ 各一份）。
这批文件名是 GBK 字节被按错误编码解出来的乱码（Python 读出来带 U+FFFD，
原始汉字已经丢了），所以**不能用文件名识别文档**。

好在文件名里的日期和股票代码是纯 ASCII，未被破坏，且 6 位代码唯一。
这里就用代码作为匹配键，公司名/简称/年度从 PDF 首页正文里取（正文是好的）。
"""
import os
import re
from pathlib import Path

# 语料根目录：**含 pdf/ 与 txt/ 两个子目录**的那一层（即 ccf_competition 本身）。
#
# 必须可用环境变量覆盖：容器里语料是挂载进来的，路径固定在 /corpus，
# 写死 Windows 路径的话 `kb-builder` 服务在容器里一份文档都找不到
# （实测：环境变量设了 /corpus，代码仍去读 D:\...，解析到 0 份文档）。
CORPUS_DIR = Path(os.getenv("CORPUS_DIR", r"D:\BW\RAG 工单\附件\ccf_competition"))
PDF_DIR = CORPUS_DIR / "pdf"

# 9 份年报。code 是文件名里唯一没被乱码破坏的字段，用它定位文件。
DOCS = [
    {"key": "平安银行2019", "code": "000001", "short": "平安银行", "year": 2019,
     "company": "平安银行股份有限公司", "industry": "银行"},
    {"key": "中国平安2019", "code": "601318", "short": "中国平安", "year": 2019,
     "company": "中国平安保险（集团）股份有限公司", "industry": "保险"},
    {"key": "招商银行2019", "code": "600036", "short": "招商银行", "year": 2019,
     "company": "招商银行股份有限公司", "industry": "银行"},
    {"key": "邮储银行2019", "code": "601658", "short": "邮储银行", "year": 2019,
     "company": "中国邮政储蓄银行股份有限公司", "industry": "银行"},
    {"key": "中信证券2020", "code": "600030", "short": "中信证券", "year": 2020,
     "company": "中信证券股份有限公司", "industry": "证券"},
    {"key": "中国人寿2020", "code": "601628", "short": "中国人寿", "year": 2020,
     "company": "中国人寿保险股份有限公司", "industry": "保险"},
    {"key": "中国太保2021", "code": "601601", "short": "中国太保", "year": 2021,
     "company": "中国太平洋保险（集团）股份有限公司", "industry": "保险"},
    {"key": "招商证券2021", "code": "600999", "short": "招商证券", "year": 2021,
     "company": "招商证券股份有限公司", "industry": "证券"},
    {"key": "国泰君安2021", "code": "601211", "short": "国泰君安", "year": 2021,
     "company": "国泰君安证券股份有限公司", "industry": "证券"},
]


def resolve(doc):
    """按股票代码在附件目录里找到对应的 PDF 路径。"""
    hits = [p for p in PDF_DIR.glob("*.pdf") if f"__{doc['code']}__" in p.name]
    if len(hits) != 1:
        raise FileNotFoundError(
            f"股票代码 {doc['code']}（{doc['key']}）在 {PDF_DIR} 下匹配到 {len(hits)} 个文件，"
            f"期望恰好 1 个")
    return hits[0]


def available():
    """返回附件目录里实际存在的文档（路径缺失的跳过，便于在别处复现）。"""
    ready = []
    for doc in DOCS:
        try:
            ready.append({**doc, "path": resolve(doc)})
        except FileNotFoundError:
            pass
    return ready


def label(doc):
    """报告和图里用的展示名，如 平安银行（000001，2019年）。"""
    return f"{doc['short']}（{doc['code']}，{doc['year']}年）"


# 问题里的简称 → 文档 key。路由和命中统计都要用：
# 「平安」既是平安银行也是中国平安，「中信」只在证券里出现，必须显式列全，
# 不能靠子串猜。
ALIASES = {
    "平安银行": "平安银行2019",
    "中国平安": "中国平安2019", "平安集团": "中国平安2019", "平安保险": "中国平安2019",
    "招商银行": "招商银行2019", "招行": "招商银行2019",
    "邮储银行": "邮储银行2019", "邮政储蓄银行": "邮储银行2019",
    "中信证券": "中信证券2020", "中信": "中信证券2020",
    "中国人寿": "中国人寿2020", "国寿": "中国人寿2020",
    "中国太保": "中国太保2021", "太平洋保险": "中国太保2021", "太保": "中国太保2021",
    "招商证券": "招商证券2021",
    "国泰君安": "国泰君安2021",
}

# 行业词 → 该行业包含的文档 key（跨文档问题用）
INDUSTRY = {
    "银行": ["平安银行2019", "招商银行2019", "邮储银行2019"],
    "保险": ["中国平安2019", "中国人寿2020", "中国太保2021"],
    "证券": ["中信证券2020", "招商证券2021", "国泰君安2021"],
}


def mentioned_docs(question):
    """问题里直接点名的文档 key 集合（用于判断跨文档检索的召回覆盖面）。"""
    return {key for alias, key in ALIASES.items() if alias in (question or "")}


_CODE_IN_Q = re.compile(r"(?<!\d)(000001|601318|600036|601658|600030|601628|601601|600999|601211)(?!\d)")


def mentioned_codes(question):
    """问题里写股票代码时同样算点名。"""
    return {d["key"] for d in DOCS if d["code"] in _CODE_IN_Q.findall(question or "")}
