# app/core/metadata_service.py
"""元数据路召回：按法条号/法律名精确命中。

为什么需要这一路：向量检索对**编号不敏感**。用户问「刑法第一百三十三条」，
纯向量检索会把语义相近的条文一起召回，真正那一條未必排在前面。
而编号是**精确信息**，用元数据过滤可以直接命中，不该交给语义相似度去猜。

链路：问题 -> 抽取(法律名, 条文号) -> MySQL 索引表查 doc_id -> Milvus 取原文。

这也是本项目的"多路召回"中 MySQL 那一路：MySQL 承担结构化索引，
Milvus 承担向量与原文，各做擅长的事。
"""
import re
from typing import Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.db.milvus_conn import expr_eq, get_milvus
from app.models.tables import LawIndex
import logging

logger = logging.getLogger(__name__)

CN_DIGITS = "零一二三四五六七八九"
CN_NUM = {c: i for i, c in enumerate(CN_DIGITS)}
CN_UNIT = {"十": 10, "百": 100, "千": 1000}

# 法条号里可能出现的数字字符：阿拉伯数字 + 中文数字（含「〇」这种写法）。
# 三个模块的正则都要用（本模块的 ARTICLE_RE、validate_service 的引用抽取、
# fetch_statutes 的法条切分），定义在这里供它们共用，避免各写一份漏改。
CN_NUMERAL = "0-9一二三四五六七八九十百千零〇"
# 「之一」「之二」这类增补条款的后缀
CN_SUFFIX = "一二三四五六七八九十"

ARTICLE_RE = re.compile(r"第([" + CN_NUMERAL + r"]+)条(之[" + CN_SUFFIX + r"]+)?")


# ============================================================
# 中文数字互转（法条号的书写形式）
# ============================================================
def num_to_cn(n: int) -> str:
    """1079 -> 一千零七十九；133 -> 一百三十三。"""
    if n < 0:
        return str(n)
    if n < 10:
        return CN_DIGITS[n]
    if n < 20:
        return "十" + (CN_DIGITS[n - 10] if n > 10 else "")
    if n < 100:
        t, o = divmod(n, 10)
        return CN_DIGITS[t] + "十" + (CN_DIGITS[o] if o else "")
    if n < 1000:
        h, r = divmod(n, 100)
        s = CN_DIGITS[h] + "百"
        if r == 0:
            return s
        return s + ("零" + CN_DIGITS[r] if r < 10 else num_to_cn(r))
    if n < 10000:
        th, r = divmod(n, 1000)
        s = CN_DIGITS[th] + "千"
        if r == 0:
            return s
        return s + ("零" + num_to_cn(r) if r < 100 else num_to_cn(r))
    return str(n)


def cn_to_num(s: str) -> Optional[int]:
    """把「一百三十三」「133」都解析成 133。"""
    s = s.strip().replace("〇", "零")
    if s.isdigit():
        return int(s)
    if not s:
        return None
    section, number = 0, 0
    for ch in s:
        if ch in CN_NUM:
            number = CN_NUM[ch]
        elif ch in CN_UNIT:
            unit = CN_UNIT[ch]
            section += (number or 1) * unit
            number = 0
        else:
            return None
    return section + number


def normalize_article(text: str) -> Optional[str]:
    """「第133条」/「第一百三十三条之一」->「第一百三十三条(之一)」统一形式。"""
    m = ARTICLE_RE.search(text or "")
    if not m:
        return None
    n = cn_to_num(m.group(1))
    if n is None:
        return None
    return "第%s条%s" % (num_to_cn(n), m.group(2) or "")


# ============================================================
# 法律名解析
# ============================================================
# 口语简称 -> 库中正式名称的关键片段
LAW_ALIAS = {
    "刑法": "中华人民共和国刑法",
    "民法典": "中华人民共和国民法典",
    "民事诉讼法": "中华人民共和国民事诉讼法",
    "刑事诉讼法": "中华人民共和国刑事诉讼法",
    "公司法": "中华人民共和国公司法",
    "劳动法": "中华人民共和国劳动法",
    "劳动合同法": "中华人民共和国劳动合同法",
    "治安管理处罚法": "中华人民共和国治安管理处罚法",
    "道路交通安全法": "中华人民共和国道路交通安全法",
    "行政处罚法": "中华人民共和国行政处罚法",
    "行政诉讼法": "中华人民共和国行政诉讼法",
    "消费者权益保护法": "中华人民共和国消费者权益保护法",
    "社会保险法": "中华人民共和国社会保险法",
    "宪法": "中华人民共和国宪法",
}


def extract_law(question: str) -> Optional[str]:
    """从问题里识别法律名，长名优先（避免"劳动法"抢先匹配"劳动合同法"）。"""
    if not question:
        return None
    for alias in sorted(LAW_ALIAS, key=len, reverse=True):
        if alias in question:
            return LAW_ALIAS[alias]
    return None


class MetadataService:
    """基于 MySQL 结构化索引的精确召回。"""

    def __init__(self):
        self.milvus = get_milvus()
        self.collection = settings.MILVUS_COLLECTION

    # ---------------- 建索引 ----------------
    @staticmethod
    def rebuild_index(db: Session, records: List[Dict]) -> int:
        """用数据集记录重建 MySQL 法条索引表。"""
        db.query(LawIndex).delete()
        seen, rows = set(), []
        for r in records:
            meta = r.get("meta") or {}
            law, article = meta.get("law"), meta.get("article")
            if not law or not article:
                continue
            key = (law, article)
            if key in seen:
                continue
            seen.add(key)
            rows.append(LawIndex(
                role_id=r.get("role_id", "lawyer"),
                law_name=law,
                article=article,
                chapter=(meta.get("chapter") or "")[:255],
                doc_ref=r.get("doc_id", ""),
                source=r.get("source", ""),
            ))
        db.add_all(rows)
        db.flush()
        return len(rows)

    # ---------------- 召回 ----------------
    def search(self, db: Session, question: str, role_id: str = "lawyer",
               limit: int = 5) -> List[Dict]:
        """按法条号精确召回。识别不到编号时返回空，不影响其他路。"""
        article = normalize_article(question)
        if not article:
            return []
        law = extract_law(question)

        try:
            stmt = select(LawIndex).where(LawIndex.article == article)
            if law:
                stmt = stmt.where(LawIndex.law_name == law)
            else:
                stmt = stmt.where(LawIndex.role_id == role_id)
            hits = db.execute(stmt.limit(limit)).scalars().all()
        except Exception as e:                              # pragma: no cover
            logger.warning("MySQL 法条索引查询失败: %s", e)
            return []

        if not hits:
            return []
        logger.debug("元数据路命中 %s 条: %s %s", len(hits),
                     hits[0].law_name, article)

        # 用 (法律名, 条文号) 去 Milvus 取原文
        out = []
        for h in hits:
            expr = expr_eq(law=h.law_name, article=h.article)
            try:
                rows = self.milvus.query(self.collection, expr,
                                         output_fields=["text", "source",
                                                        "title", "doc_type"],
                                         limit=1)
            except Exception as e:                          # pragma: no cover
                logger.warning("按 %s %s 取原文失败: %s", h.law_name, h.article, e)
                continue
            for row in rows:
                out.append({
                    "text": row.get("text", ""),
                    "title": "《%s》%s" % (h.law_name, h.article),
                    "source": row.get("source", ""),
                    "doc_type": row.get("doc_type", "statute"),
                    "law": h.law_name,
                    "article": h.article,
                    "route": "metadata",
                })
        return out


_service: Optional[MetadataService] = None


def get_metadata_service() -> MetadataService:
    global _service
    if _service is None:
        _service = MetadataService()
    return _service
