# app/core/validate_service.py
"""生成结果校验：核对回答里的法条引用是否真实存在。

**为什么需要**：大模型有时会「顺手」写出一条读起来很像真的、知识库里
根本没有的法条号（例如把《刑法》第七十二条的缓刑条件安到第七十四条上）。
在心理与金融场景里，一句不准的话是建议不专业；在法律场景里，
一个编造的条文号可能让人据此行事。所以生成之后要有第二道关。

校验方式：把回答里出现的「法律名 + 条文号」抽出来，逐条到 MySQL 的
law_index（法条索引表）里核对。核对不上的，标记出来供上层处理。

**注意**：核对不上不等于模型错了——知识库只覆盖 21 部法规，
模型提到库外的法规（如《治安管理处罚法》之外的行政法规）是正常的。
因此这里只做「提示」，不擅自改写回答。
"""
import re
from typing import Dict, List, Optional

import logging

logger = logging.getLogger(__name__)

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.metadata_service import (CN_NUMERAL, CN_SUFFIX, LAW_ALIAS,
                                       normalize_article)
from app.models.tables import LawIndex

# 法条号的数字字符类与「之一」后缀共用 metadata_service 的定义，
# 避免几处正则各写一份、改的时候漏掉
_ART = r"第[" + CN_NUMERAL + r"]+条(?:之[" + CN_SUFFIX + r"]+)?"

# 《中华人民共和国刑法》第一百三十三条 / 《刑法》第一百三十三条之一
QUOTED_RE = re.compile(r"《([^》]{2,40})》\s*(" + _ART + r")")
# 不带书名号：中华人民共和国刑法第一百三十三条 / 刑法第133条 / 民法典第1079条
# 结尾词必须带上「典」——《民法典》不以此前的「法」结尾，漏了它就抽不出来
BARE_RE = re.compile(
    r"([一-龥]{2,20}?(?:法|典|条例|解释|规定|细则|办法|决定))\s*(" + _ART + r")")


def _canonical_law(name: str) -> Optional[str]:
    """把回答里的法律名归一化到库中正式名称。"""
    name = (name or "").strip()
    if not name:
        return None
    # 已经是全称
    if name.startswith("中华人民共和国"):
        return name
    # 简称查表：长名优先，避免「劳动法」抢先匹配「劳动合同法」
    for alias in sorted(LAW_ALIAS, key=len, reverse=True):
        if alias in name:
            return LAW_ALIAS[alias]
    return None


def extract_citations(answer: str) -> List[Dict[str, str]]:
    """从回答里抽取法条引用，返回 [{raw, law, article}]（已去重）。"""
    found, seen = [], set()

    for m in QUOTED_RE.finditer(answer or ""):
        law, art = _canonical_law(m.group(1)), normalize_article(m.group(2))
        if law and art and (law, art) not in seen:
            seen.add((law, art))
            found.append({"raw": m.group(0).strip(), "law": law, "article": art})

    # 书名号形式已覆盖的，不再用宽松式重复抽取
    for m in BARE_RE.finditer(answer or ""):
        law, art = _canonical_law(m.group(1)), normalize_article(m.group(2))
        if law and art and (law, art) not in seen:
            seen.add((law, art))
            found.append({"raw": m.group(0).strip(), "law": law, "article": art})

    return found


class ValidateService:
    """生成结果的后处理校验。"""

    def check_citations(self, answer: str, db: Session) -> Dict:
        """核对回答中的法条引用，返回校验结果。"""
        citations = extract_citations(answer)
        if not citations:
            return {"checked": 0, "verified": 0, "unverified": [],
                    "coverage_note": None}

        verified, unverified = [], []
        for c in citations:
            if self._exists(db, c["law"], c["article"]):
                verified.append(c)
            else:
                unverified.append(c)

        if unverified:
            logger.warning("回答中有 %s 条法条引用未在知识库中找到: %s",
                           len(unverified),
                           [c["raw"][:30] for c in unverified[:3]])

        return {
            "checked": len(citations),
            "verified": len(verified),
            "unverified": unverified,
            # 库里只覆盖 21 部法规，提及库外法规属正常，这句话用于向用户解释
            "coverage_note": ("知识库覆盖 21 部常用法规，"
                              "未收录的法规无法核对，不代表引用有误"
                              if unverified else None),
        }

    @staticmethod
    def _exists(db: Session, law: str, article: str) -> bool:
        try:
            row = db.execute(
                select(LawIndex.id)
                .where(LawIndex.law_name == law)
                .where(LawIndex.article == article)
                .limit(1)).first()
            return row is not None
        except Exception as e:                              # pragma: no cover
            logger.warning("法条核对查询失败 %s %s: %s", law, article, e)
            return False

    # ---------------- 后处理 ----------------
    @staticmethod
    def build_note(result: Dict) -> str:
        """把校验结果转成给用户看的提示（无问题则返回空串）。"""
        bad = result.get("unverified") or []
        if not bad:
            return ""
        items = "、".join(c["raw"] for c in bad[:3])
        more = "等 %d 处" % len(bad) if len(bad) > 3 else ""
        return ("\n\n---\n提示：以上回答引用的 %s%s 未能在本系统知识库中核对到，"
                "请以现行法律法规原文为准。" % (items, more))

    def postprocess(self, answer: str, db: Session,
                    append_note: bool = True) -> tuple:
        """后处理入口：校验并（可选）追加提示。返回 (最终回答, 校验结果)。"""
        result = self.check_citations(answer, db)
        if append_note and result.get("unverified"):
            return answer + self.build_note(result), result
        return answer, result


_service: Optional[ValidateService] = None


def get_validate_service() -> ValidateService:
    global _service
    if _service is None:
        _service = ValidateService()
    return _service
