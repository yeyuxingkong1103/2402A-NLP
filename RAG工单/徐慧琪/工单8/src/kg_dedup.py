# 工单编号：人工智能NLP-RAG-基于GraphRAG实现金融问答
"""
模块：实体消歧与归一（硬性要求1「多文档实体对齐」）

三类消歧规则，按优先级：
  1. 公司别名表（000001 风格：全称/简称/英文名/曾用名 → 唯一规范名）；
  2. 规则归一（空白、全半角、括号全角→半角、公司后缀）；
  3. 同类型同名即同实体（由 entity_id = `{类型}:{规范名}` 天然保证）。

跨文档融合与「中国平安 vs 平安银行」这类易混实体的区分，都靠这里的规范名。
"""
from __future__ import annotations

import unicodedata

from src.kg_dict import COMPANY_TABLE
from src.kg_schema import Entity, Relation, make_entity_id

# 公司后缀：归一为规范简称时剥离
# 注：不把「（集团）/(集团)」列为后缀——括号集团标记是名称的一部分，
#     含集团的法定全称由别名表直接映射（如「中国平安保险（集团）股份有限公司」）。
_COMPANY_SUFFIXES = (
    "股份有限公司", "有限责任公司", "有限公司", "股份公司", "公司", "集团",
)


def _to_halfwidth(text: str) -> str:
    """全角 → 半角（NFKC 归一，ASCII 部分）。"""
    return unicodedata.normalize("NFKC", text)


# 别名 → 规范简称（构建一次，全局复用）
# 键同时登记原文与 NFKC 归一形式：查表前 surface 会被 NFKC 归一，
# 若只登记原文，全角括号全称（「中国平安保险（集团）股份有限公司」）将查不到。
_ALIAS_TO_CANONICAL: dict[str, str] = {}
for _canon, _meta in COMPANY_TABLE.items():
    for _surface in (_canon, _meta["full"], *_meta["aliases"]):
        _ALIAS_TO_CANONICAL[_surface] = _canon
        _ALIAS_TO_CANONICAL[_to_halfwidth(_surface)] = _canon


def normalize_name(name: str, etype: str) -> str:
    """规范名归一。

    · 去首尾空白、内部空白压缩；
    · NFKC（全角字母/数字/括号 → 半角）；
    · Company 类型剥离公司后缀（「平安银行股份有限公司」→「平安银行」）。
    """
    out = _to_halfwidth((name or "").strip())
    out = "".join(out.split())

    if etype == "Company":
        canon = canonical_company(out)
        if canon:
            return canon
        for suf in _COMPANY_SUFFIXES:
            if out.endswith(suf) and len(out) > len(suf):
                out = out[: -len(suf)]
                break
    return out


def canonical_company(surface: str) -> str | None:
    """任意公司表述 → 规范简称；未知返回 None。"""
    if not surface:
        return None
    probe = _to_halfwidth(surface.strip())
    if probe in _ALIAS_TO_CANONICAL:
        return _ALIAS_TO_CANONICAL[probe]
    # 退化匹配：去空白后再试一次
    probe2 = "".join(probe.split())
    return _ALIAS_TO_CANONICAL.get(probe2)


def resolve_id(etype: str, raw_name: str) -> str:
    """原始表面名 → 稳定 entity_id（应用归一规则）。"""
    return make_entity_id(etype, normalize_name(raw_name, etype))


def dedup_entities(entities: list[Entity]) -> dict[str, Entity]:
    """合并同义实体。

    合并规则：以 `resolve_id` 后的 ID 为键；重复时
      · 别名取并集；
      · 属性取并集（已有值不覆盖，缺失值补全）；
      · 保留先出现的 name。
    """
    merged: dict[str, Entity] = {}
    for e in entities:
        eid = resolve_id(e.type, e.name)
        # 别名也走归一，避免「平安银行股份有限公司」与「平安银行」并存
        aliases = {e.name, e.entity_id.split(":", 1)[-1]}
        aliases.update(e.aliases or [])
        aliases.discard(eid.split(":", 1)[-1])

        if eid not in merged:
            new = Entity(entity_id=eid, name=normalize_name(e.name, e.type),
                         type=e.type, aliases=sorted(a for a in aliases if a),
                         attrs=dict(e.attrs))
            merged[eid] = new
        else:
            cur = merged[eid]
            for a in aliases:
                if a and a not in cur.aliases:
                    cur.aliases.append(a)
            for k, v in e.attrs.items():
                if k not in cur.attrs or cur.attrs[k] in (None, "", []):
                    if v not in (None, "", []):
                        cur.attrs[k] = v
    return merged


def remap_relations(relations: list[Relation],
                    merged: dict[str, Entity]) -> list[Relation]:
    """把边的端点重写为归一后的 ID；端点不存在于 merged 的边丢弃（防悬空引用）。"""
    out: list[Relation] = []
    seen: set[tuple] = set()
    for r in relations:
        head = resolve_id(r.head_id.split(":", 1)[0], r.head_id.split(":", 1)[-1])
        tail = resolve_id(r.tail_id.split(":", 1)[0], r.tail_id.split(":", 1)[-1])
        if head not in merged or tail not in merged:
            continue
        # source_doc 必须入键：同页同证据前缀但来自不同文档的两条边是不同来源，
        # 合并会丢掉一个 provenance 溯源点（「返回答案所在的文本块」的依据）。
        key = (head, r.rel_type, tail, r.source_doc, r.evidence[:40], r.page)
        if key in seen:
            continue
        seen.add(key)
        out.append(Relation(
            head_id=head, rel_type=r.rel_type, tail_id=tail,
            evidence=r.evidence, source_doc=r.source_doc,
            page=r.page, confidence=r.confidence, attrs=dict(r.attrs)))
    return out
