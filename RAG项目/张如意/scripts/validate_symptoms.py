# -*- coding: utf-8 -*-
"""症状覆盖层自检：data/processed/symptoms.jsonl 的挂卡与字段校验。

症状条目是**外部来源**（期刊/农科院/科普）挂在国标卡上的覆盖层，
三类错误都会造成实际损害：

  1. **挂空**（card_id 不存在）→ 症状静默丢失，农户问症状永远查不到
  2. **挂了不进索引的卡**（附录A 36 张 / 光杆标题 8 张）→ 同样静默丢失
  3. **挂错卡**（pest 与目标卡 subtype 对不上）→ 把 A 病的症状安到 B 病头上，
     检索命中 B 卡 → 农户拿到**错药**

用法：
    python scripts/validate_symptoms.py        # 退出码 0=通过，1=有错，2=缺文件
"""
import json
import os
import re
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(ROOT, "data", "processed")
CHUNKS = os.path.join(ROOT, "data", "chunks", "chunks.jsonl")
SYMPTOMS = os.path.join(PROCESSED, "symptoms.jsonl")
# 三份卡片文件（与 validate_cards.py 口径一致：247 张）
CARD_FILES = ["appendixB_cards.jsonl", "body_cards.jsonl", "appendixA_cards.jsonl"]

# part 九值枚举（spec §3.2）。扩枚举必须同步 tests/test_symptoms.py 与本表。
PART_ENUM = ("叶片", "茎", "茎基部", "根", "鳞茎", "果实", "心叶", "花", "全株")


def load_cards():
    """三份卡片 JSONL → {card_id: 卡}。"""
    cards = {}
    for fn in CARD_FILES:
        p = os.path.join(PROCESSED, fn)
        if not os.path.exists(p):
            continue
        for line in open(p, encoding="utf-8"):
            if line.strip():
                c = json.loads(line)
                cards[c["card_id"]] = c
    return cards


def load_chunks(path=None):
    """chunks.jsonl → 进索引的 chunk_id 集合（203 张里实际被检索的那些）。"""
    path = path or CHUNKS
    if not os.path.exists(path):
        return set()
    return {json.loads(l)["chunk_id"] for l in open(path, encoding="utf-8") if l.strip()}


def load_symptoms(path=None):
    """症状覆盖层 → 条目列表；文件不存在返回空列表。"""
    path = path or SYMPTOMS
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


# 词元切分：病虫害名在卡片里可能以 、｜: 等分隔（subtype 与 quote 都按此切）
_TOKEN_SPLIT = re.compile(r"[、,，;；:：|｜/]+")


def _tokens(text):
    """把一段卡片文本切成独立词元集合（去空白、丢空串）。"""
    return {t.strip() for t in _TOKEN_SPLIT.split(text or "") if t.strip()}


def validate(rows, card_ids, chunk_ids, cards=None):
    """校验条目，返回错误字符串列表（空列表 = 通过）。

    rows      : 症状条目列表
    card_ids  : 全部卡片 id 集合（247 张）
    chunk_ids : 进索引的 chunk id 集合（203 张）
    cards     : {card_id: 卡}，给「挂错卡」检查用；None 时跳过该项
    """
    errs = []
    for i, r in enumerate(rows):
        tag = r.get("symptom_id") or f"第 {i+1} 行"
        cid = r.get("card_id")
        # ① 卡号必须真实存在
        if cid not in card_ids:
            errs.append(f"{tag}: card_id {cid!r} 在卡片里不存在")
            continue
        # ② 卡必须真的进了索引，否则症状静默丢失
        if cid not in chunk_ids:
            errs.append(f"{tag}: card_id {cid!r} 不进索引（附录A 或光杆标题卡），症状会丢失")
        # ③ 症状文本非空
        if not (r.get("symptom_text") or "").strip():
            errs.append(f"{tag}: symptom_text 为空")
        # ④ part 必须在枚举内
        for p in r.get("part") or []:
            if p not in PART_ENUM:
                errs.append(f"{tag}: part {p!r} 不在枚举 {PART_ENUM}")
        # ⑤ pest 必须是目标卡里的**独立词元**（subtype 或 source.quote）
        #   为什么不用子串（2026-09-28 用户裁决）——子串两个方向都错：
        #     假阳性：「疫病」是「早疫病、晚疫病」的子串，会让疫病症状挂上早/晚疫
        #             方案卡，正检索错药
        #     假阴性：正文卡的 subtype 就是章节号（b6.2 → "6.2"），病虫害名只出现在
        #             quote 里，子串法会把合法条目全部拒收
        if cards is not None:
            card = cards.get(cid) or {}
            sub = card.get("subtype") or ""
            quote = (card.get("source") or {}).get("quote") or ""
            pest = (r.get("pest") or "").strip()
            if pest and pest not in (_tokens(sub) | _tokens(quote)):
                errs.append(f"{tag}: pest {pest!r} 不是目标卡的独立词元"
                            f"（subtype={sub!r}；quote 前 40 字={quote[:40]!r}）")
        # ⑥ source 必须是对象（来源不可回查的条目等于没有来源）
        if not isinstance(r.get("source"), dict):
            errs.append(f"{tag}: source 不是对象")
    # ⑦ symptom_id 不许重复（重复会让复核清单与快照文件名互相覆盖）
    dup = sorted(k for k, v in Counter(r.get("symptom_id") for r in rows).items()
                 if k and v > 1)
    if dup:
        errs.append(f"symptom_id 重复: {dup}")
    return errs


def collect(path=None):
    """给 pytest 用的整包：返回 (条目列表, 错误列表, 统计)。"""
    cards = load_cards()
    chunks = load_chunks()
    rows = load_symptoms(path)
    errs = validate(rows, set(cards), chunks, cards)

    # 统计时对 source 做类型防护：source 坏掉的条目走的是错误列表，不该拖垮统计
    def _kind(r):
        src = r.get("source")
        return src.get("kind") if isinstance(src, dict) else None

    stats = {"n": len(rows), "by_crop": dict(Counter(r.get("crop") for r in rows)),
             "by_kind": dict(Counter(_kind(r) for r in rows))}
    return rows, errs, stats


def main():
    """CLI：打印条目数与错误明细。返回 0=通过，1=有错，2=缺文件。"""
    if not os.path.exists(SYMPTOMS):
        print(f"缺 {SYMPTOMS}（先采条目，见实施计划 Task 3）")
        return 2
    rows, errs, stats = collect()
    print(f"症状条目 {stats['n']} 条")
    print(f"  按作物: {stats['by_crop']}")
    print(f"  按来源: {stats['by_kind']}")
    if errs:
        print(f"\n✗ 校验未通过，{len(errs)} 个错误：")
        for e in errs:
            print(f"  - {e}")
        return 1
    print("\n✓ 校验通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
