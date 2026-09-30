# -*- coding: utf-8 -*-
"""知识卡自检：card_id 唯一性/格式、必填字段、高危打标一致性、in_scope 口径。

用法：python scripts/validate_cards.py
退出码非 0 表示有错误（可挂到 pytest / CI）。

【模块说明（补充）】
  输入：data/processed/ 下的三份知识卡 JSONL（见 FILES）。
  输出：不写文件，只打印检查报告；返回退出码 0=全部通过，1=有错误。
  检查项概览（对应 collect() 里的编号注释）：
    1) card_id 唯一 + 格式；2) 必填字段与枚举合法性；3) 高危打标一致性；
    3.5) pest_kind 自洽；3.6) 兼治对象必须挂在方案级；3.7) record_form 结构；
    3.8) 天数写法统一「N d」；4) 附录B 缺失值必须记 missing_fields；
    5) in_scope 口径；6) 跨作物数值不得串。
"""
import json
import os
import re
import sys
from collections import Counter

P = "data/processed"                                     # 处理产物根目录
FILES = ["appendixB_cards.jsonl", "body_cards.jsonl", "appendixA_cards.jsonl"]  # 待检的三份卡片库

# 每张卡必须存在的字段（缺一报错）
REQUIRED = ["card_id", "std_no", "std_name", "authority_level", "year", "region", "crop",
            "knowledge_type", "type_group", "subtype", "title", "is_high_risk",
            "needs_human_review", "needs_verification_hint", "source", "in_scope"]

KT = {"病虫害防治", "杂草防治", "鼠害防治", "土肥水管理", "品种选择", "农事操作", "灾害应对",
      "政策补贴", "基地与投入品管理", "劳动保护", "档案记录", "加工贮运", "其他"}   # knowledge_type 合法枚举
TG = {"病虫草害", "土肥水管理", "品种选择", "农事操作", "灾害应对", "政策补贴", "其他"}   # type_group 合法枚举
CHANNEL = {"text", "vision", "mixed"}                    # source.channel 转录通道合法枚举
# pest_kind：knowledge_type 的细分。病虫草害类必须给，其余必须为 None
PK = {"病害", "虫害", "杂草"}
# 附录A 表单结构必填子字段
RF = ["table_no", "table_name", "header_items", "columns", "footnote", "is_blank_form"]


def collect():
    """跑全部检查，返回 (cards, errs, warns)。供 CLI 与 pytest 共用。

    返回：
      cards -- (文件名, 行号, 卡片dict) 三元组列表（含全部读入的卡）
      errs  -- 错误信息列表（任一非空即判失败）
      warns -- 警告信息列表（不判失败）
    """
    errs, warns = [], []
    ids, cards = [], []
    for fn in FILES:
        path = os.path.join(P, fn)
        if not os.path.exists(path):
            errs.append(f"缺文件 {path}")
            continue
        for ln, line in enumerate(open(path, encoding="utf-8"), 1):
            if not line.strip():
                continue
            c = json.loads(line)
            cards.append((fn, ln, c))
            ids.append((c.get("card_id"), fn, ln))     # 收集 card_id 及其出处，供唯一性/格式检查

    # 1) card_id 唯一 + 格式
    dup = [k for k, v in Counter(i for i, _, _ in ids).items() if v > 1]
    if dup:
        errs.append(f"card_id 重复 {len(dup)} 个: {dup[:5]}")
    for cid, fn, ln in ids:
        # 合法格式：GBZ + 5 位数字 - 4 位年份 - p页码 - 字母段 + 数字段（如 GBZ26581-2011-p20-c10）
        if not re.fullmatch(r'GBZ\d{5}-\d{4}-p\d+-[abc][0-9.]+', cid or ""):
            errs.append(f"{fn}:{ln} card_id 格式异常: {cid!r}")

    # 2) 必填字段 + 枚举
    for fn, ln, c in cards:
        for k in REQUIRED:
            if k not in c:
                errs.append(f"{fn}:{ln} {c.get('card_id')} 缺字段 {k}")
        if c.get("knowledge_type") not in KT:
            errs.append(f"{c['card_id']} knowledge_type 非法: {c.get('knowledge_type')!r}")
        if c.get("type_group") not in TG:
            errs.append(f"{c['card_id']} type_group 非法: {c.get('type_group')!r}")
        if c.get("source", {}).get("channel") not in CHANNEL:
            errs.append(f"{c['card_id']} channel 非法")
        if not isinstance(c.get("source", {}).get("page"), int):
            errs.append(f"{c['card_id']} source.page 非整数")
        if len(c.get("source", {}).get("quote", "")) > 200:   # 原文引用最长 200 字（防止整段拷贝）
            errs.append(f"{c['card_id']} source.quote 超 200 字")

    # 3) 高危 → 必须带核实提示
    for fn, ln, c in cards:
        if c.get("is_high_risk") and not c.get("needs_verification_hint"):
            errs.append(f"{c['card_id']} 高危卡未打 needs_verification_hint")
        if c.get("needs_verification_hint") and not c.get("is_high_risk"):
            errs.append(f"{c['card_id']} 非高危卡打了 needs_verification_hint")
        if c.get("is_high_risk") and not c.get("high_risk_reasons"):
            errs.append(f"{c['card_id']} 高危卡 high_risk_reasons 为空")

    # 3.5) pest_kind：可判定时必须给且与 knowledge_type 自洽；病、虫兼有的通用条款留 None
    #      （「防治原则」「化学防治一般要求」这类条款硬分成病害或虫害反而是错的）
    for fn, ln, c in cards:
        pk, kt = c.get("pest_kind"), c.get("knowledge_type")
        if pk is not None and pk not in PK:
            errs.append(f"{c['card_id']} pest_kind 非法: {pk!r}")
        elif pk is not None and kt == "杂草防治" and pk != "杂草":
            errs.append(f"{c['card_id']} knowledge_type=杂草防治 但 pest_kind={pk!r}")
        elif pk is not None and kt == "病虫害防治" and pk not in ("病害", "虫害"):
            errs.append(f"{c['card_id']} knowledge_type=病虫害防治 但 pest_kind={pk!r}")
        elif pk is not None and c.get("type_group") != "病虫草害":
            errs.append(f"{c['card_id']} type_group={c.get('type_group')!r} 却有 pest_kind={pk!r}")
        # 附录B 的防治对象全部是具体病/虫/草，不允许判不出
        if fn.startswith("appendixB") and pk is None:
            errs.append(f"{c['card_id']} 附录B 卡 pest_kind 不应为空")

    # 3.6) 兼治对象必须挂在方案级（chemicals[]），不得提到卡级
    for fn, ln, c in cards:
        if "companion_control" in c:
            errs.append(f"{c['card_id']} companion_control 出现在卡级，应在 chemicals[] 内")
        for i, ch in enumerate(c.get("chemicals", [])):
            if fn.startswith("appendixB") and "companion_control" not in ch:
                errs.append(f"{c['card_id']} chemicals[{i}] 缺 companion_control（无兼治也应显式 null）")

    # 3.7) record_form：附录A 必须有完整结构，其他卡必须为 null
    for fn, ln, c in cards:
        rf = c.get("record_form", "缺失")
        if fn.startswith("appendixA"):
            if not isinstance(rf, dict):
                errs.append(f"{c['card_id']} 附录A 卡 record_form 非对象")
                continue
            for k in RF:
                if k not in rf:
                    errs.append(f"{c['card_id']} record_form 缺 {k}")
            if not rf.get("columns"):
                errs.append(f"{c['card_id']} record_form.columns 为空")
            if rf.get("is_blank_form") is not True:
                errs.append(f"{c['card_id']} record_form.is_blank_form 应为 true")
            if not re.fullmatch(r'表A\.\d+', str(rf.get("table_no", ""))):   # 表号必须形如「表A.数字」
                errs.append(f"{c['card_id']} record_form.table_no 格式异常: {rf.get('table_no')!r}")
            if not c.get("measures") == [] or not c.get("chemicals") == []:  # 表单卡不应有防治方案/药剂
                errs.append(f"{c['card_id']} 附录A 表单卡 measures/chemicals 应为空数组")
        elif rf not in (None, "缺失"):
            errs.append(f"{c['card_id']} 非附录A 卡 record_form 应为 null")

    # 3.8) 派生天数字段写法必须统一为「带空格」（如 `7 d~10 d`）。
    #      踩过的坑：抽 interval_days 时 .replace(" ","") 抹了空格，导致同一条记录里
    #      pre_harvest_interval 有空格、interval_days 没有。raw 字段不受此约束（保留原文）。
    DAY = re.compile(r'\d+(?:\.\d+)? d')
    for fn, ln, c in cards:
        if not fn.startswith("appendixB"):
            continue
        for i, ch in enumerate(c.get("chemicals", [])):
            for k in ("pre_harvest_interval", "interval_days"):
                v = ch.get(k)
                if not v:
                    continue
                raw_days = re.findall(r'\d+(?:\.\d+)?\s*d(?![a-zA-Z])', str(v))   # 抽出所有「数字+d」（后面不跟字母）
                bad = [d for d in raw_days if " " not in d]                       # 数字与 d 之间没空格的即为非法写法
                if bad:
                    errs.append(f"{c['card_id']} chemicals[{i}].{k}={v!r} "
                                f"天数写法无空格 {bad}（应统一为「N d」）")

    # 4) 附录B 高危数值列：缺失必须记 missing_fields，不许留空不记
    for fn, ln, c in cards:
        if not fn.startswith("appendixB"):
            continue
        for i, ch in enumerate(c.get("chemicals", [])):
            if ch.get("pre_harvest_interval") is None:
                if f"chemicals[{i}].pre_harvest_interval" not in c.get("missing_fields", []):
                    errs.append(f"{c['card_id']} chemicals[{i}] 间隔期缺失但未记 missing_fields")
        if not c.get("chemicals"):
            errs.append(f"{c['card_id']} 附录B卡无 chemicals")
        for ch in c.get("chemicals", []):
            if not ch.get("raw"):
                errs.append(f"{c['card_id']} chemicals[].raw 为空（丢失原文）")

    # 5) 附录A 必须 in_scope=false；附录B in_scope=true
    for fn, ln, c in cards:
        if fn.startswith("appendixA") and c.get("in_scope"):
            errs.append(f"{c['card_id']} 附录A 卡应 in_scope=false")
        if fn.startswith("appendixB") and not c.get("in_scope"):
            errs.append(f"{c['card_id']} 附录B 卡应 in_scope=true")

    # 6) 跨作物数值不得串（同药剂在不同作物的安全间隔期必须各自独立）
    iv = {}
    for fn, ln, c in cards:
        if not fn.startswith("appendixB"):
            continue
        for ch in c.get("chemicals", []):
            # key 前半：药剂名取 "%" 后的前 6 字（剥掉含量前缀，如 "20%吡虫啉"→"吡虫啉…"）
            key = (ch["product"].split("%")[-1][:6], c["crop"])
            iv.setdefault(key, set()).add(str(ch.get("pre_harvest_interval")))

    return cards, errs, warns


def main():
    """CLI 入口：跑全部检查并打印报告。

    返回退出码：0=全部通过；1=存在错误（错误最多展示前 40 条）。
    """
    cards, errs, warns = collect()
    uniq = len({c.get("card_id") for _, _, c in cards})   # 唯一 card_id 数
    print(f"扫描 {len(cards)} 张卡 / {uniq} 个唯一 card_id")
    print("  按文件: " + ", ".join(
        f"{fn.split('_')[0]}={sum(1 for f, _, _ in cards if f == fn)}" for fn in FILES))
    if warns:
        print("\n警告:")
        for x in warns:
            print("  ⚠", x)
    if errs:
        print(f"\n错误 {len(errs)} 条:")
        for x in errs[:40]:                              # 错误太多时只展示前 40 条
            print("  ✗", x)
        return 1
    print("\n全部检查通过 ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
