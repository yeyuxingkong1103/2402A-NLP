# -*- coding: utf-8 -*-
"""数据质量体检：量一下知识卡到底脏不脏，别拍脑袋说「不用清洗」。

只诊断、不改数据。读取 data/processed 下的三份知识卡 JSONL（附录B、正文、附录A），
对全库做 8 项检查：页眉页脚残留、上标格式、数值写法一致性、全半角混用、
缺失值分布、重复内容、异常字符、空白与换行。纯打印报告，无输出文件。
"""
import json
import os
import re
from collections import Counter

P = "data/processed"   # 知识卡所在目录
# 载入三个卡源的全部知识卡（附录B、正文、附录A）
cards = []
for f in ("appendixB_cards.jsonl", "body_cards.jsonl", "appendixA_cards.jsonl"):
    for line in open(os.path.join(P, f), encoding="utf-8"):
        cards.append(json.loads(line))   # JSONL：一行一个 JSON 卡片
blob = "\n".join(json.dumps(c, ensure_ascii=False) for c in cards)  # 全库文本拼接，供正则统计


def head(n, title):
    """打印体检报告的小节标题分隔条。参数 n=小节序号，title=小节名。"""
    print("\n" + "=" * 72)
    print(f"{n}. {title}")
    print("=" * 72)


head(1, "页眉页脚残留")
# 统计标准号在全文的出现次数（合法位置只有 std_no / source 字段，混进正文即页眉残留）
h = re.findall(r'GB/Z\s*265\d\d[—\-–]2011', blob)
print(f"   全库出现 {len(h)} 次（合法：只在 std_no / source 字段）")
in_body = [c["card_id"] for c in cards
           if re.search(r'GB/Z\s*265\d\d', json.dumps(c.get("measures"), ensure_ascii=False))]
print(f"   混进 measures 正文的: {len(in_body)} 张 {'✓' if not in_body else '⚠️'}")

head(2, "上标与格式残留")
# 四种格式问题各配一个正则：m2 未还原上标、MinerU LaTeX 残留、667m/hm 后缺 ²
for pat, name in [(r'\bm2\b', "m2（上标未还原）"),
                  (r'\$m\^', "$m^2$（MinerU LaTeX 格式）"),
                  (r'667m(?![²2])', "667m 后缺 ²"),
                  (r'hm(?![²2])', "hm 后缺 ²")]:
    n = len(re.findall(pat, blob))
    print(f"   {'⚠️' if n else '✓'} {name}: {n} 处")

head(3, "数值格式一致性")
sp = Counter()
for c in cards:
    for ch in c.get("chemicals", []):
        for k in ("pre_harvest_interval", "interval_days"):
            v = ch.get(k)
            if v:
                # 抽「数字+d」写法，按有无空格分桶（20 d vs 20d）
                for m in re.findall(r'\d+(?:\.\d+)?\s*d(?![a-z])', str(v)):
                    sp["有空格(20 d)" if " " in m else "无空格(20d)"] += 1
print(f"   卡片字段里的天数写法: {dict(sp)}")
raw = Counter()
for c in cards:
    for ch in c.get("chemicals", []):
        for m in re.findall(r'\d+(?:\.\d+)?\s*d(?![a-zA-Z])', ch.get("raw", "")):
            raw[m] += 1
print(f"   raw 原文里的写法抽样: {list(raw)[:6]}")

head(4, "全角/半角混用")
# 逐个标点统计全角/半角出现次数，混用多说明清洗有遗漏
for p, name in [("，", "全角逗号"), (",", "半角逗号"), ("（", "全角括号"), ("(", "半角括号"),
                ("、", "顿号"), ("；", "全角分号"), (";", "半角分号")]:
    print(f"   {name}: {blob.count(p)}")

head(5, "缺失值分布")
mf = Counter()
for c in cards:
    for x in c.get("missing_fields", []):   # 把 chemicals[0].xxx 的下标抹平再统计
        mf[re.sub(r'\[\d+\]', '[]', x)] += 1
print(f"   missing_fields: {dict(mf) if mf else '无'}")
# 允许为 null 的可选字段清单（其余字段为 null 视为异常）
OPTIONAL = ("symptoms", "fertilizer", "precautions", "trigger", "growth_stage",
            "crop_alias", "pest_kind", "record_form", "crop")
nulls = Counter()
for c in cards:
    for k, v in c.items():
        if v is None and k not in OPTIONAL:  # 只统计非可选字段为 null 的情况
            nulls[k] += 1
print(f"   非可选字段为 null: {dict(nulls) if nulls else '无 ✓'}")

head(6, "重复内容")
t = Counter(c["title"] for c in cards)          # 按标题分组计数
dup = {k: v for k, v in t.items() if v > 1}     # 只留出现次数 > 1 的组
print(f"   标题重复 {len(dup)} 组:")
for k, v in list(dup.items())[:8]:              # 最多展示 8 组
    print(f"      ×{v}  {k}")
q = Counter(c["source"]["quote"] for c in cards)
# 引文重复且长度 > 25 才算（短引文如「表A.1」重复属正常）
d2 = {k: v for k, v in q.items() if v > 1 and len(k) > 25}
print(f"   source.quote 重复 {len(d2)} 组")

head(7, "异常字符")
# 控制字符：ASCII < 32 且排除换行符
ctrl = Counter(repr(ch) for ch in blob if ord(ch) < 32 and ch != "\n")
print(f"   控制字符: {dict(ctrl) if ctrl else '无 ✓'}")
# 全角 ASCII 区（FF00–FFEF）与零宽空格/零宽不折行空格
weird = Counter(ch for ch in blob if 0xFF00 <= ord(ch) <= 0xFFEF or ord(ch) in (0x200B, 0xFEFF))
print(f"   全角ASCII/零宽字符: {dict(weird) if weird else '无 ✓'}")

head(8, "空白与换行")
print(f"   连续空格(≥2): {len(re.findall(r'  +', blob))} 处")
print(f"   行首尾空白: {len(re.findall(r'(?m)^ +| +$', blob))} 处")  # 多行模式，逐行查首尾
print(f"   空字符串字段: {len(re.findall(r':\s*\"\"', blob))} 处")   # 形如 "key": "" 的字段
