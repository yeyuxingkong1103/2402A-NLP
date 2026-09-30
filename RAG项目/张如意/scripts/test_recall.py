# -*- coding: utf-8 -*-
"""口语映射的召回自测（关键词版，非最终检索）。

用途：在还没有向量索引前，先验证「口语 -> 标准术语」这一步有没有用。
     不做语义检索，只做「映射前 vs 映射后，能命中多少张卡」的对比。

跑法：python scripts/test_recall.py

【模块说明（补充）】
  输入：
    - configs/colloquial_map.json               口语→标准术语映射表
    - data/processed/appendixB_cards.jsonl      附录 B 知识卡（in_scope=true 才入池）
    - data/processed/body_cards.jsonl           正文知识卡（in_scope=true 才入池）
  输出：只打印到 stdout——22 条农户问法的「映射命中 / 检索键 / 卡数」统计表，
        以及前 6 条问法命中的卡片抽样。不写文件。
  ⚠️ 阅读提示：本文件部分全局量（MAP/CARDS/SPOKEN_IDX/TERMS/QUERIES）若未定义
     单独运行会抛 NameError，以实际代码为准。
"""
import json
import os
import re

P = "data/processed"
MAP = json.load(open("configs/colloquial_map.json", encoding="utf-8"))  # 口语→标准术语映射表

# 只拿进检索的卡（附录A in_scope=false 已排除）
CARDS = []
for f in ("appendixB_cards.jsonl", "body_cards.jsonl"):
    for line in open(os.path.join(P, f), encoding="utf-8"):
        c = json.loads(line)
        if c["in_scope"]:                              # 附录A 表单卡 in_scope=false，不入检索池
            CARDS.append(c)


def searchable(c):
    """卡的检索文本（与后续 embedding 字段口径一致：标题+对象+适期+药剂）

    参数 c：一张知识卡 dict。返回：拼接后的检索文本字符串。
    """
    parts = [c["title"], c["subtype"], c.get("trigger") or "", c["crop"]]
    parts += [ch["product"] for ch in c.get("chemicals", [])]   # 逐条药剂名也拼进检索文本
    return " ".join(parts)


# 同一 spoken 可能有多条：带 crop 的（作物专属）与 crop=None 的（通用）。
# 应用时**作物专属优先，没有专属才回落到通用** —— 不能因为指定了作物就把通用全跳过
# （那样「起腻虫→蚜虫」这类通用口语一条都用不上），也不能无条件用通用的
# （那样「辣椒蚜虫」会被「蚜虫→瓜蚜」误映射成瓜蚜）。
SPOKEN_IDX = {}
for _r in MAP:
    if not _r["spoken"]:
        continue
    cur = SPOKEN_IDX.get(_r["spoken"])
    if cur is None or (_r["crop"] and not cur["crop"]):          # 作物专属条目优先占据索引位
        SPOKEN_IDX[_r["spoken"]] = _r


def normalize(q, crop=None):
    """把口语词替换成该作物下的标准术语；返回 (归一化query, 命中的映射)"""
    out, hits = q, []
    for spoken in sorted(SPOKEN_IDX, key=len, reverse=True):   # 长词优先：防止短词先替换破坏长词匹配
        if spoken not in out:
            continue
        best = SPOKEN_IDX[spoken]
        # 若该口语存在作物专属条目，优先取它
        for r in MAP:
            if r["spoken"] == spoken and r["crop"] == crop:
                best = r
                break
        # 作物专属条目要求作物匹配；通用条目任何作物都适用
        if best["crop"] and crop and best["crop"] != crop:
            continue
        out = out.replace(spoken, best["canonical"])
        hits.append(best)
    return out, hits


# 语料里的标准术语池：用它们去扫 query，而不是靠空格分词
# （中文问法没有空格，"黄瓜起腻虫了打什么药" 整句当一个 token 匹配不到任何东西）
#
# 注意：**不要把作物名放进池子**。黄瓜/辣椒/大蒜 出现在每张卡的检索文本里，
# 放进去会让任何带作物名的问句召回该作物的全部卡片（实测 79/211），等于没检索。
# 作物是**过滤条件**，不是检索键。
# 注意：**要把复合 subtype 拆开**。辣椒表把两种病虫合成一行（「棉铃虫、烟青虫」
# 「早疫病、晚疫病」「猝倒病、立枯病」），不拆的话农户只问其中一种（「棉铃虫」）
# 匹配不到任何 TERM，召回为 0 —— 实测踩过。
TERMS = set()
for _c in CARDS:
    for _t in re.split(r'[、,，]', _c["subtype"]):    # 复合防治对象按顿号/逗号拆成独立术语
        TERMS.add(_t.strip())
    for _ch in _c.get("chemicals", []):
        TERMS.add(_ch["product"])                    # 药剂原名入池
        TERMS.add(re.sub(r'^\d+(?:\.\d+)?%', '', _ch["product"]))  # 去掉含量前缀（如 "20%"）的药剂名也入池
TERMS = {t for t in TERMS if t and len(t) >= 2}      # 过滤空串与单字（单字命中率过高、噪声大）


def recall(q, crop=None):
    """关键词召回：归一化后，用【语料里的标准术语池】扫 query，命中的术语即检索键

    参数：q=问句；crop=作物过滤（None 不限）。
    返回：(归一化query, 命中的映射列表, 召回的卡列表, 命中的术语集合)。
    """
    q2, hits = normalize(q, crop)
    matched = {t for t in TERMS if t in q2}          # 归一化后 query 里包含哪些标准术语
    got = []
    for c in CARDS:
        if crop and c["crop"] != crop:               # 作物是过滤条件，不是检索键
            continue
        txt = searchable(c)
        if any(t in txt for t in matched):           # 卡的检索文本含任一命中术语即召回
            got.append(c)
    return q2, hits, got, matched


# 农户口吻的测试问法（**注意：这些是我编的示例，不是真实采集**）
QUERIES = [
    ("黄瓜起腻虫了打什么药", "黄瓜"),
    ("辣椒白粉怎么治", "辣椒"),
    ("大蒜烂根怎么办", "大蒜"),
    ("黄瓜叶子上有鬼画符", "黄瓜"),
    ("辣椒钻心虫打什么药", "辣椒"),
    ("大蒜根蛆怎么防", "大蒜"),
    ("黄瓜霜霉病用什么药", "黄瓜"),
    ("辣椒蚜虫", "辣椒"),
    ("黄瓜打药后几天能摘", "黄瓜"),
    ("大蒜叶枯病", "大蒜"),
    ("菜青虫", "辣椒"),
    ("黄瓜叶子卷了", "黄瓜"),
    ("黄瓜白粉病打什么药", "黄瓜"),
    ("辣椒早疫病", "辣椒"),
    ("大蒜刺足根螨", "大蒜"),
    ("黄瓜美洲斑潜蝇", "黄瓜"),
    ("大蒜蓟马怎么治", "大蒜"),
    ("辣椒茶黄螨", "辣椒"),
    ("黄瓜灰霉病用什么药", "黄瓜"),
    ("大蒜种蝇", "大蒜"),
    ("黄瓜细菌性角斑病", "黄瓜"),
    ("辣椒猝倒病", "辣椒"),
]

if __name__ == "__main__":
    if not MAP:
        print("映射表为空"); raise SystemExit(1)
    print(f"检索池: {len(CARDS)} 张卡（in_scope=true）\n")
    print(f"{'农户问法':<22}{'映射命中':<22}{'检索键':<24}{'卡数'}")
    print("-" * 96)
    tot = miss = 0                                   # tot=问法总数  miss=召回为 0 的条数
    for q, crop in QUERIES:
        q2, hits, got, matched = recall(q, crop)
        h = "、".join(f"{x['spoken']}→{x['canonical']}" for x in hits) or "（无）"
        tot += 1
        if not got: miss += 1
        print(f"{q:<22}{h[:20]:<22}{'、'.join(sorted(matched))[:22]:<24}{len(got)}")
    print("-" * 96)
    print(f"共 {tot} 条问法，召回为 0 的: {miss} 条")

    print("\n\n各条命中的卡片（抽样看是否合理）:")
    for q, crop in QUERIES[:6]:                      # 只抽样前 6 条问法做命中明细展示
        q2, hits, got, matched = recall(q, crop)
        if not got:
            print(f"\n  ✗ 「{q}」→ 一张都没命中")
            continue
        ev = {h["evidence"] for h in hits}
        flag = "⚠️ 依赖待验证映射" if ev & {"unverified", "seed"} else "✅"   # 命中依赖未经证实映射时打警告
        print(f"\n  {flag} 「{q}」({crop}) → {len(got)} 张")
        for c in got[:4]:
            print(f"      {c['card_id']:<24} {c['title']}")
