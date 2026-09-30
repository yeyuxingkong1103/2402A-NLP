# -*- coding: utf-8 -*-
"""导出待筛的 unverified 映射 -> data/processed/口语映射_待筛清单.md

给每条补两列实话：
  来源  —— 标准术语是否真的出现在本项目语料里（可验证，不是猜的）
  建议  —— 我的判断 + 理由，供人工筛选时参考

⚠️ 「来源」列只描述**标准术语**的出处。俗称那一侧**全部是推测**，
   本项目没有任何真实农户语料，这是整张表最薄弱的地方。

【模块说明（补充）】
  输入：
    - configs/colloquial_map.json        口语→标准术语映射全表（筛 evidence=="unverified" 的行）
    - data/processed/appendixB_cards.jsonl、body_cards.jsonl、appendixA_cards.jsonl
    - data/processed/body_text/{作物}_body.md（存在才读）
        以上共同构成本项目语料全文，用于核验标准术语是否「有卡可召」
  输出：data/processed/口语映射_待筛清单.md —— 按虫害/病害/杂草/其他分节的
        待人工筛选表（含「来源」「建议」两列与统计）。
"""
import json
import os
import re

P = "data/processed"

# 语料全文（三份国标正文 + 247 张卡），用来核「标准术语在不在语料里」
CORPUS_TEXT = ""
for f in ("appendixB_cards.jsonl", "body_cards.jsonl", "appendixA_cards.jsonl"):
    CORPUS_TEXT += open(os.path.join(P, f), encoding="utf-8").read()   # 三份卡片 JSONL 原文全部拼入语料
for crop in ("黄瓜", "辣椒", "大蒜"):
    p = os.path.join(P, "body_text", f"{crop}_body.md")
    if os.path.exists(p):
        CORPUS_TEXT += open(p, encoding="utf-8").read()                # 正文 Markdown 存在才追加

# 该术语出现在哪些作物里（避免跨作物误用，如「蒜蛆」用在辣椒上）
CROP_OF_TERM = {}
for line in open(os.path.join(P, "appendixB_cards.jsonl"), encoding="utf-8"):
    c = json.loads(line)
    for t in re.split(r'[、,，]', c["subtype"]):        # subtype 可能是复合词（如「棉铃虫、烟青虫」），按顿号/逗号拆开
        CROP_OF_TERM.setdefault(t.strip(), set()).add(c["crop"])
for line in open(os.path.join(P, "body_cards.jsonl"), encoding="utf-8"):
    c = json.loads(line)
    for t in re.split(r'[、,，]', c["subtype"]):
        CROP_OF_TERM.setdefault(t.strip(), set()).add(c["crop"])       # 记录每个术语出现过的作物集合

# 只在正文「主要防治对象」或兼治里出现、没有独立卡片的术语
NO_CARD = {"根腐病", "疫病", "枯萎病", "病毒病", "菌核病", "烟粉虱", "蝼蛄",
           "地老虎", "地下害虫", "黄化", "卷叶病/虫害", "化学防治",
           "安全间隔期", "每茬最多使用次数", "稀释", "剂量", "稀释倍数"}


def in_corpus(term):
    """判断标准术语是否出现在本项目语料全文里。

    参数 term：术语字符串。返回：bool（简单子串判断，无分词）。
    """
    return term in CORPUS_TEXT


def advice(spoken, canon, cat):
    """给一条建议。理由必须落到「有没有卡可召」上，而不是泛泛的'看起来对'。

    参数：
      spoken -- 农户俗称；canon -- 标准术语；cat -- 类别（虫害/病害/杂草/其他）
    返回：(处置建议字符串, 理由字符串)。
      处置优先级：术语不在语料→删；特定口语词→限作物保留；语料有词无卡→仅提示性保留；
      术语只在单一作物→保留并建议加 crop 限定；否则→正常保留。
    """
    if not in_corpus(canon):
        return "**删**", "标准术语不在本项目语料里 → 映射了也无卡可召"
    crops = CROP_OF_TERM.get(canon, set())
    if cat == "虫害" and spoken in ("蒜蛆", "根蛆"):
        return "保留", "限大蒜，建议加 `crop=大蒜`（`蒜蛆` 用在辣椒上不通）"
    if spoken in ("拉拉蛄", "蝼蛄", "土蚕", "地蚕", "土蚕子", "切根虫"):
        return "保留", "仅大蒜 6.3.2.2 提到，建议加 `crop=大蒜`；且无独立卡，只能作提示"
    if canon in NO_CARD:
        return "保留", "语料里有此词但**无独立卡**，只能命中正文条款，不能当主检索键"
    if crops and len(crops) == 1:                      # 该术语只出现在一种作物 -> 建议加 crop 限定
        return "保留", f"仅 {list(crops)[0]} 有，建议加 `crop={list(crops)[0]}`"
    return "保留", "语料有对应卡"


# 只取 evidence == "unverified"（未经证实）的映射行进清单
rows = [r for r in json.load(open("configs/colloquial_map.json", encoding="utf-8"))
        if r["evidence"] == "unverified"]

L = ["# 待筛：`unverified` 农户口语映射\n",
     "> 共 %d 条。**俗称那一侧全部是我的推测**，本项目无真实农户语料，这是最薄弱处。\n" % len(rows),
     "## 怎么用这张表\n",
     "筛完三种处置，告诉我最终结果即可：",
     "- **删**：明显不对的（不同作物混用、地区差异过大）",
     "- **verified**：保留靠谱的，evidence 改成 `verified`",
     "- **pending**：拿不准的，单独讨论\n",
     "> 「建议」列是我的判断，只作参考，最终以你的领域经验为准。\n",
     "> 「来源」列只描述**标准术语**在不在本项目语料里（可复现验证），**不代表俗称可信**。\n",
     "---\n"]

# 按四大类别分节输出
for cat, title in [("虫害", "一、虫害"), ("病害", "二、病害"),
                   ("杂草", "三、杂草"), ("其他", "四、非病虫害问法")]:
    sub = [r for r in rows if r["category"] == cat]    # 该类别下的待筛行
    L.append(f"\n## {title}（{len(sub)} 条）\n")
    L.append("| 序号 | 标准术语 | 农户俗称 | 来源 | 建议 |")
    L.append("| ---: | --- | --- | --- | --- |")
    for i, r in enumerate(sub, 1):
        src = "国标有" if in_corpus(r["canonical"]) else "**国标无**"   # 来源列：标准术语是否在语料
        a, why = advice(r["spoken"], r["canonical"], cat)              # 建议列：脚本判断 + 理由
        L.append(f"| {i} | {r['canonical']} | {r['spoken']} | {src}；俗称推测 | "
                 f"{a}　{why} |")

L.append("\n---\n\n## 统计\n")
n_in = sum(1 for r in rows if in_corpus(r["canonical"]))               # 标准术语在语料里的条数
L.append(f"- 标准术语**在语料里**：{n_in} 条")
L.append(f"- 标准术语**不在语料里**（映射了也无卡可召，建议删）：{len(rows)-n_in} 条")
L.append(f"- 术语在语料里但**无独立卡**（只能命中正文条款）："
         f"{sum(1 for r in rows if r['canonical'] in NO_CARD)} 条")

dst = os.path.join(P, "口语映射_待筛清单.md")
open(dst, "w", encoding="utf-8").write("\n".join(L))
print(f"已写出 {dst}（{len(rows)} 条）")
print(f"  标准术语在语料里: {n_in} / 不在: {len(rows)-n_in}")
