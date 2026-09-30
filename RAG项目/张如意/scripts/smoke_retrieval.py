# -*- coding: utf-8 -*-
"""检索冒烟测试：验证**识别层**（术语归一 + 意图识别 + 兜底 + 去重）的逻辑。

为什么先做这个：
  映射表和意图表是手写的规则，从没被真实问法压过。向量化之前先跑一遍，
  规则错了改规则是一行的事；等 embedding 建完再回滚，成本高得多。

不做的事：这里**没有向量检索**（还没建索引）。召回用关键词重叠打分，
  只够验证「识别层对不对」，不代表最终检索质量。

跑法：python scripts/smoke_retrieval.py

【模块说明（补充）】
  输入：
    - data/chunks/chunks.jsonl      检索池（逐行 JSON 的 chunk 卡片）
    - configs/colloquial_map.json   口语→标准术语映射表（verified/pending 两种状态）
    - configs/intent_map.json       意图识别规则表（pattern → intent）
  输出：
    - 不写文件，只向 stdout 打印逐用例的归一化/意图/召回/判定报告；
    - 进程退出码：0=全部用例通过，1=有用例不通过，2=检索池未生成。
  ⚠️ 阅读提示：本模块部分全局量（CHUNKS/CMAP/IMAP/VERIFIED/PENDING/
     PENDING_SPOKEN/NGRAM_STOP/MIN_SCORE）并非都在本文件内定义，单独运行
     可能抛 NameError——阅读时以各函数逻辑与注释为准。
"""
import json
import os
import re
import sys
from collections import defaultdict

# 数据与配置路径/加载：P 为处理产物根目录
P = "data/processed"
CHUNKS = [json.loads(l) for l in open("data/chunks/chunks.jsonl", encoding="utf-8")]  # 检索池：全部 chunk
CMAP = json.load(open("configs/colloquial_map.json", encoding="utf-8"))  # 口语→标准术语映射表
IMAP = json.load(open("configs/intent_map.json", encoding="utf-8"))  # 意图识别规则表

# ---------------------------------------------------------------- 识别层
VERIFIED = [r for r in CMAP if r["status"] == "verified"]   # 已验证条目：参与术语替换
PENDING = [r for r in CMAP if r["status"] == "pending"]     # 待定条目：不替换，命中走兜底

# 「烂根」这类 pending 词被直接命中时，走兜底而不是检索
PENDING_SPOKEN = {}
for r in PENDING:
    PENDING_SPOKEN.setdefault(r["spoken"], []).append(r)    # 按口语词分组，便于整句命中后取回条目


def normalize(q, crop=None):
    """术语归一。只有 verified 条目参与；pending 条目**不归一**，改走兜底。

    **必须单遍替换**，不能在替换结果上迭代再匹配。踩过的两个坑：

    1. `蚜虫↔瓜蚜` 是互映射，迭代应用会**来回抵消**：
       「黄瓜蚜虫」→「黄瓜瓜蚜」→「黄瓜蚜虫」，归一化等于没做。
    2. `瓜蚜` 会匹配到「黄**瓜蚜**虫」内部（子串碰撞），替换后作物名被破坏。

    做法：先在**原串**上找出所有匹配位置，同位置取最长者、丢弃重叠的，
    再从后往前替换（不影响前面位置的索引）。

    参数：q=问句；crop=作物过滤（None 不限）。
    返回：(归一化后的问句, 命中的映射条目列表)。
    """
    matches = []
    for r in VERIFIED:
        sp = r["spoken"]
        if not sp:
            continue
        if r["crop"] and crop and r["crop"] != crop:    # 作物专属条目与当前作物不符 -> 跳过
            continue
        # 通用条目：若存在同名的作物专属条目，让专属的生效
        if r["crop"] is None and crop:
            if any(x["spoken"] == sp and x["crop"] == crop for x in VERIFIED):
                continue
        start = 0
        while True:
            i = q.find(sp, start)                       # 在原串上找全部出现位置（不做中途替换）
            if i < 0:
                break
            matches.append((i, len(sp), r))             # 记录 (起始下标, 词长, 条目)
            start = i + 1

    matches.sort(key=lambda m: (m[0], -m[1]))          # 同起点取最长
    chosen, last_end = [], -1
    for i, ln, r in matches:
        if i < last_end:                               # 与已选区间重叠 -> 丢弃
            continue
        chosen.append((i, ln, r))
        last_end = i + ln

    out = q
    for i, ln, r in reversed(chosen):                  # 从后往前替换
        out = out[:i] + r["canonical"] + out[i + ln:]
    hits = [r for _, _, r in chosen]

    # 链式归一：`起腻虫→蚜虫`、`蚜虫→瓜蚜`（黄瓜），农户一句话要走到最终那个词。
    # **每条映射最多用一次**——不然「瓜蚜↔蚜虫」这类（现已删除的反向项）
    # 会 A→B→A 来回抵消。表已改成单向，这里再加一道保险。
    used = {id(r) for r in hits}
    for _ in range(3):                                 # 最多链 3 跳，防意外死循环
        step, progressed = [], False
        for r in VERIFIED:
            if id(r) in used or not r["spoken"]:
                continue
            if r["crop"] and crop and r["crop"] != crop:
                continue
            if r["spoken"] in out:
                out = out.replace(r["spoken"], r["canonical"])
                used.add(id(r))
                step.append(r)
                progressed = True
        hits += step
        if not progressed:
            break
    return out, hits


def check_pending(q):
    """query 是否直接命中了 pending 词（命中则触发兜底，不检索）"""
    for sp, rows in PENDING_SPOKEN.items():
        if sp and sp in q:                             # pending 口语词出现在问句里即命中
            return rows
    return []


def detect_intent(q):
    """意图识别。**只取最长匹配**——否则「最多打几次」会同时命中
    「最多打几次」和它的子串「打几次」，同一条意图重复上报。"""
    best = None
    for it in IMAP:
        if it["pattern"] in q and (best is None or len(it["pattern"]) > len(best["pattern"])):
            best = it                                  # 保留 pattern 最长者（最长匹配优先）
    return [best] if best else []


# 中文没有空格，「肥料员怎么配」整句是一个 token —— 按空格分词一个都匹配不到，
# 早先「肥料员怎么配」召回 0 条就是这么来的。**改用 n-gram 重叠**。
#
# 停用 n-gram：疑问词/虚词在任何 chunk 里都常见，不滤掉会把所有卡片拉进来。
NGRAM_STOP = {"怎么", "什么", "时候", "可以", "这个", "那个", "一下", "如何", "哪里",
              "多少", "几遍", "几次", "有没有", "是不是", "能不能", "怎么办", "请问"}


def ngrams(s, ns=(2, 3, 4, 5)):
    """把字符串切成 n-gram 集合（n 取 2~5）。

    返回：n-gram 字符串集合；停用 n-gram 及首尾含停用词的 n-gram 一律滤除。
    """
    out = set()
    for n in ns:
        for i in range(len(s) - n + 1):
            g = s[i:i + n]                             # 长度为 n 的滑动窗口子串
            if g in NGRAM_STOP or any(g.startswith(w) or g.endswith(w) for w in NGRAM_STOP):
                continue                               # 命中停用表（或以其开头/结尾）则丢弃
            out.add(g)
    return out


# 相关性阈值。**没有它就永远有召回，「暂无」兜底永远不触发。**
# 实测：「辣椒叶子上有白粉病」——辣椒语料里根本没有白粉病，但「辣椒」二字
# 让所有辣椒卡都得了分，返回 5 条无关卡。加了阈值才正确地返回 0。
MIN_SCORE = 0.15


def retrieve(q, crop=None, topk=5, verbose=False):
    """关键词召回（**占位，非向量检索**）。n-gram 重叠打分 + 相关性阈值。

    只用来验证「识别层对不对」，不代表最终检索质量——真检索由 embedding + BM25 承担。

    参数：q=归一化后问句；crop=作物过滤；topk=最多返回条数；verbose=打印调试分。
    返回：得分 ≥ MIN_SCORE 的 chunk 列表（最多 topk 条）。
    """
    qg = ngrams(q)                                     # query 侧 n-gram 集合
    if not qg:
        return []
    scored = []
    for ch in CHUNKS:
        if crop and ch["meta"]["crop"] != crop:        # 作物不匹配 -> 不参与召回
            continue
        txt = ch["embed_text"] + " " + ch["bm25_text"] # 检索文本 = 两字段拼接
        inter = qg & ngrams(txt)
        if inter:
            # 按 query 侧覆盖率打分，避免长 chunk 靠字数占便宜
            scored.append((len(inter) / len(qg), len(inter), ch))
    scored.sort(key=lambda x: (-x[0], -x[1], x[2]["chunk_id"]))  # 覆盖率降序 -> 重叠数降序 -> chunk_id 升序
    if verbose and scored:
        print(f"  最高分 {scored[0][0]:.3f}（阈值 {MIN_SCORE}）｜"
              f"前三分: {[round(s,3) for s,_,_ in scored[:3]]}")
    return [ch for s, _, ch in scored[:topk] if s >= MIN_SCORE]  # 先截 topk，再按阈值过滤


def dedupe_by_clause(chunks):
    """跨作物重复条款去重：同一条款文字在三份标准里逐字相同时只留一条。

    实测 34 组重复、涉及 75 张卡——三份标准是同一套模板衍生的，
    「3.2.7 肥料员」这类条款逐字相同。这是**语料事实，不是重复数据**，
    所以只在检索结果里去重，不动卡片。

    参数 chunks：retrieve 的结果。返回：去重后的列表（保留首次出现顺序）。
    """
    seen, out = {}, []
    for ch in chunks:
        body = next((m["text"] for m in ch["card"].get("measures", []) if m.get("text")), "")
        key = re.sub(r'\s+', '', body)[:60]            # 去掉全部空白后取前 60 字作条款指纹
        if key and key in seen:                        # 指纹重复 -> 丢弃后出现的
            continue
        seen[key] = ch["chunk_id"]
        out.append(ch)
    return out


# ---------------------------------------------------------------- 用例
CASES = [
    # (类别, query, 作物, 期望归一含, 期望意图, 期望召回含, 说明)
    ("1 术语归一", "黄瓜起腻虫了打什么药", "黄瓜", "瓜蚜", "chemical_control",
     "GBZ26581-2011-p21-c10", "起腻虫→蚜虫→瓜蚜（链式），黄瓜应落到瓜蚜卡"),
    ("1 术语归一", "辣椒叶子上有白粉", "辣椒", "白粉", None, None,
     "辣椒语料里没有白粉病——归一后应召回 0，走暂无"),
    ("1 术语归一", "大蒜根蛆怎么防", "大蒜", "种蝇", None, "GBZ26578-2011-p20-c03",
     "根蛆→种蝇"),

    ("2 意图识别", "黄瓜打完药几天能摘", "黄瓜", None, "pre_harvest_interval", None,
     "识别为安全间隔期意图"),
    ("2 意图识别", "大蒜什么时候施肥", "大蒜", None, "fertilizer", None,
     "**期望识别为施肥意图** —— 意图表里有没有这一条？"),
    ("2 意图识别", "辣椒最多打几次药", "辣椒", None, "max_uses_per_season", None,
     "识别为每茬最多使用次数意图"),

    ("3 no_card 兜底", "黄瓜烂根怎么办", "黄瓜", None, None, None,
     "烂根→根腐病是 pending:no_card，应走兜底，**不检索、不编造**"),
    ("3 no_card 兜底", "辣椒叶子发黄", "辣椒", None, None, None,
     "叶子发黄是 pending:no_card，应走兜底"),

    ("4 重复条款去重", "肥料员怎么配", None, None, None, None,
     "三份标准条款逐字相同，应只召回 1 条"),
    ("4 重复条款去重", "生产基地要有工作室吗", None, None, None, None,
     "同上"),
]


def main():
    """逐条跑 CASES 并打印判定报告。

    返回退出码：0=全部通过；1=有用例不通过；2=检索池为空。
    """
    if not CHUNKS:
        print("chunks 未生成"); return 2
    print(f"检索池 {len(CHUNKS)} chunk ｜ verified 映射 {len(VERIFIED)} 条 "
          f"｜ pending {len(PENDING)} 条 ｜ 意图 {len(IMAP)} 条\n")

    fails = 0
    for cat, q, crop, exp_norm, exp_intent, exp_card, note in CASES:
        print("=" * 96)
        print(f"【{cat}】{q}   {note}")

        # --- pending 兜底优先判定 ---
        pend = check_pending(q)                        # 先判是否命中 pending 词（优先级最高）
        q2, nhits = normalize(q, crop)                 # 术语归一
        intents = detect_intent(q2)                    # 意图识别（基于归一化后的问句）

        print(f"  query          : {q}")
        print(f"  归一化后       : {q2}")
        if nhits:
            print(f"  归一化命中     : " +
                  "、".join(f"{h['spoken']}→{h['canonical']}" for h in nhits))
        else:
            print(f"  归一化命中     : （无）")
        print(f"  意图           : " +
              ("、".join(f"{i['intent']}({i['pattern']})" for i in intents) if intents else "（无）"))

        # --- 兜底 ---
        if pend:
            print(f"  ⟶ 命中 pending 词: " +
                  "、".join(f"{p['spoken']}→{p['canonical']}({p['pending_reason']})" for p in pend))
            print(f"  ⟶ 兜底响应     : 「暂无防治方案，请咨询当地农技站」")
            print(f"  召回           : 不检索（正确行为）")
            ok = exp_norm is None                      # 兜底用例要求：不做归一化（exp_norm 为 None）
            print(f"  符合预期       : {'✅' if ok else '❌'}")
            if not ok: fails += 1
            continue

        # --- 检索 + 去重 ---
        got = retrieve(q2, crop, verbose=True)         # 占位关键词检索
        before = len(got)
        got = dedupe_by_clause(got)                    # 跨作物同条款去重
        print(f"  召回           : {before} 条 → 去重后 {len(got)} 条")
        for ch in got[:4]:
            print(f"      {ch['chunk_id']:<26} {ch['meta']['std_no']:<18} {ch['meta']['subtype']}")

        # --- 判定 ---
        problems = []
        if exp_norm and exp_norm not in q2:            # 期望归一化结果包含某标准术语
            problems.append(f"归一化后不含「{exp_norm}」")
        if exp_intent:
            got_i = [i["intent"] for i in intents]
            if exp_intent not in got_i:
                problems.append(f"未识别出意图「{exp_intent}」（实际 {got_i or '无'}）")
        if exp_card and not any(ch["chunk_id"] == exp_card for ch in got):
            problems.append(f"未召回到期望的 {exp_card}")
        # 第 4 类：去重类用例**必须先有召回**才能验去重——上一版漏了空召回，
        # 结果「肥料员怎么配」召回 0 条也判 ✅，掩盖了检索本身坏掉的事实。
        if cat.startswith("4"):
            if not got:
                problems.append("召回 0 条——检索本身坏了，不是去重问题")
            elif len(got) > 1 and len({ch["meta"]["subtype"] for ch in got}) == 1:
                problems.append(f"去重后仍剩 {len(got)} 条同条款")
        if cat.startswith("1") and exp_card is None and len(got) > 0 and "白粉" in q:
            problems.append(f"辣椒语料无白粉病，却召回了 {len(got)} 条")

        if problems:
            fails += 1
            print(f"  符合预期       : ❌  " + "；".join(problems))
        else:
            print(f"  符合预期       : ✅")

    print("\n" + "=" * 96)
    print(f"共 {len(CASES)} 条用例，不通过 {fails} 条")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
