# -*- coding: utf-8 -*-
"""知识卡 → 切分后的 chunk（双路检索字段）。

输入：data/processed/{appendixB,body}_cards.jsonl 里 in_scope=true 的卡
输出：data/chunks/chunks.jsonl（一行一个 chunk）

## 切分粒度：一卡一 chunk
247 张卡已满足「一条知识 = 一个对象 × 一个措施」，硬拆破坏语义、合并稀释语义。
附录 A 那 36 张 in_scope=false，**不进检索**（供后续过滤用）。

## 双路检索字段（按 2026-09-17 定稿）
| 字段 | 喂给 | 内容 |
| --- | --- | --- |
| `embed_text` | 向量检索 | **标题 + 防治对象 + 防治适期 + 药剂名** |
| `bm25_text`  | BM25 精确匹配 | 全卡文本，**含剂量/安全间隔期等数值** |

设计理由：高危数值（剂量、安全间隔期）**不喂 embedding**——数字会干扰语义相似度，
而"代森锰锌多少倍"这种问法需要的是精确匹配，交给 BM25。
两路各司其职，检索时融合。

## 用法
    python src/index/build_chunks.py
"""
# ============================================================================
# 【本文件在流水线里的位置】解析产物 → 索引，第 1 步（共 3 步）：
#   build_chunks.py（本文件）→ embed_chunks.py → build_milvus_index.py
# 【做什么】从卡片里挑出可检索的 203 张（一卡一 chunk），每张卡生成两个检索字段：
#   embed_text → 喂稠密向量（只管语义）
#   bm25_text  → 喂稀疏向量（含剂量等数值，负责精确匹配）
# ============================================================================
import json
import os
import re
import sys
from collections import Counter

# 输入/输出路径与卡片文件名（只取附录B 与正文两类；附录A 不进检索）
P = "data/processed"
OUT = "data/chunks"
CARDS = ["appendixB_cards.jsonl", "body_cards.jsonl"]
# 症状覆盖层：外部来源的症状条目，按 card_id 挂到卡上（卡本体不动）
SYMPTOMS = os.path.join(P, "symptoms.jsonl")


# 读卡：两个文件逐行解析 JSON，只收 in_scope=true 的
def load_cards():
    """读取附录B 与正文两份卡片文件，返回 in_scope=true 的卡片 dict 列表。

    缺文件时打印提示并退出（退出码 2），提示先跑解析脚本。
    """
    cards = []
    for fn in CARDS:
        p = os.path.join(P, fn)
        # 缺文件直接退出，并提示先跑解析脚本
        if not os.path.exists(p):
            print(f"缺 {p}，先跑 scripts/build_cards_*.py")
            sys.exit(2)
        for line in open(p, encoding="utf-8"):
            c = json.loads(line)
            # in_scope 过滤：附录A 的 36 张记录表卡在这里被挡掉
            if c["in_scope"]:          # 附录A 的 in_scope=false，天然排除
                cards.append(c)
    return cards


# 【症状覆盖层】读 data/processed/symptoms.jsonl → {card_id: [条目, ...]}
def load_symptoms(path=None):
    """读症状覆盖层，按目标卡分组。文件不存在返回空 dict（覆盖层是可选的）。"""
    path = path or SYMPTOMS
    if not os.path.exists(path):
        return {}
    by_card = {}
    for line in open(path, encoding="utf-8"):
        if line.strip():
            s = json.loads(line)
            by_card.setdefault(s["card_id"], []).append(s)
    return by_card


# 症状挂到不存在的卡（或该卡不进索引）→ 返回孤儿 card_id 列表
def check_orphans(symptom_card_ids, chunk_card_ids):
    """挂空的症状会静默丢失，必须在切分层拦下。两个参数都是 card_id 集合。"""
    return sorted(set(symptom_card_ids) - set(chunk_card_ids))


# 【检索字段 1/2】embed_text：喂稠密向量，只放语义线索，不放高危数值
def embed_text(c):
    """向量检索字段。**刻意不含剂量/安全间隔期**——数值交给 BM25。

    两类卡形状不同，不能套同一个模板：

    - **附录B（防治方案卡）**：标题 + 防治对象 + 防治适期 + 药剂名（按定稿口径）
    - **正文条款卡**：没有「防治对象/适期/药剂名」三件套，硬套会拼出
      「黄瓜 黄瓜 1 范围本指导性技术文件规定了黄瓜生产的 1」这种噪音。
      改用 **作物 + 条款号 + 条款正文前 80 字** —— 条款正文本身就是它的知识。
    """
    # 有 chemicals[] 的就是附录B 防治方案卡，否则是正文条款卡
    is_plan_card = bool(c.get("chemicals"))
    # 正文条款卡没有「防治对象+适期+药剂」，改拼「作物 + 条款标题 + 正文前 80 字」
    if not is_plan_card:
        body = ""
        # 取该条款的第一条措施正文，作为这张卡的知识本体
        for m in c.get("measures", []):
            if m.get("text"):
                body = m["text"]
                break
        sec = c["source"]["section"]
        title = c["subtype"] if c["subtype"] != sec else ""
        return " ".join(p for p in [c["crop"], title or sec, body[:80]] if p).strip()

    # 防治方案卡：作物 + 标题 + 防治对象
    parts = [c["crop"], c["title"], c["subtype"] or ""]
    # 防治适期（原文表述）也进 embed_text，帮语义匹配「什么时候打药」这类问法
    if c.get("growth_stage"):
        parts.append(c["growth_stage"])
    if c.get("trigger"):
        parts.append(c["trigger"])
    # 药剂名去浓度前缀，保留有效成分名（浓度是数值，归 BM25）
    # 药剂名去掉浓度前缀（浓度是数值，交给 BM25），只留有效成分名
    seen = []
    for ch in c.get("chemicals", []):
        # 去掉开头的「数字%」，如 25%嘧菌酯 → 嘧菌酯
        name = re.sub(r'^\d+(?:\.\d+)?%', '', ch["product"])
        if name not in seen:
            seen.append(name)
            parts.append(name)
    return " ".join(p for p in parts if p).strip()


# 【检索字段 2/2】bm25_text：喂稀疏向量，必须含剂量/安全间隔期等数值
def bm25_text(c):
    """BM25 字段：全卡文本，**含高危数值**，供精确匹配。"""
    # 基础字段：作物 + 标准号 + 标题 + 防治对象
    parts = [c["crop"], c["std_no"], c["title"], c["subtype"] or ""]
    if c.get("growth_stage"):
        parts.append(c["growth_stage"])
    if c.get("trigger"):
        parts.append(c["trigger"])
    # 逐条药剂把高危字段全部拼进来，数值一个不漏
    for ch in c.get("chemicals", []):
        parts.append(ch["product"])
        for k in ("dose", "dilution", "pre_harvest_interval", "max_uses_per_season",
                  "interval_days", "method", "companion_control"):
            if ch.get(k):
                parts.append(str(ch[k]))
        if ch.get("raw"):
            parts.append(ch["raw"])
    # 防治措施 / 注意事项 / 原文引用也拼进来，保证精确匹配能命中
    for m in c.get("measures", []):
        if m.get("text"):
            parts.append(m["text"])
    if c.get("precautions"):
        parts.append(c["precautions"])
    if c.get("source", {}).get("quote"):
        parts.append(c["source"]["quote"])
    return " ".join(p for p in parts if p).strip()


# 去噪：条款正文去掉条款号后只剩章节标题本身的「光杆标题卡」，不进索引
def is_heading_only(c):
    """条款正文去掉条款号后，是否就只剩章节标题本身（无实质内容）。

    这类卡如「6 有害生物防治」，整张卡的全部内容就是那个标题，教不了任何东西。
    留在索引里会污染召回——农户问「黄瓜有害生物防治」可能召回这张空卡，
    而不是真正有用的 `6.1 防治原则`。

    在 **chunk 层**过滤：卡片层保留全部 247 张（parse 产物该完整），
    索引层才该去噪。实测 8 张 / 3.8%。
    """
    # 取出条款正文（measures 第一条的 text）
    body = next((m["text"] for m in c.get("measures", []) if m.get("text")), "")
    sec = c["source"]["section"]
    # 剥掉开头的条款号，剩下的字符数 ≤12 就认定是光杆标题
    rest = body[len(sec):].strip() if body.startswith(sec) else body.strip()
    return len(rest) <= 12


# 全角标点 -> 半角。原文本来就用半角（CLAUDE.md 注明「原文多为半角 ,」），
# 但我生成的注记/说明用了全角，混在一起。不归一化的话，农户输入「黄瓜，蚜虫」
# 与「黄瓜,蚜虫」会得到不同检索结果。**只在检索字段上做，不动卡片数据。**
# 全角标点 → 半角归一表（只在检索字段上做，不动卡片原始数据）
FW2HW = {"，": ",", "；": ";", "：": ":", "（": "(", "）": ")",
         "、": ",", "？": "?", "！": "!", "。": ".", "｜": "|"}


# 逐字符替换：农户输入全角还是半角标点，检索结果都一致
def halfwidth(s):
    for a, b in FW2HW.items():
        s = s.replace(a, b)
    return s


# 把一张卡组装成一个 chunk：两个检索字段 + 症状覆盖层 + 过滤用 meta + 原始 card
def build_chunk(c, symptoms=None):
    """symptoms：挂到本卡的覆盖层条目列表（可空）。

    ⚠️ 症状只进 `sym_text`（编码时才与两路拼在一起），**不写进 embed_text /
    bm25_text**——检索层会用 bm25_text 现算「语料里提到过」的实体表，
    混入外部来源会把兜底档位改掉（spec §4.1）。
    """
    symptoms = symptoms or []
    sym_text = " ".join((s.get("symptom_text") or "").strip() for s in symptoms).strip()
    return {
        "chunk_id": c["card_id"],
        "card_id": c["card_id"],
        # ---- 两路检索字段（已做全角->半角归一）----
        "embed_text": halfwidth(embed_text(c)),
        "bm25_text": halfwidth(bm25_text(c)),
        # ---- 症状覆盖层（外部来源，转半角后落盘；symptoms 保留原文供答案层引）----
        "sym_text": halfwidth(sym_text),
        "symptoms": symptoms,
        # ---- 过滤/排序/展示用元数据（不参与相似度）----
        "meta": {
            "crop": c["crop"],
            "std_no": c["std_no"],
            "knowledge_type": c["knowledge_type"],
            "type_group": c["type_group"],
            "pest_kind": c.get("pest_kind"),
            "subtype": c["subtype"],
            "growth_stage": c.get("growth_stage"),
            "is_high_risk": c["is_high_risk"],
            "needs_verification_hint": c["needs_verification_hint"],
            "source_section": c["source"]["section"],
            "source_page": c["source"]["page"],
            "channel": c["source"]["channel"],
            "has_symptoms": bool(symptoms),
        },
        # ---- 生成阶段要用的完整卡（自带，省一次 join）----
        "card": c,
    }


# 主流程
if __name__ == "__main__":
    # 确保输出目录存在
    os.makedirs(OUT, exist_ok=True)
    cards = load_cards()
    # 先统计要丢弃的光杆标题卡（只影响索引，卡片产物仍保留 247 张）
    dropped = [c["card_id"] for c in cards if is_heading_only(c)]
    if dropped:
        print(f"过滤光杆章节标题卡 {len(dropped)} 张（卡片层保留，仅不进索引）:")
        for d in dropped:
            print(f"    - {d}")
    # 真正过滤掉这些卡，再逐卡切 chunk
    cards = [c for c in cards if not is_heading_only(c)]
    # 症状覆盖层挂卡；挂空 = 静默丢失，直接拦下
    syms = load_symptoms()
    orphan = check_orphans(syms, {c["card_id"] for c in cards})
    if orphan:
        print(f"✗ 症状挂到了不存在的卡（或该卡不进索引）: {orphan}")
        sys.exit(1)
    chunks = [build_chunk(c, syms.get(c["card_id"])) for c in cards]

    # 自检：embed_text 不能为空；bm25_text 必须比 embed_text 长（否则数值没进去）
    # 自检 1：embed_text 不能为空（空了这条就永远召回不了）
    bad = [ch["chunk_id"] for ch in chunks if not ch["embed_text"]]
    if bad:
        print(f"✗ embed_text 为空的 chunk: {bad[:5]}")
        sys.exit(1)
    # 自检 2：高危卡的 bm25_text 必须比 embed_text 长，否则说明数值没进稀疏路
    no_num = [ch["chunk_id"] for ch in chunks
              if ch["meta"]["is_high_risk"] and len(ch["bm25_text"]) <= len(ch["embed_text"])]
    if no_num:
        print(f"✗ 高危卡的 bm25_text 未比 embed_text 长（数值可能没进去）: {no_num[:5]}")
        sys.exit(1)

    # 全部检查通过 → 落盘 data/chunks/chunks.jsonl（一行一个 chunk）
    dst = os.path.join(OUT, "chunks.jsonl")
    with open(dst, "w", encoding="utf-8") as f:
        for ch in chunks:
            # ensure_ascii=False 保持中文原样；一行一个 JSON 对象（JSONL 格式）
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")

    # 打印统计：数量、按作物、按来源、高危卡数、两个字段的平均长度
    print(f"{dst}: {len(chunks)} 个 chunk")
    print(f"  按作物: {dict(Counter(c['crop'] for c in cards))}")
    print(f"  按来源: {dict(Counter(c['source']['section'].split()[0] for c in cards))}")
    print(f"  高危卡: {sum(1 for c in cards if c['is_high_risk'])}")
    print(f"  带症状的 chunk: {sum(1 for ch in chunks if ch['meta']['has_symptoms'])}")
    print(f"\n  embed_text 平均 {sum(len(embed_text(c)) for c in cards)//len(cards)} 字")
    print(f"  bm25_text  平均 {sum(len(bm25_text(c)) for c in cards)//len(cards)} 字")
