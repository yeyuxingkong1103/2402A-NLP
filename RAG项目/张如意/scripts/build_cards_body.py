# -*- coding: utf-8 -*-
"""正文条款 → 知识卡 JSONL（一行一卡，一条款一卡）。

数据源：data/processed/body_text/{作物}_body.md（pymupdf 文本层，已剔除页眉页脚）。
输出：  data/processed/body_cards.jsonl

预处理（configs/prompts/parse_pdf_to_knowledge.md 第 2 节）：
  P1 上标还原：文本层把 m² / hm² 的上标拆到下一行，此处还原并记入 ocr_uncertain
  P3 断行合并：被排版切断的句子合并，不改字词
  P4 乱码页：正文页无字形码页，不触发

抽卡粒度：正文条款按最小可独立执行单元成卡，source.section 填条款号。
高危判定（CLAUDE.md 第六节 高危知识清单）：农药（名称/剂型/剂量/施用方法）、安全间隔期、
防治指标、禁限用农药、农药混用、人身安全操作（施药防护/器械清洗/中毒急救）。
"""
import json
import os
import re

STD = {"黄瓜": ("GB/Z 26581-2011", "黄瓜生产技术规范"),
       "辣椒": ("GB/Z 26583-2011", "辣椒生产技术规范"),
       "大蒜": ("GB/Z 26578-2011", "大蒜生产技术规范")}

# card_id 前缀：去掉斜杠与空格 -> GBZ26581-2011（必须含标准序号，否则三份标准会撞 id）
STD_ID = {c: s.replace("/", "").replace(" ", "") for c, (s, _) in STD.items()}

# 章节 → (knowledge_type, type_group)
def classify(sec):
    """按条款号的一级章节号粗分类。

    参数 sec：条款号字符串（如 "6.1.2"）。返回 (knowledge_type, type_group)：
    3/4→基地与投入品管理，7→劳动保护，8/9→档案记录，5→农事操作（含个别
    品种选择/土肥水管理特判），6→病虫害防治；其余归「其他」。
    """
    p = [int(x) for x in sec.split(".")]    # 按点拆条款号，取各级数字
    top = p[0]                              # 一级章节号
    if top == 3: return "基地与投入品管理", "其他"
    if top == 4: return "基地与投入品管理", "其他"
    if top == 7: return "劳动保护", "其他"
    if top == 8: return "档案记录", "其他"
    if top == 9: return "档案记录", "其他"
    if top == 5:
        if sec.startswith("5.4") or sec.startswith("5.2.1") or sec.startswith("5.2.2") \
           or sec.startswith("5.2.3") and False:
            return "品种选择", "品种选择"
        if sec.startswith("5.1.3") or sec.startswith("5.5.4") or sec.startswith("5.1.2") and False:
            return "土肥水管理", "土肥水管理"
        return "农事操作", "农事操作"
    if top == 6:
        if sec.startswith("6.1"): return "病虫害防治", "病虫草害"
        if sec.startswith("6.2"): return "病虫害防治", "病虫草害"
        return "病虫害防治", "病虫草害"
    return "其他", "其他"


# 品种选择 / 土肥水管理 的分节（各标准编号不同，按条款内容判定）
def refine(sec, text):
    """在 classify 粗分类之上，按条款开头文字二次细化类别。

    参数 text：条款正文。开头命中品种/种子/椒苗字样 → 品种选择；
    开头命中施肥/整地/浇水/灌溉等字样 → 土肥水管理；否则回落到 classify。
    返回 (knowledge_type, type_group)。
    """
    if text.startswith("品种选择") or "品种选择" in text[:12] or "种子质量" in text[:12] \
       or "种蒜质量" in text[:12] or "椒苗标准" in text[:12]:
        return "品种选择", "品种选择"
    # 【2026-09-27 修】窗口 [:8] → [:30]，并补「浇/追施」单字动词：
    #   大蒜 5.4.x 这类卡开头是生育期标题（「苗期管理幼苗长出3片叶后」），
    #   水肥动词出现在第 13~30 字，且原文写法是「浇一次促苗水」「追施复合肥」——
    #   连续词「浇水/追肥」永远匹配不上。曾致 3 张大蒜 5.4.x 卡误标「品种选择」，
    #   施肥/浇水意图路由（knowledge_type 过滤）把它们挡在池外（known_issue 一半）。
    if "基肥" in text[:30] or "追肥" in text[:30] or "追施" in text[:30] \
       or "施肥" in text[:30] or "肥料" in text[:30] \
       or "耕翻土壤" in text[:30] or "整地" in text[:30]:
        return "土肥水管理", "土肥水管理"
    if "浇" in text[:30] or "排涝" in text[:30] or "灌溉" in text[:30]:
        return "土肥水管理", "土肥水管理"
    return classify(sec)


# 高危：按条款内容判定（原文出现即标，不做常识推断）
HR = [
    ("农药剂量",      r"倍液|g/667|mL/667|kg/667|兑水"),
    ("农药名称",      r"%[^。]{0,12}(可湿性粉剂|乳油|悬浮剂|水分散|EC|WP|SC|WG|AS|FS|GR|CS|SPX)"),
    ("安全间隔期",    r"安全间隔期"),
    ("禁限用农药",    r"禁止使用的农药|不应采购下列农药|限用"),
    ("施用方法",      r"施药|喷雾|灌根|烟熏|拌种|浸种"),
    ("人身安全操作",  r"防护服|急救|清洗|中毒|吸烟"),
    ("农药使用准则",  r"GB/T8321|GB4285"),
    ("农药轮换使用",  r"轮换用药|轮换使用化学防治农药|交替使用农药"),
]


def preprocess(raw):
    """P1 上标还原 + P3 断行合并。返回 (正文文本, 不确定标记列表)"""
    t = raw
    notes = []
    # 上标：hm\n2 / m\n2 / m\n2~ / P2O5 等
    for unit, sup in [("hm", "2"), ("m", "2")]:
        pat = f"{unit}\\s*\\n\\s*{sup}(?![0-9])"
        if re.search(pat, t):
            t = re.sub(pat, f"{unit}²", t)
            if f"{unit}² 上标由拆行还原" not in notes:
                notes.append(f"{unit}² 上标由拆行还原")
    # 其余断行：中文之间直接拼接，中英文之间不加空格
    lines = [l.rstrip() for l in t.splitlines() if l.strip()]
    out = ""
    for l in lines:
        if out and not re.match(r'^[0-9]+(\.[0-9]+)*\s', l):
            out += l.lstrip()
        else:
            out += ("\n" if out else "") + l
    return out, notes


CLAUSE = re.compile(r'(?m)^([0-9]+(?:\.[0-9]+)*)\s+(.*?)(?=\n[0-9]+(?:\.[0-9]+)*\s|\Z)', re.S)


# pest_kind 细分词表。只在病虫草害类条款上用；命中不了就留 None
# （「防治原则」「防治措施」这类条款同时覆盖病与虫，硬分类反而错）。
PEST_WORDS = {
    # 「蚜」单字要单列：原文有「瓜蚜」「避蚜」「诱蚜」等写法，只写「蚜虫」会漏
    "虫害": ("蚜", "螨", "蝇", "螟", "蛾", "虱", "蓟马", "害虫", "杀虫灯", "黄板", "蓝板",
             "性信息素", "防虫网", "潜叶", "地老虎", "金龟子", "蝼蛄"),
    "病害": ("霜霉", "疫病", "枯病", "霉病", "白粉病", "炭疽", "病毒病", "根腐", "菌核",
             "黑斑", "紫斑", "黄斑"),
}


def pest_kind(text):
    """按 PEST_WORDS 词表判断条款属于「病害」还是「虫害」。

    参数 text：条款正文。只命中一类 → 返回该类；两类都命中或都不命中
    → 返回 None（不做硬分类）。
    """
    hits = {k for k, ws in PEST_WORDS.items() if any(w in text for w in ws)}
    return hits.pop() if len(hits) == 1 else None   # 病、虫都命中 → 无法断定，留 None


def context_prefixes(body):
    """按各标准自己的编号找出「化学防治」「劳动保护」小节的条款号前缀。
    三份标准的章节号并不一致（黄瓜化学防治=6.6，辣椒=6.3.4，大蒜=6.3.3），
    因此不能硬编码，必须从正文里定位。"""
    chem = labor = None
    for m in re.finditer(r'(?m)^([0-9]+(?:\.[0-9]+)*)\s+(化学防治|劳动保护)\s*$', body):
        (chem, labor) = (m.group(1), labor) if m.group(2) == "化学防治" else (chem, m.group(1))
    return chem, labor


def build():
    """逐作物读取正文文本，切页、预处理、按条款切分并生成知识卡。

    输入：data/processed/body_text/{作物}_body.md（带 ===== PDF pN ===== 页分隔符）。
    返回：list[dict]，一条款一卡，card_id 形如 {标准号}-p{页}-b{条款号}；
    高危判定由 HR 正则命中 + 化学防治/劳动保护/农药管理上下文共同决定。
    """
    cards = []
    for crop, (std_no, std_name) in STD.items():   # 逐作物处理三份标准
        src = os.path.join("data", "processed", "body_text", f"{crop}_body.md")  # 正文文本输入路径
        txt = open(src, encoding="utf-8").read()
        chem_pfx, labor_pfx = context_prefixes(txt)  # 定位化学防治/劳动保护小节的条款号前缀
        pages = re.split(r'===== PDF p(\d+) =====', txt)  # 按页分隔符切开：奇数位是页码、偶数位是页文本
        for pi in range(1, len(pages), 2):
            page_no = int(pages[pi])            # PDF 物理页号（1-based）
            body, ocr_notes = preprocess(pages[pi + 1])  # 上标还原 + 断行合并
            for m in CLAUSE.finditer(body):     # 逐条款匹配
                sec, raw = m.group(1), " ".join(m.group(2).split())  # 条款号；正文压平空白
                if not raw or len(raw) < 6:     # 过滤空条款与过短片段
                    continue
                # 章节标题行（如「3.2.6 植保员」后面紧跟子条款）→ 无独立内容，跳过
                if re.fullmatch(r'[^,。,；:]{1,14}', raw) and "." in sec and len(sec.split(".")) >= 2:
                    continue
                kt, tg = refine(sec, raw)       # 知识类型/分组二次细化
                reasons = [name for name, pat in HR if re.search(pat, raw)]  # 命中的高危原因列表
                # 高危上下文：① 化学防治小节 ② 农药采购/贮藏/处理条款 ③ 劳动保护小节
                in_chem = bool(chem_pfx) and sec.startswith(chem_pfx)
                in_labor = bool(labor_pfx) and sec.startswith(labor_pfx)
                # 仅「农药采购/贮藏/剩余农药/包装物」条款（各标准均为 4.1.x）；
                # 不按「文本里出现农药二字」判定，否则 3.2.2 基地仓库会被误标
                is_pesticide_mgmt = sec.startswith("4.1")
                is_hr = bool(reasons) and (in_chem or in_labor or is_pesticide_mgmt)
                n = sum(1 for c in cards if c["crop"] == crop and c["source"]["page"] == page_no) + 1  # 同作物同页内序号（每页从 1 重新计数）
                cards.append({
                    "card_id": f"{STD_ID[crop]}-p{page_no}-b{sec}",
                    "std_no": std_no, "std_name": std_name, "authority_level": 1, "year": 2011,
                    "region": "全国", "crop": crop, "crop_alias": [],
                    "growth_stage": None,
                    "knowledge_type": kt, "type_group": tg,
                    "pest_kind": pest_kind(raw) if tg == "病虫草害" else None,
                    "subtype": sec, "title": f"{crop} {sec} {raw[:18]}",
                    "symptoms": None, "trigger": None,
                    "measures": [{"type": "其他", "text": raw}],
                    "chemicals": [], "fertilizer": None,
                    "precautions": raw if is_hr and re.search(r"禁止|不应|不得|不应采购", raw) else None,
                    "record_form": None,
                    "is_high_risk": is_hr,
                    "high_risk_reasons": reasons if is_hr else [],
                    "needs_human_review": False,
                    "needs_verification_hint": is_hr,
                    "source": {"std_no": std_no, "section": sec, "page": page_no,
                               "channel": "text", "quote": raw[:200]},
                    "ocr_uncertain": ocr_notes if pi else [],
                    "missing_fields": [],
                    "in_scope": True,
                })
    return cards


if __name__ == "__main__":
    # 生成全部正文条款知识卡并写出 JSONL（一行一卡）
    cards = build()
    dst = os.path.join("data", "processed", "body_cards.jsonl")  # 输出文件路径
    with open(dst, "w", encoding="utf-8") as f:
        for c in cards:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"{dst}: {len(cards)} 张卡")
    for crop in STD:
        sub = [c for c in cards if c["crop"] == crop]
        hr = sum(1 for c in sub if c["is_high_risk"])
        print(f"  {crop}: {len(sub)} 张（高危 {hr}）")
    print("\n高危卡条款号:")
    for c in cards:
        if c["is_high_risk"]:
            print(f"  {c['crop']} {c['source']['section']:<10} {c['high_risk_reasons']}")
