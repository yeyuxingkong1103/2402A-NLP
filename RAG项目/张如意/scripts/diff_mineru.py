# -*- coding: utf-8 -*-
"""MinerU 输出 × 现有转录 交叉比对。

用法：
    python scripts/diff_mineru.py                    # 默认读 data/external/mineru/
    python scripts/diff_mineru.py --mineru-dir 某目录

比对对象（现有两条通道）：
    正文 p3–p7      <- data/processed/body_text/{作物}_body.md   （pymupdf 文本层）
    附录B p20/p21   <- data/processed/vision_transcribe_appendixB.md （视觉转录）
    高危数值        <- data/processed/appendixB_cards.jsonl

输出三段：① 页面覆盖对照 ② 高危数值差异 ③ 正文文本差异
每条差异都给出「哪边可能错」的判据，并汇总成人工抽检重点。

设计前提（不是失败，是结论）：附录A 的字形码页（p11/p12/p14/p18/p19）与
附录B 全部 5 页，MinerU 预期产出为空或乱码——这正是当初走视觉通道的原因。
"""
import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

CROPS = {"26581": "黄瓜", "26583": "辣椒", "26578": "大蒜"}  # 标准号 → 作物名（识别 MinerU 产物用）
P = "data/processed"   # 我方转录产物所在目录

# 高危数值模式：这些是错一个字符就会害人的字段
HR_PATTERNS = [
    ("安全间隔期", re.compile(r'(\d+(?:\.\d+)?)\s*d(?![a-zA-Z一-鿿])')),
    ("稀释倍数",   re.compile(r'(\d[\d\s,]*)\s*倍液')),
    ("每667m²剂量", re.compile(r'(\d+(?:\.\d+)?)\s*(?:g|mL|kg|m L)\s*/\s*667')),
    ("每100kg剂量", re.compile(r'(\d+(?:\.\d+)?)\s*(?:g|mL|kg)\s*/\s*100\s*kg')),
    ("施药次数",   re.compile(r'(\d+)\s*次')),
]


def cjk_ratio(s):
    """中文字符占全文比例。参数 s：文本；空串返回 0.0。"""
    if not s:
        return 0.0
    return len(re.findall(r'[一-鿿]', s)) / len(s)   # 「一-鿿」覆盖 CJK 基本区汉字


def garbled_ratio(s):
    """非中文、非可打印 ASCII、非常见单位符号的字符占比 —— 用来判字形码/乱码。"""
    if not s:
        return 0.0
    ok = re.compile(r'[一-鿿＀-￯0-9A-Za-z\s.,;:%~\-—–_/()\[\]²³°℃×÷+\'"=*&^$#@!?<>|\\{}]')
    return 1 - len(ok.findall(s)) / len(s)


# 「药剂名 ↔ 安全间隔期」配对。这是附录B 真正的要害：数值集合相同不代表配对了，
# 「代森锰锌 20d」和「氧化亚铜 20d」在集合比对下看不出区别，却是两条不同的知识。
PAIR_PAT = [
    # 允许药剂名与冒号之间夹一个括号说明：大蒜的清单形如「肟硫磷(拌种、灌根):63 d」，
    # 不容忍括号的话这条就抓不到，会被误报成「只有我有」。
    re.compile(r'([一-鿿]{2,8})\s*(?:\([^)]{0,20}\)|（[^）]{0,20}）)?\s*[:：]\s*(\d+(?:\.\d+)?)\s*d'),
    re.compile(r'([一-鿿]{2,8})\s+(\d+(?:\.\d+)?)\s*d(?![a-zA-Z])'),      # 代森锰锌 20 d
]
# 药剂名里常见但会误配的通用词，排除掉
PAIR_STOP = {"安全间隔期", "每茬最多", "施药间隔", "防治次数", "使用次数", "间隔期"}


# 剂型后缀：归一化药剂名时剥掉，好让「80%代森锰锌可湿性粉剂」与「代森锰锌」对得上
DOSE_FORM = ("可分散浓缩液", "水分散粒剂", "可湿性粉剂", "水分散剂", "微乳剂", "悬乳剂", "可溶粉剂",
             "可溶液剂", "烟熏剂", "悬浮剂", "乳油", "颗粒剂", "种衣剂", "水剂", "粉剂",
             "WDG", "SPX", "WP", "EC", "SC", "WG", "AS", "FS", "GR", "CS", "SP", "WJ", "WC",
             "SL", "EW", "ME")


def canon_drug(s):
    """药剂名 → 有效成分名。剥浓度前缀与剂型后缀；复配剂再按 '+' 拆开。
    例：'80%代森锰锌可湿性粉剂' -> {'代森锰锌'}；'52.5%霜脲氰+噁唑菌酮WJ' -> {'霜脲氰','噁唑菌酮'}"""
    s = re.sub(r'^\d+(?:\.\d+)?%', '', s.strip())
    for f in sorted(DOSE_FORM, key=len, reverse=True):
        if f.isascii():
            s = re.sub(rf'{f}$', '', s, flags=re.I)
        elif s.endswith(f):
            s = s[: -len(f)]
    s = s.strip()
    parts = {p.strip() for p in s.split("+") if len(p.strip()) >= 2}
    return parts or ({s} if len(s) >= 2 else set())


# 行文词：配对正则会把散文里的「X 天」也当成「药剂:天数」，命中这些词就丢弃
PROSE = ("需要", "每隔", "终霜", "采收", "晒种", "催芽", "以后", "然后", "如果",
         "以下", "应在", "上述", "间隔", "天数", "表示", "注", "表中", "一般")


def drug_interval_pairs(text):
    # 注意：这里**不能**用 norm() —— norm 会抹掉冒号与空格，而配对正则正是靠这两个
    # 分隔符区分「药剂:天数」和普通行文。用轻量归一化，只统一全半角与多余空白。
    t = text.replace("：", ":").replace("　", " ")
    t = re.sub(r'[ \t]+', ' ', t)
    t = re.sub(r'<[^>]+>', ' ', t)          # 表格 HTML 标签剥掉，保留单元格文本
    out = defaultdict(set)
    for pat in PAIR_PAT:
        for name, v in pat.findall(t):
            # 黄瓜的间隔期清单形如「代森锰锌:20d,3次氧化亚铜:21d」——名字前的「3次」的
            # 「次」是中文，会被贪婪的 [一-鿿]{2,8} 吸进来，配出「次氧化亚铜」这种名字。
            # 辣椒/大蒜的清单没有次数，不受影响，所以只有黄瓜会塌——必须剥掉前导「次」。
            name = re.sub(r'^[次、，,。]+', '', name).strip()
            if name in PAIR_STOP or len(name) < 2:
                continue
            if any(w in name for w in PROSE):
                continue
            for drug in canon_drug(name):          # 与卡片侧用同一套归一化，否则对不上
                out[drug].add(v)
    return out


def norm(s):
    """比对用归一化：去空白、统一上标与全半角。只用于比对，不回写数据。

    MinerU 把上标输出成 LaTeX（`667 $m^2$`、`667 $m^{2}$`），我们这边是 `667m²`——
    不归一化的话「我独有/它独有」会凭空多出一堆假差异。
    """
    s = s.replace(" ", "").replace("　", "")
    s = re.sub(r'\$m\s*\^?\s*\{?\s*2\s*\}?\s*\$', 'm²', s)      # $m^2$ / $m^{2}$
    s = re.sub(r'\$m\s*\^?\s*\{?\s*3\s*\}?\s*\$', 'm³', s)
    s = s.replace("m2", "m²").replace("hm2", "hm²").replace("M2", "m²")
    s = re.sub(r'[，,。；;：:、]', '', s)
    return s


def tokens(text):
    """抽高危数值 token -> {类型: Counter(值)}"""
    out = defaultdict(Counter)
    t = norm(text)
    for name, pat in HR_PATTERNS:
        for m in pat.finditer(t):
            out[name][m.group(1).replace(" ", "")] += 1
    return out


def find_mineru(d):
    """在目录里找 MinerU 产物，按标准号分组。

    识别依据是**整个相对路径**（目录名也算）——MinerU 常把产物放进以文件命名的子目录，
    例如 data/external/mineru/黄瓜/xxx_content_list.json。只认文件名会全部漏掉。
    """
    found = defaultdict(lambda: {"md": None, "content_list": None, "middle": None})
    if not os.path.isdir(d):
        return found
    for root, _, files in os.walk(d):
        for fn in files:
            low = fn.lower()
            full = os.path.join(root, fn)
            rel = os.path.relpath(full, d)
            # 标准号 或 作物名 命中其一即可
            key = next((k for k in CROPS if k in rel), None)
            if key is None:
                key = next((k for k, n in CROPS.items() if n in rel), None)
            if key is None:
                continue
            if low.endswith("_content_list.json"):
                found[key]["content_list"] = full
            elif low.endswith("_middle.json"):
                found[key]["middle"] = full
            elif low.endswith(".md"):
                found[key]["md"] = full
    return found


def load_mineru_pages(paths, offset=0):
    """按页返回 MinerU 文本。优先 content_list.json（带 page_idx）。

    offset：MinerU 若用 `-s/-e` 只跑了部分页，page_idx 是从 0 起的**相对**页号，
            必须加上起始页偏移才等于 PDF 物理页。全量跑时 offset=0。
    """
    pages = {}
    cl = paths.get("content_list")
    if cl and os.path.exists(cl):
        try:
            data = json.load(open(cl, encoding="utf-8"))
        except Exception as e:
            return {}, f"content_list.json 解析失败: {e}"
        for blk in data:
            idx = blk.get("page_idx")
            if idx is None:
                continue
            # MinerU 0-based，本项目 page 是 PDF 物理页（1-based）
            pno = idx + 1 + offset
            piece = ""
            for k in ("text", "table_body", "table_caption", "img_caption", "eq_content"):
                v = blk.get(k)
                if isinstance(v, str):
                    piece += v + "\n"
                elif isinstance(v, list):
                    piece += " ".join(str(x) for x in v) + "\n"
            pages[pno] = pages.get(pno, "") + piece
        return pages, None
    md = paths.get("md")
    if md and os.path.exists(md):
        return {}, "只有 .md（无页码信息），无法做页级对齐——请一并提供 _content_list.json"
    return {}, "无可用产物"


def load_mine_body(crop):
    """读取我方正文文本，按页分隔符切开，返回 {PDF页号: 页文本}。"""
    p = os.path.join(P, "body_text", f"{crop}_body.md")   # pymupdf 文本层产物
    if not os.path.exists(p):
        return {}
    txt = open(p, encoding="utf-8").read()
    parts = re.split(r'===== PDF p(\d+) =====', txt)      # 奇数位是页码、偶数位是页文本
    return {int(parts[i]): parts[i + 1] for i in range(1, len(parts), 2)}


def load_my_intervals(crop):
    """从附录B 卡片抽我的安全间隔期（按药剂名），作为数值比对基准。"""
    p = os.path.join(P, "appendixB_cards.jsonl")
    if not os.path.exists(p):
        return {}
    out = {}
    for l in open(p, encoding="utf-8"):
        c = json.loads(l)
        if c["crop"] != crop:
            continue
        for ch in c.get("chemicals", []):
            phi = ch.get("pre_harvest_interval")
            if not phi:
                continue
            days = re.findall(r'(\d+(?:\.\d+)?)\s*d', str(phi))
            for drug in canon_drug(ch["product"]):
                out.setdefault(drug, set()).update(days)
    return out


def main():
    """执行三段比对（页面覆盖/高危数值/药剂配对）并汇总人工抽检重点。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--mineru-dir", default=os.path.join("data", "external", "mineru"))
    a = ap.parse_args()

    print("=" * 76)
    print("MinerU × 现有转录 交叉比对")
    print("=" * 76)
    found = find_mineru(a.mineru_dir)
    if not found:
        print(f"\n在 {a.mineru_dir} 里没找到 MinerU 产物。")
        print("请放置成如下结构（文件名含标准号或作物名即可识别）：")
        print(f"  {a.mineru_dir}/黄瓜/xxx.md")
        print(f"  {a.mineru_dir}/黄瓜/xxx_content_list.json")
        print(f"  {a.mineru_dir}/黄瓜/xxx_middle.json")
        print("\n三种文件都要——页级对齐靠 _content_list.json，"
              "_middle.json 用来核版面块类型（尤其表格是否被识别成表格）。")
        return 2

    agree, diffs, focus = [], [], []   # 一致项 / 差异项 / 人工抽检重点

    for key, name in CROPS.items():
        if key not in found:
            print(f"\n### {name}（{key}）: 未提供 MinerU 产物，跳过")
            continue
        paths = found[key]
        print(f"\n{'='*76}\n### {name}（GB/Z {key}-2011）\n{'='*76}")
        print("  MinerU 文件: " + ", ".join(
            f"{k}={'有' if v else '无'}" for k, v in paths.items()))

        pages, err = load_mineru_pages(paths)
        if err:
            print(f"  ⚠ {err}")
        if not pages:
            continue    # MinerU 无页级文本（如只有 .md）→ 该作物跳过页级比对

        mine = load_mine_body(name)

        # ---------- ① 页面覆盖 ----------
        print("\n【① 页面覆盖对照】")
        print(f"  {'PDF页':<7}{'MinerU':<12}{'中文字数':<10}{'乱码率':<9}{'我这边的通道':<14}判定")
        for pno in sorted(pages):
            t = pages[pno]
            n, g = len(re.findall(r'[一-鿿]', t)), garbled_ratio(t)  # 中文字数与乱码率
            # 我方通道判定：正文 p3–p7 走文本层；附录A 字形码页与附录B 走视觉转录
            if pno <= 7:
                minech = "text"
            elif pno <= 19:
                minech = "vision" if pno in (11, 12, 14, 18, 19) else "text"  # 11/12/14/18/19 为字形码页
            else:                                   # 附录B：全部走视觉通道
                minech = "vision"
            if n == 0:
                verdict = "MinerU 空 → 预期（字形码/旋转页）" if minech == "vision" else "MinerU 空，异常"
                if minech != "vision":
                    focus.append(f"{name} p{pno}: MinerU 空但该页文本层正常 → 必看")
            elif g > 0.5:                           # 乱码率超过 50% 视为乱码页
                verdict = "MinerU 乱码 → 预期（字形码页）" if minech == "vision" else "MinerU 可能乱码"
            else:
                verdict = "有内容，可比对"
                if minech == "vision":
                    verdict += "（我做的是视觉转录，重点比数值）"
            print(f"  p{pno:<6}{('有' if t.strip() else '空'):<12}{n:<10}{g:<9.2f}{minech:<10}{verdict}")

        # ---------- ② 高危数值 ----------
        # 只在**双方都有内容且可比**的正文页（p3–p7）上比数值集合。
        # 附录B 的数值比对走 ②b（按药剂名配对），这里若把整份文档拉进来，
        # 就会变成「MinerU 有附录B、我的 body_text 没有」的苹果比橘子。
        print("\n【② 高危数值差异（仅正文 p3–p7，两侧对等）】")
        mu_body = "\n".join(v for k, v in pages.items() if 3 <= k <= 7)  # MinerU 侧只取正文页
        mt = tokens(mu_body)                 # MinerU 侧高危数值集合
        mytxt = "\n".join(mine.values())
        myt = tokens(mytxt)                  # 我方高危数值集合
        if not mu_body.strip():
            print("    MinerU 在 p3–p7 无内容，跳过")
            mt = myt = defaultdict(Counter)
        for kind in [k for k, _ in HR_PATTERNS]:
            m_set, y_set = set(mt[kind]), set(myt[kind])
            both, only_m, only_y = m_set & y_set, m_set - y_set, y_set - m_set
            if both:
                agree.append(f"{name} {kind}: 一致值 {sorted(both)}")
            if only_m:
                diffs.append((name, kind, "-", sorted(only_m), "MinerU 独有 → 我这边可能漏"))
                focus.append(f"{name} {kind}: MinerU 有而我没有 {sorted(only_m)} → 核我的转录")
            if only_y:
                diffs.append((name, kind, sorted(only_y), "-", "我独有 → MinerU 可能丢表"))
                focus.append(f"{name} {kind}: 我有而 MinerU 没有 {sorted(only_y)} → 看 MinerU 是否丢了整表")
            if not (both or only_m or only_y):
                print(f"  {kind:<12} 两边都没抽到")
            else:
                print(f"  {kind:<12} 一致 {len(both)}  仅MinerU {len(only_m)}  仅我 {len(only_y)}")
                if both:
                    print(f"      一致值: {sorted(both)}")

        # ②b 药剂↔安全间隔期 配对级比对（附录B 的要害）
        print("\n【②b 药剂↔安全间隔期 配对比对】")
        mp = drug_interval_pairs("\n".join(pages.values()))
        yp = drug_interval_pairs("\n".join(mine.values()))
        # 我这边的权威来源是卡片，不是正文文本（附录B 正文里没有这张表）
        card_iv = load_my_intervals(name)
        for drug, v in sorted(card_iv.items()):
            yp.setdefault(drug, set()).update(v)
        common = sorted(set(mp) & set(yp))   # 双方都出现的药剂
        only_m = sorted(set(mp) - set(yp))   # 仅 MinerU 抽到的药剂
        only_y = sorted(set(yp) - set(mp))   # 仅我方有的药剂
        if not (common or only_m or only_y):
            print("    两边都没抽出「药剂:天数」配对（MinerU 可能没识别成表格）")
            if name in ("黄瓜", "辣椒", "大蒜"):
                focus.append(f"{name}: MinerU 未产出可配对的药剂↔天数，附录B 数值列仍需纯人工抽检")
        else:
            print(f"    两边都出现的药剂 {len(common)} 个，逐一核天数：")
            for d in common:
                a, b = sorted(mp[d]), sorted(yp[d])
                mark = "✓ 一致" if a == b else "✗ **天数不同**"
                print(f"      {d:<22} MinerU={a}  我={b}   {mark}")
                if a != b:
                    diffs.append((name, f"间隔期·{d}", b, a, "天数冲突 → 必查原图"))
                    focus.append(f"{name} {d}: MinerU={a} vs 我={b} → 回原图核")
            if only_m:
                print(f"    仅 MinerU 有: {only_m}  → 我可能漏了这些药剂")
                focus.append(f"{name}: MinerU 多出药剂 {only_m} → 核我是否漏抽")
            if only_y:
                print(f"    仅我有: {only_y}  → MinerU 可能丢了这些行")

    # ---------- ③ 汇总 ----------
    print(f"\n\n{'='*76}\n【③ 汇总】\n{'='*76}")
    print(f"\n■ 一致的部分（{len(agree)} 条）")
    for x in agree:
        print(f"    ✓ {x}")
    if not agree:
        print("    （无）")

    print(f"\n■ 差异的部分（{len(diffs)} 条）")
    if diffs:
        print(f"    {'作物':<6}{'字段':<13}{'我的值':<28}{'MinerU':<28}判断")
        for crop, kind, mine_v, mu_v, why in diffs:
            print(f"    {crop:<6}{kind:<13}{str(mine_v):<28}{str(mu_v):<28}{why}")
    else:
        print("    （无）")

    print(f"\n■ 人工抽检重点（{len(focus)} 条）")
    for x in focus:
        print(f"    ▸ {x}")
    if not focus:
        print("    （无 —— 但附录B 数值列仍建议按原清单抽检 ≥20%）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
