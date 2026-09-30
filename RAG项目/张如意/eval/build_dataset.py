# -*- coding: utf-8 -*-
"""生成评估集 -> eval/datasets/auto.jsonl

## 两类用例，生成方式不同

| 类型 | 来源 | 生成方式 | 数量 |
| --- | --- | --- | --- |
| 附录B（高危） | 25 张卡 → 21 个「作物×病虫害」组 | 模板：`{作物}{病虫害}打什么药` | 21 |
| 正文（普通） | 186 张条款卡 | **DeepSeek 生成农户问法**（条款文本没有现成问法） | 186 |

## ⚠️ 这份评估集的局限（必须知道）

它是**从语料反推**的，测的是「能不能召回自己」，不是「能不能回答农户」。
**自问自答，结果偏乐观。** 真实表述差异（口语、错字、语序）测不到。

真实问法用 `source: "collected"` 追加到同一个数据集，指标会分层统计。

## 用法
    python eval/build_dataset.py                # 全量
    python eval/build_dataset.py --no-llm       # 只生成附录B 模板用例，不调模型
"""
import argparse
import json
import os
import sys
from collections import defaultdict

# 预处理后的卡片数据目录（输入）
P = "data/processed"
# 输出：生成的评估集 JSONL 路径
OUT = os.path.join("eval", "datasets", "auto.jsonl")
# DeepSeek 对话接口地址与模型名（用于把条款改写成农户问法）
API = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-chat"

# 正文里这几类不生成问法：不是可执行的农事知识
SKIP_SECTIONS = {"1", "2"}          # 范围、规范性引用文件
SKIP_KEYWORDS = ("填写《", "见附录", "见表")  # 条款前 20 字命中任一关键词即跳过（纯引用/指引性文字）


def load(name):
    """读取 data/processed/ 下指定名称的 JSONL 文件，返回字典列表。"""
    return [json.loads(l) for l in open(os.path.join(P, name), encoding="utf-8")]


def clause_text(card):
    """取卡片的第一条非空措施文本作为条款正文。

    参数：card —— 条款卡字典，可能含 measures 列表
    返回：第一条措施的 text；没有则返回空字符串
    """
    return next((m["text"] for m in card.get("measures", []) if m.get("text")), "")


# ---------------------------------------------------------------- 附录B（模板）
def build_appendix_b():
    """按「作物 × 病虫害」分组：问一种病，同组的所有卡都算命中。

    黄瓜霜霉病在附录B 有两张卡（发病前预防 / 发病初期），两者都是正确答案——
    期望集若只写一张，会把另一张的正确召回误判成错误。
    """
    groups = defaultdict(list)  # (作物, 防治对象) -> 同组卡片 id 列表
    for c in load("appendixB_cards.jsonl"):
        # subtype 可能是「A、B」这类顿号/逗号分隔的多对象，逐个拆开归组
        for sub in c["subtype"].replace("、", ",").split(","):
            groups[(c["crop"], sub.strip())].append(c["card_id"])

    rows = []
    for (crop, sub), ids in sorted(groups.items()):
        rows.append({
            "query": f"{crop}{sub}打什么药",  # 模板问法：作物名+防治对象+打什么药
            "crop": crop,
            "expect_cards": sorted(ids),  # 同组所有卡都算命中（排序保证输出稳定）
            "expect_fallback": False,
            "source": "auto",
            "is_high_risk": True,  # 问用药 → 高危分层
            "note": f"附录B 模板生成；{len(ids)} 张卡同属一个防治对象",
        })

    # ---- 负例：拿 A 作物的病虫害去问 B 作物 ----
    # 这是**确定性**的负例——大蒜语料里确实没有霜霉病，系统就该答「暂无」。
    # 没有负例就无法测兜底（全是正例时，兜底率恒为 0）。
    #
    # ⚠️ 必须排除「同物异名」：只做 subtype 精确匹配会造出假负例——
    #    「黄瓜蚜虫打什么药」被当成负例，但黄瓜的规范名就是「瓜蚜」，
    #    系统返回瓜蚜卡是**正确**的。所以要先过一遍口语映射表，
    #    把「在本作物下有同义实体」的情况剔掉。
    alias = {}  # (作物, 口语名) -> 规范名集合（只收已验证 verified 的映射）
    cpath = os.path.join("configs", "colloquial_map.json")
    if os.path.exists(cpath):
        for m in json.load(open(cpath, encoding="utf-8")):
            if m.get("status") != "verified":  # 未验证的映射不参与负例排除
                continue
            alias.setdefault((m["crop"], m["spoken"]), set()).add(m["canonical"])

    def equivalent(entity, crop):
        """entity 在 crop 下有没有同义实体（含链式一跳）"""
        # 收集所有口语名等于 entity 的映射对应的规范名；crop 为 None 表示跨作物通用映射
        direct = {a for (c, s), cans in alias.items() if c in (crop, None)
                  and s == entity for a in cans}
        for a in direct:
            # 规范名就是它本身，或该作物下存在以规范名命名的防治对象组 → 视为同物异名
            if a == entity or (crop, a) in groups:
                return True
        return False

    crops = sorted({c for c, _ in groups})  # 所有作物名（用于交叉造负例）
    for (crop, sub), _ in sorted(groups.items()):
        for other in crops:
            # other 作物自己也有 sub，或本作物自身 → 跳过，不是负例
            if other == crop or (other, sub) in groups:
                continue
            if equivalent(sub, other):
                continue          # 同物异名，不是真负例
            rows.append({
                "query": f"{other}{sub}打什么药",  # 拿 A 作物的病虫害去问 B 作物
                "crop": other,
                "expect_cards": [],  # 期望空：语料没有，不该召回任何卡
                "expect_fallback": True,  # 应硬兜底
                "source": "auto",
                "is_high_risk": True,
                "note": f"负例：{other} 语料中无「{sub}」，应兜底",
            })
    return rows


# ---------------------------------------------------------------- 正文（LLM）
def gen_question(card, text):
    """让 DeepSeek 把条款改写成农户口吻的问句。失败时退回关键词式问法。"""
    import urllib.request
    # 提示词：要求改写成一句口语化农户问法，且不得出现「国标/标准/条款」字眼
    prompt = (
        "下面是一条农业国家标准条款。请改写成**一个农户会问的问题**。\n"
        "要求：\n"
        "1. 只用一句话，口语化，像农户问农技员那样\n"
        "2. 问题要能指向这条条款的内容\n"
        "3. **不要出现「国标」「标准」「条款」这类词**\n"
        "4. 只输出问题本身，不要任何前缀或解释\n\n"
        f"作物：{card['crop']}\n条款：{text[:300]}"  # 条款只取前 300 字，控制 prompt 长度
    )
    # 组装请求体：temperature=0.7 保证口语多样性，max_tokens=60 足够一句话
    body = json.dumps({"model": MODEL,
                       "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0.7, "max_tokens": 60}).encode()
    # 从环境变量读取 API Key，构造带鉴权头的 HTTP 请求
    req = urllib.request.Request(
        API, data=body,
        headers={"Authorization": f"Bearer {os.environ.get('DEEPSEEK_API_KEY','')}",
                 "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            q = json.load(r)["choices"][0]["message"]["content"].strip()
            # 只保留第一行；去掉首尾的全角空格/句号/问号后统一补「？」
            return q.split("\n")[0].strip("　 。？?") + "？"
    except Exception:
        return None  # 网络/解析失败时返回 None，由调用方退回关键词式问法


def build_body(use_llm=True, limit=0):
    """从 chunks.jsonl 读，**不是** body_cards.jsonl。

    两者差 8 张「光杆章节标题卡」（如「6 有害生物防治」，整张卡就一行标题）——
    chunk 层已按 is_heading_only() 过滤，评估集必须用同一口径，
    否则会拿空标题卡当期望目标，把正确的召回判成未召回（实测 18 条假失败）。
    """
    rows = []
    # 逐行读 chunks.jsonl，排除附录B 表B.1 来源的 chunk（附录B 由模板部分单独生成）
    cards = [json.loads(l)["card"] for l in open("data/chunks/chunks.jsonl", encoding="utf-8")
             if json.loads(l)["meta"]["source_section"] not in ("附录B 表B.1",)]
    # 再按 section 前缀兜底过滤一遍附录B 卡
    cards = [c for c in cards if not c["source"]["section"].startswith("附录B")]
    if limit:
        cards = cards[:limit]  # 调试用：限制正文卡数量
    for i, c in enumerate(cards, 1):
        sec = c["source"]["section"]
        if sec in SKIP_SECTIONS:  # 范围、规范性引用等章节，不生成问法
            continue
        text = clause_text(c)
        # 无正文，或条款开头 20 字内出现引用/指引关键词（非可执行知识）→ 跳过
        if not text or any(k in text[:20] for k in SKIP_KEYWORDS):
            continue
        q = gen_question(c, text) if use_llm else None
        if not q:
            # 退回关键词式：取条款号后前 12 字当检索词
            q = text[len(sec):len(sec) + 12].strip() or text[:12]
        rows.append({
            "query": q,
            "crop": c["crop"],
            "expect_cards": [c["card_id"]],  # 一条条款对应一张期望卡
            "expect_fallback": False,
            "source": "auto-llm" if use_llm else "auto-keyword",  # 区分 LLM 生成与关键词退回
            "is_high_risk": bool(c["is_high_risk"]),  # 沿用卡片自带的高危标志
            "note": f"正文条款 {sec} 反推",
        })
        if i % 20 == 0:  # 进度提示，flush 保证实时输出
            print(f"    ...{i}/{len(cards)}", flush=True)
    return rows


if __name__ == "__main__":
    # 命令行参数解析
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true", help="不调模型，正文用关键词式问法")
    ap.add_argument("--limit", type=int, default=0, help="正文卡数量上限（调试用）")
    a = ap.parse_args()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)  # 确保输出目录存在
    rows = build_appendix_b()  # 第一部分：附录B 模板用例 + 确定性负例
    print(f"附录B 模板用例: {len(rows)} 条")
    body = build_body(use_llm=not a.no_llm, limit=a.limit)  # 第二部分：正文条款用例
    print(f"正文用例: {len(body)} 条")
    rows += body

    # 写出 JSONL：每行一条用例，ensure_ascii=False 保留中文
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    hr = sum(1 for r in rows if r["is_high_risk"])  # 高危用例数统计
    print(f"\n→ {OUT}  共 {len(rows)} 条（高危 {hr} / 普通 {len(rows)-hr}）")
    print("\n⚠️ 这是**从语料反推**的评估集，自问自答、结果偏乐观。")
    print("   真实农户问法请以 source=collected 追加。")
