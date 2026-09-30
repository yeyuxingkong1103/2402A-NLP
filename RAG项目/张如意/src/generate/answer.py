# -*- coding: utf-8 -*-
"""答案生成：高危走模板，普通走模型，骨架由模板保证。

## 分工（2026-09-20 定稿）

| 内容 | 谁生成 | 理由 |
| --- | --- | --- |
| **高危字段**（药剂名/剂型/剂量/兑水/施用方法/安全间隔期/每茬最多使用次数） | **模板**，数值逐字来自卡片 `chemicals[]` | 模型一个字符都不许碰——错一位数就是害人 |
| 普通内容（症状、防治原则、农事操作） | DeepSeek | 这些是叙述性内容，改写不影响安全 |
| 整体骨架（结论前置/分步骤/来源/核实提示） | 模板 | 农户友好格式必须稳定，不能靠模型自觉 |

## 核实提示在生成层就带

`needs_verification_hint=true` 的卡，答案里**必定**出现
「以当地最新登记信息与农技站指导为准」。
前端再怎么渲染是前端的事，**生成层是底线**——前端改版不该把提示弄丢。

## 用法
    python src/generate/answer.py --query "黄瓜霜霉病打什么药" --crop 黄瓜
    python src/generate/answer.py --query "..." --crop 黄瓜 --dry-run   # 不调模型，只看模板
"""
# ============================================================================
# 【本文件在流水线里的位置】在检索层（search.py）之后：
#   ① 拿 Top-K 的 chunk_id 查回完整卡片（_CARDS）
#   ② 高危卡 → 模板渲染，每个数值逐字取自 chemicals[]，模型不参与
#   ③ 普通卡 → 交给 DeepSeek 组织成农户听得懂的话
#   ④ 统一补【来源】与「以当地最新登记信息为准」核实提示
# 【为什么不交给模型写药】错一位剂量就是害人，模板渲染才能保证逐字照抄国家标准
# ============================================================================
import argparse
import json
import os
import sys

# 把 src/retrieve 加进 import 路径，后面才能 from search import Searcher, search
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "retrieve"))

# 【三种出口话术】① 完全没依据 / ② 有正文但没有用药方案 / ③ 高危核实提示
FALLBACK_MSG = "暂无防治方案，请咨询当地农技站"
# 放宽后的提示：语料里没有该病虫害的**用药方案**，但正文有相关内容可给。
# 与 FALLBACK_MSG 是两回事——一个「有话说但不含药」，一个「完全没依据」。
NOPLAN_MSG = "本知识库暂无该病虫害的具体用药方案，以上为通用农事建议，请咨询当地农技站"
VERIFY_HINT = "以当地最新登记信息与农技站指导为准"
# 症状问法的结论句：**不做诊断，只做对照**——「与你描述相符」而不是「这是X病」。
# 农户描述可能是疫病也可能是别的，一旦用肯定口吻下诊断，后面的药方就建立在一个
# 未经确认的判断上；对照口吻把最后一道校验交回农户（spec §5.3）。
# 中间插的卡片主语（作物 + 病虫害名）取自**症状条目自己的 `pest`**，不取卡的 subtype——
# 正文卡的 subtype 是章节号（b6.2 → "6.2"），照它念就是「辣椒 6.2」这种垃圾
# （2026-09-28 用户裁决，spec §5.2）。此处保留 SYM_HEAD 前缀常量：
# 「症状段渲染过没有」靠它判（核实提示那句条件句），三处渲染共用 _sym_block。
SYM_HEAD = "【结论】你描述的症状与下面这张卡（"
SYM_HEAD_TAIL = "）记载的相符，请对照确认："
# 症状来源的类型标签（外部来源，与卡片的一级国标来源分开标）
SYM_KIND = {"journal": "期刊文献", "gov_site": "政府/农科院网站",
            "book": "图书", "hotline": "农技热线"}
MODEL = "deepseek-chat"          # 普通内容改写用的对话模型（DeepSeek）
API = "https://api.deepseek.com/chat/completions"   # DeepSeek Chat Completions 接口地址
# 完整建库语料：模板要取结构化 chemicals[] 字段，所以生成层直接读 chunks.jsonl
# ⚠️ 按本文件位置向上定位仓库根，不能写相对路径——否则 pytest 换个 cwd 跑就找不到文件
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHUNKS = os.path.join(_ROOT, "data", "chunks", "chunks.jsonl")

# chunk_id -> 完整卡片（模板渲染高危字段要用结构化 chemicals[]，
# Milvus 里只存了拼平的 bm25_text，取不回字段边界）
# chunk_id -> 症状覆盖层条目（外部来源，答案里要与卡片来源分开标）
_CARDS = {}
_SYMPTOMS = {}
for _line in open(CHUNKS, encoding="utf-8"):
    _c = json.loads(_line)
    _CARDS[_c["chunk_id"]] = _c["card"]
    _SYMPTOMS[_c["chunk_id"]] = _c.get("symptoms") or []


# ---------------------------------------------------------------- 症状覆盖层
# 症状问法判据：**无语料实体「且」无意图**（农户在描述现象、既没说出病名、也没在问某类问题）
def _is_symptom_query(q):
    """⚠️ `q` 必须是**归一化后的问句**——检索层是在归一化问句上做 `coverage_check` 与
    `detect_intent` 的，判据必须同一把尺子、同一个输入（spec §5.1；入口在 `compose` 的 `nq`）。

    **判据本体在检索层**（`search.is_symptom_query`）——这里只做**延迟导入 + 保守兜底**，
    **不再自己实现一遍**（2026-09-29 用户裁决）。原先这里抄了一份「抽实体 + 过语料验证闸」
    的逻辑；检索层那份后来收窄成「**无实体 且 无意图**」（修一条实测回归：问施药规矩的
    技术问句同样没有病名，却被 `chemical_control` 认领，不该按症状问法处理）。
    两处各写一遍就必然漂——**「同一把尺子」的落实办法是同一个函数，不是两句注释**。

    ⚠️ 延迟导入检索层：生成层测试不该被 Milvus/BGE 依赖拖下水。
    导入失败保守返回 False——宁可少渲染一段症状，也不能让生成层挂掉。

    ⚠️ 判据为什么不能只看 `extract_entities` 的原始输出（该口径已随函数一起搬到检索层）：
    已知实体表全落空时它会退回贪婪正则 `[一-鿿]{2,6}(?:病|虫|螨|蝇|螟|蛾|虱|草)`，把散文里的
    「是什么虫」「白色的虫」也抽成噪声实体（`_PEST_STOP` 只收了裸的「什么虫」）——实测
    「叶子上有白色的虫道，是什么虫」抽出 `{'是什么虫', '白色的虫'}`，于是**症状问法被判成
    非症状问法**，整层覆盖层对农户最常用的口吻失效。故必须再过一道语料验证闸。
    """
    if not q:
        return False
    try:
        from search import is_symptom_query
        return bool(is_symptom_query(q))
    except Exception:
        return False


# 【症状对照段】逐字引用外部来源的症状描述，**不过模型**
def render_symptoms(symptoms):
    """症状条目 → 【症状对照】段。数值/文本一字不改，来源逐条标。

    每条冠以自己的病虫害名（`pest`，schema 必填，`scripts/validate_symptoms.py` 有校验）：
    农户问的是「这是什么病」，只给症状文本等于没回答（2026-09-28 用户裁决，spec §5.2）。
    """
    out = ["【症状对照】"]
    for s in symptoms:
        # 防御式取值：两个字段 schema 必填，但答案路径不能因一条脏数据 500（同 build_chunks.py 的 .get 口径）
        text = (s.get("symptom_text") or "").strip()
        pest = (s.get("pest") or "").strip()
        out.append(f"  【{pest}】{text}" if pest else f"  {text}")
        src = s.get("source") or {}
        ref = "　".join(x for x in (src.get("title"), src.get("publisher")) if x)
        kind = SYM_KIND.get(src.get("kind"), "外部来源")
        out.append(f"  —— 来源：{ref}（{kind}，非国标）")
    return "\n".join(out)


# 【症状对照整块】结论句 + 症状段，三条渲染路径共用（改一处不会漏另一处）
def _sym_block(crop, symptoms):
    """【症状对照】整块（对照结论句 + 症状段）。

    `render_high_risk` / `render_high_text` / `compose` 三处**共用这一个构造函数**——
    各自拼一遍就会改一处漏一处（2026-09-28 最终审查 Item 4）。
    结论句主语（作物 + 病虫害名）取自症状条目自己的 `pest`，去重后顿号连接，
    **不用卡的 subtype**（正文卡是章节号，见 SYM_HEAD 处的说明）。
    """
    pests = []
    for s in symptoms:
        p = (s.get("pest") or "").strip()
        if p and p not in pests:
            pests.append(p)
    return f"{SYM_HEAD}{crop} {'、'.join(pests)}{SYM_HEAD_TAIL}\n" + render_symptoms(symptoms)


# 取出 top-1 卡的症状条目（只在症状问法下取）
def _top_symptoms(q, results):
    """返回 top-1 卡的症状列表；非症状问法 / 无症状 → 空列表。

    ⚠️ `q` 必须是**归一化后的问句**（`compose` 的 `nq`）——见 `_is_symptom_query`。
    """
    if not results or not _is_symptom_query(q):
        return []
    return _SYMPTOMS.get(results[0]["chunk_id"]) or []


# ---------------------------------------------------------------- 高危模板
# 【高危模板】同一病虫害的多张卡（不同防治适期）合并成一个结论块
def render_high_risk(cards, crop, symptoms=None):
    """高危卡（同一病虫害的多张，按防治适期分组）→ 固定骨架。

    **每个数值都逐字来自 chemicals[]，无模型参与。**

    为什么要合并：黄瓜的霜霉病在附录B 里有两张卡（发病前预防 / 发病初期），
    逐卡渲染会输出两个并列的【结论】，违背"结论前置"。按病虫害合并后，
    防治适期降级为小节标题。

    ⚠️ **只有带药剂的卡才进这套模板**（它的骨架就是"用药方案"）。
    施药防护、农药采购储藏、器械清洗这类高危正文卡 `chemicals` 是空的，
    走 `render_high_text`——它们压根没有"可用下列药剂"这回事。

    症状问法时用对照口吻并插入【症状对照】段。
    """
    # 一张都没有 chemicals → 是条款型高危卡，不是用药方案卡 → 换渲染器
    if not any(c.get("chemicals") for c in cards):
        return render_high_text(cards, crop, symptoms=symptoms)

    # 病虫害名（同组卡片共用一个 subtype）
    sub = cards[0]["subtype"]
    # 收集各卡的防治适期，用来生成「…均可施药」的结论句
    stages = [c.get("trigger") for c in cards if c.get("trigger") and c["trigger"] != "—"]

    out = []
    if symptoms:
        # 症状问法：不下诊断，先让农户对照确认，再看药（spec §5.2）
        out.append(_sym_block(crop, symptoms))
    elif len(stages) > 1:
        # 多个适期：一句话给出全部适用时机（结论前置）
        out.append(f"【结论】{crop}的{sub}：{'、'.join(stages)}均可施药，方案如下")
    elif stages:
        # 只有一个适期：直接说该时机可用下列药剂
        out.append(f"【结论】{crop}的{sub}，{stages[0]}时可用下列药剂")
    else:
        out.append(f"【结论】{crop}的{sub}，可用下列药剂")

    # 固定小节：农户先看结论，再看用药清单
    out.append("【用药方案】")
    merged = []          # 兼治（同名方案在各卡间一致，去重）
    # 逐卡渲染；多卡时把防治适期降级为小节标题，避免出现多个并列【结论】
    for card in cards:
        trig = card.get("trigger")
        if len(cards) > 1 and trig and trig != "—":
            out.append(f"  ▸ {trig}")
        # 逐条药剂编号渲染（产品名/用量/兑水/施用方法）
        for i, ch in enumerate(card.get("chemicals", []), 1):
            _render_chem(out, ch, i)
            cc = ch.get("companion_control")
            if cc and cc != "—" and cc not in merged:
                merged.append(cc)

    # 兼治：各卡去重后合并成一行
    if merged:
        out.append(f"【兼治】{'；'.join(merged)}")

    # 「轮换使用」是全表最高危的语义，单独提示——它绝不能被读成"混用"
    # 「轮换使用」是全表最高危的语义，单独加一句防误读成「混用」
    if any("轮换" in (ch.get("method") or "")
           for c in cards for ch in c.get("chemicals", [])):
        out.append("  ⚠️ 上述药剂是【轮换使用】，不是混在一起打。")
    return "\n".join(out)


# ------------------------------------------- 无药剂的高危正文卡（2026-09-24 修）
def render_high_text(cards, crop, symptoms=None):
    """无 chemicals 的高危正文卡 → 逐字念 measures[] / precautions，不给药方。

    known_issue（2026-09-23 发现，2026-09-24 修）：这类卡此前也走 chemicals 模板，
    渲染成「【结论】黄瓜的7.1，可用下列药剂」+ 一个空的【用药方案】——三个错一起：
      ① 章节号 7.1 被当成病虫害名念进结论句；
      ② 凭空给了一个不存在的药方，而卡片真正的正文一个字都没出来；
      ③ 核实提示还说"农药名称、剂量、安全间隔期均照抄自原文"，名不副实。
    触发条件：状态 ok / ok_relaxed 且 top-1 是这类正文卡
    （如「打药的人得有啥证」→ 7.1 施药人员防护）。此前该问句一直走 fallback 被掩盖，
    2026-09-23 修完路由劫持后才答得出来、才暴露。

    ⚠️ **仍然不允许交给模型**：这些卡的高危理由可能是人身安全操作 / 轮换使用 / 禁限用，
    交给模型改写就是违反核心约束 1（高危知识逐字照抄）。所以这里自有一套模板渲染，
    全文来自卡片的 measures[] / precautions，一个字都不改。
    """
    first = cards[0]
    # 用卡自带的 knowledge_type（劳动保护 / 基地与投入品管理 / 病虫害防治）当标题，
    # 不新造词；章节号降级为括号里的定位信息，不再当主语念
    topic = first.get("knowledge_type") or "要求"
    out = []
    if symptoms:
        # 症状问法：不念结论句，先让农户对照症状确认（与 render_high_risk 同一口径）
        out.append(_sym_block(crop, symptoms))
    else:
        head = f"【结论】{crop}的{topic}要求如下"
        if len(cards) == 1 and first["source"].get("section"):
            head += f"（第 {first['source']['section']} 条）"
        out.append(head)

    for c in cards:
        # 多卡时按章节号分小节，避免一堆条款糊在一起
        if len(cards) > 1 and c["source"].get("section"):
            out.append(f"  ▸ {c['source']['section']}")
        out.extend(_render_clauses(c))
    return "\n".join(out)


def _render_clauses(card):
    """measures[] / precautions → 编号原文行，去重后返回。

    precautions 常与 measures[0] 完全同文（如 4.1.1 农药采购条款），不去重的话
    农户会看到同一整段国家标准连着念两遍。
    """
    seen, lines = set(), []
    for m in card.get("measures") or []:
        t = (m.get("text") or "").strip()
        if t and t not in seen:
            seen.add(t)
            lines.append(t)
    prec = (card.get("precautions") or "").strip()
    if prec and prec not in seen:
        seen.add(prec)
        lines.append(prec)
    return [f"    {i}. {t}" for i, t in enumerate(lines, 1)]


# 渲染单条药剂：产品名 + 用量 + 兑水 + 施用方法 + 安全间隔期等，全部照抄原文
def _render_chem(out, ch, i):
    """把一条药剂 dict 渲染成两行文本，追加到 out（行列表）里。

    ch：单条药剂的结构化字段（product/dose/dilution/method/pre_harvest_interval…）；
    i ：药剂序号（从 1 起），用于编号；返回值无（副作用是往 out 里 append）。
    """
    line = f"    {i}. {ch['product']}"
    # 用量：原文没有就整段不出现，绝不猜
    if ch.get("dose"):
        line += f"　用量 {ch['dose']}"
    if ch.get("dilution"):
        line += f"　{ch['dilution']}"
    if ch.get("method"):
        line += f"　{ch['method']}"
    out.append(line)

    # 明细行：安全间隔期 / 每茬最多使用次数 / 施药间隔，都是高危字段
    det = []
    if ch.get("pre_harvest_interval"):
        det.append(f"安全间隔期 {ch['pre_harvest_interval']}")
    else:
        # 原文没给安全间隔期 → 显式写明「原文未给出」，不留空白让农户误以为没有要求
        det.append("安全间隔期 原文未给出")
    if ch.get("max_uses_per_season"):
        det.append(f"每茬最多 {ch['max_uses_per_season']}")
    if ch.get("interval_days"):
        det.append(f"施药间隔 {ch['interval_days']}")
    out.append("       ⚠️ " + "　".join(det))


# ---------------------------------------------------------------- 普通内容
# 【普通内容】非高危卡（症状、防治原则、农事操作）交给模型改写
def call_model(query, cards, dry_run=False):
    """普通卡（非高危）交给模型组织成农户能看懂的话。"""
    # dry-run：不调模型，只验证模板骨架
    if dry_run:
        return "（dry-run 跳过模型调用）"
    import urllib.request
    # 把原文条款拼成上下文，作为模型的唯一依据
    src = "\n".join(f"- {c['subtype']}：{c['source']['quote'][:200]}" for c in cards)
    # 提示词硬约束：结论前置/短句分步/不得编造/不得提农药名/来源后面统一加
    prompt = (
        "你是农业技术助手，服务对象是农户。下面是农业国家标准里的原文条款。\n"
        "请用农户能听懂的话回答，要求：\n"
        "1. 结论前置，先说做什么\n"
        "2. 分步骤，每步一句话，短句\n"
        "3. **不要编造原文没有的内容**，不要提具体农药名称或剂量\n"
        "4. 不要写来源和免责声明（后面会统一加）\n\n"
        f"农户问：{query}\n\n国标原文：\n{src}"
    )
    # DeepSeek Chat Completions；temperature 0.3 求稳（叙述性内容，不涉及数值）
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0.3, "max_tokens": 600}).encode()
    # 用标准库 urllib 直接发请求，不引入第三方 SDK 依赖
    req = urllib.request.Request(
        API, data=body,
        headers={"Authorization": f"Bearer {os.environ.get('DEEPSEEK_API_KEY','')}",
                 "Content-Type": "application/json"})
    # 调用失败不抛异常：降级为「仅列原文」，保证答案里始终有真实依据
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            return json.load(r)["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"（模型调用失败：{type(e).__name__}，以下仅列原文）\n" + src


# ---------------------------------------------------------------- 组装
def top1_high_group(results):
    """compose() 实际会渲染的那一组高危卡 ＝ **含 top-1 的那一组**；没有则空列表。

    【2026-09-29 用户裁决·变更 A】`compose()` 原来把 top-5 里**每一组**高危卡
    都渲染成【用药方案】。症状问法下这就是「端出别家病虫害的药方」——
    collected #18「辣椒茎基黑褐斑…整株萎蔫」的 top-1 已是正文卡 b6.2（疫病，无药），
    但 c03（早疫/晚疫）、c05（茶黄螨）仍排在第 2、3 位，农户照样看到别家药方。
    只渲染 top-1 所在那一组后，该条才够得上验收口径「b6.2 正文 + **不带别家药方**」。

    ⚠️ **分组口径只有这一处定义**：`compose()` 渲染、`eval/run_eval.py` 的生成层
    数值校验收（校验「真正被渲染的卡」）**都调它**——「渲染了哪些卡」与
    「校验了哪些卡」各写一遍必然漂（同 `is_symptom_query` 的教训：
    同一个函数才是「同一把尺子」，不是两句注释）。

    top-1 不是高危卡（如 #18 的 b6.2 正文卡）→ 返回空：**一张用药方案都不渲染**。
    宁可少给药方，也不给农户没有依据的药方。
    """
    if not results:
        return []
    top_card = _CARDS.get(results[0]["chunk_id"])
    # top-1 查不到卡片，或它本身不是高危卡 → 没有任何一组"属于它"，一张都不渲染
    if top_card is None or not top_card["needs_verification_hint"]:
        return []
    # 分组键与 compose() 一致：(作物, 病虫害)。top_card 自己必在 high 里，
    # 所以「同键的高危卡」＝「按对象同一性把 top_card 归进去的那一组」。
    # 组内顺序沿用 results 顺序（与改动前的 groups 列表逐位相同）。
    return [c for c in (_CARDS.get(r["chunk_id"]) for r in results)
            if c and c["needs_verification_hint"]
            and c["crop"] == top_card["crop"] and c["subtype"] == top_card["subtype"]]


# 【组装】高危模板 + 普通生成 + 来源 + 核实提示 = 最终答案
def compose(query, crop, results, dry_run=False, noplan=False, nq=None):
    """把高危模板 + 普通生成 + 来源 + 核实提示拼成最终答案。

    `nq`：检索层返回的**归一化问句**（`search()` 的第二个返回值）。症状问法的判据
    必须跑在它上面——检索层就是在归一化问句上做 `coverage_check` 的，两边必须是
    **同一把尺子、同一个输入**；拿原始问句去判，「大蒜小白虫打什么药」（小白虫→蓟马）、
    「大蒜蒜蛆/黑蛆/根蛆打什么药」（→种蝇）都会被当成症状问法，农户明明点了病名
    却拿到对照口吻——「有病名的问法答案零变化」这条安全绳恰好断在口语映射表存在的
    意义上（2026-09-28 最终审查，spec §5.1）。不传时退回 `query`（只给单独调模板的
    测试用；线上三个入口一律传 `q2`）。

    【2026-09-29 用户裁决·变更 A】**高危卡只渲染 top-1 所在的那一组**
    （`top1_high_group`）；top-1 不是高危卡就一张用药方案都不渲染。
    核实提示与分支措辞跟着「**渲染了**哪一组」走，不再按「命中了几组」走。
    """
    # chunk_id 换回完整卡片（Milvus 里只有拼平的文本，取不回字段边界）
    cards = [_CARDS.get(r["chunk_id"]) for r in results]
    cards = [c for c in cards if c]
    # top-1 的 chunk_id：症状段只给含 top-1 的那一组渲染
    top_id = results[0]["chunk_id"] if results else None

    # 普通卡（不需要核实提示的正文条款/农事操作）：交给模型组织成农户话
    normal = [c for c in cards if not c["needs_verification_hint"]]

    # 【2026-09-29 用户裁决·变更 A】高危卡**只渲染 top-1 所在的那一组**，
    # 不再渲染 top-5 里每一组——否则症状问法会把别家病虫害的【用药方案】
    # 一并端给农户（#18：top-1 是正文卡 b6.2，c03/c05 却在第 2、3 位）。
    # 分组口径在 `top1_high_group`（评估器的生成层校验同调它，见该函数说明）。
    # 组内多张卡按防治适期合并渲染：同一病虫害的不同适期合成一个结论块，
    # 否则会输出多个并列的【结论】，违背「结论前置」。
    rendered_high = top1_high_group(results)

    # 症状问法（问句里没有病虫害实体）→ 只给 top-1 卡渲染【症状对照】，
    # 且**全篇只渲染一次**；若 top-1 落不到任何高危组（普通卡/正文卡），
    # 整块提到所有高危块之前，保证它永远排在【用药方案】之前。
    # （2026-09-28 用户裁决，修计划初稿的「无条件叠加」——那会让症状段重复出现，
    #   或落到【用药方案】之后。原写法用 card_id 比 chunk_id 也比错了命名空间。）
    # 判据跑在归一化问句上（nq 缺席才退回原句）——与检索层 coverage_check 同一输入
    top_syms = _top_symptoms(nq or query, results)
    top_card = _CARDS.get(top_id)      # 用对象同一性判组，不依赖 card_id ≡ chunk_id

    # 渲染那一组（症状只挂在它身上：top-1 必在组内，见 top1_high_group）
    blocks = []
    if rendered_high:
        blocks.append(render_high_risk(rendered_high, rendered_high[0]["crop"],
                                       symptoms=top_syms or None))
    # top-1 不在任何高危组（普通卡/正文卡）→ 症状块独立提前，必须在【用药方案】之前
    elif top_syms:
        blocks.insert(0, _sym_block(top_card["crop"], top_syms))
    # 放宽场景：语料无该病虫害用药方案 → 提示放**最前面**（农户先看到"没有药"）
    # 放宽场景：把「暂无用药方案」放到最前面，农户第一眼就看到没有药
    if noplan:
        blocks.insert(0, f"【注意】{NOPLAN_MSG}")
    if normal:
        # 普通卡交给模型生成，作为「相关农事操作」附在高危块后面
        # （症状段已在上面统一挂到唯一一处，这里再挂一次就会重复一整段）
        blocks.append("【相关农事操作】\n" + call_model(query, normal, dry_run))

    body = "\n\n".join(blocks) if blocks else ""

    # 核实提示：**渲染了高危卡**就必定出现（生成层底线，不只靠前端）
    # ⚠️ 变更 A 后判据从「命中高危卡」收窄成「**渲染了**高危卡」：没渲染任何高危内容
    #    却挂着「农药名称、剂量、安全间隔期均照抄自原文」就是不实——与 2026-09-24
    #    「没有药剂就不许说剂量照抄」同一原则（措辞必须跟着内容走）。
    if rendered_high:
        # 措辞必须跟着内容走：全是条款型高危卡时（如 7.1 施药防护）说
        # 「农药名称、剂量、安全间隔期照抄自原文」是误导，本轮一并纠。
        # 两条分支都强制带上 VERIFY_HINT，少一个字都不算过底线。
        # 症状那句同理——**症状段真的渲染了才许说**「症状描述来自外部文献」，
        # 没渲染却挂着这句就是不实（与「没有药剂就不许说剂量照抄」同一原则）。
        sym_note = ("症状描述来自外部文献，是否为你田里的情况请对照确认。"
                    if any(SYM_HEAD in b for b in blocks) else "")
        if any(c.get("chemicals") for c in rendered_high):
            body += f"\n\n⚠️ 农药名称、剂量、安全间隔期均照抄自国家标准原文，" \
                    f"{VERIFY_HINT}。{sym_note}"
        else:
            body += f"\n\n⚠️ 以上要求均照抄自国家标准原文，" \
                    f"{VERIFY_HINT}。{sym_note}"

    # 来源：逐条列出，含 PDF 物理页码，便于农户/农技员回查
    # 来源去重后逐条列出，带 PDF 物理页码，便于农户/农技员回查原文
    seen, srcs = set(), []
    for c in cards:
        s = c["source"]
        k = (s["std_no"], s["section"], s["page"])
        if k in seen:
            continue
        seen.add(k)
        srcs.append(f"  · {s['std_no']}　{s['section']}　PDF 第 {s['page']} 页")
    if srcs:
        body += "\n\n【来源】\n" + "\n".join(srcs)

    return body


# 命令行入口：检索 + 生成一条龙，便于端到端验证
def main():
    """命令行入口：检索 + 生成一条龙。

    参数：--query 问句（必填）、--crop 作物、--topk 取前几条（默认 3）、
    --dry-run 不调模型只看模板骨架。返回退出码 0。
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--crop", default=None)
    ap.add_argument("--topk", type=int, default=3)   # 进入生成阶段的 Top-K 条数
    ap.add_argument("--dry-run", action="store_true")   # 跳过模型调用，只验证模板骨架
    a = ap.parse_args()

    # 复用检索层的 Searcher 和 search()，生成层不自己碰向量库
    from search import Searcher, search
    s = Searcher()
    status, q2, intent, results, why = search(s, a.query, crop=a.crop, topk=a.topk)

    print("=" * 88)
    print(f"问：{a.query}    （作物={a.crop}）")
    print(f"检索状态：{status}　{why}")
    print(f"归一化后：{q2}")
    print("=" * 88)

    # 检索层判定无依据 → 直接兜底，**不交给模型编**
    # 检索层已判定无依据 → 直接输出兜底话术，绝不交给模型去编
    if status.startswith("fallback"):
        print(f"\n{FALLBACK_MSG}")
        print(f"\n（检索层已判定语料中无依据：{why}）")
        return 0

    print()
    # 有结果 → 按状态决定是否带 noplan 提示（ok_noplan 时要提示「暂无用药方案」）；
    # nq 传归一化问句，症状判据与检索层同输入（见 compose）
    print(compose(a.query, a.crop, results, a.dry_run,
                  noplan=(status == "ok_noplan"), nq=q2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
