# -*- coding: utf-8 -*-
"""混合检索：口语归一 → 意图识别 → Milvus 稠密+稀疏 RRF 融合 → 兜底判定。

## 流程

    query
      │
      ├─ ① 口语归一（configs/colloquial_map.json，只用 verified 条目；支持链式）
      ├─ ② 意图识别（configs/intent_map.json，只取最长匹配）
      │
      ├─ ③ 兜底判定 (a)：归一化命中 pending:no_card？
      │      命中 → 直接返回「暂无防治方案」，**不检索**
      │
      ├─ ④ Milvus 混合检索：稠密 HNSW/COSINE + 稀疏 SPARSE_INVERTED/IP
      │      RRFRanker 融合，标量字段过滤（作物等）
      │      **稠密单路不设阈值**——实测余量仅 0.06，固定值会误杀
      │
      └─ ⑤ 兜底判定 (b)：融合分低于阈值 / 结果为空
              → 返回「暂无防治方案」

**兜底放在检索层**，不放生成层——这是"检索结果为空"的情况，
不该让生成模型去判断自己有没有依据。

## 用法
    python src/retrieve/search.py --query "黄瓜起腻虫了打什么药" --crop 黄瓜
    python src/retrieve/search.py --query "辣椒叶子上有白粉" --crop 辣椒 --show-score
"""
# ============================================================================
# 【函数地图】按调用顺序，也就是一次查询实际走的步骤：
#   _protected_spans / normalize       → 第①步   口语归一（农户口语 → 标准术语）
#   detect_intent                      → 第②步   意图识别（问法 → 14 个意图之一）
#   check_pending                      → 第③(a)步 确定性兜底（问到 pending 词直接说暂无）
#   extract_entities / coverage_check  → 第③(b)步 实体覆盖检查（这个病虫害有没有卡）
#   is_symptom_query                   → 症状问法判据（**无实体 且 无意图**；
#                                        检索层重排 passage 与生成层症状段共用这一个谓词）
#   search() 内的 narrow               → 第③(c)步 实体收窄（只留点名病虫害的卡）
#   build_route_expr / ROUTE_*         → 第③(d)步 意图路由（该找哪一类知识）
#   Searcher.encode / hybrid           → 第④步   混合检索（稠密+稀疏 RRF 融合）
#   Searcher.rerank_gate               → 兜底闸门（分数低于阈值 → 返回「暂无」）
#   search()                           → 主流程：把上面各步串起来，出口只有一个
# ============================================================================
import argparse
import json
import os
import re
import sys

# 【第 0 步：载入静态配置】三份 JSON 在模块导入时读一次，进程内常驻内存
CMAP = json.load(open("configs/colloquial_map.json", encoding="utf-8"))
IMAP = json.load(open("configs/intent_map.json", encoding="utf-8"))
PARAMS = json.load(open("configs/retrieval_params.json", encoding="utf-8"))
# 按 status 分成两档：verified 参与归一化；pending 只在被直接命中时触发兜底
VERIFIED = [r for r in CMAP if r["status"] == "verified"]
PENDING = [r for r in CMAP if r["status"] == "pending"]
FALLBACK_MSG = "暂无防治方案，请咨询当地农技站"

# 索引文件与模型权重的固定位置（全本地路径，不依赖网络）
DEFAULT_URI = os.path.join("data", "index", "milvus_rag.db")
COLLECTION = "agri_knowledge"
MODEL = r"D:\zg6\bge-m3"
# 交叉编码器重排/闸门。本地已有，2.2G，HF 缓存布局故指向 snapshots/master
RERANKER = r"D:\zg6\bge-reranker-v2-m3\snapshots\master"


# ---------------------------------------------------------------- ① 归一化
# 【第①步-A：归一化前的保护区】先算出「被更长实体覆盖」的字符位置集合
def _protected_spans(q, min_len=4):
    """返回「被更长已知实体覆盖」的字符位置集合。

    用于防止口语替换破坏长实体：「豌豆潜叶蝇」里的「潜叶蝇」不该被换成
    「美洲斑潜蝇」（黄瓜的规范名），否则整个问句的语义就变了。

    **只保护 ≥min_len 字的实体**：2~3 字的跨词边界巧合太多——
    「瓜蚜」恰好出现在「黄【瓜蚜】虫」里，保护它反而挡住应有的链式归一。
    """
    # 结果集：被长实体盖住的字符位置（这些位置上的字不许做口语替换）
    prot = set()
    # 候选长实体 = 有卡的病虫害 ∪ 语料提及过的病虫害；按长度降序，让长实体先占位
    for e in sorted(set(_CARD_CROPS) | _MENTIONED, key=len, reverse=True):
        # 只保护 ≥4 字的实体；2~3 字跨词边界巧合太多，保护它反而挡住应有的替换
        if len(e) < min_len:
            continue
        # 找出该实体在问句里的全部出现位置
        k = q.find(e)
        while k >= 0:
            # 标记「首字之后」的字符；首字留给实体自身，保证标准名本身仍可被替换
            prot.update(range(k + 1, k + len(e)))   # 首字除外，实体自身仍可替换
            k = q.find(e, k + 1)
    return prot


# 【第①步-B：口语归一】把问句里的农户口语换成标准术语，返回 (新问句, 命中的映射条目)
def normalize(q, crop=None):
    """与 scripts/smoke_retrieval.py 同逻辑：单遍定位 + 链式，每条约映射最多用一次。"""
    # **更长实体保护区**：若某段被一个更长的已知实体覆盖，段内的口语词不替换。
    #   否则「黄瓜豌豆潜叶蝇」里的「潜叶蝇」会被换成「美洲斑潜蝇」
    #   （黄瓜的规范名），整个问句变成「黄瓜豌豆美洲斑潜蝇」——长实体在进入
    #   覆盖检查前就被毁了，系统会当成黄瓜的潜叶蝇去答。实测 1 条漏兜底。
    #   只对 **≥4 字** 的实体设保护：2~3 字的会跨词边界巧合命中——
    #   「瓜蚜」恰好出现在「黄【瓜蚜】虫」里，保护它反而挡住了「蚜虫→瓜蚜」
    #   这条应有的链式归一。
    protected = _protected_spans(q)

    matches = []
    # 单遍扫描：只用 verified 条目（pending 条目一律不参与替换）
    for r in VERIFIED:
        sp = r["spoken"]
        if not sp:
            continue
        # 作物过滤①：条目限定了作物且与当前作物不符 → 跳过
        if r["crop"] and crop and r["crop"] != crop:
            continue
        # 作物过滤②：不限作物的条目若存在同词的同作物条目 → 跳过，别让泛化条目抢专用条目
        if r["crop"] is None and crop and any(
                x["spoken"] == sp and x["crop"] == crop for x in VERIFIED):
            continue
        # 在问句里找出该口语词的全部出现位置
        start = 0
        while True:
            i = q.find(sp, start)
            if i < 0:
                break
            # 落在「长实体保护区」内的出现位置跳过，不替换
            if any(p in protected for p in range(i, i + len(sp))):
                start = i + 1
                continue
            matches.append((i, len(sp), r))
            start = i + 1
    # 出现位置排序：同一起点优先取更长的词；再贪心选出互不重叠的一组
    matches.sort(key=lambda m: (m[0], -m[1]))
    chosen, last_end = [], -1
    # 与上一个已选匹配重叠 → 丢弃（保证一次只替换一层）
    for i, ln, r in matches:
        if i < last_end:
            continue
        chosen.append((i, ln, r))
        last_end = i + ln
    out = q
    # 从后往前替换：先改后面的，前面的下标才不会失效
    for i, ln, r in reversed(chosen):
        out = out[:i] + r["canonical"] + out[i + ln:]
    # 本轮真正用上的映射条目，供链式轮排除
    hits = [r for _, _, r in chosen]

    used = {id(r) for r in hits}
    # 链式归一：替换结果可能又暴露出口语词，最多迭代 normalize_chain_max_hops 轮
    for _ in range(PARAMS["normalize_chain_max_hops"]["value"]):
        # 本轮是否有过替换；一次都没有就收敛退出
        progressed = False
        # 链式这一轮也要遵守**更长实体保护区**——只在单遍里加是不够的：
        # 「豌豆潜叶蝇」在单遍被保护住，链式轮却照样把其中的「潜叶蝇」替换掉。
        # 保护区每轮重算——只算一次的话链式轮会绕过它
        prot = _protected_spans(out)
        for r in VERIFIED:
            # 已用过的条目本轮不再用（同一条映射最多替换一次）
            if id(r) in used or not r["spoken"]:
                continue
            if r["crop"] and crop and r["crop"] != crop:
                continue
            sp = r["spoken"]
            # 若该词的所有出现位置都被更长实体覆盖，本轮跳过
            positions = []
            k = out.find(sp)
            while k >= 0:
                positions.append(k)
                k = out.find(sp, k + 1)
            # 该词的所有出现位置都在保护区内 → 本轮跳过
            free = [k for k in positions if not any(p in prot for p in range(k, k + len(sp)))]
            if not free:
                continue
            for k in reversed(free):
                out = out[:k] + r["canonical"] + out[k + len(sp):]
            used.add(id(r))
            hits.append(r)
            progressed = True
        # 一整轮毫无进展 → 提前结束，避免空转
        if not progressed:
            break
    return out, hits


# ---------------------------------------------------------------- ② 意图 → 路由
# 【第②步-A：意图识别的底座】从正文现算「章节号 → 小节标题」表
def _load_section_titles():
    """从正文标题行现算各标准的小节标题 → 章节号。

    **不能硬编码**：三份标准的章节号不一致——
      化学防治：黄瓜 6.6 / 辣椒 6.3.4 / 大蒜 6.3.3
      物理防治：黄瓜 6.5 / 辣椒 6.3.3 / 大蒜 6.3.2
    写死 `6.3/6.4/6.5` 的话，辣椒和大蒜的**化学防治会被当成「防治原则」**，
    农户问药反而去查原则条款。（同附录A 表号错位是同一类坑。）
    """
    out = {}
    # 三份作物各读一次正文文本，互不干扰
    for crop in ("黄瓜", "辣椒", "大蒜"):
        p = os.path.join("data", "processed", "body_text", f"{crop}_body.md")
        if not os.path.exists(p):
            out[crop] = {}
            continue
        titles = {}
        # 逐行匹配纯标题行，如「2.1 品种选择」
        for line in open(p, encoding="utf-8"):
            m = re.fullmatch(r'\s*(\d+(?:\.\d+)*)\s+([^\s,，。;；:：、]{2,12})\s*', line)
            if m:
                titles[m.group(1)] = m.group(2)
        out[crop] = titles
    return out


# 模块导入时执行一次，章节标题表常驻内存（之后每次查询直接查表）
_SECTION_TITLES = _load_section_titles()

# 路由：意图 -> 该找哪一类知识。值是要匹配的**小节标题**，运行时按作物解析成章节号。
# 【第②步-B：路由表】意图 → 该匹配的【小节标题】；运行时按作物解析成章节号
ROUTE_BY_TITLE = {
    "chemical_control":  ["化学防治"],          # 走附录B（见 appendix_only）
    # 问「施药规矩」而非「打什么药」——该查正文的化学防治小节
    # （6.6/6.3.4/6.3.3 的「一般要求」「施药器械」「轮换用药」），不是附录B 用药方案
    "chemical_rules":    ["化学防治"],
    "control_principle": ["防治原则", "农业防治", "生物防治", "物理防治"],
    "physical_control":  ["物理防治"],
    "symptom":           ["主要防治对象"],
}
# 路由：意图 -> 该匹配的 knowledge_type（跨作物通用，无需按作物解析）
# 意图 → knowledge_type（跨作物通用，不用按作物解析，可直接进过滤表达式）
ROUTE_BY_KT = {
    "cultivation":   ["农事操作"],
    "fertilization": ["土肥水管理"],
    "policy":        ["政策补贴"],
}
# 已有意图名 -> 路由名（意图表里的命名与本表略有差异）
# 意图表里的名字 → 本文件的路由名（两处命名不一致，这里做一层翻译）
INTENT_TO_ROUTE = {
    "fertilizer": "fertilization",
    # 【2026-09-27 修】浇水/排涝属于土肥水管理（5.4.x 浇水/施肥条款），不是农事操作——
    #   原映射到 cultivation[农事操作] 会把鳞茎膨大期浇水卡挡在池外
    "irrigation": "fertilization",
    "variety": "cultivation", "planting": "cultivation", "harvest": "cultivation",
}


# 把「路由 → 目标标题」解析成【该作物下真实存在的章节号】列表
def route_sections(route, crop):
    """把路由解析成该作物下的章节号列表"""
    titles = _SECTION_TITLES.get(crop, {})
    want = ROUTE_BY_TITLE.get(route)
    if not want:
        return []
    return [sec for sec, title in titles.items() if title in want]


# 路由 → Milvus 标量过滤表达式；返回 None 表示不加这层过滤
def build_route_expr(route, crop):
    """路由 -> Milvus 过滤表达式。返回 None 表示不加过滤（未识别意图）。"""
    if not route:
        return None
    if route in ROUTE_BY_KT:
        kts = ROUTE_BY_KT[route]
        return "knowledge_type in [" + ", ".join(f'"{k}"' for k in kts) + "]"
    secs = route_sections(route, crop)
    if not secs:
        # 该作物没有这类小节 → 不加过滤，退回全库（宁可多召回，也别漏）
        return None                      # 该作物没有这类小节 -> 不拦，退回全库
    # 连同子条款（6.1 也要覆盖 6.1.1）
    # like "6.6%" 是前缀匹配，连带覆盖子条款 6.6.1、6.6.2
    parts = [f'source_section like "{s}%"' for s in secs]
    return "(" + " or ".join(parts) + ")"


# 意图识别：问句命中多个模式时取【最长】的那个，短模式（如「防治」）不抢长模式
def detect_intent(q):
    best = None
    for it in IMAP:
        if it["pattern"] in q and (best is None or len(it["pattern"]) > len(best["pattern"])):
            best = it
    return best


# ---------------------------------------------------------------- ③ 兜底 (a)
# 【第③(a)步：确定性兜底】pending 词出现在原始问句里 → 直接判「暂无」，与分数无关
def check_pending(q):
    """query 直击 pending 词 -> 确定性兜底（与分数无关）"""
    hits = []
    # 逐条 pending 映射扫原始问句（用原句，不依赖归一化结果）
    for sp, rows in ((r["spoken"], r) for r in PENDING):
        if sp and sp in q:
            hits.append(rows)
    return hits


# ---- 实体级覆盖检查 ----------------------------------------------------------
# 病虫害名的形态：2~6 个汉字 + 病/虫/螨/蝇/螟/蛾/虱/草 结尾
# 病虫害名形态：2~6 个汉字 + 病/虫/螨/蝇/螟/蛾/虱/草 结尾（仅在已知表全落空时才用）
_PEST_RE = re.compile(r'[一-鿿]{2,6}(?:病|虫|螨|蝇|螟|蛾|虱|草)')
# 疑问/泛指短语，形态像病虫害但不是——不滤掉会把「什么病」当成实体
# 停用词：形态像病虫害、实际是疑问或泛指短语，不滤掉会被当成实体
_PEST_STOP = {"什么病", "啥病", "这种病", "什么虫", "啥虫", "这种虫", "啥草",
              "什么草", "什么药", "咋办", "怎么治", "怎么回事",
              # 「病虫害」会被正则切成「病虫」，它不在语料里 → 触发硬兜底，
              # 把「种大蒜的时候地里的病虫害该咋防」这类可答问句误杀（实测 3 条）
              "病虫"}

# 剥前缀用：疑问词 + 动词 + 作物名。正则贪婪时会往前吞这些字
# 剥前缀词表：正则贪婪会把疑问词/动词/作物名一起吞进来，这里逐层剥掉
_STRIP = ("怎么", "如何", "是否", "可以", "防治", "预防", "治疗", "用药", "打药",
          "黄瓜", "辣椒", "大蒜", "蔬菜", "作物", "什么", "咋", "该", "要")


# 生育期前缀：「辣椒播种期猝倒病」「辣椒生长期早疫病」——卡标题自带生育期
# 【2026-09-23 补】卡标题里的生育期（播种期/生长期/苗期/花期）会被正则一起吞进来。
# 只按「取末 4 字」截断会切出「期猝倒病」这种**伪实体**：它不在有卡实体表里，
# 于是 coverage_check 把它判成「仅正文提及」→ no_plan=True → text_only 屏蔽附录B，
# 农户问「辣椒苗期猝倒病怎么防治」反而查不到用药方案。实测 4 个伪实体：
# 期猝倒病 / 期早疫病 / 期棉铃虫 / 期茶黄螨。
_STAGE_RE = re.compile(r'^[一-鿿]{1,3}期')


# 清洗单条候选：剥掉被吞进来的前缀，只保留词尾 ≤4 字的病虫害名
def _clean(e):
    """把正则多吞的前缀剥掉。'黄瓜霜霉病' -> '霜霉病'，'么防治小麦锈病' -> '小麦锈病'"""
    for p in sorted(_STRIP, key=len, reverse=True):
        if e.startswith(p) and len(e) - len(p) >= 2:
            e = e[len(p):]
    # 再剥生育期前缀；剥完不足 2 字就不剥（可能本身就是个正经名字）
    m = _STAGE_RE.match(e)
    if m and len(e) - m.end() >= 2:
        e = e[m.end():]
    return e[-4:] if len(e) > 4 else e      # 病虫害名在词尾，取末 4 字


# entity -> 有独立卡的作物；entity -> **该作物正文里提到过**（含兼治/6.2 清单）
# 两档的区别决定兜底方式，见 coverage_check
_CARD_CROPS = {}
_MENTION_CROPS = {}
_MENTIONED = set()
# 【第③(b)步的底座】从 chunks.jsonl 现算两张「已知实体」表（模块导入时执行一次）
#   _CARD_CROPS     : 实体 → 有独立卡的作物集合（能正常回答）
#   _MENTION_CROPS  : 实体 → 正文提到过的作物集合（只有名字，没有配套用药方案）
for _line in open(os.path.join("data", "chunks", "chunks.jsonl"), encoding="utf-8"):
    _c = json.loads(_line)
    # 该卡所属作物
    _crop = _c["meta"]["crop"]
    # 一张卡可能登记多个防治对象（用 、 或 , 分隔），逐个登记
    for _e in (_c["meta"]["subtype"] or "").replace("、", ",").split(","):
        _e = _e.strip()
        if len(_e) >= 2 and _e not in _PEST_STOP:
            _CARD_CROPS.setdefault(_e, set()).add(_crop)
    # ⚠️ 必须 _clean 后再入池。正则从 bm25_text 抽时会吞前缀，
    #    不清洗的话「黄瓜霜霉病」这种脏实体也会进已知表，
    #    之后 query 里一提「黄瓜霜霉病」就命中它，反而把干净的「霜霉病」盖掉。
    # 再从 bm25_text 正则抽「语料里提及过」的实体，作为第三档判据
    for _e in _PEST_RE.findall(_c.get("bm25_text") or ""):
        _e = _clean(_e)
        if len(_e) >= 2 and _e not in _PEST_STOP:
            _MENTIONED.add(_e)
            _MENTION_CROPS.setdefault(_e, set()).add(_crop)


def extract_entities(q):
    """从 query 提取病虫害实体。

    **优先匹配已知实体**（有卡的 + 语料提及过的），只有全都没命中才用正则兜底。
    纯正则不可靠：`[一-鿿]{2,6}(?:病|...)` 贪婪，会把动词和作物名一起吞进去——
    实测「黄瓜怎么防治小麦锈病」提取出「么防治小麦锈病」，
    「黄瓜病毒病」提取出带作物名的整串。
    """
    # 先查已知实体表（可靠）；全部落空才退回正则（不可靠，会吞前缀）
    known = sorted(set(_CARD_CROPS) | _MENTIONED, key=len, reverse=True)
    hits = {e for e in known if e in q}
    if not hits:
        hits = {_clean(m) for m in _PEST_RE.findall(q)}
        hits = {e for e in hits if len(e) >= 2 and e not in _PEST_STOP}

    # **更长实体优先**：同一次提问里若短实体被长实体包含，只保留长的。
    #   否则「黄瓜豌豆潜叶蝇」会同时抽出「豌豆潜叶蝇」（黄瓜无）和「潜叶蝇」
    #   （黄瓜有，指美洲斑潜蝇），后者把前者"救"过去，系统返回了美洲斑潜蝇的
    #   用药方案——而农户问的是豌豆潜叶蝇。实测 2 条漏兜底，都出自这里。
    kept = []
    # 按长度降序遍历，已被更长实体包含的跳过
    for e in sorted(hits, key=len, reverse=True):
        if any(e != k and e in k for k in kept):
            continue
        kept.append(e)
    return set(kept)


# 【第③(b)步：实体级覆盖检查】问的病虫害在本作物下有没有可用知识（确定性判断，与分数无关）
def coverage_check(q, crop):
    """query 里提到的病虫害实体，在这个作物下**有没有可用的卡**。

    为什么必须有这一层：
      「黄瓜枯萎病用什么药」与「黄瓜炭疽病用什么药」结构完全一样——同作物、
      同意图、同句式，只是前者语料里没有防治方案。任何**文档级相关性打分**
      （稠密余弦、RRF、交叉编码器）都会给它高分。实测 reranker 给
      「黄瓜枯萎病」0.8951，高于「辣椒叶子上有蚜虫」的 0.8699 —— 分布重叠，
      阈值分不开。

      但换个问法就清楚了：这不是"文档多相关"，而是"这个病虫害语料里到底
      有没有方案" —— **实体级判断，确定性，与分数无关**。
      这正是 `pending:no_card` 的机制，此处泛化到全部实体。

    返回 (缺失实体, 命中实体)。没提到任何病虫害时两者都空（不拦）。
    """
    q_entities = extract_entities(q)
    # 没提到任何病虫害 → 不拦，交给后面的分数路径决定
    if not q_entities:
        return [], [], []
    # 先过一道**实体验证**：抽出来的必须真的是个病虫害名。
    # 正则会从散文里吞出「防病治虫」「耐热抗病」这种短语——
    # 实测它们触发硬兜底，把「打药的人要不要考植保员证」这类可答问句误杀了。
    # 判据：该词在语料里**至少出现过一次**（任何作物）。
    #   真病虫害（如刺足根螨，大蒜有）→ 保留，本作物没有就该兜底
    #   散文短语（如防病治虫）→ 丢掉，当作没提取到实体
    # 实体验证闸：抽出来的词必须真的在语料里出现过，否则当作没提取到（滤掉散文短语）
    q_entities = {e for e in q_entities if e in _MENTIONED or e in _CARD_CROPS}
    if not q_entities:
        return [], [], []

    # 第一档 missing：本作物没有配套的卡（不能用用药方案回答）
    missing = sorted(e for e in q_entities if crop and crop not in _CARD_CROPS.get(e, set()))
    # 第二档 present：有卡 → 走正常路径
    present = sorted(e for e in q_entities if e not in missing)
    # 第三档：**本作物正文里提到过吗**（哪怕没有配套用药方案）
    #   有卡           → 正常返回
    #   正文提到过      → 放宽到正文 + 「暂无用药方案」提示（如黄瓜病毒病，6.2.2 列了名）
    #   完全没提过      → 硬兜底（如黄瓜刺足根螨，黄瓜语料里压根没有这词）
    # 第三档 mentioned：本作物正文提到过（如黄瓜病毒病）→ 放宽到正文 + 「暂无用药方案」
    mentioned = [e for e in missing if crop and crop in _MENTION_CROPS.get(e, set())]
    return missing, present, mentioned


# ------------------------------------------- ③(b)附加 症状问法判据（两层共用）
def is_symptom_query(q):
    """症状问法判据——**检索层（重排 passage）与生成层（【症状对照】段）唯一的尺子**。

    成立条件＝**两件都占**：
      ① **语料不认识问句里的病虫害实体**：`extract_entities` 抽出来的词再过一道
         语料验证闸（`_MENTIONED ∪ _CARD_CROPS`），与 `coverage_check` 同一个谓词
         ——已知实体表全落空时 `extract_entities` 会退回贪心正则，把「是什么虫」
         「白色的虫」这类散文噪声当成实体，**不能**只看它的原始输出；
      ② **意图识别落空**：`detect_intent(q) is None`。

    **为什么必须两件都占**（2026-09-29 用户裁决，修一条实测回归）：
      - 农户描述现象（「蒜苗叶子上有一条条白色弯曲的斑道，是什么危害？」）——
        既说不出病名、也不带任何问法模式 → 成立，症状文本才该架起
        「农户的描述 ↔ 这张卡」这座桥；
      - 技术问句「种辣椒打药的时候，是不是得换着用不同的药，免得虫子产生抗药性啊？」
        ——**没有病虫害实体，但被 `chemical_control`（「打药」）认领** → **不成立**。
        它问的是**施药规矩**（答案在正文 6.3.4.4 轮换用药），不是症状描述。
        若按症状问法处理，它的重排 passage 会被塞进方案卡的症状文本，
        把附录B 池顶过闸门（实测 c05 池内最高分 0.0754 → 0.1156 > 0.1）——
        农户问「要不要轮换用药」却拿到 5 张附录B 药方，正是本项目专防的
        **「问技术拿到药方」**。

    一句话：**意图是"这是个问句、不是一段描述"的证据**；症状判据只在
    「无病名 **且** 无问法」时才认账。`q` 必须是**归一化后的问句**（与
    `coverage_check` / `detect_intent` 同一个输入，见 `search()` 主流程）。
    """
    # 空问句不是症状问法（也避免下游对 None 取成员运算）
    if not q:
        return False
    # 条件②：意图认得出来的问句，一律不算症状问法（技术问句可能同样没有病名）
    if detect_intent(q) is not None:
        return False
    # 条件①：语料验证闸后的实体集为空——与 coverage_check 第三个 return 同一个谓词
    return not (extract_entities(q) & (_MENTIONED | set(_CARD_CROPS)))


# ---------------------------------------------------------------- ④ 混合检索
# 【第④步-A：重排 passage】reranker 逐对精算时喂给它的「文档」文本
def _rerank_passage(c, with_sym):
    """重排 passage ＝ 卡本体前 200 字 ＋（仅症状问法时）空格 ＋ 症状文本前 200 字。

    **症状是「条件纳入」**：只有 `is_symptom_query(q)` 成立时才拼 `sym_text`
    （`with_sym` 由调用方**每次调用算一次**、不是每个候选算一次——同一池子里的候选
    同进同出，池内分数才可比）。理由见 `is_symptom_query`：这条桥只在
    「农户在描述现象」时该架；否则一条普通技术问句会被方案卡的症状文本顶过闸门。

    **为什么症状进 passage、不进 `bm25_text`**：
      `bm25_text` 必须保持**卡本体口径**——检索层在模块导入时用正则扫它现算出
      `_MENTIONED` / `_MENTION_CROPS` 两张实体表（`coverage_check` 第三档
      「本作物正文提到过」的唯一判据）。症状是**外部二级来源**，写进去等于把
      「手册里提到过」记成「国标正文提到过」，兜底档位会莫名漂移（见 spec §4.1）。
      而重排器判的是**查询 ↔ 文档相关性**——症状文本正是「农户的描述」与
      「这张卡」之间的桥，它天然属于重排，不属于实体表。

    **追加而非前置**：无症状（或非症状问法）时 `sym` 为空串，返回的就是卡本体原文
    （连那个空格都不加），198 张无症状卡的 passage 逐字节不变——回归面压到最小。

    `rerank_gate` 与 `rerank_sort` **必须都走这个函数**：历史上两处各写了一遍
    `c["bm25_text"][:200]`，各改各的迟早会漂成两套口径（docstring 里写着
    「分数口径完全一致」这条不变量，靠的正是这个唯一的 passage 构造点）。
    """
    base = (c.get("bm25_text") or "")[:200]
    sym = ((c.get("sym_text") or "") if with_sym else "")[:200]
    # 无症状文本 → 原样返回卡本体；有 → 追加（两条路径的 base 部分完全一致）
    return (base + " " + sym).strip() if sym else base


# 【第④步：混合检索】一个 Searcher 把 Milvus、BGE-M3、reranker 一次装好，供反复查询
class Searcher:
    def __init__(self, uri=DEFAULT_URI, collection=COLLECTION, model_path=MODEL):
        from pymilvus import MilvusClient
        from FlagEmbedding import BGEM3FlagModel
        # 打开 Milvus（Lite 版就是打开本地 .db 文件，无服务、无 Docker）
        self.client = MilvusClient(uri)
        # 表不存在 → 早退并提示先跑索引脚本，避免后面报难懂的错
        if not self.client.has_collection(collection):
            raise SystemExit(f"collection「{collection}」不存在，先跑 "
                             f"src/index/build_milvus_index.py")
        # **必须 load**：Milvus 的 collection 在连接关闭后回到 released 状态，
        # 新连接直接 search 会报 `Collection ... is in state 'released'`。
        # 索引脚本里 load 过不算数——那是在它自己的连接里。
        # 必须显式 load：新连接的 collection 处于 released 状态，直接 search 会报错
        self.client.load_collection(collection)
        self.collection = collection
        # 设备自动选择。硬编码 cpu 的话，装了 CUDA torch 也用不上——
        # reranker 在 CPU 上 ~1s/对，每次查询要 5 秒，农户等不了。
        import torch
        # 设备自动选择：有 CUDA 就走 GPU（reranker 在 CPU 上 ~1s/对，农户等不起）
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        use_fp16 = self.device == "cuda"
        self.model = BGEM3FlagModel(model_path, use_fp16=use_fp16, devices=self.device)
        # 交叉编码器 reranker：唯一能分开「可答/不可答」的打分器，用来当兜底闸门
        from FlagEmbedding import FlagReranker
        self.reranker = FlagReranker(RERANKER, use_fp16=use_fp16)
        self._enc_q, self._enc_v = None, None      # encode 单条缓存，见 encode()

    # 问句编码：一次前向同时得到稠密向量与稀疏权重
    def encode(self, q):
        """query 同时出稠密与稀疏。稀疏喂原始 query（与入库时 bm25_text 同为含数值文本）。

        **单条缓存**：一次 search() 会调 encode 两次（dense_gate 一次、hybrid 一次），
        BGE-M3 在 CPU 上约 1s/次——不缓存等于白算一遍。评估 241 条时这是主要开销之一。
        只缓存最近一条：检索是逐条串行的，多槽缓存没有收益还占内存。
        """
        # 单条缓存：一次 search() 会调两次 encode，不缓存就等于白算一遍
        if self._enc_q == q and self._enc_v is not None:
            return self._enc_v
        # BGE-M3 同时输出 dense_vecs(1024 维) 与 lexical_weights(稀疏)
        out = self.model.encode([q], batch_size=1, max_length=256,
                                return_dense=True, return_sparse=True,
                                return_colbert_vecs=False)
        # 稠密 → list[float]，可直接喂 Milvus 的 FLOAT_VECTOR
        dense = out["dense_vecs"][0].tolist()
        lw = out["lexical_weights"][0]
        # 稀疏 → {token_id: 权重}，Milvus SPARSE_FLOAT_VECTOR 要的格式
        sparse = {int(k): float(v) for k, v in lw.items()}
        self._enc_q, self._enc_v = q, (dense, sparse)
        return self._enc_v

    # 构造 Milvus 标量过滤表达式（作物 + 附录B/正文二选一）
    @staticmethod
    def build_expr(crop=None, appendix_only=False, text_only=False):
        """过滤表达式。

        `appendix_only` 用于 `chemical_control` 意图：农户问「打什么药」，
        正确答案只可能在附录B 的防治方案表里；正文 6.6 只有原则性描述
        （「应符合 GB/T 4285 的要求」），答不了具体用什么药。

        `text_only` 用于**放宽兜底**：语料里没有该病虫害的用药方案时，只回正文
        （6.2 防治对象、6.3 农业防治等通用内容）。**绝不能放宽到全部卡片**——
        实测「大蒜霜霉病用什么药」放宽后捞回了**灰霉病**的附录B 用药方案，
        农户照着打就是打错药，比直接说「暂无」危险得多。

        ⚠️ **只用 source_section 过滤，不加 knowledge_type 条件。**
           附录B 25 张卡里有 1 张是 `knowledge_type=杂草防治`（大蒜的除草剂方案
           33%二甲戊乐灵EC）——它同样是「打什么药」的答案。若按
           `knowledge_type == '病虫害防治'` 过滤会把它误杀。
        """
        parts = []
        # 作物过滤，走 crop 字段的 INVERTED 倒排索引
        if crop:
            parts.append(f'crop == "{crop}"')
        # appendix_only：只查附录B 用药方案表（农户问「打什么药」时用）
        if appendix_only:
            parts.append('source_section like "附录B%"')
        elif text_only:
            # Milvus 不支持 `not like`，用 `not (...)` 包裹（实测可用）
            # text_only：排除附录B，只回正文（放宽兜底时用）
            parts.append('not (source_section like "附录B%")')
        return " and ".join(parts) or None

    # 稠密单路打分（现仅作粗筛；真正的兜底闸门是下面的 rerank_gate）
    def dense_gate(self, q, crop=None, ef=64, appendix_only=False, text_only=False):
        """稠密单路取最高余弦分。**已降级为粗筛，不再是兜底闸门。**

        为什么不能当闸门：实测「大蒜杂草用什么药」（**可答**）0.6168 低于
        「辣椒叶子上有白粉」（不可答）0.6387 —— **两个分布重叠，任何阈值都分不开**。
        见 `rerank_gate`。
        """
        dense, _ = self.encode(q)
        res = self.client.search(self.collection, data=[dense], anns_field="dense_vector",
                                 search_params={"metric_type": "COSINE", "params": {"ef": ef}},
                                 limit=1, output_fields=["chunk_id"],
                                 filter=self.build_expr(crop, appendix_only, text_only))
        if not res or not res[0]:
            return 0.0, None
        h = res[0][0]
        return float(h.distance), h.id

    # 兜底闸门：用交叉编码器给候选逐对精算，最高分决定「有依据 / 无依据」
    def rerank_gate(self, q, candidates):
        """用 BGE-reranker-v2-m3 给候选打分，取最高分当**兜底闸门**。

        为什么必须上交叉编码器：
          - RRF 融合分是 `Σ 1/(60+rank)`，近似常数（余量 −0.0015），只能排序
          - 稠密余弦可答/不可答两个分布**重叠**（0.6168 vs 0.6387），分不开
          - 交叉编码器逐对精算，实测余量由负转正（−0.0015 → +0.0259），
            且把被误杀的「大蒜杂草用什么药」从 0.6168 提到 0.9545

        ⚠️ 余量 +0.0259 仍然不宽，且样本是自编问法，必须用真实农户问法重校准。

        passage 与 `rerank_sort` 同口径：一律由 `_rerank_passage` 构造
        （卡本体前 200 字，**症状问法时**再追加症状前 200 字），
        两处的分数因此可直接互比。
        """
        # 没有候选 → 0 分，上层据此兜底
        if not candidates:
            return 0.0, None
        # 问句与卡片文本拼成 (query, passage) 逐对喂给 reranker 精算相关度
        # passage 一律由 _rerank_passage 构造，与 rerank_sort 同口径；
        # 症状是否拼进去**每次调用算一次**（判据见 is_symptom_query）
        with_sym = is_symptom_query(q)
        pairs = [[q, _rerank_passage(c, with_sym)] for c in candidates]
        scores = self.reranker.compute_score(pairs, normalize=True)
        if not isinstance(scores, list):
            scores = [scores]
        # 取最高分与它对应的 chunk_id 返回
        best = max(scores)
        return float(best), candidates[scores.index(best)]["chunk_id"]

    # 重排：同 rerank_gate 的逐对精算，但把**全部候选按分数降序**返回。
    # 【2026-09-27 增】常规路径原来只拿最高分当闸门、排序仍用 RRF——而 RRF 融合分
    #   近似常数（见上），正文条款之间的排序基本是噪声（known_issue：4 条正文条款
    #   排序不准）。交叉编码器既然已经逐对算过分，排序就该听它的。
    def rerank_sort(self, q, candidates):
        """按 reranker 分数降序返回候选，附带最高分（供闸门判定）。

        返回 (最高分, 排序后的候选列表)。没有候选 → (0.0, [])。
        与 rerank_gate 用**同一套 (query, passage) 对**——passage 两处都由
        `_rerank_passage` 构造（卡本体前 200 字，症状问法时再追加症状前 200 字），
        症状开关两处都取自同一个 `is_symptom_query(q)`，
        分数口径完全一致，且结构上不可能再漂成两套。
        """
        if not candidates:
            return 0.0, []
        # 症状是否拼进 passage：**每次调用算一次**，与 rerank_gate 同款
        with_sym = is_symptom_query(q)
        pairs = [[q, _rerank_passage(c, with_sym)] for c in candidates]
        scores = self.reranker.compute_score(pairs, normalize=True)
        if not isinstance(scores, list):
            scores = [scores]
        ranked = [c for _, c in sorted(zip(scores, candidates),
                                       key=lambda t: t[0], reverse=True)]
        return float(max(scores)), ranked

    # 正式检索：稠密 + 稀疏两路并行召回，RRF 融合排名
    def hybrid(self, q, crop=None, topk=5, ef=64, appendix_only=False, text_only=False,
               extra_expr=None):
        from pymilvus import AnnSearchRequest, RRFRanker
        dense, sparse = self.encode(q)
        expr = self.build_expr(crop, appendix_only, text_only)
        # 意图路由：在作物过滤之上再叠一层「该找哪类知识」
        if extra_expr:
            # 意图路由表达式 and 到作物过滤上（两层过滤都要生效）
            expr = f"({expr}) and {extra_expr}" if expr else extra_expr

        # 两路各多取一倍候选（topk*2），给 reranker 留挑选余地
        reqs = [
            AnnSearchRequest(data=[dense], anns_field="dense_vector",
                             param={"metric_type": "COSINE", "params": {"ef": ef}},
                             limit=topk * 2, expr=expr),
            AnnSearchRequest(data=[sparse], anns_field="sparse_vector",
                             param={"metric_type": "IP"},
                             limit=topk * 2, expr=expr),
        ]
        # Milvus 原生混合检索：RRFRanker(k=60) 把两路排名融成一份
        # ⚠️ `sym_text` 必须取回来：它不进 `bm25_text`（实体表口径，见 _rerank_passage），
        #    不取就等于重排器物理上看不到症状——症状覆盖层会一整层失效。
        res = self.client.hybrid_search(
            self.collection, reqs, RRFRanker(k=60), limit=topk,
            output_fields=["crop", "std_no", "subtype", "is_high_risk",
                           "needs_verification_hint", "source_section", "bm25_text",
                           "sym_text"])
        # 连同标量元数据一起返回：上层覆盖检查/收窄/模板生成都要用
        return [{"chunk_id": h.id, "score": float(h.distance),
                 "crop": h.entity.get("crop"), "std_no": h.entity.get("std_no"),
                 "subtype": h.entity.get("subtype"),
                 "is_high_risk": h.entity.get("is_high_risk"),
                 "needs_verification_hint": h.entity.get("needs_verification_hint"),
                 "source_section": h.entity.get("source_section"),
                 "bm25_text": h.entity.get("bm25_text") or "",
                 "sym_text": h.entity.get("sym_text") or ""} for h in res[0]]


# ---------------------------------------------------------------- 主流程
# 【主流程 search()】一次查询的全部判定都收在这个函数里，出口只有一个 (status, ...)
def search(s, query, crop=None, topk=5, threshold=None):
    """返回 (状态, 归一化query, 意图, 结果列表, 说明)"""
    # 第①②步：先归一化，再识别意图（意图决定后面走哪条分支）
    q2, nhits = normalize(query, crop)
    intent = detect_intent(q2)

    # ③ 兜底 (a)：确定性
    #   **只在问「用药方案」时才触发**。pending 说的是「本知识库无该病虫害的
    #   **用药方案**」，若农户问的是物理防治（「我该挂几盏灯杀蝼蛄」），
    #   那条知识库里其实有（6.3.2.2 电子杀虫灯），拦掉就是误兜底——实测 2 条。
    # 第③(a)步：pending **不再硬兜底**（2026-09-23 改）
    #
    #   原逻辑：命中 pending 词 → 直接返回「暂无」，不检索。
    #
    #   问题：pending 的含义是「术语在语料里但**没有独立用药方案卡**」，
    #     而「没有用药方案」≠「没有知识」——枯萎病、病毒病在正文
    #     6.2「主要防治对象」里是有内容的。硬兜底把这部分仅有的正文
    #     也一起丢了，农户什么都拿不到。真实问法评测里这造成 3 条
    #     「该放宽却兜底」（辣椒病毒病 / 辣椒萎蔫 / 黄瓜枯萎病）。
    #
    #   为什么可以交给 coverage_check：它已实现三档判定，比 pending 的
    #     二档更精确——
    #       有卡        → ok
    #       正文提到过  → ok_noplan（正文 + 「暂无用药方案」提示）
    #       完全没提过  → 仍然硬兜底（fallback_nocoverage）
    #     pending 管的正是中间那档，交给它判即可。
    #
    #   安全性不降：no_plan 时检索带 text_only=True，**只回正文**，
    #     不会捞回别的病虫害的附录B 用药方案（见第⑤步）。
    #     2026-09-20 定稿口径即 ok_noplan，此前代码未落地。
    #
    #   check_pending 仍保留导出（serve/app.py 会 import PENDING），
    #     但主流程不再用它短路。
    pend = []

    # 闸门阈值：默认取 configs/retrieval_params.json，命令行可覆盖
    gate = PARAMS["relevance_threshold_rerank"]["value"] if threshold is None else threshold

    # ③b 实体级覆盖检查：问的病虫害在这个作物下有没有卡
    #     「黄瓜枯萎病用什么药」与「黄瓜炭疽病用什么药」结构一样，打分路分不开；
    #     但前者语料里根本没这个卡——确定性判断，与分数无关。
    # 第③(b)步 实体覆盖检查 → 三档结果：有卡 / 正文提到过 / 完全没提过
    miss, present, mentioned = coverage_check(q2, crop)
    # 三档判定：
    #   有卡      -> 正常返回
    #   正文提到过 -> 放宽到正文 + 「暂无用药方案」提示（no_plan=True）
    #   完全没提过 -> 硬兜底（不能拿本作物的通用内容糊弄农户）
    # 只有「正文提到过但没有用药方案」时才走放宽路径（no_plan=True）
    no_plan = bool(mentioned) and not present
    # 完全没提过 → 硬兜底，绝不拿本作物的通用内容糊弄农户
    if miss and not present and not mentioned:
        return ("fallback_nocoverage", q2, intent, [],
                f"语料中{crop or ''}无此病虫害：{'、'.join(miss)}")

    # ③c 实体收窄：query 点名了病虫害，就**只返回它**的卡。
    #     否则问「黄瓜霜霉病打什么药」会把灰霉病的方案一起带回来——
    #     农户可能照着打错药。这是覆盖检查的自然延伸：既然能判断"有没有"，
    #     也就能"只返回它"。
    # 第③(c)步 实体收窄：问句点名了病虫害，就只保留它自己的卡
    def narrow(rs):
        if not present:
            return rs
        # 只留 subtype 命中「问句点名实体」的结果
        keep = [r for r in rs
                if any(e in (r["subtype"] or "") for e in present)]
        # 收窄后为空则退回原结果，宁可不收窄也不丢答案
        return keep or rs       # 收窄后为空则退回原结果，不因收窄而丢答案

    # ③d 意图路由：决定「该找哪一类知识」。问「防治原则」就不该召回用药方案——
    #     实测 7 条粒度错位，农户问技术措施却拿到附录B 的农药方案。
    # 第③(d)步 意图路由：把意图翻译成「该找哪类知识」的过滤表达式
    route = None
    if intent is not None:
        route = INTENT_TO_ROUTE.get(intent["intent"], intent["intent"])
    # 【2026-09-23 修】点名了病虫害时，control_principle / symptom 路由会**劫持**它：
    #   实测「辣椒苗期猝倒病怎么防治」被 control_principle 的「怎么防」命中，
    #   路由到 6.1/6.3.1（防治原则/农业防治），反而绕开了猝倒病自己的附录B 卡 c01。
    #   试过「present 非空就不路由」，但太宽：它同时误伤正常的原则问句，
    #   auto 普通 95.6%→93.3%、collected 普通 22.2%→11.1%，已撤回。
    #   现在改成**不放弃路由，只把「点名实体自己的卡」并进来**（subtype like %实体%）：
    #   narrow() 本就会把结果收窄到这些卡，所以并进来只补回「有卡却召不回」。
    #   **只并 control_principle**：它的模式（「怎么防」「防病治虫」「综合防治」）都是
    #   能插进任何问句的泛词，路由本身就不可信；物理防治（「烫」「闷」）、
    #   主要防治对象（「有哪些病虫害」）这些是农户点名的知识类型，路由可信——
    #   并进实体卡会被 narrow 把路由选中的卡挤掉（实测 auto 普通 −2、collected 普通 −1）。
    route_expr = build_route_expr(route, crop) if route else None
    if route_expr and present and route == "control_principle":
        route_expr = (f"({route_expr}) or ("
                      + " or ".join(f'subtype like "%{e}%"' for e in present) + ")")

    # ④ chemical_control 意图：先只在附录B 里找，并用 reranker 当闸门
    # 第④步 高危路径：农户问「打什么药」→ 先在附录B 用药方案表里检索
    want_appendix = (intent is not None and intent["intent"] == "chemical_control" and not no_plan)
    # 【2026-09-27】宽召回池：RRF 只能出个大致排序（融合分近似常数），期望卡常被
    #   挤在 top-5 之外（known_issue 的 4 条正文排序不准即此因）。把最终候选池从
    #   topk 扩到 topk*2，交给 rerank_sort 精排后再截断——检索宽一点，排序交给
    #   交叉编码器。闸门照旧用池内最高分。
    pool_k = max(topk, 5)
    # 宽召回池只用于**无路由**的常规检索与营救用的全库池：RRF top-5 池会漏掉
    # 正确答案（如 collected「储存蒜头蛀食」的 c07 排 RRF 第 6+），交给 rerank_sort
    # 精排即可；带路由的检索维持 top-5 池——宽池会把路由拦住的方案卡带回来
    # （实测宽池 + 路由并存会回归 5 条 auto 用例）。
    wide_k = max(topk, 5) * 2
    if want_appendix:
        # 附录B 内混合检索；reranker 过闸门即返回（数值逐字来自卡片，安全）
        cand = s.hybrid(q2, crop=crop, topk=pool_k, appendix_only=True)
        if cand:
            # 2026-09-27：闸门 + 排序都听交叉编码器的（rerank_sort 同口径）
            r_app, cand_ranked = s.rerank_sort(q2, cand)
            if gate is None or r_app >= gate:
                return ("ok", q2, intent, narrow(cand_ranked)[:topk],
                        f"附录B 过滤（rerank {r_app:.4f}）")
        # 附录B 无结果或没过闸门 → **放宽过滤**，用正文兜底
        #   触发场景：问的对象在附录B 里没有方案（黄瓜的病毒病/枯萎病/疫病等，
        #   只在 6.2 主要防治对象或兼治里列了名，没有配套用药方案）
        # 附录B 无结果或没过闸门 → 放宽过滤，到正文里兜底
        cand_all = s.hybrid(q2, crop=crop, topk=pool_k, appendix_only=False,
                            text_only=no_plan)   # no_plan 只回正文，不碰别的病虫害方案
        # 放宽后仍为空 → 空结果兜底
        if not cand_all:
            return "fallback_empty", q2, intent, [], "检索结果为空"
        r_all, cand_all_ranked = s.rerank_sort(q2, cand_all)
        # 放宽后仍没过闸门 → 低分兜底（说明语料里确实没依据）
        if gate is not None and r_all < gate:
            return ("fallback_lowscore", q2, intent, [],
                    f"附录B 与全文均未过闸门（全文 rerank {r_all:.4f} < {gate}）")
        # no_plan：语料里没有该病虫害的用药方案，状态标成 ok_noplan，前端据此提示
        st = "ok_noplan" if no_plan else "ok_relaxed"
        return (st, q2, intent, narrow(cand_all_ranked)[:topk],
                f"已放宽到正文（rerank {r_all:.4f}）"
                + (f"；语料中{crop or ''}无此病虫害的用药方案：{'、'.join(miss)}" if no_plan else ""))

    # ⑤ 常规路径：RRF 取候选 → reranker 当闸门
    #    no_plan 时**只回正文**——否则会捞回别的病虫害的附录B 用药方案
    #    （实测「大蒜霜霉病用什么药」捞回了灰霉病的方案，比说「暂无」更危险）
    # 第⑤步 常规路径：混合检索（带意图路由过滤）+ reranker 闸门
    results = s.hybrid(q2, crop=crop, topk=pool_k if route_expr else wide_k,
                       text_only=no_plan, extra_expr=route_expr)
    # 空结果 → 兜底
    if not results:
        return "fallback_empty", q2, intent, [], "检索结果为空"

    # **覆盖检查已确认实体的问句，跳过分数闸门。**
    #   覆盖检查是确定性的：它说"这个病虫害在本作物下有卡"，答案就存在，
    #   再让 reranker 分数否决是多余的。实测这一步造成 15% 的误兜底——
    #   如「打完药的空瓶子空袋子…」RRF 已把 4.1.4 排第一，却因 rerank 0.185
    #   （长口语问句的分数系统性偏低）被拦掉。
    # 覆盖检查已确认「本作物有卡」→ 跳过分数闸门（分数路会误杀长口语问句）
    if present:
        return "ok", q2, intent, narrow(results)[:topk], f"实体已确认（{'、'.join(present)}）"

    # 其余情况：用 reranker 分数决定是返回结果还是兜底
    # 2026-09-27：rerank_sort 一次算分同时完成闸门判定与排序（原 RRF 排序对
    #   正文条款之间基本是噪声——known_issue 的 4 条排序不准即源于此）
    rscore, ranked = s.rerank_sort(q2, results)
    # 【2026-09-27 修】路由劫持仲裁（冠军营救版）：路由只是「该找哪类知识」的启发式，
    #   长问句里一个模式词（「主要病虫害」「防病治虫」）就能把路由拽错方向，把正确
    #   答案挡在池外——known_issue 的 4 条全部源于此（问品种被 symptom 路由到 6.2、
    #   问植保员被 control_principle 路由到 6.x）。
    #   规则：全库（无路由）池另排一遍，若其冠军是**正文卡**且 rerank 分比路由池
    #   冠军高出 0.05 以上，判为劫持，把这张正文卡**插到队首**（路由池其余结果保留）。
    #   只救正文卡、不救附录B 方案卡——路由的本职就是防止「问技术措施却拿到农药
    #   方案」，营救不能破坏这条底线（直接换成全库池会让方案卡回流，实测会回归 5 条）。
    if route_expr:
        open_results = s.hybrid(q2, crop=crop, topk=wide_k, text_only=no_plan)
        open_score, open_ranked = s.rerank_sort(q2, open_results)
        _champ = open_ranked[0] if open_ranked else None
        _is_plan = bool(_champ and _champ["chunk_id"].rsplit("-", 1)[0].endswith(tuple("0123456789"))
                        and "-p2" in _champ["chunk_id"] and "-c" in _champ["chunk_id"])
        if _champ and not _is_plan and open_score > rscore + 0.05:
            ranked = [_champ] + [c for c in ranked if c["chunk_id"] != _champ["chunk_id"]]
            rscore = max(rscore, open_score)
    if gate is not None and rscore < gate:
        # 【2026-09-23 修】路由过滤可能是误判——路由只是「精度」装置，不是安全装置
        #   （安全由 rerank 闸门把）。实测「打药防病治虫的人得有啥证」被「防病治虫」
        #   路由到 6.x 防治章节，把 3.2.6.2 植保员条款筛掉，只剩 0.0585 误兜底；
        #   去掉路由后同一张卡 0.9429。故路由没过闸门时去掉路由再问一次，
        #   仍不过才兜底——重试不是放水，闸门照旧（见 tests/test_retrieval_routing.py）。
        if route_expr:
            alt = s.hybrid(q2, crop=crop, topk=pool_k, text_only=no_plan)
            if alt:
                r_alt, alt_ranked = s.rerank_sort(q2, alt)
                if r_alt >= gate:
                    return ("ok_relaxed", q2, intent, narrow(alt_ranked)[:topk],
                            f"路由过滤没过闸门（rerank {rscore:.4f}），"
                            f"去掉路由重试 rerank {r_alt:.4f} 过")
        return ("fallback_lowscore", q2, intent, [],
                f"rerank {rscore:.4f} < 闸门 {gate}")
    st = "ok_noplan" if no_plan else "ok"
    why = f"rerank {rscore:.4f} 过"
    if no_plan:
        why += f"；语料中{crop or ''}无此病虫害的用药方案：{'、'.join(miss)}"
    return st, q2, intent, narrow(ranked)[:topk], why


# 命令行入口，便于单条调试：python src/retrieve/search.py --query "..." --crop 黄瓜
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--crop", default=None)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--uri", default=DEFAULT_URI)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--show-score", action="store_true")
    a = ap.parse_args()

    # 建 Searcher（加载模型 + 连库），然后跑一次主流程
    s = Searcher(uri=a.uri)
    status, q2, intent, results, why = search(s, a.query, a.crop, a.topk, a.threshold)

    print(f"query      : {a.query}")
    print(f"归一化     : {q2}")
    print(f"意图       : {intent['intent'] + ' / ' + intent['canonical_field'] if intent else '（无）'}")
    print(f"状态       : {status}" + (f"  ({why})" if why else ""))
    print()
    # 兜底状态：统一输出同一句「暂无防治方案」，不打印检索结果
    if status.startswith("fallback"):
        print(f"  响应 ⟶ 「{FALLBACK_MSG}」")
        return 0
    # 正常状态：逐条打印 分数/卡片ID/作物/防治对象，并标出高危需核实
    for i, r in enumerate(results, 1):
        flag = " ⚠️高危需核实" if r.get("needs_verification_hint") else ""
        print(f"  {i}. {r['score']:.5f}  {r['chunk_id']}  [{r['crop']}] {r['subtype']}{flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
