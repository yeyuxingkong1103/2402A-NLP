# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-Graph RAG优化任务
"""
工单编号：人工智能NLP-RAG-Graph RAG优化任务

模块：【层面1】第2层 LLM 抽取结果 → 合并入图谱

## 本模块解决的核心问题

工单08 的图谱（`kg.pkl`）实测只有**第 1 层**（词典 + 正则）的产物：
560 节点 / 873 边，仅 4 种关系（HAS_METRIC / REPORTED_IN / BELONGS_TO / IN_INDUSTRY），
**没有一个人物节点、没有一条控股或任职关系**。

但 08 的第 2 层 LLM 抽取其实**已经跑出了结果**
（`data/kg/llm_extract_results.jsonl`：349 块 / 2110 实体 / 127 关系，
含 Person 441 个、SERVES_AS 83 条、SHAREHOLDER_OF 37 条）——
**只是从来没有被合并进最终图谱**。08 的 `docs/12` 自己也承认：
「涉及『谁控股谁』『董监高任职』这类问题目前只能靠文本通道，图谱通道无贡献」。

本模块补上这缺失的一环，并在此基础上做**抽取质量的优化**（工单硬性要求1）。

## 四道闸门（相比 08 的三道，新增第 4 道）

08 在 `kg_prompt.to_entities_relations` 已设三道闸门（本体类型 / 类型组合 /
evidence 字面落地）。实测这三道**拦不住语义错误的关系**，例如：

    招商证券 --SHAREHOLDER_OF--> 招商资管
    evidence: 中国证监会及派出机构对招商资管出具了《…责令改正措施的决定》…

这句原文**同时出现了两个实体名**，字面落地闸门因此放行；
但句子讲的是**监管处罚**，完全没有表达「招商证券是招商资管的股东」。
这类「实体共现 ≠ 关系成立」的假阳性，是图谱精度的主要杀手。

故本模块新增：

  **闸门4（关系证据句式约束）** —— 每条关系类型要求 evidence 中出现
  该关系在金融年报中的**标志性表述**（如 SHAREHOLDER_OF 需要「股东/持股/控股/
  出资/股权」等）。没有标志性表述就说明这句话根本不是在陈述该关系，一律丢弃。
  这是「宁可漏抽、不可抽错」原则在关系层面的落地。

## 实体归一（08 完全缺失的能力）

  · **指代消解**：「本公司 / 该公司 / 本集团」按**来源文档**映射到主体公司
    （年报通篇用「本公司」自称，不消解就只是 441 个孤立的人名与一堆代词节点）；
  · **别名归一**：公司全称 / 简称统一到规范名（`COMPANY_TABLE` 为单一事实源）；
  · **泛称过滤**：剔除「公司」「银行」「集团」这类无指称价值的泛称节点。
"""
from __future__ import annotations

import io
import json
import os
import re
from collections import Counter

from w9 import config9

from src import config as c8
from src.kg_dict import COMPANY_TABLE
from src.kg_schema import ENTITY_TYPES, RELATION_TYPES, Entity, Relation
from src.kg_store import KnowledgeGraph

# ---------------------------------------------------------------------------
# 关系证据句式约束（闸门4）
# ---------------------------------------------------------------------------
# 每类关系在金融年报中的标志性表述。evidence 必须命中其中之一，
# 否则判定「这句话不是在陈述该关系」→ 丢弃。
# 关键词取自本语料（9 份年报）的真实用词，不是凭空拟定。
REL_EVIDENCE_KEYWORDS: dict[str, tuple[str, ...]] = {
    # 注意别把「股份」「合伙」放进来：它们几乎出现在每个公司全称里
    # （「招商证券**股份**有限公司」），会让闸门形同虚设。
    # 实测踩坑：因为收了「合伙」，把「杨如生先生现任瑞华会计师事务所合伙人」
    # 当成了「平安银行持有瑞华」的证据 —— 该句与持股毫无关系。
    "SHAREHOLDER_OF": ("股东", "持股", "控股", "出资", "股权",
                       "所有者权益", "发起人", "表决权"),
    "CONTROLS": ("控股", "控制", "全资子公司", "纳入合并范围", "实际控制"),
    "SERVES_AS": ("董事", "监事", "总经理", "行长", "总裁", "董事长",
                  "任职", "担任", "首席", "委员会", "高管", "职工代表",
                  "独立非执行", "秘书", "负责人"),
    "INVESTS_IN": ("投资", "出资", "入股", "认购", "增资", "参股", "持有"),
    "OPERATES": ("经营", "开展", "从事", "业务", "主营", "提供"),
    "COMPETES_WITH": ("竞争", "同业", "市场份额", "对手"),
    "ISSUES": ("发行", "上市", "募集", "承销", "配股", "增发"),
    "HAS_METRIC": ("率", "收入", "利润", "余额", "总额", "价值", "资产",
                   "负债", "收益", "每股", "分红", "投入", "覆盖率", "比"),
}

# 不参与合并的关系类型：这两类由第 1 层的词典/正则确定性产出，
# 精度 100%，让 LLM 边混进来只会拉低质量。
REL_SKIP_FROM_LLM = {"HAS_METRIC", "REPORTED_IN", "BELONGS_TO", "IN_INDUSTRY"}

# 明显不是实体的产物（LLM 常见噪声）：纯数字、单字、超长句片段
_PURE_NUM_RE = re.compile(r"^[\d\.,%％\-—]+$")

# 文档标题（不是「业务」）：年报里大量出现《…管理办法》《…章程》这类文件名，
# LLM 会把它们当成 Business 抽出来。实测样例：
#   招商银行 --OPERATES--> 招商银行股份有限公司高级管理人员薪酬管理办法
# 该「业务」实体毫无意义，且会污染图谱，故按后缀直接剔除。
_DOC_TITLE_RE = re.compile(
    r"(办法|规定|制度|准则|章程|细则|通知|公告|报告|方案|意见|指引|条例|"
    r"守则|政策|决议|议案|说明|附注)$")

# 不可作为实体的泛称（含英文）
GENERIC_NAMES = {"公司", "集团", "银行", "机构", "部门", "市场", "客户",
                 "股东", "员工", "board", "company", "the company"}

# ---------------------------------------------------------------------------
# 实体归一
# ---------------------------------------------------------------------------
# 名称规范化：全角括号 → 半角、去空白、去尾部标点
_WS_RE = re.compile(r"\s+")
_TRAIL_RE = re.compile(r"[，。；、,.;:：的]+$")


def norm_name(name: str) -> str:
    """名称规范化：统一括号、压缩空白、去尾标点。

    注意：`COMPANY_TABLE` 里写的是**全角**括号（如
    「中国平安保险（集团）股份有限公司」），而年报原文常写**半角**括号
    （「中国平安保险(集团)股份有限公司」）。若不对词典键做同样的规范化，
    这两种写法会被当成两个不同实体 —— 实测就是这样凭空多出了一个
    「中国平安保险(集团)股份有限公司」重复节点，并与「中国平安」之间
    连出一条自指式的 SHAREHOLDER_OF 假边。
    """
    s = _WS_RE.sub("", str(name or "").strip())
    s = s.replace("（", "(").replace("）", ")")
    s = _TRAIL_RE.sub("", s)
    return s.strip()


# 别名 → 规范简称（由 COMPANY_TABLE 构造，全称/简称/别名都指向同一个规范名）。
# 键**统一过 norm_name**，才能与归一后的实体名对上（全角/半角括号的坑见上）。
ALIAS_TO_CANON: dict[str, str] = {}
for _canon, _meta in COMPANY_TABLE.items():
    for _raw in (_canon, _meta.get("full", ""), *_meta.get("aliases", [])):
        _k = norm_name(_raw)
        if _k:
            ALIAS_TO_CANON.setdefault(_k, _canon)

# 公司全称 → 规范简称（指代消解用）
FULL_TO_CANON: dict[str, str] = {
    _meta.get("full", ""): _canon for _canon, _meta in COMPANY_TABLE.items()}

# 领域同义合并：历史名称 / 更名前称谓 → 现行规范名。
# 这些是**语料特有**的知识，词典（COMPANY_TABLE）里没有，但年报正文反复出现，
# 不合并就会把一个主体拆成两个节点。
DOMAIN_ALIASES: dict[str, str] = {
    "深圳发展银行": "平安银行",
    "深发展": "平安银行",
    "深圳发展银行股份有限公司": "平安银行",
    "中国平安保险(集团)股份有限公司": "中国平安",
    "中国太平洋保险(集团)股份有限公司": "中国太保",
    "中国邮政储蓄银行": "邮储银行",
    "中国邮政储蓄银行股份有限公司": "邮储银行",
}

# 来源文档 → 该文档的**主体公司**规范名（「本公司」指谁）
SOURCE_TO_CANON: dict[str, str] = {}
for _src, _full in c8.COMPANY_BY_SOURCE.items():
    _canon = FULL_TO_CANON.get(_full)
    if _canon:
        SOURCE_TO_CANON[_src] = _canon

# 指代表述（年报通篇以「本公司」自称）
COREF_SURFACES = {"本公司", "该公司", "本集团", "本公司股东", "本行",
                  "本公司及其子公司", "本集团及其子公司", "公司",
                  "本公司董事会", "本公司监事会", "母公司", "本银行"}

def _is_doc_title(name: str) -> bool:
    """是否像一份文件名（而非业务/实体）。"""
    return bool(_DOC_TITLE_RE.search(name)) and len(name) > 6


def resolve_entity(name: str, etype: str, source_doc: str
                   ) -> tuple[str, str] | None:
    """把 LLM 抽出的实体归一为图谱实体。

    返回 (规范名, 类型)；判定为非实体返回 None。
    归一顺序：规范化 → 指代消解 → 别名归一 → 泛称过滤。
    """
    nm = norm_name(name)
    if not nm or len(nm) < 2:
        return None
    if _PURE_NUM_RE.match(nm):
        return None
    # 超长「实体名」基本是 LLM 把整句抄成了实体
    if len(nm) > 40:
        return None

    # ① 指代消解：「本公司」等 → 来源文档的主体公司
    #    注意只在 Company/Department 语境下做，避免把「公司」当成实体名保留
    if nm in COREF_SURFACES:
        canon = SOURCE_TO_CANON.get(source_doc)
        if canon:
            return (canon, "Company")
        return None

    # ①.5 领域同义合并（历史名称 / 更名前称谓 → 现行规范名）。
    #     必须放在 Company 分支**之前** —— 「深圳发展银行」也是 Company 类型，
    #     若先落到下面的 Company 分支就直接原样返回了，历史名永远合并不掉
    #     （实测踩坑：`胡跃飞 --SERVES_AS--> 深圳发展银行` 与
    #      `胡跃飞 --SERVES_AS--> 平安银行` 并存，同一段履历被拆成两个主体）。
    dom = DOMAIN_ALIASES.get(nm)
    if dom:
        return (dom, "Company" if etype == "Company" else etype)

    # ② 别名归一：公司全称/别名 → 规范简称（仅 Company 类型）
    if etype == "Company":
        hit = ALIAS_TO_CANON.get(nm)
        if hit:
            return (hit, "Company")
        if nm in GENERIC_NAMES:
            return None
        return (nm, "Company")

    # ②.5 人物敬称剥离：「周建国先生」「蔡方方女士」与「周建国」「蔡方方」是同一人
    if etype == "Person":
        for suf in ("先生", "女士", "同志", "博士", "硕士"):
            if nm.endswith(suf) and len(nm) > len(suf) + 1:
                nm = nm[: -len(suf)]
                break

    # ④ 泛称过滤（非公司类型）
    if nm in GENERIC_NAMES:
        return None
    if etype not in ENTITY_TYPES:
        return None
    # 职务名不是人物实体
    if etype == "Person" and nm in ROLE_STOPWORDS:
        return None
    # 文档标题不是业务/实体（《…管理办法》《…章程》之类）
    if etype in ("Business", "Event") and _is_doc_title(nm):
        return None
    return (nm, etype)


# ---------------------------------------------------------------------------
# 闸门4：关系证据句式约束
# ---------------------------------------------------------------------------
def evidence_supports(rel_type: str, evidence: str) -> bool:
    """evidence 是否**在陈述**该关系（而非仅仅同时提到两个实体）。

    这是本模块相对工单08 新增的精度闸门。返回 False 即丢弃该关系。
    """
    kws = REL_EVIDENCE_KEYWORDS.get(rel_type)
    if not kws:
        return True                      # 未设约束的关系类型不拦
    ev = norm_name(evidence)
    return any(k in ev for k in kws)


def _evidence_grounded(evidence: str, source_text: str) -> bool:
    """闸门3（复用 08 口径）：evidence 必须真实出现在原文中（容忍空白差异）。"""
    if not evidence or not evidence.strip():
        return False
    norm = lambda s: "".join(str(s).split())        # noqa: E731
    return norm(evidence) in norm(source_text)


# 方向敏感的关系类型：方向搞反会把「子公司」说成「母公司」，是**语义错误**，
# 比漏抽更糟（错的关系会直接污染答案）。故单独设方向闸门。
DIRECTIONAL_RELS = ("SHAREHOLDER_OF", "CONTROLS", "INVESTS_IN")

# 「X 是股东」句式：股东/控股股东/实际控制人 后面紧跟的实体才是股东方。
# 实测踩坑：邮储银行年报原文「本行的控股股东邮政集团领取薪酬」，
# LLM 抽成了 `邮储银行 --SHAREHOLDER_OF--> 邮政集团`（方向完全反了）。
_SHAREHOLDER_HEAD_RE = re.compile(
    r"(控股股东|实际控制人|第一大股东|股东|发起人)[为是：:，,]?\s*$")


# 指代表述：证据里用「本行/本公司」自称主体公司时，等同于写明了主体公司名。
# 注意**不要**收「集团」：它出现在几乎每个保险/券商全称里
# （「中国平安保险（**集团**）股份有限公司」），收进来会让闸门5 形同虚设 ——
# 实测就是这样把「储一昀在别家任独董」误判成「证据提到了平安银行」。
COREF_TERMS = ("本行", "本公司", "本集团", "该行", "该公司", "本银行", "本集团")

# 职务名词不是人物实体：年报里「高级管理人员」「董事」「监事」是**角色**而非具体的人，
# LLM 会把它们当 Person 抽出来，形成「高级管理人员 --SERVES_AS--> 平安银行」这类
# 没有指称对象的节点。检索时还会与真实人名抢命中。
ROLE_STOPWORDS = {
    "高级管理人员", "董事", "监事", "股东", "高管", "管理层", "员工",
    "独立董事", "非执行董事", "执行董事", "职工代表", "董事会", "监事会",
    "全体股东", "投资者", "高级管理层", "股东代表", "关联方", "实际控制人",
}


def mentions_tail(tail_name: str, evidence: str, aliases: list | None,
                  tail_is_subject: bool = False) -> bool:
    """闸门6：证据是否**提到了关系的对象**（尾实体）。

    为什么需要：08 的闸门3 只检查「evidence 这段文字确实出自原文」，
    并不检查这段文字**跟本关系的对象有没有关系**。于是出现：

        储一昀 --SERVES_AS--> 平安银行
        evidence: 兼任中国平安保险（集团）股份有限公司、泰豪科技股份有限公司、
                  环旭电子股份有限公司和嘉兴银行股份有限公司独立董事。

    这句原文确凿无误地出自平安银行年报，也确实在讲「任职」，闸门1~4 全部放行
    —— 但它讲的是储一昀在**别的四家公司**任独立董事，与平安银行无关。

    本闸门要求：证据必须点到关系的对象（尾实体），或对象正是本文件的主体公司
    而证据用了「本行/本公司」自称。这是「证据必须切题」的底线检查。
    """
    ev = norm_name(evidence)
    if not ev:
        return False
    for nm in [tail_name, *(aliases or [])]:
        n = norm_name(nm)
        if len(n) >= 2 and n in ev:
            return True
    if tail_is_subject and any(t in ev for t in COREF_TERMS):
        return True
    return False


def direction_ok(rel_type: str, head_name: str, tail_name: str,
                 evidence: str) -> bool:
    """闸门5：关系方向是否与证据一致（仅对方向敏感的关系生效）。

    判据：如果证据里出现「（控股）股东/实际控制人 + 尾实体名」的句式，
    说明**尾实体**才是股东，此时若把 head 当股东就是方向反了 → 拒绝。
    宁可漏掉这条事实，也不让一条方向相反的关系进图谱。
    """
    if rel_type not in DIRECTIONAL_RELS:
        return True
    ev = norm_name(evidence)
    if not ev or not tail_name:
        return True
    for m in re.finditer(re.escape(tail_name), ev):
        prefix = ev[max(0, m.start() - 12):m.start()]
        if _SHAREHOLDER_HEAD_RE.search(prefix):
            # 尾实体名前面就是「股东」字样 → 尾才是股东方
            # 例外：证据同时以 head 为主语陈述了「head 持有 tail」
            if re.search(re.escape(norm_name(head_name)) + r"(对|持有|持有对)?"
                         r"[^。；]{0,8}" + re.escape(tail_name), ev):
                return True
            return False
    return True


# ---------------------------------------------------------------------------
# 载入 08 的第2层结果
# ---------------------------------------------------------------------------
def load_wo8_layer2(path: str | None = None) -> list[dict]:
    """读 08 的 LLM 抽取结果（JSONL，每行 {chunk_id, entities, relations}）。"""
    path = path or config9.WO8_LLM_RESULTS
    if not os.path.isfile(path):
        return []
    out: list[dict] = []
    with io.open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


# 图谱节点的统一表示：(规范名, 类型, 别名, 来源文档, 页码)
# resolve_entity 只产出前两项，name2ent 产出全部五项，统一在此补齐，
# 避免下游按不同元数解包（实测踩坑：少写一位就 ValueError 崩在入图环节）。
NodeTuple = tuple[str, str, list, str, int]


def _as_node(v, src: str, page: int) -> NodeTuple | None:
    """把 (name, type) 或 (name, type, aliases, src, page) 统一成五元组。"""
    if not v:
        return None
    if len(v) >= 5:
        return (v[0], v[1], list(v[2] or []), v[3] or src, v[4] if v[4] is not None
                else page)
    return (v[0], v[1], [], src, page)


def chunk_key(source: str, chunk_id: str) -> tuple[str, str]:
    """块的全局唯一键 = (来源文档, chunk_id)。

    ## 为什么不能用 chunk_id 单独作键（实测踩坑，重要）

    工单08 的 `chunks.json` 有 7757 条块，但 `chunk_id` 只有 **1185 个不同值**
    —— 每个文档的分块序号都**从 c000001 重新开始**，`c000001` 在 9 份年报里
    各出现一次。

    若按 `chunk_id` 单独建索引，9 份文档的同号块会互相覆盖，只剩最后一份的文本。
    后果：闸门3 拿**B 文档的原文**去核验**A 文档的 evidence**，必然不匹配，
    把大量正确的关系误判为「证据不落地」——实测 127 条关系里被误杀 118 条。

    这本身也是工单08 在多文档场景下的一个隐患：任何以 chunk_id 为全局主键的
    下游逻辑（引用回溯、块级去重）都会跨文档串味。本模块按 (source, chunk_id)
    建键，从根上避开。
    """
    return (str(source or ""), str(chunk_id or ""))


def load_chunk_texts(path: str | None = None) -> dict[tuple[str, str], str]:
    """(来源文档, chunk_id) → 原文（闸门3 需要原文核验 evidence）。"""
    path = path or config9.WO8_CHUNKS
    if not os.path.isfile(path):
        return {}
    with io.open(path, encoding="utf-8") as f:
        chunks = json.load(f)
    out: dict[tuple[str, str], str] = {}
    for c in chunks if isinstance(chunks, list) else []:
        cid = c.get("chunk_id") or c.get("id")
        if not cid:
            continue
        src = c.get("source") or (c.get("attrs") or {}).get("source_doc") or ""
        out[chunk_key(src, cid)] = c.get("text") or ""
    return out


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def merge_layer2(kg: KnowledgeGraph, records: list[dict],
                 chunk_texts: dict[tuple[str, str], str] | None = None,
                 verbose: bool = True) -> tuple[KnowledgeGraph, dict]:
    """把第2层 LLM 抽取结果合并进图谱（原地修改 kg 并返回统计）。

    ## 为什么是两遍扫描，而不是「边抽边入图」

    08 的 LLM 抽取产出了 1276 个 Company 实体，其中绝大多数是**孤立噪声**
    （句子片段、泛称、只被提到一次的子机构名）。若先无条件入图，会：
      · 把图谱从 560 节点膨胀到 2500+ 节点，多数毫无信息；
      · 污染 `find_by_alias` —— 检索时一个常见词会命中几十个噪声节点，
        实体链接直接被淹没（这正是 08 检索层 R1 问题的放大器）。

    故本模块**先判关系、后定节点**：

      第一遍：把每块的关系过完四道闸门，收集**被接受的关系**及其归一后端点；
      第二遍：只把「在已接受关系中真正充当端点的实体」以及「文档主体公司」
            写入图谱 —— 一个实体之所以进图，是因为它在某条被证实的关系里
            扮演了角色，而不是因为它被 LLM 提了一嘴。

    结果是图谱**更小但更密**：节点数可控，而每条边都经过四道闸门。

    统计里逐类记录**每条闸门拒绝了多少**，供优化方案文档引用 ——
    不给出拒绝原因的「合并了 N 条」是不可审计的。
    """
    chunk_texts = chunk_texts or load_chunk_texts()

    stats = {
        "blocks": len(records),
        "ent_in": 0, "ent_kept": 0, "ent_dropped_isolated": 0,
        "rel_in": 0, "rel_kept": 0,
        "reject": Counter(),
        "by_type_added": Counter(),
        "by_rel_added": Counter(),
        "coref_resolved": 0, "alias_merged": 0,
    }

    # ---- 第一遍：逐块归一实体、过滤关系，收集候选 --------------------------
    # accepted: [(rel_dict, head_id, tail_id, src, page, confidence, chunk_id)]
    accepted: list[tuple] = []
    # endpoints: eid -> (name, type, aliases, src, page) —— 只登记进入关系者
    endpoints: dict[str, tuple] = {}

    stats["chunk_text_miss"] = 0
    for rec in records:
        cid = str(rec.get("chunk_id") or "")
        ents = rec.get("entities") or []
        rels_raw = rec.get("relations") or []
        if not ents:
            continue

        # 该块的来源文档/页码（08 序列化时把 provenance 写进了每个实体 attrs）
        src0 = ""
        page0 = -1
        for e in ents:
            if isinstance(e, dict) and e.get("attrs"):
                src0 = e["attrs"].get("source_doc") or src0
                if src0:
                    page0 = int(e["attrs"].get("page", -1) or -1)
                    break
        if not src0:
            src0 = (rels_raw[0].get("source_doc") if rels_raw else "") or ""
        # 必须用 (来源文档, chunk_id) 取原文 —— 单用 chunk_id 会跨文档串味
        text = chunk_texts.get(chunk_key(src0, cid), "")
        if not text:
            stats["chunk_text_miss"] += 1

        # ① 归一本块实体 → 名称到 (规范名, 类型, 别名) 的映射（暂不入图）
        name2ent: dict[str, tuple] = {}
        for raw in ents:
            if not isinstance(raw, dict):
                continue
            stats["ent_in"] += 1
            orig = norm_name(raw.get("name", ""))
            rn = resolve_entity(raw.get("name", ""), str(raw.get("type", "")), src0)
            if rn is None:
                stats["reject"]["实体被归一规则剔除"] += 1
                continue
            cname, ctype = rn
            if orig != cname:
                if orig in COREF_SURFACES:
                    stats["coref_resolved"] += 1
                else:
                    stats["alias_merged"] += 1
            aliases = [norm_name(a) for a in (raw.get("aliases") or [])
                       if norm_name(a)]
            # 原文各种写法都登记，供关系端点按字面命中
            for key in (orig, cname, *aliases):
                if key:
                    name2ent.setdefault(key, (cname, ctype, aliases, src0, page0))

        # ② 关系过四道闸门
        for raw in rels_raw:
            if not isinstance(raw, dict):
                continue
            stats["rel_in"] += 1
            rtype = str(raw.get("rel_type") or raw.get("type") or "").strip()

            # 闸门1：关系类型在 08 本体之内
            if rtype not in RELATION_TYPES:
                stats["reject"]["闸门1 关系类型不在本体"] += 1
                continue
            # 第1层已确定性产出的关系类型不接受 LLM 版本（精度保护）
            if rtype in REL_SKIP_FROM_LLM:
                stats["reject"]["第1层已确定性覆盖，跳过LLM边"] += 1
                continue

            head_raw = norm_name(raw.get("head_name") or
                                 str(raw.get("head_id", "")).split(":", 1)[-1])
            tail_raw = norm_name(raw.get("tail_name") or
                                 str(raw.get("tail_id", "")).split(":", 1)[-1])
            head_type = str(raw.get("head_id", "")).split(":", 1)[0]
            tail_type = str(raw.get("tail_id", "")).split(":", 1)[0]

            # 端点归一（含指代消解）；能归一则以其类型为准
            hn = _as_node(resolve_entity(head_raw, head_type, src0)
                          or name2ent.get(head_raw), src0, page0)
            tn = _as_node(resolve_entity(tail_raw, tail_type, src0)
                          or name2ent.get(tail_raw), src0, page0)
            if not hn or not tn:
                stats["reject"]["端点实体无法归一"] += 1
                continue
            head_id, tail_id = f"{hn[1]}:{hn[0]}", f"{tn[1]}:{tn[0]}"

            # 闸门2：类型组合合法（用**归一后**的类型判定）
            spec = RELATION_TYPES[rtype]
            if spec.get("head") and hn[1] not in spec["head"]:
                stats["reject"][f"闸门2 {rtype} 头类型 {hn[1]} 非法"] += 1
                continue
            if spec.get("tail") and tn[1] not in spec["tail"]:
                stats["reject"][f"闸门2 {rtype} 尾类型 {tn[1]} 非法"] += 1
                continue
            if head_id == tail_id:
                stats["reject"]["自环关系"] += 1
                continue

            evidence = str(raw.get("evidence") or "").strip()
            # 闸门3：evidence 必须在原文中真实出现
            if text and not _evidence_grounded(evidence, text):
                stats["reject"]["闸门3 evidence 未在原文落地"] += 1
                continue
            # 闸门4（本工单新增）：evidence 必须**在陈述**该关系
            if not evidence_supports(rtype, evidence):
                stats["reject"][f"闸门4 {rtype} 证据未表达该关系"] += 1
                continue
            # 闸门5（本工单新增）：证据必须点到关系对象（否则是「文不对题」）
            if not mentions_tail(tn[0], evidence, tn[2],
                                 tail_is_subject=(tn[0] in SOURCE_TO_CANON.values()
                                                  and src0 in SOURCE_TO_CANON
                                                  and SOURCE_TO_CANON[src0] == tn[0])):
                stats["reject"][f"闸门5 {rtype} 证据未提及关系对象"] += 1
                continue
            # 闸门6（本工单新增）：方向敏感关系的方向必须与证据一致
            if not direction_ok(rtype, hn[0], tn[0], evidence):
                stats["reject"][f"闸门6 {rtype} 方向与证据相反"] += 1
                continue

            try:
                conf = min(max(float(raw.get("confidence", 0.8) or 0.8), 0.0), 0.99)
            except (TypeError, ValueError):
                conf = 0.8
            accepted.append((Relation(
                head_id=head_id, rel_type=rtype, tail_id=tail_id,
                evidence=evidence[:120],
                source_doc=str(raw.get("source_doc") or src0),
                page=int(raw.get("page") or page0 or -1),
                confidence=conf,
                attrs={"layer": "llm", "chunk_id": cid}), hn, tn))

    # ---- 第二遍：只入「在已接受关系中充当端点」的实体 + 文档主体公司 --------
    for rel, hn, tn in accepted:
        for node in (hn, tn):
            endpoints.setdefault(f"{node[1]}:{node[0]}", node)
    for src, canon in SOURCE_TO_CANON.items():
        endpoints.setdefault(f"Company:{canon}", (canon, "Company", [], src, -1))

    for eid, (cname, ctype, aliases, src, page) in endpoints.items():
        try:
            kg.add_entity(Entity(entity_id=eid, name=cname, type=ctype,
                                 aliases=list(aliases),
                                 attrs={"layer": "llm", "source_doc": src,
                                        "page": page}))
        except ValueError:                       # noqa: BLE001
            stats["reject"]["实体非法"] += 1
            continue
        stats["ent_kept"] += 1
        stats["by_type_added"][ctype] += 1

    for rel, _hn, _tn in accepted:
        if not (kg.has_entity(rel.head_id) and kg.has_entity(rel.tail_id)):
            stats["reject"]["端点不在图中"] += 1
            continue
        if kg.add_relation(rel):
            stats["rel_kept"] += 1
            stats["by_rel_added"][rel.rel_type] += 1
        else:
            stats["reject"]["重复边"] += 1

    stats["ent_dropped_isolated"] = max(
        stats["ent_in"] - stats["ent_kept"], 0)
    stats["reject"] = dict(stats["reject"])
    stats["by_type_added"] = dict(stats["by_type_added"])
    stats["by_rel_added"] = dict(stats["by_rel_added"])

    if verbose:
        print(f"[kg_merge] 实体 {stats['ent_in']} → 入图 {stats['ent_kept']}"
              f"（孤立实体不入图 {stats['ent_dropped_isolated']}）", flush=True)
        print(f"[kg_merge] 关系 {stats['rel_in']} → 入图 {stats['rel_kept']}",
              flush=True)
        print(f"[kg_merge] 指代消解 {stats['coref_resolved']} 次，"
              f"别名归一 {stats['alias_merged']} 次", flush=True)
        for k, v in sorted(stats["reject"].items(), key=lambda x: -x[1])[:10]:
            print(f"[kg_merge]   丢弃 {v:5d}  {k}", flush=True)
    return kg, stats


def build_kg9(verbose: bool = True) -> dict:
    """构建工单09 图谱：08 第1层 + 08 第2层（四道闸门 + 实体归一）。

    返回统计并落盘到 config9 的路径。**不修改 08 的任何文件**。
    """
    import time
    t0 = time.perf_counter()
    kg = KnowledgeGraph.load(config9.WO8_KG_PICKLE)      # 08 第1层（只读加载）
    base = kg.stats()

    records = load_wo8_layer2()
    texts = load_chunk_texts()
    if verbose:
        print(f"[kg_merge] 08 第1层图谱：{base['n_nodes']} 节点 / "
              f"{base['n_edges']} 边", flush=True)
        print(f"[kg_merge] 08 第2层 LLM 结果：{len(records)} 块，"
              f"原文索引 {len(texts)} 块", flush=True)

    _, mstats = merge_layer2(kg, records, texts, verbose=verbose)
    after = kg.stats()

    out = {
        "baseline_08": base,
        "merged_09": after,
        "merge": mstats,
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }
    kg.save(config9.KG9_PICKLE, config9.KG9_GRAPHML)
    kg.dump_jsonl(config9.KG9_ENTITIES, config9.KG9_RELATIONS)
    with io.open(config9.KG9_STATS, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    # 合并过程逐条留痕（可审计）
    with io.open(config9.KG9_MERGE_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({**mstats, "baseline_08": base, "merged_09": after},
                           ensure_ascii=False) + "\n")
    if verbose:
        print(f"[kg_merge] 09 图谱：{after['n_nodes']} 节点 / "
              f"{after['n_edges']} 边（{out['elapsed_s']}s）", flush=True)
        print(f"[kg_merge] 已写出 {config9.KG9_PICKLE}", flush=True)
    return out


if __name__ == "__main__":
    import sys
    from w9 import compat  # noqa: F401
    res = build_kg9()
    print(json.dumps(res["merged_09"], ensure_ascii=False, indent=2))
