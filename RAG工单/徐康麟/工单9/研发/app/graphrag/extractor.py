# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-Graph RAG 优化任务
# 来源说明：本文件复用自工单7/工单8（上游只读基座），工单9 未修改其业务逻辑，
#           仅补齐工单9 的编号头注释以满足交付纪律（_共享约定.md §4.2）。
"""工单8 图谱抽取器：**规则 / 词典 / 正则** 为主，可选 Ollama 关系补抽。

工单编号：人工智能NLP-RAG-基于Graph RAG 实现金融问答

为什么以规则为主（设计取舍，不是偷懒）：
    1. 语料是 9 份格式高度规范的年报，「公司全称 / 指标 / 金额 / 比例 / 日期」都有稳定书写形态，
       正则的**精确率**显著高于小模型自由生成，而且**零幻觉**（每个实体都能回指原文片段）；
    2. 抽取产物要写「来源块 + 证据片段 + 置信度」，规则抽取天然满足可审计要求；
    3. 本机只有 CPU 与 3B 小模型，全量 7325 块走 LLM 需数小时，与「最小闭环优先」冲突。
       因此 LLM **只用于抽样补抽同句内未匹配到正则的隐性关系**（默认 500 块），
       补抽结果单独记账（``extractor="llm"``），与规则关系分开统计，便于对比增量。

抽取器输出（纯 Python 数据结构，落盘由 ``build_graph.py`` 负责）：
    * 实体：``{id, name, type, confidence, count, chunks, aliases}``
    * 关系：``{id, head, head_type, relation, tail, tail_type, confidence, extractor, evidence, chunk_id, page, file_name}``
    * 提及：``{entity_id, chunk_id, page, file_name, confidence, snippet}``
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .schema import (CITY_TERMS, ENTITY_TYPES, METRIC_TERMS, PERSON_STOPWORDS, PRODUCT_TERMS,
                     RELATION_TYPES, SURNAMES, relation_canonical, is_valid_relation)

WORK_ORDER = "人工智能NLP-RAG-基于Graph RAG 实现金融问答"

# ---------------------------------------------------------------------------
# 正则表（每条都带命名与置信度键，便于「抽取规则的迭代」在 优化/ 里逐条对比）
# 说明：所有模式都加了**左边界**（`(?<![\u4e00-\u9fff])` 或标点前置），
#       否则「与平安集团」「年政策性银行」这类跨词粘连会被当成实体（第一轮实测踩坑）。
# ---------------------------------------------------------------------------
_RE_BOUNDARY = r"(?<![\u4e00-\u9fffA-Za-z])"
RE_ORG = re.compile(_RE_BOUNDARY + r"[\u4e00-\u9fffA-Za-z]{2,20}(?:股份有限公司|有限责任公司|有限公司|集团)")
RE_INSTITUTION = re.compile(
    _RE_BOUNDARY + r"[\u4e00-\u9fff]{2,15}"
    r"(?:交易所|监督管理委员会|管理委员会|委员会|会计师事务所|律师事务所|研究院|大学|"
    r"银行|证券公司|保险公司)"
)
# 金额：整段裸数字/千分位 + 单位（`50亿元` / `1,379.58亿元` / `269,005.23万元`）
RE_AMOUNT = re.compile(_RE_BOUNDARY + r"(?:人民币)?\d[\d,]*(?:\.\d+)?\s*(?:亿元|万元|千元|百万元|元|亿美元|亿港元)")
# 比例：**必须带 % / 个百分点 / 倍**，且不紧跟中文（`14.5%的` 也是合法比例，故只限左边界）
RE_RATIO = re.compile(_RE_BOUNDARY + r"\d+(?:\.\d+)?\s*(?:%|个百分点|倍)")
# 日期：只收「完整日期」与「年度报告期」；**不收裸年份**（`2019年` 在每页都出现，是无信息量的高频噪声）
RE_DATE = re.compile(r"(?:19|20)\d{2}\s*[-/年]\s*\d{1,2}\s*[-/月]\s*\d{1,2}\s*日?")
RE_PERIOD = re.compile(r"(?:19|20)\d{2}\s*年(?:度|末|上半年|1-12月)")
# 指标：必须含**指标中心词**（否则正则的“率/额/…”会吃掉普通名词）
_METRIC_CORE = ("收入", "利润", "资产", "负债", "权益", "收益", "资本", "保费", "股息", "分红",
                "占比", "增速", "成本", "支出", "准备", "覆盖", "价值", "缺口", "余额", "总额",
                "每股", "收益率", "充足率", "覆盖率", "拨贷比", "净利差", "净息差", "现金流")
RE_METRIC = re.compile(
    _RE_BOUNDARY + r"[\u4e00-\u9fff]{2,12}"
    r"(?:率|额|收入|利润|资产|负债|权益|收益|资本|保费|股息|分红|占比|增速|成本|支出|准备|覆盖|价值|缺口)"
)
# 产品：只认「书名号/引号包裹的专名」与「带业务后缀的词」，避免把口号、章节名当产品
RE_PRODUCT_QUOTED = re.compile(r"[“《]([^”》]{2,20})[”》]")
RE_PRODUCT_SUFFIX = re.compile(
    _RE_BOUNDARY + r"[\u4e00-\u9fff]{2,10}(?:计划|平台|家园|管家|蓝本|联盟|模式|体系|机制|工程|指引|产品线)"
)
_TITLE = ("董事长", "副董事长", "行长", "副行长", "总经理", "副总经理", "总裁", "副总裁",
          "首席执行官", "首席财务官", "首席信息官", "首席风险官", "监事长", "监事",
          "财务负责人", "会计机构负责人", "董事会秘书", "法定代表人")
RE_PERSON_TITLE = re.compile(
    "(?:" + "|".join(_TITLE) + r")(?:兼[\u4e00-\u9fff]{1,6}?)?[\s:：，,、]{0,3}([\u4e00-\u9fff]{2,4})"
)
RE_PERSON_NAME_TITLE = re.compile(
    r"([\u4e00-\u9fff]{2,3})(?:先生|女士|同志)?[\s，,]{0,2}(?:担任|任|系|为)?"
    r"(?:" + "|".join(_TITLE) + ")"
)
#: 人名黑名单词（出现在“姓名”里就说明抽到的是职务/机构残片而不是人）
_PERSON_BAD_CHARS = ("书", "记", "理", "事", "长", "行", "裁", "监", "管", "总", "任", "职",
                     "部", "级", "员", "会", "团", "办", "局", "厅", "委", "席", "秘",
                     "计", "划", "度", "期", "项", "目", "组", "司", "业")
#: 公司名黑名单（不含真实主体的后缀残片 / 泛称）
ORG_STOPWORDS: frozenset[str] = frozenset({
    "股份有限公司", "有限责任公司", "有限公司", "公司", "集团", "本集团", "本公司", "本行", "该行",
    "母公司", "子公司", "全资子公司", "控股子公司", "保险集团", "银行集团", "企业集团", "集团公司",
})
#: 产品黑名单（章节名、口号、非产品词）
PRODUCT_STOPWORDS: frozenset[str] = frozenset({
    "第十章财务报告", "财务报告", "审计报告", "年度报告", "内部控制", "公司治理", "组织结构图",
    "子公司", "分支机构", "营业网点", "联营企业", "合营企业",
})
#: 机构黑名单（业务条线/口号被误判时兜底）
INSTITUTION_STOPWORDS: frozenset[str] = frozenset({
    "交易银行", "零售银行", "商业银行", "政策性银行", "投资银行", "银行集团",
})
_SENTENCE_SPLIT = re.compile(r"[。；;!?！？\n]")
_CLAUSE_SPLIT = re.compile(r"[，,、（）()【】\[\]“”\s]")
_NUM_IN_NAME = re.compile(r"[0-9]")

#: 关系触发词 → 规范关系名（正则抽取的第一层）
RELATION_TRIGGERS: tuple[tuple[str, str], ...] = (
    (r"(?:发行|公开发行|上市|派发|每股现金分红|每股现金股息|回购)", "发行"),
    (r"(?:披露|公告|年报显示|年报载明|报告显示|报告载明)", "披露"),
    (r"(?:持有|持股|控股|参股|认购|全资子公司|控股子公司)", "持股"),
    (r"(?:董事长|行长|总经理|总裁|首席执行官|监事长|财务负责人|董事会秘书|法定代表人)", "任职"),
    (r"(?:归属于|归属为|属于)", "归属于"),
    (r"(?:同比|比上年同期|较上年末|同比增减)", "同比变动"),
    (r"(?:位于|设在|地处|总部)", "位于"),
    (r"(?:母公司|子公司|本行|本公司|集团)", "母子公司"),
    (r"(?:拥有|设立|开展|打造|推出|发布|筹建|建立)", "拥有"),
)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class Entity:
    """图谱节点。"""

    id: str
    name: str
    type: str
    confidence: float
    count: int = 0
    aliases: list[str] = field(default_factory=list)
    #: 该实体出现过的块（按置信度降序保留 top-N，N = extract.mentions_per_entity）
    chunks: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "type": self.type,
                "type_zh": ENTITY_TYPES.get(self.type, {}).get("zh", self.type),
                "confidence": round(float(self.confidence), 4), "count": int(self.count),
                "aliases": list(self.aliases), "chunks": list(self.chunks)}


@dataclass(slots=True)
class Relation:
    """图谱边（``提及`` 关系不在此结构里单独存，由 Entity.chunks 派生，避免 6 万条重复记录）。"""

    head: str
    relation: str
    tail: str
    head_type: str
    tail_type: str
    confidence: float
    extractor: str
    evidence: str
    chunk_id: str = ""
    page: int = 0
    file_name: str = ""

    def key(self) -> tuple[str, str, str]:
        return (self.head, self.relation, self.tail)

    def to_dict(self) -> dict[str, Any]:
        return {"head": self.head, "relation": self.relation, "tail": self.tail,
                "head_type": self.head_type, "tail_type": self.tail_type,
                "confidence": round(float(self.confidence), 4), "extractor": self.extractor,
                "evidence": self.evidence[:200], "chunk_id": self.chunk_id, "page": int(self.page),
                "file_name": self.file_name}


# ---------------------------------------------------------------------------
# 抽取器
# ---------------------------------------------------------------------------
class GraphExtractor:
    """规则 / 词典 / 正则抽取器（无任何网络依赖，可离线跑；LLM 补抽是独立可选步骤）。"""

    def __init__(self, cfg: Any, *, logger: Any = None) -> None:
        self.cfg = cfg
        self.log = logger
        self.settings: dict[str, Any] = dict(getattr(cfg, "extract", {}) or {})
        self.conf: dict[str, float] = dict(self.settings.get("confidence", {}))
        self.max_scan = int(self.settings.get("max_scan_chars", 4000))
        self.max_entity_len = int(self.settings.get("max_entity_len", 30))
        self.min_entity_len = int(self.settings.get("min_entity_len", 2))
        self.metrics = sorted(set(METRIC_TERMS), key=len, reverse=True)
        self.products = sorted(set(PRODUCT_TERMS), key=len, reverse=True)
        self.cities = set(CITY_TERMS)
        self._entities: dict[str, Entity] = {}
        self._relations: dict[tuple[str, str, str], Relation] = {}
        self._stats: dict[str, int] = {}

    # -- 工具 -------------------------------------------------------------
    @staticmethod
    def normalize_name(name: str) -> str:
        """实体名归一化：去空白/全角空格、去首尾标点、全角括号转半角。"""
        text = str(name or "").strip()
        text = text.replace("\u3000", "").replace(" ", "")
        text = text.replace("（", "(").replace("）", ")")
        return text.strip("，,。.、；;：:\"'“”()《》【】[]")

    @staticmethod
    def entity_id(entity_type: str, name: str) -> str:
        """节点 id = **规范化实体名**（不掺类型前缀）。

        ⚠️ 实测踩坑（pytest 抓到）：原实现用 ``f"{type}:{name}"``，于是**同名不同型**会变成两个节点
        （「私人银行」既被判为 product 又被判为 metric），关系表里就出现
        ``head == tail == 私人银行`` 的**假自环**（``head_type`` / ``tail_type`` 不同，但名字相同），
        图谱的「无自环」不变量被破坏，可视化里也会出现自连边。
        改为「一名一节点」后：同名实体只保留首次判定的类型，后续关系都挂到同一个节点上，
        自环检查（``head.id == tail.id``）也就真正等价于「名字相同」。
        """
        return str(name)

    def _bump(self, key: str, amount: int = 1) -> None:
        self._stats[key] = self._stats.get(key, 0) + amount

    def _add_entity(self, name: str, entity_type: str, confidence: float,
                    chunk: Mapping[str, Any], snippet: str = "") -> Entity | None:
        clean = self.normalize_name(name)
        if not clean or len(clean) < self.min_entity_len or len(clean) > self.max_entity_len:
            return None
        if clean in PERSON_STOPWORDS:
            self._bump(f"dropped_stopword_{entity_type}")
            return None
        # 类型各自的形态闸门（第一轮全量前实测补的规则，见 优化/图谱抽取优化说明.md）
        if entity_type == "company" and (clean in ORG_STOPWORDS or clean.startswith(("与", "原", "该", "本", "年"))):
            self._bump("dropped_org_form")
            return None
        if entity_type == "institution" and clean in INSTITUTION_STOPWORDS:
            self._bump("dropped_institution_stopword")
            return None
        if entity_type == "product" and (clean in PRODUCT_STOPWORDS or _NUM_IN_NAME.search(clean)):
            self._bump("dropped_product_form")
            return None
        if entity_type == "person":
            if len(clean) > 3 or clean[0] not in SURNAMES or any(ch in clean for ch in _PERSON_BAD_CHARS):
                self._bump("dropped_person_form")
                return None
        if entity_type == "metric" and not any(core in clean for core in _METRIC_CORE):
            self._bump("dropped_metric_form")
            return None
        if entity_type == "ratio" and _is_uninformative_ratio(
                clean, limit=float(self.settings.get("uninformative_ratio_max", 10.0))):
            # 单位数比例（0% / 5% / 10% / 1倍）几乎出现在每一页财务表里：度极大但零信息量，
            # 两跳扩展会把整张图拉进来（实测：`0%` / `5%` 曾进入度数 Top-20，占据可视化图的中心）
            self._bump("dropped_ratio_uninformative")
            return None
        if entity_type not in ENTITY_TYPES:
            self._bump("dropped_unknown_type")
            return None
        eid = self.entity_id(entity_type, clean)
        entity = self._entities.get(eid)
        if entity is None:
            entity = Entity(id=eid, name=clean, type=entity_type, confidence=confidence)
            self._entities[eid] = entity
            self._bump(f"entity_new_{entity_type}")
        else:
            entity.confidence = max(entity.confidence, confidence)
        entity.count += 1
        # 记录提及块（同一块只记一次；按置信度降序 + 出现次数截断）
        cid = str(chunk.get("chunk_id") or "")
        mine = [c for c in entity.chunks if c.get("chunk_id") == cid]
        if not mine:
            entity.chunks.append({
                "chunk_id": cid, "page": int(chunk.get("page") or 0),
                "file_name": str(chunk.get("file_name") or ""), "type": str(chunk.get("type") or "text"),
                "confidence": round(float(confidence), 4), "snippet": snippet[:120],
            })
            limit = int(self.settings.get("mentions_per_entity", 6))
            if len(entity.chunks) > limit:
                # 保留置信度最高 + 页码最小的（页码小 ⇒ 更接近「董事长致辞/重要提示」等摘要位置）
                entity.chunks.sort(key=lambda c: (-float(c.get("confidence") or 0.0), int(c.get("page") or 0)))
                del entity.chunks[limit:]
        return entity

    def _add_relation(self, head: Entity, relation: str, tail: Entity, *,
                      confidence: float, extractor: str, evidence: str,
                      chunk: Mapping[str, Any]) -> None:
        canonical = relation_canonical(relation)
        if not is_valid_relation(canonical):
            self._bump("dropped_unknown_relation")
            return
        spec = RELATION_TYPES.get(canonical, {})
        head_ok = (not spec.get("head")) or head.type in spec["head"]
        tail_ok = (not spec.get("tail")) or tail.type in spec["tail"]
        if not (head_ok and tail_ok):
            # 头尾类型不符合约束 ⇒ 丢弃并计数（schema 校验用例会核对这个计数不异常增长）
            self._bump(f"dropped_type_mismatch_{canonical}")
            return
        if head.id == tail.id:
            self._bump("dropped_self_loop")
            return
        rel = Relation(head=head.name, relation=canonical, tail=tail.name,
                       head_type=head.type, tail_type=tail.type, confidence=confidence,
                       extractor=extractor, evidence=str(evidence or "").strip(),
                       chunk_id=str(chunk.get("chunk_id") or ""), page=int(chunk.get("page") or 0),
                       file_name=str(chunk.get("file_name") or ""))
        old = self._relations.get(rel.key())
        if old is None or rel.confidence > old.confidence:
            self._relations[rel.key()] = rel
            self._bump(f"relation_{canonical}")
        else:
            self._bump("relation_duplicate_kept_stronger")

    # -- 主入口 -----------------------------------------------------------
    def extract_chunk(self, chunk: Mapping[str, Any]) -> None:
        """单块抽取：实体（8 类）→ 句内关系 → 指标取值绑定。"""
        try:
            self._extract_chunk_inner(chunk)
        except Exception as exc:  # noqa: BLE001 —— 单块失败不得中断全量构建（如实计数）
            self._bump("chunk_failed")
            if self.log is not None:
                self.log.log_event("graph.chunk_failed", level="WARNING",
                                   chunk_id=chunk.get("chunk_id"),
                                   error=f"{type(exc).__name__}: {exc}")

    def _extract_chunk_inner(self, chunk: Mapping[str, Any]) -> None:
        self._bump("chunk_scanned")
        text = str(chunk.get("content") or "")[: self.max_scan]
        if not text:
            return
        # ① 公司 / 机构
        orgs = self._match(RE_ORG, text, "company", self.conf.get("org_fullname", 0.95), chunk)
        insts = self._match(RE_INSTITUTION, text, "institution", self.conf.get("institution", 0.90), chunk)
        # 公司名里嵌的机构名（如「中信证券股份有限公司」里的「中信证券」）已由 ORG 覆盖，去重
        org_names = {o.name for o in orgs}
        insts = [i for i in insts if i.name not in org_names]
        # ② 指标（词典优先，再正则补）
        metrics = self._match_metrics(text, chunk)
        # ③ 金额 / 比例 / 日期（**不抽裸年份**：`2019年` 每页都出现，是零信息量的高频噪声）
        amounts = self._match(RE_AMOUNT, text, "amount", self.conf.get("amount", 0.90), chunk)
        ratios = self._match(RE_RATIO, text, "ratio", self.conf.get("ratio", 0.90), chunk)
        dates = self._match(RE_PERIOD, text, "date", self.conf.get("date", 0.85), chunk)
        dates += self._match(RE_DATE, text, "date", self.conf.get("date", 0.90), chunk)
        # ④ 产品/业务（词典 + 引号 + 后缀）
        products = self._match_products(text, chunk)
        # ⑤ 人物（职务+姓名，强约束）
        persons = self._match_persons(text, chunk)
        # ⑥ 关系：逐句做「触发词 + 同句实体」组合
        for sentence in _SENTENCE_SPLIT.split(text):
            sentence = sentence.strip()
            if len(sentence) < 6:
                continue
            self._relations_in_sentence(sentence, chunk)
        # ⑦ 指标取值绑定（指标 ↔ 同子句数值）
        self._bind_metric_values(text, chunk, metrics, amounts, ratios)
        # ⑧ 跨类型归属：公司 ↔ 指标 / 金额（同句共现，弱关系）
        self._bind_org_facts(text, chunk, orgs + insts, metrics, amounts, products, dates)

    # -- 各类型匹配 --------------------------------------------------------
    def _match(self, pattern: re.Pattern[str], text: str, entity_type: str, confidence: float,
               chunk: Mapping[str, Any]) -> list[Entity]:
        out: list[Entity] = []
        for m in pattern.finditer(text):
            entity = self._add_entity(m.group(0), entity_type, confidence, chunk,
                                      snippet=_snippet(text, m.start(), m.end()))
            if entity is not None:
                out.append(entity)
        return out

    def _match_metrics(self, text: str, chunk: Mapping[str, Any]) -> list[Entity]:
        out: list[Entity] = []
        seen: set[str] = set()
        for term in self.metrics:                     # 词典优先（长词优先，避免「营业收入」被切成「收入」）
            start = text.find(term)
            if start < 0:
                continue
            name = self.normalize_name(term)
            if name in seen:
                continue
            seen.add(name)
            entity = self._add_entity(term, "metric", self.conf.get("metric", 0.92), chunk,
                                      snippet=term)
            if entity is not None:
                out.append(entity)
        for m in RE_METRIC.finditer(text):            # 正则补充词典未覆盖的指标写法
            name = self.normalize_name(m.group(0))
            if name in seen or len(name) < 3:
                continue
            if any(name in term for term in seen):
                continue
            seen.add(name)
            entity = self._add_entity(name, "metric", self.conf.get("metric", 0.92) - 0.07, chunk,
                                      snippet=_snippet(text, m.start(), m.end()))
            if entity is not None:
                out.append(entity)
        return out

    def _match_products(self, text: str, chunk: Mapping[str, Any]) -> list[Entity]:
        out: list[Entity] = []
        seen: set[str] = set()
        for term in self.products:
            if len(term) >= 3 and term in text and term not in seen:
                seen.add(term)
                entity = self._add_entity(term, "product", self.conf.get("product", 0.70) + 0.15, chunk,
                                          snippet=term)
                if entity is not None:
                    out.append(entity)
        for pattern in (RE_PRODUCT_QUOTED, RE_PRODUCT_SUFFIX):
            for m in pattern.finditer(text):
                name = self.normalize_name(m.group(1) if m.re.groups else m.group(0))
                if name in seen or len(name) < 3 or len(name) > 16:
                    continue
                seen.add(name)
                entity = self._add_entity(name, "product", self.conf.get("product", 0.70), chunk,
                                          snippet=_snippet(text, m.start(), m.end()))
                if entity is not None:
                    out.append(entity)
        return out

    def _match_persons(self, text: str, chunk: Mapping[str, Any]) -> list[Entity]:
        """人名抽取（**强约束**）：职务 + 2~4 字姓名，且姓名须「百家姓开头 + 不含职务字」。

        为什么这么严（第一轮 300 块实测踩坑）：年报的「董事、监事和高级管理人员」章节是
        「职务 + 履历」罗列，宽松正则会抽出「任平安银行」「兼深圳分」「党委书记」等**非人名**，
        进而污染 ``任职`` 关系（整张图的人事边全是错的）。
        """
        out: list[Entity] = []
        seen: set[str] = set()
        for m in RE_PERSON_TITLE.finditer(text):
            name = self.normalize_name(m.group(1))
            if name in seen or len(name) < 2 or len(name) > 3:
                continue
            if name[0] not in SURNAMES or any(ch in name for ch in _PERSON_BAD_CHARS):
                self._bump("dropped_person_form")
                continue
            if name.endswith(("致辞", "会议", "报告", "声明", "意见", "任期")):
                continue
            seen.add(name)
            entity = self._add_entity(name, "person", self.conf.get("person_title", 0.88), chunk,
                                      snippet=_snippet(text, m.start(), m.end()))
            if entity is not None:
                out.append(entity)
        return out

    # -- 关系抽取 ----------------------------------------------------------
    def _lookup_person(self, raw: str) -> Entity | None:
        """按同样的**人名强约束**回查已入库的人名实体（未通过约束的返回 None）。"""
        name = self.normalize_name(raw)
        if len(name) < 2 or len(name) > 3 or name[0] not in SURNAMES:
            return None
        if any(ch in name for ch in _PERSON_BAD_CHARS):
            return None
        return self._get(name, "person")

    def _relations_in_sentence(self, sentence: str, chunk: Mapping[str, Any]) -> None:
        budget = int(self.settings.get("max_relations_per_sentence", 6))
        produced = 0
        orgs = [self._get(self.normalize_name(m.group(0)), "company") for m in RE_ORG.finditer(sentence)]
        orgs = [e for e in orgs if e is not None]
        insts = [e for e in (self._get(self.normalize_name(m.group(0)), "institution")
                             for m in RE_INSTITUTION.finditer(sentence)) if e is not None]
        orgs = orgs + [e for e in insts if e.name not in {o.name for o in orgs}]
        # 人名必须**已作为 person 实体存在**（即已通过强约束），否则不产出任职关系
        persons = [e for e in (self._lookup_person(m.group(1)) for m in RE_PERSON_TITLE.finditer(sentence))
                   if e is not None]
        metrics = [e for e in (self._get(self.normalize_name(t), "metric") for t in self.metrics
                               if t in sentence) if e is not None]
        amounts = [e for e in (self._get(self.normalize_name(m.group(0)), "amount")
                               for m in RE_AMOUNT.finditer(sentence)) if e is not None]
        ratios = [e for e in (self._get(self.normalize_name(m.group(0)), "ratio")
                              for m in RE_RATIO.finditer(sentence)) if e is not None]
        products = [e for e in (self._get(self.normalize_name(t), "product") for t in self.products
                                if t in sentence) if e is not None]
        if not orgs and not persons and not metrics:
            return
        primary_org = orgs[0] if orgs else None
        for pattern, relation in RELATION_TRIGGERS:
            if produced >= budget:
                break
            if not re.search(pattern, sentence):
                continue
            # (a) 公司 --关系--> 指标/金额/比例/产品
            if primary_org is not None:
                for tail in (metrics[:1] + amounts[:2] + ratios[:2] + products[:1]):
                    self._add_relation(primary_org, relation, tail,
                                       confidence=self.conf.get("relation_pattern", 0.86),
                                       extractor="rule_trigger", evidence=sentence, chunk=chunk)
                    produced += 1
                    if produced >= budget:
                        break
            # (b) 人物 --任职--> 公司
            if relation == "任职" and persons and primary_org is not None:
                self._add_relation(persons[0], "任职", primary_org,
                                   confidence=self.conf.get("person_title", 0.88) + 0.04,
                                   extractor="rule_title", evidence=sentence, chunk=chunk)
                produced += 1
            # (c) 公司 --母子公司--> 公司（同句第二家公司）
            if relation == "母子公司" and len(orgs) >= 2:
                self._add_relation(orgs[0], "母子公司", orgs[1],
                                   confidence=self.conf.get("relation_cooccur", 0.55) + 0.15,
                                   extractor="rule_cooccur", evidence=sentence, chunk=chunk)
                produced += 1
        # (d) 归属关系：「归属于…的X」+ 同句公司
        if primary_org is not None and re.search(r"归属于", sentence):
            for tail in (metrics[:2] or amounts[:1]):
                self._add_relation(tail, "归属于", primary_org,
                                   confidence=self.conf.get("relation_pattern", 0.86) - 0.06,
                                   extractor="rule_belong", evidence=sentence, chunk=chunk)
                produced += 1
        # (e) 同比变动：指标 ↔ 比例
        if ratios and (metrics or amounts) and re.search(r"(?:同比|比上年|较上年|增减|增长|下降|减少|提高)",
                                                         sentence):
            for head in (metrics[:2] or amounts[:1]):
                self._add_relation(head, "同比变动", ratios[0],
                                   confidence=self.conf.get("relation_cooccur", 0.55) + 0.2,
                                   extractor="rule_change", evidence=sentence, chunk=chunk)
                produced += 1
        # (f) 位于：公司/产品 ↔ 城市（城市来自 schema.CITY_TERMS 词典，不是硬编码答案）
        if re.search(r"(?:位于|设在|地处|总部)", sentence):
            cities = [c for c in self.cities if c in sentence]
            if cities:
                tail = self._add_entity(cities[0], "city", 0.78, chunk, snippet=cities[0])
                for head in ([primary_org] if primary_org else []) + products[:1]:
                    if head is None or tail is None:
                        continue
                    self._add_relation(head, "位于", tail,
                                       confidence=self.conf.get("relation_pattern", 0.86) - 0.1,
                                       extractor="rule_city", evidence=sentence, chunk=chunk)
                    produced += 1

    def _bind_metric_values(self, text: str, chunk: Mapping[str, Any],
                            metrics: Sequence[Entity], amounts: Sequence[Entity],
                            ratios: Sequence[Entity]) -> None:
        """指标取值：指标与数值出现在**同一子句**内即绑定（距离约束见 ``_CLAUSE_SPLIT``）。"""
        if not metrics or not (amounts or ratios):
            return
        clauses = _CLAUSE_SPLIT.split(text)
        if len(clauses) < 2:
            clauses = [text]
        for clause in clauses:
            if len(clause) < 4:
                continue
            clause_metrics = [e for e in metrics if e.name in clause]
            if not clause_metrics:
                continue
            clause_values = [e for e in amounts if e.name in clause] + \
                            [e for e in ratios if e.name in clause]
            for metric in clause_metrics[:2]:
                for value in clause_values[:3]:
                    self._add_relation(metric, "指标取值", value,
                                       confidence=self.conf.get("relation_cooccur", 0.55) + 0.25,
                                       extractor="rule_clause", evidence=clause, chunk=chunk)

    def _bind_org_facts(self, text: str, chunk: Mapping[str, Any], orgs: Sequence[Entity],
                        metrics: Sequence[Entity], amounts: Sequence[Entity],
                        products: Sequence[Entity], dates: Sequence[Entity]) -> None:
        """公司 ↔ 指标/产品/日期 的同句共现（弱关系，供图谱扩展时「同主体」聚合）。"""
        if not orgs:
            return
        for sentence in _SENTENCE_SPLIT.split(text):
            if len(sentence) < 8:
                continue
            sent_orgs = [e for e in orgs if e.name in sentence]
            if not sent_orgs:
                continue
            head = sent_orgs[0]
            for tail in [e for e in metrics if e.name in sentence][:2]:
                self._add_relation(head, "披露", tail,
                                   confidence=self.conf.get("relation_cooccur", 0.55),
                                   extractor="rule_cooccur", evidence=sentence, chunk=chunk)
            for tail in [e for e in products if e.name in sentence][:2]:
                self._add_relation(head, "拥有", tail,
                                   confidence=self.conf.get("relation_cooccur", 0.55),
                                   extractor="rule_cooccur", evidence=sentence, chunk=chunk)
            for tail in [e for e in dates if e.name in sentence][:1]:
                for metric in [e for e in metrics if e.name in sentence][:1]:
                    self._add_relation(metric, "指标取值", tail,
                                       confidence=self.conf.get("relation_cooccur", 0.55),
                                       extractor="rule_cooccur", evidence=sentence, chunk=chunk)

    # -- 读取 -------------------------------------------------------------
    def _get(self, name: str, entity_type: str) -> Entity | None:
        return self._entities.get(self.entity_id(entity_type, name))

    def entities(self) -> list[Entity]:
        return sorted(self._entities.values(), key=lambda e: (-e.count, e.type, e.name))

    def relations(self) -> list[Relation]:
        return sorted(self._relations.values(), key=lambda r: (-r.confidence, r.head, r.relation, r.tail))

    # -- 最低证据量剪枝 ---------------------------------------------------
    def prune_low_support(self, *, min_mentions: int = 2,
                          protect_types: Sequence[str] = ("company", "institution", "city", "person"),
                          logger: Any = None) -> dict[str, Any]:
        """剪掉「只在 1 个块里出现过」的实体（**公司/机构/地区/人物豁免**）。

        依据（全量 7430 块实测）：正则派生指标里混入大量「实现营业收入 / 现营业收入 / 年实现营业收入」
        这类**跨词残片**，它们只出现一次、对图谱扩展毫无价值，却会污染问句的实体命中
        （问句「…实现营业收入和净利润分别是多少」一次命中 5 个残片，把扩展预算吃光）。
        数据自证判据：**只出现一次的非主体实体 = 噪声**。
        """
        protected = set(protect_types)
        dropped: list[dict[str, Any]] = []
        for eid, entity in list(self._entities.items()):
            if entity.type in protected:
                continue
            if entity.count >= int(min_mentions):
                continue
            dropped.append({"id": eid, "name": entity.name, "type": entity.type, "count": entity.count})
            del self._entities[eid]
        if dropped:
            names = {d["name"] for d in dropped}
            for key in [k for k in self._relations if k[0] in names or k[2] in names]:
                del self._relations[key]
            self._bump("pruned_low_support", len(dropped))
        out = {"min_mentions": int(min_mentions), "pruned": len(dropped),
               "samples": [d["name"] for d in dropped[:15]]}
        if logger is not None:
            logger.log_event("graph.pruned_low_support", **out)
        return out

    # -- 高频噪声剪枝（**数据自证**的规则，不依赖硬编码黑名单） -------------
    def prune_high_frequency(self, *, chunks_scanned: int, max_ratio: float | None = None,
                             protect_types: Sequence[str] = ("company",),
                             logger: Any = None) -> dict[str, Any]:
        """剪掉「出现在超过 ``max_ratio`` 比例块里」的实体（连同其关系）。

        依据（全量 7325 块实测）：高频实体（如年份、栏目名、页眉词）度极大，
        在**两跳扩展**里会把半张图都拉进来 ⇒ 图谱通道召回退化成噪声，反而拉低精度。
        高频 = 信息量低，这是可复现的数据判据，比手写黑名单稳健。
        公司/机构类**豁免**（发行主体天然贯穿全文，剪掉就没有主体锚点了）。
        """
        ratio = float(max_ratio if max_ratio is not None
                      else self.settings.get("max_doc_ratio", 0.08))
        limit = max(int(chunks_scanned * ratio), 3)
        protected = set(protect_types)
        dropped: list[dict[str, Any]] = []
        for eid, entity in list(self._entities.items()):
            if entity.type in protected:
                continue
            if entity.count <= limit:
                continue
            dropped.append({"id": eid, "name": entity.name, "type": entity.type,
                            "count": entity.count, "limit": limit})
            del self._entities[eid]
        if dropped:
            names = {d["name"] for d in dropped}
            for key in [k for k in self._relations if k[0] in names or k[2] in names]:
                del self._relations[key]
            self._bump("pruned_high_frequency", len(dropped))
        out = {"chunks_scanned": int(chunks_scanned), "max_doc_ratio": ratio, "limit_chunks": limit,
               "pruned": len(dropped), "samples": dropped[:15]}
        if logger is not None:
            logger.log_event("graph.pruned_high_frequency", **out)
        return out

    def stats(self) -> dict[str, Any]:
        by_type: dict[str, int] = {}
        for entity in self._entities.values():
            by_type[entity.type] = by_type.get(entity.type, 0) + 1
        by_relation: dict[str, int] = {}
        by_extractor: dict[str, int] = {}
        for rel in self._relations.values():
            by_relation[rel.relation] = by_relation.get(rel.relation, 0) + 1
            by_extractor[rel.extractor] = by_extractor.get(rel.extractor, 0) + 1
        return {
            "entities": len(self._entities),
            "relations": len(self._relations),
            "entities_by_type": by_type,
            "relations_by_type": by_relation,
            "relations_by_extractor": by_extractor,
            "mentions": sum(len(e.chunks) for e in self._entities.values()),
            "counters": dict(sorted(self._stats.items())),
        }

    # -- LLM 补抽（抽样，独立步骤） ---------------------------------------
    def add_llm_relations(self, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """把 Ollama 补抽结果并入图谱。

        ``rows`` 每条形如 ``{"chunk_id", "page", "file_name", "head", "relation", "tail", "evidence"}``；
        头尾实体若不在图谱内则**不新建**（避免 LLM 造出不存在的实体污染图谱），只做「已存在实体」之间的补边。
        """
        added = 0
        skipped_unknown = 0
        for row in rows:
            head = self._find_any(str(row.get("head") or ""))
            tail = self._find_any(str(row.get("tail") or ""))
            if head is None or tail is None:
                skipped_unknown += 1
                continue
            chunk = {"chunk_id": row.get("chunk_id"), "page": row.get("page"),
                     "file_name": row.get("file_name")}
            self._add_relation(head, str(row.get("relation") or ""), tail,
                               confidence=self.conf.get("relation_llm", 0.75),
                               extractor="llm_qwen2.5-3b",
                               evidence=str(row.get("evidence") or ""), chunk=chunk)
            added += 1
        return {"llm_relations_merged": added, "llm_relations_skipped_unknown_entity": skipped_unknown}

    def _find_any(self, name: str) -> Entity | None:
        clean = self.normalize_name(name)
        if not clean:
            return None
        for entity in self._entities.values():
            if entity.name == clean:
                return entity
        for entity in self._entities.values():     # 子串兜底（LLM 常写简称）
            if len(clean) >= 3 and (clean in entity.name or entity.name in clean):
                return entity
        return None


def _snippet(text: str, start: int, end: int, *, radius: int = 18) -> str:
    """取命中位置附近的证据片段（用于可审计：图上每个实体都能回指原文）。"""
    lo = max(0, start - radius)
    hi = min(len(text), end + radius)
    return text[lo:hi].replace("\n", " ").strip()


_RATIO_NUM = re.compile(r"^(\d+(?:\.\d+)?)")


def _is_uninformative_ratio(name: str, *, limit: float = 10.0) -> bool:
    """比例是否「无信息量」（小数比例、拨备覆盖率这类真实指标都算有信息量）。

    判据（数据自证，阈值配置化，不写死具体数字）：
        * ``数值 ≤ limit 且为整数``（0% / 1% / 5% / 10% / 1倍）⇒ 财务表格的通用刻度，剪除；
        * ``数值 ≥ 100 且为整数``（100% / 200%…）⇒ 同样是通用「全额/整倍」刻度，剪除。
          实测依据：`100%` 在「在子公司中的权益」清单里与几十家子公司相连，
          在可视化图里形成一张 **星形噪声网**（第一版截图里占满全图），但它不含任何公司事实；
          真实指标（183.12% 拨备覆盖率 / 389.45%）**不是整数**，不受影响。
        * 其余（0.86% / 18.2% / 183.12% / 389.45% / 2.96%）⇒ 保留。
    """
    match = _RATIO_NUM.match(str(name).replace(",", ""))
    if not match:
        return False
    try:
        value = float(match.group(1))
    except ValueError:
        return False
    if abs(value - round(value)) > 1e-9:      # 非整数 ⇒ 是真实指标值，保留
        return False
    return value <= float(limit) or value >= 100.0


def iter_chunks(path: Any, *, include_table: bool = False, logger: Any = None) -> Iterable[dict[str, Any]]:
    """逐行读分块 jsonl（流式，避免一次性把 21 MB 读进内存）。"""
    import json
    from pathlib import Path

    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"分块文件不存在：{file_path}（图谱构建需要工单7 的 chunks_indexed.jsonl）")
    with file_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError as exc:
                if logger is not None:
                    logger.log_event("graph.bad_jsonl_line", level="WARNING", line=line_no,
                                     error=f"{type(exc).__name__}: {exc}")
                continue
            if not include_table and str(row.get("type")) == "table":
                continue
            yield row
