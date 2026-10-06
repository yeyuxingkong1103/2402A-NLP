# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
# 工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
# 工单01 - 基于PDF文档的问答系统
# 工单02 - 基于PDF文档的问答系统的优化
# 工单03 - PDF文档的表格解析及检索优化
"""
评估器：三轨指标，自实现（不依赖 ragas）。

【为什么不用 ragas】
ragas 走 Ollama 有三个必炸点：
  1. 它依赖结构化输出，而 Ollama 的 OpenAI 兼容端点 /v1 对
     response_format 的支持不合规（ollama#7978）；
  2. `think:false` 在 /v1 上不保证透传 —— 实测 /v1 返回的是空串，
     qwen3 的思考链会污染 judge 的 JSON；
  3. 依赖地狱：ragas 会拉进 langchain 全家桶，与 pymilvus 的
     `setuptools<81` 约束冲突风险高。
自己实现只要一个强约束 JSON 的调用 + 重试，可控得多。

【三轨设计】
轨1 规则化数值命中 —— **最可信**。5 题的答案是数字/人名，用正则硬匹配，
    没有 LLM 打分的主观性。演示时「命中 4/4 个年度数字」比
    「LLM 打了 0.87 分」有说服力得多，而且可复现、可争辩。
轨2 LLM judge —— 无参考答案也能算：Context Relevance / Faithfulness /
    Answer Relevance。用于衡量检索质量和幻觉程度。
轨3 对比人工参考答案 —— Context Recall / Answer Correctness。
    10 条参考答案写好了，工单02 的「90% 准确率」才有分子分母。

【数字匹配的归一化】
模型可能把 `5,520.00` 写成 `5520` 或 `5,520`，所以匹配前把候选答案与
关键词都去掉逗号、空格再比对 —— 否则会把正确回答误判为未命中。
"""

from __future__ import annotations

import json
import re
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Sequence

from app.config import PROJECT_ROOT, settings
from app.core.doc_profiles import get_doc_profile
from app.core.generator import Generator, warmup
from app.core.profiles import RetrievalProfile, get_profile
from app.core.retriever import Retriever, build_context
from app.core.vectorstore import SearchHit

QUESTIONS_FILE = PROJECT_ROOT / "eval" / "questions.json"

# ----------------------------------------------------------------------
# 参考答案的「脱敏原文」提示：招股书里大量使用「某」代替真实名称
# （如《某视频技术规范1.0》），评估时必须原样接受，不能因"名字不完整"扣分。
DESENSITIZED_NOTE = (
    "注意：原文中大量使用「某」字对敏感信息脱敏（例如《某视频技术规范1.0》、"
    "「某情报、指挥、控制与通信网络一体化工程」）。这些是原文原貌，"
    "回答中出现带「某」的名称属于正确，不得因此判为错误或不完整。"
)


@dataclass
class EvalItem:
    id: int
    question: str
    category: str = ""
    reference_answer: str = ""
    evidence_page: str = ""
    rag_answer: str = ""
    no_rag_answer: str = ""
    citations: list[str] = field(default_factory=list)
    rule_hit: bool | None = None
    rule_detail: str = ""
    no_rag_rule_hit: bool | None = None
    context_relevance: float | None = None
    faithfulness: float | None = None
    answer_relevance: float | None = None
    context_recall: float | None = None
    answer_correctness: float | None = None
    ttft_ms: int = 0
    total_ms: int = 0
    error: str = ""
    # ---------- 工单02 新增：检索侧指标 ----------
    # 工单02 的产出物要「对比优化前后**检索精确度**的变化」，
    # 而 rule_hit / LLM judge 都是端到端的（掺了生成的成分）。
    # 这一组是纯检索侧，能把"检索没召回"和"召回了但模型没答对"分开。
    profile: str = ""
    retrieved_pages: list[str] = field(default_factory=list)
    evidence_pages: list[str] = field(default_factory=list)
    page_precision: float | None = None   # 召回页里有多少落在证据页上
    page_recall: float | None = None      # 证据页里有多少被召回
    # ★锚点指标：召回上下文里覆盖了多少 rule_keywords。
    # 它 = 1.0 而 rule_hit = False，就说明是**生成侧**没答出来，不是检索问题。
    context_keyword_coverage: float | None = None
    context_chars: int = 0                # 送进 LLM 的上下文字数（TTFT 的直接驱动量）
    n_merged: int = 0                     # 被同事实聚合合并掉的候选数
    n_added_neighbors: int = 0            # 补进来的邻块数
    # ---------- 工单03 新增：实体路由留痕 ----------
    # 报告里要能看出每道题被路由到了哪份文档、有没有回退全库 ——
    # 否则「页级指标变差」时无法判断是路由串了还是检索本身退化了。
    routed_doc: str = ""                  # 空 = 没过滤（没点名 / 跨文档）
    route_matched: str = ""
    route_fallback: bool = False
    # ---------- 工单06 新增：检索模式与「答案要点召回率」 ----------
    # 【为什么要单独一个召回口径】工单06 要求「召回率 ≥95%」，而原先的三个
    # 检索指标都有各自的问题：
    #   · page_recall 是**页级**的，证据页标注宽泛（如「1-1-21；1-1-51 亦载明」），
    #     天然偏低且推不动（工单02 扫了 72 格参数，它恒为 0.567）；
    #   · context_keyword_coverage 的口径**把邻块排除了**，但邻块**确实送进了 prompt** ——
    #     实测 id=2 的答案要点全在邻块里，于是被系统性低估（0.6 而实际 1.0）。
    # 所以新增一个口径：**召回上下文（含邻块）覆盖了多少条要点**。
    retrieval_mode: str = ""
    fusion: str = ""
    reranker: str = ""
    n_from_dense: int = 0
    n_from_keyword: int = 0
    recall_keywords: list[str] = field(default_factory=list)
    recall_coverage: float | None = None   # 含邻块的上下文对要点的覆盖率（0~1）
    recall_full: bool | None = None        # 是否**全部**要点都被召回（更严的伴随口径）


# ----------------------------------------------------------------------
# 轨1：规则化数值命中
# ----------------------------------------------------------------------
def _norm_digits(s: str) -> str:
    """归一化数字写法差异：去逗号、去空格、全角转半角。"""
    s = s.replace("，", ",").replace("．", ".").replace("％", "%")
    s = s.replace(" ", "").replace(",", "")
    return s


_NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def _numbers_in(s: str) -> set[float]:
    """
    抽取文本中所有数字的数值。

    【为什么需要】同一事实有多种合法写法，字符串匹配太脆：
      真值 `15,000.00万元`，模型可能答 `15000万元`、`1.5亿元` 的不同形式里
      前两种应算命中。按**数值**比较即可统一。
      注意 `5,520.00` 与 `5,520万元` 数值都是 5520.0，都能对上。
    """
    out: set[float] = set()
    for m in _NUM_RE.finditer(s.replace(",", "").replace("，", "")):
        try:
            out.add(round(float(m.group()), 2))
        except ValueError:
            continue
    return out


def _keyword_is_numeric(kw: str) -> bool:
    """关键词是否本质上是数值（去掉数字、百分号、千分位、小数点后为空）。"""
    stripped = re.sub(r"[\d,.\s%％]", "", kw)
    return stripped == "" and bool(re.search(r"\d", kw))


def _keyword_variants(kw: str) -> list[str]:
    """关键词的「同义写法」：用 `|` 分隔，命中任一即算命中这条关键词。

    【工单05 补】id=5 的判据原本写死「4个部门」。实测同一系统同一道题，
    模型有时答「销售部由**4**个部门构成」、有时答「由…四个部门构成」——
    后者**语义完全正确**，却因为一个字（4 / 四）被判成未命中。
    这是**判据写得太死**，不是系统答错了（与工单04 文档 §5.1 记的是同一类问题）。

    【为什么用 `|` 而不是去改数值归一化】在 `_norm_digits` 里做「中文数字 → 阿拉伯数字」
    会波及**全部**题目的匹配（例如「一体化工程」里的「一」可能被卷进来），
    改动面与收益不成比例。用显式分隔符把"这两种写法都算对"写在数据里，
    影响面精确地限制在这一条关键词上，且验收时一眼能看懂。
    """
    return [v for v in (p.strip() for p in kw.split("|")) if v]


def keyword_hits(text: str, keywords: Sequence[str]) -> list[str]:
    """返回文本里命中了哪些关键词（数字按**数值**比较，不按字面）。

    【为什么要抽成独立函数】工单02 的 CKC 指标（上下文关键词覆盖）必须用
    与 rule_hit **完全同一套**匹配规则。实测踩过：CKC 最初写成字面子串匹配，
    于是 `15,000万` / `15000` 这类同一数值的不同写法被误判成"缺失"，
    id=207/543 的 CKC 被压到 0.33（实际规则的判法是 3/3、2/3）。
    两套口径不一致会让整个对比表整体偏低，且**偏向于惩罚数字题**。
    """
    norm = _norm_digits(text)
    text_nums = _numbers_in(text)
    hits: list[str] = []
    for k in keywords:
        for v in _keyword_variants(k):
            # ① 先做字符串包含（精确、快）
            if _norm_digits(v) in norm:
                hits.append(k)
                break
            # ② 数值型关键词退化为数值比较（容忍 15,000.00 / 15000 / 5,520万 写法差异）
            if _keyword_is_numeric(v):
                want = _numbers_in(v)
                if want and want <= text_nums:
                    hits.append(k)
                    break
    return hits


def rule_hit(answer: str, keywords: Sequence[str], rule_type: str = "any"
              ) -> tuple[bool, str]:
    """
    规则化命中判定。
    rule_type='all' → 全部关键词都要出现（多值答案，如四个年度的收入）
    rule_type='any' → 任一出现即可（同一事实存在多种合法表述）
    """
    if not keywords:
        return False, "无关键词"

    hits = keyword_hits(answer, keywords)
    missed = [k for k in keywords if k not in hits]
    ok = (len(hits) == len(keywords)) if rule_type == "all" else bool(hits)
    detail = f"命中 {len(hits)}/{len(keywords)}：{hits}"
    if missed:
        detail += f"；缺失：{missed}"
    return ok, detail


# ----------------------------------------------------------------------
# 轨2/轨3：LLM judge
# ----------------------------------------------------------------------
_JUDGE_SYS = (
    "你是一个严格的评估员。只输出 JSON，不要输出任何解释文字、不要用 markdown 代码块。"
)

_JUDGE_SCHEMA_HINT = (
    '你必须只输出如下 JSON 结构：\n'
    '{"context_relevance": <0到1的小数>, "faithfulness": <0到1的小数>, '
    '"answer_relevance": <0到1的小数>, "context_recall": <0到1的小数>, '
    '"answer_correctness": <0到1的小数>, "reason": "<一句话中文理由>"}'
)


def _parse_json(text: str) -> dict[str, Any] | None:
    """从模型输出里抠出 JSON —— 容忍它偶尔加代码块围栏。"""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?|```$", "", text).strip()
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return None


def _clamp01(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, f))


class LLMJudge:
    """单指标单次调用 + 重试，避免一次失败全盘皆输。"""

    def __init__(self, generator: Generator | None = None, retries: int = 2) -> None:
        self.gen = generator or Generator()
        self.retries = retries

    async def judge(self, question: str, context: str, answer: str,
                    reference: str) -> dict[str, Any]:
        prompt = f"""请评估下面这次检索问答的质量，按 0~1 打分（1 最好）。

{DESENSITIZED_NOTE}

【用户问题】
{question}

【检索到的资料片段】
{context[:2500] if context else "（无检索资料）"}

【系统回答】
{answer}

【人工参考答案】
{reference}

评分项：
- context_relevance：检索到的资料与问题是否相关（无资料则为 0）
- faithfulness：系统回答是否完全有资料支撑、无编造（无资料且回答"未提及"应给 1）
- answer_relevance：回答是否切题、是否直接回应了问题
- context_recall：资料是否覆盖了参考答案所需的全部要点
- answer_correctness：回答与参考答案的事实一致程度

{_JUDGE_SCHEMA_HINT}"""

        msgs = [{"role": "system", "content": _JUDGE_SYS},
                {"role": "user", "content": prompt}]

        last_err = ""
        for _ in range(self.retries + 1):
            try:
                raw = await self.gen.client.chat(
                    msgs, fmt="json", temperature=0.0, max_tokens=300
                )
            except Exception as e:  # noqa: BLE001
                last_err = str(e)
                continue
            obj = _parse_json(raw)
            if obj is None:
                last_err = f"JSON 解析失败：{raw[:120]!r}"
                continue
            return {
                "context_relevance": _clamp01(obj.get("context_relevance")),
                "faithfulness": _clamp01(obj.get("faithfulness")),
                "answer_relevance": _clamp01(obj.get("answer_relevance")),
                "context_recall": _clamp01(obj.get("context_recall")),
                "answer_correctness": _clamp01(obj.get("answer_correctness")),
                "reason": str(obj.get("reason", ""))[:200],
            }
        return {"error": last_err}


# ----------------------------------------------------------------------
# 工单02：检索侧指标（与生成解耦）
# ----------------------------------------------------------------------
_PAGE_RE = re.compile(r"1-1-\d+")


def page_re_for(hits: Sequence[SearchHit], routed_doc: str = "") -> re.Pattern:
    """证据页字符串该按**哪一份文档**的页码格式来抽。

    【工单03 为什么必须按文档走】招股书1 的页脚印 `1-1-N`，招股书2 印的是裸数字
    `N`。统一拿 `1-1-\\d+` 去抽书2 的 evidence_page（形如 `"21"`）**一条都抽不到** ——
    gold 为空，page_recall 变成 None，页级指标整列静默失效。

    取法：问题路由到哪份文档就按哪份；没路由（问题里没点公司名）时退回命中块
    所属的文档；都不确定就用书1 的格式（保持工单01/02 的既有行为）。
    """
    names = {h.doc_name for h in hits if h.doc_name}
    name = routed_doc or (names.pop() if len(names) == 1 else "")
    prof = get_doc_profile(name)
    return (prof.evidence_re or _PAGE_RE) if prof else _PAGE_RE


def retrieval_metrics(hits: Sequence[SearchHit], keywords: Sequence[str],
                      evidence_page: str,
                      page_re: re.Pattern | None = None,
                      recall_keywords: Sequence[str] | None = None) -> dict[str, Any]:
    """只测量检索本身，不看生成。三个指标 + 明细。

    【为什么必须有这一组】工单02 的产出物要「对比优化前后检索精确度的变化」，
    但 rule_hit 和 LLM judge 都是端到端的 —— 它们把"检索没召回"和
    "召回了、模型却没答对" 混成一个数字。实测就撞上了这种情况：
    id=95/795 的缺失关键词其实**都在召回上下文里**（CKC=3/3），
    是生成侧漏掉的。没有这组指标，就会把生成问题误判成检索问题。

    【邻块为什么不计入】`is_neighbor=True` 的块是邻块扩展补进来的上下文，
    本来就不指望它落在证据页上。把它算进精确率的分母，等于让我们新加的机制
    机械地压低自己的成绩 —— 统计口径必须和新机制的设计意图一致。
    """
    got = [h.page_label for h in hits if not h.is_neighbor and h.page_label]
    gold = sorted(set((page_re or _PAGE_RE).findall(evidence_page or "")))
    inter = set(got) & set(gold)

    blob = "".join(h.content for h in hits if not h.is_neighbor)
    kw = list(keywords or [])
    # 必须复用 rule_hit 的同一套匹配（含数字的数值比较），否则数字写法差异
    # 会被误判成"缺失"，把 CKC 整体压低且系统性偏向惩罚数字题。见 keyword_hits。
    kw_hit = keyword_hits(blob, kw)

    # 【召回口径：含邻块】邻块**确实进了 prompt**，所以"重要信息有没有被召回"
    # 必须把它算进去。上面三个指标继续排除邻块（那是精度口径，排除是对的），
    # 两套口径**各有各的用途**，不混用。
    rkw = list(recall_keywords if recall_keywords is not None else keywords or [])
    blob_all = "".join(h.content for h in hits)          # ← 不排除邻块
    r_hit = keyword_hits(blob_all, rkw)
    coverage = (len(r_hit) / len(rkw)) if rkw else None

    return {
        "retrieved_pages": sorted(set(got)),
        "evidence_pages": gold,
        "page_precision": (round(len(inter) / len(got), 4) if got else None),
        "page_recall": (round(len(inter) / len(gold), 4) if gold else None),
        "context_keyword_coverage": (round(len(kw_hit) / len(kw), 4) if kw else None),
        "recall_keywords": rkw,
        "recall_coverage": (round(coverage, 4) if coverage is not None else None),
        "recall_full": (coverage >= 1.0) if coverage is not None else None,
    }


# ----------------------------------------------------------------------
class Evaluator:
    def __init__(self, profile: RetrievalProfile | None = None,
                 *, collection: str | None = None,
                 embed_model: str | None = None) -> None:
        # 剖面决定评测用哪套检索策略。三个剖面各跑一遍 = 工单02 的对比表。
        self.profile = profile or get_profile()
        # 【工单06】可指向另一个 collection / 嵌入模型 —— 多模型对比用。
        if collection or embed_model:
            from app.core.embedder import Embedder
            from app.core.vectorstore import VectorStore
            self.retriever = Retriever(
                store=VectorStore(collection=collection) if collection else None,
                embedder=Embedder(model=embed_model) if embed_model else None,
                profile=self.profile)
        else:
            self.retriever = Retriever(profile=self.profile)
        self.collection = collection or settings.milvus_collection
        self.embed_model = embed_model or settings.embed_model
        self.generator = Generator()
        self.judge = LLMJudge(self.generator)

    # ------------------------------------------------------------------
    @staticmethod
    def load_questions(path=None) -> list[dict[str, Any]]:
        p = path or QUESTIONS_FILE
        data = json.loads(open(p, encoding="utf-8").read())
        return data["questions"]

    # ------------------------------------------------------------------
    async def evaluate_one(self, q: dict[str, Any], *,
                           with_no_rag: bool = True,
                           with_judge: bool = True) -> EvalItem:
        item = EvalItem(
            id=q["id"], question=q["question"],
            category=q.get("category", ""),
            reference_answer=q.get("reference_answer", ""),
            evidence_page=q.get("evidence_page", ""),
            profile=self.profile.name,
        )
        t0 = time.perf_counter()
        try:
            # ---------- RAG ----------
            rres = await self.retriever.retrieve(item.question)
            hits: list[SearchHit] = rres.hits
            # 上下文宽度取自剖面（baseline 会是 300，这会静默截断答案 ——
            # 那正是它作为"优化前地板"要展示的行为）
            ctx = build_context(hits, query=item.question, profile=self.profile)
            # 邻块不算引用：它只是上下文补充，不该出现在"答案依据的页码"里
            item.citations = [h.page_label for h in hits if not h.is_neighbor]
            item.context_chars = len(ctx)
            item.n_merged = rres.n_merged
            item.n_added_neighbors = rres.n_added_neighbors
            # 工单03：路由留痕（报告里要能看出这题用的哪份文档的检索范围）
            item.routed_doc = rres.routed_doc
            item.route_matched = rres.route_matched
            item.route_fallback = rres.route_fallback
            item.retrieval_mode = rres.retrieval_mode
            item.fusion = rres.fusion
            item.reranker = rres.reranker
            item.n_from_dense = rres.n_from_dense
            item.n_from_keyword = rres.n_from_keyword

            # ---------- 工单02：检索侧指标（与生成解耦）----------
            rm = retrieval_metrics(hits, q.get("rule_keywords", []),
                                   item.evidence_page,
                                   page_re=page_re_for(hits, rres.routed_doc),
                                   recall_keywords=q.get("recall_keywords"))
            item.retrieved_pages = rm["retrieved_pages"]
            item.evidence_pages = rm["evidence_pages"]
            item.page_precision = rm["page_precision"]
            item.page_recall = rm["page_recall"]
            item.context_keyword_coverage = rm["context_keyword_coverage"]
            item.recall_keywords = rm["recall_keywords"]
            item.recall_coverage = rm["recall_coverage"]
            item.recall_full = rm["recall_full"]

            t_gen = time.perf_counter()
            # doc 决定 system prompt 口径：不传的话 prompt 会点名兴图新科，
            # 力源的题会被模型判成"问题与文档不符"而拒答（见 generator.system_prompt_for）
            item.rag_answer = await self.generator.generate(
                item.question, hits, context=ctx,
                doc=get_doc_profile(rres.routed_doc)
            )
            item.ttft_ms = int((time.perf_counter() - t_gen) * 1000)  # 非流式，近似
            item.total_ms = int((time.perf_counter() - t0) * 1000)

            # ---------- 轨1 ----------
            ok, detail = rule_hit(item.rag_answer, q.get("rule_keywords", []),
                                  q.get("rule_type", "any"))
            item.rule_hit, item.rule_detail = ok, detail

            # ---------- 轨1 对照：纯 LLM ----------
            if with_no_rag:
                item.no_rag_answer = await self.generator.baseline_no_rag(item.question)
                ok2, _ = rule_hit(item.no_rag_answer, q.get("rule_keywords", []),
                                  q.get("rule_type", "any"))
                item.no_rag_rule_hit = ok2

            # ---------- 轨2/轨3 ----------
            if with_judge:
                j = await self.judge.judge(item.question, ctx, item.rag_answer,
                                           item.reference_answer)
                if "error" in j:
                    item.error = j["error"]
                else:
                    item.context_relevance = j["context_relevance"]
                    item.faithfulness = j["faithfulness"]
                    item.answer_relevance = j["answer_relevance"]
                    item.context_recall = j["context_recall"]
                    item.answer_correctness = j["answer_correctness"]

        except Exception as e:  # noqa: BLE001
            item.error = f"{type(e).__name__}: {e}"
        return item

    # ------------------------------------------------------------------
    async def evaluate_all(self, *, limit: int | None = None,
                           with_no_rag: bool = True,
                           with_judge: bool = True,
                           progress=None) -> dict[str, Any]:
        # 【必须预热，否则 TTFT 取决于运气】
        # 实测踩过：不预热时第 1 题的耗时是 7229ms（qwen3 冷启动被算进去），
        # 后 9 题都在 1.1–2.0s —— 整组 TTFT P95 直接被这一题从 2047ms 拉到 7229ms，
        # 而这只反映"评测前模型是否恰好热着"，不反映系统性能。
        # 应用侧本来就有 lifespan warmup（generator.warmup），评测脚本绕过应用，
        # 所以这里补上，让两边口径一致 —— 不是"把数字做漂亮"，是不让口径失真。
        await warmup()

        qs = self.load_questions()
        if limit:
            qs = qs[:limit]

        items: list[EvalItem] = []
        for i, q in enumerate(qs, 1):
            items.append(await self.evaluate_one(
                q, with_no_rag=with_no_rag, with_judge=with_judge))
            if progress:
                progress(i, len(qs), items[-1])

        return self.build_report(items)

    # ------------------------------------------------------------------
    @staticmethod
    def _avg(vals: Sequence[float | None]) -> float:
        xs = [v for v in vals if v is not None]
        return round(statistics.mean(xs), 4) if xs else 0.0

    @staticmethod
    def _pct(vals: Sequence[int], p: float) -> int:
        xs = sorted(v for v in vals if v)
        if not xs:
            return 0
        idx = min(len(xs) - 1, int(round((p / 100) * (len(xs) - 1))))
        return xs[idx]

    def build_report(self, items: Sequence[EvalItem]) -> dict[str, Any]:
        n = len(items)
        rag_hits = sum(1 for it in items if it.rule_hit)
        no_rag_hits = sum(1 for it in items if it.no_rag_rule_hit)

        summary = {
            "n": n,
            "profile": self.profile.name,
            # 显式披露：本次评测前做了模型预热，所以 TTFT **不含冷启动**。
            # 口径必须写进文档，否则"3 秒达标"会被质疑（冷启动单次约 7s）。
            "warmup": True,
            "rule_hit_rate": round(rag_hits / n, 4) if n else 0.0,
            "rag_rule_hits": rag_hits,
            "no_rag_rule_hits": no_rag_hits,
            # ---------- 工单02：检索侧汇总 ----------
            "avg_page_precision": self._avg([it.page_precision for it in items]),
            "avg_page_recall": self._avg([it.page_recall for it in items]),
            "avg_context_keyword_coverage": self._avg(
                [it.context_keyword_coverage for it in items]),
            # ---------- 工单06：召回率（工单点名 ≥95%）----------
            # 【口径】召回上下文**含邻块** —— 邻块确实送进了 prompt，排除它才是失真
            # （实测 id=2 的答案要点全在邻块里）。
            "avg_recall_coverage": self._avg([it.recall_coverage for it in items]),
            "recall_full_count": sum(1 for it in items if it.recall_full),
            "recall_full_rate": round(
                sum(1 for it in items if it.recall_full) / n, 4) if n else 0.0,
            "retrieval_mode": (items[0].retrieval_mode if items else ""),
            "fusion": (items[0].fusion if items else ""),
            "reranker": (items[0].reranker if items else ""),
            "avg_context_chars": int(self._avg([it.context_chars for it in items])),
            "n_merged_total": sum(it.n_merged for it in items),
            "n_added_neighbors_total": sum(it.n_added_neighbors for it in items),
            "avg_context_relevance": self._avg([it.context_relevance for it in items]),
            "avg_faithfulness": self._avg([it.faithfulness for it in items]),
            "avg_answer_relevance": self._avg([it.answer_relevance for it in items]),
            "avg_context_recall": self._avg([it.context_recall for it in items]),
            "avg_answer_correctness": self._avg([it.answer_correctness for it in items]),
            "ttft_p50_ms": self._pct([it.ttft_ms for it in items], 50),
            "ttft_p95_ms": self._pct([it.ttft_ms for it in items], 95),
            # 【为什么必须有这一行】n=10 时 P95 就是 10 个样本里的**最大值**
            # （`_pct` 取 round(0.95*9)=9 号）。单个离群点就能把它拉爆 ——
            # 实测同配置同题目跑两次得到 3060ms 与 2038ms，差 1 秒以上。
            # 所以「≤3 秒」这个验收结论不能只看 P95，必须同时给出
            # P50（稳定）与达标比例（不受离群点绑架）。
            "ttft_under_3s": sum(1 for it in items if 0 < it.ttft_ms < 3000),
            "ttft_under_3s_rate": round(
                sum(1 for it in items if 0 < it.ttft_ms < 3000) / n, 4) if n else 0.0,
            "ttft_max_ms": max((it.ttft_ms for it in items), default=0),
            "total_p50_ms": self._pct([it.total_ms for it in items], 50),
            "total_p95_ms": self._pct([it.total_ms for it in items], 95),
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        return {
            "summary": summary,
            "items": [it.__dict__ for it in items],
        }

    # ------------------------------------------------------------------
    @staticmethod
    def save_report(report: dict[str, Any], *,
                    tag: str | None = None) -> dict[str, str]:
        """同时落 JSON 与 CSV —— 前端表格用 JSON，交作业用 CSV。

        `tag` 非空时文件名带上它（`eval-<stamp>-<tag>.json`）且**不更新
        eval-latest.json**。工单02 的三剖面对比就是靠它：三个剖面各存一份，
        谁都不覆盖谁，`eval-latest.json` 保持"最近一次单剖面运行"的语义。
        """
        import csv
        out = settings.data_path / "eval"
        out.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        suffix = f"-{tag}" if tag else ""

        jp = out / f"eval-{stamp}{suffix}.json"
        jp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        if not tag:
            (out / "eval-latest.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        cp = out / f"eval-{stamp}{suffix}.csv"
        cols = ["id", "profile", "question", "category", "rule_hit", "no_rag_rule_hit",
                # 工单02 检索侧指标
                "page_precision", "page_recall", "context_keyword_coverage",
                "context_chars", "n_merged", "n_added_neighbors", "retrieved_pages",
                "context_relevance", "faithfulness", "answer_relevance",
                "context_recall", "answer_correctness", "ttft_ms", "total_ms",
                "citations", "rag_answer", "no_rag_answer", "rule_detail"]
        with cp.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for it in report["items"]:
                row = dict(it)
                row["citations"] = " / ".join(it.get("citations") or [])
                row["retrieved_pages"] = " / ".join(it.get("retrieved_pages") or [])
                w.writerow(row)
        return {"json": str(jp), "csv": str(cp)}
