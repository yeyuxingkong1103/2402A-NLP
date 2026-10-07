# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
# 工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
# 工单编号：人工智能NLP-RAG-混合检索任务
# 工单01 - 基于PDF文档的问答系统
# 工单02 - 基于PDF文档的问答系统的优化
# 工单03 - PDF文档的表格解析及检索优化
# 工单06 - 混合检索（检索模式 / 融合 / 重排器）
"""
检索器：实体路由 → Query理解 → 双路召回 → 同事实聚合 → 词法重排 → 冗余过滤 → top-k → 邻块扩展。

【工单03 加了什么】两份 PDF 进库后有两处必须改：

1. **实体路由**（`app/core/router.py` + 本模块 retrieve 的第 0 步）
   两份招股书章节结构雷同，靠排序区分不了谁是谁；按问题里的公司专名
   硬过滤 `doc_name`，过滤后零召回才回退全库。

2. **邻块扩展按 doc_id 分组**（`expand_with_neighbors`）
   原先只取 `hits[0].doc_id`，top-k 一旦跨文档，就会拿第一份文档的 doc_id
   去查邻块，把另一份文档的块静默拉进上下文。

【工单02 加了什么】在工单01 的链路里插了两个新环节，各自独立可开关，
由 `app/core/profiles.py` 的剖面控制：

1. **同事实聚合 + 权威页优先**（`aggregate_facts` / `authority_bias`）
   招股书同一事实会在十几页重复披露，且各页表述互相矛盾。实测 id=95
   （参与制定了哪个技术标准）召回的 top-3 就是同一句话的三种说法：
       316  1-1-158  全军第一个…（即2019年制订的《某视频指挥系统技术规范（1.0版）》）
       355  1-1-179  国防用户第一个…（即《某视频技术规范1.0》）
       358  1-1-181  全军第一个…（即《某视频技术规范1.0》）
   "全军"与"国防用户"打架、括号里的标准名打架，模型于是把两个名字用"和"连起来输出。
   **现有的 select_diverse() 抓不到它们** —— 实测整块 3-gram Jaccard 只有
   316~355=0.055、316~358=0.071：500 字的块之间只共享约 40 字的同一事实句，
   整块相似度被稀释得毫无区分度。所以相似度必须下沉到**句子粒度**。

2. **邻块扩展**（`expand_neighbors`，small-to-big 的近似）
   命中块之后，按 (doc_id, chunk_index±1) 取同页邻块补进上下文，救回被切在
   相邻块里的答案。必须带"互补性过滤"：邻块得带来原块没有的查询词才收，
   否则白涨 prompt —— 而 TTFT 只剩 ~950ms 余量（P95 2047ms / 3000ms 线）。

【为什么不填 parent_id 做真正的父子块】见 docs/工单02-优化方案.md：
想要的"父块"就是所在章节，而 `section_path` 已经编码了它；真要填得全量重入库，
而嵌入路径没有重试，演示前冒不起这个险。用 chunk_index±1 + section_path 近似。

【Query 理解：为什么要"抽象"掉公司全称 —— 这是实测出来的关键一步】
工单01 的 10 道题全都写成「武汉兴图新科电子股份有限公司…」，
但招股书正文**从不自称全称**，一律用「公司」「发行人」「兴图新科」。
于是问题向量被这 16 个字主导，与真正含答案的片段（讲的是经营模式、
募集资金）语义偏离。

实测（每题取 top-30，看首个含全部答案的片段排名）：

    题号     原问句排名   去掉公司全称排名
    260        >20           4
    33           9           1
    207        >20           3
    543          1           3
    957          1           6

去掉全称后 recall@5 从 7/10 升到 9/10。但单独用抽象版会丢掉 957
（原问句里"武汉兴图新科电子股份有限公司在哪个领域"整体语义更贴）。
**两条路互补，所以都要走**，并集后用词法重排融合。

【词法重排】稠密向量对"哪些/多少"这类疑问词不敏感。把问题里
长度≥2 的实词（jieba 分词，去停用词）与候选片段求词集合召回率，
与归一化后的余弦分数线性加权。实测这一步把 957 从第 6 提到第 1，
且对权重不敏感（0.4~0.8 都是 10/10），因此它是稳健的、不是过拟合。

最终实测：**recall@3 = 10/10，recall@5 = 10/10**。

【冗余过滤保留在检索时，不在入库时】见下方 select_diverse 的注释。
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field

import jieba

from app.config import settings
from app.core.chunker import is_atomic_chunk
from app.core.doc_profiles import all_entity_names
from app.core.embedder import Embedder
from app.core.fusion import fuse
from app.core.profiles import RetrievalProfile, get_profile
from app.core.rerank import RerankInfo, rerank
from app.core.text_analysis import abstract_query, entity_names, query_tokens
from app.core.router import route, route_multi
from app.core.vectorstore import SearchHit, VectorStore

# 查询时判定「两条召回内容过于雷同」的 Jaccard 阈值（整块 3-gram）。
# 实测同话题但内容不同的 chunk 重合度 0.36–0.54，取 0.85 有充足安全边界。
# 注意这个阈值**只抓字面重复**；招股书的"同事实不同表述"（Jaccard 仅 0.055）
# 它抓不到，那是 cluster_facts() 的活 —— 两者是组合关系，不是替代关系。
DUP_JACCARD = 0.85



@dataclass
class RetrievalResult:
    query: str
    hits: list[SearchHit]
    n_candidates: int = 0
    n_filtered: int = 0
    seconds: float = 0.0
    query_views: list[str] = field(default_factory=list)
    # ---------- 工单02 新增 ----------
    profile: str = "delivered"
    n_merged: int = 0            # 被同事实聚合塌掉几个候选（≠ n_filtered）
    n_added_neighbors: int = 0   # 补进来几个邻块
    context_chars: int = 0       # 送进 LLM 的上下文字数（TTFT 的直接驱动量）
    stage_ms: dict[str, float] = field(default_factory=dict)
    # ---------- 工单03 新增：实体路由 ----------
    routed_doc: str = ""         # 命中的文档名；空 = 全库检索
    route_matched: str = ""      # 命中的公司专名（排查用）
    route_fallback: bool = False  # 路由后召回为空、已回退全库
    # ---------- 工单06 新增：检索模式与两路留痕 ----------
    retrieval_mode: str = "vector"     # vector | fulltext | hybrid
    fusion: str = ""                   # rrf / weighted / milvus-native（仅 hybrid 有值）
    reranker: str = ""                 # 本轮用的是哪个重排器
    n_from_dense: int = 0              # 向量路召回条数
    n_from_keyword: int = 0            # 关键词路召回条数
    n_shared: int = 0                  # 两路都召回到的块数
    rerank_ms: int = 0
    rerank_fallback: bool = False      # 重排器是否因异常回退
    feedback_n: int = 0                # 反馈重排实际用上的反馈条数（0 = 冷启动）

    @property
    def context_text(self) -> str:
        return "\n\n".join(h.content for h in self.hits)

    @property
    def evidence_hits(self) -> list[SearchHit]:
        """只算真正的检索结果，不含邻块补充。

        精确率/召回率的分母必须用它 —— 否则邻块扩展这个新机制会机械地
        把自己的精确率压低（补进来的邻块本来就不指望落在证据页上）。
        """
        return [h for h in self.hits if not h.is_neighbor]


# ----------------------------------------------------------------------
# Query 理解
# ----------------------------------------------------------------------
# 【工单06 挪走】`_entity_names` / `abstract_query` 已移到
# `app/core/text_analysis.py`（重排器也要用同一套抽象，否则覆盖率会被
# 公司全称稀释 —— 实测差异大到能把页召回从 60.4% 打到 39.6%）。
# 这里保留旧名字作别名，既有调用点与测试按 `retriever.X` 引用，行为不变。
_entity_names = entity_names


def query_views(question: str) -> list[str]:
    """返回去重后的查询视角列表（原问句 + 抽象版）。"""
    views = [question.strip()]
    ab = abstract_query(question)
    if ab and ab != views[0]:
        views.append(ab)
    return views


# 【工单06 挪走】原实现 `_tokens` 已移到 `app/core/text_analysis.query_tokens`
# （重排器也要用它，放在 retriever 里会形成循环导入）。这里保留这个名字作为
# 别名 —— 既有测试与调用点按 `retriever._tokens` 引用，行为逐字节不变。
_tokens = query_tokens


def _rank_legacy(cands: list[SearchHit], qt: set[str],
                 p: RetrievalProfile) -> list[SearchHit]:
    """工单02 的排序：`lexical_rerank` 开则做 dense+词法线性融合，否则纯按 dense 分。

    【为什么单独抽成一个函数】工单06 加了可插拔重排器之后，这条老路必须能被
    **原样**调用、且只被 vector 模式调用 —— 抽出来是为了让"老路有没有被改动"
    变成一眼可查的事实（对照 `git diff`），而不是散在分支里的几行算术。

    【它为什么不能给混合模式用】`(score + 1) / 2` 把 COSINE ∈ [-1,1] 压到 [0,1]；
    BM25 无上界、RRF 只有 1/61 量级 —— 套进去会被压成近乎常数，
    `w_dense * dense` 变成常数项，排序**悄悄**退化成"只按词法覆盖率排"，且不报错。
    """
    if p.lexical_rerank and qt:
        scored: list[tuple[float, SearchHit]] = []
        for h in cands:
            lex = len(qt & _tokens(h.content)) / len(qt)
            dense = (h.score + 1.0) / 2.0     # COSINE ∈ [-1,1] → [0,1]
            scored.append((p.w_dense * dense + (1 - p.w_dense) * lex, h))
        scored.sort(key=lambda x: -x[0])
        return [h for _, h in scored]
    return sorted(cands, key=lambda h: -h.score)


# ----------------------------------------------------------------------
# 同事实聚合（工单02）
# ----------------------------------------------------------------------
def _split_sentence_units(text: str) -> list[str]:
    """按句末标点切句。

    直接复用分块器的刀（chunker._split_sentences），不另写一套 ——
    两套切句逻辑迟早会分叉，而分块器和聚合器对"一句话"的定义必须一致。
    """
    from app.core.chunker import _split_sentences
    return [s for s in _split_sentences(text) if s.strip()]


def _char_ngrams(text: str, n: int) -> set[str]:
    """字符 n-gram 集合（先去掉所有空白，避免换行把 n-gram 打断）。"""
    t = "".join(text.split())
    if len(t) < n:
        return {t} if t else set()
    return {t[i:i + n] for i in range(len(t) - n + 1)}


def fact_key_sentences(hit: SearchHit, query_tokens: set[str],
                       top: int = 2) -> list[str]:
    """挑出这个 chunk 里最可能承载「答案」的 1~2 句话。

    【为什么要挑句子而不是拿整块比】
    见模块顶部注释：500 字的块之间只共享约 40 字的同一事实句，整块相似度被
    稀释到 0.055，完全没有区分度。事实活在句子粒度上，比较就必须在句子粒度上。

    【挑不到怎么办】退回整块的开头。**保守优先**：不聚类只是没优化，
    错聚类会把不同事实合并、直接丢答案 —— 后者严重得多。
    """
    cands = [s for s in _split_sentence_units(hit.content)
             if query_tokens & _tokens(s)]
    if not cands:
        return [hit.content[:160]]
    cands.sort(key=lambda s: -len(query_tokens & _tokens(s)))
    return cands[:top]


def _sentences_same_fact(a: list[str], b: list[str], n: int, thr: float) -> bool:
    """两组关键句之间是否存在「同一事实」——用 n-gram **包含率**判定。

    用 `|gx∩gy| / min(|gx|,|gy|)` 而不是 Jaccard：两处表述长短不一
    （316 是 538 字、358 是 367 字），Jaccard 会被长度差拖低，包含率不会。

    【阈值 0.65 是标定过的，不是拍的】实测句子级 6-gram 包含率：
        真同一事实：316~358=0.74  355~358=0.78  358~44=1.00  358~43=1.00
                    （id=795）202~358=1.00  202~307=1.00
        真不同事实：316~42=0.00   316~202=0.00
        边界假阳：  42~358=0.59    ← 被 0.65 挡掉，这就是取 0.65 而不是 0.5 的原因
    """
    for x in a:
        gx = _char_ngrams(x, n)
        if not gx:
            continue
        for y in b:
            gy = _char_ngrams(y, n)
            if gy and len(gx & gy) / min(len(gx), len(gy)) >= thr:
                return True
    return False


def cluster_facts(hits: list[SearchHit], query: str,
                  p: RetrievalProfile) -> list[list[SearchHit]]:
    """把「同一事实的不同表述」聚成组。返回组列表，组内保持原顺序。

    【为什么必须用并查集做传递闭包，而不是"和组长比"】
    实测 316~355 的句子包含率是 **0.00** —— 一个写「全军第一个」、一个写
    「国防用户第一个」，措辞完全分叉。但它们各自都和 358 相似（0.74 / 0.78）。
    贪心式"跟首个代表比"会把 355 漏在外面，只有传递闭包能把三块并成一组。

    复杂度：n ≤ pool_size(30)，每块 2 句，每对最多 4 次集合交 —— 实测 <5ms。
    """
    qt = _tokens(abstract_query(query))
    keys = [fact_key_sentences(h, qt) for h in hits]
    parent = list(range(len(hits)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]      # 路径压缩
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(len(hits)):
        for j in range(i + 1, len(hits)):
            if _sentences_same_fact(keys[i], keys[j], p.fact_ngram, p.fact_sim):
                union(i, j)

    groups: dict[int, list[SearchHit]] = {}
    for i, h in enumerate(hits):
        groups.setdefault(find(i), []).append(h)
    return list(groups.values())


# 招股书的「第二节 概览」是全文的权威摘要：它给出的表述最完整，
# 也是参考答案的出处（例如 id=795 的完整句在 1-1-26 第二节，而不在附录里）。
_AUTHORITY_SECTION = ("第二节",)


def _authority_key(hit: SearchHit) -> tuple[int, int]:
    """组内排序键：**升序 = 更权威**。

    优先「第二节概览」，其次页码靠前（正文在附录前的披露更完整、更权威）。
    实测收益：id=795 的组 {202, 307, 358, 44} 里选中 44（1-1-26），
    其正文含完整的「（即相当于美军的C4ISR系统）荣获国家科技进步一等奖」。

    ⚠️ 诚实边界：它**只能在候选池里选**，不能把没召回的页变出来。
    """
    in_overview = 0 if hit.section_path.startswith(_AUTHORITY_SECTION) else 1
    return (in_overview, hit.page_no)


def merge_fact_group(group: list[SearchHit], rep: SearchHit,
                     query_tokens: set[str], budget: int) -> str:
    """代表块 +（可选）组内互补句。

    v1 默认 budget=0，也就是只用代表块：两个目标问题的代表块本身都含完整事实，
    合并只会白涨 prompt，而 TTFT 余量只剩 ~950ms。只有扫参显示有余量才打开。
    """
    if budget <= 0:
        return rep.content
    out, used = [rep.content], len(rep.content)
    for h in sorted(group, key=_authority_key):
        if h.chunk_id == rep.chunk_id:
            continue
        for s in _split_sentence_units(h.content):
            if s in rep.content:                  # 与代表块重复的整句不并入
                continue
            if not (query_tokens & _tokens(s)):   # 只并入与问题相关的补充
                continue
            if used + len(s) > budget:
                return "\n".join(out)
            out.append(s)
            used += len(s)
    return "\n".join(out)


# ----------------------------------------------------------------------
# 冗余过滤
# ----------------------------------------------------------------------
def _shingles(text: str) -> set[str]:
    """字符 3-gram 集合。"""
    t = "".join(text.split())
    if len(t) < 3:
        return {t}
    return {t[i:i + 3] for i in range(len(t) - 2)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def select_diverse(hits, k: int,
                   threshold: float = DUP_JACCARD) -> tuple[list, int]:
    """
    贪心挑选：按分数从高到低扫，与已选中任一条过于雷同就跳过。

    【为什么冗余控制在检索时做，而不是只在入库时做】
    原方案打算靠入库 SimHash 解决「重复段落占满 top-k」。实测这个前提不成立：
    12 个「重要供应商」chunk 两两海明距离是 14–34（入库阈值 3），字符重合度
    0.36–0.54 —— 它们不是重复文本，而是不同内容提到同一话题。放宽阈值会
    连带删掉大量真实信息（「程家明」在 73 个页面出现，每处都是有效内容）。
    放在检索阶段判断"这一次召回的这几条"，既精准又不损失入库数据。
    """
    chosen, chosen_sets, filtered = [], [], 0
    for h in hits:
        if len(chosen) >= k:
            break
        s = _shingles(h.content)
        if any(jaccard(s, cs) >= threshold for cs in chosen_sets):
            filtered += 1
            continue
        chosen.append(h)
        chosen_sets.append(s)
    return chosen, filtered


# ----------------------------------------------------------------------
def expand_with_neighbors(hits: list[SearchHit], query: str,
                          p: RetrievalProfile, fetch) -> tuple[list[SearchHit], int]:
    """邻块扩展：给每个命中块补上同文档的 chunk_index±radius 邻块。

    【工单03 修：必须按 doc_id 分组】原先只取 `hits[0].doc_id`
    （见下方 _expand_one_doc 的注释），两份文档进库后，top-k 一旦跨文档，
    邻块就会用第一份文档的 doc_id 去查，**把另一份文档的块拉进来**（静默串库）。
    现在按 doc_id 分组，各组用自己的 id 查。
    """
    qt = _tokens(abstract_query(query))
    groups: dict[str, list[SearchHit]] = {}
    for h in hits:
        groups.setdefault(h.doc_id, []).append(h)
    added: list[SearchHit] = []
    total = 0
    for doc_id, sub in groups.items():
        extra, used = _expand_one_doc(sub, qt, p, fetch)
        added.extend(extra)
        total += used
    return added, total


def _expand_one_doc(hits: list[SearchHit], qt: set[str],
                    p: RetrievalProfile, fetch) -> tuple[list[SearchHit], int]:
    """单份文档内部的邻块扩展（`expand_with_neighbors` 按 doc_id 分组后逐组调用）。

    【互补性过滤是省 prefill 的关键】
    邻块必须带来**原块没有的查询词**才收，否则丢弃。实测 id=95 的邻块
    （316/355/358 互相之间）会被这条规则全部挡掉 —— 因为它们本身都完整，
   这正是想要的行为：省下 prompt，不占那 ~950ms 的 TTFT 余量。

    【工单03：表格邻块无条件收】上面这条规则有个反例，正是表格题。
    实测 id=2（"本次募集资金拟投资哪些项目？"）page 21 被切成两块：

        chunk 42（正文）「五、募集资金用途本次募集资金拟投资以下项目：…」
        chunk 41（表格）「| 1 | 仓储及物流中心 | 3,393.40 | … | 5 | 其他与主营业务相关的营运资金 |」

    引子句和表格被**分到了不同的块**，而引子句与问法高度相似 → 稠密检索把
    chunk 42 排到第 1、整张表连候选池都进不去。更糟的是互补性过滤也救不回来：
    chunk 41 里全是项目名和金额，**不含任何新的查询词**（"募集资金/投资项目"
    这些词它和引子句共有），于是被判为"没带来新信息"丢掉 —— 可答案是它。
    查询词集合这套判据天然测不出"邻块里有答案"。

    表格是稠密事实载体、且是"整表不可切分"的原子块，所以表格邻块一律收。

    【双重字数上限】单个**正文**邻块走 best_window 截到 neighbor_chars（不是把
    500~600 字的整块塞进去），全部邻块合计不超过 neighbor_total_chars。
    表格不走 best_window —— 按查询词密度切窗口会把表切在行中间，破坏原子性。

    【邻块不占 top-k 名额、不参与重排】命中块（rank 出来的）才是检索结果；
    邻块只是上下文补充，标记 is_neighbor=True，用于精确率统计时排除。
    """
    if not hits:
        return [], 0
    doc_id = hits[0].doc_id
    known = {h.chunk_index for h in hits}
    want = {h.chunk_index + d for h in hits
            for d in range(-p.neighbor_radius, p.neighbor_radius + 1) if d}
    want -= known
    if not want:
        return [], 0
    try:
        cands = fetch(sorted(want), doc_id)
    except Exception:  # noqa: BLE001
        # 邻块是锦上添花，取不到就当作没有 —— 绝不让问答因此失败
        return [], 0

    # qt 由调用方 expand_with_neighbors 算好传入（工单03 按 doc_id 分组后，
    # 每组共用同一份查询词，不再各自重复算）
    owner_tokens = {h.chunk_index: _tokens(h.content) for h in hits}
    added: list[SearchHit] = []
    total = 0
    for c in cands:                                  # fetch 已按 chunk_index 排序
        owner = min(hits, key=lambda h: abs(h.chunk_index - c.chunk_index))
        is_atomic = is_atomic_chunk(c.chunk_type)
        # 表格邻块不参与互补性过滤（理由见 docstring：答案是它，可它不含新查询词）
        if not is_atomic:
            new_tok = (qt & _tokens(c.content)) - owner_tokens.get(owner.chunk_index, set())
            if not new_tok:
                continue
            body = best_window(c.content, qt, p.neighbor_chars)
            if total + len(body) > p.neighbor_total_chars:
                break        # 预算用尽，后面的正文邻块也放不下
        else:
            body = c.content  # 整表不可切分，不切窗口
            if total + len(body) > p.neighbor_total_chars:
                continue     # 这张表放不下就跳过它，别打断后面更小的邻块
        c.content = body
        c.serves = owner.chunk_index
        c.is_neighbor = True
        added.append(c)
        total += len(body)
    return added, total


class Retriever:
    def __init__(self, store: VectorStore | None = None,
                 embedder: Embedder | None = None,
                 profile: RetrievalProfile | None = None) -> None:
        self.store = store or VectorStore()
        self.embedder = embedder or Embedder()
        # 剖面是**不可变对象、显式传递**，不在请求里改写全局 settings（见 profiles.py）
        self.profile = profile or get_profile()

    async def retrieve(self, query: str, k: int | None = None, *,
                       profile: RetrievalProfile | None = None,
                       diverse: bool | None = None,
                       pool_size: int | None = None) -> RetrievalResult:
        """检索主链路。

        `diverse`/`pool_size` 是工单01 遗留的覆盖参数，保留兼容。

        【工单06：三种检索模式】
          vector   —— 稠密向量召回（工单01-05 的路径，**逐字节不变**）
          fulltext —— 倒排索引直接检索（不起向量、不嵌入）
          hybrid   —— 两路并行 + 融合（RRF 投票 / 加权平均）
        分流点是**显式的**：只有 `retrieval_mode == "vector"` 且
        `reranker == "lexical"` 时才走工单05 那条老路（见第 3 步）。
        """
        p = profile or self.profile
        t0 = time.perf_counter()
        stage: dict[str, float] = {}
        k = k or p.top_k
        n_pool = pool_size or p.pool_size
        mode = p.retrieval_mode
        views = query_views(query) if p.dual_view else [query.strip()]

        # ---------- 0. 实体路由（工单03）----------
        # 两份招股书章节结构雷同，「发行股数」「募集资金」两边都有，靠排序**无法可靠
        # 区分**（分数会随上下文措辞浮动），但问题里点了公司名，匹配是确定的 ——
        # 所以按 doc_name 硬过滤，不赌排序。理由详见 app/core/router.py。
        r = route(query)
        multi = route_multi(query)
        # 一句话点了两家公司（"A 和 B 分别…"）时不过滤：限定任何一份都会漏掉另一半。
        # 【已知限度】不过滤只是"不主动漏"，并不保证两家的答案都进 top-k ——
        # 见 docs/工单03-表格解析与多文档检索.md 第七节第 5 条（问"两家注册资本
        # 各是多少"时力源那条压根没被召回）。跨文档提问不在工单要求的 14 题内。
        expr = r.expr if len(multi) <= 1 else ""
        route_doc = r.doc_name if expr else ""

        # ---------- 1a. 稠密路（vector / hybrid）----------
        dense_hits: list[SearchHit] = []
        route_fallback = False
        if mode in ("vector", "hybrid") and p.fusion_impl != "milvus":
            t = time.perf_counter()
            vectors = await asyncio.gather(*(self.embedder.embed_one(v) for v in views))
            stage["embed_ms"] = (time.perf_counter() - t) * 1000

            t = time.perf_counter()

            def _search(vec, filter_expr: str):
                return (self.store.search(vec, n_pool, expr=filter_expr) if filter_expr
                        else self.store.search(vec, n_pool))

            pools = await asyncio.gather(*(
                asyncio.to_thread(_search, vec, expr) for vec in vectors
            ))

            # 路由后一条都没召回到 → 回退全库。
            # 【为什么必须回退】过滤是"确定的"，但"问题里点了名的公司"和"答案所在的公司"
            # 并不总是同一家（如"和兴图比，力源的发行股数是多少"）。宁可多召回也不能空手，
            # 空上下文会让模型只能凭空编。回退这件事必须记进 route_fallback，否则
            # 事后无法区分"路由生效了"和"路由白过滤了一次"。
            if expr and not any(pools):
                route_fallback = True
                pools = await asyncio.gather(*(
                    asyncio.to_thread(_search, vec, "") for vec in vectors
                ))
            stage["search_ms"] = (time.perf_counter() - t) * 1000

            merged: dict[int, SearchHit] = {}
            for pool in pools:
                for h in pool:
                    merged.setdefault(h.chunk_id, h)
            dense_hits = list(merged.values())

        # ---------- 1b. 关键词路（fulltext / hybrid）----------
        keyword_hits: list[SearchHit] = []
        if mode in ("fulltext", "hybrid"):
            t = time.perf_counter()
            keyword_hits = await asyncio.to_thread(self._keyword_search, query, n_pool, route_doc)
            stage["keyword_ms"] = (time.perf_counter() - t) * 1000

        # ---------- 2. 合成候选 ----------
        fusion_info: dict = {}
        if mode == "fulltext":
            cands = keyword_hits
        elif mode == "hybrid":
            if p.fusion_impl == "milvus":
                cands = await self._milvus_hybrid(query, views, k, expr)
                fusion_info = {"fusion": "milvus-native"}
                if expr and not cands:                     # 与稠密路同样的回退规则
                    route_fallback = True
                    cands = await self._milvus_hybrid(query, views, k, "")
            else:
                fused = fuse(dense_hits, keyword_hits,
                             method=p.fusion, w_keyword=p.w_keyword)
                cands = fused.hits
                fusion_info = fused.as_dict()
        else:
            cands = dense_hits

        # ---------- 3. 排序 / 重排（工单02 + 工单06）----------
        # 【回归红线】**只有** vector 模式 + lexical 重排器才走工单05 那条老路，
        # 而且它是**原样提取**出来的 `_rank_legacy`（算术一模一样）。
        # 其余组合一律走可插拔重排器 —— 因为老路的 `(score+1)/2` 只对 COSINE 成立，
        # BM25 无上界、RRF 只有 0.016 量级，套进去会被压成常数、排序静默退化。
        t = time.perf_counter()
        qt = _tokens(abstract_query(query))
        rerank_info = RerankInfo(reranker=p.reranker)
        if mode == "vector" and p.reranker == "lexical" and p.fusion_impl != "milvus":
            ranked = _rank_legacy(cands, qt, p)
        else:
            ranked, rerank_info = await rerank(cands, query, profile=p)
            stage["rerank_ms"] = float(rerank_info.ms)
        stage["rank_ms"] = (time.perf_counter() - t) * 1000

        # ---------- 4. 同事实聚合（工单02）----------
        # 把"同一事实的不同表述"塌成一组、每组只留一个代表。
        # 注意顺序：**先聚合、后多样性过滤** —— 聚合消的是语义重复，
        # select_diverse 消的是字面重复，聚合完再看字面重复更干净。
        n_merged = 0
        groups: list[list[SearchHit]] = []
        if p.aggregate_facts and len(ranked) > 1:
            t = time.perf_counter()
            groups = cluster_facts(ranked, query, p)
            # 组按"组内最优排名"排序，保证代表块的相对次序不乱
            order = {h.chunk_id: i for i, h in enumerate(ranked)}
            groups.sort(key=lambda g: min(order[h.chunk_id] for h in g))
            reps: list[SearchHit] = []
            for g in groups:
                rep = min(g, key=_authority_key) if p.authority_bias else g[0]
                if p.fact_merge_chars > 0:
                    rep.content = merge_fact_group(g, rep, qt, p.fact_merge_chars)
                reps.append(rep)
            n_merged = len(ranked) - len(reps)
            ranked = reps
            stage["cluster_ms"] = (time.perf_counter() - t) * 1000

        # ---------- 5. 冗余过滤 → top-k ----------
        # 名额天然够：聚合是在整个候选池（≤30）上做的，所以一定能凑满 k 组。
        want_diverse = p.diverse if diverse is None else diverse
        filtered = 0
        if want_diverse and ranked:
            hits, filtered = select_diverse(ranked, k, threshold=p.dup_jaccard)
        else:
            hits = ranked[:k]

        # ---------- 6. 邻块扩展（工单02）----------
        n_neighbors = 0
        if p.expand_neighbors and hits:
            t = time.perf_counter()
            extra, _ = expand_with_neighbors(
                hits, query, p, self.store.fetch_by_chunk_index)
            hits = hits + extra
            n_neighbors = len(extra)
            stage["neighbor_ms"] = (time.perf_counter() - t) * 1000

        total = time.perf_counter() - t0
        stage["retrieval_ms"] = total * 1000
        return RetrievalResult(
            query=query, hits=hits, n_candidates=len(cands),
            n_filtered=filtered, seconds=total, query_views=views,
            profile=p.name, n_merged=n_merged, n_added_neighbors=n_neighbors,
            stage_ms=stage,
            # 跨文档时 routed_doc 留空（我们确实没有过滤），命中的两家只在
            # route_matched 里留痕，供排查 —— 别让"没过滤"看起来像"过滤了"。
            routed_doc=(r.doc_name or "") if expr else "",
            route_matched=(r.matched if expr
                           else (f"跨文档({'、'.join(multi)})，不过滤" if multi else "")),
            route_fallback=route_fallback,
            # ---------- 工单06 ----------
            retrieval_mode=mode,
            fusion=str(fusion_info.get("fusion", "")),
            reranker=rerank_info.reranker,
            n_from_dense=(fusion_info.get("n_dense", len(dense_hits))
                          if mode == "hybrid" else len(dense_hits)),
            n_from_keyword=(fusion_info.get("n_keyword", len(keyword_hits))
                            if mode == "hybrid" else len(keyword_hits)),
            n_shared=int(fusion_info.get("n_shared", 0) or 0),
            rerank_ms=rerank_info.ms,
            rerank_fallback=rerank_info.fallback,
            feedback_n=rerank_info.feedback_n,
        )

    # ------------------------------------------------------------------
    # 工单06：关键词路
    # ------------------------------------------------------------------
    def _keyword_search(self, query: str, k: int, doc_name: str) -> list[SearchHit]:
        """全文检索那一路。

        【为什么传 doc_name 而不是 Milvus 的 expr 字符串】`expr` 是
        `doc_name == "招股说明书2.pdf"` 这种 Milvus 表达式，在 Python 里解析它
        太脆（引号、转义、将来支持 and 之后更麻烦）。路由结果本来就是结构化的，
        直接传结构化字段。
        """
        from app.core.fulltext import get_index
        return get_index(self.store.collection, store=self.store).search(
            query, k, doc_name=doc_name or None)

    async def _milvus_hybrid(self, query: str, views: list[str], k: int,
                             expr: str) -> list[SearchHit]:
        """向量库原生的混合检索（`hybrid_search` + WeightedRanker / RRFRanker）。

        【它与应用层融合的区别】Milvus 用稀疏向量表达关键词路（词袋），
        做不了布尔/短语/模糊；好处是两路在**库内**一次算完、少一次往返。
        作为对照路线保留（`--fusion-impl milvus`），不是交付默认。
        """
        from app.core.text_analysis import term_id, tokenize  # noqa: F401
        from app.core.vectorstore import lexical_sparse

        vec = await self.embedder.embed_one(views[0])
        sparse = lexical_sparse(query)
        ranker_name = "rrf" if self.profile.fusion == "rrf" else "weighted"
        return await asyncio.to_thread(
            self.store.hybrid_search, vec, sparse, k,
            expr=expr, ranker=ranker_name, w_keyword=self.profile.w_keyword)

    def retrieve_sync(self, query: str, k: int | None = None, **kw) -> RetrievalResult:
        return asyncio.run(self.retrieve(query, k, **kw))


def best_window(text: str, query_tokens: set[str], width: int) -> str:
    """
    在长片段里取「与查询词重合最多」的窗口，而不是死截开头。

    【为什么不能用 text[:300]】实测 id=793（行业下游包括哪些）：
    召回片段 1-1-151 开头是上下游关系图和资质壁垒的描述，含答案的那句
    「下游行业为各类终端用户……主要包括军队、政府机关、能源等行业企业」
    在 300 字之后，被截掉了。模型只看到图注里的「军队企事业单位」，
    于是答成「军队、企事业单位」，漏了政府机关与能源 ——
    **答案不是检索错了，是被截断毁了**。

    窗口按滑动步长扫描，取命中查询词最多的一段；并列时取靠前的
    （招股书里同一事实常先给结论后给论证）。
    """
    if len(text) <= width:
        return text
    if not query_tokens:
        return text[:width] + "…"

    def score_window(seg: str) -> tuple[int, int]:
        """
        打分 = (命中的查询词个数, −命中位置的跨度)。

        【为什么第二个维度是"跨度"而不是"靠前"】
        实测 id=793 的失败正是出在这里：候选片段 1-1-151 长 408 字，
        开头就出现「行业竞争格局」「行业上下游情况」，含答案的那句
        「下游行业为各类终端用户……主要包括军队、政府机关、能源」
        在 336 字处。窗口 [0:300] 和 [100:400] 命中同样多的查询词，
        按"靠前优先"会选中前者 —— 恰好把答案切掉，模型于是只答出
        「军队、企事业单位」（那是前文图注里的词）。
        真实答案几乎总在查询词**聚集**的地方，所以同分时选跨度更小的窗口。
        """
        matched, positions = 0, []
        for t in query_tokens:
            i = seg.find(t)
            if i >= 0:
                matched += 1
                positions.append(i)
        if not positions:
            return (0, 0)
        return (matched, -(max(positions) - min(positions)))

    step = max(1, width // 6)
    starts = list(range(0, max(1, len(text) - width + 1), step))
    if not starts or starts[-1] != len(text) - width:
        starts.append(len(text) - width)      # 收尾：确保覆盖到末尾窗口

    best_start, best_key = 0, (-1, 0)
    for start in starts:
        key = score_window(text[start:start + width])
        if key > best_key:
            best_start, best_key = start, key

    seg = text[best_start:best_start + width]
    return ("…" if best_start > 0 else "") + seg + \
           ("…" if best_start + width < len(text) else "")


def build_context(hits, max_chars_each: int | None = None,
                  query: str | None = None, *,
                  profile: RetrievalProfile | None = None) -> str:
    """
    把召回的片段拼成给 LLM 的上下文。

    【为什么截断】工单01 要求响应 ≤3 秒，prefill 耗时随上下文线性增长。
    实测每个片段压到 300 字、共 3 条（约 900 字 ≈ 1100 token）后 TTFT 压进预算。
    完整片段仍留在库里，前端「检索证据」展示的是召回原文。

    【为什么是"选窗口"而不是"截开头"】见 best_window 的注释 —— 截开头会
    静默丢掉答案。传入 query 后按查询词定位最佳窗口。
    """
    width = (max_chars_each
             or (profile.context_chunk_chars if profile is not None else None)
             or settings.context_chunk_chars)
    qt = _tokens(abstract_query(query)) if query else set()

    blocks: list[str] = []
    i = 0
    for h in hits:
        # 邻块不占用 [片段N] 编号 —— 否则模型会把它当独立证据去引用，
        # 而它只是给某一条命中块做补充的上下文。
        if h.is_neighbor:
            head = (f"[邻接补充] 页码 {h.page_label}"
                    f"（片段{h.serves} 的相邻内容）")
        else:
            i += 1
            head = f"[片段{i}] 页码 {h.page_label}"
            # 【工单04】只给图块加标记。**刻意不动 text/table 的拼装字节** ——
            # 那会让前 14 题的 prompt 逐字节改变，既打掉 Ollama 的前缀 KV cache，
            # 又可能改变模型行为，把已交付的 14/14 搅了。
            if h.chunk_type == "image":
                head += "（图）"
            if h.section_path:
                head += f"｜{h.section_path}"
        # 【工单04：原子块绝不切窗口】表格与图像块是**条目式**内容，
        # `best_window` 按查询词密度取窗口，会把表/图切在条目中间，
        # **静默丢掉答案**（图块尤甚：页 71 的转写超过 600 字，而"负增长的是
        # 哪个行业"那一行排在末尾，切窗口正好把它切掉）。这与工单01 的
        # id=793 是同一类失败，只是换了种内容形态。
        body = (h.content
                if is_atomic_chunk(h.chunk_type) or len(h.content) <= width
                else best_window(h.content, qt, width))
        blocks.append(f"{head}\n{body}")
    return "\n\n".join(blocks)
