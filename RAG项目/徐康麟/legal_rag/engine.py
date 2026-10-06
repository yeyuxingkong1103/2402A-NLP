# -*- coding: utf-8 -*-
"""RAG 引擎：把检索、重排、提示词组装、生成、后处理串成一条链路。

    question
      -> 向量化 -> 混合检索(向量 + BM25) -> 重排
      -> 组装提示词(角色设定 + 检索知识 + 历史对话 + 用户问题)
      -> 模型路由调用大模型 -> 后处理 -> Answer(answer + citations)
"""
from __future__ import annotations

import logging
from collections.abc import Iterator

from . import metrics as M
from .config import RagConfig
from .embedding.base import Embedder, build_embedder
from .generate.llm_base import LLMClient
from .generate.postprocess import postprocess
from .generate.llm_metrics import take_context_trimmed, take_route_degraded
from .generate.prompt import build_citations, build_general_messages, build_messages
from .generate.router import MOCK_PROVIDER, ModelRouter
# P0.5：函数级耗时观测（L1 直方图 + L2 慢调用清单），见 docs/REFACTOR-PLAN.md §3.7
from .observability import timed
from .retrieve.hybrid import HybridRetriever
from .retrieve.law_scope import LawScope
from .retrieve.rerank import Reranker, build_reranker
from .retrieve.version_filter import VersionFilter
from .route_intent import non_legal_reason
from .schemas import Answer, Message, Role, SearchHit
from .store.base import VectorStore, build_store

logger = logging.getLogger(__name__)

#: 「这可能在问法律」的口语线索词（t109 立、t115 F1 扩充）。用途只有一个：当检索阶段
#: 没给出任何够格的碎片时，把「法律问题但资料不够」（引导补充/换说法）与「压根不是法律
#: 问题」（通用对话）分开 —— **不能**把「朋友欠我钱不还怎么办」「楼上漏水把我家泡了，
#: 他不赔钱怎么办」这种大白话判成闲聊。
#: 口径：收**法律场景词/行为词**（欠钱、漏水、辞退、起诉…），不收「法」「规」这类通用字。
#: ⚠️ 这是一张**人工维护**的清单，不是说它完备：新出现的口语场景要往这里补词
#: （t115 F1 的反例「楼上漏水…不赔钱」就是因为缺 `漏水`/`泡了`/`赔钱` 才被判成闲聊）。
_LEGAL_SIGNAL_WORDS: tuple[str, ...] = (
    # 借钱/欠款/赖账
    "欠钱", "欠款", "借钱", "还钱", "不还", "借条", "欠条", "转账", "催收", "老赖",
    "赖账", "赖着", "拖欠", "欠薪", "讨债", "要回", "退款",
    "高利贷", "网贷", "利息", "违约金", "定金",
    # 劳动关系
    "工资", "加班", "社保", "公积金", "劳动合同", "试用期", "辞退", "开除", "离职",
    "解雇", "裁员", "工伤", "劳动仲裁", "仲裁", "竞业", "年假",
    # 住房/租赁/邻里
    "押金", "房租", "房东", "租房", "退租", "物业", "中介",
    "漏水", "渗水", "泡了", "淹了", "楼上", "邻居", "噪音", "装修",
    # 家庭/继承
    "离婚", "抚养", "赡养", "继承", "遗产", "彩礼", "抚养费", "家暴",
    # 交易/侵权/事故
    "违约", "合同", "赔偿", "赔钱", "不赔", "侵权", "车祸", "交通事故", "酒驾",
    "保险理赔", "受伤", "打人", "打架",
    # 纠纷处理
    "起诉", "法院", "判决", "律师", "报警", "拘留", "坐牢", "判刑", "刑事", "民事",
    "诈骗", "维权", "投诉",
    # 消费 / 商品质量 / 网络打赏
    # t-评测 V01 的反例：「买到假货能不能要求退一赔三」曾因缺 `假货`/`退一赔三`
    # 被判成闲聊 → 走 general → 模型被"这轮不是法律咨询"的通用提示词带偏，
    # 只回了一句"请把你的具体情况说清楚"（27 字），既没答也没说资料局限。
    "假货", "假一赔", "退一赔三", "三倍", "双倍", "欺诈", "虚假宣传", "以次充好",
    "消费者", "消协", "退货", "七天无理由", "质量问题", "售后", "打赏", "未成年",
)


def _has_legal_signal(question: str) -> bool:
    """问题里有没有「法律场景」的口语线索（判定口径见 ``_LEGAL_SIGNAL_WORDS``）。"""
    text = str(question or "")
    return any(word in text for word in _LEGAL_SIGNAL_WORDS)


#: 兜底到演示后端（mock）时，答案里**必须**带这句用户看得见的提示（t120 F1 方案②）。
#: 口径沿用 handoff/COPY-STANDARD.md：说人话、不甩锅、不吓人，并说明"现在能做什么"。
FALLBACK_NOTICE = ("[提示] 真实大模型当前不可用，这条回答由演示用的兜底模型生成，"
                   "仅供参考、不能当作正式结论；等服务恢复后请重新提问。")


def out_of_scope_reply(question: str) -> str:
    """**拿不准**时的回法（检索没给出够格资料、但问题带法律信号）。

    不调用大模型、不带任何引用，也**不锚定**上一轮话题（这里根本读不到历史）：
    说清"资料不够就直说、不硬答、不编条文"，再给一个可以怎么问的例子，并邀请补充
    关键信息（时间/地点/书面合同/对方是谁/走到哪一步）或换个更具体的说法。
    """
    example = "「劳动合同到期不续签，用人单位要不要给补偿」"
    return (
        "这个问题看起来是法律问题，但我在法规资料里没找到足够贴近的内容，"
        "先不硬答、也不给您编条文。\n"
        "您可以补充一点关键信息再问（比如时间、地点、有没有书面合同、对方是谁、"
        "已经走到哪一步），或者换个更具体的说法，例如：" + example + "。"
    )


class RagEngine:
    """RAG 主链路：**装配 → 检索 → 拼提示 → 调大模型 → 返回答案与引用**。

    一次问答的完整顺序（改动这里等于改产品行为，务必先读 `docs/ARCHITECTURE.md`）：

    1. **预路由**（`_preroute`）：明显非法律的问题（口算/翻译/天气/闲聊…）直接跳过检索、
       锁定 ``general`` —— 这是 §D7 的修复，避免"口算题检索一堆法条再把提示撑爆"；
    2. **检索**（`prepare`）：按 role 分区召回 → 融合 → 重排 → 相关性闸门；
    3. **路由**（`scope_route`）：``legal`` / ``clarify`` / ``general`` 三分类；
    4. **生成**（`ask` / `ask_stream`）：拼 messages 交给 `ModelRouter`；
       提示超窗时由 `generate/openai_compat` 按证据块逐档裁剪重试（§D7）。

    诚实边界：``store_actual`` 与 ``degraded`` 记录**真实**后端（配置要 milvus、
    实际可能是降级的内存库）—— ``/health`` 与答案的可用性判断都读它们。
    """

    def __init__(self, config: RagConfig | None = None, *,
                 embedder: Embedder | None = None,
                 store: VectorStore | None = None,
                 reranker: Reranker | None = None,
                 router: ModelRouter | None = None) -> None:
        self.config = config or RagConfig.from_env()
        self.embedder = embedder or build_embedder(
            self.config.embedding_provider, self.config.embedding_model, self.config.embedding_dim,
            config=self.config,
        )
        # 关键：必须把 config 与 embedder 一起交给 build_store。
        # 否则 store 侧拿不到「实测维度」（只会退回 config.embedding_dim 这个离线兜底值），
        # 与 Milvus collection 的 1024 维冲突 → 静默降级成内存库，服务却照报 ok。
        self.store = store or build_store(
            self.config.vector_store, persist_dir=self.config.index_dir,
            collection=self.config.collection, config=self.config, embedder=self.embedder,
        )
        # 降级可见性：记录「配置想要的后端」与「实际拿到的后端」，
        # /health 必须能一眼看出是不是真的降级了（t7 F-t2-01 藏了很久就是因为这里没信息）。
        self.store_requested = self.config.vector_store
        self.store_actual = self.store.name
        self.store_primary = getattr(self.store, "primary", "")
        self.degraded_reason = getattr(self.store, "fallback_reason", "")
        self.degraded = bool(self.degraded_reason) and (
            self.store_actual != str(self.store_requested).strip().lower())
        if self.degraded:
            logger.warning(
                "向量库已降级：配置要求 %s，实际使用 %s（原因见 store 侧 WARNING）；"
                "本次服务不是真实后端，请勿据此判定链路可用",
                self.store_requested, self.store_actual)
        self.reranker = reranker if reranker is not None else build_reranker(
            self.config.rerank_provider, self.config.rerank_model, self.config.retrieval
        )
        # 法名定向召回索引：优先 ``index/law_sources.json``，否则扫 ``knowledge/`` 的文件名
        # （只读文件名，毫秒级）。两者都拿不到就返回空索引 ⇒ 该路不生效，其余行为不变。
        self.law_scope = LawScope.load_or_build(self.config.index_dir, self.config.knowledge_dir)
        # 版本过滤：读 index/version_inventory.json（由 scripts/audit_corpus_versions.py 生成）。
        # 文件缺失/损坏 ⇒ 返回 None ⇒ 不过滤（fail-open），日志里会说清怎么生成。
        self.version_filter = (
            VersionFilter.load_or_none(self.config.index_dir)
            if getattr(self.config.retrieval, "version_filter_enabled", True) else None
        )
        self.router = router or ModelRouter(self.config)
        # LLM 列表式选择器（P8）：**速度与质量取中间值**的实现 —— 只在识别到法名时、用
        # 便宜口径（少量候选 + 短片段）让大模型挑"最能回答问题的条"。默认开（实测增益见
        # `docs/RERANK-EXPERIMENT.md`），构造失败/没有可用后端一律降级为"不启用"。
        self.selector = self._build_selector()
        self.retriever = HybridRetriever(self.embedder, self.store, self.config.retrieval,
                                         self.reranker, law_scope=self.law_scope,
                                         cache_dir=self.config.index_dir,
                                         version_filter=self.version_filter,
                                         selector=self.selector)

    # ---------- 准备阶段 ----------
    def _build_selector(self) -> object | None:
        """构造检索侧「列表式选择器」；**任何失败都只降级、绝不让服务起不来**。

        用户口径（2026-09-23）：「速度与质量取中间值，速度也不能太慢」。实测三条：
        ① 候选数/片段长度是主要成本项 —— 60×120 字 ≈ 5.3 s/次，24×60 字 ≈ **1.43 s/次**，
           而且**指标更好**（cosine Recall@5 0.5513 vs 0.5385）；
        ② 只在识别到法名时才调，平均摊到每题 ≈ **+1.0 s**；
        ③ 触发开关 ``LAW_SELECTOR_TRIGGER=low_confidence`` 可进一步收紧（本套题集上判据几乎
           总成立 ⇒ 与 always 等价，但换语料/换阈值后能省）。
        """
        if not getattr(self.config.retrieval, "law_selector_enabled", False):
            logger.info("列表式选择器未启用（LAW_SELECTOR_ENABLED=false）")
            return None
        provider = str(self.config.llm_provider or "").strip()
        if provider.lower() == MOCK_PROVIDER:
            # mock 只会回显、没有推理能力：调它只会多一次往返 + 一堆解析告警
            logger.info("列表式选择器在 mock 后端下不启用（mock 没有推理能力）")
            return None
        try:
            from .retrieve.selector import ListwiseSelector

            selector = ListwiseSelector(
                self.router.selector_client(
                    timeout=float(getattr(self.config.retrieval, "law_selector_timeout", 8.0) or 8.0)),
                top_k=int(getattr(self.config.retrieval, "law_selector_top_k", 3) or 3),
                max_candidates=int(
                    getattr(self.config.retrieval, "law_selector_max_candidates", 24) or 24),
                snippet_chars=int(
                    getattr(self.config.retrieval, "law_selector_snippet_chars", 60) or 60),
            )
        except Exception as exc:  # noqa: BLE001 - 选择器起不来不该影响服务
            logger.warning("列表式选择器构造失败（检索照常，不启用选择器）：%s: %s",
                           type(exc).__name__, exc)
            return None
        logger.info("列表式选择器已就绪：provider=%s 候选上限=%d 每条字数=%d 触发=%s（min_law_hits=%s）",
                    provider, selector.max_candidates, selector.snippet_chars,
                    getattr(self.config.retrieval, "law_selector_trigger", "always"),
                    getattr(self.config.retrieval, "law_selector_min_law_hits", 2))
        return selector

    @staticmethod
    def role_scope(role: Role | None, where: dict | None = None) -> dict:
        """确定检索过滤条件：**角色知识互不可见**。

        * 显式传入 ``where`` 时以调用方为准（运维/管理用途，HTTP 层不暴露该参数）；
        * 否则按 ``role.role_id`` **严格等值**过滤：律师角色只能召回打上
          ``role_id=lawyer`` 的知识，其它角色同理，避免串味；
        * ``role_id`` 为空（未标注归属）的语料**不会**被任何角色召回 ——
          宁可查不到，也不能让 A 角色的知识漏给 B 角色。要让它可见，
          必须显式传 ``where``。
        """
        if where is not None:
            return where
        scope: dict = {"is_parent": False}
        role_id = str(getattr(role, "role_id", "") or "").strip()
        if role_id:
            scope["role_id"] = role_id
        return scope

    # P0.5：prepare 覆盖「检索 + 拼提示」整段，是问答链路第一个可观测的大段耗时。
    @timed(operation="engine.prepare", slow_ms=1000.0)
    def prepare(self, question: str, role: Role | None = None,
                history: list[Message] | None = None, top_k: int | None = None,
                where: dict | None = None,
                memory_block: str = "") -> tuple[list[SearchHit], list[dict]]:
        """检索 + 拼 messages，返回 ``(命中, messages)``。

        ``where`` 不传时按 role 推导作用域（``role_scope``）；``memory_block`` 是
        长期记忆块，**原样拼进提示**（它不参与裁剪 —— 裁剪只动检索证据，见 §D7）。
        """
        scope = self.role_scope(role, where)
        logger.info("入口 RagEngine.prepare(role=%s, history=%d, scope=%s, memory_block=%d字)",
                    getattr(role, "role_id", None), len(history or []), scope,
                    len(memory_block or ""))
        hits = self.retriever.retrieve(question, top_k=top_k, where=scope)
        messages = build_messages(role, hits, history, question, memory_block=memory_block)
        logger.info("出口 RagEngine.prepare -> 命中 %d 条知识", len(hits))
        return hits, messages

    # ---------- 三分类路由（t109）----------
    @staticmethod
    def scope_route(question: str, hits: list[SearchHit]) -> str:
        """三分类路由：``legal`` / ``clarify`` / ``general``。

        * ``legal``   —— 召回阶段给出了够格依据（分数达标 **且** 有词面证据）：走既有法律
          路径，依据资料作答并给引用；
        * ``clarify`` —— 没依据、但问题带法律信号（"朋友欠我钱不还怎么办"）：**不硬答**，
          说清资料不够、邀请补充或换说法（不编条文、不给引用）；
        * ``general`` —— 没依据、也没有法律信号（"你好"、"水的沸点是多少度"）：这是
          **日常对话 / 通用问题**，**不得套法律知识** —— 换成通用提示词、不注入任何资料、
          引用恒为空，交给大模型用常识自然回答（用户裁决：要有一定的通用对话能力）。
        """
        if hits:
            return "legal"
        return "clarify" if _has_legal_signal(question) else "general"

    def _note_route(self, route: str, question: str) -> None:
        M.counter("chat_route_total").inc(route=route)
        logger.info("本轮路由 = %s：query=%r", route, question)

    # ---------- 预路由（§D7：明显非法律的问题**不进检索**）----------
    def _preroute(self, question: str) -> str:
        """问题是否**明显不是**法律咨询？返回命中理由（空串 = 不拦），并留痕。

        为什么放在检索之前：`prepare()` 是"先检索后路由"，口算题也会白跑一遍检索；
        一旦闸门放行就被当成法律问题 ⇒ 拼出几千 token 的法律提示词 ⇒ 撞上下文窗口
        （2026-09-28 实测：7169 token + 1024 输出 > 4090 上 4B 服务的 8192 ⇒ vLLM 400）。
        """
        reason = non_legal_reason(question)
        if reason:
            M.counter("chat_preroute_total").inc(rule=reason.split(":", 1)[0])
            logger.info("[预路由] 判为非法律咨询（%s）-> 跳过检索，直接 general：query=%r",
                        reason, question)
        return reason

    def _prepare_or_general(self, question: str, role: Role | None,
                            history: list[Message] | None, top_k: int | None,
                            where: dict | None,
                            memory_block: str) -> tuple[list[SearchHit], list[dict], bool]:
        """返回 ``(hits, messages, prerouted)``；预路由命中时不检索、直接给通用提示词。"""
        if self._preroute(question):
            return [], build_general_messages(question, memory_block=memory_block), True
        hits, messages = self.prepare(question, role, history, top_k, where,
                                      memory_block=memory_block)
        return hits, messages, False

    def _warmup_note(self) -> str:
        """本轮是否处于**预热/降级窗口**（检索层给出），以及要贴在答案里的那句提示。

        t115 F2：BM25 索引是**后台**建的（真实语料 11.9 万块约 30~75s），这段窗口里
        召回只有向量通道 —— 用户必须看得见"资料可能不全"，而不是拿到一个看起来完整的
        答案（COPY-STANDARD 第 C 条：解释系统在忙什么）。提示由检索层产生，这里只负责
        贴到用户可见处并计数。
        """
        note = str(getattr(self.retriever, "last_retrieval_note", "") or "")
        if note:
            M.counter("retrieval_warmup_notice_total").inc(
                store=str(getattr(self.store, "name", "unknown")))
            logger.info("本轮处于预热/降级窗口 -> 答案里带上可见提示：query-note=%r", note[:40])
        return note

    def _clarify_answer(self, question: str, role: Role | None, session_id: str,
                        note: str = "") -> Answer:
        """``clarify`` 分支：**不调用大模型**，也不给任何引用（拿不准就说拿不准）。"""
        self._note_route("clarify", question)
        logger.info("入口命中为空且带法律信号 -> 引导补充/换说法（不调用大模型、引用恒为空）："
                    "query=%r role=%s", question, getattr(role, "role_id", None))
        text = out_of_scope_reply(question)
        return Answer(
            text=(note + "\n\n" + text) if note else text,
            citations=[],
            role_id=getattr(role, "role_id", "") or "",
            session_id=session_id,
            provider="scope-guard",
        )

    def _take_degradation(self) -> tuple[bool, str, str]:
        """本轮是否走了兜底、为什么、兜到了谁（读一次即清空，不污染下一轮）。

        数据由 :func:`legal_rag.generate.router.ModelRouter._degrade` 落在 ContextVar 上
        （与客户端消费的那份是**两份**，互不干扰）。同一线程内读得到 —— ``ask`` /
        ``ask_stream`` 都在线程池的同一个工作线程里跑完这一步。
        """
        record = take_route_degraded()
        if not record:
            return False, "", ""
        _from, to, reason = record
        return True, reason or "unknown", to

    def _fallback_prefix(self, degraded: bool, reason: str, fallback_to: str,
                         note: str) -> str:
        """拼答案前缀：预热提示（t115 F2）在前，兜底提示（t120 F1）在后。"""
        prefix = note + "\n\n" if note else ""
        if degraded and fallback_to == "mock":
            # 方案②：必须兜底时，**不许**让假答案看起来像正常回答 —— 答案里给可见提示，
            # 响应里 provider=mock / degraded=true / degraded_reason 也如实标（接口层负责）。
            logger.warning("[LLM-FALLBACK] 本轮回答来自演示兜底后端 mock（reason=%s）："
                           "已在答案里加入可见提示，并在响应里标 degraded=true", reason)
            prefix += FALLBACK_NOTICE + "\n\n"
        return prefix

    # ---------- 上下文裁剪的可见提示（§D7）----------
    def _context_trim_prefix(self) -> str:
        """本轮若因超出窗口裁过依据，就在答案前**说清楚**（不许静默少给依据）。"""
        info = take_context_trimmed()
        if not info:
            return ""
        dropped = int(info.get("dropped_chars") or 0)
        logger.warning("[上下文裁剪] 本轮因超出模型窗口裁掉了部分依据（丢 %d 字，保留比例 %s）"
                       "⇒ 答案里带可见提示", dropped, info.get("keep_ratio"))
        return (f"[提示] 这轮的参考资料偏长、超出了模型上下文窗口，"
                f"已按相关性保留最相关的部分（约裁掉 {dropped} 字）。"
                f"如果需要更完整的依据，请把问题问得更具体一些。\n\n")

    # P0.5：整轮问答（检索 + 生成 + 后处理）的总耗时。阈值 3000ms——
    # 这是"用户到底等了多久"的口径，也是判断「慢在检索还是慢在生成」的分界。
    @timed(operation="engine.ask", slow_ms=3000.0)
    def ask(self, question: str, role: Role | None = None, session_id: str = "",
            user_id: str = "", history: list[Message] | None = None,
            top_k: int | None = None, where: dict | None = None,
            memory_block: str = "") -> Answer:
        """非流式问答：返回 ``Answer``（正文 + 引用 + 是否降级）。

        三条出口（互斥）：
        * ``clarify`` —— 没依据但有法律信号：**不调用大模型**，返回"资料不足、请补充"，
          引用为空（绝不编条文）；
        * ``general`` —— 预路由命中、或没有法律信号的通用问题：**不注入任何资料**，
          引用恒为空，用常识回答；
        * ``legal``   —— 正常链路：注入检索证据、给引用。

        正文可能带两种前缀（都要让用户看见）：上下文裁剪提示（§D7）与后端降级提示。
        """
        hits, messages, prerouted = self._prepare_or_general(
            question, role, history, top_k, where, memory_block)
        note = self._warmup_note()
        # 防御性清空：clarify 分支**不调用大模型**，因而不会消费裁剪标记；若不清，
        # 上一轮留下的标记会贴到这一轮的答案上（ContextVar 在线程复用时可能串）。
        take_context_trimmed()
        # 预路由命中就**锁定** general：不能让"翻译合同条款"这类话被后置路由判成 clarify 去拒答
        route = "general" if prerouted else self.scope_route(question, hits)
        if route == "clarify":
            return self._clarify_answer(question, role, session_id, note)
        if route == "general":
            # 日常对话：**不注入任何资料**（hits 清空 ⇒ 引用必为空），也不带历史；
            # 但长期记忆照常带上（那是"关于用户本人的事实"，不是上一轮的法律话题）
            messages, hits = build_general_messages(question, memory_block=memory_block), []
        self._note_route(route, question)
        text, provider = self.router.chat(messages, role=role, task="chat")
        text = postprocess(text)
        degraded, reason, fallback_to = self._take_degradation()
        prefix = self._context_trim_prefix() + self._fallback_prefix(
            degraded, reason, fallback_to, note)
        return Answer(
            text=prefix + text if prefix else text,
            citations=build_citations(hits),
            role_id=getattr(role, "role_id", "") or "",
            session_id=session_id,
            provider=provider,
            degraded=degraded,
            degraded_reason=reason,
        )

    def ask_stream(self, question: str, role: Role | None = None, session_id: str = "",
                   user_id: str = "", history: list[Message] | None = None,
                   top_k: int | None = None, where: dict | None = None,
                   memory_block: str = "",
                   ) -> tuple[Answer, Iterator[str]]:
        """流式问答：先返回 Answer 骨架（含引用）与一个文本迭代器。

        迭代过程中会同步累积到 Answer.text，迭代结束后为后处理过的完整文本。
        """
        hits, messages, prerouted = self._prepare_or_general(
            question, role, history, top_k, where, memory_block)
        note = self._warmup_note()
        # 防御性清空：clarify 分支**不调用大模型**，因而不会消费裁剪标记；若不清，
        # 上一轮留下的标记会贴到这一轮的答案上（ContextVar 在线程复用时可能串）。
        take_context_trimmed()
        route = "general" if prerouted else self.scope_route(question, hits)
        if route == "clarify":
            answer = self._clarify_answer(question, role, session_id, note)

            def clarify_stream() -> Iterator[str]:
                yield answer.text

            return answer, clarify_stream()
        if route == "general":
            messages, hits = build_general_messages(question, memory_block=memory_block), []
        self._note_route(route, question)
        provider, iterator = self.router.stream(messages, role=role, task="chat")
        # 降级信息**必须在这里读**：router.stream 已经探过第一个 chunk（降级就发生在那一步），
        # 而生成器本体可能在别的线程里被推进 —— 那时读不到本次请求的上下文。
        degraded, reason, fallback_to = self._take_degradation()
        prefix = self._context_trim_prefix() + self._fallback_prefix(
            degraded, reason, fallback_to, note)
        answer = Answer(
            text="",
            citations=build_citations(hits),
            role_id=getattr(role, "role_id", "") or "",
            session_id=session_id,
            provider=provider,
            degraded=degraded,
            degraded_reason=reason,
        )

        def generator() -> Iterator[str]:
            buffer: list[str] = []
            if prefix:
                yield prefix
            for piece in iterator:
                buffer.append(piece)
                yield piece
            text = postprocess("".join(buffer))
            answer.text = (prefix + text) if prefix else text

        return answer, generator()

    # ---------- 运维 ----------
    # ---------- 预热 ----------
    def warm_up(self, role: Role | None = None) -> dict:
        """启动期预热：把「嵌入冷加载」与「BM25 建索引」的成本挪到用户第一问之前。

        真机实测：首查 7.6–9.3s（bge-m3 冷加载），而 BM25 懒触发时要流式取 11.9 万块
        （约 160s），这期间关键词通道为空、闸门退化、用户会看到「资料还在预热」。
        **必须传真实角色**（默认角色）：BM25 索引按 scope 分签名，用不带 role_id 的
        scope 预热会建出一个运行时用不上的索引，白花 160s。
        """
        scope = self.role_scope(role)
        logger.info("开始预热（scope=%s）", scope)
        result = self.retriever.warm_up(scope)
        result["scope"] = scope
        return result

    def health(self) -> dict:
        """组件健康状态。

        **降级必须可见**：``store`` 是实际生效的后端，``store_requested`` 是配置想要的后端。
        只要两者不一致（或 ``degraded`` 为 true），``/health`` 的调用方就不能只看
        ``status == "ok"`` 就判定真实链路可用 —— 这正是 t7 抓到「服务报 ok 但从未连过
        Milvus」的根因，所以这里把实测后端与降级原因显式摊开。
        """
        actual = self.store_actual or self.store.name
        status = "ok"
        if self.degraded:
            status = "degraded"
        elif actual != str(self.store_requested).strip().lower():
            status = "degraded"
        store_detail: dict = {
            "name": actual,
            "backend": type(self.store).__name__,
            "requested": self.store_requested,
            "actual": actual,
            "degraded": bool(self.degraded),
            "status": status,
        }
        if self.store_primary:
            store_detail["primary"] = self.store_primary
        if self.degraded_reason:
            store_detail["degraded_reason"] = self.degraded_reason
        for attr, key in (("collection", "collection"), ("dim", "dim"),
                          ("uri", "uri"), ("metric_type", "metric_type"),
                          ("host", "host")):
            value = getattr(self.store, attr, None)
            if value not in (None, "", 0):
                store_detail[key] = value
        describe = getattr(self.store, "describe", None)
        if callable(describe):
            try:
                store_detail["describe"] = describe()
            except Exception:  # noqa: BLE001 - 健康检查自身绝不抛
                logger.debug("store.describe() 失败", exc_info=True)
        manifest = getattr(self.store, "manifest", None)
        if callable(manifest):
            try:
                store_detail["manifest"] = manifest()
            except Exception:  # noqa: BLE001
                logger.debug("store.manifest() 失败", exc_info=True)
        deep_health = getattr(self.store, "health", None)
        if callable(deep_health):
            try:
                store_detail["probe"] = deep_health()
            except Exception as exc:  # noqa: BLE001
                store_detail["probe_error"] = f"{type(exc).__name__}: {exc}"

        return {
            "status": status,
            "embedder": self.embedder.name,
            "store": actual,
            "store_actual": actual,
            "store_requested": self.store_requested,
            "degraded": bool(self.degraded),
            "degraded_reason": self.degraded_reason or None,
            "store_detail": store_detail,
            "reranker": self.reranker.name,
            "chunks": self.store.count(),
            "llm_chain": self.router.chain(),
            "llm": self.router.health(),
        }
