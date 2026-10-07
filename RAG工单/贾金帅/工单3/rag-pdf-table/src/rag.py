"""
RAG 问答引擎
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

对应工单功能「问答引擎：能够根据用户问题从PDF文档中提取相关信息并生成回答」。

完整链路：
    Query 理解 → Query 归一化 → 混合检索 → 阈值闸门 → 组装上下文 → LLM 生成（带页码引用）
                ↘ 多子问题：分别检索后按 score 合并去重

两个工程约束：
1) **上下文里必须给页码**。招股说明书问答的答案需要可核对，
   模型被要求在每个事实后标注 [第X页]。前端引用面板也用同一份页码。
2) **检索为空时必须如实说**。闸门拦空后不能让模型自由发挥，
   否则退回「纯 LLM 幻觉」——那正是本工单要对比出来的反面案例。
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

from . import llm
from .config import RETRIEVAL_FINAL_TOP_K, WORK_ORDER_NO  # noqa: F401
from .query_understanding import QueryUnderstanding, understand
from .retriever import RetrievedChunk, build_context, retrieve

logger = logging.getLogger(__name__)

NO_EVIDENCE_TEXT = (
    "资料中没有找到足以回答该问题的内容。"
    "该问题可能超出本项目知识库（《武汉兴图新科电子股份有限公司招股意向书》与"
    "《武汉力源信息技术股份有限公司招股说明书》）的披露范围，"
    "建议换个说法，或确认问题是否属于这两份文档涵盖的内容。"
)

_RAG_SYSTEM = """你是本项目知识库的专业问答助手。知识库包含两份文件：
《武汉兴图新科电子股份有限公司招股意向书》（下称「兴图新科」）与
《武汉力源信息技术股份有限公司招股说明书》（下称「力源信息」）。

严格遵守以下规则：
1. **只依据提供的资料回答**，不得使用你自己的先验知识补充任何数字、名称或结论。
2. **先看来源文档**：每个片段开头都写了「来源：……第X页」。两家的数据完全不同，
   回答**必须**说明是**哪家公司**的数据，绝不允许把 A 公司的数字安到 B 公司头上。
3. 资料不足或与问题无关时，直接回答"资料中没有相关内容"，绝不猜测。
4. 答案**简洁**：先给结论，再给必要的数据与口径说明。总长度控制在 250 字以内。
   **绝对不要**描述检索过程，不要出现"资料片段""【片段1】""根据上述资料"这类字样，
   也不要罗列与问题无关的片段信息。
5. 引用格式：在每个关键事实后标注来源页码，形如 `[第123页]`。多个来源写 `[第123页][第125页]`。
   **页码必须与片段开头的「来源：…第X页」完全一致**：片段写「第342-343页」就照写该范围，
   不要只取其中一个数、也不要自己推断或统一改写页码。
6. 涉及金额、比例、年份时，**逐字照抄**原文数字与单位，不要换算、不要四舍五入。
7. 若资料中同时存在多个期间的数据（如2017年度、2018年度、2019年1-6月），**必须分别列出**，
   不要只挑一个。可以用短列表呈现。
8. 表格类问题（募投项目、关联方、发行情况）**必须逐行列全**，不要把多行并成一句话，
   也不要用"等"字省略。列名与值要一一对应（表格资料里已写成「列名：值」的形态）。
9. **必须带上资料中的限定条件、例外与说明**，并且**分清获奖/获批主体**：
   涉及"参与/获奖/入选/中标"时，要写明是**工程或项目本身**获奖，还是**发行人**获奖。
   若发行人只是参与者（例如工程由某研究所牵头承担、发行人未列入获奖名单），
   **禁止**使用"发行人……荣获/获得……"这类句式（即使工程确实获奖，那样写也会被读成发行人获奖）。
   正确写法是先说工程获奖、再说"发行人参与该工程，但未列入获奖名单"，
   且该限定**必须出现在答案第一句**，不能放到末尾当补充。
10. 使用与问题相同的语言回答（中文问→中文答，英文问→英文答）。"""

_LLM_ONLY_SYSTEM = """你是本项目知识库的专业问答助手，知识库包含《武汉兴图新科电子股份有限公司招股意向书》与《武汉力源信息技术股份有限公司招股说明书》两份文件。
请直接回答问题。使用与问题相同的语言回答。"""


@dataclass
class Answer:
    """一次问答的完整产出。"""

    question: str
    answer: str
    mode: str = "rag"                      # rag | llm_only | no_evidence
    understanding: QueryUnderstanding | None = None
    citations: list[dict] = field(default_factory=list)
    retrieval: dict = field(default_factory=dict)
    timing: dict = field(default_factory=dict)
    # 真正喂给生成模型的上下文原文（含「来源：…第X页」表头）。
    # 评估时必须用这一份，而不是只用引用片段的正文 —— 否则评审模型看不到页码，
    # 会把「[第479页]」这类**有依据**的引用判成 unsupported（实测 faithfulness 因此
    # 从 1.0 掉到 0.75，属于评测口径问题，不是生成质量问题）。
    context_text: str = ""

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "mode": self.mode,
            "understanding": self.understanding.to_dict() if self.understanding else None,
            "citations": self.citations,
            "retrieval": self.retrieval,
            "timing": self.timing,
            "context_text": self.context_text,
        }


# ------------------------------------------------------------------ 多子问题合并


def retrieve_for(qu: QueryUnderstanding, top_k: int = RETRIEVAL_FINAL_TOP_K):
    """
    按 Query 理解结果执行检索。返回 (items, info)。

    **主查询优先，子查询只补位**（这是一个踩过坑的修正）：

    最初的写法是「主查询 + 全部子查询各自召回 → 合并 → 按分数重排 → 取 top_k」。
    实测这样做会翻车：模型把「报告期内来自军用领域的收入分别是多少」拆成了
    「报告期第一年/第二年/第三年来自军用领域的收入」——文档里根本没有"第一年"
    这种说法，这三路召回的全是噪音，且噪音分（0.99）压过了主查询正确答案（0.97），
    正确答案被挤出上下文，答案变成"资料中没有相关内容"。

    现在的规则：
      1) 主查询（改写后的检索式）的结果**整体占位**，不受子查询影响；
      2) 子查询只用于**填补主查询没占满的名额**，且分数必须不低于主查询的最低分；
      3) 最多再引入 top_k/2 条，最终仍截断到 top_k —— 上下文预算不能被子查询撑爆。
    """
    main = retrieve(qu.retrieval_query, top_k=top_k, override_query=qu.retrieval_query)
    items = list(main.items)
    seen = {it.chunk_id for it in items}

    sub_queries = [q for q in (qu.sub_questions or []) if q and q != qu.retrieval_query][:2]
    used = [qu.retrieval_query]

    if len(items) < top_k and sub_queries:
        floor = min((it.score for it in items), default=0.0)
        extra: list[RetrievedChunk] = []
        for sq in sub_queries:
            try:
                r = retrieve(sq, top_k=top_k, override_query=sq)
            except Exception as exc:  # noqa: BLE001
                logger.warning("子查询检索失败（%s）：%s", sq, exc)
                continue
            used.append(sq)
            for it in r.items:
                # 只在主查询没找到更强证据时补位
                if it.chunk_id not in seen and it.score >= floor:
                    extra.append(it)
                    seen.add(it.chunk_id)
        extra.sort(key=lambda x: -x.score)
        items = items + extra[: max(1, top_k // 2)]

    items = sorted(items, key=lambda x: -x.score)[:top_k]

    if not items:
        return [], {
            "gated": True,
            "trace": main.trace,
            "used_query": main.used_query,
            "normalized_query": main.normalized_query,
        }

    return items, {
        "gated": False,
        "trace": main.trace,
        "used_query": main.used_query,
        "normalized_query": main.normalized_query,
        "queries_used": used,
    }


def build_citations(items: list[RetrievedChunk]) -> list[dict]:
    return [
        {
            "index": i,
            "chunk_id": it.chunk_id,
            "doc": it.doc,
            "doc_key": it.doc_key,
            "page": it.page,
            "page_end": it.page_end,
            "section": it.section,
            "type": it.type,
            "score": round(it.score, 4),
            "evidence": round(it.evidence, 4),
            "text": it.text,
        }
        for i, it in enumerate(items, 1)
    ]


# ------------------------------------------------------------------ RAG 回答


def answer(question: str, top_k: int = RETRIEVAL_FINAL_TOP_K) -> Answer:
    """非流式：完整跑一遍 RAG 链路。"""
    t_start = time.perf_counter()

    qu = understand(question)

    if not qu.needs_retrieval:
        return Answer(
            question=question,
            answer="你好，我是这份招股意向书的问答助手。请提出与文档内容相关的问题，例如"
                   "「武汉兴图新科电子股份有限公司的注册资本是多少？」",
            mode="no_evidence",
            understanding=qu,
            timing={"total_ms": round((time.perf_counter() - t_start) * 1000, 2)},
        )

    t = time.perf_counter()
    items, rinfo = retrieve_for(qu, top_k=top_k)
    retrieval_ms = (time.perf_counter() - t) * 1000

    if not items:
        return Answer(
            question=question,
            answer=NO_EVIDENCE_TEXT,
            mode="no_evidence",
            understanding=qu,
            retrieval=rinfo,
            timing={
                "query_understanding_ms": round(qu.ms, 2),
                "retrieval_ms": round(retrieval_ms, 2),
                "total_ms": round((time.perf_counter() - t_start) * 1000, 2),
            },
        )

    contexts = build_context(items, max_chars=4800)  # 控制提示长度：上下文越长，首字延迟越高
    t = time.perf_counter()
    text = llm.chat(
        [
            {"role": "system", "content": _RAG_SYSTEM},
            {
                "role": "user",
                "content": f"【资料片段】\n{contexts}\n\n【问题】\n{question}\n\n请依据上述资料片段回答。",
            },
        ]
    )
    gen_ms = (time.perf_counter() - t) * 1000

    return Answer(
        question=question,
        answer=text.strip(),
        mode="rag",
        understanding=qu,
        citations=build_citations(items),
        retrieval=rinfo,
        timing={
            "query_understanding_ms": round(qu.ms, 2),
            "retrieval_ms": round(retrieval_ms, 2),
            "generation_ms": round(gen_ms, 2),
            "total_ms": round((time.perf_counter() - t_start) * 1000, 2),
        },
        context_text=contexts,
    )


# ------------------------------------------------------------------ 纯 LLM 基线


def answer_llm_only(question: str) -> Answer:
    """
    对照组：不检索，直接问 LLM。
    工单明确要求「对比基于pdf的返回结果和只使用LLM返回的答案的对比分析」。
    """
    t0 = time.perf_counter()
    try:
        text = llm.chat(
            [
                {"role": "system", "content": _LLM_ONLY_SYSTEM},
                {"role": "user", "content": question},
            ]
        )
    except Exception as exc:  # noqa: BLE001
        text = f"（LLM 调用失败：{exc}）"
    return Answer(
        question=question,
        answer=text.strip(),
        mode="llm_only",
        timing={"total_ms": round((time.perf_counter() - t0) * 1000, 2)},
    )


# ------------------------------------------------------------------ 流式回答

_PAGE_RE = re.compile(r"\[第\s*\d+(?:\s*[-–~]\s*\d+)?\s*页\]")


def answer_stream(question: str, top_k: int = RETRIEVAL_FINAL_TOP_K):
    """
    流式生成。产出事件元组 (event, payload)：

      ("stage",   {"name": "understand"|"retrieve"|"generate", "status": "running"|"done", ...})
      ("meta",    {...})          # 不含引用，引用必须排在正文之后
      ("delta",   "文本增量")
      ("citations", [...])
      ("done",    {"answer": 完整答案, ...})

    帧序是产品要求，不要改：stage→meta→delta*→citations→done。
    引用排在正文之后，用户才能「先读到答案，再看依据」。
    """
    t_start = time.perf_counter()

    yield ("stage", {"name": "understand", "status": "running"})
    qu = understand(question)
    yield (
        "stage",
        {
            "name": "understand",
            "status": "done",
            "ms": round(qu.ms, 2),
            "detail": qu.to_dict(),
        },
    )

    if not qu.needs_retrieval:
        msg = (
            "你好，我是这份招股意向书的问答助手。请提出与文档内容相关的问题，例如"
            "「武汉兴图新科电子股份有限公司的注册资本是多少？」"
        )
        yield ("meta", {"mode": "no_evidence"})
        yield ("delta", msg)
        yield ("done", {"answer": msg, "mode": "no_evidence", "timing": _timing(t_start, qu, 0, 0)})
        return

    yield ("stage", {"name": "retrieve", "status": "running"})
    t = time.perf_counter()
    items, rinfo = retrieve_for(qu, top_k=top_k)
    retrieval_ms = (time.perf_counter() - t) * 1000
    yield (
        "stage",
        {
            "name": "retrieve",
            "status": "done",
            "ms": round(retrieval_ms, 2),
            "detail": rinfo,
        },
    )

    if not items:
        yield ("meta", {"mode": "no_evidence"})
        yield ("delta", NO_EVIDENCE_TEXT)
        yield (
            "done",
            {
                "answer": NO_EVIDENCE_TEXT,
                "mode": "no_evidence",
                "timing": _timing(t_start, qu, retrieval_ms, 0),
            },
        )
        return

    contexts = build_context(items, max_chars=4800)
    yield ("meta", {"mode": "rag", "context_chars": len(contexts)})
    yield ("stage", {"name": "generate", "status": "running"})

    t = time.perf_counter()
    acc: list[str] = []
    first_token_ms = None
    try:
        for kind, piece in llm.chat_stream(
            [
                {"role": "system", "content": _RAG_SYSTEM},
                {
                    "role": "user",
                    "content": f"【资料片段】\n{contexts}\n\n【问题】\n{question}\n\n请依据上述资料片段回答。",
                },
            ]
        ):
            if kind == "__thinking__":
                yield ("thinking", piece)
                continue
            if first_token_ms is None:
                first_token_ms = (time.perf_counter() - t) * 1000
            acc.append(piece)
            yield ("delta", piece)
    except Exception as exc:  # noqa: BLE001
        logger.warning("流式生成失败：%s", exc)
        fallback = "（生成中断，请稍后重试）"
        acc.append(fallback)
        yield ("delta", fallback)

    gen_ms = (time.perf_counter() - t) * 1000
    full = "".join(acc)
    yield (
        "stage",
        {
            "name": "generate",
            "status": "done",
            "ms": round(gen_ms, 2),
            "first_token_ms": round(first_token_ms, 2) if first_token_ms else None,
            "answer_chars": len(full),
        },
    )

    # 引用**必须**排在正文之后
    yield ("citations", build_citations(items))
    yield (
        "done",
        {
            "answer": full,
            "mode": "rag",
            "cited_pages": sorted({int(p) for p in _PAGE_RE.findall(full) for p in re.findall(r"\d+", p)}),
            "timing": _timing(t_start, qu, retrieval_ms, gen_ms),
        },
    )


def _timing(t_start: float, qu: QueryUnderstanding, retrieval_ms: float, gen_ms: float) -> dict:
    return {
        "query_understanding_ms": round(qu.ms, 2),
        "retrieval_ms": round(retrieval_ms, 2),
        "generation_ms": round(gen_ms, 2),
        "total_ms": round((time.perf_counter() - t_start) * 1000, 2),
    }
