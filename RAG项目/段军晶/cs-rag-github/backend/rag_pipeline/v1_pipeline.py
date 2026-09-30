# -*- coding: utf-8 -*-
"""
V1 基线问答链路（MVP）

链路（朴素稠密检索）：
    ① 会话管理    —— 从 Redis 读取该会话的历史对话
    ② 问题向量化  —— BGE-M3 编码为 1024 维稠密向量
    ③ 稠密检索    —— 在 Milvus 中做 Top-K 相似度检索
    ④ 元数据回填  —— 按 chunk_id 从 MySQL 取回文件名与页码（溯源真相源）
    ⑤ 提示词组装  —— 系统约束 + 历史对话 + 检索原文片段
    ⑥ 答案生成    —— 调用 deepseek-v4-flash
    ⑦ 溯源构造    —— (源文件名, 页码) 去重排序，与答案一并返回

★ 页码溯源贯穿第 ④ 与第 ⑦ 步 ★
    向量库只告诉我们「哪些块相关」，文件名与页码一律回 MySQL 取回。
    这是 V1 必须守住、且 V2/V3 不得降级的硬性要求。

Trace 约定：
    链路每一步的中间结果（命中块、相似度分数、命中页码、耗时）
    通过 logging 写入服务端日志，**不进入 HTTP 响应体**，
    前端也不展示任何检索链路细节。
"""

from __future__ import annotations

import time
from typing import Any, Dict, Iterator, List, Optional

from backend.config import settings
from backend.db import milvus_client, mysql, redis_client
from backend.embedder import get_embedder
from backend.llm_client import ChatLLM, LLMError, get_chat_llm
from backend.logging_config import get_logger, log_retrieval_trace, set_session_id

logger = get_logger(__name__)


# ===========================================================================
# 提示词
# ===========================================================================

# 系统提示词：本项目的防幻觉底线（对应需求 N2 与 ADR-012）。
# 用户拿答案去投标和应对审计，编造的标准条款会造成实际业务损失，
# 因此这里的约束是硬性的，V2/V3 迭代时不得放宽。
SYSTEM_PROMPT = """你是企业标准知识库的问答助手，服务于需要查阅国家标准的工程师与合规人员。

请严格遵守以下要求：

1. 只依据【参考资料】中的内容回答问题，绝对不要使用你自己的知识进行补充、推测或延伸。
2. 如果【参考资料】中没有回答问题所需的信息，必须明确回答：「根据现有知识库，未找到相关内容。」
   此时不要编造任何条款内容、数值或出处。
3. 回答要准确、简洁、条理清晰。涉及多项要求或等级划分时，使用分点列出。
4. 涉及具体数值、等级、时限、比例等关键信息时，必须与参考资料原文保持一致，
   不得改写、换算或近似。
5. 不要在回答中自行撰写「依据某某文件第几页」这类引用，出处由系统在答案下方单独展示。
6. 使用中文回答，语言平实专业。"""

# ---------------------------------------------------------------------------
# 身份提示词（增量改造：问答页可切换「学生身份 / 职场身份」）
# ---------------------------------------------------------------------------
# 设计原则（与上面 SYSTEM_PROMPT 的关系）：
#   1. 上面那份 SYSTEM_PROMPT 保留不动，作为默认值继续可用 —— 任何旧调用方
#      （评测脚本、curl、JMeter 压测）不传 role 时行为与改造前完全一致。
#   2. 两套身份提示词的防幻觉红线**逐字相同**：只依据【参考资料】、未找到即明说、
#      数值等级不得改写换算近似、不自行撰写引用。差异只在语气、术语与结构要求，
#      因此「事实与数值」不受身份影响 —— 事实一致，只是讲法不同。
#   3. 不为「轻松愉悦」开举生活例子的口子 —— 一举例必然动用模型自身知识，
#      直接击穿第 1 条防幻觉底线。
ROLE_PROMPTS: Dict[str, str] = {
    "student": """你是「智查 AI」知识库助手，正在以【学生模式】和一位在校学生交流。

语气要求：轻松、亲切、有耐心，像学长学姐讲题那样好懂。可以用「咱们」「其实不难」「这一点记住就行」
这样的口吻，句子别太长，重要结论可以单独成行。

但下面这些底线一条都不能松：

1. 只依据【参考资料】中的内容回答问题，绝对不要使用你自己的知识进行补充、推测或延伸。
2. 如果【参考资料】中没有回答问题所需的信息，必须明确回答：「根据现有知识库，未找到相关内容。」
   此时不要编造任何条款内容、数值或出处。
3. 答案比较长时，先用一句话把结论说清楚，再分点展开，让人一眼抓住重点。
4. 涉及具体数值、等级、时限、比例等关键信息时，必须与参考资料原文保持一致，
   不得改写、换算或近似。
5. 不要在回答中自行撰写「依据某某文件第几页」这类引用，出处由系统在答案下方单独展示。
6. 使用中文回答。""",

    "workplace": """你是「智查 AI」知识库助手，正在以【职场模式】为工程师、合规人员等专业读者提供服务。

语气要求：严谨、简洁、权威，直接给结论和依据要点，不寒暄、不打比方、不使用调侃语气，
不出现「咱们」「其实」这类口语化表达。

请严格遵守以下要求：

1. 只依据【参考资料】中的内容回答问题，绝对不要使用你自己的知识进行补充、推测或延伸。
2. 如果【参考资料】中没有回答问题所需的信息，必须明确回答：「根据现有知识库，未找到相关内容。」
   此时不要编造任何条款内容、数值或出处。
3. 回答要准确、简洁、条理清晰。涉及多项要求、等级划分或流程步骤时，使用分点列出，
   并沿用参考资料中的规范术语，不要替换为同义表述。
4. 涉及具体数值、等级、时限、比例等关键信息时，必须与参考资料原文保持一致，
   不得改写、换算或近似。
5. 参考资料中若给出适用范围、适用条件或例外情形，应一并说明，不要只摘取结论。
6. 不要在回答中自行撰写「依据某某文件第几页」这类引用，出处由系统在答案下方单独展示。
7. 使用中文回答。""",
}

# 默认身份：非法 / 缺失的 role 一律回退到该身份，只回退、不报错
DEFAULT_ROLE = "student"


def normalize_role(role: Optional[str]) -> str:
    """
    归一化身份入参。

    只认 student / workplace；大小写与首尾空白自动规整；
    非法值（None、空串、未知字符串）**回退默认身份而不报错** ——
    演示现场参数传错不会让整个问答挂掉。
    """
    key = (role or "").strip().lower()
    return key if key in ROLE_PROMPTS else DEFAULT_ROLE


# 参考资料模板：每段原文都明确标注来源文件与页码，

# 让生成模型清楚知道每段内容的出处，减少张冠李戴。
CONTEXT_TEMPLATE = """【资料{index}】来源：《{file_name}》 第 {page_no} 页
{content}"""

USER_TEMPLATE = """【参考资料】
{context}

【用户问题】
{question}"""

NO_RESULT_ANSWER = "根据现有知识库，未找到相关内容。"

# 生成失败的兜底文案（单独常量化是为了能判断"这次到底答没答出来"）。
# 关键约束：这段文案**绝不写入查询缓存**。
# 原因：生成失败是临时故障（模型超时、输出被长度截断等），
# 若把这段提示缓存 1 小时，用户在这一小时内反复提问都会命中这份"错误答案"，
# 表现成"服务彻底坏了、怎么问都是同一句"，而实际检索链路完全正常。
LLM_FAILED_ANSWER = (
    "抱歉，答案生成服务暂时不可用，请稍后重试。\n"
    "（检索已命中相关原文，可在下方来源中查看）"
)


# ===========================================================================
# V1 链路
# ===========================================================================

class V1Pipeline:
    """V1 基线问答链路（朴素稠密检索）"""

    PIPELINE_NAME = "v1"

    def __init__(self, llm: Optional[ChatLLM] = None) -> None:
        self.llm = llm or get_chat_llm()

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def answer(
        self,
        question: str,
        session_id: Optional[str] = None,
        *,
        top_k: Optional[int] = None,
        use_cache: bool = True,
        role: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        回答一个问题，返回答案与溯源信息。

        返回结构（对前端稳定，V2/V3 只允许新增字段，不得删除或改名）：
            {
              "session_id": str,
              "answer":     str,
              "sources":    [{file_name, page_no, page_nums, chunk_id,
                              score, summary, content_type}, ...],
              "pipeline":   "v1",
              "latency_ms": int,
              "cached":     bool
            }
        """
        started = time.time()
        set_session_id(session_id)

        # 身份归一化（增量改造）：非法值回退默认身份，不报错
        role = normalize_role(role)

        # ---- ① 会话管理 ----
        if not session_id:
            session_id = redis_client.create_session()
        history = redis_client.get_history(session_id)

        # ---- 缓存：重复问题直接返回（F8）----
        if use_cache:
            cached = redis_client.get_cached_answer(question, role=role)
            if cached:
                latency = int((time.time() - started) * 1000)
                logger.info("命中查询缓存 | 问题=%r | 耗时=%dms", question, latency)
                cached["session_id"] = session_id
                cached["cached"] = True
                cached["latency_ms"] = latency
                # 缓存命中也要写入会话历史，保证多轮对话连续
                redis_client.append_turn(
                    session_id,
                    question=question,
                    answer=cached.get("answer", ""),
                    sources=cached.get("sources", []),
                )
                return cached

        top_k = top_k or settings.retrieve_top_k

        # ---- ②③④ 检索：向量化 -> 稠密检索 -> 元数据回填 ----
        ordered = self.retrieve(question, top_k=top_k)

        if not ordered:
            logger.warning("检索无结果 | 问题=%r | 知识库可能尚未构建或元数据缺失", question)
            return self._finish(
                session_id=session_id, question=question,
                answer=NO_RESULT_ANSWER, sources=[], started=started,
                use_cache=use_cache, retrieved=[], role=role,
            )

        # ---- ⑤ 提示词组装 ----
        context = self._build_context(ordered)
        messages = self._build_messages(history, context, question, role)

        # ---- ⑥ 答案生成 ----
        try:
            answer_text = self.llm.chat(messages)
        except LLMError as exc:
            # 生成失败也要给出可理解的中文提示，且不能编造内容
            logger.error("答案生成失败：%s", exc)
            answer_text = LLM_FAILED_ANSWER

        # ---- ⑦ 溯源构造 ----
        sources = self._build_sources(ordered)

        return self._finish(
            session_id=session_id, question=question, answer=answer_text,
            sources=sources, started=started, use_cache=use_cache,
            retrieved=ordered, role=role,
        )

    # ------------------------------------------------------------------
    # 流式回答（新增，与 answer() 并行；检索与溯源逻辑完全复用）
    # ------------------------------------------------------------------

    def answer_stream(
        self,
        question: str,
        session_id: Optional[str] = None,
        *,
        top_k: Optional[int] = None,
        role: Optional[str] = None,
        use_cache: bool = True,
    ) -> Iterator[Dict[str, Any]]:
        """
        流式回答：先给溯源，再逐段吐答案。

        与 answer() 的关系（重要）：
            answer() 保持原样，一行未改 —— 非流式调用方（评测脚本、JMeter 压测、
            /api/chat/query 接口）行为完全不变。
            本方法只是把「生成」那一步由一次性调用换成流式调用，
            检索（retrieve）、上下文拼装（_build_context）、提示词组装（_build_messages，
            含 role 身份切换）、溯源构造（_build_sources）、收尾（_finish：写会话历史、
            写缓存、写问答记录）**全部复用同一套方法**，因此流式与非流式的事实、
            溯源、缓存规则完全一致。

        产出的事件（前端按 type 分发）：
            {"type": "meta",  "session_id", "sources", "role", "pipeline", "cached"}
                —— 检索已完成，先把溯源给前端（用户立刻看到"命中了哪几页"）
            {"type": "delta", "text": "..."}      —— 答案增量
            {"type": "done",  "session_id", "answer", "latency_ms", "cached"}
                —— 收尾（历史与缓存已在 _finish 里落好）
        """
        started = time.time()
        set_session_id(session_id)
        role = normalize_role(role)

        # ---- ① 会话管理 ----
        if not session_id:
            session_id = redis_client.create_session()
        history = redis_client.get_history(session_id)

        # ---- 缓存命中：直接一次性给完（与流式的对外行为保持一致）----
        if use_cache:
            cached = redis_client.get_cached_answer(question, role=role)
            if cached:
                latency = int((time.time() - started) * 1000)
                cached["session_id"] = session_id
                cached["cached"] = True
                cached["latency_ms"] = latency
                redis_client.append_turn(
                    session_id,
                    question=question,
                    answer=cached.get("answer", ""),
                    sources=cached.get("sources", []),
                )
                logger.info("命中查询缓存（流式接口）| 问题=%r | 耗时=%dms", question, latency)
                yield {
                    "type": "meta", "session_id": session_id,
                    "sources": cached.get("sources", []), "role": cached.get("role", role),
                    "pipeline": self.PIPELINE_NAME, "cached": True,
                }
                yield {"type": "delta", "text": cached.get("answer", "")}
                yield {
                    "type": "done", "session_id": session_id,
                    "answer": cached.get("answer", ""),
                    "latency_ms": latency, "cached": True,
                }
                return

        top_k = top_k or settings.retrieve_top_k

        # ---- ②③④ 检索：与 answer() 调的是同一个方法 ----
        ordered = self.retrieve(question, top_k=top_k)

        if not ordered:
            logger.warning("检索无结果（流式）| 问题=%r | 知识库可能尚未构建", question)
            result = self._finish(
                session_id=session_id, question=question,
                answer=NO_RESULT_ANSWER, sources=[], started=started,
                use_cache=use_cache, retrieved=[], role=role,
            )
            yield {
                "type": "meta", "session_id": session_id, "sources": [],
                "role": role, "pipeline": self.PIPELINE_NAME, "cached": False,
            }
            yield {"type": "delta", "text": NO_RESULT_ANSWER}
            yield {
                "type": "done", "session_id": session_id, "answer": NO_RESULT_ANSWER,
                "latency_ms": result.get("latency_ms", 0), "cached": False,
            }
            return

        # ---- ⑤ 提示词组装（含身份切换，与 answer() 同一套）----
        context = self._build_context(ordered)
        messages = self._build_messages(history, context, question, role)

        # ---- ⑦ 溯源构造（先算好，随 meta 一起发给前端）----
        sources = self._build_sources(ordered)
        yield {
            "type": "meta", "session_id": session_id, "sources": sources,
            "role": role, "pipeline": self.PIPELINE_NAME, "cached": False,
        }

        # ---- ⑥ 流式生成 ----
        pieces: List[str] = []
        failed = False
        try:
            for item in self.llm.chat_stream(messages):
                if item.get("kind") == "content":
                    pieces.append(item.get("text") or "")
                    yield {"type": "delta", "text": item.get("text") or ""}
                elif item.get("kind") == "reasoning":
                    # 思考进度：模型在推理、正文尚未开始，把这个进度透给前端
                    yield {"type": "thinking", "chars": item.get("chars") or 0}
        except LLMError as exc:
            logger.error("流式生成失败：%s", exc)
            failed = True

        answer = "".join(pieces).strip()
        if failed or not answer:
            # 生成失败：给可理解的提示；_finish 里的缓存规则会自动跳过这段文案
            answer = LLM_FAILED_ANSWER
            result = self._finish(
                session_id=session_id, question=question, answer=answer,
                sources=sources, started=started, use_cache=use_cache,
                retrieved=ordered, role=role,
            )
            yield {"type": "delta", "text": answer}
            yield {
                "type": "done", "session_id": session_id, "answer": answer,
                "latency_ms": result.get("latency_ms", 0), "cached": False,
            }
            return

        # ---- 收尾：写会话历史 / 写缓存 / 写问答记录（与 answer() 同一个方法）----
        result = self._finish(
            session_id=session_id, question=question, answer=answer,
            sources=sources, started=started, use_cache=use_cache,
            retrieved=ordered, role=role,
        )
        yield {
            "type": "done", "session_id": session_id, "answer": answer,
            "latency_ms": result.get("latency_ms", 0), "cached": False,
        }

    # ------------------------------------------------------------------
    # 检索环节（②③④ 三步，可独立调用）
    # ------------------------------------------------------------------

    def retrieve(
        self,
        question: str,
        *,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        执行检索，返回带页码的命中列表，**不调用生成模型**。

        单独抽出来的用途：
            评测脚本需要独立衡量「检索能力」（recall@k / MRR）。
            若把检索与生成混在一起衡量，指标会被生成质量干扰，
            无法判断问题究竟出在检索环节还是生成环节。

        返回：[{chunk_id, score, file_name, page_no, page_nums,
                content, content_type, section_title}, ...]
        """
        top_k = top_k or settings.retrieve_top_k

        # ---- ② 问题向量化 ----
        # 把用户的自然语言问题编码成 1024 维稠密向量。
        # 这是 V1 唯一的检索输入（V2 会额外再要一份稀疏权重）。
        query_vector = get_embedder().encode_query(question)
        if not query_vector:
            raise RuntimeError("问题向量化失败，请检查 BGE-M3 模型是否可用")

        # ---- ③ 稠密检索 ----
        # 拿问题向量去 Milvus 里找最相似的 top_k 个块（余弦相似度）。
        hits = milvus_client.search_dense(query_vector, top_k=top_k)
        # 检索链路细节只写服务端日志，不进 HTTP 响应体（对齐接口边界要求）。
        log_retrieval_trace(
            logger,
            stage="v1_dense_retrieve",
            question=question,
            hits=hits,
            extra={"top_k": top_k, "collection": settings.milvus_collection},
        )
        # 一个都没命中：返回空列表，上层会回答「未找到相关内容」。
        if not hits:
            return []

        # ---- ④ 元数据回填（页码溯源的真相源）----
        # Milvus 只知道"哪些块相关"，至于这些块出自哪份文件、第几页，
        # 一律回 MySQL 查 —— MySQL 才是页码溯源的唯一真相源。
        chunk_ids = [h["chunk_id"] for h in hits]
        chunk_map = {c["chunk_id"]: c for c in mysql.fetch_chunks_by_ids(chunk_ids)}

        ordered: List[Dict[str, Any]] = []
        # 按检索名次逐条组装结果（返回顺序即相似度从高到低）。
        for hit in hits:
            meta = chunk_map.get(hit["chunk_id"])
            if not meta:
                # 向量库与 MySQL 不一致（如 MySQL 被清空），跳过并告警
                logger.warning("chunk 元数据缺失（MySQL 中不存在）：%s", hit["chunk_id"])
                continue
            ordered.append({
                "chunk_id": hit["chunk_id"],
                # 这里是 V1 的余弦相似度。量纲与 V2/V3 的 RRF 融合分不同，
                # 跨版本比较时不能直接拿 score 的绝对值说话。
                "score": hit["score"],
                # ★ 页码与文件名一律以 MySQL 为准 ★
                "file_name": meta.get("source_file") or meta.get("doc_id", ""),
                "page_no": int(meta.get("page_no") or 1),
                "page_nums": meta.get("page_nums") or [int(meta.get("page_no") or 1)],
                "content": meta.get("content", ""),
                "content_type": meta.get("content_type", "text"),
                "section_title": meta.get("section_title", ""),
            })
        return ordered

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _build_context(self, ordered: List[Dict[str, Any]]) -> str:
        """把检索到的原文片段拼装成参考资料，每段标注来源文件与页码"""
        blocks: List[str] = []
        for index, item in enumerate(ordered, start=1):
            blocks.append(CONTEXT_TEMPLATE.format(
                index=index,
                file_name=item["file_name"],
                page_no=item["page_no"],
                content=item["content"].strip(),
            ))
        return "\n\n".join(blocks)

    def _build_messages(
        self,
        history: List[Dict[str, Any]],
        context: str,
        question: str,
        role: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        组装对话消息。

        历史轮次以 user/assistant 交替的形式放在参考资料之前，
        使模型能理解「那第 4 级呢」这类省略主语的追问。
        """
        # ★ 增量改造的唯一注入点：只替换 system 那一条 ★
        # 参考资料、会话历史、用户问题的组装方式完全不变，
        # 检索 / 分块 / 重排 / 改写 / 溯源链路一行未动。
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": ROLE_PROMPTS[normalize_role(role)]}
        ]

        for turn in history:
            prev_q = (turn.get("question") or "").strip()
            prev_a = (turn.get("answer") or "").strip()
            if prev_q:
                messages.append({"role": "user", "content": prev_q})
            if prev_a:
                messages.append({"role": "assistant", "content": prev_a})

        messages.append({
            "role": "user",
            "content": USER_TEMPLATE.format(context=context, question=question),
        })
        return messages

    @staticmethod
    def _build_sources(ordered: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        构造前端展示用的溯源列表。

        按 (文件名, 主页码) 去重：同一个位置命中多个 chunk 时只保留分数最高的，
        避免前端出现重复条目；同时保留完整页码列表与命中摘要。
        """
        best: Dict[tuple, Dict[str, Any]] = {}
        for item in ordered:
            key = (item["file_name"], item["page_no"])
            if key not in best or item["score"] > best[key]["score"]:
                best[key] = item

        sources: List[Dict[str, Any]] = []
        for item in sorted(best.values(), key=lambda x: -x["score"]):
            content = (item["content"] or "").strip().replace("\n", " ")
            sources.append({
                "file_name": item["file_name"],
                "page_no": item["page_no"],
                "page_nums": item["page_nums"],
                "chunk_id": item["chunk_id"],
                "score": round(float(item["score"]), 4),
                "summary": content[:120] + ("…" if len(content) > 120 else ""),
                "content_type": item["content_type"],
                "section_title": item.get("section_title", ""),
            })
        return sources

    def _finish(
        self,
        *,
        session_id: str,
        question: str,
        answer: str,
        sources: List[Dict[str, Any]],
        started: float,
        use_cache: bool,
        retrieved: List[Dict[str, Any]],
        role: Optional[str] = None,
    ) -> Dict[str, Any]:
        """收尾：写会话历史、写缓存、写问答记录、返回响应体"""
        latency_ms = int((time.time() - started) * 1000)
        role = normalize_role(role)

        # 拒答时不展示来源。
        # 原因：检索环节可能召回了一些低相关片段（分数低于阈值但仍在 Top-K 中），
        # 若在「未找到相关内容」下方仍列出这些来源，用户会困惑「究竟找到没有」。
        # 语义一致性优先于「多给点信息」。
        if answer == NO_RESULT_ANSWER:
            sources = []

        payload: Dict[str, Any] = {
            "session_id": session_id,
            "answer": answer,
            "sources": sources,
            "pipeline": self.PIPELINE_NAME,
            "latency_ms": latency_ms,
            "cached": False,
            # 新增字段（增量改造）：本次回答使用的身份，前端历史记录据此标注
            "role": role,
        }

        # 会话历史（多轮对话记忆）
        redis_client.append_turn(
            session_id, question=question, answer=answer, sources=sources)

        # 只缓存"真正作答"的结果：拒答不缓存（避免知识库更新后仍返回旧拒答），
        # 生成失败也不缓存（否则临时故障会在缓存 TTL 内被反复复现）
        if use_cache and answer not in (NO_RESULT_ANSWER, LLM_FAILED_ANSWER) and sources:
            redis_client.set_cached_answer(question, payload, role=role)

        # 问答记录（仅后端排查用，不对外提供查询接口）
        mysql.insert_qa_record(
            session_id=session_id,
            question=question,
            answer=answer,
            sources=sources,
            retrieved=[
                {"chunk_id": r["chunk_id"], "score": r["score"], "page_no": r["page_no"]}
                for r in retrieved
            ],
            pipeline=self.PIPELINE_NAME,
            latency_ms=latency_ms,
        )

        logger.info(
            "V1 问答完成 | 问题=%r | 身份=%s | 来源数=%d | 耗时=%dms | 答案长度=%d",
            question, role, len(sources), latency_ms, len(answer),
        )
        return payload


# ---------------------------------------------------------------------------
# 单例
# ---------------------------------------------------------------------------

_pipeline: Optional[V1Pipeline] = None


def get_pipeline() -> V1Pipeline:
    """获取 V1 链路单例"""
    global _pipeline
    if _pipeline is None:
        _pipeline = V1Pipeline()
    return _pipeline


def answer_question(
    question: str,
    session_id: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """便捷函数：直接调用 V1 链路回答问题"""
    return get_pipeline().answer(question, session_id, **kwargs)
