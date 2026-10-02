"""在线法律问答的 RAG 总流程。

一轮提问进入这里后会依次经历：

``读取记忆 -> 理解问题 -> 规划检索 -> 并行查私有/公共/网页资料``
``-> 融合与重排 -> 构建模型上下文 -> 生成答案 -> 保存本轮记忆``

本文件主要负责“编排”，具体的问题理解、检索、重排、上下文和答案生成分别由
understand.py、search.py、context.py 和 answer.py 实现。
"""

# logging 记录检索计划、耗时、置信度和失败通道，便于生产环境排查。
import logging
# inspect 用来检查被调用函数是否支持新参数，兼容旧实现和测试替身。
import inspect
# ThreadPoolExecutor 让私有库、公共库和网页检索同时执行；as_completed 按完成顺序取结果。
from concurrent.futures import ThreadPoolExecutor, as_completed
# perf_counter 是高精度计时器，用于统计各阶段耗时。
from time import perf_counter

# Settings 提供检索数量、上下文预算、模型开关等配置。
from ..config import Settings
# ModelGateway 统一封装问题理解、向量化、重排和大语言模型调用。
from ..models import ModelGateway
# RedisStore 保存短期对话历史和缓存。
from ..storage.redis import RedisStore
# MilvusStore 查询公共知识库和用户私有向量。
from ..storage.vector import MilvusStore
# set_user_context 把当前用户和会话写入日志上下文，便于追踪一次请求。
from ..system import set_user_context
# WorkspaceService 管理用户上传文件、解析结果和私有资料检索。
from ..workspace import WorkspaceService
# MemoryOrchestrator 统一管理短期记忆、长期记忆和指代消解上下文。
from ..memory.api import MemoryOrchestrator
# ContextBuilder 控制最终喂给大模型的历史和证据数量，避免超过上下文长度。
from .context import ContextBuilder
# AnswerGenerator 根据证据生成回答、引用和可下载解决方案。
from .answer import AnswerGenerator
# QueryUnderstanding 负责意图识别、关键词、法律领域和问题改写。
from .understand import QueryUnderstanding
# SearchRouter 根据问题理解结果决定查哪些集合、是否查网页等。
from .plan import SearchRouter
# EvidenceBuilder 整理证据；ProcessingPipeline 融合与重排；Reranker 调用重排模型。
from .search import EvidenceBuilder, ProcessingPipeline, Reranker
# Retriever 执行用户资料、公共知识库和网页三个检索通道。
from .search import Retriever


# 本模块日志器名称与 API 层不同，便于区分接口问题和 RAG 流程问题。
logger = logging.getLogger("law_rag.rag")

# 模型可以用这对标签把内部草稿和最终回答分开；这里只向用户展示标签内部内容。
FINAL_ANSWER_OPEN_TAG = "<final_answer>"
FINAL_ANSWER_CLOSE_TAG = "</final_answer>"
# 无标签流式内容至少积累 24 个字符才开始展示，降低把模型草稿误发给用户的风险。
UNTAGGED_STREAM_MIN_CHARS = 24
# 这些口语化开头常见于模型内部自我修正，不适合作为正式法律回答开头。
NOISY_ANSWER_PREFIXES = ("对，", "对。", "不对，", "不对。", "等下，", "等下。", "然后", "首先得", "用户现在")
# 当用户说“和上次无关”时，把这条系统消息作为唯一历史，明确禁止沿用旧事实。
RESET_CONTEXT_SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "用户明确表示本轮问题与之前或上次内容无关。"
        "本轮必须按新的独立问题处理，不得引用之前对话、旧材料、旧结论或旧上传文件；"
        "如果当前事实不足，应说明缺少本轮事实；只有用户明确要求查看附件或上传材料时，才说明当前会话没有可用材料。"
    ),
}


class RagWorkflow:
    """法律问答主流程：理解问题、规划检索、召回证据、重排融合并生成答案。"""

    def __init__(self, settings: Settings, model: ModelGateway, milvus: MilvusStore, redis: RedisStore, workspace: WorkspaceService, memory: MemoryOrchestrator | None = None, user_state=None):
        """组装一套可运行的在线问答服务。

        这些依赖在应用启动时创建一次并传入，后续每个请求复用，避免反复加载模型和
        建立数据库连接。``memory`` 与 ``user_state`` 可选，用于兼容尚未启用完整记忆
        功能的环境。
        """

        # 保存全局配置，后续读取数量限制、阈值和开关。
        self.settings = settings
        # 保存统一模型网关，多个子组件可以复用同一配置和客户端。
        self.model = model
        # Redis 既是旧版短期记忆后备，也可被其他流程用于缓存。
        self.redis = redis
        # 新版统一记忆服务，可能为 None。
        self.memory = memory
        # 用户资料/偏好状态服务，可能为 None。
        self.user_state = user_state
        # 创建问题理解器，用大模型与规则提取意图、关键词和改写问题。
        self.understanding = QueryUnderstanding(model)
        # 创建检索路由器，根据理解结果生成检索计划。
        self.router = SearchRouter(settings)
        # 创建检索器，并把模型、Milvus 与用户工作区交给它。
        self.retriever = Retriever(settings, model, milvus, workspace)
        # 创建重排器；两个限制避免一次向重排模型发送过多或过长文本。
        self.ranker = Reranker(model, input_limit=settings.retrieval_rerank_input_max, doc_max_chars=settings.reranker_doc_max_chars)
        # EvidenceBuilder 把重排结果整理成最终可引用证据，并限制数量和正文长度。
        self.evidence = EvidenceBuilder(
            limit=settings.retrieval_evidence_limit,
            max_content_chars=settings.evidence_max_chars,
            exact_limit=settings.retrieval_evidence_limit,
        )
        # ProcessingPipeline 先用 RRF 融合多路召回，再调用重排器和证据整理器。
        self.processing = ProcessingPipeline(
            self.ranker,
            self.evidence,
            rrf_limit=settings.retrieval_candidate_pool_max,
            rrf_k=settings.retrieval_rrf_k,
        )
        # ContextBuilder 控制历史与证据占用的 token，防止超过大模型上下文窗口。
        self.context_builder = ContextBuilder(
            max_tokens=settings.context_max_tokens,
            budget={"evidence": settings.context_evidence_token_budget},
            evidence_max_chars=settings.context_evidence_max_chars,
            evidence_limit=settings.retrieval_evidence_limit,
        )
        # 最后创建答案生成器，负责把问题、历史和证据组织成提示词并调用大模型。
        self.generator = AnswerGenerator(model)

    @staticmethod
    def user_facing_answer(text: str) -> str:
        """从模型可能返回的草稿/标签内容中截取真正展示给用户的回答。"""

        # 把 None 等输入安全地转成字符串并去除首尾空白。
        value = str(text or "").strip()
        # 找到开始标签时，只保留它后面的内容。
        if FINAL_ANSWER_OPEN_TAG in value:
            value = value.split(FINAL_ANSWER_OPEN_TAG, 1)[1]
        # 找到结束标签时，只保留它前面的内容。
        if FINAL_ANSWER_CLOSE_TAG in value:
            value = value.split(FINAL_ANSWER_CLOSE_TAG, 1)[0]
        # 最后清理不自然的模型开头。
        return RagWorkflow.sanitize_answer(value)

    @staticmethod
    def sanitize_answer(text: str) -> str:
        """去掉模型草稿式前缀，但尽量不改写正式回答正文。"""

        # 统一成去除首尾空白的字符串。
        value = str(text or "").strip()
        # 这些前缀表达的是排版提示，删除后不改变结论内容。
        conclusion_prefixes = ("先说结论：", "先说结论:", "结论：", "结论:")
        # 如果回答正好以结论前缀开头，只删除该前缀。
        for prefix in conclusion_prefixes:
            if value.startswith(prefix):
                return value[len(prefix):].lstrip()
        # 正式回答以章节、序号或“根据”开头时直接保留。
        markers = ["一、", "一.", "根据"]
        if any(value.startswith(prefix) for prefix in markers):
            return value
        # “关于……”也是可接受的正式开头。
        if value.startswith("关于"):
            return value
        # 如果前面有少量草稿文字，找到正文标志第一次出现的位置。
        starts = [index for marker in [*conclusion_prefixes, *markers] if (index := value.find(marker)) > 0]
        # 从最早正文位置重新清理，避免遗漏紧随其后的结论前缀。
        if starts:
            return RagWorkflow.sanitize_answer(value[min(starts):].strip())
        # 口语化噪声开头无法安全精确截断时，优先保留最后三段较成熟内容。
        if value.startswith(NOISY_ANSWER_PREFIXES):
            # 统一 Windows/Linux 换行符后按行切分。
            sentences = value.replace("\r\n", "\n").split("\n")
            # 去除空行并清理每行首尾空白。
            clean_sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
            # 超过三段时取最后三段；内容很短则原样返回，避免误删答案。
            return "\n".join(clean_sentences[-3:]).strip() if len(clean_sentences) > 3 else value
        # 没发现已知噪声时保持模型原文。
        return value

    @staticmethod
    def should_stream_untagged_answer(buffer: str) -> bool:
        """判断没有 final_answer 标签的累计文字能否安全开始流式展示。"""

        # 统一待判断文本。
        value = str(buffer or "").strip()
        # 内容太短时信息不足，继续等待后续字符。
        if len(value) < UNTAGGED_STREAM_MIN_CHARS:
            return False
        # 明显像自我修正的开头暂不展示。
        if value.startswith(NOISY_ANSWER_PREFIXES):
            return False
        # 前 80 字出现这些词时更像内部推理或草稿。
        draft_markers = ("思考", "草稿", "内部", "先分析", "自我纠错")
        # 没有草稿标志才允许无标签流式输出。
        return not any(marker in value[:80] for marker in draft_markers)

    @staticmethod
    def user_facing_answer_stream(deltas):
        """过滤模型流式增量，只逐步产出适合直接展示的最终回答。"""

        # buffer 暂存还不能确定是否安全展示的字符。
        buffer = ""
        # 标记是否已经读到 <final_answer>。
        inside_final_answer = False
        # 标记是否已经读到 </final_answer>，之后的模型文字全部忽略。
        final_answer_closed = False
        # 模型没有标签但内容被判断为安全后，进入直接透传模式。
        streaming_untagged = False
        # deltas 是模型逐段返回的文本生成器。
        for delta in deltas:
            # 空片段没有展示价值，跳过。
            if not delta:
                continue
            # 最终答案已经闭合后，忽略可能附带的内部说明。
            if final_answer_closed:
                continue
            # 无标签安全模式已经开启时，后续片段可以直接发给前端。
            if streaming_untagged:
                yield delta
                continue
            # 尚未确定时，把新片段加入缓冲区。
            buffer += str(delta)
            # 还没进入 final_answer 标签时，先寻找开始标签。
            if not inside_final_answer:
                open_index = buffer.find(FINAL_ANSWER_OPEN_TAG)
                # 找到标签后丢弃标签前的草稿内容。
                if open_index >= 0:
                    inside_final_answer = True
                    buffer = buffer[open_index + len(FINAL_ANSWER_OPEN_TAG):]
                # 长度足够且不像草稿时，允许兼容没有标签的模型输出。
                elif RagWorkflow.should_stream_untagged_answer(buffer):
                    streaming_untagged = True
                    # 开始输出前再清理一次开头。
                    piece = RagWorkflow.sanitize_answer(buffer)
                    buffer = ""
                    if piece:
                        yield piece
                    continue
                else:
                    # 证据仍不足时继续积累字符，不向用户展示。
                    continue
            # 已进入最终答案区域，查找结束标签。
            close_index = buffer.find(FINAL_ANSWER_CLOSE_TAG)
            if close_index >= 0:
                # 只输出结束标签之前的正文。
                piece = buffer[:close_index]
                buffer = ""
                final_answer_closed = True
            else:
                # 保留最多“结束标签长度-1”的尾部，防止标签被模型拆成多个增量。
                keep = len(FINAL_ANSWER_CLOSE_TAG) - 1
                if len(buffer) <= keep:
                    continue
                # 前半部分确定不是结束标签，可以安全输出；尾部留待下一轮判断。
                piece = buffer[:-keep]
                buffer = buffer[-keep:]
            # 非空正文片段交给上层逐字符发送。
            if piece:
                yield piece
        # 无标签模式已经边读边输出，无需再处理缓冲区。
        if streaming_untagged:
            return
        # 模型结束但没有给结束标签时，把 final_answer 内剩余文字输出。
        if inside_final_answer and not final_answer_closed and buffer:
            yield buffer
        # 整个回答都没有标签且之前未达到流式阈值时，最后统一清理并输出。
        elif not inside_final_answer and buffer:
            piece = RagWorkflow.user_facing_answer(buffer)
            if piece:
                yield piece

    @staticmethod
    def source_snapshot(rows: list[dict]) -> list[dict]:
        """从完整证据中提取适合写日志的来源摘要，避免记录整段法律正文。"""

        # 列表推导会为每条来源生成一个只含标识、类型、标题和分数的小字典。
        return [
            {
                # 内部统一把 source_id 记录为 chunk_id，便于定位原始分块。
                "chunk_id": row.get("source_id", ""),
                "source_type": row.get("source_type", "unknown"),
                "collection": row.get("collection", ""),
                "retrieval_channel": row.get("retrieval_channel", ""),
                "title": row.get("title", ""),
                "score": row.get("score"),
                "rerank_score": row.get("rerank_score"),
            }
            for row in rows
        ]

    @staticmethod
    def thinking_event(summary: str, step: str, status: str = "running", mode: str = "append", sources: list[dict] | None = None) -> tuple[str, dict]:
        """构造统一的 thinking 事件，供前端展示检索和分析进度。"""

        # payload 中的字段保持稳定，前端无需针对每个阶段写不同解析逻辑。
        payload = {
            # running/completed 表示步骤是否完成。
            "status": status,
            # summary 是用户首先看到的简短状态。
            "summary": summary,
            # append 累积步骤，replace 用当前内容替换之前的思考展示。
            "mode": mode,
            # step 是这一时刻较详细的说明。
            "step": step,
            # steps 保留列表形式，兼容前端原有组件。
            "steps": [step] if step else [],
            # smooth 告诉前端使用平滑过渡动画。
            "smooth": True,
        }
        # 只有明确传入来源时才添加字段，避免把 None 误当成“已加载空来源”。
        if sources is not None:
            payload["sources"] = sources
        # 工作流事件统一返回二元组：(事件名称, 事件数据)。
        return "thinking", payload

    def warn_low_confidence(self, rows: list[dict]) -> int:
        """统计低置信度证据并写警告日志，返回低置信度记录数量。"""

        # 从配置读取阈值；旧配置没有该项时默认使用 0.7。
        threshold = getattr(self.settings, "log_low_confidence_score", 0.7)
        # 重排降级，或重排/召回分数低于阈值，都归为需要人工复核。
        low_rows = [
            row for row in rows
            if row.get("rerank_degraded") or float(row.get("rerank_score", row.get("score", 1)) or 0) < threshold
        ]
        # 只有确实存在低置信度记录时才写日志，避免无意义噪声。
        if low_rows:
            logger.warning(
                "检索结果置信度低，建议人工复核",
                extra={
                    "event": "low_confidence_retrieval",
                    "fields": {
                        "threshold": threshold,
                        "low_confidence_sources": self.source_snapshot(low_rows),
                    },
                },
            )
        # 上层根据数量决定是否在最终回答中增加风险提示。
        return len(low_rows)

    @staticmethod
    def _search_task(label: str, func, *args):
        """包装一个检索通道，统一记录耗时和错误。"""

        # 记录检索开始时间。
        start = perf_counter()
        # 单个检索通道失败不能让另外两个通道一起失败，因此在这里捕获异常。
        try:
            # func 可能是私有材料、公共知识库或网页检索函数。
            rows = func(*args)
            # 成功时错误字符串为空。
            error = ""
        except Exception as exc:
            # 失败通道按无结果处理，让流程继续使用其他来源。
            rows = []
            # 保存错误文字，稍后写日志并返回前端检索状态。
            error = str(exc)
        # 返回通道名、结果、毫秒耗时和错误文字。
        return label, rows, round((perf_counter() - start) * 1000, 2), error

    @staticmethod
    def channel_counts(rows: list[dict]) -> dict[str, int]:
        """按 retrieval_channel/source_type 统计各检索路线返回多少条记录。"""

        # counts 的键是通道名，值是累计数量。
        counts: dict[str, int] = {}
        # 遍历所有原始召回结果。
        for row in rows:
            # 优先用更具体的 retrieval_channel，否则退回来源类型。
            channel = str(row.get("retrieval_channel") or row.get("source_type") or "unknown")
            # 第一次出现从 0 开始，然后每条加 1。
            counts[channel] = counts.get(channel, 0) + 1
        # 统计结果用于日志和前端调试信息。
        return counts

    def _thinking_enabled(self, thinking_enabled: bool | None = None) -> bool:
        """决定本轮是否展示深度思考进度：请求参数优先，其次使用全局配置。"""

        # 前端明确传 True/False 时尊重用户本轮选择。
        if thinking_enabled is not None:
            return bool(thinking_enabled)
        # 没有传时使用配置，旧配置缺少该项则默认开启。
        return bool(getattr(self.settings, "deepseek_thinking", True))

    @staticmethod
    def _stream_answer_text(generator, question: str, evidence: list[dict], history: list[dict], thinking_enabled: bool, answer_detail: str = "standard"):
        """兼容新旧 AnswerGenerator，流式调用时按能力传 answer_detail。"""

        # 取得实际要调用的流式生成方法。
        method = generator.stream_answer_text
        # 读取方法签名，测试替身或旧实现可能没有 answer_detail 参数。
        parameters = inspect.signature(method).parameters
        # 显式存在 answer_detail，或方法接受 **kwargs，都代表可以传详细度。
        accepts_detail = "answer_detail" in parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
        )
        # thinking_enabled 是所有版本都支持的参数。
        kwargs = {"thinking_enabled": thinking_enabled}
        # 只在被调用方支持时增加 answer_detail，避免 TypeError。
        if accepts_detail:
            kwargs["answer_detail"] = answer_detail
        # yield from 将模型生成器的每个文本增量原样向上产出。
        yield from method(question, evidence, history, **kwargs)

    @staticmethod
    def _generate_answer(generator, question: str, evidence: list[dict], history: list[dict], thinking_enabled: bool, answer_detail: str = "standard") -> dict:
        """兼容新旧 AnswerGenerator，执行一次非流式完整答案生成。"""

        # 取得生成完整答案的方法。
        method = generator.generate
        # 检查它是否接受 answer_detail。
        parameters = inspect.signature(method).parameters
        accepts_detail = "answer_detail" in parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
        )
        # 先准备所有版本共有的开关。
        kwargs = {"thinking_enabled": thinking_enabled}
        # 新版方法支持时再传回答详细度。
        if accepts_detail:
            kwargs["answer_detail"] = answer_detail
        # 返回包含 answer、citations、risk_notice 等字段的答案字典。
        return method(question, evidence, history, **kwargs)

    @staticmethod
    def _understand_query(understanding, question: str, memory_context):
        """调用问题理解器；单独封装便于测试替换。"""

        # memory_context 帮助理解“他、这个合同、上一条”等多轮指代。
        return understanding.understand(question, memory_context)

    def _run_stream(
        self, question: str, user: dict | None = None, session_id: str | None = None,
        include_web: bool = True, thinking_enabled: bool | None = None,
        retrieval_mode: str = "auto", answer_detail: str = "standard",
    ):
        """执行一轮完整 RAG，并依次产出可供 SSE 返回的事件。

        这是在线问答最核心的生成器。它不是一次 ``return`` 所有结果，而是在每个阶段
        ``yield`` 状态或文字片段，因此前端能实时看到“正在理解、正在检索”和逐字回答。
        最后一个事件名称为 ``complete``，其中包含完整答案、证据和运行元数据。
        """

        # 记录整轮请求开始时间，用于最终总耗时。
        start = perf_counter()
        # 确保 include_web 一定是布尔值。
        include_web = bool(include_web)
        # 只接受三个检索模式，未知值按 auto 处理，防止路由器收到非法状态。
        retrieval_mode = retrieval_mode if retrieval_mode in {"local", "auto", "force"} else "auto"
        # 只接受三个回答详细度，未知值回退到标准回答。
        answer_detail = answer_detail if answer_detail in {"concise", "standard", "detailed"} else "standard"
        # 登录用户使用真实 ID；游客使用 anonymous，仅用于日志且不持久化会话。
        user_id = user.get("user_id") if user else "anonymous"
        # 游客不能通过伪造 session_id 使用记忆或查看私有资料。
        if not user:
            session_id = None
        # 将用户和会话加入本请求日志上下文，之后日志可自动带上这些标识。
        set_user_context(user_id, session_id)
        # 检测用户是否明确说“这是新问题/与前文无关”。只有登录会话才有旧上下文可重置。
        reset_previous_context = bool(
            user
            and session_id
            # getattr 提供后备函数，兼容没有该能力的旧 workspace 或测试对象。
            and getattr(getattr(self.retriever, "workspace", None), "query_resets_previous_context", lambda _query: False)(question)
        )

        # 合并本轮参数和全局配置，得到最终思考展示开关。
        thinking_enabled = self._thinking_enabled(thinking_enabled)
        # 开启时先告诉前端已经开始处理，避免用户面对空白等待。
        if thinking_enabled:
            yield self.thinking_event("正在启动深度思考", "正在接收问题并准备拆解事实、诉求和可用证据。")

        # getattr 兼容测试或旧对象没有 memory 属性的情况。
        memory = getattr(self, "memory", None)
        # 用户明确开启新上下文时，先清理这个会话的旧记忆。
        if reset_previous_context and session_id:
            try:
                # 新版使用统一记忆服务删除短期和持久化会话数据。
                if memory:
                    memory.delete_session(user_id, session_id)
                else:
                    # 没有统一记忆服务时退回旧 Redis 历史删除。
                    self.redis.delete_history(user_id, session_id)
            except Exception:
                # 清理失败不能把旧上下文重新加入本轮，因此后面仍按 reset 状态继续。
                logger.warning(
                    "清理隔离会话记忆失败，继续按无历史上下文处理本轮问题",
                    extra={"event": "rag_reset_context_cleanup_failed", "fields": {"session_id": session_id or ""}},
                    exc_info=True,
                )
        # 登录信息中可能已经附带用户偏好，复制字典避免后续修改原 user 对象。
        user_profile = dict((user or {}).get("profile") or {})
        # user_state 是可选的持久化用户资料服务。
        user_state = getattr(self, "user_state", None)
        # 已登录但请求中没有 profile 时，再从用户状态服务读取。
        if user_state and user and not user_profile:
            try:
                user_profile = user_state.get_profile(user_id)
            except Exception:
                # 偏好读取失败时使用空偏好，不阻断法律问答主流程。
                user_profile = {}
        # 重置时绝不加载旧上下文；否则仅对登录且有 session_id 的请求加载记忆。
        memory_context = None if reset_previous_context else (memory.load_context(user_id, session_id, user_profile=user_profile, query=question) if memory and session_id else None)
        # 记忆服务可能已经完成指代消解；有 resolved_query 时用它检索，否则用原问题。
        effective_question = getattr(memory_context, "resolved_query", "") or question
        # 重置时只放入禁止沿用旧内容的系统消息；正常情况优先用统一记忆，最后退回 Redis。
        history = [RESET_CONTEXT_SYSTEM_MESSAGE] if reset_previous_context else (memory_context.short_term if memory_context else (self.redis.get_history(user_id, session_id) if session_id else []))
        # stage_timings 保存每个阶段的毫秒耗时，最终返回给日志和调试界面。
        stage_timings: dict[str, float] = {}

        # ---------- 第一阶段：理解问题 ----------
        # status 是面向普通用户的简短状态事件。
        yield "status", {"message": "正在理解问题", "stage": "understanding"}
        # 单独记录问题理解开始时间。
        stage_start = perf_counter()
        # 提取意图、关键词、法律领域、法条号、改写问题和指代消解信息。
        info = self._understand_query(self.understanding, effective_question, memory_context)
        # 将秒转换为毫秒并保留两位小数。
        stage_timings["understanding"] = round((perf_counter() - stage_start) * 1000, 2)
        # step 事件提供比 status 更详细、可展开的阶段结果。
        yield "step", {
            "stage": "understanding",
            "status": "completed",
            "message": "已识别问题结构",
            "detail": f"关键词 {len(info.get('keywords', []))} 个 / 意图 {info.get('intent', 'general')}",
            "elapsed_ms": stage_timings["understanding"],
            "route": info.get("route", "full_retrieval"),
            "keywords": info.get("keywords", []),
            "article_numbers": info.get("article_numbers", []),
            "legal_domains": info.get("legal_domains", []),
            "intent": info.get("intent", "general"),
            "confidence": info.get("confidence", 0.0),
            "intent_reason": info.get("intent_reason", ""),
            "understanding_source": info.get("understanding_source", ""),
            "understanding_error": info.get("understanding_error", ""),
            "route_memory_used": info.get("route_memory_used", False),
            "query_variants": info.get("query_variants", []),
            "original_query": info.get("original_query", question),
            "rewritten_query": info.get("rewritten_query", question),
            "resolved_references": {**getattr(memory_context, "resolved_references", {}), **info.get("resolved_references", {})} if memory_context else info.get("resolved_references", {}),
            "reset_previous_context": reset_previous_context,
        }
        # 深度思考展示开启时，用最多三个法律领域描述下一步方向。
        if thinking_enabled:
            domains = "、".join(info.get("legal_domains", [])[:3]) or "相关法律关系"
            yield self.thinking_event(
                "已完成问题拆解，正在规划检索",
                f"已识别 {len(info.get('keywords', []))} 个关键词，初步聚焦 {domains} 和用户的具体诉求。",
            )

        # ---------- 第二阶段：规划并执行检索 ----------
        # retrieval_start 覆盖规划和所有检索通道，用于计算整个召回阶段耗时。
        retrieval_start = perf_counter()
        # 告诉前端已经从问题理解切换到检索规划。
        yield "status", {"message": "正在规划检索", "stage": "retrieval"}
        # planning 单独计时，方便发现意图路由是否过慢。
        stage_start = perf_counter()
        # 取得路由方法并检查签名，兼容旧版路由器和测试替身。
        route_method = self.router.route
        route_parameters = inspect.signature(route_method).parameters
        # 新版支持 retrieval_mode 时传入用户选择的“仅知识库/智能/强制联网”。
        if "retrieval_mode" in route_parameters:
            plan = route_method(info, include_web, retrieval_mode=retrieval_mode)
        else:
            # 旧版只接收问题理解和联网开关。
            plan = route_method(info, include_web)
        # 保存规划耗时。
        stage_timings["planning"] = round((perf_counter() - stage_start) * 1000, 2)
        # 将最终检索计划写入结构化日志，便于判断为什么查了这些集合。
        logger.info(
            "RAG 检索计划生成",
            extra={
                "event": "rag_plan_created",
                "fields": {
                    "include_web": include_web,
                    "retrieval_mode": retrieval_mode,
                    "question_length": len(question),
                    "rewritten_query": plan.rewritten_query,
                    "session_id": session_id or "",
                    "searched_collections": plan.collections,
                    "priority_collections": plan.priority_collections,
                    "route": plan.route,
                    "route_reason": plan.route_reason,
                    "query_variants": plan.query_variants,
                    "article_numbers": plan.article_numbers,
                    "legal_domains": plan.legal_domains,
                    "intent": plan.intent,
                    "route_memory_used": info.get("route_memory_used", False),
                    "retrieval_notes": plan.retrieval_notes,
                },
            },
        )

        # 默认公共知识库是主要检索目标。
        search_targets = ["公共知识库"]
        # 登录用户且计划允许私有检索时，把上传材料放到目标列表最前面。
        if user and plan.search_private:
            search_targets.insert(0, "私有材料")
        # 路由计划最终决定联网时，加入网页目标。
        if plan.include_web:
            search_targets.append("网页")
        # 思考事件向用户说明系统会查哪些来源。
        if thinking_enabled:
            yield self.thinking_event(
                f"正在并行检索{'、'.join(search_targets)}",
                f"检索计划已确定：优先从 {'、'.join(search_targets)} 中寻找能对应具体事实的材料。",
            )
        # 普通状态事件供没有展开思考面板的前端显示。
        yield "status", {"message": f"正在并行检索{'、'.join(search_targets)}", "stage": "retrieval"}
        # 三个列表分别保存私有、公共和网页检索结果，未启用的通道保持空列表。
        search_results = {"private": [], "public": [], "web": []}
        # 失败通道的错误信息单独保存，最终写日志但不阻断其他通道。
        search_errors: dict[str, str] = {}
        # 先为三个通道写入 0，确保最终 meta 字段结构稳定。
        stage_timings["private_search"] = 0.0
        stage_timings["public_search"] = 0.0
        stage_timings["web_search"] = 0.0
        # 最多三个工作线程，正好对应三个互不依赖的检索来源。
        with ThreadPoolExecutor(max_workers=3) as executor:
            # futures 保存“异步任务 -> 通道名称”，完成后据此放回正确位置。
            futures = {}
            # 只有登录用户且计划要求时才提交私有资料检索任务。
            if user and plan.search_private:
                futures[executor.submit(self._search_task, "private", self.retriever.search_user_materials, plan, user, session_id)] = "private"
            # 大多数法律问题都需要公共知识库，具体由 plan.search_public 决定。
            if plan.search_public:
                futures[executor.submit(self._search_task, "public", self.retriever.search_public_knowledge, plan)] = "public"
            # 只有最终计划 include_web=True 时才真正提交网页检索。
            if plan.include_web:
                futures[executor.submit(self._search_task, "web", self.retriever.search_web_pages, plan)] = "web"
            # as_completed 会先处理先完成的通道，不必等待较慢网页后才显示其他结果。
            for future in as_completed(futures):
                # 取回提交任务时保存的通道名。
                label = futures[future]
                # 每个任务已经在 _search_task 中统一捕获异常和统计耗时。
                task_label, rows, elapsed_ms, error = future.result()
                # 把结果放进相应通道。
                search_results[label] = rows
                # 保存实际通道耗时。
                stage_timings[f"{task_label}_search"] = elapsed_ms
                # 有错误时记录，但继续使用其他通道的检索结果。
                if error:
                    search_errors[label] = error
                    logger.warning(
                        "RAG %s 检索失败",
                        label,
                        extra={
                            "event": "retrieval_failed",
                            "fields": {
                                "source": label,
                                "error": error,
                                "include_web": include_web,
                                "session_id": session_id or "",
                            },
                        },
                    )
                # 如果前端展示思考过程，就实时报告每个通道的完成或失败状态。
                if thinking_enabled:
                    # 内部英文通道名转换为用户容易理解的中文。
                    channel_names = {"private": "私有材料", "public": "公共法律库", "web": "网页资料"}
                    channel_name = channel_names.get(label, label)
                    if error:
                        # 先标记该通道结束，再回到整体检索仍在运行的状态。
                        yield self.thinking_event("检索仍在继续，部分来源暂不可用", f"{channel_name} 检索暂未成功，正在保留其他可用来源继续分析。", status="completed")
                        yield self.thinking_event("检索仍在继续，部分来源暂可恢复", f"{channel_name} 正在切回检索准备状态，继续补齐其他来源。", status="running")
                    else:
                        # 成功时显示候选数量和秒级耗时。
                        yield self.thinking_event(
                            "正在汇总检索结果",
                            f"{channel_name} 已返回 {len(rows)} 条候选材料，用时 {(elapsed_ms / 1000):.1f} 秒。",
                        )

        # 将三路结果合并成原始候选列表，供日志、统计和后续融合。
        raw = search_results["private"] + search_results["public"] + search_results["web"]
        # 更细地统计向量、关键词、网页等实际召回通道数量。
        retrieval_channel_counts = self.channel_counts(raw)
        # ProcessingPipeline 需要“多路列表”，因此只保留非空来源列表。
        channels = [
            rows for rows in (
                search_results["private"],
                search_results["public"],
                search_results["web"],
            )
            if rows
        ]
        # 整个检索阶段到这里结束，记录规划加召回总耗时。
        stage_timings["retrieval"] = round((perf_counter() - retrieval_start) * 1000, 2)
        # 记录原始召回来源和错误，便于分析召回是否正常。
        logger.info(
            "RAG 原始检索完成",
            extra={
                "event": "retrieval_completed",
                "fields": {
                    "retrieval_sources": self.source_snapshot(raw),
                    "raw_result_count": len(raw),
                    "retrieval_counts": {
                        "private": len(search_results["private"]),
                        "public": len(search_results["public"]),
                        "web": len(search_results["web"]),
                    },
                    "retrieval_channel_counts": retrieval_channel_counts,
                    "search_errors": search_errors,
                },
            },
        )
        # 向前端发送检索阶段的结构化完成事件。
        yield "step", {
            "stage": "retrieval",
            "status": "completed",
            "message": f"检索到 {len(raw)} 条候选",
            "detail": f"私有 {len(search_results['private'])} 条 / 公共 {len(search_results['public'])} 条 / 网页 {len(search_results['web'])} 条",
            "elapsed_ms": stage_timings["retrieval"],
            "count": len(raw),
            "searched_collections": plan.collections,
            "priority_collections": plan.priority_collections,
            "route": plan.route,
            "route_reason": plan.route_reason,
            "query_variants": plan.query_variants,
            "retrieval_notes": plan.retrieval_notes,
            "retrieval_mode": plan.retrieval_mode,
            "web_used": plan.include_web,
        }
        # 思考面板提示接下来会从候选中筛选证据。
        if thinking_enabled:
            yield self.thinking_event(
                "检索完成，正在筛选核心证据",
                f"共找到 {len(raw)} 条候选材料，接下来会融合排序并保留最能支撑判断的证据。",
            )

        # ---------- 第三阶段：融合、重排并选出核心证据 ----------
        yield "status", {"message": "正在融合与重排", "stage": "ranking"}
        # ranking 总计时包含融合、重排和证据整理。
        stage_start = perf_counter()
        # 优先使用初始化好的处理管道；getattr 后备值方便精简测试对象。
        processing = getattr(self, "processing", ProcessingPipeline(self.ranker, self.evidence))
        # 先记录 RRF 融合开始时间。
        fusion_start = perf_counter()
        # 合并私有、公共、网页等多路结果；全空时仍传 [raw] 保持参数结构。
        fused = processing.merge_results(channels or [raw])
        # 保存融合耗时。
        stage_timings["fusion"] = round((perf_counter() - fusion_start) * 1000, 2)
        # 接下来用 BGE reranker 对融合候选重新按问题相关度打分。
        rerank_start = perf_counter()
        ranked = processing.rank(plan.query, fused)
        # 保存重排模型耗时，它通常是需要重点优化的部分。
        stage_timings["rerank"] = round((perf_counter() - rerank_start) * 1000, 2)
        # 最后开始整理可引用证据。
        evidence_start = perf_counter()
        # 统计低分或降级结果，稍后可能向用户增加风险提示。
        low_confidence = self.warn_low_confidence(ranked)
        # 根据限制和去重规则，从排名结果中构建最终证据列表。
        evidence = processing.build_evidence(ranked)
        # 保存证据整理和整个排名阶段耗时。
        stage_timings["evidence"] = round((perf_counter() - evidence_start) * 1000, 2)
        stage_timings["ranking"] = round((perf_counter() - stage_start) * 1000, 2)
        # 告诉前端候选经过融合后最终保留多少条证据。
        yield "step", {
            "stage": "ranking",
            "status": "completed",
            "message": f"保留 {len(evidence)} 条证据",
            "detail": f"融合 {len(fused)} 条 / 证据 {len(evidence)} 条",
            "elapsed_ms": stage_timings["ranking"],
            "count": len(evidence),
        }
        # 思考面板说明证据筛选已完成，即将生成正式回答。
        if thinking_enabled:
            yield self.thinking_event(
                "已筛出核心证据，准备生成答案",
                f"已从 {len(fused)} 条融合候选中保留 {len(evidence)} 条核心证据，正在对应事实、规则和风险点。",
            )

        # ---------- 第四阶段：构建上下文并生成答案 ----------
        yield "status", {"message": "正在生成回答", "stage": "generation"}
        # 获取初始化好的上下文构建器；后备实例方便测试中的精简工作流。
        context_builder = getattr(self, "context_builder", ContextBuilder())
        # 按 token 预算组合改写问题、会话历史、用户偏好和检索证据。
        context_payload = context_builder.build_context(plan.query, memory_context, evidence, user_profile=user_profile)
        # 上下文构建器可能因预算裁剪证据，后续回答和来源必须只使用真正进入提示词的部分。
        evidence = context_payload["evidence"]
        # answer_history 是经过隐私和长度控制后真正喂给大模型的历史。
        answer_history = context_payload["history"]
        # 新版生成器提供 answer_metadata；测试替身或旧版可能没有。
        metadata_builder = getattr(self.generator, "answer_metadata", None)
        # 在正式生成前准备引用摘要，供“思考完成”事件展示。
        pre_answer_metadata = metadata_builder(context_payload["evidence"]) if callable(metadata_builder) else {}
        # 深度思考模式展示几条固定、可解释的分析步骤，但不展示模型内部推理草稿。
        if thinking_enabled:
            for step in [
                "正在核对用户问题中的法律关系、具体诉求和会影响结果的关键事实。",
                f"正在结合 {len(evidence)} 条核心证据区分事实材料、法律规则和补充参考。",
                "正在把检索到的法条、司法解释或类案对应到具体事实，避免只罗列来源名称。",
                "正在同步检查证据缺口、对方可能抗辩和下一步可操作路径。",
            ]:
                yield self.thinking_event(f"正在结合问题和 {len(evidence)} 条核心证据做案情分析", step)
            # 提取即将引用的来源列表。
            thinking_sources = pre_answer_metadata.get("citations", [])
            # replace 表示用完成状态替换之前不断追加的运行状态。
            yield self.thinking_event(
                "思考完成，开始输出正式答案",
                "已完成事实、证据和法律依据核对，下面开始输出正式答案。",
                status="completed",
                mode="replace",
                sources=thinking_sources,
            )
        # 从这里开始计算大模型生成耗时。
        stage_start = perf_counter()
        # 保存已经发给前端的正式回答字符，最后拼回完整答案。
        answer_text_parts: list[str] = []
        # 有证据时优先使用流式生成，让用户更快看到首字。
        if evidence:
            # 取得大模型文本增量生成器。
            deltas = self._stream_answer_text(
                self.generator, plan.query, context_payload["evidence"], answer_history,
                thinking_enabled, answer_detail,
            )
            # 过滤 final_answer 标签和草稿，只保留面向用户的文本片段。
            for piece in self.user_facing_answer_stream(deltas):
                # 再逐字符发送，让前端打字效果平滑。
                for character in piece:
                    # 回答开头的空格和换行不发送。
                    if not answer_text_parts and not character.strip():
                        continue
                    answer_text_parts.append(character)
                    yield "chunk", {"delta": character}
            # 把已发送字符拼成完整字符串，再统一清理一次。
            answer_text = self.user_facing_answer("".join(answer_text_parts))

            # 某些模型可能只输出标签或草稿，过滤后为空时改用非流式生成兜底。
            if not answer_text:
                answer = self._generate_answer(
                    self.generator, plan.query, context_payload["evidence"], answer_history,
                    thinking_enabled, answer_detail,
                )
                # 从结构化答案中取正文并清理标签。
                answer_text = self.user_facing_answer(str(answer.get("answer", "")))
                # 低置信度时把人工复核提醒追加到风险说明。
                if low_confidence:
                    answer["risk_notice"] = (answer.get("risk_notice") or "") + " 本次检索依据置信度较低，结论仅供参考，建议补充更直接的证据后再次确认。"
                # 兜底答案此前尚未发给前端，因此现在逐字符补发。
                for character in answer_text:
                    answer_text_parts.append(character)
                    yield "chunk", {"delta": character}
            else:
                # 流式正文生成成功后，尝试补足引用来源摘要。
                highlight_enricher = getattr(self.generator, "ensure_source_highlights", None)
                enriched_answer = (
                    highlight_enricher(answer_text, plan.query, context_payload["evidence"])
                    if callable(highlight_enricher)
                    else answer_text
                )
                # 补充后的内容与现有回答不同时，只发送新增部分，避免前端重复正文。
                if enriched_answer != answer_text:
                    extra = enriched_answer[len(answer_text):] if enriched_answer.startswith(answer_text) else f"\n\n{enriched_answer}"
                    for character in extra:
                        yield "chunk", {"delta": character}
                    answer_text = enriched_answer
                # 流式方法只生成文字，这里再组合引用、来源等结构化元数据。
                answer = {"answer": answer_text, **self.generator.answer_metadata(context_payload["evidence"])}
                # 同样对低置信度检索结果追加风险提醒。
                if low_confidence:
                    answer["risk_notice"] = (answer.get("risk_notice") or "") + " 本次检索依据置信度较低，结论仅供参考，建议补充更直接的证据后再次确认。"
        else:
            # 没检索到证据时使用非流式结构化生成，让模型说明信息不足而不是编造依据。
            answer = self._generate_answer(
                self.generator, plan.query, context_payload["evidence"], answer_history,
                thinking_enabled, answer_detail,
            )
            # 清理最终答案文本并写回 answer 字典。
            answer_text = self.user_facing_answer(str(answer.get("answer", "")))
            answer["answer"] = answer_text
            # 即便没有证据，也以相同 chunk 事件逐字返回，前端无需特殊处理。
            for character in answer_text:
                answer_text_parts.append(character)
                yield "chunk", {"delta": character}
        # 某些生成结果已带案件分析，优先复用。
        case_analysis = answer.get("case_analysis") or []
        # 没有案件分析且生成器支持补建时，再调用轻量分析函数补充。
        if not case_analysis and hasattr(self.generator, "build_case_analysis"):
            case_analysis = self.generator.build_case_analysis(plan.query, context_payload["evidence"], answer_text, thinking_enabled=False)
            answer["case_analysis"] = case_analysis
        # 记录从提示词准备完成到全部答案生成完毕的耗时。
        stage_timings["generation"] = round((perf_counter() - stage_start) * 1000, 2)
        # 告诉前端生成阶段已完成，并附带正文长度。
        yield "step", {
            "stage": "generation",
            "status": "completed",
            "message": "回答已生成",
            "detail": "输出结论、依据和建议",
            "elapsed_ms": stage_timings["generation"],
            "answer_length": len(str(answer.get("answer", ""))),
        }

        # ---------- 第五阶段：保存记忆并整理最终结果 ----------
        # 只有登录用户才保留 session_id，因此游客问答不会进入记忆存储。
        if session_id:
            # 优先由 MemoryOrchestrator 同时处理短期历史和长期记忆。
            if memory:
                memory.save_turn(
                    # 用户 ID 和会话 ID 共同保证数据隔离。
                    user_id,
                    session_id,
                    # 保存用户原始问题，而不是内部改写后的检索问题。
                    question,
                    # 保存最终向用户展示的回答。
                    answer["answer"],
                    # 结构化答案包含引用、风险提示和案件分析，后续可恢复完整页面。
                    answer_payload=answer,
                    # 记录本轮实际进入上下文的证据数量。
                    source_count=len(evidence),
                )
            else:
                # 没有统一记忆服务时，把用户消息写入 Redis 短期历史。
                self.redis.append_history(user_id, session_id, {"role": "user", "content": question}, self.settings.history_ttl)
                # 再写入助手回答；两条使用相同 TTL 自动过期。
                self.redis.append_history(user_id, session_id, {"role": "assistant", "content": answer["answer"]}, self.settings.history_ttl)

        # 从原始召回结果中提取真正命中的公共集合名称，并排序保证输出稳定。
        retrieved_collections = sorted({
            str(row.get("collection", ""))
            for row in raw
            if row.get("source_type") == "public" and row.get("collection")
        })
        # 收集答案声称引用的 source_id，稍后检查它们是否都来自真实召回结果。
        citation_source_ids = [citation.get("source_id", "") for citation in answer.get("citations", [])]
        # 计算整轮问题从开始到现在的总毫秒数。
        elapsed_ms = round((perf_counter() - start) * 1000, 2)
        # 形成稳定的三路召回数量统计。
        retrieval_counts = {
            "private": len(search_results["private"]),
            "public": len(search_results["public"]),
            "web": len(search_results["web"]),
            "total": len(raw),
        }
        # shared_meta 同时用于日志和返回前端，避免两边统计口径不同。
        shared_meta = {
            # 整轮耗时和进入本轮前已有的历史条数。
            "elapsed_ms": elapsed_ms,
            "history_before": len(history),
            # 原问题和检索改写问题用于解释检索行为。
            "original_query": plan.original_query,
            "rewritten_query": plan.rewritten_query,
            # 合并记忆层和理解层的指代消解结果。
            "resolved_references": {**getattr(memory_context, "resolved_references", {}), **info.get("resolved_references", {})} if memory_context else info.get("resolved_references", {}),
            # 以下字段描述检索计划、命中情况和每阶段耗时。
            "searched_collections": plan.collections,
            "priority_collections": plan.priority_collections,
            "route": plan.route,
            "route_reason": plan.route_reason,
            "query_variants": plan.query_variants,
            "article_numbers": plan.article_numbers,
            "legal_domains": plan.legal_domains,
            "intent": plan.intent,
            "route_memory_used": info.get("route_memory_used", False),
            "retrieval_notes": plan.retrieval_notes,
            "retrieved_collections": retrieved_collections,
            "raw_result_count": len(raw),
            "evidence_count": len(evidence),
            "stage_timings": stage_timings,
            "context": context_payload["meta"],
            "retrieval_counts": retrieval_counts,
            "retrieval_channel_counts": retrieval_channel_counts,
            "retrieval_errors": search_errors,
            "retrieval_mode": plan.retrieval_mode,
            "web_used": plan.include_web,
            "answer_detail": answer_detail,
        }
        # 写入一条完整成功日志，生产环境可据此分析慢请求和引用准确性。
        logger.info(
            "RAG 回答生成完成",
            extra={
                "event": "rag_answer_generated",
                "fields": {
                    **shared_meta,
                    "include_web": include_web,
                    "question_length": len(question),
                    "evidence_sources": self.source_snapshot(evidence),
                    "citation_source_ids": citation_source_ids,
                    # 所有引用 ID 都应存在于本轮真实召回结果，防止模型凭空制造来源。
                    "citation_to_retrieval_matched": set(citation_source_ids).issubset({row.get("source_id", "") for row in raw}),
                },
            },
        )
        # result 是前端最终收到的完整数据结构。
        result = {
            # answer 包含正文、引用、风险提示和案件分析。
            "answer": answer,
            # sources 是真正进入模型上下文的证据，不是全部原始候选。
            "sources": evidence,
            # meta 保存检索路线、耗时和问题理解信息，用于展示与调试。
            "meta": {
                **shared_meta,
                "confidence": info.get("confidence", 0.0),
                "intent_reason": info.get("intent_reason", ""),
                "understanding_source": info.get("understanding_source", ""),
                "understanding_error": info.get("understanding_error", ""),
            },
        }
        # source_id 是服务端内部追踪字段，不直接暴露给普通前端响应。
        result["sources"] = [{k: v for k, v in row.items() if k != "source_id"} for row in result.get("sources", [])]
        # complete 是本轮最后一个事件；run() 也依靠它取得最终返回值。
        yield "complete", result

    def run(
        self, question: str, user: dict | None = None, session_id: str | None = None,
        include_web: bool = True, thinking_enabled: bool | None = None,
        retrieval_mode: str = "auto", answer_detail: str = "standard",
    ) -> dict:
        """非流式调用入口：内部仍执行同一流程，只返回最后的 complete 数据。"""

        # result 初始为 None；只有工作流正常走到 complete 才会被赋值。
        result = None
        # 消费 _run_stream 产生的全部事件；中间 status/chunk 在普通接口中不单独返回。
        for event, data in self._run_stream(
            question, user, session_id, include_web, thinking_enabled,
            retrieval_mode=retrieval_mode, answer_detail=answer_detail,
        ):
            # 只保存最后的完整结果。
            if event == "complete":
                result = data
        # 如果生成器提前结束却没有 complete，说明流程内部状态异常。
        if result is None:
            raise RuntimeError("RAG workflow did not complete")
        # 普通 /ask 接口会把这个字典一次性返回前端。
        return result

    def stream(self, *args, **kwargs):
        """流式调用入口：把 _run_stream 的所有事件直接转交给 API 层。"""

        # *args 和 **kwargs 原样透传，避免重复维护一份长参数列表。
        yield from self._run_stream(*args, **kwargs)
