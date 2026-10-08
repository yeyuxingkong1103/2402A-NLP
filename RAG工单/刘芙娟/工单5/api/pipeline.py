"""提问链路的两个处理步骤：**检索接入**与**生成接入**。

与 `stream.py` 的分界：那边管**事件的顺序**（谁先谁后、终帧怎么拼），
这边管**每一步做了什么、失败怎么办**。分开的理由是两者的"错误"完全不同 ——
顺序错是协议问题（前端会解析不了），步骤失败是运行问题（用户看到拒答），
混在一起排查时要先分辨是哪一类。

---

⚠️ **本模块的两个函数都 MUST NOT 让异常冒泡。**

调用方（`stream.py`）已经处在 SSE 流**开始之后**的位置：`status` 帧已发出、
响应头已确定，此时无法再改 HTTP 状态码。异常冒泡会让客户端拿到一个中断的流，
而那与"网络断了"不可区分 —— 两者该做的处置完全不同（重试 vs 改问题）。

这不违反 constitution 的「禁止 try/except 吞掉失败条件」：判定标准是
"事后能不能查出来"，这里每个失败分支都 MUST 留 ERROR 级日志，因此**可查**。
同一条推理已在 `backend/api/capture.py` 的模块文档字符串中确立。
"""

import logging
from collections.abc import AsyncIterator

from backend.generate import DEFAULT_MODE
from backend.generate.client import LLMError
from backend.generate.service import start as start_generation
from backend.retrieve import RetrievalError
from backend.retrieve import service as retrieval_service
from backend.retrieve.models import RetrievalResult

from . import ANSWER_JOINER, DISCLAIMER
from .config import AppConfig
from .schemas import Citation

logger = logging.getLogger(__name__)

__all__ = [
    "retrieve",
    "citation_payload",
    "preamble_chunks",
    "generation_deltas",
    "assemble_answer_text",
]


def assemble_answer_text(preamble: list[str], body: str) -> str:
    """拼出对外的完整回答文本。**全仓唯一拼接 `answer_text` 的地方。**

        [紧急话术 + 分隔符] + 正文或兜底话术 + 分隔符 + 免责声明

    ---

    ## 为什么它必须只有一个实现

    `docs/05` §4.4 要求「全仓检索这两句逐字文本，只应命中一处」。
    分在两处拼接的后果不是测试失败 —— **是没有任何测试会发现**：
    话术是数据，没有测试覆盖它。而其中一处迟早会漂移，
    表现是"同一个系统在不同入口给出不同的免责声明"。

    ## 为什么放在 `pipeline.py` 而不是 `chat/`

    `backend/chat/` 是非 SSE 的对话层。让它提供 `stream.py` 依赖的函数，
    会把耦合方向倒置（生成层反过来依赖会话层）。
    `pipeline.py` 已经是 `/ask` 的装配中枢（`retrieve` / `citation_payload` /
    `preamble_chunks` / `generation_deltas` 都在此），
    是**唯一不引入新依赖方向**的位置。

    ## 参数用"片段列表"而不是"字符串"

    `preamble` 是 `preamble_chunks()` 的返回值 —— 一个**列表**，元素自带分隔符。
    调用方 MUST NOT 先 `"".join()` 再传进来：那样 `/ask` 与 chat 两条路径
    就会各自决定"分隔符加不加、加在哪"，而它正是本函数要收拢的细节。
    """

    return "".join(preamble) + body + ANSWER_JOINER + DISCLAIMER


def retrieve(
    question: str,
    answer_id: str,
    config: AppConfig,
    query_vector: list[float] | None,
) -> RetrievalResult | None:
    """执行检索。**MUST NOT 让异常冒泡到调用方。**

    返回 `None` 表示"检索没能进行" —— 调用方据此走拒答路径（`_body_chunks`）。

    ---

    ⚠️ **为什么异常不冒泡成 500**（research R9）：

    检索发生在 SSE 流**已经开始之后**（`status` 帧已发出、响应头已定），此时
    无法再改 HTTP 状态码。若让它冒泡，客户端拿到的是流中断 —— 与"网络断了"
    不可区分，而这两件事该做的处置完全不同（重试 vs 改问题）。

    ⚠️ **为什么这不违反 constitution 的「禁止 try/except 吞掉失败条件」**：

    判定标准是"事后能不能查出来"。这里 MUST 留 ERROR 级日志（含 `answer_id`
    与异常类型），因此失败**是可查的**。同一条推理已在 `capture.py` 的模块
    文档字符串中确立，本函数沿用。

    ⚠️ **向量缺失时 MUST NOT 降级为"只跑关键词路"**（FR-020）：
    那会让"语义检索坏了"表现为"检索结果少了些"，用户与服务端都看不出来。
    正确的处置是**不检索、直接拒答 + ERROR 日志**。
    """

    if query_vector is None:
        # S8 的编码失败（capture.py 已记过 WARNING，这里补一条定位到请求的 ERROR）。
        logger.error(
            "retrieval_skipped answer_id=%s reason=query_vector_unavailable", answer_id
        )
        return None

    try:
        return retrieval_service.search(
            question=question,
            query_vector=query_vector,
            top_k=config.top_k,
            threshold=config.similarity_threshold,
        )
    except RetrievalError as exc:
        logger.error(
            "retrieval_failed answer_id=%s code=%s reason=%s",
            answer_id,
            exc.code,
            exc.message.replace("\n", " "),
        )
        return None


def citation_payload(retrieval: RetrievalResult | None) -> list[dict]:
    """把检索结果映射成响应契约里的 citation 对象。

    ⚠️ 映射是刻意的，不是多余的转换：检索层的 `RetrievedPassage` 不该依赖
    HTTP 传输模型（`schemas.Citation`）。二者字段名当前一致，但**它们会因
    不同的原因变化** —— 前端契约变了要改 `Citation`，检索内部结构调整要改
    `RetrievedPassage`。让其中一处能独立变化，是这个映射存在的全部意义。

    `citation_id` 从 1 起编号，与将来正文里 `【n】` 标记的序号一致
    （`docs/05` §3.1.3）。**编号在服务端产生，MUST NOT 由前端推算** ——
    前端推算等于把排序规则复刻了一遍。
    """

    if retrieval is None:
        return []

    return [
        Citation(
            citation_id=index,
            file_name=passage.file_name,
            page_start=passage.page_start,
            page_end=passage.page_end,
            section=passage.section,
            block_type=passage.block_type,
            text=passage.text,
            score=passage.score,
        ).model_dump()
        for index, passage in enumerate(retrieval.passages, start=1)
    ]


def preamble_chunks(config: AppConfig) -> list[str]:
    """正文的**前置**片段。**这是 FR-030（紧急话术前置）的落点。**

    返回的片段按顺序拼接即等于"正文之前的那一段"（片段自带分隔符，不做二次拼接）。

    目前恒为空列表：I-04（紧急判定）未实现，话术因此没有真实触发条件，只有
    验收注入 `test_preamble` 时才有内容。

    ⚠️ 它与正文**分开**、且**先于**正文 yield —— 这是 constitution 原则 IV
    要求的"话术 MUST 出现在任何其他内容之前"。做成"正文之前的一个独立步骤"，
    而不是"正文的第一段"，是为了让这条约束在代码结构上无法被后来的改动破坏。
    """

    if not config.test_preamble:
        return []
    return [config.test_preamble, ANSWER_JOINER]


def encode_query(text: str) -> list[float] | None:
    """把一段文本编码成查询向量。**不带任何副作用。**

    ---

    ## ⚠️ 为什么不复用 `capture_question()`

    `capture_question(question, answer_id)` 做两件事：**留存提问** + 返回向量。

    改写是**内部行为**，不是用户又问了一次。用它会往 `data/questions/`
    再写一条记录 —— 于是留存里会出现"用户问过『高血压 怎么预防』"，
    而用户从未打过这句话。那条记录还会污染后续任何基于留存的统计。

    ## 为什么失败返回 `None` 而不是抛

    调用方（`dialogue.py`）拿不到向量时的处置是唯一的：**照旧拒答**。
    一个必然被处理、处置唯一的失败，抛异常只是把同一件事写成两层。
    这里留 ERROR 日志，因此失败**事后可查**（与 `capture.py` 的判定标准一致）。

    ## 为什么编码器用 `get_encoder()` 取常驻单例

    它就是 `/ask` 用的那一个（启动期已加载）。**编码口径只有一处**（S8 的 FR-003）——
    这里 MUST NOT 自己构造 `Encoder`，那会多出一个可能在别处加载不同权重的入口。
    """

    # 延迟导入：`backend.query` 会牵出 torch（约 2.3 GB 的权重）。
    # 本模块在 `/ask` 的路由链上被导入，而那条链上并不需要这个函数 ——
    # 放在模块级会让 import 成本提前到进程启动。
    from backend.query import service as query_service
    from backend.retrieve import DIM

    try:
        vector = query_service.get_encoder().encode_query(text)
    except Exception as exc:  # noqa: BLE001 —— 见上：失败即拒答，但必须留痕
        logger.error("encode_query_failed error=%s", type(exc).__name__)
        return None

    values = vector.tolist()

    # ⚠️ 维度不符 MUST 抛错而不是继续查 —— 与 `backend/retrieve/store.py` 的
    #    `search_semantic` 同一处置：真发生了说明指纹门禁被绕过或索引被换过，
    #    此时"少一条结果"比"用错维度的向量查到一堆看似合理的垃圾"好得多。
    if len(values) != DIM:
        logger.error("encode_query_dim_mismatch expected=%d actual=%d", DIM, len(values))
        return None

    return values


async def generation_deltas(
    question: str,
    answer_id: str,
    passages: list,
    config: AppConfig,
    holder: dict,
    history: list | None = None,
) -> AsyncIterator[str]:
    """流式生成正文。**MUST NOT 让异常冒泡到调用方。**

    逐段产出模型增量，并在**成功**时把 `GenerationResult` 放进 `holder["result"]`。
    任何失败都不放 —— 调用方据此走拒答路径。

    ---

    ⚠️ **产出的是"未校验"的原始文本。**

    引用白名单校验（`docs/05` §3.1.7）需要完整正文，而流式输出没法等。所以：
    这里吐给用户的是模型原样，**终帧的 `answer_text` 才是校验过的版本**。前端
    `transcript.js` 的 `renderDone` 会用终帧整体替换 —— S7 的规格写明"渐进渲染
    只是预览，最终显示的必须与服务端逐字一致"。**引用校验正是这条设计要挡的
    差异，不是需要消除的副作用。**

    ⚠️ **失败不冒泡**（与 `retrieve` 同一条推理）：流已经开始，异常无法变成
    HTTP 状态码；冒泡只会让客户端拿到一个中断的流 —— 与"网络断了"不可区分。
    这里 MUST 留 ERROR 日志，因此失败仍**可查**。

    ⚠️ **`mode` 固定为 `patient`。** `docs/05` §3.1.2 的 `AskRequest` 只有
    `question` 一个字段，没有让用户选模式的位置；而 `docs/01`/`02` 与
    constitution 原则 V 把本系统定位为**面向群众**。开放模式选择是一次契约
    变更（要改 §3.1.2、前端请求体、以及"面向群众"的定位），MUST 走独立的
    规格，MUST NOT 在这里顺手加一个可选字段。
    """

    generation = None
    try:
        generation = start_generation(
            question,
            passages,
            mode=DEFAULT_MODE,
            answer_id=answer_id,
            config=config,
            history=history,
        )
    except Exception as exc:  # noqa: BLE001 —— 见函数文档字符串
        # 到这一步还失败，说明是**参数或提示词**的问题（片段为空、模式非法、
        # 提示词文件缺失）—— 都是编程错误，不是网络问题。这类错误不会自愈，
        # 因此不重试、不静默，直接判为生成失败并记 ERROR。
        logger.error(
            "generation_setup_failed answer_id=%s error=%s: %s",
            answer_id,
            type(exc).__name__,
            exc,
        )
        return

    try:
        async for delta in generation.deltas():
            yield delta
    except LLMError as exc:
        logger.error(
            "generation_failed answer_id=%s attempts=%s reason=%s",
            answer_id,
            exc.attempts,
            exc.message.replace("\n", " "),
        )
        return

    result = generation.result
    if result is None:
        logger.error("generation_no_result answer_id=%s", answer_id)
        return

    if result.is_citation_failure:
        # 模型产出了正文，但里面没有一个**有效的**引用编号（或它自己说了资料不足）。
        # docs/05 §4.3：调用方 MUST 转拒答，不得返回 answer_text。
        logger.warning(
            "generation_citation_failure answer_id=%s used=%d chars=%d",
            answer_id,
            len(result.used_citation_ids),
            len(result.answer_text),
        )
        return

    holder["result"] = result
