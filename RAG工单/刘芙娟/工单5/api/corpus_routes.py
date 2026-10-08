"""语料入库的 HTTP 路由（specs/009）。

与 `routes.py`（S7 提问链路）**分开**，理由不是"文件太长"，而是：

1. `routes.py` 被记录为 S7 契约的落点（见其模块文档字符串）。把语料入库的
   端点混进去，会让"这个文件对应哪个特性"变得不可回答，而本仓其余 8 个特性
   都维持着"一个特性一个包"的可追溯性。
2. 两条链路的失败模式完全不同：问答链路要保证回答不中断，入库链路可以失败
   重试。分开后，入库子进程崩溃不会触碰问答路由的任何一行代码。

本模块只做 **HTTP ↔ 模型** 的协议转换 —— 与 `routes.py` 同一条约束：
MUST NOT 含业务判定。编排在 `backend/corpus/runner.py`，数据在
`backend/corpus/store.py`。
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from ..corpus import (
    TASK_INTERRUPTED,
    TASK_NEEDS_DECISION,
    TASK_PROCESSING,
    TASK_SUCCEEDED,
)
from ..corpus import decisions as corpus_decisions
from ..corpus import stats as corpus_stats
from ..corpus import store as corpus_store
from ..corpus import tasks as corpus_tasks
from ..corpus import runner as corpus_runner
from ..corpus.runner import (
    done_payload,
    prepare_retry,
    run_pipeline,
    snapshot_payload,
)
from ..corpus.uploads import UploadRejected, receive
from . import (
    CORPUS_CHUNKS_PATH,
    CORPUS_DECISIONS_PATH,
    CORPUS_DOC_PATH,
    CORPUS_LIST_PATH,
    CORPUS_PROGRESS_PATH,
    CORPUS_RETRY_PATH,
    CORPUS_UPLOAD_PATH,
    ERR_BAD_ANSWERS,
    ERR_CORPUS_BUSY,
    ERR_DOC_DELETE_ROLLBACK,
    ERR_DOC_NOT_FOUND,
    ERR_DOC_PROCESSING,
    ERR_INTERNAL,
    ERR_INVALID_UPLOAD,
    ERR_NOT_FOUND,
    ERR_NO_DECISIONS,
    ERR_RETRY_NO_FILE,
    EV_CORPUS_DONE,
    EV_CORPUS_SNAPSHOT,
    EV_CORPUS_STEP,
    MSG_BAD_ANSWERS,
    MSG_CORPUS_BUSY,
    MSG_DECISION_BUSY,
    MSG_DOC_NOT_FOUND,
    MSG_DOC_PROCESSING,
    MSG_INTERNAL,
    MSG_NO_DECISIONS,
    MSG_RETRY_BUSY,
    MSG_RETRY_NO_FILE,
)
from .errors import error_response, request_id_of
from .events import sse_event
from .sse_headers import SSE_HEADERS

logger = logging.getLogger(__name__)

router = APIRouter()

# SSE 上检查任务状态的间隔（秒）。
#
# 取 0.3 而非 1.0：specs/009 SC-002 要求"每步完成后 1 秒内界面更新"，
# 而这条链路有两段延迟 —— 服务端轮询间隔 + 网络往返。0.3 秒让轮询几乎不
# 占预算，留出的余量足以覆盖抖动。
#
# 代价是每次检查都要读一次任务 JSON。该文件是几百字节，且只在有活跃订阅时
# 才读，可以忽略。
_POLL_INTERVAL_S = 0.3

# 一条进度流的最长存活时间（秒）。
#
# 兜底而非限流：编排器每步都有超时，任务终会进入终态。这个上限防的是
# "编排器因某种未预料的路径没能写出终态"——那种情况下若不设上限，一条流会
# 无限期挂在那里，而用户看到的是永远转圈。
#
# 取 2 小时：远大于任何正常入库（分钟级），因此正常情况下永不触发。
_MAX_STREAM_S = 7200.0

# 进度流的终止状态。
#
# ⚠️ `needs_decision` **在此集合中**：任务停在等人裁决的位置时，不会再有
#    任何事件产生（除非用户去答复）。若不终止，流会一直挂到 _MAX_STREAM_S
#    的上限，用户看到的就是"卡住了"。
_TERMINAL = {TASK_SUCCEEDED, "failed", TASK_INTERRUPTED, TASK_NEEDS_DECISION}

# 后台编排任务的强引用。
#
# ⚠️ 必须有。`asyncio.create_task` 返回的 Task 若无人持有，可能被垃圾回收，
# 任务会在执行到一半时**静默消失** —— 没有异常、没有日志，只是不再推进。
# 这是 Python 里一个著名且极难定位的坑。
_BACKGROUND: set[asyncio.Task] = set()


def _json_error(code: str, message: str, status_code: int, request_id: str) -> JSONResponse:
    """统一错误体的薄包装，避免每个端点重复三行。"""
    return error_response(code, message, status_code, request_id)


@router.post(CORPUS_UPLOAD_PATH, status_code=202)
async def upload(request: Request, file: UploadFile) -> JSONResponse:
    """接收一份 PDF，创建入库任务。

    校验顺序见 `contracts/http.md` §1。**并发检查放在最前**，与契约里写的
    次序（文件校验在前）有一处刻意的偏差，理由如下：

    两项都成立时（既有任务在跑、文件又不合法），先报"文件不是 PDF"会让用户
    去改文件，改完再传一次，然后才被告知"其实有任务在跑"。先报并发能让用户
    一步走到正确的动作。对只满足其中一项的请求，两种次序的结果完全相同。
    """
    request_id = request_id_of(request)

    active = corpus_tasks.active_task()
    if active is not None:
        # 拒绝而非排队（research R8）：五个步骤共享 data/ 下的工作区并写同一个
        # Milvus collection，并发会让两份文档的产物交错，并触发非预期的先删后插。
        logger.info("拒绝上传：已有任务进行中 task=%s", active.task_id)
        return _json_error(ERR_CORPUS_BUSY, MSG_CORPUS_BUSY, 409, request_id)

    task_id = str(uuid.uuid4())

    try:
        stored = await receive(file, task_id)
    except UploadRejected as exc:
        logger.info("拒绝上传：%s", exc.message)
        return _json_error(ERR_INVALID_UPLOAD, exc.message, 422, request_id)

    # is_rerun：该 doc_id 是否已在 manifest 中（FR-008）。
    # 界面据此提示"该文档已存在，本次为重跑"，MUST NOT 静默当作新增。
    is_rerun = corpus_stats.manifest_document(stored.doc_id) is not None

    task = corpus_tasks.new_task(
        task_id=task_id,
        file_name=stored.file_name,
        file_size=stored.file_size,
        source_hash=stored.source_hash,
        doc_id=stored.doc_id,
        stored_path=stored.stored_path,
        is_rerun=is_rerun,
    )
    corpus_tasks.save_task(task)

    _start_background(task)

    logger.info(
        "接收上传 task=%s doc=%s file=%s size=%d rerun=%s",
        task_id,
        stored.doc_id,
        stored.file_name,
        stored.file_size,
        is_rerun,
    )

    return JSONResponse(
        status_code=202,
        content={
            "task_id": task.task_id,
            "doc_id": task.doc_id,
            "file_name": task.file_name,
            "file_size": task.file_size,
            "is_rerun": is_rerun,
        },
        headers={"Location": CORPUS_PROGRESS_PATH.format(task_id=task.task_id)},
    )


def _start_background(task, start_from: str | None = None) -> None:
    """把编排器放到后台跑，与本次 HTTP 请求的生命周期解耦。

    **必须解耦**：上传响应要立刻返回（SC-001 要求 3 秒内看到第一步亮起），
    而流水线要跑几分钟。若在请求里 await 它，响应会等到全部跑完才发出，
    界面上什么都看不到。

    `start_from` 供「裁决后续跑」使用（见 runner.prepare_retry）。
    """
    bg = asyncio.create_task(_drain(task, start_from))
    _BACKGROUND.add(bg)
    bg.add_done_callback(_BACKGROUND.discard)


async def _drain(task, start_from: str | None = None) -> None:
    """消费编排器的事件流。状态由编排器自己落盘，这里只负责驱动。"""
    try:
        async for _ in run_pipeline(task, start_from):
            pass
    except Exception:  # noqa: BLE001
        # 编排器内部已把可预料的失败写成终态；能飘到这里的是未预料的异常。
        # 必须留痕（否则任务会停在 processing，界面永远转圈），同时把任务
        # 落成一个确定状态 —— 这是 FR-038"不得无限停留在处理中"的要求，
        # 不因失败原因是"未预料"而豁免。
        logger.exception("流水线异常终止 task=%s", task.task_id)
        try:
            fresh = corpus_tasks.load_task(task.task_id)
            if fresh is not None and fresh.status == TASK_PROCESSING:
                fresh.status = "failed"
                fresh.error = "处理过程中发生未预期的错误，请查看服务端日志"
                corpus_tasks.save_task(fresh)
        except Exception:  # noqa: BLE001
            logger.exception("写入失败终态也失败了 task=%s", task.task_id)


@router.get(CORPUS_PROGRESS_PATH)
async def progress(request: Request, task_id: str) -> StreamingResponse:
    """订阅任务进度。

    **任务不存在返回 404，而不是 200 空流。** 空流会让前端永远等待 ——
    它无法区分"任务还没有状态"与"这个 task_id 根本不存在"，于是只能一直
    显示加载中。契约见 `contracts/sse.md`。
    """
    request_id = request_id_of(request)

    if corpus_tasks.load_task(task_id) is None:
        return _json_error(
            ERR_NOT_FOUND, MSG_DOC_NOT_FOUND, 404, request_id
        )  # type: ignore[return-value]

    return StreamingResponse(
        _progress_events(task_id),
        media_type="text/event-stream",
        # 与 /ask 同一个常量，不复制第二份（specs/009 research R1）
        headers=SSE_HEADERS,
    )


async def _progress_events(task_id: str):
    """产出进度事件。

    设计要点：**本端点不订阅编排器的内存状态，而是轮询任务记录文件。**

    这样做的收益是"重连的正确性不需要额外机制"—— 编排器把每一步的状态
    落盘，进度流只是这份记录的一个只读视图。于是：

    * 刚上传就订阅 → 读到全 pending 的快照
    * 处理到一半刷新重连 → 读到部分完成的快照，界面立刻正确
    * 任务已结束才订阅 → 读到终态，随即关闭

    三种情形用同一套逻辑覆盖。若改成"编排器直接往流里推"，这三种情形都要
    单独处理，而且断线期间的增量会永久丢失。
    """
    task = corpus_tasks.load_task(task_id)
    if task is None:
        return

    # 快照必须是**第一个**事件，且恰好一次。
    yield sse_event(EV_CORPUS_SNAPSHOT, snapshot_payload(task))

    seen = {s.key: s.status for s in task.steps}
    elapsed = 0.0

    while task.status not in _TERMINAL:
        if elapsed >= _MAX_STREAM_S:
            logger.error("进度流超过最长存活时间 task=%s", task_id)
            break

        await asyncio.sleep(_POLL_INTERVAL_S)
        elapsed += _POLL_INTERVAL_S

        fresh = corpus_tasks.load_task(task_id)
        if fresh is None:
            # 记录被删了（例如文档已被删除）。终止流，不报错 ——
            # 前端会把流关闭当作"结束了"，并刷新列表。
            logger.info("任务记录已消失，进度流结束 task=%s", task_id)
            return

        task = fresh
        for step in task.steps:
            if seen.get(step.key) != step.status:
                seen[step.key] = step.status
                yield sse_event(
                    EV_CORPUS_STEP,
                    {
                        "key": step.key,
                        "label": step.label,
                        "status": step.status,
                        "detail": step.detail,
                        "error": step.error,
                        "started_at": step.started_at,
                        # 每一步都带上最新进度 —— 前端据此重设进度条的
                        # 速率与终点（见 contracts/sse.md 与 runner.progress_percent）
                        "progress": corpus_runner.progress_percent(task),
                    },
                )

    yield sse_event(EV_CORPUS_DONE, done_payload(task))


@router.get(CORPUS_LIST_PATH)
async def list_docs(request: Request, offset: int = 0, limit: int = 20) -> JSONResponse:
    """已入库文档列表（contracts/http.md §3）。

    分页参数做**夹取**而不是报错：`limit` 超上限时截到上限、为 0 或负数时
    用默认值。理由是这两个参数只影响返回多少条，不改变语义 —— 为它们返回
    422 会让前端因为一个无关紧要的越界值而整页失败。

    `offset` 为负则归零；`limit` 上限 100，防止一次拉全库。
    """
    request_id = request_id_of(request)
    offset = max(0, offset)
    limit = 100 if limit > 100 else (limit if limit > 0 else 20)

    try:
        payload = corpus_store.list_documents(offset=offset, limit=limit)
    except Exception:  # noqa: BLE001
        logger.exception("读取文档列表失败")
        return _json_error(
            ERR_INTERNAL, MSG_INTERNAL, 500, request_id
        )  # type: ignore[return-value]

    return JSONResponse(content=payload)


@router.post(CORPUS_RETRY_PATH)
async def retry(request: Request, doc_id: str) -> JSONResponse:
    """重新入库：从第一个未成功的步骤续跑。

    这是失败/待裁决行唯一的"再来一次"入口。已成功的步骤（解析、清洗）不重跑
    —— 解析是整条链路上最贵的一步，重跑它只是白等几分钟，而结果必然相同。

    ⚠️ 复用的是**原来那条任务记录与原来那份上传文件**，不新建任务、不要求
       重新上传。上传件仍在 `data/uploads/{task_id}/` 下（`stored_path`）。
       文件若已被清掉，只能请用户重新上传 —— 那种情况下 MUST 明确告知，
       而不是启动一条注定在第一步失败的流水线。
    """
    request_id = request_id_of(request)

    task = corpus_tasks.latest_task_for_doc(doc_id)
    if task is None:
        return _json_error(
            ERR_DOC_NOT_FOUND, MSG_DOC_NOT_FOUND, 404, request_id
        )  # type: ignore[return-value]

    active = corpus_tasks.active_task()
    if active is not None and active.task_id != task.task_id:
        return _json_error(
            ERR_CORPUS_BUSY, MSG_RETRY_BUSY, 409, request_id
        )  # type: ignore[return-value]

    if task.status == TASK_PROCESSING:
        return _json_error(
            ERR_DOC_PROCESSING, MSG_DOC_PROCESSING, 409, request_id
        )  # type: ignore[return-value]

    if not Path(task.stored_path).is_file():
        return _json_error(
            ERR_RETRY_NO_FILE, MSG_RETRY_NO_FILE, 409, request_id
        )  # type: ignore[return-value]

    try:
        start_from = prepare_retry(task)
    except Exception:  # noqa: BLE001
        logger.exception("重置任务失败 doc=%s", doc_id)
        return _json_error(
            ERR_INTERNAL, MSG_INTERNAL, 500, request_id
        )  # type: ignore[return-value]

    _start_background(task, start_from)

    logger.info("重新入库 doc=%s task=%s 从 %s 续跑", doc_id, task.task_id, start_from)

    return JSONResponse(
        status_code=202,
        content={
            "doc_id": doc_id,
            "task_id": task.task_id,
            "resumed_from": start_from,
        },
        headers={"Location": CORPUS_PROGRESS_PATH.format(task_id=task.task_id)},
    )


@router.get(CORPUS_DECISIONS_PATH)
async def get_decisions(request: Request, doc_id: str) -> JSONResponse:
    """列出该文档的分块待裁决项，以及每项**当前会应用**的取值。

    `applied` 是考虑四档优先级之后的结果，不是"用户为它单独填了什么" ——
    界面据此显示"已答复/未答复"，与服务端脚本的判断保持一致。
    """
    request_id = request_id_of(request)

    if corpus_stats.manifest_document(doc_id) is None:
        task = corpus_tasks.latest_task_for_doc(doc_id)
        if task is None:
            return _json_error(
                ERR_DOC_NOT_FOUND, MSG_DOC_NOT_FOUND, 404, request_id
            )  # type: ignore[return-value]

    try:
        payload = corpus_decisions.summarize(doc_id)
    except Exception:  # noqa: BLE001
        logger.exception("读取待裁决项失败 doc=%s", doc_id)
        return _json_error(
            ERR_INTERNAL, MSG_INTERNAL, 500, request_id
        )  # type: ignore[return-value]

    return JSONResponse(content=payload)


@router.post(CORPUS_DECISIONS_PATH)
async def post_decisions(request: Request, doc_id: str) -> JSONResponse:
    """写回裁决答复，并从分块步续跑。

    请求体：`{"answers": {"<作用域键>": "<选项>"}}`

    作用域键按 `chunk_clean.py` 的四档约定（精确 / parent: / kind: / `*`）。
    本端点**不解释**这些键的语义，只负责落盘并触发续跑 —— 语义归
    `backend/chunk/decide.py` 所有，在这里再实现一遍必然与它漂移。
    """
    request_id = request_id_of(request)

    task = corpus_tasks.latest_task_for_doc(doc_id)
    if task is None:
        return _json_error(
            ERR_DOC_NOT_FOUND, MSG_DOC_NOT_FOUND, 404, request_id
        )  # type: ignore[return-value]

    # 续跑同样受单任务限制：否则会出现"一边处理 A 一边裁决 B"，
    # 两者都写 data/chunks/ 与同一个 Milvus collection。
    active = corpus_tasks.active_task()
    if active is not None and active.task_id != task.task_id:
        return _json_error(
            ERR_CORPUS_BUSY, MSG_DECISION_BUSY, 409, request_id
        )  # type: ignore[return-value]

    if not corpus_decisions.pending_decisions(doc_id):
        return _json_error(
            ERR_NO_DECISIONS,
    ERR_RETRY_NO_FILE, MSG_NO_DECISIONS, 409, request_id
        )  # type: ignore[return-value]

    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        return _json_error(
            ERR_BAD_ANSWERS, MSG_BAD_ANSWERS, 422, request_id
        )  # type: ignore[return-value]

    answers = (body or {}).get("answers") if isinstance(body, dict) else None
    if not isinstance(answers, dict) or not answers:
        return _json_error(
            ERR_BAD_ANSWERS, MSG_BAD_ANSWERS, 422, request_id
        )  # type: ignore[return-value]
    if not all(isinstance(k, str) and isinstance(v, str) for k, v in answers.items()):
        return _json_error(
            ERR_BAD_ANSWERS, MSG_BAD_ANSWERS, 422, request_id
        )  # type: ignore[return-value]

    try:
        corpus_decisions.write_answers(doc_id, answers)
        start_from = prepare_retry(task)
    except Exception:  # noqa: BLE001
        logger.exception("写入裁决答复失败 doc=%s", doc_id)
        return _json_error(
            ERR_INTERNAL, MSG_INTERNAL, 500, request_id
        )  # type: ignore[return-value]

    _start_background(task, start_from)

    logger.info("裁决已提交 doc=%s 条目=%d 从 %s 续跑", doc_id, len(answers), start_from)

    return JSONResponse(
        status_code=202,
        content={
            "doc_id": doc_id,
            "task_id": task.task_id,
            "resumed_from": start_from,
            "answer_count": len(answers),
        },
        headers={"Location": CORPUS_PROGRESS_PATH.format(task_id=task.task_id)},
    )


@router.get(CORPUS_CHUNKS_PATH)
async def doc_chunks(request: Request, doc_id: str) -> JSONResponse:
    """某文档的全部分块（contracts/http.md §4）。"""
    request_id = request_id_of(request)

    doc = corpus_stats.manifest_document(doc_id)
    if doc is None:
        return _json_error(
            ERR_DOC_NOT_FOUND, MSG_DOC_NOT_FOUND, 404, request_id
        )  # type: ignore[return-value]

    task = corpus_tasks.latest_task_for_doc(doc_id)
    if task is not None and task.status == TASK_PROCESSING:
        return _json_error(
            ERR_DOC_PROCESSING, MSG_DOC_PROCESSING, 409, request_id
        )  # type: ignore[return-value]

    try:
        payload = corpus_store.list_chunks(doc_id)
    except Exception:  # noqa: BLE001
        logger.exception("读取分块失败 doc=%s", doc_id)
        return _json_error(
            ERR_INTERNAL, MSG_INTERNAL, 500, request_id
        )  # type: ignore[return-value]

    return JSONResponse(content=payload)


@router.delete(CORPUS_DOC_PATH)
async def delete_doc(request: Request, doc_id: str) -> JSONResponse:
    """彻底删除一份已入库文档（contracts/http.md §5）。

    二次确认由前端承担（FR-029）—— 服务端不再弹窗，但它仍然拒绝处理中的文档，
    因为那不是"确不确认"的问题，而是操作本身没有意义。
    """
    request_id = request_id_of(request)

    try:
        result = corpus_store.delete_document(doc_id)
    except corpus_store.DocNotFound:
        return _json_error(
            ERR_DOC_NOT_FOUND, MSG_DOC_NOT_FOUND, 404, request_id
        )  # type: ignore[return-value]
    except corpus_store.DocProcessing:
        return _json_error(
            ERR_DOC_PROCESSING, MSG_DOC_PROCESSING, 409, request_id
        )  # type: ignore[return-value]
    except corpus_store.DeleteRollback as exc:
        # 回滚发生过，库可能不一致。message 里带备份路径 —— 那是用户唯一
        # 能拿到的可用信息（contracts/http.md §5 的失败矩阵）。
        return _json_error(
            ERR_DOC_DELETE_ROLLBACK, exc.message, 500, request_id
        )  # type: ignore[return-value]
    except Exception:  # noqa: BLE001
        logger.exception("删除失败 doc=%s", doc_id)
        return _json_error(
            ERR_INTERNAL, MSG_INTERNAL, 500, request_id
        )  # type: ignore[return-value]

    return JSONResponse(
        content={
            "doc_id": result.doc_id,
            "deleted_chunks": result.deleted_chunks,
            "artifacts_removed": result.removed,
            "warnings": result.warnings,
        }
    )
