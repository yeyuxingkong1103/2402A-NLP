"""FastAPI 应用装配。

**装配顺序是被锁定的，不是随意的：**

    中间件 → 异常处理器 → API 路由 → 静态挂载

⚠️ API 路由 MUST 在静态挂载**之前**注册。

Starlette 按**注册顺序**匹配路由。若先把 `frontend/` 挂到 `/`，那么 `/ask`
会被静态处理器接走，它的处理方式是"在 frontend/ 下找一个叫 ask 的文件"，
找不到就返回 404 —— 于是接口看起来像是"路由没写对"。

**而这个过程在启动时不报任何错。** 没有警告、没有异常，只有一个莫名其妙的
404。这是本模块存在的主要理由：把顺序固定下来并写明原因，避免后来者
"整理一下代码顺序"时把它拆掉。
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from . import STATIC_DIR_NAME
from .chat_routes import router as chat_router
from .config import PROJECT_ROOT, AppConfig, load
from .corpus_routes import router as corpus_router
from .errors import register_exception_handlers
from .routes import router

logger = logging.getLogger(__name__)

__all__ = ["create_app", "NoCacheStaticFiles"]


class NoCacheStaticFiles(StaticFiles):
    """静态资源一律带 `Cache-Control: no-cache`。

    ⚠️ **这不是"开发期专用"的临时措施。**

    `no-cache` 不是"不缓存"—— 它是"用缓存之前先向服务端确认有没有变"。
    文件没变时服务端回 304，不重复传输；只有真的变了才重新下载。
    所以它的代价（一次协商往返）在有网络连接的场景下可以忽略，
    而它挡掉的问题很具体：

    **改了前端，用户刷新后看到的还是旧页面。**

    Starlette 的 `StaticFiles` 默认不发任何缓存头，浏览器于是用启发式规则
    长期缓存 JS/CSS。这在传统部署里靠文件名加 hash 解决（`app.a1b2c3.js`），
    而本项目的资源名是固定的 `app.js` / `nav.js` —— 没有那个机制可用。

    真实代价示例：删掉一个导航项后，源码里已经没有它了，浏览器侧栏却照旧
    显示，且**普通刷新无效** —— 排查方向会被引向"是不是没保存""是不是改错
    文件了"，而真相只是缓存。`routes.py` 给 SSE 响应加 `no-cache` 是同一条
    取向（那边防的是"看到上一个问题的答案"）。
    """

    def file_response(self, *args, **kwargs):  # type: ignore[override]
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


def _reconcile_corpus_tasks() -> None:
    """把上次进程遗留的 `processing` 入库任务收敛为确定状态（specs/009 FR-038）。

    失败不阻断启动：收敛是"让界面不说谎"的尽力而为，而服务本身能否启动
    关系到问答链路 —— 后者重要得多。但这不等于静默：异常会被记录，
    因为"任务状态没收敛"正是那种会让用户看到永久转圈、却无从排查的问题。
    """
    try:
        from ..corpus.tasks import reconcile_interrupted

        reconciled = reconcile_interrupted()
        if reconciled:
            logger.warning(
                "有 %d 个入库任务因上次进程退出而未完成，已标记为失败", reconciled
            )
    except Exception:  # noqa: BLE001 —— 见上方说明：不阻断启动，但必须留痕
        logger.exception("入库任务状态收敛失败，界面可能显示过期的「处理中」")


# 检测语料变化的间隔（秒）。
#
# 检查本身是读一个几 KB 的本地 JSON，微秒级，因此间隔可以短。取 10 秒是
# "命令行重跑入库之后，最多 10 秒内服务就能检索到新语料"—— 足够快，
# 又不至于每秒都去碰磁盘。
_CORPUS_WATCH_INTERVAL_S = 10.0


async def _watch_corpus(app: FastAPI) -> None:
    """后台检测语料变化并重建检索索引。**这是命令行入库路径的补丁。**

    ## 为什么需要它

    检索的关键词路与融合查表都用一份常驻内存的语料快照。两条入库路径中：

      * **界面上传** —— 编排器就跑在本进程内，入库成功后直接调
        `set_index()` 刷新（见 `backend/corpus/runner.py`）；
      * **命令行** —— `python -m backend.index_milvus` 是**另一个进程**，
        它能写 Milvus 与清单，却没有任何办法通知本进程里的单例。

    后者若不处理，就会出现和界面路径同样的症状：库里明明有新文档，
    提问却检索不到，而且要等到下次重启服务才恢复。而"绕过界面、直接用
    命令行重跑入库"在本项目里是**正常用法**（管线本来就是 CLI 驱动的）。

    ## 为什么不在请求路径上检查

    `serve.py` 写着"请求期绝不重建"，理由是惰性重建会把构建耗时压给第一次
    提问。本函数不改变那条约束：它在**后台**跑，重建也在线程里做
    （`asyncio.to_thread`），不阻塞事件循环，也不占用任何一次请求。

    ## 判据

    比对 `index_manifest.json` 的 `(total_chunks, pipeline_config_hash)`
    与常驻索引构建时记下的值（见 `bundle.stale_reason`）。选磁盘文件而不是
    去问 Milvus：读本地 JSON 是微秒级，而问库要一次网络往返。
    """
    from ..retrieve import bundle as retrieve_bundle

    while True:
        await asyncio.sleep(_CORPUS_WATCH_INTERVAL_S)

        try:
            current = retrieve_bundle.get_index()
        except Exception:  # noqa: BLE001
            # 索引尚未构建（例如 create_app 被单独使用、没走 serve.py）。
            # 不是错误，下一轮再看。
            continue

        try:
            # ⚠️ 先判"参数是否变了" —— 那一项**不能靠重建解决**。
            #
            # 参数变了意味着切分或编码口径变了，而本进程里的 BGE-M3 仍是旧的。
            # 拿旧编码器的向量去查按新口径建的索引，相似度会"看起来正常但完全
            # 错位"：排序乱掉、引用对不上原文，且没有任何报错。那种结果比
            # "检索不到"糟得多 —— 用户会照着一份错位的引用去核对原文。
            #
            # 因此这里只报警、不重建，让人去重启。
            if retrieve_bundle.pipeline_changed(current):
                logger.error(
                    "**索引参数已变化**（切分/编码口径变了）。本进程仍持旧口径，"
                    "重建语料索引无法修复 —— 用旧编码器查新索引会得到错位的相似度。"
                    "**请重启服务**（backend/serve.py）。"
                )
                await asyncio.sleep(_CORPUS_WATCH_INTERVAL_S * 6)  # 别每 10 秒刷屏
                continue

            reason = retrieve_bundle.stale_reason(current)
        except Exception:  # noqa: BLE001
            logger.exception("判断语料是否变化时出错，本轮跳过")
            continue

        if reason is None:
            continue

        logger.warning("检测到语料变化，正在后台重建检索索引 —— %s", reason)
        try:
            # 线程里构建：拉语料 + BM25 重建是阻塞操作（实测约 1 秒），
            # 放在事件循环里会卡住同一时刻的所有请求。
            fresh = await asyncio.to_thread(
                retrieve_bundle.build_from_config, app.state.config
            )
        except Exception:  # noqa: BLE001 —— 重建失败不该拖垮服务
            # 继续用旧索引：检索结果会少了新文档，但**旧的仍可检索** ——
            # 比让整个检索报错好。失败会每 10 秒重试一次，直到成功。
            logger.exception("后台重建检索索引失败，继续使用旧索引")
            continue

        retrieve_bundle.set_index(fresh)
        logger.info("检索索引已重建：%d 条语料", fresh.size)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """应用生命周期：起后台语料监视 + 探测会话存储，关闭时收掉。

    必须显式取消任务：`asyncio.create_task` 出来的协程不会随进程函数返回而
    自动结束，而一个永远 sleep 的循环在测试里会让事件循环无法退出。
    """

    await _probe_chat_store(app)

    watcher = asyncio.create_task(_watch_corpus(app))
    try:
        yield
    finally:
        watcher.cancel()
        with suppress(asyncio.CancelledError):
            await watcher
        await _close_chat_store(app)


async def _probe_chat_store(app: FastAPI) -> None:
    """启动期主动 ping 一次会话存储。**失败不阻止启动。**

    ---

    ⚠️ **这是 `REDIS_URL` 不进 `REQUIRED_NOW` 这个判定的附条件**
    （research R13）。

    判定"Redis 不可用不该让整个服务起不来"是合理的，但它有一个必须补齐的部分：
    **"Redis 连不上"必须在启动时可见**，而不能推迟到第一次用户提问 ——
    那时用户看到的是一个 503，而部署者会先怀疑服务本身，而不是 Redis。

    ⚠️ **探测放在 lifespan 而不是 `serve.py` 的 sync `main()` 里**：
    `serve.py` 构造客户端时不做 I/O（连接池是惰性的）。若在那里用
    `asyncio.run(client.ping())`，连接会绑在那个**临时**事件循环上，
    到了 uvicorn 的循环里就失效 —— 一个只有"探测过"才会出现的诡异故障。
    在 lifespan 里探测，用的就是真正会跑请求的那个循环。

    ⚠️ 通过 `getattr` 取 `chat_client` 而不是 import `backend.chat`：
    `create_app` 也被测试与 `serve.py` 之外的调用方使用，那里可能根本没有
    会话功能。没有该属性时静默跳过是**正确**的，不是降级 ——
    单轮问答不依赖 Redis。
    """

    client = getattr(app.state, "chat_client", None)
    if client is None:
        return

    try:
        await client.ping()
    except Exception as exc:  # noqa: BLE001 —— 见上：不阻止启动，但必须留痕
        logger.warning(
            "会话存储（Redis）不可达：%s。**服务照常启动** —— "
            "/api/chat/* 会返回 503，其余功能不受影响。请检查 REDIS_URL 与 Redis 是否在运行。",
            type(exc).__name__,
        )
        return

    logger.info("会话存储已就绪（Redis ping 成功）")


async def _close_chat_store(app: FastAPI) -> None:
    """关闭会话存储连接池。

    不关的后果在长驻进程里不明显，但在**测试**里是致命的：每个用例新建一个
    应用就多一个永不释放的连接池，跑几十个用例后会看到一片
    "Unclosed client session" 警告，把真正的失败淹没。
    """

    client = getattr(app.state, "chat_client", None)
    if client is None:
        return
    with suppress(Exception):
        await client.aclose()


def create_app(config: AppConfig | None = None) -> FastAPI:
    """构造应用。`config` 缺省时自行加载（供 `serve.py` 之外的调用方使用）。"""

    config = config or load()
    static_dir = PROJECT_ROOT / STATIC_DIR_NAME

    app = FastAPI(
        title="医知源",
        # 关闭自动文档。
        #
        # 理由不是"少暴露信息"，而是**它会生成一份错误的契约**：
        # OpenAPI 会把这个接口描述成返回 JSON 的普通端点，而它实际返回
        # text/event-stream。一份自动生成的、与真实行为不符的文档，比没有
        # 文档更危险 —— 前端会照着它写解析代码。
        # 真实契约以 specs/006-medical-qa-input/contracts/sse.md 为准。
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        # 后台语料监视（命令行入库路径的补丁，见 _watch_corpus）
        lifespan=_lifespan,
    )

    # 配置在启动期注入，请求路径上不再读环境变量（FR-022）。
    app.state.config = config

    # 0) 僵尸任务收敛（specs/009 FR-038）—— 必须在任何请求之前完成。
    #
    # 进程刚启动时不可能存在真正在跑的入库任务：编排器是本进程内的 asyncio
    # 任务，上一个进程一死它们就都没了。因此此刻任何 `processing` 记录都必然
    # 是遗留的，必须改成确定状态 —— 否则界面会永远转圈（specs/009 Edge Cases）。
    #
    # 放在这里而不是 startup 事件里：它是一次快速的目录扫描，同步执行即可，
    # 且放在构造期能保证"应用可服务"与"状态已收敛"是同一时刻成立的。
    _reconcile_corpus_tasks()

    # 1) 异常处理器 —— 必须早于路由，否则特殊异常路径会走框架默认输出。
    register_exception_handlers(app)

    # 2) API 路由 —— 必须在静态挂载之前，理由见模块文档字符串。
    app.include_router(router)
    #    语料入库路由同为 API，同样必须在静态挂载之前（specs/009 T009）。
    app.include_router(corpus_router)
    #    多轮对话路由同为 API，同样必须在静态挂载之前（specs/010 T042）。
    #    ⚠️ 漏掉这一行不会报错，只会让 `/api/chat/*` 全部 404 ——
    #       因为 StaticFiles 会接走它们，去 frontend/ 下找同名文件。
    app.include_router(chat_router)

    # 3) 静态资源 —— 挂在根路径，`html=True` 让 `/` 返回 index.html。
    #
    # 用 NoCacheStaticFiles 而不是 StaticFiles，理由见那个类的文档字符串
    # （一句话：资源名没有内容指纹，默认缓存会让"改了前端但看不到变化"
    #   反复发生，而排查方向会被引向错误的地方）。
    if static_dir.is_dir():
        app.mount(
            "/",
            NoCacheStaticFiles(directory=str(static_dir), html=True),
            name="static",
        )
    else:
        # 不静默跳过：目录缺失会让页面 404，而原因（"没建 frontend/ 目录"）
        # 与现象（"接口通了但页面打不开"）之间没有可见的因果链。
        logger.warning(
            "静态资源目录不存在，界面将无法访问：%s（期望位置：%s）",
            static_dir,
            Path(STATIC_DIR_NAME),
        )

    return app
