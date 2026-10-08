"""医知源服务入口。**本文件是全特性唯一的进程入口。**

运行方式（MUST 用模块形式，以仓库根为工作目录）：

    D:/zg6_Project/9/med_rag/rag/python.exe -m backend.serve

不要用 `rag/python.exe backend/serve.py` —— 那样 `sys.path[0]` 会变成
`backend/` 而不是仓库根，`import backend.api` 会失败。既有管线的
`-m backend.pipeline` 约定同理（docs/05 §5）。

若要留存日志供验收（quickstart §9 会 grep 它）：

    D:/zg6_Project/9/med_rag/rag/python.exe -m backend.serve > .smoke_out/serve.log 2>&1

---

⚠️ **启动前提（specs/008 起新增）：Milvus MUST 已运行。**

混合检索的关键词索引在启动期从向量库拉全量语料（`backend/retrieve/`，FR-010）。
库不可达即**启动失败** —— MUST NOT 以"能启动但关键词路是空的"状态继续。
在此之前（S8），服务在 Milvus 未运行时也能起来（指纹门禁只读
`index_manifest.json`，不连库）。这是本特性带来的**行为变更**。

启动序列（顺序有理由，不要随意调换）：

    ① 读配置            —— 缺失必需项 → 退出码 2
    ② 指纹门禁          —— **最便宜的检查最先**（S8 的取向；它不需要 torch）
    ③ 加载 BGE-M3 权重  —— 约 10 s
    ④ 构建关键词索引    —— 连 Milvus 拉语料 + jieba 预热
    ⑤ 起 HTTP
"""

import logging
import sys
import time

import uvicorn

from backend.api import EXIT_CONFIG, EXIT_OK
from backend.api.app import create_app
from backend.api.config import ConfigError, load
from backend.generate import PromptError
from backend.generate import prompt as generate_prompt
from backend.query import EXIT_MODEL, QueryError, gate, service
from backend.retrieve import EXIT_DEP, RetrievalError, assert_dim_matches_index_package
from backend.retrieve import bundle as retrieve_bundle
from backend.retrieve import lexical as retrieve_lexical

logger = logging.getLogger("backend.serve")

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"


def _force_utf8_output() -> None:
    """把标准输出/错误切到 UTF-8。

    Windows 控制台默认 cp936。管道重定向到日志文件时（quickstart §9 会 grep
    它），中文日志会写成 cp936 字节，`grep` 与编辑器都会读成乱码 ——
    **读不了的日志等于没记**，而本项目排障高度依赖日志（answer_id 关联、
    client_disconnected 线索）。

    必须在 `uvicorn.run` 之前调用：uvicorn 在 run() 内部配置自己的日志处理器，
    那时会捕获已经切换过的流。
    """

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]


def main() -> int:
    _force_utf8_output()
    logging.basicConfig(
        level=logging.INFO, format=LOG_FORMAT, stream=sys.stderr
    )

    # 配置在启动期一次性加载。必需项缺失即失败并非 0 退出，MUST NOT 用默认值兜底
    #（FR-022、constitution 原则 III）。报错只报变量名，不回显值。
    try:
        config = load()
    except ConfigError as exc:
        print(f"启动失败：{exc}", file=sys.stderr)
        print(
            "请复制仓库根的 .env.example 为 .env 并补齐必需项。",
            file=sys.stderr,
        )
        return EXIT_CONFIG

    # 指纹门禁（specs/007 FR-014~FR-018）。
    #
    # ⚠️ **门禁 MUST 在加载模型之前**。
    #
    # 它只读 index_manifest.json 并比对参数，**不需要 torch**（fingerprint()
    # 只读 config.json 与 stat 权重文件）。放在加载之后的话，参数写错要等
    # 7–11 秒的权重加载完才被告知 —— 而那句报错本身一个字都不依赖模型。
    # **便宜的检查放前面**，失败快 70 倍。
    #
    # 只在这里校验一次，不在请求路径上重复：编码参数在进程生命周期内不会变，
    # 每请求比对等于反复读一个 JSON 去回答一个恒定不变的问题。
    #
    # ⚠️ 已知局限：服务运行期间若重新入库（S6 被重跑、索引指纹变了），
    #    本进程仍持旧口径。**重新入库后 MUST 重启服务**。
    #    正解是 docs/05 §3.3 的 /health 里那个 matches_index 检查，
    #    那不在本特性范围内。
    try:
        gate_info = gate.check_gate()
    except QueryError as exc:
        print(f"启动失败：{exc.message}", file=sys.stderr)
        return exc.code

    # 加载 BGE-M3 权重（specs/007 FR-031）。
    #
    # ⚠️ 这一步让启动从 < 1 s 变成 7–11 s，进程私有内存增加约 3.26 GB。
    #    这是**设计内的**，不是卡死。因此前后各打一行 —— 把 10 秒的沉默变成
    #    可见的预期，否则使用者会以为命令打错了（research R7）。
    print("正在加载 BGE-M3 权重（约 2.3 GB，首次约 10 秒）…", flush=True)
    started = time.time()
    try:
        encoder = service.get_encoder()
    except Exception as exc:  # noqa: BLE001 —— 加载失败即启动失败
        # 不以"能提问但算不出向量"的状态继续（research R2）：
        # 那种状态下用户照常拿到 200，只有翻开留存记录才会发现全部失败了 ——
        # 这正是 constitution 原则 III 要防的静默降级。
        print(f"启动失败：无法加载 BGE-M3 权重 —— {exc}", file=sys.stderr)
        print(
            "请确认权重目录存在且完整（EMBED_MODEL_PATH / "
            "backend/embed/__init__.py 的 MODEL_DIR）。",
            file=sys.stderr,
        )
        return EXIT_MODEL
    elapsed = time.time() - started
    print(
        f"权重已加载（{elapsed:.1f} s），编码指纹 "
        f"{str(encoder.fingerprint.get('config_sha256', '?'))[:8]}…",
        flush=True,
    )

    print(
        f"指纹与索引一致（{gate_info['collection']} / "
        f"{gate_info['total_chunks']} chunks）",
        flush=True,
    )

    # 混合检索的关键词索引（specs/008 / FR-010）。
    #
    # ⚠️ **这一步让 Milvus 从"可选"变成"启动的硬前提"。**
    #
    # 在此之前（S8），服务在 Milvus 未运行时也能正常启动 —— 指纹门禁只读
    # index_manifest.json，不连库。接入检索后不能：语料在启动期拉取，
    # 拉不到就没有关键词索引。而"启动成功但每次检索降级"被 spec 的 Edge Case
    # 明令禁止（静默降级会让向量检索坏了这件事永远不被发现）。
    #
    # ⚠️ 顺序：放在加载权重之后只是**不改动既有顺序**（两者的耗时不在一个量级，
    # 且索引构建依赖 Milvus 的可用性，与权重无关）。若将来要缩短启动时间，
    # 这两个步骤可以并行 —— 但那需要先验证 Milvus 客户端与 BM25 索引的线程
    # 安全性（research R12），不在本特性范围内。
    try:
        assert_dim_matches_index_package()
    except RetrievalError as exc:
        print(f"启动失败：{exc.message}", file=sys.stderr)
        return exc.code

    # 提示词自检（specs/008 之后的生成模块）。
    #
    # ⚠️ 放在**连库之前**：它只读一个本地文件，比 Milvus 往返和 jieba 预热都便宜。
    #    与指纹门禁同一取向 —— 便宜的检查放前面。
    #
    # 检查的核心是"三个变量各自有一个真正的注入点"。缺注入点时渲染**不会报错**
    # （没有记号可替换，也就没有残留），但模型手里会是一篇没有原文的提示词 ——
    # 启动期挡下来，比等用户拿到一个没有出处的回答好。
    try:
        prompt_info = generate_prompt.check_prompt()
    except PromptError as exc:
        print(f"启动失败：{exc.message}", file=sys.stderr)
        return EXIT_CONFIG
    print(
        "提示词已加载（%s，%d 字）" % (prompt_info["path"], prompt_info["chars"]),
        flush=True,
    )

    print("正在构建关键词索引（从向量库拉取语料）…", flush=True)
    started = time.time()
    try:
        retrieve_lexical.warm_up()   # jieba 首次调用要加载词典，约 0.5–1 s
        bundle = retrieve_bundle.build_index(
            uri=config.milvus_uri,
            collection=config.milvus_collection,
            candidates=config.retrieval_candidates,
            rrf_k=config.rrf_k,
            admit_rank=config.lexical_admit_rank,
            min_coverage=config.lexical_min_coverage,
            bm25_k1=config.bm25_k1,
            bm25_b=config.bm25_b,
        )
    except RetrievalError as exc:
        print(f"启动失败：{exc.message}", file=sys.stderr)
        print(
            "请确认 Milvus 已启动（docker ps 应含 milvusdb/milvus:v2.6.9）、"
            "MILVUS_URI / MILVUS_COLLECTION 正确，且已完成 S6 入库。",
            file=sys.stderr,
        )
        return exc.code
    except Exception as exc:  # noqa: BLE001 —— 建索引失败即启动失败
        print(f"启动失败：构建关键词索引时出错 —— {exc}", file=sys.stderr)
        return EXIT_DEP

    # 把构建好的索引装进常驻单例。**请求期绝不重建** —— 惰性构建会让第一次
    # 用户提问承担这段耗时，并把"Milvus 不可达"从启动期推迟到请求期。
    retrieve_bundle.set_index(bundle)

    # 下面这几行不是装饰：它们是"关键词索引悄悄是空的"这件事**唯一**能被
    # 观测到的地方。不打印的话，一份只有语义路的检索看起来完全正常。
    matched = bundle.matches_manifest
    if matched is None:
        print("关键词索引就绪（语料条数无法与索引清单比对）", flush=True)
    elif matched:
        print(
            f"关键词索引就绪（{bundle.size} 条，与索引清单一致，"
            f"{time.time() - started:.1f} s）",
            flush=True,
        )
    else:
        print(
            f"关键词索引就绪（库中 {bundle.size} 条，索引清单记 "
            f"{bundle.manifest_total} 条 —— **不一致**，"
            "若刚重新入库属正常，否则清单已过时）",
            flush=True,
        )

    print(
        "检索配置：top_k=%d threshold=%.4f candidates=%d rrf_k=%d "
        "admit_rank=%d min_coverage=%.2f"
        % (
            config.top_k,
            config.similarity_threshold,
            config.retrieval_candidates,
            config.rrf_k,
            config.lexical_admit_rank,
            config.lexical_min_coverage,
        ),
        flush=True,
    )

    app = create_app(config)

    url = f"http://{config.medrag_host}:{config.medrag_port}/"
    print(f"医知源服务已启动  {url}", flush=True)

    # 刻意**不启用 reload**。
    #
    # 热重载会在每次改文件时重启进程并中断进行中的 SSE 流。调试期它看似方便，
    # 实际会掩盖流式行为的真实表现 —— 一个只在长连接下暴露的问题，会在
    # "文件一保存连接就断"的环境里永远复现不出来。
    uvicorn.run(
        app,
        host=config.medrag_host,
        port=config.medrag_port,
        log_level="info",
        access_log=True,
    )
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
