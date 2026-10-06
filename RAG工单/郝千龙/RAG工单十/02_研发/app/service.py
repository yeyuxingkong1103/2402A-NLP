# -*- coding: utf-8 -*-
# 【金融问答HTTP服务 · service.py】启动建库/加载卷索引，对外提供健康检查与问答接口
# 工单编号：人工智能NLP-RAG-金融问答系统部署
# 说明：容器 ENTRYPOINT 入口，也可在 app 目录下直接 ``python service.py`` 本机运行；
#       首次启动从挂载的 /data 目录 PDF 建库并持久化到 /app/index 卷，
#       后续启动（含换新容器挂同一卷）经来源清单校验直接加载，不再重建。

"""金融问答系统 HTTP 服务（Flask + 纯 CPU 离线 TF-IDF/BM25 检索）。

接口清单：
- GET  /        ：服务说明页（人工探活）；
- GET  /health  ：健康检查，返回服务状态、语料文档数、块数与索引来源；
- POST /ask     ：问答接口，入参 JSON ``{"question": "..."}``，
                  返回 answer、evidence（页码/标题/摘要）、latency_ms。
"""
import logging
import os
import sys
import time
import warnings

# 屏蔽 pdfminer/pdfplumber 在部分招股书页上的常规解析告警，避免容器日志出现噪声；
# 真实异常仍会通过 logging 与 HTTP 状态码暴露
warnings.filterwarnings("ignore")

from flask import Flask, jsonify, request

# 保证“python app/service.py”与“python service.py”两种启动方式均可导入同目录模块
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from config import CONFIG  # noqa: E402  （sys.path 注入后再导入同目录模块）
from chunker import build_chunks  # noqa: E402
from index_store import IndexStore, scan_sources  # noqa: E402
from pdf_parser import parse_pdf  # noqa: E402
from qa_engine import QAEngine  # noqa: E402
from retriever import Retriever  # noqa: E402

SERVICE_NAME = "金融问答系统（招股说明书 RAG 离线服务）"
SERVICE_VERSION = "1.0.0"

# 日志同时输出到标准输出（docker logs 直接可见），带时间与级别
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("finqa")

app = Flask(__name__)

# 运行期全局状态：由启动流程初始化，请求处理只读
_STATE = {
    "engine": None,        # QAEngine 单例
    "sources": [],         # PDF 来源清单
    "chunk_count": 0,
    "index_built": False,  # 本次启动是否执行了重建（False=复用卷内缓存）
    "build_seconds": 0.0,
    "started_at": "",
}


def _build_index_from_pdfs(data_dir: str, index_dir: str) -> IndexStore:
    """扫描数据目录 PDF -> 解析 -> 分块 -> 构建双路索引 -> 持久化到卷。

    :param data_dir: PDF 语料目录（挂载卷）
    :param index_dir: 索引持久化目录（挂载卷）
    :return: 构建完成的 IndexStore
    """
    sources = scan_sources(data_dir)
    if not sources:
        raise FileNotFoundError(f"数据目录 {data_dir} 下未找到任何 PDF 文件")
    all_blocks = []
    for src in sources:
        pdf_path = os.path.join(data_dir, src["name"])
        t0 = time.perf_counter()
        logger.info("解析 PDF：%s（%.2f MB）...",
                    src["name"], src["size"] / 1024 / 1024)
        blocks = parse_pdf(pdf_path)
        logger.info("PDF %s 解析完成：%d 个结构化块，耗时 %.1f 秒",
                    src["name"], len(blocks), time.perf_counter() - t0)
        all_blocks.extend(blocks)
    chunks = build_chunks(all_blocks)
    logger.info("分块完成：共 %d 个检索块", len(chunks))
    store = IndexStore.build(chunks, sources)
    store.save(index_dir)
    return store


def bootstrap() -> None:
    """服务启动引导：命中卷缓存则加载，否则首次建库；异常时给出明确日志。"""
    from datetime import datetime
    _STATE["started_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data_dir, index_dir = CONFIG.data_dir, CONFIG.index_dir
    logger.info("=" * 70)
    logger.info("%s v%s 启动中", SERVICE_NAME, SERVICE_VERSION)
    logger.info("数据目录（PDF 卷）：%s", data_dir)
    logger.info("索引目录（持久化卷）：%s", index_dir)
    os.makedirs(index_dir, exist_ok=True)

    sources = scan_sources(data_dir)
    t0 = time.perf_counter()
    fresh, reason = IndexStore.is_cache_fresh(index_dir, sources)
    if fresh and sources:
        # 新容器挂同一卷：只做反序列化加载，不重新解析 PDF
        logger.info("检测到持久化卷内有效索引：%s", reason)
        logger.info("直接加载卷内索引（无需重建）...")
        store = IndexStore.load(index_dir, sources)
        _STATE["index_built"] = False
    else:
        # 首次启动或语料/版本变更：全量建库
        logger.info("索引缓存未命中：%s", reason)
        store = _build_index_from_pdfs(data_dir, index_dir)
        _STATE["index_built"] = True
    elapsed = time.perf_counter() - t0
    _STATE["build_seconds"] = round(elapsed, 2)
    _STATE["sources"] = sources
    _STATE["chunk_count"] = len(store.chunks)
    _STATE["engine"] = QAEngine(Retriever(store))
    logger.info("索引就绪：文档 %d 个 / 检索块 %d 个 / 加载耗时 %.2f 秒（本次%s）",
                len(sources), len(store.chunks), elapsed,
                "执行重建" if _STATE["index_built"] else "复用卷缓存")
    logger.info("HTTP 服务监听 %s:%s", CONFIG.host, CONFIG.port)
    logger.info("=" * 70)


@app.route("/", methods=["GET"])
def index():
    """服务说明页：浏览器直接访问时给出接口用法，兼作人工探活入口。"""
    return (
        f"<html><head><meta charset='utf-8'><title>{SERVICE_NAME}</title></head>"
        f"<body style='font-family:Microsoft YaHei,Arial;margin:40px;'>"
        f"<h2>{SERVICE_NAME} v{SERVICE_VERSION}</h2>"
        f"<p>工单编号：人工智能NLP-RAG-金融问答系统部署</p>"
        f"<h3>接口用法</h3><ul>"
        f"<li>健康检查：<code>GET /health</code></li>"
        f"<li>金融问答：<code>POST /ask</code>，"
        f"请求体 <code>{{\"question\": \"发行人的注册资本是多少？\"}}</code></li>"
        f"</ul><p>语料文档：{len(_STATE['sources'])} 个；"
        f"检索块：{_STATE['chunk_count']} 个。</p></body></html>"
    ), 200


@app.route("/health", methods=["GET"])
def health():
    """健康检查：索引与引擎就绪即返回 200，否则 503（供 Docker HEALTHCHECK 使用）。"""
    ready = _STATE["engine"] is not None
    payload = {
        "status": "ok" if ready else "initializing",
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "documents": len(_STATE["sources"]),
        "chunks": _STATE["chunk_count"],
        "index_built": _STATE["index_built"],
        "index_load_seconds": _STATE["build_seconds"],
        "started_at": _STATE["started_at"],
    }
    return jsonify(payload), (200 if ready else 503)


@app.route("/ask", methods=["POST"])
def ask():
    """问答接口：入参 JSON {"question": 问题文本}，返回答案、证据与耗时。"""
    engine = _STATE["engine"]
    if engine is None:
        return jsonify({"code": 503, "message": "索引尚未就绪，请稍后重试"}), 503

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"code": 400, "message": "请求体必须为 JSON 对象"}), 400
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"code": 400, "message": "question 字段不能为空"}), 400
    if len(question) > 500:
        return jsonify({"code": 400, "message": "question 长度不能超过 500 字"}), 400

    top_k = data.get("top_k") or CONFIG.final_top_k
    try:
        result = engine.answer(question, top_k=int(top_k))
    except Exception as exc:  # 检索/抽取期异常返回 500，且打印完整堆栈到容器日志
        logger.exception("问答处理异常：%s", exc)
        return jsonify({"code": 500, "message": f"服务内部错误: {exc}"}), 500

    # 证据序列化为前端/测试友好的扁平结构：页码 + 标题 + 摘要 + 双路名次
    evidences = [
        {
            "rank": i + 1,
            "page_no": ev.page_no,
            "title": ev.heading_path or "正文",
            "snippet": ev.parent_text[:300],
            "score": round(ev.score, 4),
            "dense_rank": ev.dense_rank,
            "sparse_rank": ev.sparse_rank,
        }
        for i, ev in enumerate(result.evidences)
    ]
    logger.info("问答：%s -> 第%s页 | 耗时 %dms",
                question,
                ",".join(str(e.page_no) for e in result.evidences),
                result.latency_ms)
    return jsonify({
        "code": 0,
        "question": question,
        "answer": result.answer,
        "evidences": evidences,
        "latency_ms": result.latency_ms,
        "mode": result.mode,
        "timings": result.timings,
    }), 200


if __name__ == "__main__":
    # 先建库/加载索引，再对外监听；首次建库期间 Docker 健康检查为 503/连接失败，
    # 由 HEALTHCHECK 的 start-period 宽限覆盖
    bootstrap()
    # threaded=True 支持并发问答请求；use_reloader 关闭，避免容器内双进程重复建库
    app.run(host=CONFIG.host, port=CONFIG.port, threaded=True,
            debug=False, use_reloader=False)
