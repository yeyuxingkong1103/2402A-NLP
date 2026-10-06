#!/usr/bin/env python3
"""端到端验证 RERANKER=bge 真的接管了精排（不是只起了个服务）。

自包含：临时 Milvus Lite + 真实 Ollama bge-m3 向量 + 真实 8001 重排服务。
用一组「语义最相关的那条并非字面最像的那条」样例，观察：

  1. 粗排（RRF 融合）与精排（BGE）的顺序/分数确实不同；
  2. 最终分数落在 BGE sigmoid 区间（0.5~1.0），而非 RRF 的 ~0.016，
     这足以区分「真精排」与「服务掉线悄悄降级回 score_fusion」。

用法：.venv/bin/python scripts/verify_rerank_e2e.py
前置：先 bash scripts/run_rerank.sh 起 8001 重排服务；Ollama 需可达。

与 `scripts/verify_rerank.sh` 的分工：那个只打 8001 一个 HTTP 请求，证明「服务会算分」；
这个把 RAGPipeline 整条链路跑起来，证明「应用真的把精排接上了」——只起服务但
RERANKER 还是 score_fusion 时，前者全绿、后者必挂。

副作用与退出码：只用临时目录（见下面的环境隔离），不碰线上三库；末尾删临时目录。
0 = 三项断言全过；非 0 = 有断言失败（具体原因打印在 ❌ 列表里）。注意断言失败或
中途抛异常时临时目录不会被清（rmtree 在最后才执行），只在 /tmp 下留垃圾，不影响库。
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# 隔离环境：临时 Milvus Lite + 临时 SQLite + 进程内记忆，不碰线上三库。
# 必须在 import app 之前设置，否则 load_dotenv 已把 .env 写进 os.environ。
# 之所以「先设环境变量」就能压过 .env，有双重保险：load_dotenv 默认不覆盖已存在的
# 环境变量；而 config 加载完还会再把真实环境变量写回（见 app/core/config.py 的说明）。
# APP_ENV 置空串是显式关掉 .env.{env} 分层加载。APP_ENV 只从真实环境变量读、且早于
# load_dotenv（见 app/core/config.py），所以终端里 `export APP_ENV=prod` 之后再来跑本脚本，
# .env.prod 会把上面三条全覆盖掉，测试就打到了线上库——置空让它与环境无关。
TMP = tempfile.mkdtemp(prefix="rag-rerank-e2e-")
os.environ["APP_ENV"] = ""
os.environ["MILVUS_DB_URI"] = os.path.join(TMP, "milvus.db")
os.environ["SQL_URL"] = f"sqlite:///{TMP}/app.db"
os.environ["MEMORY_BACKEND"] = "memory"

from app.core.config import load_settings  # noqa: E402
from app.core.embedding import EmbeddingClient  # noqa: E402
from app.core.pipeline import RAGPipeline  # noqa: E402
from app.core.reranker import create_reranker  # noqa: E402
from app.core.retrieve.hybrid_retriever import HybridRetriever  # noqa: E402
from app.core.store.milvus_store import MilvusStore  # noqa: E402

ROLE = "lawyer"
QUERY = "试用期最长可以约定多久？"

# 期望 #1（六个月）语义最相关——问的是「最长」，只有它给的是上限；#3 只有「试用期」三个字
# 字面重合、答的是社保问题，与「最长多久」无关；#4 完全无关。
# 编号是**下标**（0 起），与下面的断言对应。
# #3 是关键：纯 BM25/字面匹配会把它顶上来，只有语义精排才会把它压下去。
DOCS = [
    "劳动合同期限三个月以上不满一年的，试用期不得超过一个月。",
    "三年以上固定期限和无固定期限的劳动合同，试用期不得超过六个月。",
    "同一用人单位与同一劳动者只能约定一次试用期。",
    "试用期期间，用人单位应当为劳动者缴纳社会保险。",
    "用人单位招用劳动者时，应当如实告知工作内容、工作条件、劳动报酬等情况。",
]


def _trunc(text: str, n: int = 26) -> str:
    return text if len(text) <= n else text[:n] + "…"


def main() -> int:
    """跑完整链路并按三条断言判定，返回进程退出码（0 通过 / 1 有失败项）。

    判定刻意用「顺序 + 分数区间」两种互相独立的证据：顺序答的是「结果对不对」，
    分数区间答的是「分是谁给的」——单靠任一条都可能放过错误实现（比如服务在算分、
    但结果没被采用，顺序恰好也对）。
    """
    settings = load_settings()
    # 早失败：RERANKER 没开成 bge 时后面所有断言都没有意义（pipeline 根本不会调精排），
    # 而且要在这里就拦住，否则会在临时库上白跑一遍向量化（要联网，慢）
    assert settings.reranker == "bge", f"期望 RERANKER=bge，实际 {settings.reranker}"

    embedding = EmbeddingClient(settings)
    milvus = MilvusStore(settings)

    vecs = embedding.embed_texts(DOCS)
    # 直接插 Milvus、跳过 SQL 登记：本脚本不验证入库链路，只要检索侧的数据在位
    milvus.insert(
        ROLE,
        [
            {"text": t, "title": "e2e", "source": f"e2e-{i}.md",
             "chunk_index": i, "summary": "", "vector": v}
            for i, (t, v) in enumerate(zip(DOCS, vecs))
        ],
    )

    retriever = HybridRetriever(settings, embedding, milvus)
    # 用 create_reranker 而不是直接 new BGEReranker：验的正是「配置能选出 BGE」这一步，
    # 手写构造会把被测对象换掉，测试就永远绿了
    reranker = create_reranker(settings)
    # llm/sql/memory 传 None：只走检索段，不生成答案（省一次 LLM 调用，也不需要那三个库）
    pipeline = RAGPipeline(
        settings, llm=None, embedding=embedding, milvus=milvus,
        sql=None, memory=None, retriever=retriever, reranker=reranker,
    )

    # 粗排池（RERANK_POOL=20）：这是精排前的输入顺序，score 是 RRF 融合分
    pool = retriever.retrieve(QUERY, ROLE, settings.rerank_pool)
    print("== 粗排（RRF 融合，pool=20）==")
    for i, c in enumerate(pool, 1):
        print(f"  {i}. score={c['score']:.5f}  {_trunc(c['text'])}")

    # 精排后（截断回 TOP_K=5）：score 应被 BGE 重写
    final = pipeline.retrieve(QUERY, ROLE)
    print("\n== 精排后（BGE rerank，top_k=5）==")
    for i, c in enumerate(final, 1):
        print(f"  {i}. score={c['score']:.5f}  {_trunc(c['text'])}")

    # 断言用「有没有问题」收集而不是 assert：三条都想跑完，一次把结论全打出来，
    # 否则修一条跑一次的来回太费（每次都要重新向量化 + 精排）
    problems: list[str] = []
    # 1) 分数必须是 BGE sigmoid 区间（>0.4），RRF 只有 ~0.016，能区分真精排 / 降级
    # 阈值 0.4 取在两个量纲之间靠 BGE 一侧：BGE 实际多在 0.5~1.0，留出的余量
    # 让「模型这次给分偏低」不至于被误报成降级（假阳性会让人白白去查服务）
    if not any(c["score"] > 0.4 for c in final):
        problems.append("最终分数没有 >0.4 的，疑似降级回 score_fusion（RRF 分 ~0.016）")
    # 2) 语义最相关的「六个月」要排在字面陷阱「缴纳社会保险」之前
    # 断言「相对顺序」而不是「必须第一」：问句给的是「最长」，#0（一个月）和 #1（六个月）
    # 都是合理答案，硬要求 #1 第一会把这个脚本变成对模型打分的测试；而「语义相关压过字面重合」
    # 才是要验的 RAG 行为，这条无论谁排第一都成立。
    texts = [c["text"] for c in final]
    pos_best = next((i for i, t in enumerate(texts) if "六个月" in t), None)
    pos_trap = next((i for i, t in enumerate(texts) if "社会保险" in t), None)
    if pos_best is None:
        problems.append("语义最相关的「不得超过六个月」没进 top_k")
    elif pos_trap is not None and pos_trap < pos_best:
        problems.append(
            f"字面陷阱「缴纳社会保险」（#{pos_trap + 1}）压过了语义最相关的「六个月」（#{pos_best + 1}）"
        )
    # 3) 无关的「如实告知」不能被顶到第一
    # 实现上只查「是不是第一」，不查「是不是垫底」：候选只有 5 条且全落在 top_k 内，
    # 垫底几乎是必然事件、没有信息量；被顶到第一才说明精排没起作用。
    if "如实告知" in final[0]["text"]:
        problems.append("无关文档「如实告知」排到了第一")

    shutil.rmtree(TMP, ignore_errors=True)
    if problems:
        print("\n❌ 未通过：")
        for p in problems:
            print("   -", p)
        return 1
    print("\n✅ 通过：RERANKER=bge 已接管精排（粗排→精排顺序改变，最终分数为 BGE 语义分）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
