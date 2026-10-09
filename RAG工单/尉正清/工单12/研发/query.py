# 工单编号：人工智能NLP-RAG项目-LightRAG优化
"""统一查询入口：选择用 RAG 还是 LightRAG 的知识库检索

工单验收点 2 要求「可以选择使用 RAG 还是 LightRAG 的知识库进行检索」，
这个脚本就是那个选择器。

⚠️ **两条路跑在不同的 conda 环境里**（原因见 优化/过程问题记录.md 问题 2）：

    --kb rag       → rag_gd    FlagEmbedding + BGE-M3 + 交叉编码器重排序
    --kb lightrag  → rag_gd1   lightrag-hku + neo4j + ragas

所以本脚本启动时会先自检：环境不对就直接给一句人话提示，
而不是抛一堆 ImportError 让人去猜。

用法：
    python query.py --kb rag      "武汉力源信息技术股份有限公司注册资本是多少？"
    python query.py --kb lightrag "同上"
    python query.py --kb both     "两个都问，并排打印"
"""
import argparse
import asyncio
import re
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _need(module, env_name, why):
    try:
        __import__(module)
    except ImportError:
        raise SystemExit(
            f"[环境不对] 这条路需要 `{module}`，但它不在当前 Python 里。\n"
            f"           请改用 {env_name} 环境运行：\n"
            f"           D:/Anaconda/envs/{env_name}/python.exe {why}"
        )


# ---------------- RAG 那一路 ----------------
def query_rag(question, top_k=None):
    _need("FlagEmbedding", "rag_gd", "query.py --kb rag \"问题\"")
    from config import CACHE_ROOT, TOP_K
    from rag_engine import RAGEngine
    from vector_store import BGEM3VectorStore

    kb = Path(CACHE_ROOT) / "prospectus12"
    if not kb.exists():
        raise SystemExit(f"[缺少知识库] {kb} 不存在，先在 rag_gd 里跑 build_kb.py")
    store = BGEM3VectorStore()
    if not store.load(kb):
        raise SystemExit(f"[知识库不可用] {kb} 加载失败，重建：python build_kb.py --no-cache")

    engine = RAGEngine(store, top_k=top_k or TOP_K)
    t0 = time.time()
    out = engine.answer(question)
    return {
        "kb": "rag",
        "answer": out["answer"],
        "contexts": [c["text"] for c, _ in out["contexts"]],
        "metas": [{"page": c.get("page"), "doc": c.get("doc"), "type": c.get("type")}
                  for c, _ in out["contexts"]],
        "seconds": round(time.time() - t0, 1),
    }


# ---------------- LightRAG 那一路 ----------------
_PAGE_RE = re.compile(r"【([^】]*?\.pdf)\s*第\s*(\d+)\s*页】")


def split_lightrag_context(raw):
    """把 LightRAG 返回的大块上下文切成与 RAG 可比的小段。

    `only_need_context=True` 返回的是**一整块**（实测 6.4 万字符），里面混了
    两样东西：知识图谱的实体/关系 JSON，以及召回的原文片段。

    原文片段里带着我们灌进去的页码标记（见 corpus.py），据此切开：
      · 检索结果对比要按页算命中率，不切开拿不到页码；
      · RAGAS 的 context_precision 是按「每条上下文」判相关的，
        RAG 那边是 8 条，这边要是 1 条 6 万字的，两边根本不是一个口径。

    返回 (原文片段列表, 页码元信息列表, 图谱数据字符数)。
    图谱数据不放进 contexts —— 它是另一种模态，混进来就没法和 RAG 的
    「检索到的原文块」做同口径比较了，单独统计后在报告里说明。
    """
    text = str(raw)
    first = _PAGE_RE.search(text)
    kg_chars = first.start() if first else len(text)

    hits = list(_PAGE_RE.finditer(text))
    parts, metas = [], []
    for i, m in enumerate(hits):
        end = hits[i + 1].start() if i + 1 < len(hits) else len(text)
        seg = text[m.start():end].strip()
        if seg:
            parts.append(seg)
            metas.append({"doc": m.group(1), "page": int(m.group(2))})
    return parts, metas, kg_chars


async def _query_lightrag_async(question, mode=None):
    from config import LIGHTRAG_QUERY_MODE
    from lightrag import QueryParam
    from lightrag_engine import build_engine

    engine = build_engine()
    await engine.initialize_storages()
    mode = mode or LIGHTRAG_QUERY_MODE
    try:
        t0 = time.time()
        # 分两次调用：一次只取上下文（给 RAGAS 的 context_* 指标用），
        # 一次取生成的答案。检索是确定性的，两次拿到的是同一批证据。
        ctx = await engine.aquery(question, param=QueryParam(
            mode=mode, only_need_context=True))
        ans = await engine.aquery(question, param=QueryParam(
            mode=mode, only_need_context=False))
        contexts, metas, kg_chars = split_lightrag_context(ctx)
        return {
            "kb": "lightrag",
            "answer": ans if isinstance(ans, str) else str(ans),
            "contexts": contexts,
            "metas": metas,
            "kg_chars": kg_chars,
            "mode": mode,
            "seconds": round(time.time() - t0, 1),
        }
    finally:
        await engine.finalize_storages()


def query_lightrag(question, mode=None):
    _need("lightrag", "rag_gd1", "query.py --kb lightrag \"问题\"")
    _need("neo4j", "rag_gd1", "query.py --kb lightrag \"问题\"")
    return asyncio.run(_query_lightrag_async(question, mode))


# ---------------- 命令行 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", choices=["rag", "lightrag", "both"], default="both")
    ap.add_argument("--mode", default=None, help="LightRAG 查询模式 local/global/hybrid/mix")
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("question", nargs="+", help="要问的问题")
    args = ap.parse_args()
    question = " ".join(args.question)

    for kb in (["rag", "lightrag"] if args.kb == "both" else [args.kb]):
        print("=" * 70)
        print(f"[{kb}] {question}")
        print("=" * 70)
        try:
            out = (query_lightrag(question, args.mode) if kb == "lightrag"
                   else query_rag(question, args.top_k))
        except SystemExit as exc:
            print(exc)
            continue
        print(f"\n回答（{out['seconds']}s）：\n{out['answer']}\n")
        print(f"检索到 {len(out['contexts'])} 段上下文")
        for i, c in enumerate(out["contexts"][:3], 1):
            print(f"  [{i}] {c[:110]}...".replace("\n", " "))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
