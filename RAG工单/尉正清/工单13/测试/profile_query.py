# 工单编号：人工智能NLP-RAG项目-RAG性能瓶颈识别与优化
"""应用级性能分析：cProfile 跑一遍完整的问答流程，产出 pstats + snakeviz 页面

工单把「应用级性能分析器」列为识别瓶颈的第一种手段，并点名了
Python 的 cProfile 和 snakeviz。这个脚本就是那一步。

⚠️ **cProfile 的绝对耗时不可信**：它给每个函数调用插桩，本流程里
`_sparse_score` 会被调用 7009 次，插桩开销能把这一段放大好几倍。
所以本脚本的产出只用来**定位热点函数**（看累计耗时占比和调用次数），
真实延迟以 `perf.py` 的阶段埋点为准。这一点写进了性能报告。

用法：
    D:/Anaconda/envs/rag_gd/python.exe profile_query.py [--n 3] [--snakeviz]
--snakeviz 会额外生成 HTML 火焰图（需要装了 snakeviz）
"""
import argparse
import cProfile
import pstats
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
DEV = HERE.parent / "研发"
sys.path.insert(0, str(DEV))

RESULTS = HERE / "results"
QUESTIONS = [
    "平安银行2019年末的拨备覆盖率是多少？",
    "招商银行2020年的营业收入是多少？",
    "中国平安2019年的归母净利润同比增速是多少？",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3, help="跑几道题（每道都做完整问答）")
    ap.add_argument("--snakeviz", action="store_true", help="额外生成 snakeviz HTML")
    ap.add_argument("--kb", default="ccf_competition")
    args = ap.parse_args()

    from config import CACHE_ROOT
    from rag_engine import RAGEngine
    from vector_store import BGEM3VectorStore

    kb = Path(CACHE_ROOT) / args.kb
    print(f"[加载] {kb} ...", flush=True)
    store = BGEM3VectorStore()
    if not store.load(kb):
        raise SystemExit(f"知识库加载失败：{kb}")
    engine = RAGEngine(store)
    print(f"[加载] {len(store.chunks):,} 块就绪", flush=True)

    # 先热一遍，避免把 CUDA 初始化算进 profile
    engine.answer("预热问题：平安银行")
    print("[预热] 完成\n", flush=True)

    prof = cProfile.Profile()
    t0 = time.perf_counter()
    prof.enable()
    for i in range(args.n):
        q = QUESTIONS[i % len(QUESTIONS)]
        r = engine.answer(q)
        print(f"  [{i+1}/{args.n}] {q[:26]}… 端到端 {r['elapsed']:.2f}s", flush=True)
    prof.disable()
    wall = time.perf_counter() - t0
    print(f"\n[cProfile 开启时的总耗时] {wall:.2f}s（{args.n} 题）"
          f" —— 比真实慢是正常的，见文件头的说明", flush=True)

    out = RESULTS / "profile.pstats"
    RESULTS.mkdir(parents=True, exist_ok=True)
    prof.dump_stats(str(out))
    print(f"[pstats] 已写入 {out}\n")

    st = pstats.Stats(str(out))
    st.sort_stats("cumulative")
    print("===== 累计耗时 Top 20（按函数）=====")
    st.print_stats(20)

    if args.snakeviz:
        import subprocess
        html = RESULTS / "profile_snakeviz.html"
        print(f"\n[snakeviz] 生成 {html} ...")
        # snakeviz 的 -s 是「不开浏览器只出 HTML」
        subprocess.run([sys.executable, "-m", "snakeviz", "-s",
                        "-o", str(html), str(out)], check=False)
        print(f"[snakeviz] 完成：{html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
