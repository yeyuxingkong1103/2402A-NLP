# 工单编号：人工智能NLP-RAG项目-RAG性能瓶颈识别与优化
"""优化正确性对拍：确认倒排索引给出的分数与原来的全表扫描**逐位一致**

性能优化最容易出的不是崩溃，而是**悄悄算错** —— 稀疏打分换了实现，
如果分数有偏差，检索排序就会变，答案质量下降但没人会立刻发现。
所以在跑性能对比之前，先用原来的算法对拍一遍。

原算法（全表扫描）：
    score[d] = Σ_t  query_w[t] * doc_w[d].get(t, 0)
新算法（倒排索引）：只遍历 t 命中的文档累加。

两者数学上恒等，差异只可能来自实现 bug。本脚本对 10 道测试题逐题比对。

用法：
    D:/Anaconda/envs/rag_gd/python.exe verify_optimization.py
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
DEV = HERE.parent / "研发"
sys.path.insert(0, str(DEV))


def old_sparse_scores(query_sparse, docs):
    """优化前的实现：对每篇文档遍历查询词。"""
    out = []
    for d in docs:
        if not query_sparse or not d:
            out.append(0.0)
            continue
        out.append(float(sum(query_sparse.get(t, 0.0) * d.get(t, 0.0)
                             for t in query_sparse)))
    return out


def main():
    import numpy as np
    from config import CACHE_ROOT
    from vector_store import BGEM3VectorStore

    import ccf_testset
    QUESTIONS = [t["question"] for t in ccf_testset.TEST_SET]

    kb = Path(CACHE_ROOT) / "ccf_competition"
    print(f"[加载] {kb} ...", flush=True)
    store = BGEM3VectorStore()
    if not store.load(kb):
        raise SystemExit("知识库加载失败")
    print(f"[加载] {len(store.chunks):,} 块；倒排词项 {len(store.sparse_inv or {}):,}",
          flush=True)

    bad = 0
    for i, q in enumerate(QUESTIONS, 1):
        _, q_sparse = store._encode([q])
        if not q_sparse:
            print(f"  [{i}] 跳过（无稀疏向量）")
            continue
        old = np.array(old_sparse_scores(q_sparse[0], store.sparse_emb),
                       dtype=np.float32)
        new = store._sparse_scores(q_sparse[0])
        max_diff = float(np.abs(old - new).max())
        # 浮点累加顺序不同会有极小误差，用相对容差判断
        rel = max_diff / (float(np.abs(old).max()) or 1.0)
        ok = rel < 1e-5
        if not ok:
            bad += 1
        print(f"  [{i}] {q[:26]}…  最大绝对差 {max_diff:.3e}  "
              f"相对差 {rel:.2e}  {'✓ 一致' if ok else '✗ 不一致'}")

    print(f"\n{'✅ 全部一致，优化没有改变计算结果' if not bad else f'❌ 有 {bad} 题不一致'}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
