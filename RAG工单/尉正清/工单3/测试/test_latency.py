# 工单编号：人工智能NLP-RAG-PDF 文档的表格解析及检索优化
"""性能验收测试：工单 10 个问题，验证「从提问到返回答案不超过 3 秒」

用法：python test_latency.py [PDF路径]
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "研发"
sys.path.insert(0, str(ROOT))

from config import CACHE_ROOT, TEST_QUESTIONS, TOP_K   # noqa: E402
from document import build_chunks, load_pdf       # noqa: E402
from rag_engine import RAGEngine                  # noqa: E402
from vector_store import BGEM3VectorStore         # noqa: E402

DEFAULT_PDF = r"D:\BW\RAG 工单\附件\招股说明书1.pdf"
LIMIT = 3.0


def build_store(pdf_path):
    """优先复用 kb_cache，避免每次重跑都重新解析与编码。"""
    cache_dir = CACHE_ROOT / Path(pdf_path).stem
    store = BGEM3VectorStore()
    if cache_dir.exists() and store.load(cache_dir):
        print(f"复用知识库缓存：{cache_dir}（{len(store.chunks)} 个块）")
        return store

    t0 = time.time()
    pages, tables = load_pdf(pdf_path)
    print(f"PDF 解析：{len(pages)} 页文字、{len(tables)} 个表格，耗时 {time.time()-t0:.1f}s")

    t0 = time.time()
    store.build(build_chunks(pages, tables))
    print(f"向量索引：{len(store.chunks)} 个块，耗时 {time.time()-t0:.1f}s")
    store.save(cache_dir)
    return store


def main(pdf_path):
    print("=" * 66)
    print("性能验收：响应时间（工单要求 <= 3.00s）")
    print("=" * 66)

    engine = RAGEngine(build_store(pdf_path), top_k=TOP_K)
    print(f"\n{'ID':<6}{'耗时(s)':<10}{'检索页码':<26}达标")
    print("-" * 66)

    times = []
    for item in TEST_QUESTIONS:
        started = time.time()
        result = engine.answer(item["question"])
        elapsed = time.time() - started
        times.append(elapsed)
        pages_hit = [c["page"] for c, _ in result["contexts"]]
        flag = "OK" if elapsed <= LIMIT else "超标"
        print(f"{item['id']:<6}{elapsed:<10.2f}{str(pages_hit):<26}{flag}")

    print("-" * 66)
    avg = sum(times) / len(times)
    print(f"平均 {avg:.2f}s | 最快 {min(times):.2f}s | 最慢 {max(times):.2f}s "
          f"| 超标 {sum(1 for t in times if t > LIMIT)}/{len(times)} 题")
    print("结论：" + ("全部达标" if max(times) <= LIMIT else "存在超标题目"))

    # 中英文双语验收
    print("\n多语言验收：")
    for q, tag in [("注册资本是多少？", "中文"),
                   ("What is the registered capital?", "英文")]:
        started = time.time()
        result = engine.answer(q)
        print(f"  [{tag}] {time.time()-started:.2f}s -> {result['answer'][:70]}")

    return times


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PDF)
