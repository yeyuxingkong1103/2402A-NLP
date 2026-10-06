"""验收脚本：演示检索服务的完整链路和统计信息。

使用 mock 数据模拟完整流程，展示：
- 向量召回条数
- 关键词召回条数
- RRF 融合后条数（去重）
- 重排后条数
"""
from app.retrieval.service import RetrievalService, RetrievalResult
from app.retrieval.context_builder import RetrievedArticle
from app.retrieval.fusion import fuse_results, RankedItem


def mock_vector_retriever_results():
    """模拟向量检索结果"""
    return [
        RetrievedArticle(
            chunk_key=f"vector_chunk_{i}",
            content=f"向量召回的第{i}条内容",
            article_number=str(i),
            document_title=f"法规A",
            source_url=f"https://example.com/a{i}",
            recall_score=0.9 - i * 0.01,
            document_type="法律",
            jurisdiction="中国大陆",
            is_current=True,
        )
        for i in range(1, 21)  # 20条
    ]


def mock_keyword_searcher_results():
    """模拟关键词检索结果（部分与向量结果重复）"""
    results = []
    # 前5条与向量结果重复
    for i in range(1, 6):
        results.append(
            RetrievedArticle(
                chunk_key=f"vector_chunk_{i}",  # 相同的 chunk_key
                content=f"关键词召回的第{i}条内容",
                article_number=str(i),
                document_title=f"法规A",
                source_url=f"https://example.com/a{i}",
                recall_score=0.85 - i * 0.01,
                document_type="法律",
                jurisdiction="中国大陆",
                is_current=True,
            )
        )
    # 后10条是独立的
    for i in range(21, 31):
        results.append(
            RetrievedArticle(
                chunk_key=f"keyword_chunk_{i}",
                content=f"关键词召回的第{i}条内容",
                article_number=str(i),
                document_title=f"法规B",
                source_url=f"https://example.com/b{i}",
                recall_score=0.80 - (i - 21) * 0.01,
                document_type="行政法规",
                jurisdiction="中国大陆",
                is_current=True,
            )
        )
    return results


def demo_retrieval_pipeline():
    """演示完整检索链路"""
    print("=" * 60)
    print("检索服务链路演示")
    print("=" * 60)

    # 模拟两路召回结果
    vector_results = mock_vector_retriever_results()
    keyword_results = mock_keyword_searcher_results()

    print(f"\n【第一步】两路召回")
    print(f"  向量召回: {len(vector_results)} 条")
    print(f"  关键词召回: {len(keyword_results)} 条")

    # 转换为 RankedItem 用于融合
    vector_items = [
        RankedItem(
            chunk_key=art.chunk_key,
            score=art.recall_score,
            source="vector",
            content=art.content,
        )
        for art in vector_results
    ]
    keyword_items = [
        RankedItem(
            chunk_key=art.chunk_key,
            score=art.recall_score,
            source="keyword",
            content=art.content,
        )
        for art in keyword_results
    ]

    # RRF 融合
    fused_items = fuse_results([vector_items, keyword_items])
    print(f"\n【第二步】RRF 融合")
    print(f"  融合后: {len(fused_items)} 条（去重后）")
    print(f"  去重数量: {len(vector_results) + len(keyword_results) - len(fused_items)}")

    # 重排（取 top 5）
    reranked_items = fused_items[:5]
    print(f"\n【第三步】重排")
    print(f"  重排后: {len(reranked_items)} 条")

    # 显示最终结果的 chunk_key 和 RRF 分数
    print(f"\n【最终结果】Top 5")
    for i, item in enumerate(reranked_items, 1):
        both_hit = item.chunk_key.startswith("vector_chunk_") and int(item.chunk_key.split("_")[-1]) <= 5
        marker = " ← 两路都命中" if both_hit else ""
        print(f"  {i}. {item.chunk_key} (RRF={item.score:.4f}){marker}")

    # 统计信息
    stats = {
        "vector_recall_count": len(vector_results),
        "keyword_recall_count": len(keyword_results),
        "fused_count": len(fused_items),
        "reranked_count": len(reranked_items),
    }
    print(f"\n【统计信息】")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    demo_retrieval_pipeline()
