"""演示入口：问一个问题，用真实法条生成带引用的回答。

跑法（在 backend 目录下）：

    python -m app.cli.demo_ask "公司解除劳动合同需要提前多久通知？"

链路：问题 → 检索服务（向量 + 关键词 + RRF融合）→ 问答服务（提示词 + LLM + 护栏）→ 打印回答

完整链路已实现：时效与法域过滤、引用校验、绝对化表述拦截、固定免责声明。
"""

import argparse
import sys

from app.chat.bootstrap import build_default_chat_service


def print_retrieval_stats(stats: dict[str, int]) -> None:
    """打印检索统计信息。"""
    print(f"\n【检索统计】")
    print(f"  向量召回: {stats.get('vector_recall_count', 0)} 条")
    print(f"  关键词召回: {stats.get('keyword_recall_count', 0)} 条")
    print(f"  融合后: {stats.get('fused_count', 0)} 条")
    print(f"  重排后: {stats.get('reranked_count', 0)} 条")


def format_article_label(article_number: str | None, paragraph_number: str | None = None) -> str:
    """拼出"条第X款"的显示文本。

    条号在数据里有两种形态：完整写法（"第四十七条"）与纯编号（"47"），
    两种都要正确显示，不能出现"第 第四十七条 条"这种重复。
    条号缺失时如实标注，不用别的字段顶替。
    """
    if not article_number:
        return "未识别条号"
    text = str(article_number).strip()
    label = text if text.startswith("第") else f"第 {text} 条"
    if paragraph_number:
        label += f"第 {paragraph_number} 款"
    return label


def print_sources(sources: list[dict]) -> None:
    """打印引用的法源列表。

    字段名与 ChatService 产出的法源结构保持一致：
    chunk_id / law_name / article_number / paragraph_number / page。
    """
    if not sources:
        return

    print(f"\n【引用法源】共 {len(sources)} 条")
    for i, source in enumerate(sources, start=1):
        law_name = source.get("law_name") or "未知法规"
        label = format_article_label(source.get("article_number"), source.get("paragraph_number"))
        print(f"  [{i}] 《{law_name}》{label}")


def main(argv: list[str] | None = None) -> int:
    """命令行入口：问答完整闭环。"""
    parser = argparse.ArgumentParser(description="法律 RAG 完整闭环：检索 + 问答 + 护栏。")
    parser.add_argument("question", help="要提问的法律问题")
    parser.add_argument("--recall-limit", type=int, default=20, help="向量召回条数")
    parser.add_argument("--keyword-recall-limit", type=int, default=20, help="关键词召回条数")
    parser.add_argument("--top-n", type=int, default=5, help="重排后进入上下文的条数")
    parser.add_argument(
        "--as-of-date",
        type=str,
        default=None,
        help="适用时间点（YYYY-MM-DD 格式），默认不限制时间",
    )
    parser.add_argument(
        "--jurisdiction",
        type=str,
        default="中国大陆",
        help="法域（默认：中国大陆）",
    )
    args = parser.parse_args(argv)

    # 初始化问答服务
    try:
        chat_service = build_default_chat_service()
    except Exception as error:
        print(f"初始化服务失败：{type(error).__name__} {error}", file=sys.stderr)
        return 2

    # 执行完整问答流程
    try:
        result = chat_service.chat(
            args.question,
            vector_recall_limit=args.recall_limit,
            keyword_recall_limit=args.keyword_recall_limit,
            rerank_top_n=args.top_n,
            as_of_date=args.as_of_date,
            jurisdiction=args.jurisdiction,
        )
    except Exception as error:
        print(f"问答失败：{type(error).__name__} {error}", file=sys.stderr)
        return 3

    # 打印检索统计
    if result.retrieval_stats:
        print_retrieval_stats(result.retrieval_stats)

    # 打印引用法源
    if not result.refused:
        print_sources(result.sources)

    # 打印最终回答
    print("\n【回答】")
    print(result.answer)

    # 打印护栏信息（调试用）
    if result.guardrail_applied:
        print(f"\n【护栏】{', '.join(result.guardrail_applied)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

