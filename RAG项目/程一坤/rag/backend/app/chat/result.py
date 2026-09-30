"""问答结果与上下文摘录（结果组装职责，自 service.py 拆出）。

ChatResult 是问答链路的统一产出物：同步 chat() 与流式 chat_stream() 都以它
收口，API 层 / CLI / 评测脚本都消费这个类型。放在独立文件是为了让
service.py 只保留"主流程编排"，结果的形状定义与组装工具单列。
"""
from dataclasses import dataclass
from typing import Any


@dataclass
class ChatResult:
    """问答结果。

    Attributes:
        answer: 最终回答文本
        sources: 引用的法源列表（简化信息）
        refused: 是否拒答
        guardrail_applied: 应用的护栏列表
        retrieval_stats: 检索统计信息（可选）
    """
    answer: str
    sources: list[dict[str, Any]]
    refused: bool
    guardrail_applied: list[str]
    retrieval_stats: dict[str, int] | None = None
    # 注入提示词的法条原文（编号与回答里的 [n] 对应）。仅供评测（Faithfulness
    # 打分）与内部审计使用；API 层不序列化该字段，前端响应不受影响。
    context_excerpts: list[dict[str, Any]] | None = None


def _build_context_excerpts(articles: list[Any]) -> list[dict[str, Any]]:
    """把注入上下文的法条原文整理成带编号的列表（[n] 与提示词一致）。"""
    return [
        {
            "index": position,
            "law_name": art.document_title,
            "article_number": art.article_number,
            "content": art.content or "",
        }
        for position, art in enumerate(articles, start=1)
    ]
