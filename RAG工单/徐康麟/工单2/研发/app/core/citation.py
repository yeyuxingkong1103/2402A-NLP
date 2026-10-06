"""引用处理：把检索片段转成带页码的可追溯引用。

工单：人工智能NLP-RAG-基于PDF文档的问答系统优化
阶段：研发 / 引用层（对应 设计/接口设计.md §2.13）

工单要求（5.6）：答案后列出引用来源——页码、原文片段摘要、chunk_id；前端可展开查看原文。
本模块同时负责**引用合法性校验**（评估要用）：引用必须带真实 ``chunk_id``，
且页码落在文档页码范围内（本语料 = 1..548）；无法回查的引用在 ``validate()`` 中列出
``invalid_pages``，绝不伪造带空 chunk_id 的引用（那样会让引用正确率统计失真）。
"""

from __future__ import annotations

import re

from app.core.logging_conf import logger, trace
from app.core.text_utils import truncate
from app.models.schemas import Answer, Citation, RetrievedChunk

CITATION_PATTERN = re.compile(r"\[页码:\s*(\d+)\]")
#: 英文引用标签（英文提问时模型输出 ``[Page: N]``）
CITATION_PATTERN_EN = re.compile(r"\[Page:\s*(\d+)\]", re.IGNORECASE)


class CitationManager:
    """引用管理器。"""

    def __init__(self, valid_pages: set[int] | None = None) -> None:
        """Args:
        valid_pages: 文档真实存在的页码集合；用于校验引用是否越界。
        """
        self.valid_pages = valid_pages or set()

    # ------------------------------------------------------------------
    def set_valid_pages(self, pages: set[int] | list[int]) -> None:
        """更新文档合法页码集合（通常是 1..page_count）。"""
        self.valid_pages = set(pages)

    def set_valid_chunk_ids(self, chunk_ids: set[str] | list[str]) -> None:
        """更新可回查的 chunk_id 集合（引用需能回查 SQLite chunks 表）。"""
        self.valid_chunk_ids = set(chunk_ids)

    # ------------------------------------------------------------------
    @trace
    def build(
        self,
        answer_text: str,
        contexts: list[RetrievedChunk],
        language: str = "zh",
        primary_chunk_id: str = "",
        limit: int = 5,
        primary: RetrievedChunk | None = None,
    ) -> list[Citation]:
        """依据答案正文中的 ``[页码: N]`` 与候选片段构造引用列表。

        优先级（越靠前越优先，同一页码只保留分数最高的一条）：

        1. 答案正文里**显式引用**的页码（``[页码: N]``）；
        2. ``primary``：真正产出该答案的证据片段（抽取式回答必传）；
        3. 其余高分片段，仅作补充，最多补到 ``limit`` 条。

        这样既保证“引用一定可追溯到真实片段”，又不会把 5 条检索结果
        全部堆到答案后面——引用正确率（评估指标之一）才站得住。
        """
        cited = self.extract_pages(answer_text)
        ordered: list[RetrievedChunk] = []
        seen: set[str] = set()

        def push(item: RetrievedChunk | None) -> None:
            if item is None or item.chunk.chunk_id in seen:
                return
            seen.add(item.chunk.chunk_id)
            ordered.append(item)

        # 1) 显式引用的页码
        if cited:
            for page in cited:
                best_for_page = max(
                    (item for item in contexts if item.chunk.page == page),
                    key=lambda item: item.score,
                    default=None,
                )
                if best_for_page is not None:
                    push(best_for_page)
                else:
                    # 模型引用了候选中不存在的页码：**丢弃该引用**并写告警日志。
                    # （不再伪造一条 chunk_id 为空的引用——那样会让引用正确率
                    #   统计失真，也无法在前端展开原文。）
                    logger.warning(
                        "app.core.citation",
                        "答案引用了候选片段中不存在的页码，已丢弃该引用（疑似幻觉）",
                        missing_page=page,
                        available_pages=sorted({item.chunk.page for item in contexts}),
                    )
        # 2) 真正产出答案的证据片段
        if primary is None and primary_chunk_id:
            primary = next((item for item in contexts if item.chunk.chunk_id == primary_chunk_id), None)
        push(primary)
        # 3) 其余高分片段补足
        for item in sorted(contexts, key=lambda candidate: candidate.score, reverse=True):
            if len(ordered) >= limit:
                break
            push(item)

        return [
            Citation(
                page=item.chunk.page,
                chunk_id=item.chunk.chunk_id,
                snippet=truncate(item.chunk.content, 220),
                section=item.chunk.section,
                score=round(item.score, 4),
            )
            for item in ordered[:limit]
        ]

    # ------------------------------------------------------------------
    @staticmethod
    def extract_pages(answer_text: str) -> list[int]:
        """从答案正文提取 ``[页码: N]`` 中的页码（去重、保持顺序）。"""
        seen: set[int] = set()
        pages: list[int] = []
        for match in CITATION_PATTERN.finditer(answer_text or ""):
            page = int(match.group(1))
            if page not in seen:
                seen.add(page)
                pages.append(page)
        return pages

    # ------------------------------------------------------------------
    def validate(self, answer: Answer) -> dict[str, object]:
        """校验答案的引用是否合法。

        “合法”指：引用带有真实 ``chunk_id``（说明它来自检索片段而非模型臆造），
        且页码落在文档页码范围内。

        Returns:
            ``{"total": n, "valid": n, "invalid_pages": [...], "has_snippet": bool,
               "valid_ratio": float}``
        """
        citations = answer.citations
        invalid: list[int] = []
        valid = 0
        with_snippet = 0
        for citation in citations:
            page_ok = (not self.valid_pages) or citation.page in self.valid_pages
            has_source = bool(citation.chunk_id)
            if has_source and self.valid_chunk_ids:
                has_source = citation.chunk_id in self.valid_chunk_ids
            if page_ok and has_source:
                valid += 1
            else:
                invalid.append(citation.page)
            if citation.snippet:
                with_snippet += 1
        total = len(citations)
        result = {
            "total": total,
            "valid": valid,
            "invalid_pages": invalid,
            "has_snippet": with_snippet > 0,
            "valid_ratio": round(valid / total, 4) if total else 0.0,
            "is_unknown": answer.is_unknown,
        }
        if invalid:
            logger.warning(
                "app.core.citation",
                "存在无效引用（页码越界或缺少来源片段）",
                invalid_pages=invalid,
                document_pages=len(self.valid_pages),
            )
        return result

    # ------------------------------------------------------------------
    @staticmethod
    def inline_labels(citations: list[Citation]) -> str:
        """把引用渲染成 ``[页码: 12] [页码: 129]`` 形式，供纯文本报告使用。"""
        return " ".join(citation.label() for citation in citations)

    def render_markdown(self, answer: Answer) -> str:
        """把答案渲染成带引用清单的 Markdown（供评估报告/导出使用）。"""
        lines = [answer.answer, ""]
        if answer.citations:
            lines.append("**引用来源**")
            for citation in answer.citations:
                label = citation.label()
                snippet = citation.snippet or "（正文中未找到对应片段）"
                lines.append(f"- {label} `{citation.chunk_id}` {snippet}")
        else:
            lines.append("_（无引用：该回答被判定为无法从文档中获得答案）_")
        return "\n".join(lines)


def get_citation_manager(valid_pages: set[int] | list[int] | None = None) -> CitationManager:
    """工厂函数。"""
    return CitationManager(valid_pages=set(valid_pages) if valid_pages else None)
