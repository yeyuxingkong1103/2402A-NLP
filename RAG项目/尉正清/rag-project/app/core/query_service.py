# app/core/query_service.py
"""查询理解：追问改写 + 多查询扩写。

两者服务于同一个目标——让检索命中该命中的内容，但解决的是不同问题：

    改写  多轮对话里「那他呢」这类省略句，脱离上下文无法检索
    扩写  单一表述只能覆盖知识库中的一种说法，换个问法就检索不到

扩写出来的多个查询**分别检索**，结果再做融合，相当于用不同角度
各撒一次网，能显著提升召回率。
"""
import re
from typing import List, Optional

from app.core.llm_service import get_llm
import logging

logger = logging.getLogger(__name__)

# 一次扩写出多少个查询（含原查询，实际新增 n-1 个）
DEFAULT_VARIANTS = 3
MAX_QUERY_CHARS = 120

EXPAND_PROMPT = """请针对下面的问题，生成 {n} 个不同角度的检索查询，用于在知识库中查找相关资料。

要求：
1. 每个查询从不同角度切入：同义改写、上位概念、关键要素拆解、专业术语表述；
2. 每行一个查询，不要编号、不要引号、不要任何解释；
3. 保留原问题的核心意图，不要引入原问题没有的新条件；
4. 如果原问题包含法律条文号或专业术语，务必在其中一条查询里原样保留。

问题：{question}"""


class QueryService:
    """查询改写与扩写。两者都是尽力而为，失败时回退到原问题。"""

    def rewrite(self, question: str, history_lines: Optional[List[str]] = None) -> str:
        """多轮追问改写：把「那它呢」补全成可独立检索的问题。"""
        if not history_lines:
            return question
        try:
            out = get_llm().rewrite_query(question, history_lines)
            return (out or question).strip()
        except Exception as e:                              # pragma: no cover
            logger.warning("追问改写失败，使用原问题: %s", e)
            return question

    def expand(self, question: str, n: int = DEFAULT_VARIANTS) -> List[str]:
        """围绕一个问题生成多个检索查询（含原问题本身）。"""
        question = (question or "").strip()
        if not question or n <= 1:
            return [question]

        prompt = EXPAND_PROMPT.format(n=n - 1, question=question)
        try:
            raw = get_llm().chat([{"role": "user", "content": prompt}],
                                 temperature=0.3, max_tokens=1024)
        except Exception as e:                              # pragma: no cover
            logger.warning("查询扩写失败，只用原查询: %s", e)
            return [question]

        variants = []
        for line in (raw or "").split("\n"):
            line = line.strip()
            # 去掉可能残留的编号、项目符号、引号
            line = re.sub(r"^[\-\*\d\.、\)）]+\s*", "", line).strip().strip('"“”\'')
            if not line or len(line) < 3 or len(line) > MAX_QUERY_CHARS:
                continue
            if line == question or line in variants:
                continue
            variants.append(line)

        queries = [question] + variants[:n - 1]
        if len(queries) > 1:
            logger.debug("查询扩写 %s -> %s 个", question[:24], len(queries))
        return queries

    def prepare(self, question: str, history_lines: Optional[List[str]] = None,
                expand_n: int = DEFAULT_VARIANTS) -> List[str]:
        """完整流程：先按上下文改写，再扩写出多个检索查询。

        返回的列表首项始终是「改写后的问题」，便于日志与调试对齐。
        """
        rewritten = self.rewrite(question, history_lines)
        return self.expand(rewritten, n=expand_n)

_service: Optional[QueryService] = None


def get_query_service() -> QueryService:
    global _service
    if _service is None:
        _service = QueryService()
    return _service
