"""查询改写 / 扩写（多查询生成）。"""
from __future__ import annotations

from app.config import settings
from app.core.registry import get_llm
from app.logging_conf import log

_REWRITE_PROMPT = (
    "请针对下面的用户问题，生成 2 个不同角度的改写或扩写查询，用于提升检索召回。\n"
    "要求：保持原意，可补充同义词或专业术语；每行一个，不要编号。\n"
    "用户问题：{query}"
)


def expand_queries(query: str, enable: bool | None = None, max_queries: int = 3) -> list[str]:
    """返回 [原问题, 改写1, 改写2...]，失败时仅返回原问题。"""
    enable = settings.enable_query_rewrite if enable is None else enable
    queries = [query]
    if not enable:
        return queries
    try:
        raw = get_llm().chat(
            [{"role": "user", "content": _REWRITE_PROMPT.format(query=query)}], temperature=0.3
        )
        for line in raw.splitlines():
            q = line.strip().lstrip("-*0123456789.、) ").strip()
            if len(q) >= 4 and q not in queries:
                queries.append(q)
    except Exception as exc:  # noqa: BLE001
        log.warning("查询改写失败: %s", exc)
    return queries[:max_queries]
