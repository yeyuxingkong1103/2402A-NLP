from __future__ import annotations

"""检索结果元数据过滤工具。"""

from typing import Any


def matches_metadata(metadata: dict[str, Any], filters: dict[str, str] | None) -> bool:
    if not filters:
        return True
    for key, expected in filters.items():
        if expected and str(metadata.get(key, "")) != expected:
            return False
    return True
