from __future__ import annotations

from typing import Any

VAGUE_REFERENCES = (
    "他", "她", "它", "这个", "那个", "上述", "前面", "刚才",
    "该合同", "该行为", "该条款", "该证据", "该材料", "该问题", "该情况",
    "其责任", "其行为", "其权利", "其义务", "其主张", "其证据",
)


class CoreferenceResolver:
    def resolve(self, query: str, short_memory: dict[str, Any], long_memories: list[dict], history_summary: str = "") -> dict:
        original = str(query or "").strip()
        reference_text = original.replace("其他", "").replace("其次", "").replace("其中", "")
        if not original or not any(mark in reference_text for mark in VAGUE_REFERENCES):
            return {"query": original, "resolved_query": original, "resolved_references": {}, "used_memory_ids": []}
        target = str(short_memory.get("current_topic") or short_memory.get("current_goal") or "").strip()
        if not target and long_memories:
            target = str(long_memories[0].get("content") or "").strip()
        if not target:
            target = str(history_summary or "").strip()[-180:]
        if not target:
            return {"query": original, "resolved_query": original, "resolved_references": {}, "used_memory_ids": []}
        resolved = f"关于{target}，{original}"
        return {"query": original, "resolved_query": resolved, "resolved_references": {"target": target}, "used_memory_ids": [int(item["id"]) for item in long_memories if str(item.get("id") or "").isdigit() and str(item.get("content") or "") == target]}


__all__ = ["CoreferenceResolver", "VAGUE_REFERENCES"]
