import json
from typing import Any


DEFAULT_TOKEN_BUDGET = {
    # RAG 上下文按用途分桶，避免历史、记忆或证据中的任一类内容挤占整段 prompt。
    "system": 2000,
    "user_profile": 1000,
    "case_memory": 2000,
    "summary": 2000,
    "recent_messages": 4000,
    "current_question": 1000,
    "evidence": 12000,
    "answer_reserve": 6000,
    "safety": 2000,
}


def estimate_tokens(value: Any) -> int:
    """用字符数做轻量 token 估算，供本地预算裁剪使用。"""
    if value in (None, "", [], {}):
        return 0
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False)
    return max(1, len(value) // 2)


class ContextBuilder:
    """把用户画像、长期记忆、短期历史和检索证据组装成最终回答上下文。"""

    def __init__(self, max_tokens: int = 32000, budget: dict | None = None, evidence_max_chars: int = 1200, evidence_limit: int = 8):
        self.max_tokens = max_tokens
        self.budget = {**DEFAULT_TOKEN_BUDGET, **(budget or {})}
        self.evidence_max_chars = max(200, int(evidence_max_chars or 1200))
        self.evidence_limit = max(1, int(evidence_limit or 8))

    def allocate_budget(self) -> dict:
        """当配置预算超过模型窗口时，按比例压缩每个上下文分桶。"""
        planned = dict(self.budget)
        planned_total = sum(planned.values())
        if planned_total <= self.max_tokens:
            return planned
        scale = self.max_tokens / planned_total
        return {key: max(64, int(value * scale)) for key, value in planned.items()}

    def select_recent_messages(self, messages: list[dict], token_budget: int) -> list[dict]:
        """从最近消息向前选取，第一条就超预算时直接跳过，防止单条历史无界进入 prompt。"""
        selected: list[dict] = []
        used = 0
        for message in reversed(messages or []):
            cost = estimate_tokens(message)
            if used + cost > token_budget:
                if selected:
                    break
                continue
            selected.append(message)
            used += cost
        return list(reversed(selected))

    @staticmethod
    def trim_text_to_budget(text: str, token_budget: int) -> str:
        if token_budget <= 0:
            return ""
        value = str(text or "").strip()
        if estimate_tokens(value) <= token_budget:
            return value
        return value[: max(1, token_budget * 2)].rstrip()

    @staticmethod
    def evidence_score(row: dict) -> float:
        value = row.get("rerank_score", row.get("score", row.get("rrf_score", 0)))
        try:
            return float(value or 0)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def evidence_priority(row: dict) -> int:
        if row.get("retrieval_channel") == "exact" or row.get("retrieval_reason") == "article_exact":
            return 5
        if row.get("source_type") in {"private", "user_upload"}:
            return 4
        if row.get("retrieval_channel") == "keyword":
            return 3
        if row.get("priority_collection"):
            return 2
        if row.get("source_type") == "public":
            return 1
        return 0

    def select_evidence(self, evidence: list[dict], token_budget: int) -> list[dict]:
        """优先保留强匹配和私有材料；首条证据过大时裁剪而不是全部丢弃。"""
        selected: list[dict] = []
        used = 0
        ranked = sorted(
            evidence or [],
            key=lambda row: (self.evidence_priority(row), self.evidence_score(row)),
            reverse=True,
        )
        for row in ranked:
            if len(selected) >= self.evidence_limit:
                break
            cost = estimate_tokens(row.get("content", "")) + estimate_tokens(row.get("title", ""))
            if used + cost > token_budget:
                if selected:
                    break
                item = dict(row)
                title_tokens = estimate_tokens(item.get("title", ""))
                item["content"] = self.trim_text_to_budget(str(item.get("content") or ""), max(0, token_budget - title_tokens))
                if not item["content"]:
                    continue
                selected.append(item)
                break
            selected.append(row)
            used += cost
        return selected

    @staticmethod
    def compress_summary(summary: dict | str, max_chars: int = 2000) -> str:
        if isinstance(summary, dict):
            text = str(summary.get("summary") or "")
        else:
            text = str(summary or "")
        return text[-max_chars:]

    @staticmethod
    def compress_evidence(evidence: list[dict], max_chars: int = 1200) -> list[dict]:
        compressed = []
        for row in evidence:
            item = dict(row)
            item["content"] = str(item.get("content") or "")[:max_chars]
            compressed.append(item)
        return compressed

    @staticmethod
    def compress_original_hits(original_hits: list[dict], max_chars: int = 800, limit: int = 6) -> list[dict]:
        compressed = []
        for row in original_hits[:limit]:
            item = dict(row)
            item["content"] = str(item.get("content") or "")[:max_chars]
            compressed.append(item)
        return compressed

    @staticmethod
    def format_original_hits(original_hits: list[dict]) -> str:
        lines = []
        for index, row in enumerate(original_hits, start=1):
            role = str(row.get("role") or "历史消息")
            content = str(row.get("content") or "").strip()
            if not content:
                continue
            lines.append(f"{index}. [{role}] {content}")
        return "\n".join(lines)

    @staticmethod
    def _section(title: str, value: Any) -> str:
        if not value:
            return ""
        if isinstance(value, str):
            body = value
        else:
            body = json.dumps(value, ensure_ascii=False, indent=2)
        return f"{title}：\n{body}"

    def build_context(self, question: str, memory_context=None, evidence: list[dict] | None = None, user_profile: dict | None = None) -> dict:
        """生成 AnswerGenerator 使用的统一上下文对象，并返回可观测的裁剪统计。"""
        budget = self.allocate_budget()
        profile = user_profile or getattr(memory_context, "user_profile", {}) or {}
        legacy_case_memory = getattr(memory_context, "legacy_case_memory", getattr(memory_context, "case_memory", {})) or {}
        long_memory = getattr(memory_context, "long_memory", []) or []
        if memory_context is not None and not getattr(memory_context, "enable_long_memory", True):
            long_memory = []
        original_hits = self.compress_original_hits(getattr(memory_context, "original_hits", []) or [])
        summary = getattr(memory_context, "summary", {}) or {}
        recent = getattr(memory_context, "short_term", []) or []
        selected_recent = self.select_recent_messages(recent, budget["recent_messages"])
        selected_evidence = self.select_evidence(evidence or [], budget["evidence"])
        selected_evidence = self.compress_evidence(selected_evidence, self.evidence_max_chars)

        memory_lines = [
            f"- [{item.get('memory_type', 'fact')}] {item.get('content', '')}"
            for item in long_memory[:8]
            if item.get("content")
        ]
        long_memory_text = "\n".join(memory_lines)
        if long_memory_text:
            long_memory_text = (
                "以下只代表用户跨对话稳定信息、偏好、技能或长期项目；不得把它当作当前案件事实、上传材料内容或本次咨询证据。\n"
                f"{long_memory_text}"
            )
        sections = [
            self._section("用户画像", profile),
            self._section("长期记忆", long_memory_text),
            self._section("当前对话案件记忆", legacy_case_memory),
            self._section("当前对话历史摘要", self.compress_summary(summary)),
            self._section("当前对话原始内容回查", self.format_original_hits(original_hits)),
        ]
        memory_prompt = "\n\n".join(section for section in sections if section)
        history = []
        if memory_prompt:
            memory_prompt = self.trim_text_to_budget(memory_prompt, budget["user_profile"] + budget["case_memory"] + budget["summary"])
            if memory_prompt:
                history.append({"role": "system", "content": memory_prompt})
        history.extend(selected_recent)
        estimated = estimate_tokens(question) + estimate_tokens(history) + estimate_tokens(selected_evidence)
        return {
            "question": question,
            "history": history,
            "evidence": selected_evidence,
            "meta": {
                "estimated_tokens": min(estimated, self.max_tokens),
                "budget": budget,
                "recent_message_count": len(selected_recent),
                "original_hit_count": len(original_hits),
                "evidence_count": len(selected_evidence),
                "dropped_evidence_count": max(0, len(evidence or []) - len(selected_evidence)),
            },
        }
