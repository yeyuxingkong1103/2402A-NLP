"""在模型上下文长度内选择记忆、历史和证据，构造最终回答上下文。

检索到的内容并不是越多越好。本文件会优先保留精确法条、用户证据和高分资料，
再按 token 预算裁剪，防止历史或某一条超长材料挤掉当前问题和回答空间。
"""

# json 用于估算字典/列表的文本长度，并把结构化记忆放进提示词。
import json
# Any 表示 token 估算器可接收字符串、字典、列表等。
from typing import Any


# 法条和司法解释必须保留完整条件、例外，不能截掉后半段。
FULL_TEXT_COLLECTIONS = {"civil_code_articles", "civil_interpretations"}
# 其他资料被裁剪时追加明确提示，防止模型把节选误当完整原文。
EXCERPT_NOTICE = "【节选，非完整原文】"


def evidence_excerpt(row: dict, max_chars: int) -> dict:
    """条文保留完整内容；其他资料裁剪时明确标注节选，供各阶段共用。"""

    # 复制记录，避免裁剪检索器原始结果。
    item = dict(row)
    # 统一取正文并清理首尾空白。
    text = str(item.get("content") or "").strip()
    # 法条/司法解释不截断；其他集合超限时才裁剪。
    if item.get("collection") not in FULL_TEXT_COLLECTIONS and len(text) > max_chars:
        # 为节选提示预留字符，确保最终不超过 max_chars。
        text = text[:max(0, max_chars - len(EXCERPT_NOTICE))] + EXCERPT_NOTICE
        # 显式标记，后续展示或生成器可以识别这不是全文。
        item["content_truncated"] = True
    # 把保留后的正文写入副本。
    item["content"] = text
    return item


# 给不同用途分配默认 token 桶，避免任一类内容独占整个模型窗口。
DEFAULT_TOKEN_BUDGET = {
    # 系统指令预算。
    "system": 2000,
    # 用户稳定画像/偏好预算。
    "user_profile": 1000,
    # 当前案件结构化记忆预算。
    "case_memory": 2000,
    # 历史摘要预算。
    "summary": 2000,
    # 最近原始消息预算。
    "recent_messages": 4000,
    # 本轮问题预算。
    "current_question": 1000,
    # 检索证据是最大内容桶。
    "evidence": 12000,
    # 为大模型回答预留输出空间。
    "answer_reserve": 6000,
    # 为安全说明和估算误差预留空间。
    "safety": 2000,
}


def estimate_tokens(value: Any) -> int:
    """用字符数做轻量 token 估算，供本地预算裁剪使用。"""

    # 常见空值不占上下文。
    if value in (None, "", [], {}):
        return 0
    # 结构化值先转成不转义中文的 JSON 字符串。
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False)
    # 中文 token 通常比“4 字符一个 token”更密集，这里用约 2 字符估算得更保守。
    return max(1, len(value) // 2)


class ContextBuilder:
    """把用户画像、长期记忆、短期历史和检索证据组装成最终回答上下文。"""

    def __init__(self, max_tokens: int = 32000, budget: dict | None = None, evidence_max_chars: int = 1200, evidence_limit: int = 8):
        # 模型整段上下文最大 token 数。
        self.max_tokens = max_tokens
        # 用户配置覆盖同名默认桶，未配置项仍保留默认值。
        self.budget = {**DEFAULT_TOKEN_BUDGET, **(budget or {})}
        # 单条非规范证据最少允许 200 字，避免配置成无意义极小值。
        self.evidence_max_chars = max(200, int(evidence_max_chars or 1200))
        # 至少保留一条证据名额。
        self.evidence_limit = max(1, int(evidence_limit or 8))

    def allocate_budget(self) -> dict:
        """当配置预算超过模型窗口时，按比例压缩每个上下文分桶。"""

        # 复制配置，避免缩放时修改实例原预算。
        planned = dict(self.budget)
        # 计算所有桶总和。
        planned_total = sum(planned.values())
        # 未超过模型窗口时无需调整。
        if planned_total <= self.max_tokens:
            return planned
        # 超限时计算统一缩放比例。
        scale = self.max_tokens / planned_total
        # 每个桶至少保留 64 token，避免被压成完全不可用。
        return {key: max(64, int(value * scale)) for key, value in planned.items()}

    def select_recent_messages(self, messages: list[dict], token_budget: int) -> list[dict]:
        """从最近消息向前选取，第一条就超预算时直接跳过，防止单条历史无界进入 prompt。"""

        # selected 暂时以“新到旧”顺序收集。
        selected: list[dict] = []
        # used 累计已使用 token。
        used = 0
        # reversed 从最新消息开始，优先保留离当前问题最近的上下文。
        for message in reversed(messages or []):
            cost = estimate_tokens(message)
            # 加入当前消息会超预算时停止或跳过。
            if used + cost > token_budget:
                # 已经选到至少一条后停止，保持连续的最近对话。
                if selected:
                    break
                # 最新一条自身过长时跳过，继续寻找更短的上一条。
                continue
            selected.append(message)
            used += cost
        # 再反转回正常时间顺序交给大模型。
        return list(reversed(selected))

    @staticmethod
    def trim_text_to_budget(text: str, token_budget: int) -> str:
        """把普通文本截到估算 token 预算内。"""

        # 无预算时返回空文本。
        if token_budget <= 0:
            return ""
        value = str(text or "").strip()
        # 未超预算保持原文。
        if estimate_tokens(value) <= token_budget:
            return value
        # estimate_tokens 约按 2 字符计算，因此保留 token_budget * 2 字符。
        return value[: max(1, token_budget * 2)].rstrip()

    @staticmethod
    def evidence_score(row: dict) -> float:
        """按重排分、召回分、RRF 分的优先级取得证据分数。"""

        value = row.get("rerank_score", row.get("score", row.get("rrf_score", 0)))
        try:
            return float(value or 0)
        except (TypeError, ValueError):
            # 异常分数按 0 处理，不能让排序报错。
            return 0.0

    @staticmethod
    def evidence_priority(row: dict) -> int:
        """给不同证据来源分配业务优先级，数字越大越先进入上下文。"""

        # 精确法条号或主键命中最高。
        if row.get("retrieval_channel") == "exact" or row.get("retrieval_reason") == "article_exact":
            return 5
        # 用户亲自上传的证据比一般公共候选更贴近具体案情。
        if row.get("source_type") in {"private", "user_upload"}:
            return 4
        # 关键词精确命中排在普通向量召回之前。
        if row.get("retrieval_channel") == "keyword":
            return 3
        # 路由器标记的重点集合继续获得加权。
        if row.get("priority_collection"):
            return 2
        # 普通公共资料是基础优先级。
        if row.get("source_type") == "public":
            return 1
        # 未知来源最低。
        return 0

    def select_evidence(self, evidence: list[dict], token_budget: int) -> list[dict]:
        """优先保留强匹配和私有材料；首条证据过大时裁剪而不是全部丢弃。"""

        # selected 保存最终进入模型上下文的证据。
        selected: list[dict] = []
        # used 累计证据标题和正文的估算 token。
        used = 0
        # 先按业务优先级，再按重排/召回分数降序排列。
        ranked = sorted(
            evidence or [],
            key=lambda row: (self.evidence_priority(row), self.evidence_score(row)),
            reverse=True,
        )
        # 按排序逐条尝试加入。
        for row in ranked:
            # 达到配置证据条数上限后停止。
            if len(selected) >= self.evidence_limit:
                break
            # 单条成本由正文和标题共同组成。
            cost = estimate_tokens(row.get("content", "")) + estimate_tokens(row.get("title", ""))
            # 加入后会超过证据预算时采用不同策略。
            if used + cost > token_budget:
                # 法条和司法解释不能截掉末尾条件或例外，因此整条跳过。
                if row.get("collection") in FULL_TEXT_COLLECTIONS:
                    continue
                # 已经有证据时停止，避免后面的低优先级内容替代当前连续选择。
                if selected:
                    break
                # 第一条非规范证据过长时，尝试在预算内保留节选。
                item = dict(row)
                title_tokens = estimate_tokens(item.get("title", ""))
                # 剩余 token 按每 token 约 2 字符换算成正文字符上限。
                available = max(0, token_budget - title_tokens) * 2
                # 连“节选”提示都放不下时跳过。
                if available <= len(EXCERPT_NOTICE):
                    continue
                item = evidence_excerpt(item, available)
                # 裁剪后没有正文则仍不能使用。
                if not item["content"]:
                    continue
                selected.append(item)
                # 第一条已经用完可用预算，结束选择。
                break
            # 未超预算时保留完整记录并累计成本。
            selected.append(row)
            used += cost
        return selected

    @staticmethod
    def compress_summary(summary: dict | str, max_chars: int = 2000) -> str:
        """从摘要对象或字符串中取最近 max_chars 个字符。"""

        # 记忆服务常把摘要放在 {summary: ...} 中。
        if isinstance(summary, dict):
            text = str(summary.get("summary") or "")
        else:
            text = str(summary or "")
        # 保留尾部，因为摘要尾部通常包含更新的案件进展。
        return text[-max_chars:]

    @staticmethod
    def compress_evidence(evidence: list[dict], max_chars: int = 1200) -> list[dict]:
        """对每条非规范证据统一做展示长度裁剪。"""

        return [evidence_excerpt(row, max_chars) for row in evidence]

    @staticmethod
    def compress_original_hits(original_hits: list[dict], max_chars: int = 800, limit: int = 6) -> list[dict]:
        """压缩从历史中回查到的原始消息，最多保留指定条数。"""

        # compressed 保存复制后的记录。
        compressed = []
        # 只处理前 limit 条命中。
        for row in original_hits[:limit]:
            item = dict(row)
            # 历史回查是辅助信息，每条最多保留 max_chars 字。
            item["content"] = str(item.get("content") or "")[:max_chars]
            compressed.append(item)
        return compressed

    @staticmethod
    def format_original_hits(original_hits: list[dict]) -> str:
        """把原始历史命中格式化成编号文本，便于放入系统上下文。"""

        # lines 保存最终的“序号.[角色] 内容”。
        lines = []
        for index, row in enumerate(original_hits, start=1):
            # 缺失角色时使用“历史消息”。
            role = str(row.get("role") or "历史消息")
            content = str(row.get("content") or "").strip()
            # 空内容跳过。
            if not content:
                continue
            lines.append(f"{index}. [{role}] {content}")
        return "\n".join(lines)

    @staticmethod
    def _section(title: str, value: Any) -> str:
        """把一类记忆包装成带中文标题的提示词章节。"""

        # 空值不生成章节。
        if not value:
            return ""
        # 字符串直接使用；字典/列表格式化为易读 JSON。
        if isinstance(value, str):
            body = value
        else:
            body = json.dumps(value, ensure_ascii=False, indent=2)
        return f"{title}：\n{body}"

    def build_context(self, question: str, memory_context=None, evidence: list[dict] | None = None, user_profile: dict | None = None) -> dict:
        """生成 AnswerGenerator 使用的统一上下文对象，并返回可观测的裁剪统计。"""

        # 先取得适配当前模型窗口的各类预算。
        budget = self.allocate_budget()
        # 显式传入画像优先，否则使用记忆上下文中的画像。
        profile = user_profile or getattr(memory_context, "user_profile", {}) or {}
        # 兼容新版 legacy_case_memory 和旧版 case_memory 字段。
        legacy_case_memory = getattr(memory_context, "legacy_case_memory", getattr(memory_context, "case_memory", {})) or {}
        # 长期记忆可能包括稳定偏好、技能或跨会话项目。
        long_memory = getattr(memory_context, "long_memory", []) or []
        # 用户关闭长期记忆时，即使查到了也不能加入提示词。
        if memory_context is not None and not getattr(memory_context, "enable_long_memory", True):
            long_memory = []
        # 压缩历史回查、摘要和近期原始消息。
        original_hits = self.compress_original_hits(getattr(memory_context, "original_hits", []) or [])
        summary = getattr(memory_context, "summary", {}) or {}
        recent = getattr(memory_context, "short_term", []) or []
        # 分别在近期历史和证据预算内选择内容。
        selected_recent = self.select_recent_messages(recent, budget["recent_messages"])
        selected_evidence = self.select_evidence(evidence or [], budget["evidence"])
        # 证据入选后，再限制单条非规范材料的展示字符数。
        selected_evidence = self.compress_evidence(selected_evidence, self.evidence_max_chars)

        # 最多取八条长期记忆，并标注记忆类型。
        memory_lines = [
            f"- [{item.get('memory_type', 'fact')}] {item.get('content', '')}"
            for item in long_memory[:8]
            if item.get("content")
        ]
        # 将长期记忆逐行连接。
        long_memory_text = "\n".join(memory_lines)
        # 明确约束：稳定个人信息不能被模型当成当前案件事实或证据。
        if long_memory_text:
            long_memory_text = (
                "以下只代表用户跨对话稳定信息、偏好、技能或长期项目；不得把它当作当前案件事实、上传材料内容或本次咨询证据。\n"
                f"{long_memory_text}"
            )
        # 五类记忆分别形成章节，空章节稍后过滤。
        sections = [
            self._section("用户画像", profile),
            self._section("长期记忆", long_memory_text),
            self._section("当前对话案件记忆", legacy_case_memory),
            self._section("当前对话历史摘要", self.compress_summary(summary)),
            self._section("当前对话原始内容回查", self.format_original_hits(original_hits)),
        ]
        # 只连接有内容的章节。
        memory_prompt = "\n\n".join(section for section in sections if section)
        # history 最终按 Chat Messages 格式传给 AnswerGenerator。
        history = []
        # 有记忆内容时先按画像+案件记忆+摘要的总预算裁剪。
        if memory_prompt:
            memory_prompt = self.trim_text_to_budget(memory_prompt, budget["user_profile"] + budget["case_memory"] + budget["summary"])
            if memory_prompt:
                # 记忆作为 system 消息，和用户/助手历史区分。
                history.append({"role": "system", "content": memory_prompt})
        # 把选中的最近原始消息接在记忆系统消息之后。
        history.extend(selected_recent)
        # 估算真正将进入提示词的当前问题、历史和证据总量。
        estimated = estimate_tokens(question) + estimate_tokens(history) + estimate_tokens(selected_evidence)
        # 返回生成器所需内容，以及便于观察裁剪行为的统计。
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
