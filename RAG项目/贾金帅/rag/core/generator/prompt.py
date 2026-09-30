"""生成辅助提示词与结构化输出解析。

混合检索回答提示词位于 ``src.core.fusion.prompt``；本文件只保留图片理解、
无结果兜底以及通用 JSON/上下文工具。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.+?)\s*```", re.DOTALL)

PROMPT_VERSION = "2.0.0"
ANSWER_KEYS = ("answer", "items", "confidence", "references")


def extract_json(text: str) -> Any:
    """从纯 JSON、代码块或夹杂文字的模型输出中提取 JSON。"""
    if not text:
        return None
    cleaned = text.strip()
    match = _JSON_BLOCK_RE.search(cleaned)
    if match:
        cleaned = match.group(1).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 尝试截取最外层 JSON 对象或数组。
    for left, right in (("{", "}"), ("[", "]")):
        start = cleaned.find(left)
        end = cleaned.rfind(right)
        if start >= 0 and end > start:
            try:
                return json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError:
                continue
    return None


@dataclass(frozen=True)
class PromptTemplate:
    """system/user 二元提示词模板。"""

    template_id: str
    version: str
    system: str
    user: str

    def render(self, *args, **kwargs) -> tuple[str, str]:
        user = self.user.format(*args) if args else self.user.format(**kwargs)
        return self.system, user


NO_RESULT_PROMPT = PromptTemplate(
    template_id="no_result",
    version="3.0.0",
    system=(
        "你是自然、耐心的医疗问答助手。你现在没有足够可靠的信息回答问题，"
        "不得使用模型常识补答。像面对面交流一样，直接说明还不能确定什么，"
        "并告诉用户补充哪项具体信息最有帮助。不要提及检索、证据、资料库、"
        "知识库、数据库或系统，也不要使用客服式的‘您好’和机械编号清单。"
        "只输出 JSON 对象，字段为 answer、items、confidence、references；"
        "items 和 references 必须为空数组，confidence 必须为 0。"
    ),
    user="用户问题：{query}\n请自然地说明目前无法确定，并给出一个具体下一步。只输出 JSON。",
)


IMAGE_UNDERSTANDING_PROMPT = PromptTemplate(
    template_id="image_understanding",
    version="2.0.0",
    system=(
        "你是医疗图片信息提取器，只描述图片中清晰可见的内容，不作诊断。"
        "药品图片提取名称、规格和厂家；处方或报告提取可辨文字；"
        "症状图片只描述客观外观。无法辨认时明确说无法识别。只输出文本。"
    ),
    user="请提取图片中与医疗有关的可见信息。",
)


def estimate_tokens(text: str) -> int:
    """粗略估算中英文混合文本 token 数。"""
    if not text:
        return 0
    chinese = len(re.findall(r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]", text))
    other = len(text) - chinese
    return chinese + other // 4 + 1


def truncate_context(context: str, max_tokens: int = 2000) -> str:
    """在证据条目边界上截断上下文，极端情况下截断首条。"""
    if not context:
        return ""
    max_tokens = max(1, int(max_tokens))
    if estimate_tokens(context) <= max_tokens:
        return context

    kept: list[str] = []
    total = 0
    for line in context.splitlines():
        cost = estimate_tokens(line)
        if total + cost > max_tokens:
            break
        kept.append(line)
        total += cost
    if not kept:
        kept = [context[:max(1, max_tokens - 12)]]
    kept.append("……（证据过长已截断）")
    return "\n".join(kept)


def parse_answer_json(raw: str) -> dict[str, Any]:
    """解析统一回答 JSON；失败时以纯文本安全包装。"""
    data = extract_json(raw)
    if isinstance(data, dict) and "answer" in data:
        return _normalize_answer(data)
    return {
        "answer": (raw or "").strip(),
        "items": [],
        "confidence": 0.0,
        "references": [],
        "_fallback": True,
    }


def _normalize_answer(data: dict) -> dict[str, Any]:
    answer = data.get("answer", "")
    for _ in range(3):
        if isinstance(answer, dict):
            answer = answer.get("answer", answer)
        else:
            break

    items = data.get("items") or []
    if not isinstance(items, list):
        items = []
    normalized_items = []
    for item in items:
        if not isinstance(item, dict):
            continue
        normalized = dict(item)
        normalized.setdefault("name", str(normalized.get("value", "未知")))
        normalized_items.append(normalized)

    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    references = data.get("references") or []
    if not isinstance(references, list):
        references = []
    normalized_references: list[int] = []
    for reference in references:
        if not isinstance(reference, (int, float)):
            continue
        value = int(reference)
        if value > 0 and value not in normalized_references:
            normalized_references.append(value)

    return {
        "answer": str(answer).strip(),
        "items": normalized_items,
        "confidence": min(max(confidence, 0.0), 1.0),
        "references": normalized_references,
    }


def dumps_answer(answer: dict) -> str:
    return json.dumps(answer, ensure_ascii=False)
