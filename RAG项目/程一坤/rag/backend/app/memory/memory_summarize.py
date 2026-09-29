"""长期记忆的摘要生成（写入前的"事实提炼"，自 long_term.py 拆出）。

把一次问答压成一句可跨会话复用的用户事实（LLM 一次调用）。
独立成文件的动机：它依赖 LLM 客户端、属于"写入侧"的前置步骤，
与 Milvus 存储（long_term.py）和 MySQL 开关（memory_settings.py）无耦合，
单独放置便于单独测试与复用（chat 主流程、未来导入流程都可调用）。
"""
from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def summarize_memory_fact(llm_client: Any, question: str, answer: str) -> dict[str, Any] | None:
    """把一次问答压成一句可跨会话复用的用户事实。

    用现有 LLM 客户端做一次调用，返回 {"summary": str, "importance": float}；
    任何失败（LLM 异常、JSON 解析失败、字段缺失）都返回 None——
    写入侧跳过本条，绝不阻塞回答、绝不编造记忆。
    """
    system_prompt = (
        "你从一次法律问答中提炼用户长期事实。只输出一个 JSON 对象，不要输出任何其它文字：\n"
        '{"summary": "<一句话，第三人称，写用户的情况/偏好/背景，可跨会话复用；'
        "与用户个人情况无关的通用法律知识不要写>\", "
        '"importance": <0到1的小数，核心个人情况0.8~1.0，一般偏好0.4~0.7，闲聊0~0.2>}\n'
        "示例：{\"summary\": \"用户是 A 公司的程序员，入职两年未签书面合同\", "
        "\"importance\": 0.9}"
    )
    user_prompt = f"# 用户问题\n{question[:1000]}\n\n# 助手回答（节选）\n{answer[:1500]}"
    try:
        raw = llm_client.chat(system_prompt, user_prompt)
        # 容错：模型可能带 ```json 围栏
        text = raw.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
        payload = json.loads(text.strip())
        summary = str(payload["summary"]).strip()
        importance = float(payload["importance"])
        if not summary:
            return None
        return {"summary": summary, "importance": min(1.0, max(0.0, importance))}
    except Exception as error:  # noqa: BLE001
        logger.warning("长期记忆摘要失败，跳过写入：%s", type(error).__name__)
        return None
