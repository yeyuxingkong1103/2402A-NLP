import logging
import re

logger = logging.getLogger(__name__)


def postprocess_answer(answer: str) -> str:
    # 统一清理模型输出，避免重复空白、模型生成的代码围栏或多余来源标记影响前端展示。
    """Normalize common model artifacts while keeping the answer content intact."""
    result = answer.replace("\r\n", "\n").strip()
    result = re.sub(r"<think>.*?</think>", "", result, flags=re.IGNORECASE | re.DOTALL)
    result = re.sub(r"```(?:markdown|text)?\s*", "", result, flags=re.IGNORECASE)
    result = result.replace("```", "")
    result = re.sub(r"\n{3,}", "\n\n", result)
    result = re.sub(r"(?m)^\s*(答案|答复)\s*[:：]\s*", "", result)
    result = result.strip()
    if not result:
        result = "抱歉，我暂时没有生成有效回答，请换一种方式描述问题。"
    logger.debug("answer postprocessed: chars=%s", len(result))
    return result
