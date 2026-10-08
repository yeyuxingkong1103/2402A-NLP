"""问题校验。

**边界值的判定只此一处。** 前端也有一份等价实现（`frontend/js/app.js`），
那不是重复而是纵深：前端那份是为了"不发无谓的请求"，服务端这份才是边界。
两者 MUST 保持一致，验收脚本会同时打这两条路径。

这里的判定与 `docs/05` §3.1.2 一致：问题去除首尾空白后长度 MUST 在 1–200。
"""

from . import (
    MSG_INVALID_QUESTION,
    MSG_TOO_LONG,
    QUESTION_MAX_LEN,
    QUESTION_MIN_LEN,
)
from .errors import InvalidQuestionError

__all__ = ["validate_question"]


def validate_question(raw: str) -> str:
    """校验问题并返回**规范化后**的文本。

    规范化 = 去除首尾空白。返回它而不是原串，有两个原因：

    1. 日志与后续检索用的必须是同一串文本。若一边存原串、一边用去空白串，
       同一个问题会因为首尾多打了一个空格而在日志里对不上。
    2. 长度判定与留存内容必须基于同一个基准，否则"通过校验的字符串"
       与"被校验的字符串"可能不是同一个东西。

    为什么用 `str.strip()` 而不是只去 ASCII 空格：用户粘贴时带进的换行、
    制表符、以及**全角空格 U+3000**（中文输入法下极常见）都该算作空白。
    Python 的 `str.strip()` 按 `str.isspace()` 判定，这三类都覆盖。

    为什么长度按**字符**而非字节：200 个中文字是 200 字符 / 600 字节。
    按字节判会把 67 个汉字就拒掉，与 `docs/05` §3.1.2 的 200 字符上限
    完全不是一回事。Python 的 `len()` 对 `str` 即字符数，正合要求。

    抛 `InvalidQuestionError`（而非返回 None/布尔），使"不合法"必须被处理 ——
    调用方无法静默忽略它继续往下走。
    """

    question = raw.strip()

    if len(question) < QUESTION_MIN_LEN:
        raise InvalidQuestionError(MSG_INVALID_QUESTION)

    if len(question) > QUESTION_MAX_LEN:
        raise InvalidQuestionError(MSG_TOO_LONG)

    return question
