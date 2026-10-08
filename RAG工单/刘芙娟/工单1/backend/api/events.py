"""SSE 线格式编码。全仓唯一的事件序列化点。

契约见 `specs/006-medical-qa-input/contracts/sse.md` §1。每个事件三行，空行结束：

    event: <事件名>\\n
    data: <单行 JSON>\\n
    \\n
"""

import json
from typing import Any, Mapping

__all__ = ["sse_event"]


def sse_event(name: str, payload: Mapping[str, Any]) -> str:
    """把一个事件编码成 SSE 文本块。

    参数：
        name:    事件名，取值来自 `__init__.py` 的 `EV_*` 常量。
        payload: 载荷，会被序列化为**单行** JSON。

    ⚠️ `ensure_ascii=False` 不是可选优化：

    默认的 `ensure_ascii=True` 会把中文转成 `\\uXXXX`。功能上完全正确，但会让
    `curl -N` 的输出彻底不可读 —— 而"能直接肉眼观察流内容"正是本项目选 SSE
    而非 NDJSON 的理由之一（research R1）。转义后这个理由就消失了。

    ⚠️ 不传 `indent`：

    SSE 规范中 `data:` 后跟换行即终止该字段。带缩进的 JSON 会被拆成多个
    `data:` 行，接收端拼回来的语义取决于实现 —— 这是一个**双方各自看起来
    都对、却能悄悄错位**的地方。
    """

    body = json.dumps(dict(payload), ensure_ascii=False, separators=(",", ":"))
    _assert_single_line(body)
    return f"event: {name}\ndata: {body}\n\n"


def _assert_single_line(body: str) -> None:
    """断言序列化结果不含裸换行。

    诚实说明：以 `json.dumps` 的语义，换行必然被转义成 `\\n` 两个字符，
    因此本断言在当前实现下**不可达**。保留它是为了守住不变量 —— 若将来
    有人换成别的序列化器（如手写模板、`orjson` 的非标准选项），
    这条要求不会随之消失，而失效方式又恰好是"看起来正常、偶发丢事件"。
    断言比注释可靠。
    """

    if "\n" in body or "\r" in body:
        raise ValueError(
            "SSE data 段不得含裸换行：它会被拆成多个 data: 行，导致接收端解析错位"
        )
