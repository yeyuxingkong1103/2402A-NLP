"""测试间共用的小工具与假对象（fixture 在 conftest.py，别混在一起）。

这里只放**被两个以上测试文件用到的纯函数/假对象**。只有一个文件用得到的东西
留在原文件里——过早抽公共反而让读测试的人多跳一层。

两类东西不要混：假对象（RecordingLLM）替掉**外部依赖**，让用例离线也能断言；
纯函数（parse_sse / think_block）是为了让断言写得更短、更贴协议本身。

断言「管线/模型到底收到了什么」时，一律看入参、不去问模型它看到了什么——模型会顺着
问题编出并不存在的槽位。RecordingLLM 就是为此存在：它只记账、不生成内容。
"""
from __future__ import annotations

import json

# 形如 <think> 的裸标签在部分编辑器/工具链里会被当成 HTML 标签吞掉，写成字面量会导致
# 测试内容静默变形（本仓库踩过）。所以统一在这里用拼接生成，各测试文件别再各写一份。
T_OPEN = "<" + "think" + ">"
T_CLOSE = "<" + "/think" + ">"


def think_block(body: str) -> str:
    """包一个 think 块。"""
    return T_OPEN + body + T_CLOSE


def parse_sse(body: str) -> dict[str, list]:
    """把 SSE 响应体解析成 {事件名: [payload, ...]}。

    dict 保序（Python 3.7+），所以 list(events) 就是事件到达顺序，
    需要断言「第一个是 sources、最后一个是 done」时直接用它。
    """
    events: dict[str, list] = {}
    name = None
    for line in body.splitlines():
        if line.startswith("event: "):
            name = line[len("event: ") :]
        elif line.startswith("data: ") and name:
            events.setdefault(name, []).append(json.loads(line[len("data: ") :]))
    return events


class RecordingLLM:
    """把每次收到的 messages 记下来，用于断言「管线到底送了什么给模型」。"""

    def __init__(self, reply: str = "回答内容" * 50):
        self.reply = reply
        self.calls: list[list[dict]] = []

    def chat(self, messages) -> str:
        self.calls.append(messages)
        return self.reply

    def stream(self, messages):
        self.calls.append(messages)
        yield self.reply
