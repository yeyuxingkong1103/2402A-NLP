"""思维链清洗的端到端验证：真实推理文本 + 内联 think 的假 LLM，跑完整链路。

## 为什么这里必须用假 LLM（实测结论，别再重新推导一遍）

本项目 provider 是 ollama，实测三条路径**都把推理放在独立字段**：

| 路径 | 推理所在字段 |
| --- | --- |
| 原生 `/api/chat` | `message.thinking` |
| `/v1/chat/completions` 非流式 | `message.reasoning` |
| `/v1/chat/completions` 流式（网页走这条） | `delta.reasoning` |

而 `app/core/llm.py` 只取 `delta.content` / `message.content`，所以推理内容
**根本进不到 `postprocess`** —— `postprocess` 里的 `_THINK_BLOCK` 在 ollama 下不可达。

它只在「把推理内联进 content」的服务商上才会生效（部分 openai_compat 中转如此）。
真实模型走 ollama 产出不了内联形态，所以这条路径**无法用真模型实测**，
只能用假 LLM 覆盖 —— 这正是本文件存在的原因。

夹具正文是两个文件，取自同一次真实生成（见 `fixtures/README.md`）。
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.postprocess import postprocess
from app.main import app
from helpers import T_CLOSE, T_OPEN, parse_sse, think_block

_FIXTURES = Path(__file__).parent / "fixtures"
REASONING = (_FIXTURES / "qwen3_reasoning.txt").read_text(encoding="utf-8")
ANSWER = (_FIXTURES / "qwen3_answer.txt").read_text(encoding="utf-8")

# 一个识别不出的结束标记：模拟「各家推理标记不统一」
UNKNOWN_CLOSER = "｜end" + "▁of▁thinking｜"


def _inline(reasoning: str, answer: str) -> str:
    """模拟「推理内联在 content 里」的 provider 输出。"""
    return think_block(reasoning) + answer


class InlineThinkLLM:
    """把思维链内联在 content 里的假 LLM。

    按固定窗口切块吐出，让标签本身也可能被切到两个 chunk 里 —— 真实流式就是
    这样，而 postprocess 是对**拼接后的完整文本**做的，跨块拼接必须成立。
    """

    def __init__(self, reasoning: str = REASONING, answer: str = ANSWER, chunk_size: int = 64):
        self.reasoning = reasoning
        self.answer = answer
        self.chunk_size = chunk_size

    def _whole(self) -> str:
        return _inline(self.reasoning, self.answer)

    def chat(self, messages) -> str:
        return self._whole()

    def stream(self, messages):
        whole = self._whole()
        for i in range(0, len(whole), self.chunk_size):
            yield whole[i : i + self.chunk_size]


@pytest.fixture
def think_pipeline(make_fake_pipeline):
    """注入内联 think 的假 LLM，其余（Milvus/Retriever/SQL/记忆）都是真实装配。"""
    return make_fake_pipeline(InlineThinkLLM())


# ---------------- postprocess 单元层 ----------------


def test_strip_is_lossless():
    """去 think 之后必须与「本来就没有 think 块」完全一致。

    这是本文件的核心不变式：清洗只能去掉推理，不能顺带改动正文一个字。
    用真实推理文本（833 字、18 行，含中文口语和换行）作载荷。
    """
    assert postprocess(_inline(REASONING, ANSWER)) == postprocess(ANSWER)


def test_real_reasoning_is_gone_and_answer_survives():
    out = postprocess(_inline(REASONING, ANSWER))

    assert out != _inline(REASONING, ANSWER), "清洗没生效"
    assert REASONING[:30] not in out, "推理文本残留"
    assert ANSWER[:30] in out, "正文开头被误删"
    assert ANSWER.strip()[-30:] in out, "正文结尾被误删"


def test_answer_markdown_structure_survives():
    """真实正文里有 ### / **加粗** / --- 分隔线，清洗不能把它们压平。"""
    out = postprocess(_inline(REASONING, ANSWER))

    for marker in ("### ", "**", "---"):
        assert marker in out, f"markdown 标记 {marker!r} 被清洗破坏了"


def test_unknown_closer_keeps_answer_instead_of_eating_it():
    """结束标记认不出时，只能保留，绝不能连正文一起删。

    早期实现有「只有开始标签就删到结尾」的兜底，实测会把整篇答案清空——
    比残留一段推理文本严重得多，故已移除。这里用真实推理文本再钉一次。
    """
    src = T_OPEN + REASONING + UNKNOWN_CLOSER + ANSWER
    out = postprocess(src)

    assert ANSWER[:30] in out, "正文被兜底逻辑吃掉了"
    assert out.strip().endswith(ANSWER.strip()[-10:])


# ---------------- pipeline 装配层 ----------------


def test_pipeline_nonstream_strips_inline_think(think_pipeline):
    result = think_pipeline.answer("高血压注意什么？", "psychologist", "t1")

    assert REASONING[:30] not in result["answer"]
    assert ANSWER[:30] in result["answer"]


def test_pipeline_stream_final_is_cleaned(think_pipeline):
    """`final` 是 /chat/stream 的 done 事件来源，必须是清洗后的权威文本。"""
    final: list[str] = []
    deltas = list(
        think_pipeline.answer_stream("高血压注意什么？", "psychologist", "t2", final=final)
    )

    assert deltas, "没有产出任何增量"
    assert len(final) == 1, "final 必须恰好回传一次"
    assert REASONING[:30] not in final[0]
    assert ANSWER[:30] in final[0]


def test_stream_deltas_are_raw_by_design(think_pipeline):
    """增量是**原始**文本（含推理），清洗只作用于 done 事件 —— 这是刻意的。

    正则清洗需要完整文本（跨块语义会被逐块清洗破坏），所以流式期间前端看到的是
    原始增量，收到 done 后覆盖显示。代价是：若 provider 内联推理，用户在生成过程中
    会短暂看到推理文本。ollama 走独立字段，不存在这个问题（见文件头说明）。
    """
    raw = "".join(think_pipeline.answer_stream("高血压注意什么？", "psychologist", "t3"))

    assert REASONING[:30] in raw, "增量本就该是原始的，这里变了说明契约改了"
    assert postprocess(raw) == postprocess(_inline(REASONING, ANSWER))


def test_memory_stores_cleaned_answer_not_raw(think_pipeline):
    """写进短期记忆的必须是清洗后的版本。

    否则下一轮模型会读到自己上一轮带推理链的原始输出 —— 这正是流式绕过
    postprocess 那个 bug 的后果（P0-2）。
    """
    final: list[str] = []
    list(think_pipeline.answer_stream("高血压注意什么？", "psychologist", "t4", final=final))

    hist = think_pipeline.memory.get_history(
        think_pipeline._memory_key("t4", "psychologist")
    )
    assistant_msgs = [h["content"] for h in hist if h["role"] == "assistant"]

    assert len(assistant_msgs) == 1
    assert assistant_msgs[0] == final[0], "记忆里的答案与 done 事件不一致"
    assert REASONING[:30] not in assistant_msgs[0], "记忆里存了原始推理"


# ---------------- HTTP 接口层 ----------------


def test_api_done_event_carries_cleaned_answer(think_pipeline):
    with TestClient(app) as c:
        app.state.pipeline = think_pipeline
        resp = c.post(
            "/chat/stream", json={"question": "高血压注意什么？", "role_id": "psychologist"}
        )

    assert resp.status_code == 200
    events = parse_sse(resp.text)

    assert "error" not in events, f"流式生成报错：{events.get('error')}"
    assert len(events["done"]) == 1, "done 事件必须恰好一个"
    done = events["done"][0]["answer"]

    assert REASONING[:30] not in done, "done 事件里漏了推理内容"
    assert ANSWER[:30] in done


def test_api_deltas_plus_done_reconstruct_the_answer(think_pipeline):
    """端到端不变式：done == postprocess(所有 delta 拼接)。"""
    with TestClient(app) as c:
        app.state.pipeline = think_pipeline
        resp = c.post(
            "/chat/stream", json={"question": "高血压注意什么？", "role_id": "psychologist"}
        )

    events = parse_sse(resp.text)
    raw = "".join(d["delta"] for d in events["delta"])

    assert events["done"][0]["answer"] == postprocess(raw)
