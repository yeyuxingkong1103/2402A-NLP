"""上下文预算与截断留痕。

背景（2026-09-17 实测，qwen3:8b / n_ctx=16384）：prompt 吃满上下文时有两种坏结果，
此前**都是静默的**：

  1. 回答被砍断但内容非空 —— 既不触发 EmptyAnswerError（那只拦空串），
     还会被写进短期记忆，下一轮模型基于半截回答继续编；
  2. 历史被 Ollama 丢掉，而应用日志照样打印「历史裁剪：保留 N 轮」，日志和实际对不上。

共同根因：历史预算管不到系统消息，而系统消息里塞着 TOP_K 条检索资料，大小随
chunk_size 走。实测（用 Ollama 的 prompt_eval_count 读真值）：chunk_size=2000 时
prompt 15355 tokens，还装得下；3000 时 18850 tokens 就超了，服务端只评估 14760，
静默丢掉最老轮次。
"""
from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from app.core.config import Settings
from app.core.document import CHUNK_OVERLAP_MAX, CHUNK_SIZE_MAX, CHUNK_SIZE_MIN
from app.core.llm import LLMClient
from app.core.prompt.templates import build_messages
from helpers import RecordingLLM


# ===== 截断留痕 =====


def _resp(content: str, finish_reason: str | None):
    return NS(choices=[NS(message=NS(content=content), finish_reason=finish_reason)])


def _chunk(content: str | None = None, finish_reason: str | None = None):
    return NS(
        choices=[
            NS(delta=NS(content=content) if content is not None else None, finish_reason=finish_reason)
        ]
    )


def _llm_returning(payload) -> LLMClient:
    """造一个 provider 非 dummy、但底层 OpenAI 客户端被换掉的 LLMClient。"""
    llm = LLMClient(Settings(llm_provider="ollama", llm_model="test-model"))
    llm._client = NS(chat=NS(completions=NS(create=lambda **kw: payload)))
    return llm


def test_truncated_answer_is_logged(caplog):
    """finish_reason=length 必须留痕——否则截断完全不可见。"""
    llm = _llm_returning(_resp("半截回答", "length"))
    with caplog.at_level(logging.WARNING, logger="llm"):
        out = llm.chat([{"role": "user", "content": "问"}])

    assert out == "半截回答", "被截断的回答仍然要返回，不能改成报错"
    assert "finish_reason=length" in caplog.text


def test_complete_answer_is_not_logged(caplog):
    """正常结束不许误报，否则这条告警很快就被当成噪音忽略。"""
    llm = _llm_returning(_resp("完整回答", "stop"))
    with caplog.at_level(logging.WARNING, logger="llm"):
        llm.chat([{"role": "user", "content": "问"}])

    assert "截断" not in caplog.text


def test_stream_truncation_is_logged(caplog):
    """流式的 finish_reason 只在最后一个块里，漏掉它就等于没做这件事。"""
    llm = _llm_returning([_chunk("前半"), _chunk("后半"), _chunk(None, "length")])
    with caplog.at_level(logging.WARNING, logger="llm"):
        out = "".join(llm.stream([{"role": "user", "content": "问"}]))

    assert out == "前半后半"
    assert "finish_reason=length" in caplog.text


def test_stream_normal_finish_is_not_logged(caplog):
    llm = _llm_returning([_chunk("答"), _chunk(None, "stop")])
    with caplog.at_level(logging.WARNING, logger="llm"):
        "".join(llm.stream([{"role": "user", "content": "问"}]))

    assert "截断" not in caplog.text


# ===== 整体预算（系统消息 + 历史 + 提问）=====

PERSONA = {"name": "医生", "system_prompt": "你是医生"}
QUESTION = "问"


def _contexts(n: int, size: int) -> list[dict]:
    return [{"text": "控" * size, "title": f"资料{i}", "source": "s", "score": 0.9} for i in range(n)]


def _history(turns: int, answer_len: int) -> list[dict]:
    out: list[dict] = []
    for i in range(turns):
        out.append({"role": "user", "content": f"第{i}问"})
        out.append({"role": "assistant", "content": "答" * answer_len})
    return out


def _total(messages: list[dict]) -> int:
    return sum(len(m["content"]) for m in messages)


def test_prompt_budget_shrinks_history_when_system_message_is_large():
    """系统消息一大，历史要让位。

    不这么做的话：历史按自己的额度照留，prompt 整体超窗，Ollama 反过来静默丢最老
    轮次——正是这套裁剪想消灭的情况。
    """
    msgs = build_messages(
        PERSONA, _contexts(5, 1000), _history(30, 200), QUESTION,
        max_history_chars=12000, max_prompt_chars=8000,
    )
    assert _total(msgs) <= 8000


def test_prompt_budget_keeps_history_within_its_own_cap():
    """反向：系统消息很小时，整体预算不该把历史额度放宽。"""
    msgs = build_messages(
        PERSONA, [], _history(30, 200), QUESTION,
        max_history_chars=500, max_prompt_chars=100000,
    )
    # 用真实的 system 消息长度来扣，不要写死成「人设」：零召回时 system 里还会多一段
    # 「没有资料」的作答约束（见 templates.py），按人设长度扣会把它算进历史里。
    history_chars = _total(msgs) - len(msgs[0]["content"]) - len(QUESTION)
    assert history_chars <= 500


def test_prompt_budget_none_means_no_trimming():
    """两个预算都不给时保持原行为（不裁）。"""
    msgs = build_messages(PERSONA, [], _history(30, 200), QUESTION)
    assert _total(msgs) > 5000


def test_prompt_budget_never_drops_the_newest_turn():
    """系统消息已吃满预算时，最新一轮仍要保留——否则模型看不到上下文。"""
    msgs = build_messages(
        PERSONA, _contexts(5, 2000), _history(3, 100), QUESTION,
        max_history_chars=12000, max_prompt_chars=100,
    )
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert msgs[-1]["content"] == QUESTION


def test_warns_when_system_message_alone_eats_the_budget(caplog):
    """系统消息一项就吃满预算时必须告警。

    这是 Ollama 超窗那条路上**唯一会响的警报**：此时历史怎么裁都救不回来，
    prompt 必然超窗，而 Ollama 是静默丢最老轮次（实测 finish_reason 仍是 stop，
    不报错），日志里只剩一条与实际不符的「历史裁剪：保留 N 轮」。
    """
    with caplog.at_level(logging.WARNING, logger="prompt"):
        build_messages(
            PERSONA, _contexts(5, 2000), _history(3, 100), QUESTION,
            max_history_chars=12000, max_prompt_chars=100,
        )
    assert "预算被吃光" in caplog.text
    assert "本轮提问" in caplog.text, "要把提问长度也报出来——吃光预算的常常是提问本身，只怪 chunk_size 会带偏排查"


def test_no_warning_while_history_can_still_yield(caplog):
    """历史还能让位时不该报警——否则警报很快被当噪音。"""
    with caplog.at_level(logging.WARNING, logger="prompt"):
        build_messages(
            PERSONA, _contexts(5, 500), _history(30, 200), QUESTION,
            max_history_chars=12000, max_prompt_chars=8000,
        )
    assert "预算被吃光" not in caplog.text


def test_pipeline_applies_prompt_budget(make_fake_pipeline):
    """接线验证：整体预算得真传进 build_messages。

    参数是可选的，忘了传照样能跑、单测也照样过——所以要在真装配上验一次。
    """
    llm = RecordingLLM("回答" * 100)  # 与 test_pipeline.py 共用实现，这里只是答复长度不同
    p = make_fake_pipeline(llm)
    p.settings.history_max_chars = 1_000_000   # 关掉历史自己的预算，只剩整体预算管事
    p.settings.prompt_max_chars = 400

    for i in range(10):
        p.answer(f"第{i}问：高血压要注意什么？", "psychologist", "budget1")

    sent = llm.calls[-1]
    history_msgs = sent[1:-1]
    assert len(history_msgs) == 2, f"整体预算没生效：历史留了 {len(history_msgs)} 条"
    # 留下的必须是**最新**一轮，不是随便一轮
    assert history_msgs[-1]["content"] == "回答" * 100


# ===== chunk_size 上界 =====
#
# 这是上面那套预算的入口防线：chunk_size 直接决定系统消息里 TOP_K 条资料有多大，
# 而系统消息不受 HISTORY_MAX_CHARS 约束。此前它是无上界的查询参数，一次
# ?chunk_size=3000 就能把 prompt 顶到 18850 tokens（n_ctx=16384），
# 服务端静默丢最老轮次。

_DOC = {"file": ("doc.md", "# 标题\n\n高血压患者应限制钠盐摄入。", "text/markdown")}


def test_upload_rejects_oversized_chunk_size(client):
    """边界声明在接口参数上（Query 的 le/ge），所以它挡不住绕开接口的 CLI——下一节另有一道。"""
    resp = client.post(
        "/knowledge/upload", files=_DOC,
        params={"role_id": "psychologist", "chunk_size": 2000},
    )
    assert resp.status_code == 422


def test_upload_accepts_chunk_size_at_the_boundary(client):
    resp = client.post(
        "/knowledge/upload", files=_DOC,
        params={"role_id": "psychologist", "chunk_size": 1000},
    )
    assert resp.status_code == 200


def test_upload_rejects_negative_overlap(client):
    """越界就当场拒，不做静默夹取：chunker 里也会把 overlap 夹进区间，但那是给内部调用兜底的，
    接口上悄悄改掉用户传的值会让人以为参数生效了。
    """
    resp = client.post(
        "/knowledge/upload", files=_DOC,
        params={"role_id": "psychologist", "chunk_overlap": -1},
    )
    assert resp.status_code == 422


# ===== 离线 CLI 的同一道防线 =====
#
# scripts/ingest.py 直接写 Milvus，不经过 /knowledge/upload，所以接口上的边界拦不住
# 它：`--chunk-size 5000` 照样能入库，检索再把这种巨块整段塞进系统消息。
# 两边现在共用 app/core/document/chunker.py 里的常量。

_INGEST_SPEC = importlib.util.spec_from_file_location(
    "ingest_cli", Path(__file__).resolve().parents[1] / "scripts" / "ingest.py"
)
ingest_cli = importlib.util.module_from_spec(_INGEST_SPEC)
_INGEST_SPEC.loader.exec_module(ingest_cli)


@pytest.mark.parametrize(
    "flag,value",
    [("--chunk-size", CHUNK_SIZE_MAX + 1), ("--chunk-size", CHUNK_SIZE_MIN - 1), ("--overlap", CHUNK_OVERLAP_MAX + 1)],
)
def test_ingest_cli_rejects_out_of_range(flag, value, monkeypatch, tmp_path):
    doc = tmp_path / "doc.md"
    doc.write_text("# 标题\n\n高血压患者应限制钠盐摄入。", encoding="utf-8")
    called = []
    monkeypatch.setattr(ingest_cli, "ingest_path", lambda *a, **kw: called.append(a) or 0)
    monkeypatch.setattr(sys, "argv", ["ingest.py", "--file", str(doc), flag, str(value)])

    with pytest.raises(SystemExit) as exc:
        ingest_cli.main()

    assert exc.value.code == 2, "越界参数必须当场退出（argparse 的用法错误码）"
    assert not called, "参数非法时不该再去动 Milvus"


def test_ingest_cli_accepts_boundary_value(monkeypatch, tmp_path):
    """边界值本身要放行——上界是「不许超过」，不是「不许等于」。"""
    doc = tmp_path / "doc.md"
    doc.write_text("# 标题\n\n高血压患者应限制钠盐摄入。", encoding="utf-8")
    seen: dict = {}
    monkeypatch.setattr(
        ingest_cli,
        "ingest_path",
        lambda path, role, chunk_size, overlap, strategy, re_ingest, summary, dedup: seen.update(
            chunk_size=chunk_size, overlap=overlap
        ) or 0,
    )
    monkeypatch.setattr(
        sys, "argv", ["ingest.py", "--file", str(doc), "--chunk-size", str(CHUNK_SIZE_MAX)]
    )

    ingest_cli.main()

    assert seen["chunk_size"] == CHUNK_SIZE_MAX
