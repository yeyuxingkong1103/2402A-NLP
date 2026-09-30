from dataclasses import replace

from app.core.pipeline import RAGPipeline
from app.core.prompt.templates import build_messages, trim_history
from helpers import RecordingLLM


def test_build_messages_combines_persona_context_history():
    persona = {"name": "医生", "system_prompt": "你是医生"}
    contexts = [{"text": "限制钠盐摄入", "title": "指南", "source": "s"}]
    history = [
        {"role": "user", "content": "之前问过"},
        {"role": "assistant", "content": "之前答过"},
    ]
    msgs = build_messages(persona, contexts, history, "高血压吃什么？")

    assert msgs[0]["role"] == "system"
    assert "限制钠盐摄入" in msgs[0]["content"]
    assert msgs[-1] == {"role": "user", "content": "高血压吃什么？"}
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]


def test_build_messages_degrades_when_nothing_retrieved():
    """零召回时不能只剩人设，否则模型会按常识把专业问答答得很像样。

    实测场景：集合没建/名字配错、score_threshold 全滤、BM25 那一路失效时 /chat 返回
    200 + sources: []，而 system 里没有任何"没有资料"的约束——用户看到的是一段像模像样
    的医学回答。又不能一刀切拒答（没入库的角色一开口就是"没有相关内容"），
    所以只禁具体事实、保留角色对话。
    """
    persona = {"name": "医生", "system_prompt": "你是医生"}
    msgs = build_messages(persona, [], [], "高血压吃什么？")
    system = msgs[0]["content"]

    assert "没有检索到任何知识库资料" in system
    assert "闲聊、寒暄相关的回应照常给出" in system, "角色扮演对话不能被一刀切拒答"
    assert "不要凭空给出具体的专业事实" in system


def test_build_messages_empty_context_branch_not_added_when_contexts_exist():
    """反过来的误加：有资料时不能同时出现"没有资料"那句（自相矛盾会稀释约束）。"""
    persona = {"name": "医生", "system_prompt": "你是医生"}
    contexts = [{"text": "限制钠盐摄入", "title": "指南", "source": "s"}]
    system = build_messages(persona, contexts, [], "高血压吃什么？")[0]["content"]

    assert "唯一可用的事实来源" in system
    assert "没有检索到任何知识库资料" not in system


def test_build_messages_constraint_is_not_medical_only():
    """约束不能写死成「医学内容」：这份模板服务全部角色，其中没有一个是纯医疗角色。

    实测律师题（资料只写「超过一个月不满一年应付二倍工资」）回答补出「最长不超过
    11 个月」——写死成医学的话，这条禁令对律师那位根本没有约束力。
    """
    persona = {"name": "律师", "system_prompt": "你是律师"}
    contexts = [{"text": "超过一个月不满一年应付二倍工资", "title": "劳动法", "source": "s"}]
    system = build_messages(persona, contexts, [], "没签合同怎么办？")[0]["content"]

    assert "不要补充资料之外的专业内容" in system
    assert "医学内容" not in system
    assert "唯一可用的事实来源" in system


def test_answer_returns_answer_sources_and_writes_memory(fake_pipeline):
    """sources 里必须原样带上检索到的文本（前端来源列表靠它），记忆里必须是成对的问-答。"""
    result = fake_pipeline.answer("高血压饮食注意什么", "psychologist", "s1")

    assert "回复" in result["answer"]
    assert result["sources"][0]["text"] == "限制钠盐摄入"
    assert result["session_id"] == "s1"

    hist = fake_pipeline.memory.get_history(fake_pipeline._memory_key("s1", "psychologist"))
    assert [h["role"] for h in hist] == ["user", "assistant"]


def test_answer_stream_writes_memory(fake_pipeline):
    deltas = list(fake_pipeline.answer_stream("高血压？", "psychologist", "s2"))
    assert len(deltas) >= 1
    assert "回复" in "".join(deltas)
    assert len(fake_pipeline.memory.get_history(fake_pipeline._memory_key("s2", "psychologist"))) == 2


def test_unknown_role_raises(fake_pipeline):
    """管线层用 ValueError 表达「没这个角色」，HTTP 层靠它分流成 404——类型改了会让接口变 500。"""
    import pytest

    with pytest.raises(ValueError):
        fake_pipeline.answer("问题", "no_such_role", "s3")


# ---- 历史裁剪：按字符预算，而不是按轮数 ----
#
# 动因：Ollama 默认 num_ctx 只有 4096，超了会**静默**丢掉最老的几轮。实测单条回答
# 1670 字时只装得下 3 轮——所以 MEMORY_MAX_TURNS 设 10 还是 20 没有任何区别。
# 裁在应用侧，至少是可预测、可记录的行为。


def _turns(*pairs):
    """构造历史：[user, assistant] 交替。"""
    out = []
    for q, a in pairs:
        out.append({"role": "user", "content": q})
        out.append({"role": "assistant", "content": a})
    return out


def test_trim_history_keeps_newest_turns_within_budget():
    h = _turns(("q1", "a" * 100), ("q2", "b" * 100), ("q3", "c" * 100))
    kept = trim_history(h, max_chars=250)

    answers = [m["content"] for m in kept if m["role"] == "assistant"]
    assert answers == ["b" * 100, "c" * 100], "应该保留最新的两轮"
    assert "q1" not in [m["content"] for m in kept], "最老的一轮该被丢掉"


def test_trim_history_always_keeps_newest_turn_even_if_oversized():
    """最新一轮即使自己就超预算也要留下，否则整段历史会全空。"""
    h = _turns(("q1", "a" * 10), ("q2", "b" * 5000))
    kept = trim_history(h, max_chars=50)

    assert len(kept) == 2
    assert kept[-1]["content"] == "b" * 5000


def test_trim_history_never_splits_a_turn():
    """裁剪以「轮」为单位，不能留下有问无答或答非所问的半轮。"""
    h = _turns(("q1", "a" * 100), ("q2", "b" * 100), ("q3", "c" * 100))
    kept = trim_history(h, max_chars=210)

    roles = [m["role"] for m in kept]
    assert roles == ["user", "assistant", "user", "assistant"], f"轮被拆散了: {roles}"
    assert roles[0] == "user", "不能以 assistant 开头"


def test_trim_history_drops_headless_assistant():
    """历史若以 assistant 开头（存储损坏），那条无头消息要丢掉。"""
    h = [{"role": "assistant", "content": "孤儿回答"}] + _turns(("q1", "a" * 10))
    kept = trim_history(h, max_chars=None)

    assert [m["role"] for m in kept] == ["user", "assistant"]


def test_trim_history_none_means_no_trimming():
    h = _turns(*[(f"q{i}", f"a{i}") for i in range(30)])
    assert len(trim_history(h, max_chars=None)) == 60


def test_build_messages_applies_history_budget():
    h = _turns(("q1", "a" * 100), ("q2", "b" * 100), ("q3", "c" * 100))
    msgs = build_messages({"name": "医生", "system_prompt": "你是医生"}, [], h, "新问题",
                          max_history_chars=250)

    assert msgs[0]["role"] == "system"
    assert msgs[-1] == {"role": "user", "content": "新问题"}
    assert len(msgs) == 1 + 4 + 1, f"应该只留最新两轮，实际 {len(msgs)} 条"


def test_pipeline_passes_history_budget_from_settings(fake_pipeline):
    """管线必须把 settings 里的预算接上，否则裁剪形同虚设。"""
    fake_pipeline.settings.history_max_chars = 120

    for i in range(6):
        fake_pipeline.answer(f"第{i}问：高血压？", "psychologist", "budget1")

    hist = fake_pipeline.memory.get_history(
        fake_pipeline._memory_key("budget1", "psychologist")
    )
    assert len(hist) > 2, "记忆本身不该被裁——裁的是送进提示词的部分"

    msgs = build_messages(
        {"name": "医生", "system_prompt": "你是医生"},
        [],
        hist,
        "新问题",
        max_history_chars=fake_pipeline.settings.history_max_chars,
    )
    history_msgs = msgs[1:-1]
    assert sum(len(m["content"]) for m in history_msgs) <= 120


def test_pipeline_actually_trims_before_calling_llm(make_fake_pipeline):
    """真装配验证：裁过的历史才是发给模型的。

    这条抓的是接线——单测 `trim_history` 通过、但管线忘了传 max_history_chars 时，
    裁剪形同虚设，而那是很容易发生的疏忽（参数是可选的）。
    """
    llm = RecordingLLM()
    p = make_fake_pipeline(llm)
    p.settings.history_max_chars = 300

    for i in range(8):
        p.answer(f"第{i}问：高血压要注意什么？", "psychologist", "trim1")

    sent = llm.calls[-1]
    # 去掉开头的 system 和末尾的本轮提问，中间就是历史
    history_msgs = [m for m in sent[1:] if m["role"] in ("user", "assistant")][:-1]

    total = sum(len(m["content"]) for m in history_msgs)
    assert total <= 300, f"超预算的历史没被裁掉：{total} 字符"
    assert len(history_msgs) < 8 * 2, "8 轮全在，说明根本没裁"

    # 记忆本身不该被裁——裁的只是送进提示词的部分
    assert len(p.memory.get_history(p._memory_key("trim1", "psychologist"))) == 16


# ---- 开精排时粗排池要宽于 top_k ----
#
# 精排只能从粗排召回的池子里挑人。池宽等于 top_k 时，精排最多换个顺序，
# 真正该进上下文的第 12 名永远上不来——BGE-rerank 配了等于没配。
def test_bge_rerank_widens_recall_pool(dummy_settings):
    """开着精排时池宽必须宽于 top_k，否则精排至多换个顺序、第 12 名永远上不来（配了等于没配）。

    同时锁住「没开精排时行为与加池子之前一字不差」——score_fusion 下池宽仍等于 top_k。
    """
    retrieved_with = []

    class SpyRetriever:
        def retrieve(self, query, role_id, top_k=None):
            retrieved_with.append(top_k)
            return [
                {"id": str(i), "text": f"候选{i}", "score": 1.0 - i * 0.01}
                for i in range(top_k)
            ]

    class NoopReranker:
        @staticmethod
        def rerank(query, candidates):
            return candidates

    def make(**overrides):
        kwargs = {"top_k": 5, "rerank_pool": 20}
        kwargs.update(overrides)
        s = replace(dummy_settings, **kwargs)
        # retrieve() 只碰 retriever / reranker / settings，其余依赖传 None 即可
        return RAGPipeline(
            s, None, None, None, None, None,
            retriever=SpyRetriever(), reranker=NoopReranker(),
        )

    # score_fusion：池宽就是 top_k，与加精排池之前的行为一致
    out = make(reranker="score_fusion").retrieve("q", "psychologist")
    assert retrieved_with[-1] == 5
    assert len(out) == 5

    # bge：先按 rerank_pool 宽召回，最后仍只给 top_k 条
    out = make(reranker="bge").retrieve("q", "psychologist")
    assert retrieved_with[-1] == 20
    assert len(out) == 5

    # 池宽跟着 rerank_pool 走
    make(reranker="bge", rerank_pool=8).retrieve("q", "psychologist")
    assert retrieved_with[-1] == 8


def test_rerank_input_is_capped_by_pool(dummy_settings):
    """池宽是精排**输入的上界**，不只是「每条 query 的召回宽度」。

    开了 query 改写后，hybrid_retriever 会把 ≤4 条 query 的召回结果去重合并返回，
    条数可达 pool 的数倍（实测 rerank_pool=20 + 4 条 query → 80 条，还没算 BM25 那一路），
    于是一股脑发几十对给 CrossEncoder：本机逐对推理，60s 超时就整体降级回 score_fusion，
    精排等于白开。这里断言精排实际收到的条数不超过池宽。
    """
    seen = []
    kept_ids = []

    class FloodRetriever:
        """模拟多查询合并后的超额召回：不管要多少，都给 80 条。"""

        def retrieve(self, query, role_id, top_k=None):
            return [
                {"id": str(i), "text": f"候选{i}", "score": 1.0 - i * 0.001}
                for i in range(80)
            ]

    class SpyReranker:
        @staticmethod
        def rerank(query, candidates):
            seen.append(len(candidates))
            kept_ids.append([c["id"] for c in candidates])
            return candidates

    s = replace(dummy_settings, reranker="bge", rerank_pool=20, top_k=5)
    p = RAGPipeline(
        s, None, None, None, None, None,
        retriever=FloodRetriever(), reranker=SpyReranker(),
    )

    assert len(p.retrieve("q", "psychologist")) == 5
    assert seen == [20], f"精排收到 {seen} 条，池宽承诺失效"
    # 截掉的必须是融合分数最低的尾部：RRF 已经排好序，取前池宽条
    assert kept_ids[0] == [str(i) for i in range(20)]

    # 没超池宽时不能截（12 条进 12 条），否则等于凭空丢候选
    class SmallRetriever(FloodRetriever):
        def retrieve(self, query, role_id, top_k=None):
            return super().retrieve(query, role_id, top_k)[:12]

    p2 = RAGPipeline(
        s, None, None, None, None, None,
        retriever=SmallRetriever(), reranker=SpyReranker(),
    )
    p2.retrieve("q", "psychologist")
    assert seen[-1] == 12
