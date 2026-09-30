"""src/online/chain.py —— 在线对话链路编排（新架构的 RAG 主流程）。

在链路中的位置（新架构在线侧的总编排）：
    src/api/routers/chat.py → 【本文件】 → 检索 → 精排 → 记忆 → 提示词 → 生成 → 后处理
下游：src/online/retriever.py、reranker.py、prompt_builder.py、llm.py、postprocess.py
      src/memory/short_term.py（SQLite 短期记忆）、long_term.py（向量长期记忆）

与 backend/server.py 的差异：
    backend 那条主线把编排写在接口函数里（generate_answer / ask_stream）；
    这里抽成独立的 run_chain / stream_chain，接口层只负责协议转换。
    好处是同一套链路既能被 HTTP 接口调用、也能被测试或 CLI 直接调用。

对外提供两个入口，共用同一套步骤，区别只在"生成"那一步：
    run_chain    一次性生成完再返回（普通 POST）
    stream_chain 逐字 yield（SSE 流式）
"""
from __future__ import annotations

from typing import Any, Iterable

from src.memory.long_term import remember_turn, retrieve_memory
from src.memory.short_term import append_message, recent_messages
from src.online.llm import llm_client
from src.online.postprocess import clean_answer, enforce_refusal, references
from src.online.prompt_builder import build_prompt
from src.online.reranker import reranker
from src.online.retriever import retrieve


def run_chain(role: dict[str, Any], user_id: int, session_id: int, message: str, tenant_id: str = "default") -> dict[str, Any]:
    """执行一轮完整的非流式角色对话。

    参数：
        role: 角色配置（含 role_id / top_k / bound_kb / rerank_top_k / temperature）
        user_id / session_id: 会话定位
        message: 本轮用户消息
        tenant_id: 租户 id
    返回：
        {"answer": 答案, "references": 引用列表, "trace": 链路明细,
         "rewritten_query": 改写后的查询, "memory": {"short_count", "long_hits"}}

    步骤顺序（顺序本身就是设计，不能随意调换）：
        1. 读短期记忆（recent_messages）—— 必须在写入本轮消息**之前**读，
           否则会把本轮问题也算进"历史"，让上下文里出现重复
        2. 检索（retrieve）—— 传入 short 让检索能做指代消解
           （"它有什么要求？"里的"它"要靠历史才能还原）
        3. 精排（reranker.rerank）
        4. 读长期记忆（retrieve_memory）—— 按当前问题做向量检索
        5. 拼提示词（build_prompt）
        6. 生成 → 清洗 → 拒答检查
           clean_answer 先去噪，再交给 enforce_refusal 判断"是否该拒答"
        7. 落库：写入用户消息和助手消息
        8. 记住本轮（remember_turn）—— 只存用户说的话，不存模型输出

    为什么"先检索再写历史"而不是反过来：
        和第 1 点同因。写历史在前的话，检索时看到的历史里已经包含本轮问题，
        指代消解会把"它"错误地解析成问题里的其他词。

    memory 字段里的 short_count 是 len(short)：
        即在读历史那一刻的条数（不含本轮），与 backend/roleplay.py 的口径略有不同，
        但都如实反映"这一轮实际用了多少条历史"。
    """
    short = recent_messages(user_id, session_id)
    bundle = retrieve(message, role["role_id"], tenant_id, short, role.get("top_k"), role.get("bound_kb"))
    # 精排要用改写后的查询（rewritten_query）：它才是被检索实际使用的语义表达
    ranked = reranker.rerank(bundle["rewritten_query"], bundle["hits"], role.get("rerank_top_k"))
    long = retrieve_memory(user_id, role["role_id"], message, tenant_id)
    messages = build_prompt(role, long, short, ranked, message)
    # 三层嵌套：先清洗噪声，再检查是否需要拒答（依据不足时替换成固定拒答文案）
    answer = enforce_refusal(clean_answer(llm_client.complete(messages, role.get("temperature", 0.4))), ranked)
    append_message(user_id, session_id, role["role_id"], "user", message)
    append_message(user_id, session_id, role["role_id"], "assistant", answer)
    # 长期记忆存用户原话（而非模型回答）：需要被记住的是用户的偏好和处境
    remember_turn(user_id, role["role_id"], session_id, message, tenant_id)
    return {"answer": answer, "references": references(ranked), "trace": bundle.get("trace", []), "rewritten_query": bundle.get("rewritten_query"), "memory": {"short_count": len(short), "long_hits": len(long)}}


def stream_chain(role: dict[str, Any], user_id: int, session_id: int, message: str, tenant_id: str = "default") -> Iterable[dict[str, Any]]:
    """执行一轮流式角色对话，逐个事件 yield。

    参数：
        同 run_chain
    返回（生成器）：
        依次产出四类事件：
            {"event": "delta", "data": {"content": 增量文本}}   —— 多次
            {"event": "error", "data": {"message": 错误信息}}   —— 出错时（终止）
            {"event": "references", "data": {"references", "trace"}}
            {"event": "done", "data": {"answer", "rewritten_query"}}

    前面四步（读历史 → 检索 → 精排 → 读长期记忆 → 拼提示词）与非流式完全一致，
    只有生成那一步改成逐块 yield 出去，实现打字机效果。

    流式场景的"边生成边落库"：
        生成过程中边收边攒到 chunks 列表，全部收完后再统一做
        clean_answer + enforce_refusal，然后才写库。
        为什么不在生成过程中就写：拒答检查和清洗都需要看完整答案才能判断，
        边流边写会把未清洗的内容落进历史，污染后续轮次的上下文。

    异常处理的选择：
        生成阶段出错时 yield 一个 error 事件就 return ——
        此时答案不完整，所以**不写库**，也 yield 不出 references/done。
        让前端知道这一轮失败了，比落一条残缺记录更好。

    references 与 done 分两个事件发：
        引用和 trace 在答案生成完就已确定，可以比 done 早一点送到前端；
        前端据此可以先渲染出引用卡片，不用等整个连接关闭。
    """
    short = recent_messages(user_id, session_id)
    bundle = retrieve(message, role["role_id"], tenant_id, short, role.get("top_k"), role.get("bound_kb"))
    ranked = reranker.rerank(bundle["rewritten_query"], bundle["hits"], role.get("rerank_top_k"))
    long = retrieve_memory(user_id, role["role_id"], message, tenant_id)
    messages = build_prompt(role, long, short, ranked, message)
    chunks = []
    try:
        for delta in llm_client.stream(messages, role.get("temperature", 0.4)):
            chunks.append(delta)
            yield {"event": "delta", "data": {"content": delta}}
    except Exception as exc:
        yield {"event": "error", "data": {"message": str(exc)}}
        return  # 出错则不再落库、不发送后续事件
    # 收完增量后再统一清洗与拒答检查（见 docstring 说明）
    answer = enforce_refusal(clean_answer("".join(chunks)), ranked)
    append_message(user_id, session_id, role["role_id"], "user", message)
    append_message(user_id, session_id, role["role_id"], "assistant", answer)
    remember_turn(user_id, role["role_id"], session_id, message, tenant_id)
    yield {"event": "references", "data": {"references": references(ranked), "trace": bundle.get("trace", [])}}
    yield {"event": "done", "data": {"answer": answer, "rewritten_query": bundle.get("rewritten_query")}}
