# -*- coding: utf-8 -*-
"""问答模块：提示词组装 + DeepSeek 调用（含流式）+ 记忆读写 + 改写与缓存，不含 HTTP 接口。"""

import argparse                               # 导入 argparse，用于解析命令行参数
import json                                   # 导入 json，用于流式结束时输出来源
import time                                   # 导入 time，用于统计耗时

import config                                 # 导入配置模块，模型参数与密钥都从这里读
import db_user                                # 导入用户数据模块，用于落库会话消息
import memory                                 # 导入记忆模块，用于读写短期与长期记忆
import optimize                               # 导入优化模块，用于查询改写与答案缓存
import retrieval                              # 导入检索模块，复用第 4 步的检索链路
from logger import get_logger                 # 导入日志工具，用于记录问答过程
from prompt import (                          # 从提示词模块导入
    get_system_prompt,                        # 按角色取系统提示词
    LONG_MEMORY_HEADER,                       # 历史对话记忆段标题
    LONG_MEMORY_ITEM,                         # 单条历史记忆格式
    USER_PROMPT_TEMPLATE,                     # 用户消息模板
)                                             # 导入结束
# 后处理与校验已拆到 postprocess.py，这里再导入一次，保证 rag.postprocess 等旧调用方式仍可用
from postprocess import (                     # 从后处理模块导入
    BANNED_WORDS,                             # 违规词表（供外部引用）
    MAX_ANSWER_LEN,                           # 答案长度上限（供外部引用）
    NO_EVIDENCE_TEXT,                         # 无依据话术（供外部引用）
    TRUNCATE_SUFFIX,                          # 截断提示语（供外部引用）
    postprocess,                              # 后处理函数
    validate_answer,                          # 校验函数
)                                             # 导入结束

logger = get_logger("rag")                    # 创建本模块的 logger 实例

END_MARKER = "__END__"                        # 流式结束标记，供上层识别
INTERRUPT_TEXT = "[生成中断，请重试]"          # 网络异常时输出的提示
RETRY_TIMES = 2                               # 网络异常时的重试次数


def _memory_text(long_memory: list) -> str:          # 内部函数：拼装历史对话记忆段
    """把长期记忆拼成提示词里的一段文本；为空时返回空串，该段整体不出现。"""
    if not long_memory:                              # 没有历史记忆
        return ""                                    # 返回空串，模板里不留痕迹
    lines = [LONG_MEMORY_HEADER]                     # 段标题，含优先级声明
    for idx, item in enumerate(long_memory, 1):      # 逐条编号
        lines.append(LONG_MEMORY_ITEM.format(        # 按固定格式拼一条
            tag=f"M{idx}",                           # 编号形如 M1、M2
            question=(item.get("question") or "").strip(),          # 历史问题
            answer=(item.get("answer") or "").strip()[:300],        # 历史回答截断到 300 字
        ))                                           # 单条拼装结束
    return "\n".join(lines) + "\n"                   # 段末换行，与后续分隔线隔开


def build_messages(query: str, contexts: list, history: list = None,
                   role_name: str = "", long_memory: list = None) -> list:
    """组装 messages：system + 历史对话 + user（标准条款 + 历史记忆 + 问题）。"""
    parts = []                                       # 逐条拼装上下文
    for idx, item in enumerate(contexts, 1):         # 编号从 1 开始
        block = (f"[{idx}] 来源：{item.get('source', '未知')}"    # 每条以序号与来源开头
                 f"\n命中片段：{item.get('text', '')}")            # 检索命中的子块原文
        parent = (item.get("extra") or {}).get("parent_text")   # 取父子块回溯得到的父块全文
        if parent:                                   # 有父块才追加，没有就只拼命中片段
            block += f"\n上下文扩充：{parent}"         # 接上父块全文，补足上下文
        parts.append(block)                          # 收进列表
    context_text = "\n".join(parts)                  # 拼成完整上下文文本
    if not context_text:                             # 一条上下文都没有
        context_text = "（本次检索没有找到任何相关条款）"   # 明确告诉模型没有依据
    messages = [{"role": "system", "content": get_system_prompt(role_name)}]   # 按角色取系统提示词
    limit = config.LLM_HISTORY_LIMIT                 # 历史条数上限从配置读
    recent = list(history or [])[-limit:] if limit > 0 else []   # 只取最近 limit 条
    messages += [{"role": t.get("role", "user"), "content": t.get("content", "")}
                 for t in recent]                    # 按顺序插入历史对话
    messages.append({                                # 最后一条是用户消息
        "role": "user",                              # 用户角色
        "content": USER_PROMPT_TEMPLATE.format(      # 套模板
            context=context_text,                    # 标准条款段
            long_memory=_memory_text(long_memory),   # 历史记忆段，空时不出现
            question=query),                         # 用户问题
    })                                               # 用户消息组装结束
    return messages                                  # 返回组装好的消息列表


def _client():                                       # 内部函数：创建 DeepSeek 客户端
    """创建 OpenAI 兼容客户端，密钥与地址都来自配置，代码中不写死。"""
    from openai import OpenAI                        # 延迟导入，避免未安装时影响模块加载
    if not config.LLM_API_KEY:                       # 密钥为空说明 .env 没配好
        raise RuntimeError("LLM_API_KEY 未配置，请在 .env 中填写")   # 立即报错
    return OpenAI(                                   # 创建并返回客户端
        api_key=config.LLM_API_KEY,                  # 密钥来自 .env
        base_url=config.LLM_BASE_URL,                # 服务地址来自 .env
        timeout=60,                                  # 单次请求超时 60 秒
    )                                                # 客户端创建结束


def call_deepseek(messages: list) -> dict:
    """非流式调用 DeepSeek，网络异常自动重试 2 次，返回正文与 token 用量。"""
    client = _client()                               # 创建客户端
    last_error = None                                # 记录最后一次异常
    for attempt in range(RETRY_TIMES + 1):           # 首次加上 2 次重试，共 3 次机会
        try:                                         # 尝试调用
            response = client.chat.completions.create(   # 发起对话补全请求
                model=config.LLM_MODEL,              # 模型名称来自 .env
                messages=messages,                   # 组装好的消息
                temperature=config.LLM_TEMPERATURE,  # 生成温度
                max_tokens=config.LLM_MAX_TOKENS,    # 最大生成 token 数
                stream=False,                        # 非流式
            )                                        # 请求结束
            usage = response.usage                   # 取出 token 用量
            return {                                 # 返回结果
                "content": response.choices[0].message.content or "",   # 正文
                "usage": {f: getattr(usage, f, 0) for f in
                          ("prompt_tokens", "completion_tokens", "total_tokens")},   # 三项 token 用量
            }                                        # 结果字典结束
        except Exception as exc:                     # 调用失败
            last_error = exc                         # 记下异常
            logger.warning("DeepSeek 调用失败（第 %d 次）：%s", attempt + 1, exc)   # 记录告警
            time.sleep(1)                            # 等 1 秒再重试
    raise RuntimeError(f"DeepSeek 调用连续失败：{last_error}")   # 全部失败后抛出异常


def call_deepseek_stream(messages: list, usage_sink: dict = None):
    """流式调用 DeepSeek，逐段 yield 正文，结束时 yield 结束标记，异常时 yield 提示语。"""
    try:                                             # 流式调用容易中断，做整体保护
        client = _client()                           # 创建客户端
        stream = client.chat.completions.create(     # 发起流式请求
            model=config.LLM_MODEL,                  # 模型名称
            messages=messages,                       # 消息列表
            temperature=config.LLM_TEMPERATURE,      # 生成温度
            max_tokens=config.LLM_MAX_TOKENS,        # 最大 token 数
            stream=True,                             # 开启流式
            stream_options={"include_usage": True},  # 要求最后一段带上用量
        )                                            # 请求结束
        for chunk in stream:                         # 逐段接收
            if getattr(chunk, "usage", None) and usage_sink is not None:   # 最后一段带用量
                for field in ("prompt_tokens", "completion_tokens", "total_tokens"):   # 三项用量
                    usage_sink[field] = getattr(chunk.usage, field, 0)   # 写进调用方传入的字典
            if not chunk.choices:                    # 有些段没有选项（只带用量）
                continue                             # 跳过
            delta = chunk.choices[0].delta           # 取出增量
            piece = getattr(delta, "content", None)  # 取出正文增量
            if piece:                                # 跳过空增量
                yield piece                          # 把这一小段交给上层
    except Exception as exc:                         # 网络或服务异常
        logger.warning("DeepSeek 流式调用中断：%s", exc)   # 记录告警
        yield INTERRUPT_TEXT                         # 输出中断提示
    yield END_MARKER                                 # 无论成功失败都给出结束标记


def _sources(contexts: list) -> list:                # 内部函数：整理来源列表
    """把检索结果整理成来源清单，每条包含文件名、片段摘要与检索分数。"""
    return [{"source": c.get("source", ""),          # 来源文件名
             "summary": (c.get("text", "") or "")[:80],   # 片段摘要，截前 80 字
             "score": round(float(c.get("score", 0.0)), 4)}   # 检索分数，保留 4 位
            for c in contexts]                       # 遍历检索结果


def collect_long_memory(query: str, user_id: int) -> list:
    """取该用户的长期记忆并按问题去重（同问题只留分数最高的一条）。"""
    memories = memory.search_long_memory(user_id, query, config.MEMORY_TOP_K)   # 取历史问答
    best = {}                                        # 问题文本到最佳记忆的映射
    for item in sorted(memories, key=lambda m: m.get("score", 0.0), reverse=True):   # 按分数降序
        question = (item.get("question") or "").strip()   # 取出历史问题
        if question and question not in best:        # 同问题只保留分数最高的
            best[question] = item                    # 记下该条
    deduped = list(best.values())                    # 去重后的记忆列表
    if memories:                                     # 有记忆时记录去重前后条数
        logger.info("长期记忆：召回 %d 条 → 去重后 %d 条", len(memories), len(deduped))
    return deduped                                   # 返回去重结果


def save_turn(user_id: int, conversation_id: int, query: str, answer: str) -> None:
    """把本轮问答写入 MySQL、Redis、Milvus 三处；每组失败不影响其他组，不抛异常。"""
    # 注意：不要把「写 user」和「写 assistant」用 `or` 串成一行的 lambda。
    # save_message 返回 lastrowid、push_short_memory 返回列表长度，都是真值，
    # 而 `A() or B()` 在 A 为真时短路，B 永远不会执行——第 6 步踩过这个坑，
    # 导致助手回答全部丢失。这里改成顺序语句，保证两次写入都执行。
    def _write_messages() -> None:                # 第 1 组：会话消息落库（MySQL）
        """用户提问与助手回答各写一条，两条都要执行。"""
        db_user.save_message(conversation_id, "user", query)          # 写用户提问
        db_user.save_message(conversation_id, "assistant", answer)    # 写助手回答

    def _write_short_memory() -> None:            # 第 2 组：短期记忆（Redis）
        """用户提问与助手回答各推一条，两条都要执行。"""
        memory.push_short_memory(user_id, "user", query)          # 推用户提问
        memory.push_short_memory(user_id, "assistant", answer)    # 推助手回答

    def _write_long_memory() -> None:             # 第 3 组：长期记忆（Milvus）
        """整轮问答作为一个记忆单元存入向量库，便于以后按语义召回。"""
        memory.save_long_memory(user_id, query, answer, source=str(conversation_id))   # 存整轮

    writes = [                                    # 三组写入动作，逐组执行
        ("会话消息落库", _write_messages),         # MySQL
        ("短期记忆", _write_short_memory),         # Redis
        ("长期记忆", _write_long_memory),          # Milvus
    ]                                             # 写入动作列表结束
    for label, action in writes:                  # 逐组执行
        try:                                      # 单组失败不影响其他两组
            action()                              # 执行该组写入
        except Exception as exc:                  # 该组写入失败
            logger.warning("%s写入失败：%s", label, exc)   # 记录告警，不中断其余组


def _role_name_of(role_id: int) -> str:              # 内部函数：由角色编号反查角色名
    """按角色编号从 roles 表反查角色名，查不到时返回空串（走默认提示词）。"""
    try:                                             # 查库失败不影响主流程
        roles = {r["role_id"]: r["role_name"] for r in db_user.list_roles()}   # 编号到角色名的映射
        return roles.get(role_id, "")                # 取角色名，查不到返回空串
    except Exception as exc:                         # 查库失败
        logger.warning("角色名反查失败，使用默认提示词：%s", exc)   # 记录降级原因
        return ""                                    # 返回空串


def ask(query: str, user_id: int, role_id: int, conversation_id: int,
        top_k: int = None, stream: bool = False):
    """问答总入口：取记忆 → 改写 → 查缓存 → 检索 → 生成 → 写回；命中缓存跳过生成但仍写回。"""
    history = memory.get_short_memory(user_id)       # 第一步：取该用户的短期记忆
    rewritten = optimize.rewrite_query(query, history) if history else query   # 第二步：查询改写
    role_name = _role_name_of(role_id)               # 第三步：反查角色名，决定用哪份提示词
    key = optimize.cache_key(user_id, role_name, query)   # 第四步：按「用户+角色+归一化原问题」建键
    if not stream:                                   # 非流式才查缓存（流式不做半截缓存）
        cached = optimize.cache_get(key)             # 尝试读缓存
        if cached:                                   # 命中
            # 缓存只该跳过 LLM 调用，不该跳过副作用：写库与写记忆必须照做，
            # 否则 MySQL 缺这一轮、Redis 短期记忆缺这一轮，多轮对话就断片了。
            # save_turn 内部已含三组写入（MySQL 消息、Redis 短期记忆、Milvus 长期记忆），
            # 所以这里只调它一次，不要再单独调 save_long_memory，否则长期记忆会重复写一条。
            save_turn(user_id, conversation_id, query, cached["answer"])   # 与未命中路径完全一致
            logger.info("缓存命中：命中键=%s，仍写入 MySQL/Redis/Milvus", key)   # 记录日志
            cached["cache_hit"] = True               # 标记命中
            return cached                            # 直接返回缓存结果
    contexts = retrieval.retrieve(rewritten, top_k)  # 第五步：只把检索结果送去精排
    long_memory = collect_long_memory(rewritten, user_id)   # 第六步：长期记忆单独取，不进精排
    messages = build_messages(rewritten, contexts, history, role_name, long_memory)   # 第七步：组装提示词
    history_used = len(messages) - 2                 # 实际带入的历史条数
    long_used = len(long_memory)                     # 去重后的长期记忆条数
    logger.info("问答开始：角色 %s，改写后问题 %s，上下文 %d 条，历史 %d 条，长期记忆 %d 条",
                role_name or "默认", rewritten[:40], len(contexts), history_used, long_used)   # 日志
    if stream:                                       # 流式分支
        return _ask_stream(messages, contexts, query, user_id, conversation_id, long_used)   # 生成器
    result = call_deepseek(messages)                 # 第八步：非流式调用
    answer = postprocess(result["content"])          # 第九步：后处理
    check = validate_answer(answer)                  # 第十步：校验
    save_turn(user_id, conversation_id, query, answer)   # 第十一步：写回三处记忆
    logger.info("问答完成：校验 %s（%s），token %d",
                check["ok"], check["reason"], result["usage"]["total_tokens"])   # 记录日志
    payload = {                                      # 组装返回结果
        "answer": answer,                            # 后处理后的答案
        "raw": result["content"],                    # 后处理前的模型原文，供对比查看
        "sources": _sources(contexts),               # 来源清单
        "contexts": contexts,                        # 原始检索结果
        "usage": result["usage"],                    # token 用量
        "check": check,                              # 校验结果
        "conversation_id": conversation_id,          # 本轮所属会话
        "history_used": history_used,                # 实际带入的历史条数
        "long_memory_used": long_used,               # 实际带入的长期记忆条数
        "rewritten_query": rewritten,                # 改写后的检索问题
        "cache_hit": False,                          # 本次未命中缓存
    }                                                # 结果字典结束
    optimize.cache_set(key, payload)                 # 写入缓存，供后续相同问题复用
    return payload                                   # 返回结果


def _ask_stream(messages: list, contexts: list, query: str, user_id: int,
                conversation_id: int, long_used: int):   # 内部函数：流式问答生成器
    """流式问答：逐段产出正文，结束时产出带 sources 与 usage 的 JSON 字符串。"""
    usage = {}                                       # 用于接收 token 用量
    buffer = ""                                      # 只缓冲尾部，用于跨段清理行尾空格
    answer_parts = []                                # 收集完整答案，结束时写回记忆
    for piece in call_deepseek_stream(messages, usage):   # 逐段接收
        if piece == END_MARKER:                      # 收到结束标记
            if buffer:                               # 还有没吐出的尾巴
                tail = postprocess(buffer)           # 清理尾部
                answer_parts.append(tail)            # 计入完整答案
                yield tail                           # 吐出尾巴
            break                                    # 跳出循环
        buffer += piece                              # 累积到缓冲区
        cut = buffer.rfind("\n")                     # 找最后一个换行位置
        if cut > 0:                                  # 找到完整行
            ready, buffer = buffer[:cut + 1], buffer[cut + 1:]   # 完整行先输出，尾巴留下
            cleaned = postprocess(ready)             # 对这一批做轻量后处理
            if cleaned:                              # 有内容才输出
                answer_parts.append(cleaned)         # 计入完整答案
                yield cleaned                        # 吐出一批
    answer = "".join(answer_parts)                   # 拼出完整答案
    save_turn(user_id, conversation_id, query, answer)   # 写回三处记忆
    tail = {"sources": _sources(contexts), "usage": usage,   # 结束信息：来源与用量
            "conversation_id": conversation_id, "long_memory_used": long_used}   # 会话与记忆条数
    yield json.dumps(tail, ensure_ascii=False)       # 结束产出


def main() -> None:                                  # 命令行入口
    """命令行入口：委托给 cli 模块，保持 python -m rag 可用。"""
    import cli                                       # 延迟导入，避免模块加载时循环依赖
    cli.main()                                       # 执行命令行演示


if __name__ == "__main__":                           # 支持 python -m rag 直接运行
    main()                                           # 执行命令行入口
