"""多轮 query 改写（指代消解）+ Redis 缓存。

职责
----
把用户在多轮对话中的最新问题改写成独立完整的问题（指代消解，如「它有什么副作用」
→「硝苯地平有什么副作用」），提升后续检索的命中率；改写结果按 session_id + query
缓存到 Redis，避免相同 query 反复调用 LLM。

依赖
----
- rag：复用 DeepSeek 客户端（_get_client）与模型名（DEEPSEEK_MODEL）、Redis 客户端（_get_redis）
  ——均延迟 import，避免循环 import（rag 又 import 了本模块的 get_rewritten_query）

输入输出契约
------------
- get_rewritten_query(query, history, session_id) -> str（改写后 query 或原样 query）
- rewrite_query(query, history) -> str（无缓存的纯改写）

降级策略：无历史（第一轮）直接返回原 query；改写失败 / 改写返回空 / Redis 读写异常
都降级为原 query 或跳过缓存，绝不抛错，保证主链路（检索）不被改写环节拖垮。
"""
import hashlib
import logging

logger = logging.getLogger(__name__)

REWRITE_HISTORY_ROUNDS = 3   # 改写时最多参考最近 3 轮历史
REWRITE_CACHE_TTL = 3600     # 改写结果缓存 1 小时
REWRITE_MAX_TOKENS = 200     # 改写输出上限（一个完整问题，200 token 足够）

# 改写 prompt：让模型结合历史做指代消解，把最新问题补全成独立完整的问题。
# 占位符 {history} / {query} 在 rewrite_query 里用 .format 填充。
# 关键约束「只输出改写后的问题，不要解释」避免模型额外输出干扰下游直接使用。
REWRITE_PROMPT = (
    "根据以下对话历史，把用户的最新问题改写成独立完整的问题（指代消解）。\n"
    "如果最新问题本身已经完整，原样返回。\n"
    "只输出改写后的问题，不要解释。\n"
    "对话历史：\n"
    "{history}\n"
    "最新问题：{query}\n"
    "改写后："
)


def _client():
    """延迟 import 复用 rag 的 DeepSeek 客户端与模型名，避免循环 import。

    Returns:
        (OpenAI 客户端, 模型名) 元组。

    Raises:
        未配置 API key 时由 rag._get_client 抛 RuntimeError（向上传播）。
    """
    from rag import DEEPSEEK_MODEL, _get_client

    return _get_client(), DEEPSEEK_MODEL


def _redis():
    """延迟 import 复用 rag 的 Redis 客户端。

    Returns:
        redis.Redis 客户端实例。
    """
    from rag import _get_redis

    return _get_redis()


def _format_history(history: list[dict]) -> str:
    """取最近 REWRITE_HISTORY_ROUNDS 轮，拼成「用户：… / 助手：…」文本。

    Args:
        history: 历史消息列表（[{role, content}, ...]，旧→新）。

    Returns:
        多行文本，每行形如「用户：xxx」或「助手：xxx」。

    Raises:
        无。

    只取末尾 3 轮（= 6 条），既给足指代消解所需上下文，又不让 prompt 过长、省 token。
    """
    lines = []
    for turn in history[-REWRITE_HISTORY_ROUNDS * 2:]:  # 3 轮 = user/assistant 各 3 = 6 条
        role = "用户" if turn["role"] == "user" else "助手"
        lines.append(f"{role}：{turn['content']}")
    return "\n".join(lines)


def rewrite_query(query: str, history: list[dict]) -> str:
    """把 query 改写成独立完整的问题；无历史或失败时降级返回原 query。

    Args:
        query: 用户最新问题。
        history: 历史消息列表（旧→新）。

    Returns:
        改写后的问题；无历史 / 调用失败 / 返回空时降级为原 query。

    Raises:
        无（LLM 调用异常内部捕获并降级）。
    """
    if not history:
        return query  # 第一轮无历史，无需改写，直接返回
    prompt = REWRITE_PROMPT.format(history=_format_history(history), query=query)
    try:
        client, model = _client()
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,  # 改写是确定性任务，温度 0 保证结果稳定可复现
            max_tokens=REWRITE_MAX_TOKENS,
        )
        rewritten = (resp.choices[0].message.content or "").strip()
    except Exception as e:  # noqa: BLE001
        # 降级路径 1：改写调用失败 -> 用原始 query，不让改写拖垮检索
        logger.warning("query 改写失败，降级用原始 query: %s", e)
        return query
    if not rewritten:
        # 降级路径 2：模型返回空 -> 用原始 query
        logger.warning("query 改写返回空，降级用原始 query")
        return query
    logger.info("query 改写: %r -> %r", query, rewritten)
    return rewritten


def _rewrite_key(session_id: str, query: str) -> str:
    """生成改写缓存 key：rewrite:{session_id}:{md5(query)}。

    Args:
        session_id: 会话唯一标识。
        query: 用户问题。

    Returns:
        Redis key 字符串。

    Raises:
        无。

    用 md5(query) 而非原 query 作 key：query 可能很长且含空格/中文/特殊字符，
    md5 摘要固定长度、无非法字符，可直接当 Redis key。
    """
    digest = hashlib.md5(query.encode("utf-8")).hexdigest()
    return f"rewrite:{session_id}:{digest}"


def get_rewritten_query(query: str, history: list[dict], session_id: str) -> str:
    """带缓存的改写入口：先查 Redis 缓存，未命中再改写并写回；Redis 异常不影响主流程。

    Args:
        query: 用户最新问题。
        history: 历史消息列表（旧→新）。
        session_id: 会话唯一标识（缓存 key 的一部分）。

    Returns:
        改写后的问题（含缓存命中的历史结果）或原 query。

    Raises:
        无（Redis 读写异常内部捕获，不中断）。

    缓存策略：按 session_id + query 缓存，同一会话内相同 query 直接复用改写结果，
    省一次 LLM 调用；TTL 1 小时让缓存随历史推进自然过期（历史变化后同 query 的
    理想改写可能不同，1 小时限制其影响，属「省调用」与「改写真确」的折中）。
    """
    if not history:
        return query  # 第一轮无历史，不改写也不缓存
    key = _rewrite_key(session_id, query)
    try:
        cached = _redis().get(key)
        if cached:
            logger.info("改写缓存命中 %s", key)
            return cached
    except Exception as e:  # noqa: BLE001
        # 降级路径 3：Redis 读失败 -> 跳过缓存，直接改写
        logger.warning("改写缓存读取失败，跳过缓存: %s", e)
    rewritten = rewrite_query(query, history)
    try:
        _redis().set(key, rewritten, ex=REWRITE_CACHE_TTL)
    except Exception as e:  # noqa: BLE001
        # 降级路径 4：Redis 写失败 -> 仅告警，仍返回改写结果（下次未命中会重新改写）
        logger.warning("改写缓存写入失败: %s", e)
    return rewritten
"""
查询改写模块的核心目标是做指代消解，把用户带有省略、指代的问题补全成独立完整问句，供后续检索使用。第一部分，入口函数`get_rewritten_query`调用`_rewrite_key`，使用会话 ID 加上问题 md5 摘要生成 Redis 缓存 key。之所以这样设计，是因为用户问题可能很长，还包含中文与特殊符号，直接用作 Redis 的 key 会出现超长、非法字符等问题；带上 session_id，是为了区分不同会话，避免不同会话里相同文字的问题复用错误缓存。生成 key 之后优先查询 Redis 缓存，如果命中缓存，就直接返回已经改写好的问题，目的是减少重复调用大模型，节省 token 开销并降低响应延迟。如果读取 Redis 时报错，则跳过缓存逻辑，直接走 LLM 改写流程；缓存只是性能优化手段，不是核心能力，如果不做异常捕获，一旦 Redis 服务异常，整个查询改写流程就会中断，进而导致问答主流程失败。第二部分，缓存未命中或者读缓存异常时，就调用`rewrite_query`执行 LLM 改写。`rewrite_query`内部调用`_format_history`，只截取最近 3 轮对话历史拼入提示词。这样做是为了控制输入 token 数量，降低成本和延迟，过多的历史还可能引入无关上下文干扰模型判断；如果不截断历史，输入文本会持续膨胀，不仅开销上涨，还可能触发模型输入超限。代码使用`_client()`延迟加载 LLM 客户端，延迟导入是为了解决模块之间循环导入的问题，只有真正调用模型的时候才导入 rag 模块中的客户端，若在文件头部直接导入，会造成程序启动时报错。提示词里明确要求模型只输出改写后的问题，不增加额外解释，同时 temperature 设置为 0。下游检索会直接使用模型输出文本作为检索 query，如果模型附带多余解释文字，检索输入就会被污染，召回无关文档；temperature 等于 0 可以保证同样输入得到稳定的改写结果，适合这种确定性任务，温度过高会让结果随机波动，缓存也失去意义。代码捕获 LLM 调用异常以及模型返回空内容的场景，一旦出现这类情况就降级返回原始用户 query。查询改写属于增强优化功能，不是问答必须的步骤，如果不做降级处理，大模型接口超时或者报错时，整个问答链路就直接中断。另外，当不存在对话历史时直接返回原问题，第一轮提问没有上文，不存在指代需要消解，无需调用大模型，避免浪费 token。第三部分，拿到改写结果后，尝试写入 Redis 缓存并设置一小时的过期时间。设置过期时间是一种折中方案，缓存可以在一小时内复用，节约 LLM 调用；但对话上下文会随时间持续变化，长时间保留缓存，同一个问题在新的对话环境下会读到过时的改写结果，造成指代理解错误。写入缓存时如果发生 Redis 异常，只打印告警日志，仍然返回改写好的 query。写缓存失败只会影响后续请求，不影响当前请求结果，如果写缓存报错直接抛出异常，当前这次问答请求就会失败。整体设计思想是，查询改写只是增强功能而非基础刚需，能成功改写就提升检索准确性；改写链路中任何环节出错，都要降级回原始 query，保障问答主流程不会崩溃，同时依靠缓存和历史截断，控制 token 消耗和接口响应延迟。
"""