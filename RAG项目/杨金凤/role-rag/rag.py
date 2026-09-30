"""角色提示词 + 调用 DeepSeek 生成 + 多轮会话记忆（Redis）。

流程：
接收提问 → 急症校验 → 读取会话历史 → 改写查询 → 检索知识库 → 拼接人设、资料、历史送入大模型 → 清洗结果 + 追加免责声明 → 返回答案 → 对话存入 Redis → 定期异步抽取长期记忆

职责
----
把「角色人设 + 检索到的资料 + 多轮历史」拼成 prompt，调用 DeepSeek 生成回答，
并把每轮问答写回 Redis 做多轮记忆；对外提供 ask()（非流式）与 ask_stream()（流式 SSE）两个入口。

依赖
----
- retrieval：混合检索（retrieve / COLLECTION / load_embedder / load_reranker / _rrf_fuse / _sigmoid，
  经 rag.* 转发供 api.py / tests 使用）
- roles：角色表（get_role / first_role），按 role 解析 persona 与 collection_name
- query_rewrite：Query 改写（get_rewritten_query）
- memory：长期记忆（recall_facts / schedule_extract）
- utils：急症检测 check_emergency、免责声明 add_disclaimer、回答后处理 postprocess
- redis：多轮会话记忆存储（可用性不足时降级为无记忆，不报错）
- openai.OpenAI：DeepSeek 兼容接口

输入输出契约
------------
- ask(query, session_id, role, user_id) -> {"answer": str, "sources": list[dict]}
- ask_stream(...) -> 生成器，逐段 yield SSE 文本（event: sources / data: token / data: [DONE]）
"""
import json
import logging
import os
import sys
import time

import redis
from dotenv import load_dotenv
from openai import OpenAI

from retrieval import (  # noqa: F401  供 api.py / tests 经 rag.* 访问
    COLLECTION,
    _rrf_fuse,
    _sigmoid,
    load_embedder,
    load_reranker,
    retrieve,
)

import roles  # noqa: E402
from query_rewrite import get_rewritten_query  # noqa: E402
import memory  # noqa: E402
from utils import check_emergency, add_disclaimer, postprocess  # noqa: E402

load_dotenv()

logger = logging.getLogger(__name__)

DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")

# Redis 会话记忆配置
REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_MAX_ROUNDS = int(os.getenv("REDIS_MAX_ROUNDS", "10"))  # 每个会话最多保留的轮数
REDIS_TTL = int(os.getenv("REDIS_TTL", "86400"))  # 会话记忆过期时间（秒），默认 1 天

# 高血压医生人设（用户提供）。共 9 条回答要求：
# 1 只依知识库不编造 / 2 结论-原因-注意事项结构 / 3 用药剂量诊断转诊提醒线下就医 /
# 4 不替代面诊不开处方 / 5 语气耐心不恐吓不淡化 / 6 无相关内容明说"指南中未提及" /
# 7 引用尽量给页码 / 8 中文简洁必要时比喻 / 9 Markdown 表格格式规范。
DOCTOR_PERSONA = """你是一位严谨、温和的高血压专科医生，擅长用通俗语言向普通患者解释病情。你熟悉《国家基层高血压防治管理指南》，回答时严格依据指南内容。

回答要求：
1. 只根据提供的知识库内容回答，不编造、不臆测。
2. 回答结构：先给结论，再解释原因，最后给注意事项。
3. 涉及用药、剂量、诊断、转诊标准，必须提醒用户线下就医。
4. 不替代医生面诊，不开具处方，不给个体化用药方案。
5. 语气耐心、专业，避免恐吓性表达，也不淡化风险。
6. 如果知识库中没有相关内容，明确说"指南中未提及"，不要凭空回答。
7. 引用建议时尽量说明来源页码，便于用户溯源。
8. 回答用中文，简洁清晰，避免过多专业术语，必要时用比喻解释。
9. 输出 Markdown 表格时，严格按以下格式（每一行独立，行与行之间用换行符分隔）：

| 列1 | 列2 |
|---|---|
| 值1 | 值2 |
| 值3 | 值4 |

不要把表格写在一行里。"""

_redis_client = None


def _get_redis():
    """懒加载 Redis 客户端（进程内单例）；短超时，保证 Redis 挂了不卡请求。

    Returns:
        redis.Redis 客户端实例（decode_responses=True，读写均为 str）。

    Raises:
        不主动连接（懒连接），故此处不抛连接异常；连接异常在 get_history / save_turn 里被捕获。
    """
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=True,
            socket_connect_timeout=2,  # 连接超时 2s
            socket_timeout=2,  # 读写超时 2s
        )
    return _redis_client


def _get_client() -> OpenAI:
    """构造 DeepSeek 的 OpenAI 兼容客户端。

    Returns:
        OpenAI 实例（base_url 指向 DeepSeek）。

    Raises:
        RuntimeError: 未配置 DEEPSEEK_API_KEY1 时抛出（启动期配置错误，不降级）。
    """
    api_key = os.getenv("DEEPSEEK_API_KEY1")
    if not api_key:
        raise RuntimeError("未配置 DEEPSEEK_API_KEY1，请在 .env 中填写")
    return OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)


def _resolve_role(role: str | None) -> tuple[str, str, str]:
    """解析 (role_name, persona, collection_name)。

    Args:
        role: 角色名；为 None 时取第一个角色；明确传入但不存在时回退默认医生。

    Returns:
        (role_name, persona, collection_name) 三元组。

    Raises:
        无。

    api.py 已对「明确传入但不存在」的 role 返回 404，故此处未命中与 None 同回退，
    统一落到 default / DOCTOR_PERSONA / COLLECTION。
    """
    info = roles.get_role(role) if role else roles.first_role()
    if info is not None:
        return info["name"], info["persona"], info["collection_name"]
    return "default", DOCTOR_PERSONA, COLLECTION


def get_history(session_id: str, role: str = "default") -> list[dict]:
    """从 Redis 读取该会话该角色的最近对话，返回 [{role, content}, ...]（旧→新）。

    Args:
        session_id: 会话唯一标识。
        role: 角色名（同一 session 下不同角色各自独立记忆）。

    Returns:
        消息列表，格式为 OpenAI messages 的 user/assistant 交替序列；
        Redis 不可用或读取异常时降级返回 []（无记忆，不影响主流程）。

    Raises:
        无（内部捕获所有异常）。
    """
    key = f"session:{session_id}:{role}:messages"
    try:
        raw = _get_redis().lrange(key, 0, -1)  # LPUSH 存的，列表头是最新，取全量后需反转
        history = []
        for item in reversed(raw):  # 新→旧 反转为 旧→新，符合对话时间顺序
            turn = json.loads(item)
            history.append({"role": "user", "content": turn["user"]})
            history.append({"role": "assistant", "content": turn["assistant"]})
        logger.info("Redis 读取 %s 历史 %d 轮", key, len(history) // 2)
        return history
    except Exception as e:  # noqa: BLE001
        logger.warning("Redis 不可用，降级为无记忆: %s", e)
        return []


def save_turn(session_id: str, role: str, user_msg: str, assistant_msg: str) -> None:
    """把本轮问答写入 Redis（LPUSH + LTRIM + EXPIRE），维护最近 REDIS_MAX_ROUNDS 轮。

    Args:
        session_id: 会话唯一标识。
        role: 角色名。
        user_msg: 本轮用户问题（原始 query）。
        assistant_msg: 本轮助手回答（已后处理 + 加免责声明）。

    Returns:
        None。

    Raises:
        无（失败仅告警，不中断主流程）。

    Redis 操作：
        1) LPUSH 把本轮（JSON 串）压到列表头，最新在最前；
        2) LTRIM 只保留前 REDIS_MAX_ROUNDS 项，超出自动丢弃最旧；
        3) EXPIRE 设置 TTL，过期自动清理，避免无限累积。
    三步用 pipeline 打包成一次往返，保证原子性并减少网络开销。
    """
    key = f"session:{session_id}:{role}:messages"
    try:
        pipe = _get_redis().pipeline()
        pipe.lpush(
            key,
            json.dumps({"user": user_msg, "assistant": assistant_msg}, ensure_ascii=False),
        )
        pipe.ltrim(key, 0, REDIS_MAX_ROUNDS - 1)
        pipe.expire(key, REDIS_TTL)
        pipe.execute()
        logger.info("Redis 写入 %s 本轮问答", key)
    except Exception as e:  # noqa: BLE001
        logger.warning("Redis 写入失败，跳过记忆: %s", e)


def _maybe_extract_memory(mem_key: str, history: list[dict], query: str, answer: str) -> None:
    """每 5 轮触发一次长期记忆抽取（取最近 5 轮），异步调度，不阻塞主流程。

    Args:
        mem_key: 记忆归属键（user_id 或 session_id）。
        history: 本轮之前的对话历史（不含本轮）。
        query: 本轮用户问题。
        answer: 本轮助手回答。

    Returns:
        None。

    Raises:
        无。

    触发条件：写入本轮后总轮数（len(history)//2 + 1）能被 5 整除；
    抽取内容为「历史 + 本轮」的末尾 10 条（即最近 5 轮）交给 memory.schedule_extract 异步处理。
    """
    rounds = len(history) // 2 + 1  # 本轮写入后的总轮数
    if rounds % 5 != 0:
        return
    recent = (history + [
        {"role": "user", "content": query},
        {"role": "assistant", "content": answer},
    ])[-10:]  # 最近 5 轮 = 末尾 10 条（user/assistant 各 5）
    memory.schedule_extract(mem_key, recent)


def build_messages(
    query: str,
    contexts: list[dict],
    history: list[dict] | None = None,
    persona: str | None = None,
    facts: list[str] | None = None,
) -> list[dict]:
    """组装 OpenAI messages：system(人设 + 用户事实 + 参考资料) + 历史 + user(问题)。

    Args:
        query: 用户问题。
        contexts: 检索到的资料片段列表（每项含 content/page）。
        history: 历史消息（旧→新），None 或空则不加历史。
        persona: 角色人设，None 时用 DOCTOR_PERSONA。
        facts: 长期记忆召回的用户事实列表，空/None 则不加该段。

    Returns:
        可直接传给 chat.completions.create 的 messages 列表。

    Raises:
        无。

    拼装顺序（system 内部三段拼接）：
        1) 人设 persona（含 9 条回答要求）；
        2) 【关于用户的已知事实】（若有 facts）——让模型带上用户画像；
        3) 【参考资料】编号列出检索片段（含页码）——RAG 的知识来源；
      然后依次追加历史（多轮上下文）、最后是当前 user 问题。
    """
    persona = persona or DOCTOR_PERSONA
    # 把每条资料标成 [资料N｜第X页] 形式，方便模型引用页码、便于用户溯源
    context_text = "\n\n".join(
        f"[资料{i + 1}｜第{c['page']}页]\n{c['content']}"
        for i, c in enumerate(contexts)
    )
    system = persona
    if facts:
        fact_text = "\n".join(f"- {f}" for f in facts)
        system += f"\n\n【关于用户的已知事实】\n{fact_text}"
    system += f"\n\n【参考资料】\n{context_text}"
    messages = [{"role": "system", "content": system}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": query})
    return messages


def ask(
    query: str,
    # 多用户场景必须让客户端传唯一 session_id，否则所有请求共享 default 会话，会串记忆。
    session_id: str = "default",
    role: str | None = None,
    user_id: str | None = None,
) -> dict:
    """非流式完整链路：急症检测 → 角色解析 → 读历史 → Query 改写 → 检索 → 拼 prompt → 生成 → 后处理 → 写记忆。

    Args:
        query: 用户问题。
        session_id: 会话唯一标识（多用户必须由客户端传入，默认 "default"）。
        role: 角色名；None 取第一个角色。
        user_id: 长期记忆归属键；None 时回退为 session_id。

    Returns:
        {"answer": str, "sources": list[dict]}；急症命中时 answer 为提示语、sources 为空。

    Raises:
        DeepSeek 调用异常会向上传播（非流式不降级，由上层 api.py 处理）；
        检索/Redis 内部各自降级。

    与 ask_stream 的差异：整段等待生成完成、一次性返回 answer 字符串，
    不逐 token 输出；急症/错误不 yield SSE，而是直接返回 dict。
    """
    em = check_emergency(query)
    if em:
        # 急症检测命中：直接返回提示，跳过检索与生成
        return {"answer": em, "sources": []}
    role_name, persona, collection_name = _resolve_role(role)
    history = get_history(session_id, role_name)
    search_query = get_rewritten_query(query, history, session_id)  # 结合历史改写 query，提升检索命中
    sources = retrieve(search_query, collection_name)
    mem_key = user_id or session_id
    facts = memory.recall_facts(mem_key, query)  # 长期记忆召回（降级 []）
    messages = build_messages(query, sources, history, persona, facts)

    start = time.perf_counter()
    resp = _get_client().chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=messages,
        temperature=0.7,
    )
    elapsed = time.perf_counter() - start
    logger.info("DeepSeek 调用耗时 %.2fs", elapsed)

    answer = resp.choices[0].message.content
    answer = add_disclaimer(postprocess(answer))  # 先清洗格式，再统一追加免责声明
    save_turn(session_id, role_name, query, answer)  # 写回本轮到 Redis 多轮记忆
    _maybe_extract_memory(mem_key, history, query, answer)  # 每 5 轮异步抽长期记忆
    return {"answer": answer, "sources": sources}


def ask_stream(
    query: str,
    # 多用户场景必须让客户端传唯一 session_id，否则所有请求共享 default 会话，会串记忆。
    session_id: str = "default",
    role: str | None = None,
    user_id: str | None = None,
):
    """流式完整链路：急症检测 → 角色解析 → 读历史 → Query 改写 → 检索 → 拼 prompt → 逐 token 生成 → 后处理 → 写记忆。

    Args:
        query: 用户问题。
        session_id: 会话唯一标识（多用户必须由客户端传入，默认 "default"）。
        role: 角色名；None 取第一个角色。
        user_id: 长期记忆归属键；None 时回退为 session_id。

    Returns:
        生成器，逐段 yield SSE 文本：
        - 先 yield "event: sources\\ndata: <检索结果 JSON>\\n\\n"（把 sources 先发给前端）；
        - 生成期间逐 token yield "data: <token>\\n\\n"（token 按换行拆成多行，每行一个 data 前缀）；
        - 结束 yield "data: [DONE]\\n\\n"；
        - 流中途异常 yield "event: error\\ndata: generation interrupted\\n\\n" 后接 [DONE]。

    Raises:
        不向外抛（生成器内异常被捕获，转为 error 事件后正常结束）。

    与 ask 的差异：
        1) 生成前先 yield sources 事件，让前端先拿到来源；
        2) 逐 token 边生成边输出，而非整段等待；
        3) 异常时降级为 error 事件、不写 Redis（未完成回答不记入历史），非流式则向上抛。
    """
    em = check_emergency(query)
    if em:
        # 急症检测命中：直接以 SSE 形式返回提示语
        yield f"data: {em}\n\n"
        return
    start_all = time.perf_counter()
    role_name, persona, collection_name = _resolve_role(role)
    history = get_history(session_id, role_name)
    search_query = get_rewritten_query(query, history, session_id)  # 结合历史改写 query
    sources = retrieve(search_query, collection_name)
    mem_key = user_id or session_id
    facts = memory.recall_facts(mem_key, query)  # 长期记忆召回（降级 []）
    messages = build_messages(query, sources, history, persona, facts)

    # 生成前先发 sources，供前端展示引用来源
    yield f"event: sources\ndata: {json.dumps(sources, ensure_ascii=False)}\n\n"

    answer_parts = []
    first_token_elapsed = None  # 首 token 耗时，用于观测流式响应速度
    try:
        gen_start = time.perf_counter()
        resp = _get_client().chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=messages,
            temperature=0.7,
            stream=True,  # 流式开关
        )
        for chunk in resp:
            if not chunk.choices:
                continue  # 偶发无 choices 的 chunk，跳过
            token = chunk.choices[0].delta.content
            if not token:
                continue  # 首个 chunk 可能只有 role 无 content，跳过
            if first_token_elapsed is None:
                first_token_elapsed = time.perf_counter() - gen_start
            answer_parts.append(token)
            # token 内可能含换行，按 SSE 规范逐行加 data: 前缀（每一行独立一个 data 字段）
            for line in token.rstrip("\n").split("\n"):
                yield f"data: {line}\n"
            yield "\n"
    except Exception as e:  # noqa: BLE001
        # 流中途异常：不写 Redis（回答不完整），转为 error 事件后正常收尾
        logger.warning("流式生成中断，不写记忆: %s", e)
        yield "event: error\ndata: generation interrupted\n\n"
        yield "data: [DONE]\n\n"
        return

    answer = "".join(answer_parts)
    answer = add_disclaimer(postprocess(answer))  # 生成完成后统一后处理 + 免责声明
    save_turn(session_id, role_name, query, answer)  # 写回本轮到 Redis 多轮记忆
    _maybe_extract_memory(mem_key, history, query, answer)  # 每 5 轮异步抽长期记忆
    total_elapsed = time.perf_counter() - start_all
    logger.info(
        "流式首 token %.2fs，总耗时 %.2fs",
        first_token_elapsed if first_token_elapsed is not None else -1.0,
        total_elapsed,
    )
    yield "data: [DONE]\n\n"


if __name__ == "__main__":
    # 命令行快速自测：python rag.py "问题"，打印回答与来源
    q = sys.argv[1] if len(sys.argv) > 1 else "血压多少算高？"
    result = ask(q)
    print("回答：", result["answer"])
    print("\n来源：")
    for s in result["sources"]:
        print(f"  - 第{s['page']}页（相似度 {s['similarity']}）：{s['content'][:60]}...")
"""
这个RAG模块收到用户提问之后，首先会执行急症检测，如果识别出属于急症情况，就直接返回急救提示，不再继续后面所有流程。如果不是急症，就解析传入的角色参数，找到对应的角色人设和向量知识库集合，角色不存在或者没传角色时，自动回退到高血压医生的默认人设。接着用会话ID和角色名从Redis读取本次会话的历史对话，Redis读取失败或者不可用时，就直接使用空的对话历史，不会阻断请求；读到的原始存储数据需要调整顺序，变成从早到晚的聊天时序，方便后续传给大模型。拿到历史对话之后，系统结合历史内容对用户原始问题做查询改写，生成更适合检索的查询文本，再拿着改写后的查询调用检索模块，从指定知识库里面召回相关参考片段。之后基于用户ID或者会话ID召回长期记忆里保存的用户事实，长期记忆召回失败同样会降级为空，不影响主流程。接下来把角色人设、召回到的用户事实、带页码标注的检索参考资料拼接成系统提示词，再追加Redis读出的历史对话，最后带上用户当前问题，组装成完整消息列表，准备发给DeepSeek大模型。

如果走非流式的ask接口，会一次性调用大模型等待完整回答返回，拿到模型原始输出后，先做文本后处理清洗格式，再追加统一的免责声明。处理完成后，把本轮用户提问和最终回答写入Redis保存会话记忆，写入操作会通过pipeline一次性完成压入列表、截断最大轮数、设置过期时间，Redis写入失败只会打告警，不中断业务。保存完对话之后，判断当前会话总轮数是不是5的倍数，如果满足条件，就把最近5轮对话丢给异步任务去提取长期记忆事实，这个抽取过程不会阻塞用户响应，最后把答案和检索来源一起返回给调用方。

如果走流式的ask_stream接口，前面急症判断、角色解析、读取历史、查询改写、检索、召回长期记忆、组装消息这些步骤和非流式完全一样，但是在调用大模型生成之前，会先通过SSE把检索到的资料sources推送给前端，让前端可以提前展示引用来源。随后开启流式调用大模型，模型每吐出一小段token，就马上转成SSE数据逐行推给前端，同时收集所有返回的token，在内存里拼出完整回答。如果流式生成中途出现异常，就发送error事件，直接结束流，**不会把这轮不完整回答写入Redis，也不会触发长期记忆抽取**。只有完整生成没有报错的时候，才对拼接好的全文做后处理、加上免责声明，再把问答存入Redis，同样在满足轮数条件时调度长期记忆抽取任务，最后发送`[DONE]`标记流式会话结束。整个模块对Redis做懒加载，第一次访问Redis的时候才创建客户端，并且设置较短读写超时，保证Redis服务异常的时候，仅仅丢失会话记忆，不会卡住整个问答请求。

"""