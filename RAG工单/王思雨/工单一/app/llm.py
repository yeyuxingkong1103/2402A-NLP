# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 本模块职责：大模型调用（DeepSeek）+ Redis 缓存容错 + Query 理解，是问答链路的最上游。
# 拆出本模块是为了满足「每个 .py 不超过 300 行」的工程约束。

import re                             # 正则：JSON 抽取、意图识别、冗余引导语剥离
import json                           # 请求与响应体序列化
import time                           # 重试退避与统计
import hashlib                        # 生成缓存键

import httpx                          # 调用 DeepSeek HTTP 接口
from . import config as C             # 全局配置
from . import rag                     # 复用分词能力做关键词抽取

_REDIS = None                         # Redis 客户端单例
_REDIS_DOWN = False                   # Redis 不可用标记，避免每次请求都重试拖慢响应

# 多跳/复杂问题的判定词：命中才启用 LLM Query 改写，否则走零延迟的本地规则理解。
# 取舍依据（实测）：LLM 改写一次约 7.5s（deepseek-v4-flash 为推理模型，先产出 reasoning_tokens），
# 远超 3 秒响应预算，因此只对真正的多跳/比较类问题启用；
# 「分别」「和」这类词在事实型问题里极常见（工单 10 问里有 3 问含「分别」），不能作为复杂标记。
_COMPLEX_HINTS = ("对比", "区别", "相比", "差异", "为什么", "如何", "各自", "总结", "分析一下")
# 意图识别的关键词表：按优先级从上到下匹配
_INTENT_RULES = [("numeric", ("多少", "几", "比重", "占比", "金额", "万元", "比例", "增长率")),
                 ("list", ("哪些", "包括", "涉及", "有哪些")),
                 ("boolean", ("是否", "有没有", "能否")),
                 ("compare", ("对比", "区别", "相比", "差异")),
                 ("factoid", ("谁", "哪个", "哪个工程", "何时", "什么时候", "多少年"))]


# ============================ 大模型调用 ============================
def llm_chat(messages: list[dict], max_tokens: int = 900, temperature: float = 0.2,
             stream: bool = False, retries: int = 2):                 # DeepSeek 统一调用入口
    """调用 DeepSeek；失败按指数退避重试；stream=True 时返回逐块产出文本的生成器。"""
    url = C.LLM_BASE_URL.rstrip("/") + "/chat/completions"            # 拼接补全接口地址
    headers = {"Authorization": f"Bearer {C.LLM_API_KEY}",            # 鉴权头
               "Content-Type": "application/json"}                     # 内容类型
    payload = {"model": C.LLM_MODEL, "messages": messages, "max_tokens": max_tokens,
               "temperature": temperature, "stream": stream}           # 请求体
    if C.LLM_REASONING_EFFORT and C.LLM_REASONING_EFFORT != "default":  # 按配置控制推理开销：
        payload["reasoning_effort"] = C.LLM_REASONING_EFFORT            # none 可把首字延迟从 5.2s 降到 1.6s
    last = None                                                        # 记录最后一次异常
    for attempt in range(retries + 1):                                 # 含首次共 retries+1 次
        try:                                                           # 尝试请求
            if stream:                                                 # 流式分支
                return _stream_llm(url, headers, payload)              # 返回生成器（惰性执行）
            with httpx.Client(timeout=C.LLM_TIMEOUT) as cli:            # 非流式分支
                r = cli.post(url, headers=headers, json=payload)       # 发起请求
                r.raise_for_status()                                   # 非 2xx 抛异常
                return r.json()["choices"][0]["message"]["content"]    # 取回答正文
        except Exception as exc:                                       # 网络错误/超时/限流
            last = exc                                                 # 记录异常
            time.sleep(0.6 * (2 ** attempt))                           # 指数退避：0.6s、1.2s、2.4s
    raise RuntimeError(f"DeepSeek 调用失败：{type(last).__name__}: {last}")  # 全部重试失败才抛


def _stream_llm(url, headers, payload):                                # 流式请求内部实现
    """逐行读取 SSE 响应，产出增量文本；连接层异常向上抛出由调用方兜底。"""
    with httpx.Client(timeout=C.LLM_TIMEOUT) as cli:                    # 建立客户端
        with cli.stream("POST", url, headers=headers, json=payload) as resp:  # 流式 POST
            resp.raise_for_status()                                     # 状态码检查
            for line in resp.iter_lines():                              # 逐行读取
                if not line or not line.startswith("data:"):            # 跳过空行与非数据行
                    continue                                            # 继续下一行
                data = line[5:].strip()                                 # 去掉 "data:" 前缀
                if data == "[DONE]":                                    # 结束标记
                    break                                               # 结束迭代
                try:                                                    # 解析增量 JSON
                    delta = json.loads(data)["choices"][0].get("delta", {})  # 取 delta 字段
                    piece = delta.get("content")                        # 增量文本
                    if piece:                                           # 非空才产出
                        yield piece                                     # 交给上层拼装
                except Exception:                                       # 个别脏行不影响整体
                    continue                                            # 忽略继续


def llm_available() -> bool:                                           # LLM 可用性判断
    """Key 未配置时直接视为不可用，供降级分支使用。"""
    return bool(C.LLM_API_KEY)                                         # 只判断 Key 是否存在


# ============================ Redis 缓存（容错） ============================
def get_redis():                                                       # Redis 客户端单例
    """连接 Redis；任何异常都返回 None，调用方退化为无缓存模式（容错要求）。"""
    global _REDIS, _REDIS_DOWN                                        # 使用模块级状态
    if _REDIS is not None or _REDIS_DOWN:                              # 已有连接或已判定不可用
        return _REDIS                                                  # 直接返回（可能是 None）
    try:                                                               # 尝试连接
        import redis                                                   # 延迟导入
        _REDIS = redis.from_url(C.REDIS_URL, protocol=C.REDIS_PROTOCOL,  # Redis 5 必须用 RESP2
                                decode_responses=True, socket_timeout=1.5)  # 短超时，避免拖慢主链路
        _REDIS.ping()                                                  # 连通性验证
    except Exception as exc:                                           # 连接失败
        print(f"[缓存降级] Redis 不可用，转为无缓存模式：{type(exc).__name__}")  # 记录原因
        _REDIS, _REDIS_DOWN = None, True                               # 标记不可用，后续不再重试
    return _REDIS                                                      # 返回客户端或 None


def cache_get(key: str):                                               # 读缓存（永不抛异常）
    r = get_redis()                                                    # 取客户端
    if not r:                                                          # 不可用
        return None                                                    # 视为未命中
    try:                                                               # 防御性捕获
        v = r.get(key)                                                 # 读取
        return json.loads(v) if v else None                            # 反序列化
    except Exception:                                                  # 序列化/网络异常
        return None                                                    # 视为未命中


def cache_set(key: str, value, ttl: int = 3600) -> None:               # 写缓存（永不抛异常）
    r = get_redis()                                                    # 取客户端
    if not r:                                                          # 不可用
        return                                                         # 直接返回
    try:                                                               # 防御性捕获
        r.setex(key, ttl, json.dumps(value, ensure_ascii=False))       # 带过期时间写入
    except Exception:                                                  # 写入失败
        pass                                                           # 静默忽略，不影响主流程


def cache_key(prefix: str, text: str) -> str:                          # 生成缓存键
    """归一化后取 md5，避免大小写与空白差异导致缓存击穿。"""
    norm = re.sub(r"\s+", "", text).lower()                            # 去空白、转小写
    return f"rag:{prefix}:{hashlib.md5(norm.encode('utf-8')).hexdigest()}"  # 前缀 + md5


# ============================ Query 理解 ============================
def understand_query(question: str, use_rewrite: str = "auto") -> dict:  # Query 理解总入口
    """意图识别 + 消歧 + 分解与抽象。use_rewrite: auto/true/false，auto 下简单问题走本地规则。"""
    local = _local_understand(question)                                # 先做零延迟的本地理解
    need = (use_rewrite is True or use_rewrite == "true") or (         # 是否需要调 LLM
        use_rewrite == "auto" and (len(question) > 80 or any(h in question for h in _COMPLEX_HINTS)))
    if not need or not llm_available():                                # 不需要或没有 Key
        local["rewrite_source"] = "local"                              # 标记来源为本地规则
        return local                                                   # 直接返回本地结果
    try:                                                               # 尝试 LLM 改写
        prompt = ("你是检索查询理解模块。请对用户问题做四件事：1)意图识别(intent)；"
                  "2)消歧(把简称补全为文档中的规范全称)；3)分解为若干可独立检索的子问题；"
                  "4)抽象出用于检索的关键词。只输出 JSON，字段："
                  '{"intent":"factoid|numeric|list|boolean|compare","rewritten":"改写后的检索式",'
                  '"sub_questions":["..."],"keywords":["..."]}\n用户问题：' + question)  # 提示词
        # 注意：deepseek-v4-flash 是推理模型，会先消耗 reasoning_tokens 再输出正文。
        # max_tokens 给小了（实测 300）推理会把预算吃光，content 返回空字符串，故留足 1200。
        raw = llm_chat([{"role": "user", "content": prompt}], max_tokens=1200, temperature=0.0)  # 轻量调用
        if not raw or not raw.strip():                                 # 空回复（推理耗尽预算）
            raise ValueError("LLM 返回空内容")                          # 抛异常触发本地规则降级
        data = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))    # 从回复中抠出 JSON
        data["original"] = question                                    # 保留原问题
        data["rewrite_source"] = "llm"                                 # 标记来源为 LLM
        data.setdefault("sub_questions", [])                           # 补齐缺失字段
        return data                                                    # 返回 LLM 理解结果
    except Exception as exc:                                           # 解析失败/接口异常
        print(f"[Query理解降级] LLM 改写失败，用本地规则：{type(exc).__name__}")  # 记录原因
        local["rewrite_source"] = "local"                              # 标记来源
        return local                                                   # 降级返回本地结果


def _local_understand(question: str) -> dict:                          # 本地规则式 Query 理解
    """零延迟实现意图识别、抽象（剥离冗余引导语）、分解（并列子句拆分）。"""
    q = question.strip()                                               # 去空白
    intent = "factoid"                                                 # 默认意图
    for name, words in _INTENT_RULES:                                  # 依次匹配意图关键词
        if any(w in q for w in words):                                 # 命中该意图
            intent = name                                              # 记录意图
            break                                                      # 取优先级最高者
    core = re.sub(r"^(根据|依据|按照)?[^，,]{0,30}?(招股意向书|招股说明书|年报|报告)[，,]?", "", q)  # 剥离引导语
    core = re.sub(r"^(请问|请|那么|那|所以)[，,]?", "", core).strip() or q  # 剥离口语化前缀
    subs = [s.strip("？?。 ") for s in re.split(r"[，,]\s*(?=以及|和|还有|另外)|以及|和(?=[^，,]{6,})", core)
            if s and s.strip("？?。 ")]                                 # 按并列连词分解子问题
    keywords = [w for w in rag.tokenize(core) if len(w) > 1][:20]      # 抽取检索关键词
    return {"original": question, "intent": intent, "rewritten": core,  # 返回本地理解结果
            "sub_questions": subs if len(subs) > 1 else [],            # 只有真分解出多条才保留
            "keywords": keywords}                                      # 关键词列表
