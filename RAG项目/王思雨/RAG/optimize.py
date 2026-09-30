# -*- coding: utf-8 -*-
"""在线优化模块：查询改写、查询扩写、Redis 答案缓存，第 7 步不含离线数据增强。"""

import json                                   # 导入 json，用于缓存值的序列化
import re                                     # 导入 re，用于查询归一化的正则替换
from typing import List                       # 导入 List，用于声明列表类型

import config                                 # 导入配置模块，缓存时长从这里读
import db                                     # 导入数据库模块，复用 Redis 连接封装
from logger import get_logger                 # 导入日志工具，用于记录优化过程

logger = get_logger("optimize")               # 创建本模块的 logger 实例

CACHE_PREFIX = "cache:rag:"                   # 缓存键统一前缀，便于管理与清理
HISTORY_ROUNDS = 2                            # 改写时最多带入的历史轮数

# 归一化时要去掉的标点：中文标点 + 英文标点
PUNCTUATION_RE = re.compile(r"[，。！？；：、“”‘’（）【】,.!?;:\"'()\[\]{}]")   # 待删除的标点集合
SPACE_RE = re.compile(r"\s+")                 # 连续空白，用于合并成一个空格


def normalize_query(query: str) -> str:        # 查询归一化
    """归一化查询：去中英文标点、合并空白、转小写，用于生成稳定的缓存键。"""
    text = (query or "").strip()               # 去掉首尾空白
    text = PUNCTUATION_RE.sub("", text)        # 去掉所有中英文标点
    text = SPACE_RE.sub(" ", text)             # 多个连续空白合并为一个空格
    return text.strip().lower()                # 再去首尾空白并转小写


def cache_key(user_id: int, role_name: str, query: str) -> str:   # 生成缓存键
    """按「用户 + 角色 + 归一化后的原始问题」生成缓存键。"""
    return f"{CACHE_PREFIX}answer:{int(user_id)}:{role_name or 'default'}:{normalize_query(query)}"   # 键结构
# 注意：deepseek-v4-flash 是推理模型，max_tokens 里包含思维链消耗。
# 实测改写任务推理要烧掉约 390 token，设太小会导致正文为空（静默失败），因此下限给到 512。
REWRITE_MAX_TOKENS = 512                      # 改写调用的输出上限（含思维链）
EXPAND_MAX_TOKENS = 1024                      # 扩写调用要输出多行，上限给得更宽

# 查询改写提示词：把历史与追问合并成一句自足的检索问题
REWRITE_PROMPT = """把下面的对话历史和当前追问合并成一句独立的、语义完整的检索问题，只输出改写后的问题，不要解释。
历史：{history_text}
追问：{query}"""

# 查询扩写提示词：生成多个不同表达的同义问题
EXPAND_PROMPT = """针对下面这个问题，生成 {n} 个不同表达方式的同义问题，每行一个，不要编号，不要解释。
问题：{query}"""


def _history_text(history: list) -> str:      # 内部函数：把历史整理成文本
    """把历史对话整理成「用户：… / 助手：…」的简短文本，只取最近几轮。"""
    rounds = (history or [])[-HISTORY_ROUNDS * 2:]   # 只取最近几轮（一问一答算两条）
    lines = []                                 # 保存整理后的行
    for turn in rounds:                        # 逐条处理
        speaker = "用户" if turn.get("role") == "user" else "助手"   # 中文说话方
        content = (turn.get("content") or "").strip()[:120]          # 内容截断，避免提示词过长
        if content:                            # 有内容才记
            lines.append(f"{speaker}：{content}")   # 拼成一行
    return "\n".join(lines)                    # 拼成多行文本返回


def _chat(prompt: str, max_tokens: int) -> str:   # 内部函数：一次轻量 LLM 调用
    """调用 DeepSeek 做一次轻量生成，供改写与扩写复用；失败时抛出异常由调用方处理。"""
    from openai import OpenAI                  # 延迟导入，避免未安装时影响模块加载
    client = OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL, timeout=30)   # 建客户端
    response = client.chat.completions.create(  # 发起对话补全
        model=config.LLM_MODEL,                # 模型名称
        messages=[{"role": "user", "content": prompt}],   # 单条提示词
        temperature=0.2,                       # 改写任务要求稳定，温度调低
        max_tokens=max_tokens,                 # 输出长度上限
        stream=False,                          # 非流式
    )                                          # 调用结束
    return (response.choices[0].message.content or "").strip()   # 取正文并去空白


def rewrite_query(query: str, history: list = None) -> str:
    """查询改写：把历史与追问合并成一句自足的检索问题；失败时原样返回 query。"""
    if not history:                            # 没有历史说明是首轮提问
        return query                           # 追问不需要改写
    history_text = _history_text(history)      # 整理历史文本
    if not history_text:                       # 历史整理后为空
        return query                           # 同样不需要改写
    try:                                       # 调用失败时降级为原问题
        rewritten = _chat(REWRITE_PROMPT.format(history_text=history_text, query=query),
                          REWRITE_MAX_TOKENS)   # 执行改写
        if rewritten:                          # 改写有结果
            logger.info("查询改写：%s → %s", query, rewritten)   # 记录改写前后
            return rewritten                   # 返回改写结果
        logger.warning("查询改写返回空，使用原问题")   # 记录告警
    except Exception as exc:                   # 调用异常
        logger.warning("查询改写失败，使用原问题：%s", exc)   # 记录降级原因
    return query                               # 降级返回原问题


def expand_query(query: str, n: int = 3) -> List[str]:
    """查询扩写：生成 n 个不同角度的同义问题，返回原问题加同义问题共 n+1 条。"""
    try:                                       # 调用失败时只返回原问题
        text = _chat(EXPAND_PROMPT.format(n=n, query=query), EXPAND_MAX_TOKENS)   # 执行扩写
        variants = [line.strip() for line in text.splitlines() if line.strip()]   # 按行拆出同义问题
        variants = [v for v in variants if v != query][:n]   # 去掉与原问题重复的，取前 n 条
        logger.info("查询扩写：原问题 + %d 个同义问题", len(variants))   # 记录扩写数量
        return [query] + variants              # 原问题放最前，便于保持主查询优先级
    except Exception as exc:                   # 调用异常
        logger.warning("查询扩写失败，只返回原问题：%s", exc)   # 记录降级原因
        return [query]                         # 降级只返回原问题


def cache_get(key: str):                       # 读缓存
    """读缓存：key 为完整键（含 cache:rag: 前缀），未命中或 Redis 不可用返回 None。"""
    try:                                       # 读操作降级为返回 None
        client = db.get_redis_client()         # 获取 Redis 客户端
        raw = client.get(key)                  # 按键取值，键由调用方用 cache_key() 生成
    except Exception as exc:                   # 连不上
        logger.warning("缓存读取降级为未命中（Redis 不可用）：%s", exc)   # 记录降级原因
        return None                            # 返回 None
    if not raw:                                # 没命中
        return None                            # 返回 None
    try:                                       # 反序列化失败也视为未命中
        return json.loads(raw)                 # 返回解析后的对象
    except Exception:                          # 内容损坏
        logger.warning("缓存内容无法解析，视为未命中：%s", key)   # 记录告警
        return None                            # 返回 None


def cache_set(key: str, value, expire: int = None) -> None:
    """写缓存：key 为完整键（含 cache:rag: 前缀），Redis 不可用时静默跳过。"""
    ttl = expire if expire is not None else config.CACHE_EXPIRE   # 未指定时长则用配置值
    try:                                       # 写失败不影响主流程
        client = db.get_redis_client()         # 获取 Redis 客户端
        client.set(key, json.dumps(value, ensure_ascii=False), ex=ttl)   # 写入并设过期
        logger.info("缓存已写入：%s（%d 秒后过期）", key, ttl)   # 记录日志
    except Exception as exc:                   # 连不上或写入失败
        logger.warning("缓存写入跳过（Redis 不可用）：%s", exc)   # 记录降级原因
