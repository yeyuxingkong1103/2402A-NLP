# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-Graph RAG 优化任务
# 模块职责：封装 DeepSeek 大模型调用，提供重试、超时、JSON 解析与磁盘缓存能力（容错机制）

import os                                                       # 导入 os 用于缓存文件路径处理
import re                                                       # 导入正则用于从文本中提取 JSON 片段
import json                                                     # 导入 json 用于缓存与结构化输出解析
import hashlib                                                  # 导入 hashlib 用于生成稳定的缓存键
import time                                                     # 导入 time 用于重试退避等待
from openai import OpenAI                                       # 导入 OpenAI 兼容客户端（DeepSeek 兼容该协议）
import config                                                   # 导入本工单配置模块
from timing import LOGGER                                       # 导入统一日志对象

# 创建全局客户端实例（模块级复用，避免每次调用重复建立连接池）
_CLIENT = None                                                  # 客户端单例占位
_CACHE = None                                                   # LLM 结果缓存字典占位（懒加载）
_CACHE_DIRTY = False                                            # 标记缓存是否有未落盘的变更


def _client():                                                  # 定义获取 LLM 客户端的内部函数
    global _CLIENT                                              # 声明使用模块级变量
    if _CLIENT is None:                                         # 首次调用时才创建客户端（懒加载）
        _CLIENT = OpenAI(api_key=config.DEEPSEEK_API_KEY,       # 传入 API 密钥
                         base_url=config.DEEPSEEK_BASE_URL,     # 传入兼容的基础地址
                         timeout=config.LLM_TIMEOUT)            # 传入请求超时时间
    return _CLIENT                                              # 返回客户端实例


def _load_cache():                                              # 定义加载 LLM 缓存文件的内部函数
    global _CACHE                                               # 声明使用模块级缓存变量
    if _CACHE is None:                                          # 仅在首次调用时读取磁盘
        if config.ENABLE_LLM_CACHE and os.path.isfile(config.EVAL_CACHE_FILE):  # 缓存开关开启且文件存在
            try:                                                # 尝试读取缓存文件
                with open(config.EVAL_CACHE_FILE, "r", encoding="utf-8") as f:  # 以 UTF-8 打开缓存
                    _CACHE = json.load(f)                       # 反序列化为字典
            except Exception as e:                              # 读取或解析失败时容错
                LOGGER.warning("LLM 缓存读取失败，忽略：%r", e)  # 输出告警但不中断流程
                _CACHE = {}                                     # 回退为空缓存
        else:                                                   # 缓存关闭或无缓存文件
            _CACHE = {}                                         # 初始化为空字典
    return _CACHE                                               # 返回缓存字典


def save_cache(force=False):                                    # 定义缓存落盘函数
    global _CACHE_DIRTY                                         # 声明使用脏标记
    if not config.ENABLE_LLM_CACHE:                             # 缓存被关闭时直接返回
        return                                                  # 不做任何落盘操作
    if not _CACHE_DIRTY and not force:                          # 无变更且非强制时跳过落盘，减少 IO
        return                                                  # 提前返回
    config.ensure_dirs()                                        # 确保数据目录存在
    try:                                                        # 尝试写入缓存文件
        with open(config.EVAL_CACHE_FILE, "w", encoding="utf-8") as f:  # 以 UTF-8 覆盖写入
            json.dump(_load_cache(), f, ensure_ascii=False)     # 序列化缓存字典
        _CACHE_DIRTY = False                                    # 落盘成功后清除脏标记
    except Exception as e:                                      # 写入失败时容错
        LOGGER.warning("LLM 缓存写入失败，忽略：%r", e)          # 仅告警，不影响主流程


def _cache_key(system, user, temperature):                      # 定义缓存键生成函数
    raw = "SYS=%s||USR=%s||T=%.2f" % (system, user, temperature)  # 拼接系统提示、用户提示与温度
    # 必须使用稳定哈希：内置 hash() 受 PYTHONHASHSEED 影响，进程重启后同一提示词会得到不同键，
    # 导致缓存永远不命中（且可能把不同提示词的旧结果串用），故改用 md5
    return hashlib.md5(raw.encode("utf-8")).hexdigest()         # 返回 32 位十六进制摘要作为缓存键


def chat(system, user, temperature=0.0, use_cache=True, max_tokens=1500):  # 定义对话补全主函数
    """调用 DeepSeek 对话接口，返回纯文本；带重试、超时与可选缓存。"""
    key = _cache_key(system, user, temperature)                 # 计算本次调用的缓存键
    cache = _load_cache()                                       # 加载缓存字典
    if use_cache and key in cache:                              # 命中缓存则直接返回
        return cache[key]                                       # 返回历史结果，节省时间与费用
    messages = [{"role": "system", "content": system},          # 组装系统角色消息
                {"role": "user", "content": user}]              # 组装用户角色消息
    last_err = None                                             # 记录最后一次异常用于最终抛出
    for attempt in range(config.LLM_MAX_RETRY):                 # 按最大重试次数循环
        try:                                                    # 尝试发起请求
            resp = _client().chat.completions.create(           # 调用兼容 OpenAI 的对话接口
                model=config.DEEPSEEK_MODEL,                    # 指定模型名称
                messages=messages,                              # 传入消息列表
                temperature=temperature,                        # 传入温度（评估类任务用 0 保证稳定）
                max_tokens=max_tokens,                          # 限制输出长度，控制成本与延迟
            )
            text = resp.choices[0].message.content or ""        # 提取回复文本（空值兜底为空串）
            text = text.strip()                                 # 去除首尾空白
            if use_cache:                                       # 若启用缓存
                cache[key] = text                               # 写入内存缓存
                globals()["_CACHE_DIRTY"] = True                # 标记缓存已变更
            return text                                         # 返回模型回复
        except Exception as e:                                  # 捕获网络或接口异常
            last_err = e                                        # 记录异常
            LOGGER.warning("LLM 调用失败（第 %d 次重试）：%r", attempt + 1, e)  # 输出重试日志
            time.sleep(1.0 * (attempt + 1))                     # 线性退避等待，缓解限流
    LOGGER.error("LLM 调用最终失败：%r", last_err)               # 重试耗尽后记录错误
    return ""                                                   # 返回空串，由上层做降级处理（容错）


def chat_json(system, user, temperature=0.0, use_cache=True, max_tokens=2000):  # 定义结构化 JSON 输出函数
    """调用 LLM 并解析为 Python 对象；解析失败时返回 None 由上层降级。"""
    system_json = system + "\n你必须只输出合法 JSON，不要输出任何解释文字或 Markdown 代码块标记。"  # 强化 JSON 输出约束
    text = chat(system_json, user, temperature, use_cache, max_tokens)  # 调用对话函数获取文本
    obj = extract_json(text)                                    # 从文本中稳健提取 JSON 对象
    return obj                                                  # 返回解析结果（可能为 None）


def extract_json(text):                                         # 定义从任意文本中提取 JSON 的函数
    if not text:                                                # 文本为空时直接返回
        return None                                             # 返回 None
    cleaned = text.strip()                                      # 去除首尾空白
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()     # 去掉可能存在的 Markdown 代码块起始标记
    cleaned = re.sub(r"```$", "", cleaned).strip()              # 去掉可能存在的 Markdown 代码块结束标记
    try:                                                        # 优先尝试整体解析
        return json.loads(cleaned)                              # 直接解析为 Python 对象
    except Exception:                                           # 整体解析失败时继续尝试定位片段
        pass                                                    # 忽略异常继续
    for pattern in (r"\{.*\}", r"\[.*\]"):                      # 依次尝试匹配对象或数组片段（贪婪匹配）
        m = re.search(pattern, cleaned, re.S)                   # 在文本中搜索匹配片段
        if m:                                                   # 找到候选片段
            try:                                                # 尝试解析候选片段
                return json.loads(m.group(0))                   # 解析成功即返回
            except Exception:                                   # 片段仍无法解析
                continue                                        # 尝试下一个模式
    LOGGER.warning("无法从 LLM 输出中解析 JSON：%s", text[:200])  # 记录失败样本便于排查
    return None                                                 # 全部失败返回 None
