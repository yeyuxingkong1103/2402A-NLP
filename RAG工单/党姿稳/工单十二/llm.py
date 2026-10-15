# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG项目-LightRAG 优化任务
# 模块职责：封装 DeepSeek 大模型调用，提供重试、超时、稳健 JSON 解析与磁盘缓存能力

import os                                                       # 导入 os 用于缓存文件判断
import re                                                       # 导入正则用于剥离 Markdown 代码块标记
import json                                                     # 导入 json 用于缓存与结构化输出解析
import hashlib                                                  # 导入 hashlib 生成稳定缓存键
import time                                                     # 导入 time 用于重试退避等待
from openai import OpenAI                                       # 导入 OpenAI 兼容客户端（DeepSeek 兼容该协议）
import config                                                   # 导入本工单配置模块
from logger import LOGGER                                       # 导入统一日志对象

import threading                                                # 导入 threading 用于并发调用时的缓存加锁

_CLIENT = None                                                  # 客户端单例占位（懒加载）
_CACHE = None                                                   # LLM 结果缓存字典占位（懒加载）
_CACHE_DIRTY = False                                            # 标记缓存是否有未落盘的变更
_LOCK = threading.Lock()                                        # 并发调用时的缓存与客户端互斥锁


def _client():                                                  # 定义获取 LLM 客户端的内部函数
    """懒加载全局客户端，避免每次调用重复建立连接池。"""
    global _CLIENT                                              # 声明使用模块级变量
    with _LOCK:                                                 # 加锁保证多线程下只创建一次
        if _CLIENT is None:                                     # 首次调用时才创建
            _CLIENT = OpenAI(api_key=config.DEEPSEEK_API_KEY,    # 传入 API 密钥
                             base_url=config.DEEPSEEK_BASE_URL,  # 传入兼容的基础地址
                             timeout=config.LLM_TIMEOUT)        # 传入请求超时时间
    return _CLIENT                                              # 返回客户端实例


def _load_cache():                                              # 定义加载缓存文件的内部函数
    """首次调用时从磁盘读取缓存字典，失败则回退为空字典。"""
    global _CACHE                                               # 声明使用模块级缓存变量
    if _CACHE is None:                                          # 仅在首次调用时读取磁盘
        if config.ENABLE_LLM_CACHE and os.path.isfile(config.LLM_CACHE_FILE):  # 缓存开启且文件存在
            try:                                                # 尝试读取缓存文件
                with open(config.LLM_CACHE_FILE, "r", encoding="utf-8") as f:  # 以 UTF-8 打开
                    _CACHE = json.load(f)                       # 反序列化为字典
            except Exception as e:                              # 读取或解析失败时容错
                LOGGER.warning("LLM 缓存读取失败，忽略：%r", e)  # 输出告警但不中断流程
                _CACHE = {}                                     # 回退为空缓存
        else:                                                   # 缓存关闭或无缓存文件
            _CACHE = {}                                         # 初始化为空字典
    return _CACHE                                               # 返回缓存字典


def save_cache(force=False):                                    # 定义缓存落盘函数
    """把内存缓存写入磁盘；无变更且非强制时跳过，减少 IO。"""
    global _CACHE_DIRTY                                         # 声明使用脏标记
    if not config.ENABLE_LLM_CACHE:                             # 缓存被关闭时直接返回
        return                                                  # 不做任何落盘操作
    with _LOCK:                                                 # 加锁避免并发写坏文件
        if not _CACHE_DIRTY and not force:                      # 无变更且非强制时跳过
            return                                              # 提前返回
        config.ensure_dirs()                                    # 确保数据目录存在
        try:                                                    # 尝试写入缓存文件
            with open(config.LLM_CACHE_FILE, "w", encoding="utf-8") as f:  # 以 UTF-8 覆盖写入
                json.dump(_load_cache(), f, ensure_ascii=False)  # 序列化缓存字典
            _CACHE_DIRTY = False                                # 落盘成功后清除脏标记
        except Exception as e:                                  # 写入失败时容错
            LOGGER.warning("LLM 缓存写入失败，忽略：%r", e)      # 仅告警，不影响主流程


def _cache_key(system, user, temperature):                      # 定义缓存键生成函数
    """用 md5 生成稳定缓存键；内置 hash() 受 PYTHONHASHSEED 影响，不能用作跨进程缓存键。"""
    raw = "SYS=%s||USR=%s||T=%.2f" % (system, user, temperature)  # 拼接系统提示、用户提示与温度
    return hashlib.md5(raw.encode("utf-8")).hexdigest()         # 返回 32 位十六进制摘要


def chat(system, user, temperature=0.0, use_cache=True, max_tokens=1500):  # 定义对话补全主函数
    """调用 DeepSeek 对话接口返回纯文本；带重试、超时与可选磁盘缓存。"""
    key = _cache_key(system, user, temperature)                 # 计算本次调用的缓存键
    cache = _load_cache()                                       # 加载缓存字典
    if use_cache and key in cache:                              # 命中缓存则直接返回
        return cache[key]                                       # 返回历史结果，节省时间与费用
    messages = [{"role": "system", "content": system},          # 组装系统角色消息
                {"role": "user", "content": user}]              # 组装用户角色消息
    last_err = None                                             # 记录最后一次异常用于最终提示
    for attempt in range(config.LLM_MAX_RETRY):                 # 按最大重试次数循环
        try:                                                    # 尝试发起请求
            resp = _client().chat.completions.create(           # 调用兼容 OpenAI 的对话接口
                model=config.DEEPSEEK_MODEL,                    # 指定模型名称
                messages=messages,                              # 传入消息列表
                temperature=temperature,                        # 传入温度（评估类任务用 0 保证稳定）
                max_tokens=max_tokens)                          # 限制输出长度，控制成本与延迟
            text = (resp.choices[0].message.content or "").strip()  # 提取回复文本并去除首尾空白
            if use_cache:                                       # 若启用缓存
                with _LOCK:                                     # 加锁写入共享缓存
                    cache[key] = text                           # 写入内存缓存
                    globals()["_CACHE_DIRTY"] = True            # 标记缓存已变更
            return text                                         # 返回模型回复
        except Exception as e:                                  # 捕获网络或接口异常
            last_err = e                                        # 记录异常
            LOGGER.warning("LLM 调用失败（第 %d 次重试）：%r", attempt + 1, e)  # 输出重试日志
            time.sleep(1.0 * (attempt + 1))                     # 线性退避等待，缓解限流
    LOGGER.error("LLM 调用最终失败：%r", last_err)               # 重试耗尽后记录错误
    return ""                                                   # 返回空串，由上层做降级处理


def extract_json(text):                                         # 定义从任意文本中提取 JSON 的函数
    """按「整体解析 → 括号对候选片段」的顺序容错解析，返回 Python 对象或 None。"""
    if not text:                                                # 文本为空时直接返回
        return None                                             # 返回 None
    cleaned = text.strip()                                      # 去除首尾空白
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()     # 去掉可能存在的 Markdown 起始标记
    cleaned = re.sub(r"```$", "", cleaned).strip()              # 去掉可能存在的 Markdown 结束标记
    try:                                                        # 优先尝试整体解析（覆盖绝大多数正常返回）
        return json.loads(cleaned)                              # 直接解析为 Python 对象
    except Exception:                                           # 整体解析失败时继续尝试定位片段
        pass                                                    # 忽略异常继续
    cands = []                                                  # 收集括号对候选片段
    for open_ch, close_ch in (("[", "]"), ("{", "}")):          # 依次尝试数组与对象两种括号
        s, e = cleaned.find(open_ch), cleaned.rfind(close_ch)   # 定位最外侧的括号对
        if 0 <= s < e:                                          # 括号顺序合法时才收集
            cands.append((s, e))                                # 记录片段起止位置
    cands.sort()                                                # 按起始位置升序，优先采用出现更早的片段
    for s, e in cands:                                          # 遍历候选片段
        try:                                                    # 尝试解析该片段
            return json.loads(cleaned[s:e + 1])                 # 解析成功即返回
        except Exception:                                       # 片段仍无法解析
            continue                                            # 尝试下一个候选
    LOGGER.warning("无法从 LLM 输出中解析 JSON：%s", text[:200])  # 记录失败样本便于排查
    return None                                                 # 全部失败返回 None


def chat_json(system, user, temperature=0.0, use_cache=True, max_tokens=2000):  # 定义结构化输出函数
    """调用 LLM 并把回复解析为 Python 对象；解析失败时返回 None 由上层降级。"""
    system_json = system + "\n你必须只输出合法 JSON，不要输出任何解释文字或 Markdown 代码块标记。"  # 强化 JSON 约束
    text = chat(system_json, user, temperature, use_cache, max_tokens)  # 调用对话函数获取文本
    return extract_json(text)                                   # 返回稳健解析后的对象（可能为 None）
