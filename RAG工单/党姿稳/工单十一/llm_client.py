# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG项目-Embedding 模型微调任务
# 模块职责：封装 DeepSeek 大模型调用，供「问答对数据集生成」使用；带重试、超时与磁盘缓存
# 说明：缓存键使用 md5，避免内置 hash() 受 PYTHONHASHSEED 影响导致进程重启后永不命中

import os                                                       # 导入 os 用于缓存文件路径处理
import json                                                     # 导入 json 用于缓存读写
import time                                                     # 导入 time 用于退避等待
import hashlib                                                  # 导入 hashlib 用于生成稳定的缓存键
import threading                                                # 导入 threading 用于多线程并发下的缓存加锁
import config                                                   # 导入本工单配置模块
from openai import OpenAI                                       # 导入 OpenAI 兼容客户端

_CLIENT = None                                                  # 全局客户端单例（延迟初始化）
_CACHE = None                                                   # 全局缓存字典（延迟加载）
_LOCK = threading.Lock()                                        # 缓存写操作互斥锁，防止并发写坏文件


def _cache_key(system, user, temperature, max_tokens):          # 定义缓存键生成函数
    """用 md5 对请求参数取摘要，作为稳定且与进程无关的缓存键。"""
    raw = "|".join([system or "", user or "",                 # 依次拼接系统提示
                    str(temperature), str(max_tokens)])        # 与温度、最大长度
    return hashlib.md5(raw.encode("utf-8")).hexdigest()         # 返回 32 位十六进制摘要


def _load_cache():                                              # 定义缓存加载函数
    """首次调用时从磁盘加载缓存，之后复用内存副本。"""
    global _CACHE                                               # 声明使用全局缓存变量
    if _CACHE is None:                                          # 尚未加载时
        try:                                                    # 尝试读取缓存文件
            with open(config.LLM_CACHE_FILE, "r", encoding="utf-8") as f:  # 以 UTF-8 打开
                _CACHE = json.load(f)                           # 解析为字典
        except Exception:                                       # 文件不存在或损坏
            _CACHE = {}                                         # 使用空字典兜底
    return _CACHE                                               # 返回缓存字典


def _save_cache():                                              # 定义缓存落盘函数
    """把内存缓存写回磁盘，供下次运行复用。"""
    config.ensure_dirs()                                        # 确保数据目录存在
    with _LOCK:                                                 # 加锁避免多线程同时写
        tmp = config.LLM_CACHE_FILE + ".tmp"                    # 先写临时文件
        with open(tmp, "w", encoding="utf-8") as f:             # 以 UTF-8 打开临时文件
            json.dump(_CACHE, f, ensure_ascii=False)            # 保留中文，不转义
        os.replace(tmp, config.LLM_CACHE_FILE)                  # 原子替换，避免写一半损坏


def get_client():                                               # 定义客户端获取函数
    """返回线程安全的 OpenAI 兼容客户端单例。"""
    global _CLIENT                                              # 声明使用全局客户端变量
    if _CLIENT is None:                                         # 尚未创建时
        with _LOCK:                                             # 加锁避免并发重复创建
            if _CLIENT is None:                                 # 双重检查
                _CLIENT = OpenAI(api_key=config.DEEPSEEK_API_KEY,  # 填入密钥
                                 base_url=config.DEEPSEEK_BASE_URL,  # 填入接口地址
                                 timeout=config.LLM_TIMEOUT,     # 设置请求超时
                                 max_retries=0)                  # 关闭 SDK 自带重试，改由本模块控制
    return _CLIENT                                              # 返回客户端


def chat(system, user, temperature=0.3, max_tokens=1024, use_cache=True):  # 定义对话函数
    """执行一次大模型对话，命中缓存则直接返回；失败时线性退避重试。"""
    cache = _load_cache()                                       # 加载缓存字典
    key = _cache_key(system, user, temperature, max_tokens)     # 计算本次请求的缓存键
    if use_cache and key in cache:                              # 缓存命中
        return cache[key]                                       # 直接返回历史结果（不再计费）
    last_err = None                                             # 记录最后一次异常
    for attempt in range(config.LLM_MAX_RETRY):                 # 按最大重试次数循环
        try:                                                    # 尝试调用接口
            resp = get_client().chat.completions.create(        # 发起 ChatCompletion 请求
                model=config.DEEPSEEK_MODEL,                    # 指定模型
                messages=[{"role": "system", "content": system},  # 系统提示
                          {"role": "user", "content": user}],   # 用户提示
                temperature=temperature,                        # 采样温度
                max_tokens=max_tokens)                          # 输出长度上限
            text = (resp.choices[0].message.content or "").strip()  # 取出正文并去除首尾空白
            if text:                                            # 仅在拿到非空结果时写缓存
                cache[key] = text                               # 写入内存缓存
                _save_cache()                                   # 同步落盘
            return text                                         # 返回结果
        except Exception as e:                                  # 捕获异常
            last_err = e                                        # 记录异常
            if attempt < config.LLM_MAX_RETRY - 1:              # 还有重试机会时
                time.sleep(1.0 * (attempt + 1))                 # 线性退避等待
    print("[警告] LLM 调用失败：%r" % last_err)                  # 重试耗尽后打印告警
    return ""                                                   # 返回空串表示失败


def _strip_fence(text):                                         # 定义代码块包裹剥离函数
    """去掉模型习惯性输出的 ```json ... ``` 包裹。"""
    text = text.strip()                                         # 去除首尾空白
    if text.startswith("```"):                                  # 命中 markdown 代码块
        text = text.strip("`").strip()                          # 去掉反引号与残留空白
        if text.lower().startswith("json"):                     # 去掉语言标记
            text = text[4:].strip()                             # 截掉 "json" 四个字符
    return text                                                 # 返回剥离后的文本


def chat_json(system, user, temperature=0.2, max_tokens=1500, use_cache=True):  # 定义 JSON 对话函数
    """调用大模型并解析 JSON 结果；解析失败返回 None，由调用方决定是否重试。"""
    raw = chat(system, user, temperature=temperature,           # 复用 chat 完成调用
               max_tokens=max_tokens, use_cache=use_cache)      # 透传缓存开关
    if not raw:                                                 # 调用失败
        return None                                             # 返回 None
    text = _strip_fence(raw)                                    # 剥离代码块包裹
    try:                                                        # 先尝试整体解析（最常见且最准确）
        return json.loads(text)                                 # 直接返回解析结果
    except Exception:                                           # 整体解析失败，进入容错提取
        pass                                                    # 忽略，继续下面的兜底逻辑
    # 兜底：按括号配对截取最外层结构。必须以「先出现的那个括号」为准——
    # 早期版本先找 "{"，导致数组 [ {...}, {...} ] 被从第一个 "{" 截到最后一个 "}"，
    # 截出「对象后面跟着多余数据」，json 解析报 Extra data
    cands = []                                                  # 候选 (起始位置, 结束位置) 列表
    for open_ch, close_ch in (("[", "]"), ("{", "}")):           # 分别尝试数组与对象两种结构
        s, e = text.find(open_ch), text.rfind(close_ch)          # 找该结构的起始与结束括号
        if 0 <= s < e:                                          # 括号次序合法时
            cands.append((s, e))                                # 记录该候选的起止位置
    cands.sort()                                                # 按起始位置升序：最靠前的结构即最外层
    for s, e in cands:                                          # 依次尝试每个候选
        try:                                                    # 尝试解析该片段
            return json.loads(text[s:e + 1])                    # 成功则直接返回
        except Exception:                                       # 该片段不合法
            continue                                            # 换下一个候选
    print("[警告] 无法从模型输出中解析 JSON：%s" % raw[:120])    # 全部失败时打印前 120 字便于排查
    return None                                                 # 返回 None 表示解析失败
