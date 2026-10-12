# 工单编号：人工智能NLP-RAG-Query理解优化
# 文件说明：大模型层。封装 DeepSeek（OpenAI 兼容接口）的调用、重试与提示词构造；
#           回答严格基于检索片段，中文问中文答、英文问英文答，无依据时明确说"未找到"。
import time                                                                     # 重试退避
from typing import List, Dict, Optional                                         # 类型标注

from . import config as C                                                       # 复用集中配置

_CLIENT = None                                                                  # OpenAI 兼容客户端单例


def llm_ready() -> bool:                                                        # 健康检查
    """只判断密钥是否配置；真正可用性在首次调用时暴露。"""
    return bool(C.DEEPSEEK_API_KEY)                                             # 有密钥即视为已配置


def warmup() -> None:                                                           # 预热大模型连接
    """用一次极短的调用把 TLS 建连与连接池建好。

    实测：冷启动后第一次调用要 3.1 秒（DNS + TLS 握手 + 连接池初始化），第二次只要 0.3 秒。
    这 3 秒会直接吃掉首个"需要改写"的问题的响应预算，所以在服务启动时就付掉它。
    max_tokens=1 让这次预热的费用可以忽略。
    """
    if not llm_ready():                                                         # 未配置密钥
        return                                                                  # 无需预热
    try:                                                                        # 预热失败不该让服务起不来
        chat([{"role": "user", "content": "hi"}], max_tokens=1, timeout=30)     # 极短调用
    except Exception as exc:                                                    # 网络或额度问题
        print(f"[预热] 大模型连接预热失败：{type(exc).__name__}: {exc}")          # 明确报因


def client():                                                                   # 懒加载客户端
    """创建 OpenAI 兼容客户端；缺密钥时抛出明确异常，由接口层转成友好提示。"""
    global _CLIENT                                                              # 声明单例
    if _CLIENT is None:                                                         # 尚未创建
        if not C.DEEPSEEK_API_KEY:                                              # 缺密钥
            raise RuntimeError("未配置 DEEPSEEK_API_KEY 环境变量，无法调用大模型")  # 明确报因
        from openai import OpenAI                                               # 延迟导入
        _CLIENT = OpenAI(api_key=C.DEEPSEEK_API_KEY, base_url=C.LLM_BASE_URL,   # 指向 DeepSeek 兼容端点
                         timeout=C.LLM_TIMEOUT, max_retries=0)                  # 重试自己控制，关掉 SDK 内建重试
    return _CLIENT                                                              # 返回


def chat(messages: List[Dict], timeout: Optional[int] = None,                   # 通用对话调用
         max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> str:  # 生成参数可覆盖
    """调用对话补全并返回文本；失败按配置重试，最终失败抛出异常由调用方降级。"""
    cli = client()                                                              # 取客户端
    last_exc = None                                                             # 记录最后一次异常
    for attempt in range(C.LLM_RETRIES + 1):                                    # 首次 + N 次重试
        try:                                                                    # 调用可能因网络或限流失败
            kw = {"model": C.LLM_MODEL, "messages": messages,                   # 模型与消息
                  "max_tokens": max_tokens or C.LLM_MAX_TOKENS,                 # 生成上限
                  "temperature": C.LLM_TEMPERATURE if temperature is None else temperature,  # 温度
                  "timeout": timeout or C.LLM_TIMEOUT}                          # 超时
            if C.LLM_REASONING_EFFORT:                                          # 仅当显式配置了才传
                kw["reasoning_effort"] = C.LLM_REASONING_EFFORT                 # 坑：推理型模型不传 none 会返回空内容
            elif "reason" in C.LLM_MODEL.lower() or "flash" in C.LLM_MODEL.lower():  # 模型名像推理型
                kw["extra_body"] = {"reasoning_effort": "none"}                 # 显式关掉思考，避免吃光 max_tokens
            resp = cli.chat.completions.create(**kw)                            # 发起请求
            content = (resp.choices[0].message.content or "").strip()           # 取文本
            if content:                                                         # 有内容
                return content                                                  # 返回
            last_exc = RuntimeError("大模型返回空内容")                          # 空内容视为失败（推理模型常见）
        except Exception as exc:                                                # 网络/限流/密钥错误
            last_exc = exc                                                      # 记录
        if attempt < C.LLM_RETRIES:                                             # 还有重试机会
            time.sleep(C.LLM_RETRY_SLEEP * (attempt + 1))                       # 退避后重试
    raise RuntimeError(f"大模型调用失败：{type(last_exc).__name__}: {last_exc}")  # 重试耗尽，抛出明确异常


_SYS_ANSWER = (                                                                 # 回答用系统提示词
    "你是招股说明书问答助手。请严格遵守：\n"
    "1) 只依据【检索片段】作答，禁止编造、禁止引入片段外的知识；\n"
    "2) 金额、年份、比例等数字必须与片段完全一致，不得换算、四舍五入或估计；\n"
    "3) 若片段中没有足够信息，请直接回答“未找到相关内容”，并说明还缺什么；\n"
    "4) 在结论句末用【文件名 P页码】标注依据，页码取自片段头部的来源标记；\n"
    "5) 语言跟随提问语言：中文提问用中文回答，英文提问用英文回答(专有名词可保留中文原名)；\n"
    "6) 回答简洁，先给结论再给必要依据，不要复述整段原文。"
)


def build_answer_messages(question: str, contexts: List[Dict],                  # 构造回答用消息
                          history: Optional[List[Dict]] = None) -> List[Dict]:  # 历史可为空
    """把对话历史与检索片段拼成提示词；片段头部带来源标记，便于模型标注引用。"""
    blocks = []                                                                 # 片段文本
    for i, c in enumerate(contexts, 1):                                         # 逐个片段编号
        head = f"[片段{i}] 来源：{c.get('doc', '')} 第{c.get('page_no', 0)}页"     # 来源标记（页码是 1 起）
        if c.get("section"):                                                    # 有章节就带上，帮助模型定位
            head += f" 章节：{c['section'][:40]}"                                  # 拼接章节
        blocks.append(f"{head}\n{c.get('text', '')}")                            # 拼接片段正文
    ctx = "\n\n".join(blocks) if blocks else "（无检索片段）"                       # 无片段时明确写出
    msgs: List[Dict] = [{"role": "system", "content": _SYS_ANSWER}]             # 系统提示
    for h in (history or [])[-C.HISTORY_TURNS * 2:]:                            # 只带最近若干轮历史
        role = h.get("role")                                                    # 角色
        if role in ("user", "assistant"):                                       # 只接受标准角色
            msgs.append({"role": role, "content": str(h.get("content", ""))[:C.SESSION_MAX_CHARS]})  # 截断后加入
    msgs.append({"role": "user",                                              # 本轮提问
                 "content": f"【检索片段】\n{ctx}\n\n【用户问题】\n{question}"})   # 片段+问题
    return msgs                                                                 # 返回消息列表


def answer(question: str, contexts: List[Dict],                                 # 生成回答
           history: Optional[List[Dict]] = None) -> str:                        # 历史可选
    """检索片段 → 回答；无片段时走"无上下文"分支，明确告知未检索到依据。"""
    if not contexts:                                                            # 没有任何片段
        lang_en = _looks_english(question)                                      # 判断提问语言
        return ("No relevant content was found in the two prospectuses for this question."
                if lang_en else "未在两份招股说明书中检索到相关内容，无法回答该问题。")  # 明确说未找到
    return chat(build_answer_messages(question, contexts, history))             # 正常生成


def _looks_english(text: str) -> bool:                                          # 粗略判断是否英文提问
    """按 ASCII 字母占比判断：英文提问时占比通常很高，中文提问时很低。"""
    t = text or ""                                                              # 防空
    letters = sum(1 for ch in t if ch.isascii() and ch.isalpha())               # ASCII 字母数
    cjk = sum(1 for ch in t if "一" <= ch <= "鿿")                      # 中日韩字符数
    return letters > cjk * 2 and letters >= 5                                   # 字母远多于汉字才算英文提问
