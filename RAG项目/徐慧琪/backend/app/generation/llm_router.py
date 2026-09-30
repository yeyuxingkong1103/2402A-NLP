"""按侧返回 LLM 客户端。

存在的理由（技术方案 1.1 两条链路）：律师侧的数据不出域，只能用本地 Ollama；
公众侧允许调用白名单内的外部 API，且外部模型质量更强。两者的差异只在
"用哪个客户端"，本模块把它收成一处，上层（chain/answer）只认 side 字符串。

密钥只从环境变量读，不写进代码、不进日志。
"""
from __future__ import annotations

import os

from app.generation.profiles import SIDE_INTERNAL, SIDE_PUBLIC

OLLAMA_BASE_URL = "http://127.0.0.1:11434"
OLLAMA_MODEL = "qwen2.5:3b"

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"
# 环境变量名由用户指定（2026-09-23）：密钥已在本机用户环境变量里，名为 api
DEEPSEEK_KEY_ENV = "api"

# 技术方案 6.1：两侧共同低温度 0~0.2，引用要忠实、不要创作
TEMPERATURE = 0.1
# 超时必须有：本地 3b 卡死或外部 API 挂起时，不能让整条问答链无限等
REQUEST_TIMEOUT_S = 30


class MissingAPIKeyError(RuntimeError):
    """公众侧缺密钥。刻意不降级到本地模型——静默换模型会让
    "公众侧走外部 API"这条已定稿口径（技术方案 9.6）名存实亡，
    排查时也看不出回答质量下降的原因。"""


def get_llm(side: str):
    """返回该侧应使用的聊天模型。"""
    if side == SIDE_INTERNAL:
        # 延迟导入：两侧客户端各属独立发行包，上层导入路由时不该强制加载两边
        from langchain_ollama import ChatOllama
        # format="json" 让 Ollama 在解码层就约束成 JSON，比事后解析可靠得多。
        # 超时不能走 timeout= 形参：ChatOllama 的 extra="ignore" 会把它悄悄丢掉，
        # 只有 client_kwargs 才会真正传给底层 httpx 客户端（实测确认过）
        return ChatOllama(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL,
                          temperature=TEMPERATURE, format="json",
                          client_kwargs={"timeout": REQUEST_TIMEOUT_S})
    if side == SIDE_PUBLIC:
        key = os.environ.get(DEEPSEEK_KEY_ENV, "").strip()
        if not key:
            # 报错只提变量名，不回显任何取值——密钥进日志等于泄露
            raise MissingAPIKeyError(
                f"环境变量 {DEEPSEEK_KEY_ENV} 未设置或为空；公众侧必须走外部 API")
        from langchain_openai import ChatOpenAI
        # timeout 是 request_timeout 字段的别名，实测两者等价（httpx 客户端
        # 都收到 30.0）；这里按 brief 原样用别名
        return ChatOpenAI(base_url=DEEPSEEK_BASE_URL, model=DEEPSEEK_MODEL,
                          api_key=key, temperature=TEMPERATURE,
                          timeout=REQUEST_TIMEOUT_S,
                          # DeepSeek 的 json 模式要求提示词里出现 json 字样，
                          # profiles 的格式说明已满足
                          model_kwargs={"response_format": {"type": "json_object"}})
    raise ValueError(f"未知的侧别：{side!r}")
