"""DeepSeek LLM（OpenAI 兼容接口，用 langchain-openai 封装）。"""  # 模块说明
from langchain_openai import ChatOpenAI  # OpenAI 兼容的聊天模型客户端

import config  # 全局配置


def get_llm() -> ChatOpenAI:  # 工厂函数：构建 DeepSeek LLM 客户端
    if not config.deepseek_api_key:  # 未配置密钥时提前报错，给出明确提示
        raise ValueError("未配置 DEEPSEEK_API_KEY，请在 .env 或环境变量中设置。")
    return ChatOpenAI(  # 用 OpenAI 兼容协议连接 DeepSeek
        model=config.deepseek_model,  # 模型名
        api_key=config.deepseek_api_key,  # API 密钥
        base_url=config.deepseek_base_url,  # 把接口地址覆盖为 DeepSeek 的服务端
        temperature=config.llm_temperature,  # 生成温度
        max_tokens=config.llm_max_tokens,  # 最大生成 token 数
    )
