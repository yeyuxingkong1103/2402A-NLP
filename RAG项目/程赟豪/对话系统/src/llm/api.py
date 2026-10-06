"""在线API大模型模块"""
from typing import Dict, Any, Optional, Iterator
import requests
from openai import OpenAI

from src.config import config
from src.utils.logger import logger


class APILLM:
    """在线API大模型客户端"""
    
    PROVIDER_MAP = {
        "deepseek": {
            "base_url": "https://api.deepseek.com/v1",
            "model": "deepseek-chat"
        },
        "qwen": {
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "model": "qwen-turbo"
        },
        "doubao": {
            "base_url": "https://ark.cn-beijing.volces.com/api/v3",
            "model": "doubao-pro-32k"
        },
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-4o-mini"
        }
    }
    
    def __init__(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        llm_config = config.get_llm_config() or {}
        # 在线 API 配置位于 llm.api 下
        api_config = llm_config.get('api', {}) if isinstance(llm_config, dict) else {}

        # 确定provider：参数 > llm.api.provider > 默认 deepseek
        provider = provider or api_config.get('provider') or 'deepseek'
        # 兼容 get_llm_client 传入的顶层 llm.provider=api
        if provider == 'api':
            provider = api_config.get('provider') or 'deepseek'

        # 获取API配置
        provider_config = self.PROVIDER_MAP.get(provider, self.PROVIDER_MAP['deepseek'])

        self.base_url = base_url or api_config.get('base_url') or provider_config["base_url"]
        self.model = model or api_config.get('model') or provider_config["model"]
        self.api_key = api_key or api_config.get('api_key') or ""

        if not self.api_key or self.api_key == "your-api-key":
            raise ValueError("请在配置文件中设置API Key")

        self.temperature = api_config.get('temperature', 0.7)
        self.max_tokens = api_config.get('max_tokens', 2048)

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )

        logger.info(f"API大模型初始化: {self.model} @ {self.base_url}")
    
    def chat(
        self,
        messages: list,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """聊天"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                stream=stream
            )
            
            if stream:
                return response
            else:
                return {
                    "content": response.choices[0].message.content,
                    "finish_reason": response.choices[0].finish_reason,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens
                    } if response.usage else None
                }
                
        except Exception as e:
            logger.error(f"API大模型调用失败: {e}")
            raise
    
    def chat_stream(
        self,
        messages: list,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Iterator[str]:
        """流式聊天"""
        response = self.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        )
        
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


def get_llm_client(provider: str = "auto") -> Any:
    """
    获取大模型客户端
    
    Args:
        provider: auto / local / deepseek / qwen / doubao / openai
    
    Returns:
        LLM客户端实例
    """
    if provider == "auto":
        provider = config.get('llm.provider', 'api')
    
    if provider == "local":
        from src.llm.local import LocalLLM
        return LocalLLM()
    else:
        return APILLM(provider=provider)
