"""本地大模型模块"""
from typing import Dict, Any, Optional, Iterator
import requests
from openai import OpenAI

from src.config import config
from src.utils.logger import logger


class LocalLLM:
    """本地大模型客户端"""
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: str = "not-needed"
    ):
        llm_config = config.get_llm_config()
        
        self.base_url = base_url or llm_config.get('base_url', 'http://localhost:8000/v1')
        self.model = model or llm_config.get('model', 'qwen2.5')
        self.temperature = llm_config.get('temperature', 0.7)
        self.max_tokens = llm_config.get('max_tokens', 2048)
        
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=api_key
        )
        
        logger.info(f"本地大模型初始化: {self.model} @ {self.base_url}")
    
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
            logger.error(f"本地大模型调用失败: {e}")
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
    
    def embed(self, texts: list) -> list:
        """文本嵌入（如果模型支持）"""
        # 本地模型通常不支持嵌入
        raise NotImplementedError("本地模型不支持嵌入功能")


class VLMLocalLLM(LocalLLM):
    """多模态本地大模型"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def chat_with_image(
        self,
        messages: list,
        image_url: str,
        **kwargs
    ) -> Dict[str, Any]:
        """图文聊天"""
        # 构建多模态消息
        multimodal_messages = []
        for msg in messages:
            if msg.get("role") == "user" and image_url:
                multimodal_messages.append({
                    "role": msg["role"],
                    "content": [
                        {"type": "text", "text": msg["content"]},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                })
            else:
                multimodal_messages.append(msg)
        
        return self.chat(multimodal_messages, **kwargs)
