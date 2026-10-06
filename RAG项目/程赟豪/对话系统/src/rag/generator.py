"""生成回答模块"""
import re
from typing import List, Dict, Any, Optional, Iterator
from dataclasses import dataclass

from src.config import config
from src.utils.logger import logger


@dataclass
class GenerationResult:
    """生成结果"""
    text: str
    source_docs: List[Dict[str, Any]]
    finish_reason: str = "stop"
    usage: Optional[Dict[str, Any]] = None


class AnswerGenerator:
    """答案生成器"""
    
    def __init__(self, llm_client: Optional[Any] = None):
        self.llm_client = llm_client
        self.max_tokens = config.get('llm.api.max_tokens', 2048)
        self.temperature = config.get('llm.api.temperature', 0.7)
    
    def generate(
        self,
        query: str,
        context_docs: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> GenerationResult:
        """
        生成回答
        
        Args:
            query: 用户问题
            context_docs: 检索到的上下文文档
            system_prompt: 系统提示词
            conversation_history: 对话历史
        
        Returns:
            GenerationResult
        """
        if not context_docs:
            return GenerationResult(
                text="抱歉，我没有找到相关的知识来回答这个问题。",
                source_docs=[],
                finish_reason="no_context"
            )
        
        # 构建提示词
        prompt = self._build_prompt(query, context_docs, conversation_history)
        
        # 调用大模型
        if self.llm_client:
            response = self.llm_client.chat(
                messages=[
                    {"role": "system", "content": system_prompt or self._default_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            text = response["content"]
            finish_reason = response.get("finish_reason", "stop")
            usage = response.get("usage")
        else:
            # 模拟生成
            text = self._fallback_generate(query, context_docs)
            finish_reason = "no_llm"
            usage = None
        
        # 后处理
        text = self._post_process(text)
        
        return GenerationResult(
            text=text,
            source_docs=context_docs,
            finish_reason=finish_reason,
            usage=usage
        )
    
    def generate_stream(
        self,
        query: str,
        context_docs: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> Iterator[str]:
        """流式生成"""
        if not context_docs:
            yield "抱歉，我没有找到相关的知识来回答这个问题。"
            return
        
        prompt = self._build_prompt(query, context_docs, conversation_history)
        
        if self.llm_client:
            for chunk in self.llm_client.chat_stream(
                messages=[
                    {"role": "system", "content": system_prompt or self._default_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            ):
                yield chunk
        else:
            # 模拟流式输出
            text = self._fallback_generate(query, context_docs)
            for char in text:
                yield char
    
    def _build_prompt(
        self,
        query: str,
        context_docs: List[Dict[str, Any]],
        history: Optional[List[Dict[str, Any]]]
    ) -> str:
        """构建提示词"""
        # 构建上下文
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            source = doc.get('source', 'unknown')
            page = doc.get('page_number', '')
            text = doc.get('text', '')
            context_parts.append(f"[文档{i}]\n来源: {source} (第{page}页)\n内容: {text}")
        
        context = "\n\n".join(context_parts)
        
        # 构建历史
        history_text = ""
        if history:
            history_msgs = []
            for msg in history[-5:]:  # 最近5轮
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                if content:
                    history_msgs.append(f"{role}: {content}")
            history_text = "\n".join(history_msgs)
        
        # 构建完整提示词
        prompt = f"""基于以下参考资料回答用户问题。如果资料中没有相关信息，请如实说明。

## 参考资料:
{context}

"""
        if history_text:
            prompt += f"""## 对话历史:
{history_text}

"""
        
        prompt += f"""## 当前问题:
{query}

请根据参考资料给出回答，回答时注明参考来源。"""
        
        return prompt
    
    def _default_system_prompt(self) -> str:
        """默认系统提示词"""
        return """你是一个专业的AI助手，擅长根据给定的参考资料回答用户的问题。
请确保回答准确、简洁，并在回答中注明参考来源。
如果资料中没有相关信息，请如实告知用户。"""
    
    def _fallback_generate(self, query: str, docs: List[Dict]) -> str:
        """备用生成（无大模型时）"""
        return f"根据检索到的资料，我来回答您的问题：\n\n" + "\n\n".join([
            f"参考 {i+1}: {doc.get('text', '')[:200]}..." 
            for i, doc in enumerate(docs[:3])
        ])
    
    def _post_process(self, text: str) -> str:
        """后处理"""
        # 去除多余空白
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        # 去除特殊标记
        text = re.sub(r'\[闪避\]', '', text)
        
        return text.strip()
