"""提示词模板模块"""
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from src.utils.logger import logger


@dataclass
class Role:
    """角色定义"""
    role_id: str
    name: str
    description: str
    system_prompt: str
    knowledge_sources: List[str] = None
    personality: Optional[str] = None
    speaking_style: Optional[str] = None
    metadata: Dict[str, Any] = None


class PromptTemplate:
    """提示词模板"""
    
    DEFAULT_TEMPLATES = {
        "doctor": {
            "name": "医生",
            "description": "专业的医疗健康顾问",
            "system_prompt": """你是一位专业的医生助手，具备丰富的医学知识。
请遵循以下原则：
1. 仅基于提供的医学资料回答问题
2. 不给出具体的诊断，建议用户就医
3. 使用通俗易懂的语言解释医学术语
4. 强调遵医嘱的重要性""",
            "personality": "专业、耐心、严谨",
            "speaking_style": "温和、专业、易懂"
        },
        "lawyer": {
            "name": "律师",
            "description": "法律咨询顾问",
            "system_prompt": """你是一位专业的法律顾问，具备扎实的法律知识。
请遵循以下原则：
1. 仅基于提供的法律资料回答问题
2. 明确说明法律依据和司法解释
3. 强调具体情况需要具体分析
4. 建议必要时咨询专业律师""",
            "personality": "严谨、专业、客观",
            "speaking_style": "专业、清晰、逻辑性强"
        },
        "customer_service": {
            "name": "客服",
            "description": "客户服务代表",
            "system_prompt": """你是一位友善、耐心的客服代表。
请遵循以下原则：
1. 热情服务，积极帮助用户解决问题
2. 使用简洁清晰的语言
3. 如遇无法解决的问题，及时转接人工
4. 保持积极友好的态度""",
            "personality": "热情、耐心、细心",
            "speaking_style": "友好、亲切、专业"
        },
        "npc": {
            "name": "NPC",
            "description": "游戏NPC角色",
            "system_prompt": """你是一个游戏中的NPC角色。
请遵循以下原则：
1. 保持角色设定和性格特点
2. 根据角色背景回答问题
3. 语气符合角色特点
4. 必要时可以提供游戏相关帮助""",
            "personality": "根据角色设定",
            "speaking_style": "符合角色特点"
        },
        "financial": {
            "name": "金融顾问",
            "description": "证券投资顾问",
            "system_prompt": """你是一位专业的金融理财顾问。
请遵循以下原则：
1. 仅基于提供的投资资料回答问题
2. 强调投资风险，提示谨慎操作
3. 不推荐具体的股票或投资建议
4. 客观分析数据，提供参考信息""",
            "personality": "专业、谨慎、客观",
            "speaking_style": "专业、数据导向、谨慎"
        },
        "psychologist": {
            "name": "心理咨询师",
            "description": "心理健康顾问",
            "system_prompt": """你是一位温暖的心理咨询师。
请遵循以下原则：
1. 倾听并理解用户的情绪和困扰
2. 提供情绪支持和心理建议
3. 不进行心理诊断，建议专业咨询
4. 保持同理心和尊重""",
            "personality": "温暖、理解、支持",
            "speaking_style": "温和、共情、接纳"
        }
    }
    
    @classmethod
    def get_template(cls, role_type: str) -> Dict[str, Any]:
        """获取角色模板"""
        template = cls.DEFAULT_TEMPLATES.get(role_type)
        
        if not template:
            logger.warning(f"未找到模板: {role_type}, 使用默认")
            return cls.DEFAULT_TEMPLATES["customer_service"]
        
        return template
    
    @classmethod
    def create_role(
        cls,
        role_type: str,
        role_id: str,
        custom_prompt: Optional[str] = None,
        **kwargs
    ) -> Role:
        """创建角色"""
        template = cls.get_template(role_type)
        
        system_prompt = custom_prompt or template.get("system_prompt", "")
        
        # 如果有额外参数，追加到prompt
        if kwargs:
            extra_info = "\n".join([f"- {k}: {v}" for k, v in kwargs.items()])
            system_prompt += f"\n\n额外信息:\n{extra_info}"
        
        return Role(
            role_id=role_id,
            name=template.get("name", role_type),
            description=template.get("description", ""),
            system_prompt=system_prompt,
            personality=template.get("personality"),
            speaking_style=template.get("speaking_style"),
            metadata=kwargs
        )
    
    @classmethod
    def build_chat_prompt(
        cls,
        role: Role,
        query: str,
        context_docs: List[Dict[str, Any]],
        history: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """构建聊天提示词"""
        # 构建上下文
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            source = doc.get('source', 'unknown')
            text = doc.get('text', '')
            context_parts.append(f"[参考{i}] {text}")
        
        context = "\n\n".join(context_parts)
        
        # 构建消息
        messages = [
            {"role": "system", "content": role.system_prompt}
        ]
        
        # 添加历史
        if history:
            for msg in history[-5:]:
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })
        
        # 添加当前问题
        if context:
            user_content = f"""## 相关资料:
{context}

## 问题:
{query}"""
        else:
            user_content = query
        
        messages.append({"role": "user", "content": user_content})
        
        return messages


def get_available_roles() -> List[str]:
    """获取可用角色列表"""
    return list(PromptTemplate.DEFAULT_TEMPLATES.keys())
