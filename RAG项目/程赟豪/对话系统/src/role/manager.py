"""角色管理模块"""
from typing import Dict, Any, Optional, List
import json
from pathlib import Path

from src.config import config
from src.utils.logger import logger
from src.role.prompt_template import PromptTemplate, Role, get_available_roles


class RoleManager:
    """角色管理器"""
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path or "data/roles"
        self.roles: Dict[str, Role] = {}
        self._ensure_storage()
    
    def _ensure_storage(self):
        """确保存储目录存在"""
        Path(self.storage_path).mkdir(parents=True, exist_ok=True)
    
    def create_role(
        self,
        role_type: str,
        role_id: str,
        name: Optional[str] = None,
        custom_prompt: Optional[str] = None,
        knowledge_sources: Optional[List[str]] = None,
        **kwargs
    ) -> Role:
        """创建角色"""
        role = PromptTemplate.create_role(
            role_type=role_type,
            role_id=role_id,
            custom_prompt=custom_prompt,
            **kwargs
        )
        
        if name:
            role.name = name
        
        if knowledge_sources:
            role.knowledge_sources = knowledge_sources
        
        self.roles[role_id] = role
        self._save_role(role)
        
        logger.info(f"创建角色: {role_id} ({role.name})")
        return role
    
    def get_role(self, role_id: str) -> Optional[Role]:
        """获取角色"""
        if role_id in self.roles:
            return self.roles[role_id]
        
        # 尝试从文件加载
        return self._load_role(role_id)
    
    def list_roles(self) -> List[Dict[str, Any]]:
        """列出所有角色"""
        roles_list = []
        
        for role_id, role in self.roles.items():
            roles_list.append({
                "role_id": role.role_id,
                "name": role.name,
                "description": role.description,
                "knowledge_sources": role.knowledge_sources
            })
        
        return roles_list
    
    def update_role(
        self,
        role_id: str,
        system_prompt: Optional[str] = None,
        name: Optional[str] = None,
        **kwargs
    ) -> Optional[Role]:
        """更新角色"""
        role = self.get_role(role_id)
        
        if not role:
            logger.warning(f"角色不存在: {role_id}")
            return None
        
        if system_prompt:
            role.system_prompt = system_prompt
        
        if name:
            role.name = name
        
        for key, value in kwargs.items():
            setattr(role, key, value)
        
        self._save_role(role)
        logger.info(f"更新角色: {role_id}")
        
        return role
    
    def delete_role(self, role_id: str) -> bool:
        """删除角色"""
        if role_id in self.roles:
            del self.roles[role_id]
        
        # 删除文件
        file_path = Path(self.storage_path) / f"{role_id}.json"
        if file_path.exists():
            file_path.unlink()
            logger.info(f"删除角色: {role_id}")
            return True
        
        return False
    
    def _save_role(self, role: Role):
        """保存角色到文件"""
        file_path = Path(self.storage_path) / f"{role.role_id}.json"
        
        data = {
            "role_id": role.role_id,
            "name": role.name,
            "description": role.description,
            "system_prompt": role.system_prompt,
            "knowledge_sources": role.knowledge_sources,
            "personality": role.personality,
            "speaking_style": role.speaking_style,
            "metadata": role.metadata or {}
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _load_role(self, role_id: str) -> Optional[Role]:
        """从文件加载角色"""
        file_path = Path(self.storage_path) / f"{role_id}.json"
        
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            role = Role(
                role_id=data["role_id"],
                name=data["name"],
                description=data["description"],
                system_prompt=data["system_prompt"],
                knowledge_sources=data.get("knowledge_sources"),
                personality=data.get("personality"),
                speaking_style=data.get("speaking_style"),
                metadata=data.get("metadata")
            )
            
            self.roles[role_id] = role
            return role
            
        except Exception as e:
            logger.error(f"加载角色失败: {e}")
            return None


# 全局实例
role_manager = RoleManager()
