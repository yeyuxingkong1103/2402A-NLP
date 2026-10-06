"""角色模块"""
from src.role.prompt_template import PromptTemplate, Role, get_available_roles
from src.role.manager import RoleManager, role_manager

__all__ = [
    'PromptTemplate', 'Role', 'get_available_roles',
    'RoleManager', 'role_manager'
]
