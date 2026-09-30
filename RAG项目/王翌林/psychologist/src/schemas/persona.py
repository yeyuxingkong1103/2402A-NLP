"""心理医生角色（Persona）Schema。

“persona” 就是系统里的每个 AI 心理医生人设，例如“温和的认知行为疗法导师”。
一份 persona 最少要包含：
- 标识信息： persona_code（唯一英文编码）、name（显示名）、title（头衔）；
- 对话风格： therapy_type（流派）、style（语气）、methods（常用技术）、greeting（开场白）；
- 核心设定： system_prompt（系统提示词，决定 AI 的行为边界）、safety_boundary（安全红线）；
- 知识范围： knowledge_scope（限定可检索的知识库范围）；
- 模型参数： model_params（如 temperature、max_tokens 等采样参数）。

本模块的模型分工：
- PersonaBase   —— 列表/公开场景使用的基础字段（不含 system_prompt）；
- PersonaDetail —— 详情场景，在基础字段之上额外暴露 system_prompt 与 safety_boundary；
- Create/Update —— 后台管理角色的请求体；
- PersonaListResult —— 角色列表响应。
"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class PersonaBase(BaseModel):
    """角色基础信息（可安全对外暴露的部分，不含系统提示词）。"""
    model_config = ConfigDict(from_attributes=True)  # 允许从 ORM 对象直接构造

    id: int
    # 角色唯一英文编码（如 "cbt_coach"）：程序内部引用角色的稳定标识，
    # 与可变的 name 解耦，改名不会影响已有会话的关联关系
    persona_code: str
    # 角色显示名（中文名），给用户看
    name: str
    # 头衔/专业标签，如“资深认知行为治疗师”
    title: Optional[str] = None
    # 治疗流派：如 CBT（认知行为）、人本主义、精神动力学等
    therapy_type: Optional[str] = None
    # 语言风格：如“温和耐心”“理性直接”，影响提示词措辞
    style: Optional[str] = None
    # 常用技术：如“正念练习、认知重构”，供提示词选用
    methods: Optional[str] = None
    # 开场白：进入新会话时角色说的第一句话
    greeting: Optional[str] = None
    # 知识范围：限定该角色可检索的知识库/文档范围，实现知识隔离
    knowledge_scope: Optional[str] = None
    # 头像 URL
    avatar: Optional[str] = None
    # 角色启停状态：1=启用（用户可选），0=停用（下架但保留历史会话）
    status: int = 1
    # 模型采样参数（temperature、top_p、max_tokens 等）。
    # 用 Dict[str, Any] 是为了灵活支持不同厂商模型的不同参数名，避免频繁改表结构
    model_params: Optional[Dict[str, Any]] = None


class PersonaDetail(PersonaBase):
    """角色详情（继承 PersonaBase，额外包含敏感的系统级设定）。

    继承的好处：基础字段只在父类维护一份，详情/列表不会出现字段定义漂移；
    差异只体现在本类新增的两个“不对普通列表暴露”的字段上。
    """
    # 系统提示词：定义角色的身份、语气、回答边界，是 persona 的灵魂；
    # 只在详情接口返回，避免列表接口批量泄露（也节省响应体积）。
    # 这里无默认值，表示必填——一个人设没有系统提示词是不可用的
    system_prompt: str
    # 安全边界：心理场景专属，声明不越界内容（如不做医疗诊断、不推荐药物）；
    # 可选，因为部分轻量角色直接把这部分写进了 system_prompt
    safety_boundary: Optional[str] = None


class PersonaCreateRequest(BaseModel):
    """创建角色的请求体（管理端使用）。"""
    # 角色编码：2~64 字符。下限 2 保证编码有意义；
    # 上限 64 对齐数据库列宽；编码会被用作索引键，故不宜过长
    persona_code: str = Field(min_length=2, max_length=64)
    # 角色显示名：1~64 字符，不允许为空（列表里没有名字无法展示）
    name: str = Field(min_length=1, max_length=64)
    title: Optional[str] = None
    therapy_type: Optional[str] = None
    style: Optional[str] = None
    methods: Optional[str] = None
    greeting: Optional[str] = None
    # 系统提示词：创建时必填，即上架前必须先把人设写清楚
    system_prompt: str
    knowledge_scope: Optional[str] = None
    avatar: Optional[str] = None
    safety_boundary: Optional[str] = None
    model_params: Optional[Dict[str, Any]] = None
    # 新建角色默认即启用，省去创建后再点一次“上架”
    status: int = 1


class PersonaUpdateRequest(BaseModel):
    """更新角色的请求体（部分更新，PATCH 语义）。

    所有字段均为可选且默认 None：
    - 只传想改的字段，未传的字段保持原值（不是被清空）；
    - 服务端据此实现“只更新非 None 字段”的逻辑。
    注意 update 里不含 persona_code：编码是稳定标识，创建后不可修改。
    """
    # 注意 name 这里没有长度约束，而 create 时有——因为 update 走的是部分更新路径，
    # 长度合法性通常由服务层在合并后统一校验（历史实现保持一致，勿擅自改动）
    name: Optional[str] = None
    title: Optional[str] = None
    therapy_type: Optional[str] = None
    style: Optional[str] = None
    methods: Optional[str] = None
    greeting: Optional[str] = None
    system_prompt: Optional[str] = None
    knowledge_scope: Optional[str] = None
    avatar: Optional[str] = None
    safety_boundary: Optional[str] = None
    model_params: Optional[Dict[str, Any]] = None
    # 状态可选：不传即不改变上下架状态
    status: Optional[int] = None


class PersonaListResult(BaseModel):
    """角色列表响应。

    列表用 PersonaBase（不含 system_prompt），
    因此无需分页字段——角色数量少，一次全量返回即可。
    """
    items: List[PersonaBase]