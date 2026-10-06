# -*- coding: utf-8 -*-
"""数据模型模块：定义 HTTP 接口的请求与响应结构，供 app.py 与前端对接使用。"""

from typing import List, Optional             # 导入类型注解工具，用于声明字段类型
from pydantic import BaseModel, Field         # 导入 Pydantic 基类与字段描述工具


# ===================== 健康检查与通用 =====================

class HealthResp(BaseModel):                  # 健康检查响应
    """健康检查响应：三项依赖服务的连通状态。"""
    status: str = Field("ok", description="服务状态")            # 固定为 ok
    milvus: bool = Field(False, description="Milvus 是否可连")    # 向量库状态
    mysql: bool = Field(False, description="MySQL 是否可连")      # 关系库状态
    redis: bool = Field(False, description="Redis 是否可连")      # 缓存状态


class MsgResp(BaseModel):                     # 通用消息响应
    """通用消息响应，用于只需返回一句提示的接口。"""
    msg: str = Field("", description="提示信息")                  # 提示内容


class ErrorResp(BaseModel):                   # 统一错误响应
    """统一错误响应，异常时返回，不暴露堆栈。"""
    error: str = Field("", description="错误说明")                # 错误描述


# ===================== 认证与用户 =====================

class RegisterReq(BaseModel):                 # 注册请求
    """注册请求。"""
    username: str = Field(..., description="用户名", min_length=1, max_length=64)   # 用户名
    password: str = Field(..., description="密码明文", min_length=6, max_length=128)   # 密码


class RegisterResp(BaseModel):                # 注册响应
    """注册响应。"""
    user_id: int = Field(..., description="新用户编号")           # 用户编号
    username: str = Field(..., description="用户名")              # 用户名
    msg: str = Field("注册成功", description="提示信息")          # 提示


class LoginReq(BaseModel):                    # 登录请求
    """登录请求。"""
    username: str = Field(..., description="用户名")              # 用户名
    password: str = Field(..., description="密码明文")            # 密码


class LoginResp(BaseModel):                   # 登录响应
    """登录响应。"""
    user_id: int = Field(..., description="用户编号")             # 用户编号
    username: str = Field(..., description="用户名")              # 用户名
    token: str = Field(..., description="令牌（教学版：user_id 哈希）")   # 令牌


class UserInfoResp(BaseModel):                # 用户信息响应
    """用户信息响应。"""
    user_id: int = Field(..., description="用户编号")             # 用户编号
    username: str = Field(..., description="用户名")              # 用户名
    created_at: str = Field("", description="创建时间")           # 创建时间


# ===================== 角色 =====================

class RoleItem(BaseModel):                    # 单个角色
    """角色条目。"""
    role_id: int = Field(..., description="角色编号")             # 角色编号
    role_name: str = Field(..., description="角色名")             # 角色名
    description: str = Field("", description="角色说明")          # 角色说明


class RoleListResp(BaseModel):                # 角色列表响应
    """角色列表响应。"""
    roles: List[RoleItem] = Field(default_factory=list, description="角色列表")   # 角色数组


# ===================== 会话与消息 =====================

class CreateConversationReq(BaseModel):       # 创建会话请求
    """创建会话请求。"""
    user_id: int = Field(..., description="所属用户编号")           # 用户编号
    role_id: int = Field(..., description="使用的角色编号")         # 角色编号
    title: str = Field("新会话", description="会话标题")            # 会话标题


class CreateConversationResp(BaseModel):      # 创建会话响应
    """创建会话响应。"""
    conversation_id: int = Field(..., description="新会话编号")     # 会话编号


class ConversationItem(BaseModel):            # 单个会话
    """会话条目。"""
    conversation_id: int = Field(..., description="会话编号")       # 会话编号
    role_name: str = Field("", description="角色名")               # 角色名
    title: str = Field("", description="会话标题")                  # 标题
    created_at: str = Field("", description="创建时间")             # 创建时间


class ConversationListResp(BaseModel):        # 会话列表响应
    """会话列表响应。"""
    conversations: List[ConversationItem] = Field(default_factory=list, description="会话列表")   # 会话数组


class MessageItem(BaseModel):                 # 单条消息
    """消息条目。"""
    message_id: int = Field(..., description="消息编号")            # 消息编号
    role: str = Field(..., description="说话方：user / assistant")   # 说话方
    content: str = Field("", description="消息正文")                # 正文
    created_at: str = Field("", description="创建时间")             # 创建时间


class MessageListResp(BaseModel):             # 消息列表响应
    """消息列表响应。"""
    messages: List[MessageItem] = Field(default_factory=list, description="消息列表")   # 消息数组


# ===================== 问答 =====================

class ChatReq(BaseModel):                     # 问答请求
    """问答请求。"""
    user_id: int = Field(..., description="用户编号")               # 用户编号
    role_id: int = Field(..., description="角色编号")               # 角色编号
    conversation_id: int = Field(..., description="会话编号")       # 会话编号
    query: str = Field(..., description="用户问题", min_length=1)   # 问题
    stream: bool = Field(False, description="是否流式返回")         # 流式开关


class SourceItem(BaseModel):                  # 来源条目
    """检索来源条目。"""
    source: str = Field("", description="来源文件名")               # 文件名
    summary: str = Field("", description="片段摘要")                # 摘要
    score: float = Field(0.0, description="相关度分数")              # 分数


class ChatResp(BaseModel):                    # 问答响应（非流式）
    """问答响应（非流式）。"""
    answer: str = Field("", description="回答正文")                 # 答案
    sources: List[SourceItem] = Field(default_factory=list, description="引用来源")   # 来源清单
    rewritten_query: str = Field("", description="改写后的检索问题")   # 改写结果
    cache_hit: bool = Field(False, description="是否命中缓存")       # 缓存命中
    conversation_id: int = Field(0, description="会话编号")          # 会话编号
    usage: dict = Field(default_factory=dict, description="token 用量")   # 用量
    long_memory_used: int = Field(0, description="本轮使用到的长期记忆条数")   # 长期记忆条数


# ===================== 知识库 =====================

class UploadResp(BaseModel):                  # 上传响应
    """上传响应：只解析不入库。"""
    file_name: str = Field("", description="文件名")                # 文件名
    status: str = Field("", description="处理状态")                 # 状态
    pages: int = Field(0, description="解析出的页数")                # 页数
    chunks: int = Field(0, description="切分出的块数")               # 块数


class KbStatusResp(BaseModel):                # 知识库状态响应
    """知识库状态响应。"""
    collection: str = Field("", description="Milvus 集合名")        # 集合名
    row_count: int = Field(0, description="集合内向量条数")          # 条数
    last_update: str = Field("", description="增强结果文件更新时间")   # 更新时间


# ===================== 评测 =====================

class EvalSummary(BaseModel):                 # 指标汇总
    """指标汇总：四条评测指标的均分。"""
    # 四项都可空：某项在所有样本上都算不出来时返回 null，而不是编一个 0 出来
    faithfulness: Optional[float] = Field(None, description="忠实度均分，全算不出时为 null")       # 忠实度
    answer_relevancy: Optional[float] = Field(None, description="答案相关度均分，同上")            # 相关度
    context_precision: Optional[float] = Field(None, description="上下文精确率均分，同上")         # 精确率
    context_recall: Optional[float] = Field(None, description="上下文召回率均分，同上")            # 召回率


class EvalDetail(BaseModel):                  # 单条评测明细
    """单条评测明细：一条问题的问答与四项指标分数。"""
    question: str = Field("", description="评测问题")               # 问题
    answer: str = Field("", description="系统回答")                 # 系统答案
    ground_truth: str = Field("", description="参考答案")            # 参考答案
    # 单条同样可空：这一条上该项没算出来（如评判超时）时为 null，前端显示 "-"
    faithfulness: Optional[float] = Field(None, description="忠实度，未算出为 null")        # 忠实度
    answer_relevancy: Optional[float] = Field(None, description="答案相关度，未算出为 null")   # 相关度
    context_precision: Optional[float] = Field(None, description="上下文精确率，未算出为 null")   # 精确率
    context_recall: Optional[float] = Field(None, description="上下文召回率，未算出为 null")      # 召回率


class EvalResp(BaseModel):                    # 评测结果响应
    """评测结果响应：含时间戳、模型名、四项指标汇总与逐条明细。"""
    timestamp: int = Field(0, description="评测时间戳（秒）")             # 评测时间
    model: str = Field("", description="评估用大模型名称")                # 评估模型
    embedding: str = Field("", description="评估用向量模型名称")           # 评估向量模型
    summary: EvalSummary = Field(default_factory=EvalSummary, description="四项指标均分")   # 汇总
    details: List[EvalDetail] = Field(default_factory=list, description="逐条评测明细")      # 明细


class KbRebuildResp(BaseModel):               # 知识库重建响应
    """知识库重建响应：后台任务立即返回。"""
    status: str = Field("started", description="任务状态")           # 状态
    msg: str = Field("", description="提示信息")                    # 提示


# ===================== 缓存 =====================

class ClearCacheResp(BaseModel):              # 清理缓存响应
    """清理缓存响应：返回本次清理的明细，供前端展示。"""
    msg: str = Field("", description="提示信息")                     # 提示内容
    cache_keys: int = Field(0, description="删除的答案缓存键个数")    # 缓存键个数
    short_memory: bool = Field(False, description="短期记忆是否已清除")   # 短期记忆标记


class CacheStatsResp(BaseModel):              # 缓存统计响应
    """缓存统计响应：答案缓存总量、按用户分组明细、短期记忆用户数。"""
    total_keys: int = Field(0, description="cache:rag:* 答案缓存键总数")   # 缓存键总数
    by_user: dict = Field(default_factory=dict,   # 分组明细：user_id 字符串 → 该用户的键数量
                          description="按用户分组的键数量，键为 user_id 字符串")   # 字段说明
    short_memory_users: int = Field(0, description="存在短期记忆 mem:short:* 的用户数")   # 用户数
