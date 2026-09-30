"""src/schemas/roleplay.py —— 角色扮演业务的请求/响应模型。

在链路中的位置：
    src/api/routers/role.py    用 RoleCard（角色卡校验）
    src/api/routers/session.py 用 SessionCreate（建会话）
    src/api/routers/chat.py    用 ChatRequest（对话请求）

三个模型覆盖了"角色扮演"这条业务线的三个关键入参：
    定义角色（RoleCard）→ 开启对话（SessionCreate）→ 提问（ChatRequest）
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class RoleCard(BaseModel):
    """角色卡（角色扮演的核心数据结构）。

    这是本项目"角色"概念的完整定义，字段可分成四组：

    一、人设描述（进 L1 角色人设层，决定模型以什么身份说话）
        role_id           角色标识，唯一。用于检索过滤（Milvus 的 role_id 字段）
        name              角色名
        avatar            头像（前端展示用）
        identity          身份设定（"你是一名执业十年的律师"）
        personality       性格特征
        tone              说话语气
        catchphrase       口头禅（让角色更有辨识度）
        knowledge_scope   知识范围（告诉模型该在哪个领域内回答）
        taboo_topics      禁忌话题（不该聊或该回避的内容）
        output_format     输出格式要求（如"分点回答"）
        opening_line      开场白（新建会话时的第一句话）
        disclaimer        免责声明
                          默认 "信息仅供参考。" —— 安全相关的文案要有非空默认值，
                          否则模型可能在专业领域给出无免责声明的断言

    二、知识库绑定
        bound_kb          限定检索范围（哪些文档来源），空列表 = 全库
                          传给 src/online/retriever.py 的 _expr 做 doc_source 过滤

    三、生成参数
        temperature       采样温度，默认 0.4（比知识问答的 0.2 更有表达变化）
        top_k             召回条数，默认 12
        rerank_top_k      精排后保留条数，默认 6

    四、检索质量门限
        similarity_threshold  相似度阈值，默认 0.25
                              注意这个默认值（0.25）与全局配置里的
                              settings.similarity_threshold 是**两个独立的默认值**，
                              角色卡显式给出时会覆盖全局配置。

    为什么大部分字段都有默认值：
        减少编写角色卡的门槛 —— 只想定义"身份+语气"时也能建出一个可用角色，
        不必填满十几个字段。必填的只有 role_id 和 name。

    注意 kb_chunks 与 role 表、以及 Milvus 三者的一致性：
        role_id 会作为 Milvus 记录的过滤字段，
        所以它既是数据库里的业务标识，也是向量库里的隔离键 —— 改它会切断已建的数据关联。
    """

    role_id: str
    name: str
    avatar: str = ""
    identity: str = ""
    personality: str = ""
    tone: str = ""
    catchphrase: str = ""
    knowledge_scope: str = ""
    taboo_topics: str = ""
    output_format: str = ""
    opening_line: str = ""
    bound_kb: list[str] = Field(default_factory=list)
    disclaimer: str = "信息仅供参考。"
    temperature: float = 0.4
    top_k: int = 12
    rerank_top_k: int = 6
    similarity_threshold: float = 0.25


class SessionCreate(BaseModel):
    """新建会话的请求体。

    字段：
        role_id: 该会话绑定的角色（会话一旦建立，角色不可更改）
        title:   会话标题，默认空 —— 由首条消息自动补全更符合直觉
    """

    role_id: str
    title: str = ""


class ChatRequest(BaseModel):
    """对话请求体。

    字段：
        session_id: 会话 id（必填）。会话决定用哪个角色、以及短期记忆取自哪里
        message:    用户本轮消息（必填）
        stream:     是否用 SSE 流式返回，默认 True

    默认 stream=True 的理由：
        流式是本项目的主要交互形态（打字机效果、链路逐步点亮），
        默认开流能让前端"不传这个字段"就得到最好的体验。
        非流式（stream=false）留给脚本调用和自动化测试 ——
        那里不需要打字机效果，一次拿到完整结果更方便断言。
    """

    session_id: int
    message: str
    stream: bool = True
