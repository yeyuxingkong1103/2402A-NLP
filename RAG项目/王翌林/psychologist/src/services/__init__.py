"""服务层包（不在此处导入子模块，避免循环导入）。

服务层是"业务逻辑层"：对上层 API 提供用例级能力（注册登录、会话、双层记忆、RAG 问答、
知识库、危机干预、评测等），向下编排 core（配置/日志/异常/安全）与 rag（检索/提示词）模块。
这里刻意不做 `from . import user_service` 之类的聚合导入：各 service 之间存在相互依赖
（如 user_service 依赖 persona_service、memory_service 依赖 conversation_service），
一旦在 __init__ 里提前 import，就会在模块加载期形成循环导入，导致 ImportError。
"""