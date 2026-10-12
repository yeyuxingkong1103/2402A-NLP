# 工单编号：人工智能NLP-RAG-Graph RAG优化任务
# HTTP 请求模型：只做字段声明与校验（容错 E1：空串/超长输入在此被拦截成 422），不含任何业务逻辑
from pydantic import BaseModel, Field                                          # 数据校验基类与字段约束
class Ask(BaseModel):                                                          # POST /ask 与 /ask/before 的请求体
    question: str = Field(min_length=1, max_length=500)                        # 问题文本：限长，防止超长输入打爆上下文与缓存键
    use_cache: bool = True                                                     # 是否允许命中 Redis 缓存（评估时置 false 以测真实链路）
    lang: str = "auto"                                                         # 期望作答语言：auto/zh/en，auto 时由生成层按问题语言自适应
class Trip(BaseModel):                                                         # POST /graph/text 的请求体（演示"贴文本建图"）
    text: str = Field(min_length=1, max_length=8000)                           # 待抽取的正文片段
    code: str = ""                                                             # 归属公司代码（可选，用于给抽出的边打标）
