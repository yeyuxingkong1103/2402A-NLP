"""跨模块共用的异常。

单独放一个文件是为了**避免循环导入**：`EmptyAnswerError` 原先定义在 pipeline.py，
而 llm.py 也需要在「上游返回空 choices」时抛它——但 pipeline 反过来要 import llm。
放这里两边都能用；pipeline 仍然把它再导出一次，历史 import 路径不受影响。
"""
from __future__ import annotations


class EmptyAnswerError(RuntimeError):
    """模型没产出任何回答文本。

    典型成因：推理模型在思考阶段就把生成长度用光了（被 num_predict / num_ctx 截断），
    此时 content 为空、推理内容却有一两百字，OpenAI 兼容层给的 finish_reason 是
    "length"。实测形态：

        finish_reason='length'  content=0 字  reasoning=214 字

    另一类成因：上游返回 HTTP 200 但 `choices` 为空（中转、内容过滤、上游抖动）。

    这类失败**不能**静默通过：空字符串会让 /chat 返回 200、前端渲染一片空白，
    用户以为模型答了；空答案还会被写进短期记忆，污染下一轮上下文。
    故显式抛出，由接口层转成 502 / error 事件。
    """

    def __init__(self, message: str = "模型未产出回答（可能被截断），请重试"):
        super().__init__(message)
