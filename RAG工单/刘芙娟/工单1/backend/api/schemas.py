"""传输模型。

`AskResponse` 的字段名 MUST 与 `docs/05_接口设计.md` §3.1.3 **逐字段一致**，
MUST NOT 增删改。原因：规格 SC-009 要验证的正是「后续四个模块（向量化、检索、
提示词、生成）接入时，界面侧的响应字段解析代码不需要修改」—— 若字段名在此处
漂移，这条验证就失去了对象。

本期的取值与后续不同（答案字段承载"能力未就绪"状态），但**字段集合不变**。
"""

from typing import Literal

from pydantic import BaseModel, Field

from . import DISCLAIMER, RISK_LEVEL_NONE

RiskLevel = Literal["immediate", "soon", "observe", "none"]


class AskRequest(BaseModel):
    """`POST /ask` 的请求体。

    只做「字段存在且是字符串」这一层校验。长度与空白由 `validate.py` 负责，
    理由：移到 Pydantic 校验器里会让边界值判定散落两处，而 quickstart §4 的
    四组边界用例要求判定**只此一处**。
    """

    question: str


class EmergencyInfo(BaseModel):
    """`docs/05` §3.1.3 的 `emergency` 对象。

    刻意只暴露 `triggered`。`emergency.source`（lexicon/model/both/degraded）
    与 `matched_terms` 保留在服务端日志，不进响应体 —— 前者会把内部实现细节
    固化进公开契约，后者是安全资产（`docs/05` §3.1.8）。
    """

    triggered: bool = False


class Citation(BaseModel):
    """引用片段。本期恒为空集，但结构先定下来。

    展示契约：前端 MUST 按服务端给出的顺序渲染，MUST NOT 重排序。
    排序依据 `score` 是**服务端算出来的余弦相似度**，前端重排等于把排序规则
    复刻了一遍，两份实现必然漂移且漂移时没有测试能发现。
    """

    citation_id: int
    file_name: str
    page_start: int
    page_end: int
    section: str | None = None
    block_type: str
    text: str
    score: float = Field(description="余弦相似度分数，引用片段的排序依据")


class AskResponse(BaseModel):
    """`done` 事件的载荷，逐字段复用 `docs/05` §3.1.3。

    一条契约不变量：`answer_id` MUST 与同一次提问的 `status` 事件一致。
    前端可据此断言「拿到的终帧确实属于本次提问」。
    """

    answer_id: str
    is_refusal: bool
    emergency: EmergencyInfo
    risk_level: RiskLevel
    answer_text: str
    citations: list[Citation]
    disclaimer: str


def refusal_response(
    answer_id: str, answer_text: str, citations: list[Citation] | None = None
) -> AskResponse:
    """拒答应答。`is_refusal=True`。

    `answer_text` 由调用方给出，覆盖三种情形（都是"这次没东西看"）：

        · 检索为空 —— 知识库里没有
        · 全部低于阈值 —— 有候选但不够像
        · 检索 / 生成未能进行 —— 基础设施故障

    三者共用同一段兜底话术，不在其中区分是哪一环失败（`docs/02` §344 的取向）。
    """

    return _response(answer_id, answer_text, citations, is_refusal=True)


def answer_response(
    answer_id: str, answer_text: str, citations: list[Citation] | None = None
) -> AskResponse:
    """正常应答。`is_refusal=False` —— 本系统第一次产出**有依据的成文回答**。

    ⚠️ `answer_text` MUST 是装配层拼好的完整文本（正文 + 免责声明），
    调用方 MUST NOT 传一段裸正文进来 —— 拼装只在一处发生，见 `docs/05` §3.1.5。

    ⚠️ 生成模块接入后，`risk_level` **仍然恒为 `none`**。这不是遗漏：
    `docs/05` §3.1.6 的四条规则里，第 1 条依赖 I-04（紧急判定，未实现），
    第 2–4 条要求对**原文表述**做急症/就医判断 —— 那属于 I-07 装配层的事，
    也不在本期范围内。当前 `risk_level` 因此是一个**已知空洞**：
    模型即使在前言里给出立即就医指引，界面也不会亮起风险标识。
    补它的前提是先实现 I-04 或把 §3.1.6 的表述判定落到装配层。
    """

    return _response(answer_id, answer_text, citations, is_refusal=False)


def _response(
    answer_id: str,
    answer_text: str,
    citations: list[Citation] | None,
    *,
    is_refusal: bool,
) -> AskResponse:
    """两种应答的唯一构造点。

    ⚠️ **`citations` MUST 与实际发出的 `citations` 事件保持一致。**

    这是 S9 接入时发现并修掉的一个真实缺陷：原先 `citations` 被写死成 `[]`，
    于是 `done` 终帧与 `citations` 事件说的不是同一件事。而前端
    `frontend/js/transcript.js` 的 `renderDone` 会**用终帧的 citations 重渲染
    引用区** —— 结果是用户看到的引用片段在终帧到达那一刻**被清空**。

    之所以在开发期没有立刻暴露：`citations` 事件先到、`done` 后到，两者间隔
    极短，人工肉眼很难分辨"引用闪了一下就没了"。
    """

    return AskResponse(
        answer_id=answer_id,
        is_refusal=is_refusal,
        emergency=EmergencyInfo(triggered=False),
        risk_level=RISK_LEVEL_NONE,
        answer_text=answer_text,
        citations=list(citations or []),
        disclaimer=DISCLAIMER,
    )
