"""公众侧路由：匿名问答与推荐律师（设计 §四、§六）。

两条端点的**共同点**是「谁都能打」：它们不定身份、也不该定 —— FR-9.1 要求
公众侧免注册（设计 §六 的无账号依赖）。本模块因此不引入任何鉴权依赖，
也不写任何一条按用户过滤的查询（三层隔离里「公众侧根本没有律师侧路由」
在本文件表现为：这里只有这两条路径）。

限流与并发上限**不在本模块**（任务 6）：那条中间件按 IP 维度拦在更外层，
本模块只管「这一条请求的输入与输出」。

边界（有意不做的两件事）：
  ①**不进 Answerer 的内部**：不判 `status == "error"`、不自己拼故障语义 ——
    故障与无依据的区分收在 errors.raise_if_answer_failed 一处（任务 3 交接）。
  ②**不装配任何重资源**：连接、模型、extras 全部来自 app.state（§三 定调第 2 条）。
"""
from __future__ import annotations

from fastapi import APIRouter, Request

from app.api import deps, schemas
from app.core import errors
from app.generation.profiles import SIDE_PUBLIC
from app.recommend import cause as cause_mod
from app.recommend import lawyers

router = APIRouter(prefix="/api/v1", tags=["public"])

# 推荐卡片数（与 attach 的默认一致）。写在这里而不是引用 attach.DEFAULT_CARD_COUNT：
# 那个常量属于问答链路的区块组装，这条端点不再走那条路（见 recommend 的注释），
# 两者若要一起改说明是需要一次裁决的口径变更 —— 引用同一常量会把那种变更静默掉
CARD_COUNT = 3


@router.post("/public/qa", response_model=schemas.PublicQAAnswer)
def public_qa(body: schemas.QuestionRequest, request: Request) -> dict:
    """公众侧问答（设计 §四）：匿名、只读、响应含律师与费用区块。

    侧别**显式**传 SIDE_PUBLIC：Answerer 的默认侧是 internal（`answer(question,
    side="internal")`），漏传的后果是公众侧被当成律师侧 —— 既拿不到附加区块
    （_extras_for 按侧短路），又用上律师侧的提示词（少了公众侧那三条合规约束）。
    两者都不会报错，只会让合规要求静默失效，故它是本函数里唯一的显式传参。

    故障（Milvus/LLM 不可用）→ 503、无依据 → 200 由 status 表达：两者都由
    raise_if_answer_failed 判，本函数不重复判（重复判就是第二份口径）。
    """
    result = deps.services_of(request).answerer.answer(body.question, SIDE_PUBLIC)
    errors.raise_if_answer_failed(result)
    return schemas.public_qa_payload(result, deps.request_id_of(request))


@router.get("/lawyers/recommend", response_model=schemas.LawyerRecommendResponse)
def recommend_lawyers(request: Request) -> schemas.LawyerRecommendResponse:
    """推荐示例律师卡片（设计 §四：「接 `lawyers` + `fees`」）。

    本端点**只接 lawyers，不跑费用链路**（fees 只到免责声明与字段口径那一层）。
    四条理由，按硬度排序：
      ①**没有问句就没有案由**，而费用链路的检索需要一段查询文本：`attach.build`
        的 search_fn 在案由为空时回退到「原问句」，这条端点两样都没有。空查询在
        fee_search 里直接返回空集 → estimate 归到 `no_corpus`（「语料未收录该案由
        的收费口径」）——那是把「我们没查」说成「语料里没有」，正是 ③a 的红线。
      ②**编一个查询串（如「律师费」）会产出一个与任何案由都不相干的价格**，而
        数字回查只保证「数字能在片段里逐字找到」，挡不住「取错片段」——③b-1 的
        终审点名过这一形态。给一个没有前提的区间，比不给更误导。
      ③**唯一合法的复用点（`extras_fn`）是问句入参、且藏在 Answerer 私有属性里**：
        要么伸手取私有属性、要么在本路由里把 `build_extras_fn` 的装配再抄一遍
        （LLM 客户端、留痕表、search_fn 三处）。后者正是设计 §三 定调第 3 条
        要消灭的「两处各存一份口径」，且分叉后表现为「CLI 有区间、接口没区间」。
      ④**成本与暴露面不匹配**：这条端点匿名、当前无限流（任务 6 才有），而费用
        链路每次要一次编码 + 一次 Milvus 检索 + 一次 DeepSeek 生成。
    代价（已知）：`fee_range` 恒为 null。它在契约里的键位保留（见 schemas），
    故将来设计层回答了「cause 从哪来」之后填充它不需要改契约。

    field 取 `cause.GENERIC_FIELD`（「通用」）：没有输入可判领域，就不猜一个；
    卡片本身带 `demo` 与「示例数据」标注（lawyers.DEMO_MODE），演示期的虚假宣传
    风险由那两处挡，不在这里补一句免责。

    匿名可读（设计 §六）：本端点不挂任何鉴权依赖，故意如此 —— 它与 /public/qa
    一样属于「公众可读的三个」，把它们也 404 化会把该公开的挡掉。
    """
    cards = lawyers.recommend(cause_mod.GENERIC_FIELD, CARD_COUNT)
    return schemas.LawyerRecommendResponse(request_id=deps.request_id_of(request),
                                          field=cause_mod.GENERIC_FIELD,
                                          lawyers=cards, fee_range=None)
