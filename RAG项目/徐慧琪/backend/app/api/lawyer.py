"""律师侧路由：问答、只检索、历史案件接口位（设计 §四、§六）。

**本模块与 public.py 的差别落在路由表上，不在分支里**（设计 §六 原文：公众侧
「根本没有」律师侧路由 —— 不是「存在但拒绝」）。具体到文件层面：三条路径都挂了
`Depends(deps.current_user)`，而公众侧的问答/推荐两条完全没有鉴权依赖；反过来
公众侧也没有任何一条路径在本模块里。这样「谁能打哪条」由**路径本身**决定，
不需要在同一个处理函数里按身份分支 —— 分支写法漏一条 if 就是一次越权，
而路径缺失时探测者拿到的是与「没有这个接口」逐字节相同的 404（§二 第 9 条）。

历史案件（`/cases/search`）注册但返回 501：设计 §二 第 6 条要的是「接口清单稳定、
前端可提前对接」，所以它**挂鉴权但不实现** —— 未认证的人拿到 404（路径不暴露），
已认证的人拿到 501 与写明的 reason（前端据此知道什么时候能用上）。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

# retrieve 用 from-import 绑到本模块：接线用例要注入替身（真链路加载 2GB 模型，
# 「参数有没有递对」不该为此付费），打桩点是 app.api.lawyer.retrieve。若改成
# `from app.retrieval import pipeline` + `pipeline.retrieve(...)`，打桩点就变成
# 检索模块的**全局**属性 —— 同一进程里别的用例（并发或先后）会一起被替换掉，
# 而打桩的意图只是「这条端点用哪个检索函数」
from app.api import deps, schemas
from app.core import errors
from app.core.security import CurrentUser
from app.generation.profiles import SIDE_INTERNAL
from app.retrieval.pipeline import RERANK_OUTPUT_TOPK, retrieve

router = APIRouter(prefix="/api/v1", tags=["lawyer"])


@router.post("/qa", response_model=schemas.QAAnswer)
def lawyer_qa(body: schemas.QuestionRequest, request: Request,
              _user: CurrentUser = Depends(deps.current_user)) -> dict:
    """律师侧问答（FR-7.1）：任意已登录角色都可问，不再按角色细分。

    侧别**显式**传 SIDE_INTERNAL：Answerer.answer 的签名默认值恰好也是
    "internal"，所以漏传**今天不会红**（拿到的仍是律师侧提示词）—— 但那是
    「默认值恰好等于想要的值」，两者一旦分家（默认改成 public 或加第三个侧别），
    律师侧会静默用上公众侧提示词，而它少了「可给出条款号规范引用格式」这条
    （FR-7.4 的原料）。故测试断言的是**调用参数**（answerer.calls 里的元组），
    不是响应内容 —— 内容分不出侧别，参数才分得出。

    response_model 用 `QAAnswer`（律师侧基类）：它天然没有 `lawyers`/`fee_range`
    两个键 —— 设计 §四 的红线是「不是置空，是根本不出现」（律师侧不经过
    recommend/ 的红线在契约层也要成立）。用 `PublicQAAnswer` + 置空是另一种做法，
    那种做法要靠每个出口记得传对模型，而漏传的表现是两个键悄悄出现、没有红灯。

    故障与无依据的区分交给 `errors.raise_if_answer_failed` 一个入口（任务 3
    交接第 2 条）：本函数不判 `status == "error"` —— 那是第二份口径，
    两份口径分家时表现为「某个端点偶尔把故障答成没依据」。
    """
    result = deps.services_of(request).answerer.answer(body.question, SIDE_INTERNAL)
    errors.raise_if_answer_failed(result)
    return schemas.lawyer_qa_payload(result, deps.request_id_of(request))


@router.post("/search", response_model=schemas.SearchResponse)
def lawyer_search(body: schemas.SearchRequest, request: Request,
                  _user: CurrentUser = Depends(deps.current_user)) -> dict:
    """律师侧检索（FR-7.1 的「检索」半边）：只检索，**不调 LLM**。

    这条端点存在的理由就是「便宜、可反复打」：检索链路（抽号 → 取条 → 编码 →
    粗排 → 回填 → 精排）一次约 2 秒且不花钱，而完整问答还要一次 DeepSeek 生成。
    律师在正式提问前反复调整说法、比对召回，走的就是这条路径。

    top_k 缺省取 pipeline.RERANK_OUTPUT_TOPK（**默认值只有那一处字面量**），
    上限在 schema 层卡（超限 400）—— 见 SearchRequest 与 MAX_TOP_K 的注释。
    四个重资源全部来自 app.state（本节不装配任何东西，§三 定调第 2 条）。
    """
    services = deps.services_of(request)
    top_k = body.top_k if body.top_k is not None else RERANK_OUTPUT_TOPK
    result = retrieve(body.question, conn=services.conn,
                      client=services.client, encoder=services.encoder,
                      reranker=services.reranker, top_k=top_k)
    return schemas.search_payload(result, deps.request_id_of(request))


@router.post("/cases/search")
def search_cases(_user: CurrentUser = Depends(deps.current_user)) -> dict:
    """历史案件检索（FR-5.3）的**接口位**：已注册、已鉴权，但返回 501（设计 §二 第 6 条）。

    为什么注册一个永不成功的端点：接口清单要稳定 —— 前端可以提前把入口接上、
    按 501 显示「即将开放」，而不是等案件表落地后再改一次路由与契约；同时它让
    「律师侧专用路径」这一组在路由表里今天就是完整的（未认证探测这三条得到的
    都是同一种 404，若少了这一条，探测者能数出「现在只有两条」）。

    **不解析请求体**（有意）：案件检索的输入形状要等案件表落地才能定，现在钉一个
    形状出来就是猜；而声明了 body 参数之后，畸形请求体会先撞上 400 校验失败，
    把 501 这个唯一有信息量的答复盖掉 —— 一个「未实现」的端点对已认证调用方
    应当只有一种结局。鉴权依赖**要挂**：它属于 §六 里「律师侧专用路径」那一组，
    未认证时必须是 404 而不是 501（未认证的人不该知道这个功能存在）。

    reason 会原样进响应（FeatureNotImplemented 是唯一打开 public_detail 的类型）：
    它由服务端写死、不含用户输入、不含下游异常原文，与 503 那种「下游把原文
    塞进异常」不同类（设计 §七 第 4 行要求「reason 写明」）。
    """
    raise errors.FeatureNotImplemented(
        "历史案件检索尚未实现（FR-5.3）：案件表与 case_chunks 集合尚不存在，"
        "本轮只保留接口位，返回 501 表示功能未开放")
