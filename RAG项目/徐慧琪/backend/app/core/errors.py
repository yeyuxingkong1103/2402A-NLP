"""接口层错误的类型与状态码归属：设计 §七 那张表的可执行形式。

存在的理由：那张表是硬规格（HTTP 状态码、通用文案、503 与 200 的区别），而「按表
映射」这件事只有收成**一处**才守得住 —— 若每个路由各写各的 `raise HTTPException(...)`，
表的第 5、6 行（故障 503 / 无依据 200）就会在十几处各判一次，判错一处只表现为
「那个端点偶尔答错状态码」，没有任何红灯。所以路由只负责举手表态（抛这个模块的
类型或调 raise_if_answer_failed），映射与文案由 main.py 的处理器按这里的表统一决定。

为什么不是 `core/` 之外的新模块（设计 §三 没有它）：错误类型必须能被 api/ 与 main.py
**同时** import。放进 main.py 会成环 —— 任务 4 起 main.py 要 import api/ 的路由，
路由再回头 import main 取异常类型就是循环导入。放 core/ 则两边都合法（设计 §三：
api/ 与 main.py 可以 import core/）。

本模块**不 import fastapi**：它只回答「哪个异常归哪个状态码」，出响应是 main.py 的事。
好处是任务 4 的路由单测可以直接 import 它而不用拉起 Web 栈，层次也清楚。
"""
from __future__ import annotations

# HTTP 状态码 → (对外错误码, 对外通用文案)。表里的七行对应设计 §七 的七行，
# 多出来的 413 见 main.py 体大小中间件的注释（那一行是本任务加的，设计 §七 未列）
PUBLIC_ERRORS: dict[int, tuple[str, str]] = {
    # 400：参数校验失败 / 问句超长（输入长度上限由任务 4 的 schema 施加）
    400: ("bad_request", "请求参数不合法"),
    # 404：未认证 / 越权（设计 §二 第 9 条：不暴露路径存在性，故不是 401/403）。
    # 未知路径同样落这一行 —— 两者对外必须不可区分
    404: ("not_found", "未找到"),
    # 401：**登录端点自己**的口令/账号不对（本任务新增，设计 §七 未列）。
    # 与 404 那条不冲突：404 保的是「不暴露**路径**存在性」，而 /auth/login 是
    # 文档里的公开入口，承认它存在没有任何信息量；反过来把它 404 化会让前端
    # 无法区分「登录失败」与「服务端没有这个接口」。文案不区分「查无此人」与
    # 「口令不对」—— 那是账号存在性的泄漏口（用例逐字节比对两者响应）
    401: ("unauthorized", "用户名或密码错误"),
    # 413：请求体超过中间件上限（本任务新增的一行，理由见 main.py）
    413: ("payload_too_large", "请求体过大"),
    # 429：限流 / 并发超限（阈值与判定是任务 6 的事，这里只归它的状态码与文案）
    429: ("rate_limited", "请求过于频繁，请稍后再试"),
    # 500：内部异常。文案刻意与 503 分开：500 是「我们坏了」，503 是「依赖坏了」，
    # 运维看日志时这两行要能一眼分开
    500: ("internal_error", "服务内部错误"),
    # 501：历史案件（设计 §二 第 6 条：注册但返回 501，reason 写明未实现）
    501: ("not_implemented", "该功能尚未实现"),
    # 503：系统故障（Milvus / LLM 不可用）。**通用文案**是硬要求：异常原文
    # （如 "milvus down"）只进日志，不给公众 —— ③b-1 真跑抓到过原文外泄
    503: ("service_unavailable", "服务暂时不可用，请稍后再试"),
}

# 表外状态码（框架自己抛的 405、将来别的 4xx）的兜底。状态码保留原样，
# 只给一个不含细节的通用文案 —— 兜底存在的意义是「未知也不会泄露」，
# 而不是「未知都报成 500」：把 405 谎报成 500 会让排查方向整个跑偏
FALLBACK_ERROR = ("error", "请求未能完成")


def public_error(status_code: int) -> tuple[str, str]:
    """给状态码取 (对外错误码, 对外文案)。表外状态码走兜底，不抛错。"""
    return PUBLIC_ERRORS.get(status_code, FALLBACK_ERROR)


def public_status(status_code: int) -> int:
    """框架抛出的状态码 → **对外**状态码。目前只有一条映射：405 → 404。

    任务 6 补的洞（Task 5 审查挖到，未认证即可利用）：`GET /api/v1/qa`（该路径只收
    POST）原先回 **405 + `Allow: POST`**，而未知路径回 404 —— 换个方法打一发就能把
    「这条路径存在」从 404 里分出来，`Allow` 还把正确方法一并奉上，与设计 §二 第 9 条
    正面冲突。映射收在这里的理由：它是「对外状态码」这件事的一部分，与 PUBLIC_ERRORS
    同源；散进 main 的处理器里，加一条新映射时就会在流量最大的那条路径上漏改。

    随状态码一起被丢掉的还有 `Allow` 头（它在 405 上由框架挂）—— 只把状态码改成 404
    而留着 `Allow: POST`，探测者照样读得出「这里有一条 POST 路径」，等于没修。
    代价已知并接受：/law/* 与 /public/* 的错方法也一并变 404，它们是公开路径，
    只是少了一句「你该用 POST」的提示，不挡正经调用。
    """
    return 404 if status_code == 405 else status_code


class ApiError(Exception):
    """接口层错误的基类：自带对外状态码；`detail` 默认只进日志，不进响应。

    构造参数与 `str(exc)` 都指向 detail（便于日志与调试），而对外文案由
    main.py 按 `status_code` 查表得到 —— 两条通路刻意不共用，这样「把异常原文
    写进响应」需要显式写代码才做得到，而不是顺手 `str(exc)` 就会发生。

    public_detail 是那道显式开关，**默认关**（fail-closed）：设计里只有 §七 第 4
    行（501）要求「reason 写明」，故只有那个类打开它。test_main_errors.py 有一条
    用例钉住「打开它的类型只有 FeatureNotImplemented」—— 加新类型时顺手打开它，
    等于把「异常原文只进日志」这条悄悄改掉。
    """

    status_code = 500
    public_detail = False

    def __init__(self, detail: str = "") -> None:
        super().__init__(detail)
        self.detail = detail


class BadRequest(ApiError):
    """参数不合法（含量校验之外的自定义规则，如输入长度上限）。"""

    status_code = 400


class Unauthorized(ApiError):
    """登录失败（口令不对 / 账号不存在 / 账号停用 —— 三者对外**必须是同一种**）。

    detail 只进日志：它是排查「是哪种失败」的唯一线索，而对外只给 §七 表的
    通用文案。三合一不是偷懒：分开报会把「这个用户名存在」告诉一个没通过
    认证的人 —— 那正是本类型存在的理由（公开文案由 PUBLIC_ERRORS[401] 定，
    不在这里写，免得两处口径分家）。
    """

    status_code = 401


class RateLimited(ApiError):
    """限流 / 并发超限。retry_after 由 main.py 写进 Retry-After 响应头（§六）。"""

    status_code = 429

    def __init__(self, detail: str = "", *, retry_after: int = 1) -> None:
        super().__init__(detail)
        # 直接放 int 而不是格式化好的字符串：响应头要的是秒数，
        # 让唯一的出口去格式化，避免两处各写一遍
        self.retry_after = retry_after


class FeatureNotImplemented(ApiError):
    """已注册但未实现（历史案件检索 FR-5.3）。reason 由路由写进 detail。

    **details 是公开的**：§七 第 4 行明确要「reason 写明」（前端据此告诉用户什么
    时候能用上），故这里打开 public_detail。理由也够硬：这个 detail 由服务端自己
    写死，不含用户输入、不含下游异常原文，与 503 那种「下游把原文塞进异常」不同类。
    """

    status_code = 501
    public_detail = True


class ServiceUnavailable(ApiError):
    """系统故障：Milvus / LLM / 模型不可用。设计 §七 的红线一行。"""

    status_code = 503


# Answerer 的故障终态字面量。它是 app/generation/answer.py 里 `status="error"`
# 的那个值，本模块不 import 那边（core 与 generation 的 import 方向是 core →
# 下层，反向会成环）。防漂移靠测试：test_main_errors.py 真跑一次会失败的
# Answerer，断言它吐出来的状态与本常量相等
FAILURE_STATUS = "error"


def raise_if_answer_failed(result) -> None:
    """问答**故障**转 503；其余终态一律放行（200，由响应里的 status 字段表达）。

    这是设计 §七 第 5、6 两行的落点，也是整张表最要紧的一处：③a/③b-1 在服务层
    守的「故障 ≠ 没有依据」，到接口层就体现为这两行的区别。压成同一种对外行为
    （都 503、或都 200）会让前面那些努力在验收时看不见 —— 故判据只有一个入口，
    路由不许自己判 `status == "error"`。

    参数从宽（鸭子类型，只读 .status/.answer）：调用方传 QAResult，测试传替身，
    Answerer 换实现也不必改这里。
    """
    if getattr(result, "status", None) == FAILURE_STATUS:
        # detail 带出 Answerer 写的故障描述（含底层异常原文）——只进日志。
        # 对外文案由 PUBLIC_ERRORS[503] 决定，这里写什么都不影响响应
        raise ServiceUnavailable(f"问答链路故障：{getattr(result, 'answer', '')}")
