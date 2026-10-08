"""S7 提问链路（医知源）的运行时服务模块。

对外唯一入口是 `backend/serve.py`；本包不被单独执行。

模块划分（按「什么时机可能出错」切分，见 specs/006 的 plan.md）：

    config    启动期读环境变量        —— 与请求路径隔离
    schemas   传输模型                —— docs/05 §3.1.3 契约的落点
    validate  问题校验                —— 边界值判定只此一处
    errors    统一错误体              —— 防止框架默认 HTML 错误页漏出堆栈
    events    SSE 编码                —— 线格式只此一处
    stream    事件序列                —— **后续检索/生成模块的唯一接缝**
    routes    HTTP 协议转换            —— 不做业务判定
    app       FastAPI 装配             —— 路由注册顺序在此（先 API 后静态）

本模块只放常量，不放逻辑。这样切分是为了让「全仓只有一处定义用户可见文案」这条
要求在**结构上可见** —— docs/05 §4.4 对逐字话术提的是同类要求，此处把它扩展到
本特性的全部固定文案。
"""

# ---- 路由 ----

ASK_PATH = "/ask"
STATIC_DIR_NAME = "frontend"

# ---- SSE 事件名（契约见 specs/006-medical-qa-input/contracts/sse.md）----
#
# 事件顺序是硬约束：status → citations → token* → done。
# 类型走 `event:` 字段而非 JSON 内的 type 字段 —— 协议层能表达的东西不放进应用层。

EV_STATUS = "status"
EV_CITATIONS = "citations"
EV_TOKEN = "token"
EV_DONE = "done"

# ---- 状态取值 ----

STATE_ACCEPTED = "accepted"

# risk_level 的四个取值。本期只可能产出 RISK_LEVEL_NONE（未产出知识性内容）。
# 其余三个在本期不会被使用，列在这里是为了让取值集合完整可见，
# 接入生成模块时不必回头翻 docs/05 §3.1.6。
RISK_LEVEL_IMMEDIATE = "immediate"
RISK_LEVEL_SOON = "soon"
RISK_LEVEL_OBSERVE = "observe"
RISK_LEVEL_NONE = "none"

# ---- 输入限额 ----
#
# 上限 200 的依据见 docs/05 §3.1.2：M1 定义的是单轮、非专业的口语化提问，
# 200 字符足以覆盖；超长输入既不服务真实场景，又会放大提示词注入与成本风险。
# 这是产品边界的显式表达，不是技术限制。
QUESTION_MIN_LEN = 1
QUESTION_MAX_LEN = 200

# ---- 面向用户的固定文案 ----
#
# ⚠️ 全仓检索这四段文本，只应命中本文件。
# 任何其他位置的重复出现都应在代码评审中被拒绝 —— 分散定义会让
#「改了一句、漏改一处」成为必然，而这几句都是用户每次提问都会看到的。
#
# ⚠️ 前三段 MUST 逐字匹配 docs/01 的锁定文本，改动等于改需求。

# 紧急症状前置话术。constitution 原则 IV：必须出现在任何内容之前且逐字输出。
# 本期无紧急判定模块，因此**不会发出**；但流式结构已把它的位置固定下来
#（必须是第一个 token 事件），见 stream.py。
EMERGENCY_PREAMBLE = "请立即就医或拨打急救电话。以下信息仅供参考。"

# 免责声明。constitution 原则 V：所有路径（含拒答）末尾必加。
DISCLAIMER = "以上为基于知识库的参考信息，不能替代执业医师的当面诊断。"

# 拒答兜底话术。docs/01 §F6 只说"返回固定兜底话术"却从未给出文本，
# 本特性首次定稿（用户裁决，2026-09-27）。四条拒答路径共用同一套话术，
# 不在其中区分是哪一环失败（docs/02 §344）。
#
# 本期**不会发出**：本期不存在「检索过且为空」这一事实，发它会让用户误以为
# 知识库覆盖不足，而真相是检索能力尚未建成。详见 specs/006 research.md R5。
REFUSAL_FALLBACK = (
    "这个问题在现有知识库中找不到可以依据的原文，我无法给出有依据的回答。"
    "建议你咨询专业医生。"
)

# 能力未就绪文案。本期唯一会发出的答案区文本。
#
# 为什么不用 REFUSAL_FALLBACK 代替它：docs/05 §3.1.4 规则 3 明令禁止把
#「本该能答但基础设施坏了」伪装成拒答 —— 用户会以为"换个问法也许有"，
# 而实际是"问什么都一样"。二者对用户的含义完全不同。
CAPABILITY_NOT_READY = (
    "你的问题已收到。医知源的向量检索与大模型生成能力正在接入中，"
    "本期暂不能给出答案。"
)

# answer_text 的拼接分隔符。docs/05 §3.1.5 锁定的拼装契约：
#   answer_text = [紧急话术 + 分隔符] + 正文或兜底话术 + 分隔符 + 免责声明
ANSWER_JOINER = "\n\n"

# ---- 故障注入（仅验收用，正常路径不受影响）----
#
# 本期没有紧急判定模块，FR-030（话术必须是首个 token）因此没有真实触发条件。
# 用本变量在验收时注入一段前置文本，使这条约束可以被断言到。
# 沿用 specs/005 的 MEDRAG_TEST_FAULT 模式，不引入新范式。
PREAMBLE_ENV = "MEDRAG_TEST_PREAMBLE"

# ---- 错误码与用户可见文案（docs/05 §2.3）----

ERR_INVALID_QUESTION = "INVALID_QUESTION"
ERR_INTERNAL = "INTERNAL_ERROR"

# ⚠️ `NOT_FOUND` **不属于** docs/05 §2.3 的 API 错误码集合。
#
# 它存在的原因是：静态资源挂在根路径，访问一个不存在的路径会由 StaticFiles
# 抛出 404，那是 Web 层的事，不是 API 的事。若不为它指定错误码，响应会退回到
# 框架默认的 `{"detail": "Not Found"}` —— 与其它错误形状不一致，客户端要么
# 为它写特例解析，要么在解析失败时表现异常。
#
# 这里给的是"形状一致、语义区分"的折中：客户端只需一套错误体解析逻辑，同时
# 能看出这不是一次提问失败。docs/05 §2.3 的四码集合仍然只描述 API 行为。
ERR_NOT_FOUND = "NOT_FOUND"

MSG_INVALID_QUESTION = "请输入你的问题"
MSG_TOO_LONG = f"问题过长，请控制在 {QUESTION_MAX_LEN} 个字符以内"
MSG_INTERNAL = "服务暂时不可用，请稍后重试"
MSG_NOT_FOUND = "页面不存在"

# 客户端可选传入的关联头。未传则由服务端生成（docs/05 §2.2）。
REQUEST_ID_HEADER = "X-Request-Id"

# ---- 进程退出码 ----

EXIT_OK = 0
EXIT_CONFIG = 2
