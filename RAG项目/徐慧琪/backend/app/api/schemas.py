"""请求/响应模型与契约整形（设计 §四 的可执行形式）。

为什么单独一个模块：§四 那张契约表是**硬规格**（字段名、哪些键必须出现、哪些
必须不出现），而契约的兑现散在十几个路由里时，「多一个键」与「少一个键」都不会
有红灯。故请求模型与响应整形都收在这里：路由只负责取值 → 调整形函数 → 返回。

**内部结构与对外契约之间有一层投影，不是透传** —— 三条实据：
  ①`attach._unavailable` 把异常原文写进了 `fee["reason"]`（"费用信息暂时不可用：
    milvus down"），设计 §七 明写异常原文不给公众，③b-1 的真跑也抓到过外泄；
    CLI 的渲染层同样刻意不印它。透传 fee 字典 = 当场把这个洞重新打开。
  ②检索块（sources）里有整条法条原文 `text` 与 `status`/`law_id`/`chunk_id`：
    原文不在 §四 的契约里（要原文有 /law/{law_id}/articles/{article_no}），
    而把它塞进每个问答响应会让响应体随召回条数膨胀到几十 KB。
  ③模型给的 Citation 含 `law` 字段（未经校验的自由文本），§四 的引用形状是
    四键 —— 投影到契约键，新增内部字段就不会自动变成对外字段。
"""
from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

# 问句的字符数上限（设计 §六 防滥用表「输入长度」那行）。取值锚在审计表的
# query_text 列宽（设计 §五 VARCHAR(500)）：律师侧要写进那一列的原文必须短于它，
# 否则任务 7 的审计只能截断，而截断正是本节禁止的处理方式。这条上限与
# main.py 的体上限（64KiB）是**两层**：体上限只认 Content-Length、分块传输
# 绕得过，字符数这一层绕不过（任务 3 复审的交接项）
MAX_QUESTION_CHARS = 500

# 用户名上限取自 users 表的列宽（VARCHAR(64)）：比它长的值在库里存不下，
# 而在这一层拒掉比让 pymysql 抛「Data too long」清楚得多（后者会变成 500）。
MAX_USERNAME_CHARS = 64

# `POST /search`（律师侧，只检索不生成）的 top_k 上限。默认值是 pipeline 的
# RERANK_OUTPUT_TOPK（5，见 SearchRequest 的注释：默认值在**路由层**取，这里不写
# 第二份字面量）；上限取 20 是本任务自定的（报告「存疑与需裁决项」有记）：
#   ①它是默认值的 4 倍，够律师「多看几条」用，又不至于把响应撑成 NB 级 ——
#     每块约 300 字节（不含法条原文，见 SEARCH_BLOCK_FIELDS 的取舍），20 块 ≈ 6KB；
#   ②上限必须**小于**检索池（RECALL_TOPK=50 + 精确块），否则这个闸门只是装饰：
#     merge_blocks 本来也吐不出比池子更多的块，把上限设到 1000 与不设等价，
#     而「设了上限」这句话就会在报告里变成一句不设防的声明。
MAX_TOP_K = 20

# 口令只设下限不设上限，且下限与 tools/manage_users.MIN_PASSWORD_LEN 有意
# **不共用常量**：那是建号时的策略（8 位），这是接口层的输入校验（1 位），
# 两者可以各自演进 —— 把已建账号的旧口令挡在登录之外（比如接口层也要求 8 位）
# 是纯粹的故障。空口令在 security.hash_password 里就被拒了，这里 min_length=1
# 只是把「空串」这种畸形输入挡在校验层（否则它会走一遍 scrypt 再判否）。

# strip_whitespace 在长度判定**之前**生效：少了它，一个只含空格的问题会通过
# min_length=1，然后一路走到编码器，在那儿以「sparse 转换后为空」的形态炸成
# 服务故障 —— 把「输入不合法」说成「我们坏了」，正是 ③a 那条红线要防的形态
_Question = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1,
                                             max_length=MAX_QUESTION_CHARS)]
_Username = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1,
                                             max_length=MAX_USERNAME_CHARS)]
# 口令**不 strip**：两端空格是口令的一部分，改它等于改口令（登录就会失败，
# 而用户看到的只是「用户名或密码错误」—— 一次本可避免的排查）
_Password = Annotated[str, StringConstraints(min_length=1)]


class LoginRequest(BaseModel):
    """登录请求（设计 §四）。

    两个字段都钉成 str：pydantic v2 对 int→str **不做**隐式转换，JSON 里给个
    数字会变成 400（校验失败），而不是 500。这条不是形式 —— security.verify_password
    对 password 只做 .encode()，收到数字就是 AttributeError → 500（任务 2 交接
    点名的形态）。
    """

    username: _Username
    password: _Password


class LoginResponse(BaseModel):
    """登录响应（设计 §四）：{"access_token", "token_type", "role", "team_id"}。

    team_id 进响应体是设计定的（前端要按团队展示），它本来就是 token 载荷里的
    内容，多出现在这一层不增加暴露面。
    """

    access_token: str
    token_type: str = "bearer"
    role: str
    team_id: str


class QuestionRequest(BaseModel):
    """问答请求。公众侧与律师侧共用同一个形状（字段名 question）。

    两侧共用不是省事：`tools/ask.py` 的位置参数与用户的心智模型都是「一个问题」，
    两侧各起一个字段名（q / question）只会让前端多一套模板。
    """

    question: _Question


class QAAnswer(BaseModel):
    """问答响应的**公共部分**（设计 §四）。public 侧在它之上再加四键。

    分成基类与子类，而不是「一个模型 + exclude 掉两个键」：设计 §四 明写律师侧
    响应里**没有** `lawyers` 与 `fee_range` 两个键（不是置空，是根本不出现）——
    「不出现」用两个模型天然成立，用 exclude 则要每个出口记得传对参数，而漏传
    的表现是那两个键悄悄出现，没有任何红灯。任务 5 的律师侧响应用本类即可。
    """

    request_id: str
    status: str
    answer: str
    citations: list[dict]
    sources: list[dict]
    disclaimer: str
    failures: list[str]


class PublicQAAnswer(QAAnswer):
    """公众侧问答响应 = 公共部分 + 附加区块展开的四键（设计 §四）。

    四键**恒存在**（值可为 null/[]）：它们的有无由契约决定，不由本次回答有没有
    区块决定 —— 让客户端按「键在不在」分支，等于把服务端的一个内部判据
    （Answerer._extras_for 的侧别/状态/问价意图）泄漏成接口的形状。
    """

    cause: str | None
    field: str | None
    lawyers: list[dict]
    fee_range: dict | None


class LawyerRecommendResponse(BaseModel):
    """`/lawyers/recommend` 的响应（设计 §四：接 `lawyers` + `fees`）。

    字段与公众问答里的同名区块**同形**（lawyers / fee_range）—— 前端两处可以用
    同一套渲染组件，而形状分叉时（比如这里叫 fee、那里叫 fee_range）不会有红灯。
    field 一并带上：卡片是按领域匹配出来的，不带的话客户端看不出这批卡片属于哪个领域。
    """

    request_id: str
    field: str
    lawyers: list[dict]
    fee_range: dict | None


# 引用的对外键（设计 §四 的示例逐字）：模型那份 Citation 还带一个 `law`
# （见模块 docstring 的投影理由）。顺序照 §四 写，读代码时能与设计对上
CITATION_FIELDS = ("article", "paragraph", "item", "quote")

# 检索块的对外键（设计 §四）：article_no 与 path 是给人核对的落点，
# source 说明它来自精确置顶还是向量召回，rerank_score 是排序依据
SOURCE_FIELDS = ("article_no", "path", "source", "rerank_score")

# 费用区块的对外键（设计 §四 的 fee_range 示例 8 键）：**不含 `reason`**，
# 它就是那条会把异常原文带出去的口子（模块 docstring ①）。
# unit 与 charge_basis 必须在列：少了它们，量级（万元/元）与计价基础
# （每小时/每件）在客户端就丢成裸数字，真跑实证 N10 吃过这个亏
FEE_FIELDS = ("status", "low", "high", "unit", "charge_basis", "basis",
              "source_doc", "source_no")

# `POST /search` 每块的对外键（任务 5 自定的投影，理由接模块 docstring ②）。
# 与 SOURCE_FIELDS **刻意不共用一个常量**：两处取舍不同 —— 问答响应里的 sources
# 是「答案的出处」，四键够核对；检索端点的 blocks 是「本次召回了什么」，多一个
# article_no_cn 是因为律师的下一步动作是把条款号写进文书（FR-7.4 的「规范条款号
# 引用格式」），中文条号（"五百八十四"）是那一步的原料。共用常量会把两种用途的
# 差异静默掉：将来给 blocks 加键时问答响应会跟着多一个键而无人察觉。
SEARCH_BLOCK_FIELDS = ("article_no", "article_no_cn", "path", "source",
                       "rerank_score")


def citation_payload(citation) -> dict:
    """一条引用 → 对外四键。取属性而不是 model_dump：多一个模型字段就多一个对外键。"""
    return {name: getattr(citation, name, None) for name in CITATION_FIELDS}


def source_payload(block: dict) -> dict:
    """一个检索块 → 对外四键。用 get 而不是下标：块的键集在两条召回路上不对称
    （精确块 14 键、向量块 12 键，见 pipeline.merge_blocks），缺键时给 None
    比 KeyError 变成 500 合适 —— 缺的字段是展示用的，不是判据。
    """
    return {name: block.get(name) for name in SOURCE_FIELDS}


def fee_payload(fee: dict | None) -> dict | None:
    """费用区块 → 对外 8 键；区块整个缺席时给 None（诚实：没算过 ≠ 算出来不可用）。

    注意这里**不**把缺席改写成 `{"status": "unavailable"}`：那是在替链路
    表态（说「费用信息暂时不可用」），而真实情况可能只是「这次没走费用链路」
    （如公众侧的拒答出口）。把没算过的说成算不出来的，是本项目反复防的那类
    「把 A 说成 B」。
    """
    if fee is None:
        return None
    return {name: fee.get(name) for name in FEE_FIELDS}


def public_qa_payload(result, rid: str) -> dict:
    """QAResult + 附加区块 → 公众侧问答响应（设计 §四 的字段逐个对上）。

    extras 为 None 的出口（拒答且不含问价词、律师侧、extras_fn 关闭）里，
    四键仍在：lawyers 给 `[]`、其余给 None。理由同 PublicQAAnswer 的注释 ——
    键的有无是契约，不是判据。

    failures 原样带出：它是「为什么降级成未找到依据」的唯一线索，且内容是
    内部措辞（如「输出解析失败：…」）而非用户输入 —— 与 reason 不同，它不含
    下游异常原文以外的东西。这是本任务对设计的有意补充（§四 示例里 failures
    是空数组，说明该键必须在），不额外脱敏的理由见报告。
    """
    extras = result.extras or {}
    return {
        # request_id 由调用方从 request.state 取（任务 3 交接：不要在 api/ 里
        # import app.main 拿 _scope_request_id，那会与 main→api 的路由注册成环）
        "request_id": rid,
        "status": result.status,
        "answer": result.answer,
        "citations": [citation_payload(c) for c in result.citations],
        "sources": [source_payload(b) for b in result.sources],
        "cause": extras.get("cause"),
        "field": extras.get("field"),
        # `or []` 而不是直接 get：extras 有键但值为 None 时（没有哪个出口这么给，
        # 但 extras 是 dict、类型上允许）也不能让 [] 与 None 两种形状都流出去
        "lawyers": extras.get("lawyers") or [],
        "fee_range": fee_payload(extras.get("fee")),
        "disclaimer": result.disclaimer,
        "failures": list(result.failures),
    }


def lawyer_qa_payload(result, rid: str) -> dict:
    """QAResult → 律师侧问答响应：恰好 7 键，**没有** lawyers / fee_range。

    与 public_qa_payload 的那 7 个公共键是刻意重复的，不是抽公共函数：
    public 侧是超集（7 + 4 键），抽成 `_common(result, rid)` 再各自 extend
    会让「哪些键属于公共部分」变成一个隐式约定，而这里的**判据是键的集合本身**
    （律师侧不许出现那两个键）。重复的代价是改公共键要改两处 —— 由测试兜住：
    两侧的响应键集都有字面量断言（public 11 键、lawyer 7 键），漏改一处即红。

    failures 的取舍与 public 侧一致（原样带出，它是「为什么降级」的唯一线索）；
    disclaimer 也照带 —— 律师侧的内侧免责是空串（Answerer._side_disclaimer 按侧给），
    键仍在，前端不必为两侧写两套模板。
    """
    return {
        "request_id": rid,
        "status": result.status,
        "answer": result.answer,
        "citations": [citation_payload(c) for c in result.citations],
        "sources": [source_payload(b) for b in result.sources],
        "disclaimer": result.disclaimer,
        "failures": list(result.failures),
    }


class SearchRequest(BaseModel):
    """`POST /search` 的请求（律师侧，只检索不生成）。

    top_k 可选：省略时由**路由层**取 pipeline.RERANK_OUTPUT_TOPK —— 默认值不在这里
    写第二份字面量。两份默认值分家的形态是「接口默认 5、链路默认 3」，而两者都
    「看着对」，只有对比才看得出；`None` 是本模型对「没传」的唯一表示。
    上限卡在 schema 层（超限 → 400），与问句长度上限同一个理由：它是输入校验，
    不是业务规则；放进路由就变成「每条端点自己记得判一次」。
    """

    question: _Question
    top_k: int | None = Field(default=None, ge=1, le=MAX_TOP_K)


class SearchResponse(BaseModel):
    """`POST /search` 的响应。**不含生成内容**：这条端点的价值就是便宜、可反复打。

    带 question 回来（strip 后的值）：调用方据它对齐「我搜的是哪句话」，尤其是
    批量比对时 —— 响应里没有问句，客户端就只能靠请求顺序猜。
    exact_nos 单独成键而不是塞进 blocks：它是「第 N 条被精确置顶」的判据
    （pipeline 只上报真正取到的条号），客户端要能一眼看出这次走没走精确通路。
    """

    request_id: str
    question: str
    blocks: list[dict]
    exact_nos: list[int]


class ArticleResponse(BaseModel):
    """`GET /law/{law_id}/articles/{article_no}` 的响应（FR-7.2 的落点）。

    只给**法条原文**，不掺任何生成内容：FR-4.4/FR-8.2 要求客户端能把「原文」与
    「AI 解释」分开呈现，而一条同时混着两者的接口会让那道视觉分离在数据层就不成立。
    status 是效力（来自 law_version，不是条表副本），引用该条前要据此判断是否有效。
    """

    request_id: str
    article_no: int
    article_no_cn: str | None
    path: str
    text: str
    status: str
    law_id: str


def search_payload(result, rid: str) -> dict:
    """RetrievalResult → `/search` 响应。

    块用白名单投影（SEARCH_BLOCK_FIELDS）而不是透传：透传会把整条法条原文
    （`text`）塞进每个块，而原文有它自己的端点 —— 5 条父块 ≈ 数十 KB，
    最贵的那部分（原文）在这里毫无用处（客户端要展示原文时会去打那条端点）。
    `article_no_cn` 与 `path` 都是 None 安全的（两条召回路的块键集不对称，
    见 source_payload 的同款理由）。
    """
    return {
        "request_id": rid,
        "question": result.question,
        "blocks": [{name: block.get(name) for name in SEARCH_BLOCK_FIELDS}
                   for block in result.blocks],
        "exact_nos": list(result.exact_nos),
    }


def article_payload(block: dict, rid: str) -> dict:
    """article_lookup 的父块 → 条文响应（设计 §四：这一层只给原文）。

    取键用下标而不是 get：这里的块**只有一个来源**（article_lookup.fetch_articles
    的 _row_to_block），七个键是该函数的输出契约；缺键说明取块路径被改坏了，
    KeyError → 500 比静默给 null 好 —— 静默 null 会让客户端拿到一条空条文而
    无法与「这条真的没有原文」区分（那正是 §七 要防的「把 A 说成 B」）。
    """
    return {"request_id": rid, "article_no": block["article_no"],
            "article_no_cn": block["article_no_cn"], "path": block["path"],
            "text": block["text"], "status": block["status"],
            "law_id": block["law_id"]}


class NavArticle(BaseModel):
    """导航树的条目落点（FR-7.3 的「定位」）：条号 + 中文条号。

    只有两级必要信息：article_no 是跳转参数（打条文端点取原文），article_no_cn
    是给人看的（用户与律师说的都是「第五百八十四条」）。路径不在这一层重复 ——
    它就在父节点上（同一个 path 复制上千遍会让响应体翻倍）。
    """

    article_no: int
    article_no_cn: str | None


class NavNode(BaseModel):
    """导航树的**一级**（编 / 分编 / 章 / 节 —— 层级数不固定，按下标泛化）。

    层级数由数据决定而不是三个固定字段（编/章/节）：实测本库的 path 有 2、3、4
    三种深度（1260 行里 297/785/178），写死「编—章—节」会让四层的 178 行
    在折树时无处安放 —— 而那种丢失是静默的（少几个节点，没有任何报错）。
    title 是 path 里那一段的原文（"第一编 总则"），level 是从 0 开始的下标；
    path 是本级的**跳转落点**：它是该级全部条文的公共前缀，客户端凭它定位
    （逐级浏览），凭 articles 里的 article_no 取具体条文（逐级定位）。
    """

    title: str
    level: int
    path: str
    articles: list[NavArticle]
    children: list["NavNode"]


# 自引用模型的显式重建（pydantic v2 的要求）：少了它，NavResponse 校验时
# children 这一层会被当成未解析的前向引用，实测的直接表现是 create_app 期
# 就抛 PydanticUndefinedAnnotation —— 属于「不写会当场炸」而非「悄悄退化」
NavNode.model_rebuild()


class LawNav(BaseModel):
    """一个 law_id 的整棵树。

    按 law_id 分开而不是合成一棵：今天库里只有民法典，但 `article.law_id` 是
    设计里的多法维度（条文端点的路径段就带它），而两法的 path 前缀完全可能重名
    （"第一编 总则" 不止民法典有）。合并成一棵树会把两部法**静默**揉在一起 ——
    客户端的编章节视图凭空多出别人家的章节，没有任何红灯。分开还有个实际好处：
    客户端的跳转 URL 需要 law_id，而它不该（也不能）从服务端代码里硬编码。
    """

    law_id: str
    nodes: list[NavNode]


class NavResponse(BaseModel):
    """`GET /law/nav` 的响应（形状是本任务自定的，理由见报告「存疑与需裁决项」）。"""

    request_id: str
    laws: list[LawNav]
