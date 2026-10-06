"""法条只读路由：条文原文与编—章—节导航（设计 §四、§六）。

**两条都不挂鉴权**：它们属于设计 §六 的「公众可读的三个」（`/public/qa`、
`/lawyers/recommend`、`/law/*`）。把该公开的也 404 化会把访客挡在门外 ——
FR-8.7 给公众的权限里明写「查看法条原文」，而法条库本身是公开数据
（§六 数据层那句：法条是公开数据，不需要隔离）。

本模块只读：SQL 只有两条 SELECT，没有任何写入口（与公众侧的只读口径一致）。
两个端点的 404 语义要与「未知路径」完全同形（§二 第 9 条）：走 HTTPException(404)
由 main._on_http_error 统一出错误体，而不是自己拼一个 JSONResponse ——
自己拼的那一刻起，两处的文案/字段就有了分叉的余地。
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.api import deps, schemas
from app.retrieval.article_lookup import fetch_articles

# 导航树的原料：整张 article 表的四列。为什么不按条聚合（`SELECT DISTINCT path`）：
# 叶子要带 `article_no` 才能跳转（FR-7.3 的「定位」），而只取 path 的查询里没有
# 条号 —— 再补一次「按 path 取条」的查询等于把同一张表读两遍，且两次读之间
# 若有人改库，导航树与条文会互相矛盾。1260 行的库一次读回内存折叠，代价是
# 可忽略的，而「一次查询 = 一个一致的快照」这条性质值得留着。
# 不写 ORDER BY：排序在内存里按 int 做 —— article_no 是 VARCHAR(16)，库里的
# 字典序会把第 10 条排在第 2 条前面（实测条号分布 1~1260，全部会踩到）
NAV_SQL = "SELECT law_id, path, article_no, article_no_cn FROM article"

router = APIRouter(prefix="/api/v1/law", tags=["law"])


@router.get("/nav", response_model=schemas.NavResponse)
def nav(request: Request) -> dict:
    """编—章—节导航树（FR-7.3：逐级浏览并定位）。

    树完全由 `article.path` 折叠而来，**不猜任何层级**：库里有什么编就有什么编
    （例如不补一个不存在的「第二编」）。层级数不固定 —— 实测本库的 path 有 2、
    3、4 三种深度（297/785/178 行），故按段落下标泛化，而不是写死编/章/节三个
    字段：写死会让四层的 178 行在折叠时静默丢失（少几个节点，没有任何报错）。

    响应形状（本任务自定，理由见报告「存疑与需裁决项」）：`{"request_id", "laws": [...]}`，
    每个 law 下是 `{"law_id", "nodes"}`，节点带 `title`/`level`/`path`/`articles`/`children`。
    按 law_id 分树而不是合成一棵：跳转 URL 需要 law_id（客户端不该硬编码
    "minfadian"），而两部法的 path 前缀完全可能重名（"第一编 总则" 不止民法典有），
    合并会把别家的章节静默揉进来。今天库里只有一部法，故 laws 恒为一个元素；
    形状按多法设计，是为「加第二部法」时客户端不必改契约。

    装配未完成 → 503（services_of 的唯一出口）：那是「我们坏了」，
    与「这一条不存在」的 404 是两种结论（§七 的 503/200 同族口径）。
    """
    with deps.services_of(request).conn.cursor() as cur:
        cur.execute(NAV_SQL)
        rows = cur.fetchall()
    return {"request_id": deps.request_id_of(request),
            "laws": build_nav(rows)}


def build_nav(rows: list[tuple]) -> list[dict]:
    """`(law_id, path, article_no, article_no_cn)` 行集 → 每个 law_id 一棵树。

    纯函数（不碰连接）：折树规则值得单独测，而它的输入形状只有 SQL 那一处产出。
    折叠按 path 的 ` > ` 分段做前缀合并 —— 段的相等性由「前缀串相等」保证
    （path 是拼好的字符串，同前缀 ⇒ 同段落），故不另做归一化（如去空格：
    库里「第一分编 通 则」中间那个空格是原文自带的，去掉反而与另一处分叉）。

    条号在节点上按 **int 排序**（VARCHAR 的字典序会把 10 排到 2 前面），
    并且**允许节点同时有 articles 与 children**：今天的 109 条 path 里没有
    「某条 path 是另一条的前缀」的情形（实测 0 例），但那是数据现状而不是契约，
    四条民法典条文改挂到「第一编 总则」下就会造出这种节点 —— 那时丢掉任何
    一半都是静默的（树少一层，或条少几条，都只是看着少一点）。
    """
    laws: dict[str, list[dict]] = {}
    for law_id, path, article_no, article_no_cn in rows:
        siblings = laws.setdefault(law_id, [])
        prefix: list[str] = []
        node: dict | None = None
        for level, title in enumerate(path.split(" > ")):
            prefix.append(title)
            node = next((n for n in siblings if n["title"] == title), None)
            if node is None:
                node = {"title": title, "level": level,
                        "path": " > ".join(prefix),
                        "articles": [], "children": []}
                siblings.append(node)
            siblings = node["children"]
        # 条挂在**最深层**节点上（path 全串对应的节点），而不是挂到「第一个没有
        # children 的节点」那种猜测性的位置上：后者在「某条 path 恰是另一条的
        # 前缀」时会把条挂错层，而形态是树看着仍然完整、只是条去了别人家里
        node["articles"].append({"article_no": int(article_no),
                                 "article_no_cn": article_no_cn})
    for nodes in laws.values():
        _sort_articles(nodes)
    return [{"law_id": law_id, "nodes": nodes} for law_id, nodes in laws.items()]


def _sort_articles(nodes: list[dict]) -> None:
    """递归按条号数值排序每个节点的 articles（稳定、原地）。"""
    for node in nodes:
        node["articles"].sort(key=lambda item: item["article_no"])
        _sort_articles(node["children"])


@router.get("/{law_id}/articles/{article_no}", response_model=schemas.ArticleResponse)
def article(law_id: str, article_no: str, request: Request) -> dict:
    """条文原文（FR-7.2 的落点：引用的「一键跳转」打到这条路径）。

    取条**复用 `article_lookup.fetch_articles`**，不在这里另写 SQL：那条查询
    带着两个必须一致的口径 —— 效力读 `law_version.status`（不是条表副本）、
    条号列是 VARCHAR 而下游块里是 int。第二份 SQL 的形态是「第一份改了、这份
    没改」，而 FR-7.2 的跳转准确率正是靠这个口径：库里能查到、块里对得上。

    参数校验：`article_no` 是路径段（字符串），**显式转 int，转不动就 404**。
    不转的后果是它直接进 SQL 的参数位，pymysql 把 'abc' 交给 VARCHAR 列不会报错，
    查询返回空 → 仍然是 404，看着一样 —— 但那只是巧合（这一列的隐式转换恰好
    宽容）；若哪天列类型收紧或换成数值列，同样的输入会变成 500。转不动判 404
    而不是 400：请求的是一条不存在的路径形状，与「第 9999 条不存在」同类，
    §二 第 9 条要的也是这两种情形对外不可区分。取不到（条不存在、law_id 不在
    库中）一律 404 —— 与「未认证」同码是有意的，这一条本来就不存在
    「存在但不给你看」的情形。
    """
    try:
        number = int(article_no)
    except ValueError:
        raise HTTPException(status_code=404) from None
    blocks = fetch_articles(deps.services_of(request).conn, [number],
                            law_id=law_id)
    if not blocks:
        raise HTTPException(status_code=404)
    return schemas.article_payload(blocks[0], deps.request_id_of(request))
