# 法条只读两条端点（/law/nav 与 /law/{law_id}/articles/{article_no}）的判据：
# 设计 §四 的契约、§六 的「公众可读」、§七 的 503 与 404。
#
# **真连库**（项目惯例：能用真依赖就不用替身）：这两条端点的价值全在库里的
# 1260 行上 —— 替身能证明「接线对」，证明不了「折叠规则对真数据的 109 条 path
# 成立」。折叠规则本身（build_nav 是纯函数）另有离线用例，覆盖冲突形状。
#
# 末尾附一条律师侧 /qa 的嵌套投影用例（设计 §四 的契约那一族；安置理由见其 docstring）。
import pytest
from fastapi.testclient import TestClient

from app import main
from app.api import law as law_mod
from app.core import security
from app.db.mysql import connect
from app.generation.answer import QAResult
from app.generation.schema import Citation
from tests._fakes_http import (JWT_SECRET, FakeServices, bearer, fake_factory,
                               jwt_token, without_request_id)

# 真条号与真路径：584 挂在「第三编 合同 > 第一分编 通 则 > 第八章 违约责任」下，
# 选它是因为多段 path 才暴露得出「只按第一段折叠」这类退化
ARTICLE_NO = 584
ARTICLE_PATH = "第三编 合同 > 第一分编 通 则 > 第八章 违约责任"


def _mysql_online() -> bool:
    """MySQL 在线才跑真依赖用例（离线时跳过，而不是假装通过）。"""
    try:
        connect().close()
        return True
    except Exception:
        return False


requires_mysql = pytest.mark.skipif(not _mysql_online(), reason="MySQL 未在线")


def _client(conn=None) -> TestClient:
    """真 create_app + 假 Services（只填一条真连接）：中间件、异常映射、路由都真。

    conn 省略时给一个占位对象：只有「装配缺失 → 503」那条用例会走到它，
    而那一条恰恰要的是「services 在、连接不在」的形态（与真装配失败同形）。
    """
    services = FakeServices(conn=conn if conn is not None else object())
    factory_fn, _ = fake_factory(services)
    return TestClient(main.create_app(services_factory=factory_fn))


def _walk(nodes: list[dict]):
    """深度优先遍历导航树，产出每个节点（多条用例共用同一套遍历）。"""
    for node in nodes:
        yield node
        yield from _walk(node["children"])


def _articles_of(nodes: list[dict]) -> list[dict]:
    """树里全部条目的落点（article_no + article_no_cn），展平成一维。"""
    return [item for node in _walk(nodes) for item in node["articles"]]


def _db_rows(sql: str, params=None) -> list[tuple]:
    """直接查库取几行（不经端点）：判据「树里的落点在库里查得到」要用真库比对。"""
    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()
    finally:
        conn.close()


def _nav(client: TestClient) -> list[dict]:
    """打一次 /law/nav（**不带 token**：这两条端点是公开的）并取出每个 law 的树。"""
    response = client.get("/api/v1/law/nav")
    assert response.status_code == 200, response.text
    return response.json()["laws"]


# ---- GET /law/nav（FR-7.3）----


@requires_mysql
def test_the_nav_tree_covers_every_article_in_the_database():
    """导航树的落点必须**一个不漏**地对上库里 1260 条。

    这是本端点唯一真正承重的判据：树的形状怎么设计都可以，但「少了几条」是
    静默的（客户端只是少几个条目，没有任何报错），而少掉的形态恰恰是最可能
    发生的退化 —— 库里 path 有 2/3/4 三层深度（实测 297/785/178 行），
    把树写死成「编—章—节」三层就会把四层那 178 行丢掉。

    判据取**集合相等**而不是长度相等：长度相等挡不住「多一条少一条」，
    而多出来的条号意味着树里混进了库里没有的东西（那更糟，因为它会让
    客户端跳到一条 404 的条文上）。
    """
    with _client(connect()) as client:
        laws = _nav(client)
    assert [law["law_id"] for law in laws] == ["minfadian"]
    nodes = laws[0]["nodes"]
    tree_nos = {item["article_no"] for item in _articles_of(nodes)}
    db_nos = {int(row[0]) for row in _db_rows("SELECT article_no FROM article")}
    assert len(db_nos) == 1260, "库里条数变了，本用例的样本前提需要重新确认"
    assert tree_nos == db_nos
    # 「第一编」是真实存在的层级（不是造出来的占位）：树里必须有它
    assert any(node["title"].startswith("第一编") for node in nodes)


@requires_mysql
def test_the_nav_tree_keeps_every_level_the_database_has_not_just_every_article():
    """层级不固定：树里必须**每一级**都在 —— 条号集合相等挡不住「层被上提合并」。

    上面那条覆盖判据只数条号，而把折树的 `path.split(" > ")` 截断成 `[:3]`
    （= 简报点名要防的「假设恰好三层」）时**条一条不少**：真库实测 178 条四层
    条文被上提合并进 6 个父节点（17 个「节」节点 → 0，节点数 136 → 119）。
    失败模型是**丢层、不丢条**，所以判据必须看得见「层」，且取**前缀闭包集合
    相等**：库里每条 path 的每一级都要在树里有对应节点，反向也不许多出层级 ——
    这比「存在一个 level>=3 的节点」更难绕过（后者在「只留最深一级」的退化下
    仍可能成立）。
    """
    with _client(connect()) as client:
        nodes = _nav(client)[0]["nodes"]
    db_paths = {row[0] for row in _db_rows("SELECT DISTINCT path FROM article")}
    expected = {" > ".join(path.split(" > ")[:cut]) for path in db_paths
                for cut in range(1, len(path.split(" > ")) + 1)}
    assert {node["path"] for node in _walk(nodes)} == expected
    # 第二道：level 字段要与库里**最深** path 的深度对上（路径对而层级被截会漏过上面一条）
    assert max(node["level"] for node in _walk(nodes)) == (
        max(len(path.split(" > ")) for path in db_paths) - 1)
    # 字面量锚（本机真库实测）：17 条四层 path → 17 个 level==3 的节点、其下 178 条；
    # 库里没有「某条 path 恰是另一条的前缀」（实测 0 例），故两个数一起钉住
    # 「四层没有被上提合并」。数字变了说明样本前提变了，先重新确认本用例。
    deep = [node for node in _walk(nodes) if node["level"] == 3]
    assert len(deep) == 17, "四层节点数变了：先确认库里 path 的深度分布（样本前提）"
    assert sum(len(node["articles"]) for node in deep) == 178


@requires_mysql
def test_the_nav_tree_carries_article_numbers_sorted_numerically():
    """每个节点下的条号按**数值**升序，且至少有一个节点能区分数值与字典序。

    article_no 在库里是 VARCHAR(16)：直接 ORDER BY 或按字符串排序会把第 10 条
    排到第 2 条前面（1260 条里 10/100/1000 都会踩到），而浏览体验上这表现为
    「章节里条号跳着走」——没有任何报错。只断言「有序」是不够的：把它写成
    `sorted(key=str)` 时用例必须红，故要求样本里存在 ≥11 条的同节点
    （1 与 10 的顺序才是两种排法的分歧点），否则断言恒真。
    """
    with _client(connect()) as client:
        nodes = _nav(client)[0]["nodes"]
    longest = 0
    for node in _walk(nodes):
        nos = [item["article_no"] for item in node["articles"]]
        assert nos == sorted(nos), f"节点 {node['path']} 的条号没按数值排序"
        longest = max(longest, len(nos))
    assert longest >= 11, "样本里没有能区分数值序与字典序的节点，本用例此刻恒真"


@requires_mysql
def test_every_nav_node_is_a_jump_target_and_the_first_article_is_reachable():
    """FR-7.3 的「逐级浏览并定位」：每一级带 path（浏览落点），叶子带 article_no（定位）。

    两个方向都钉：①树里每个节点的 path 是「父 path + ' > ' + 自己」——它是客户端
    逐级下钻时认的键，写错一处就会把两个不同的章节混成一个；②拿第一编第一章的
    第一条去打条文端点，必须 200 且 article_no 回得来 —— 导航与条文两条端点
    不是各自为政的：树给出的落点必须真的能在条文端点取到（否则前端点进去是 404）。
    """
    with _client(connect()) as client:
        nodes = _nav(client)[0]["nodes"]
        for node in _walk(nodes):
            assert node["path"].split(" > ")[-1] == node["title"]
            assert node["level"] == len(node["path"].split(" > ")) - 1
        first_chapter = next(node for node in nodes
                             if node["title"].startswith("第一编"))
        leaf = first_chapter["children"][0]
        assert leaf["articles"], "第一章下应有条目"
        number = leaf["articles"][0]["article_no"]
        detail = client.get(f"/api/v1/law/minfadian/articles/{number}")
    assert detail.status_code == 200
    assert detail.json()["article_no"] == number
    assert detail.json()["text"].strip()


# ---- GET /law/{law_id}/articles/{article_no}（FR-7.2）----


@requires_mysql
def test_the_article_endpoint_returns_the_raw_text_and_exactly_the_designed_keys():
    """条文响应 = 法条**原文** + 跳转键，恰好 7 个键（设计 §四）。

    键集用字面量集合钉：多一个键（例如把生成内容或内部列塞进来）要有人看一眼 ——
    这一层的立意是「只给原文」（FR-4.4/FR-8.2 要求客户端能分开原文与解释），
    而多出来的键正是那道分离开始漏的地方。text 与库里那一行逐字比对：
    对不上说明取错了列或拼了别的东西。
    """
    db_text, db_cn, db_path = _db_rows(
        "SELECT text, article_no_cn, path FROM article WHERE article_no = %s",
        (str(ARTICLE_NO),))[0]
    with _client(connect()) as client:
        response = client.get(f"/api/v1/law/minfadian/articles/{ARTICLE_NO}")
    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == {"request_id", "article_no", "article_no_cn", "path",
                         "text", "status", "law_id"}
    assert body["article_no"] == ARTICLE_NO          # int，不是字符串
    assert body["article_no_cn"] == db_cn == "五百八十四"
    assert body["path"] == db_path == ARTICLE_PATH
    assert body["text"] == db_text
    assert body["text"].startswith("第五百八十四条")
    # 效力来自 law_version（article_lookup 的既有口径），这里只钉它非空 ——
    # 取值来源由 test_article_lookup 的 SQL 文本用例守着，不重复
    assert body["status"] == "现行有效"
    assert body["law_id"] == "minfadian"


@requires_mysql
def test_a_missing_article_a_bad_law_id_and_a_non_numeric_number_are_404_alikes():
    """三种「取不到」一律 404，且与「路径不存在」逐字段同形（§二 第 9 条）。

    三个样本各挡一种退化：
      - 9999：条不存在 → 这是常态（用户记错条号），必须 404 而不是空 200；
      - notalaw：law_id 不在库 → fetch_articles 的 WHERE law_id 过滤掉，同样 404；
      - abc：**非数字条号**是最容易被写成 500 的那条 —— 'abc' 直接落进 SQL 的
        参数位时 pymysql 不会报错（VARCHAR 列的隐式转换恰好宽容），但那是巧合，
        一旦列类型收紧或改成数值列，同一条请求就变成 500。故实现显式 int() 转换、
        转不动判 404；本用例钉「不是 500」这一条，光有状态码断言挡不住它 ——
        500 也是「不是 200」，故这里把三种情形的响应体一起与「未知路径」比对。

    与未知路径同形是硬要求：若这一层的 404 与框架的 404 在文案或 code 上有一丝
    差别，扫描器就能把「路径存在但条号不对」从「没有这条路径」里分出来。
    """
    with _client(connect()) as client:
        missing = client.get("/api/v1/no-such-path")
        responses = {
            "no-such-article": client.get("/api/v1/law/minfadian/articles/9999"),
            "no-such-law": client.get("/api/v1/law/notalaw/articles/1"),
            "not-a-number": client.get("/api/v1/law/minfadian/articles/abc"),
        }
    for name, response in responses.items():
        assert response.status_code == 404, f"{name}: {response.status_code}"
        assert without_request_id(response) == without_request_id(missing), name


def test_the_law_endpoints_are_503_when_the_services_are_not_assembled():
    """装配没起来 → 503（不是 404、更不是 500）：那是「我们坏了」，与「这条不存在」
    是两种结论（§七 的 503/200 同族口径）。这条判据同时也是「本模块没有绕开
    deps.services_of 自己取连接」的证明 —— 绕过它的实现会在这里变成 500。"""
    factory_fn, _ = fake_factory()
    app = main.create_app(services_factory=factory_fn)
    # 不进 with：lifespan 不跑，app.state 上就没有 services
    probe = TestClient(app, raise_server_exceptions=False)
    nav = probe.get("/api/v1/law/nav")
    detail = probe.get("/api/v1/law/minfadian/articles/1")
    assert nav.status_code == 503 and detail.status_code == 503
    assert nav.json()["error"]["code"] == "service_unavailable"


# ---- build_nav：折叠规则本身（纯函数，离线可跑）----


def test_build_nav_folds_any_depth_and_keeps_articles_at_their_own_level():
    """折叠规则的四条性质，用一份造出来的行集一次钉全（不依赖真库的现状）。

    为什么值得单独测：真库今天有 109 条 path，但没有一条是另一条的前缀
    （实测 0 例），于是真库跑不出下面第 ③ 条那种形状；而「path 前缀冲突」
    是随时可能出现的（把几条条文挂到「第一编 总则」这一级就造出来了），
    出现时丢条或丢层都是静默的。

    四条性质：
      ①按 law_id 分树：两法同名路径（"第一编 总则"）不许合并 —— 合并后客户端
        的章节视图里会凭空多出别人家的章节；
      ②条号按**数值**排序：行集里给 "10" 与 "2"（字符串序会排反）；
      ③某条 path 恰是另一条的前缀时，**两级都保留**：上层节点既有 articles
        又有 children（今天真库没有这种节点，但规则不能靠数据现状成立）；
      ④level 与 path 由下标生成，三段路径就是三层、四段就是四层（不写死编/章/节）。
        样本必须**比真库更深**：真库最深四层（178 行挂在 17 个「节」节点上），
        样本只用三层时，把折树写死成 `[:3]` 的退化在离线用例里看不见
        （真库那条 `keeps_every_level...` 用例是另一道）。
    """
    rows = [
        ("minfadian", "第一编 总则", "1", "一"),
        ("minfadian", "第一编 总则 > 第一章 基本规定", "2", "二"),
        ("minfadian", "第一编 总则 > 第一章 基本规定", "10", "十"),
        ("minfadian", "第一编 总则 > 第一章 基本规定 > 第一节 目的", "3", "三"),
        ("minfadian", "第一编 总则 > 第一章 基本规定 > 第一节 目的 > 第一分节 释义",
         "4", "四"),
        ("otherlaw", "第一编 总则", "1", "一"),
    ]
    laws = law_mod.build_nav(rows)
    assert [law["law_id"] for law in laws] == ["minfadian", "otherlaw"]
    assert len(laws[0]["nodes"]) == 1, "两个 law 的节点必须分开成两棵树"
    assert laws[0]["nodes"] is not laws[1]["nodes"], "两法的树是同一份对象（被合并了）"
    assert [item["article_no"] for item in laws[1]["nodes"][0]["articles"]] == [1]
    top = laws[0]["nodes"][0]
    assert (top["title"], top["level"], top["path"]) == ("第一编 总则", 0, "第一编 总则")
    assert [item["article_no"] for item in top["articles"]] == [1]
    chapter = top["children"][0]
    assert (chapter["title"], chapter["level"]) == ("第一章 基本规定", 1)
    assert chapter["path"] == "第一编 总则 > 第一章 基本规定"
    assert [item["article_no"] for item in chapter["articles"]] == [2, 10]
    section = chapter["children"][0]
    assert (section["title"], section["level"], section["path"]) == (
        "第一节 目的", 2, "第一编 总则 > 第一章 基本规定 > 第一节 目的")
    assert [item["article_no"] for item in section["articles"]] == [3]
    subsection = section["children"][0]
    assert (subsection["title"], subsection["level"], subsection["path"]) == (
        "第一分节 释义", 3,
        "第一编 总则 > 第一章 基本规定 > 第一节 目的 > 第一分节 释义")
    assert [item["article_no"] for item in subsection["articles"]] == [4]


# ---- 律师侧 /qa 的嵌套投影（用例安置理由见函数 docstring）----

LEAK = "内部原文-SECRET"  # 响应体里绝不许出现的标记（越独特越可靠）


class _Answerer:
    """最小假 Answerer（本文件只有这一条用例走问答链路）：判整形，不判接线。"""

    def __init__(self, result) -> None:
        self.result = result

    def answer(self, question: str, side: str):
        return self.result


def test_the_lawyer_qa_nested_projection_hides_the_internal_block_fields(monkeypatch):
    """律师侧 /qa 的 `sources`/`citations` 是白名单投影：内部字段一个都不出去。

    为什么住在本文件：test_api_lawyer.py 的非空行已顶到 300 的单文件闸门（check_style），
    而这条判据属于「响应整形」那一族（设计 §四 的契约），故挪到这里（理由见修复报告）。

    为什么非有不可：`QAAnswer.sources`/`citations` 是 `list[dict]`，pydantic
    **不**过滤嵌套字典的键，顶层 7 键的字面量断言管不到这里；而 test_api_lawyer.py
    里所有 QAResult 的 sources/citations 都是空列表（`_result()` 的默认值 []）。
    实测：把 `lawyer_qa_payload` 的 `[source_payload(b) ...]` 换成 `[dict(b) ...]`
    → 那 11 条全绿，实拍响应却会带出 `text`（整条法条原文）、`status`、`law_id`、
    `chunk_id`。公众侧同类用例在 test_api_public.py，但两侧走的是**不同的整形
    函数**（public_qa_payload / lawyer_qa_payload）—— 公众侧挡住了的，律师侧不会
    自动挡住，这正是它被漏掉的原因（一度是同文件里两种标准）。判据取**字面量
    键集**：少一个契约键与多一个内部键都要有人看一眼。
    """
    monkeypatch.setenv(security.JWT_SECRET_ENV, JWT_SECRET)
    block = {"article_no": ARTICLE_NO, "article_no_cn": "五百八十四", "path": ARTICLE_PATH,
             "source": "vector", "rerank_score": 0.91, "text": LEAK,
             "status": "现行有效", "law_id": "minfadian", "chunk_id": "c-584"}
    result = QAResult(status="ok", answer="依《民法典》第五百八十四条……",
                      disclaimer="", failures=[], sources=[block],
                      citations=[Citation(law="中华人民共和国民法典", article="584",
                                          quote="当事人一方不履行合同义务……")])
    factory_fn, _ = fake_factory(FakeServices(answerer=_Answerer(result)))
    with TestClient(main.create_app(services_factory=factory_fn)) as client:
        response = client.post("/api/v1/qa", json={"question": "押金怎么退"},
                               headers=bearer(jwt_token()))
    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body["sources"][0]) == {"article_no", "path", "source", "rerank_score"}
    assert set(body["citations"][0]) == {"article", "paragraph", "item", "quote"}
    assert LEAK not in response.text
    assert "law_id" not in response.text and "chunk_id" not in response.text
