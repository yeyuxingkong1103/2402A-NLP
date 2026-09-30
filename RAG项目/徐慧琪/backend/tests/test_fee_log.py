# 留痕是 AC-21 要求的证据（「参考区间」不能只是嘴上说说，要能追溯每次
# 生成用了哪个片段）。这里真连 MySQL —— 它同时是「表结构可用」的集成测。
# 注意：本表按 AC-19 不得记录用户问题原文或任何身份信息。
import uuid
from unittest.mock import MagicMock

import pytest

from app.db.mysql import connect
from app.recommend import fee_log, fees


def _unique(prefix: str) -> str:
    """每次跑都换一个标记值（request_id 或 cause）。"""
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _row_where(conn, column: str, value):
    """按**本轮唯一**的标记取回本用例写的那一行。

    为什么不取「最新一行」：留痕表只增不删，上一次跑留下的行还在库里，而它的内容
    往往与本次期望逐字节相同 —— 取最新一行会拿到那行陈货，于是「本次根本没写库」
    也照样判绿（实测把 record 改成纯 no-op 后仍有 3 条绿）。按唯一标记选行，
    「查得到」就只能来自本次插入。
    无法用 request_id 做标记的用例（那一格按题意就是 NULL）改用唯一的 cause。
    column 是测试自己写死的字面量、不来自外部输入，故可拼进 SQL。
    """
    with conn.cursor() as cur:
        cur.execute("SELECT cause, snippet_id, range_low, range_high, model, "
                    f"request_id, status FROM fee_generation_log "
                    f"WHERE {column} = %s ORDER BY id DESC LIMIT 1", (value,))
        return cur.fetchone()


def _forget(conn, column: str, value) -> None:
    """清掉本用例写的那一行：留痕表不该被测试挤满（只删自己这行，不动别的）。"""
    with conn.cursor() as cur:
        cur.execute(f"DELETE FROM fee_generation_log WHERE {column} = %s", (value,))
    conn.commit()

# 与 test_fees.py 同源的片段：数字 1/50 都在文内，便于构造 ok 与 rejected 两向
SNIPPET = "不超过1万元的，每件交纳50元"


@pytest.fixture()
def conn():
    connection = connect()
    fee_log.ensure_table(connection)
    yield connection
    connection.close()


def test_ensure_table_is_idempotent(conn):
    """跑两次不报错 —— 启动时无条件调用它。"""
    fee_log.ensure_table(conn)


def test_ensure_table_executes_the_ddl_and_commits():
    """上面那条只能证明「跑两次不报错」—— ensure_table 若被改成 `pass`，本机因为表
    早就建好了，其余用例查的仍是那张老表，**整份测试照样全绿**（实测 K8 就是这样），
    要等哪次全新部署才会炸。故这里用假连接钉住：它确实把 DDL 交给游标执行了，
    也确实提交了 —— 这是唯一不破坏现存留痕行（AC-21 的证据）又能验证「会建表」的
    办法，删表重建那种写法会把真实留痕一起清掉。"""
    fake = MagicMock()
    fee_log.ensure_table(fake)
    fake.cursor.return_value.__enter__.return_value.execute.assert_called_once_with(
        fee_log.DDL)
    fake.commit.assert_called_once()


def test_record_writes_a_row(conn):
    request_id = _unique("req-brief-1")
    fee_log.record(conn, cause="民间借贷纠纷", snippet_id="第十三条",
                   range_low=1, range_high=10, model="deepseek-chat",
                   request_id=request_id, status="ok")
    row = _row_where(conn, "request_id", request_id)
    assert row is not None, "record 一行都没写进去"
    assert row[0] == "民间借贷纠纷"
    assert row[1] == "第十三条"
    assert float(row[2]) == 1 and float(row[3]) == 10
    assert row[6] == "ok"
    _forget(conn, "request_id", request_id)


def test_record_accepts_missing_numbers_for_rejected_rows(conn):
    """回查被拒时没有区间可记 —— 那一行仍要落库（「拒绝过什么」也在追溯范围里）。

    这一行的 request_id 按题意就是 NULL，标记只能用 cause（唯一值），
    否则「取最新一行」会拿到上次跑剩的陈货，record 变成 no-op 也照样绿。
    """
    cause = _unique("cause-rejected")
    fee_log.record(conn, cause=cause, snippet_id=None,
                   range_low=None, range_high=None, model=None,
                   request_id=None, status="rejected")
    row = _row_where(conn, "cause", cause)
    assert row is not None, "判拒的那一行没落库"
    assert row[6] == "rejected"
    _forget(conn, "cause", cause)


def test_make_logger_adapter_calls_record(conn):
    """make_logger 的产物要能直接喂给 fees.estimate(log_fn=...)。"""
    request_id = _unique("req-brief-2")
    logger = fee_log.make_logger(conn)
    logger(cause="劳动争议", snippet_id="第五条", range_low=2, range_high=5,
           model="qwen2.5:3b", request_id=request_id, status="ok")
    row = _row_where(conn, "request_id", request_id)
    assert row is not None, "适配器没有写进任何一行"
    assert row[0] == "劳动争议" and row[6] == "ok"
    _forget(conn, "request_id", request_id)


def test_record_is_visible_to_another_connection(conn):
    """record 必须**提交**，否则真实调用方那边这行等于没写。

    为什么非另开一条连接不可：pymysql 默认 autocommit=False，未提交的行在**同一个
    连接**里可见 —— 所以「写没写、提没提交」用同一条连接永远看不出差别（实测把
    record 的 commit 删掉，全套用例照绿）。真实调用方是另一个会话（API 进程、
    审计查询），这里就照那个形态验：另开连接读回，读不到即红。
    这也是给 record 补的那半边 —— ensure_table 早有 commit 断言，record 之前没有。
    """
    request_id = _unique("req-commit")
    fee_log.record(conn, cause="劳动争议", snippet_id="第五条", range_low=2,
                   range_high=5, model="qwen2.5:3b", request_id=request_id,
                   status="ok")
    other = connect()
    try:
        row = _row_where(other, "request_id", request_id)
    finally:
        other.close()
    assert row is not None, "本连接看得见、别的连接看不见 —— record 没有提交"
    assert row[6] == "ok" and row[5] == request_id
    _forget(conn, "request_id", request_id)


def test_table_has_no_column_for_user_text(conn):
    """AC-19 的工程化形式：表结构里根本不给出存问题原文的列。"""
    with conn.cursor() as cur:
        cur.execute("SHOW COLUMNS FROM fee_generation_log")
        columns = {row[0].lower() for row in cur.fetchall()}
    assert "question" not in columns
    assert "user_text" not in columns
    assert "ip" not in columns


# ---- 以下为 brief 之外补的用例：brief 的断言在这几处改坏不会红（逐条配了刀） ----

def test_every_column_lands_in_its_own_field(conn):
    """上面那条只读了 cause/snippet_id/range/status 五格，model 与 request_id
    两格空着 —— 把这两列的值写串（model 写进 request_id 列）它照样绿。AC-21 的
    追溯恰恰靠这两格指认「哪次请求、哪个模型」，故把七格**逐格比一遍**。"""
    request_id = _unique("req-fields")
    fee_log.record(conn, cause="劳动争议", snippet_id="第五条",
                   range_low=2, range_high=5, model="qwen2.5:3b",
                   request_id=request_id, status="ok")
    row = _row_where(conn, "request_id", request_id)
    assert row is not None, "record 一行都没写进去"
    assert tuple(row) == ("劳动争议", "第五条", row[2], row[3],
                          "qwen2.5:3b", request_id, "ok")
    assert float(row[2]) == 2 and float(row[3]) == 5
    _forget(conn, "request_id", request_id)


def test_missing_fields_become_null_not_default(conn):
    """被拒行只断言了 status —— 实现若把缺的字段填上默认值（比如 cause 兜成
    空串、range 兜成 0）同样绿，而那会让「没有区间」在追溯时看起来像「区间是 0」。
    故整行比一遍：除 cause 与 status 外必须是 NULL。
    标记同样只能用 cause：其余五格按题意是 NULL。"""
    cause = _unique("cause-blank")
    fee_log.record(conn, cause=cause, snippet_id=None, range_low=None,
                   range_high=None, model=None, request_id=None,
                   status="rejected")
    row = _row_where(conn, "cause", cause)
    assert row is not None, "record 一行都没写进去"
    assert tuple(row) == (cause, None, None, None, None, None, "rejected")
    _forget(conn, "cause", cause)


def test_make_logger_takes_the_kwargs_estimate_sends(conn):
    """跨任务的真实契约：estimate 调 log_fn 时传的就是这七个 kwargs（Task 4 的
    `common` 四项 + range_low/range_high/status）。只测「logger 能被手工调用」挡不住
    签名不合 —— 必须让**真的 estimate** 来调它，再读库核对。

    清理按**唯一 cause** 且放在 finally 里，两条缺一不可：一旦走进效力闸门那一支
    （hits 的 status 被改成非现行、或被删掉这个键），那一行写的是 cause，而
    model / request_id 按题意是 NULL —— 按 request_id 删不掉；而断言失败又会让
    清理语句根本走不到。两件事叠加就是永远删不掉的残留行（表里 id 269/270 正是
    这样留下的，2026-09-29 已清）。cause 因此也必须是唯一值：写死的案由只说明
    「这行属于本用例」，事后没人能确定它是不是上一轮跑剩的。
    """
    hits = [{"text": SNIPPET, "source_no": "第十三条",
             "source_doc": "诉讼费用交纳办法", "status": "现行有效"}]
    request_id = _unique("req-ok")
    cause = _unique("cause-ok")
    # unit 必须报「万元」：片段写的是「1万元」，缺省闸门对紧邻量级词的数会判拒
    # （那样本用例就走不到 ok 支，测到的就不是留痕的字段链了）
    produced = {"low": 1, "high": 50, "unit": "万元", "model": "deepseek-chat",
                "request_id": request_id}
    try:
        result = fees.estimate(cause, search_fn=lambda c: hits,
                               generate_fn=lambda text, c: produced,
                               log_fn=fee_log.make_logger(conn))
        assert result["status"] == "ok"
        row = _row_where(conn, "request_id", request_id)
        assert row is not None, "estimate 跑过 ok 分支却没留下任何一行"
        assert row[0] == cause and row[1] == "第十三条"
        assert float(row[2]) == 1 and float(row[3]) == 50
        assert row[4] == "deepseek-chat" and row[5] == request_id and row[6] == "ok"
    finally:
        _forget(conn, "cause", cause)


def test_rejected_estimate_row_keeps_the_rejected_range(conn):
    """判拒的分支同样要留痕，且记的是**被拒的原始区间**（而不是两个 NULL）——
    事后要看得出「拒的是 1~999」。若把被拒行写成空区间，追溯就丢了它唯一的信息。
    清理口径与上一条同源（唯一 cause + finally），理由见那边的 docstring。"""
    hits = [{"text": SNIPPET, "source_no": "第十三条",
             "source_doc": "诉讼费用交纳办法", "status": "现行有效"}]
    request_id = _unique("req-rejected")
    cause = _unique("cause-rejected")
    produced = {"low": 1, "high": 999, "model": "deepseek-chat",
                "request_id": request_id}
    try:
        result = fees.estimate(cause, search_fn=lambda c: hits,
                               generate_fn=lambda text, c: produced,
                               log_fn=fee_log.make_logger(conn))
        assert result["status"] == "rejected"
        row = _row_where(conn, "request_id", request_id)
        assert row is not None, "判拒的分支没留痕，AC-21 的「拒绝过什么」就断了"
        assert row[0] == cause and row[1] == "第十三条"
        assert float(row[2]) == 1 and float(row[3]) == 999
        assert row[6] == "rejected"
    finally:
        _forget(conn, "cause", cause)


def test_record_tolerates_keys_the_caller_omits_entirely(conn):
    """上面那条「缺字段」用例实际把七个字段**全传了**（值给 None），所以它钉的是
    「None 能落库」而不是「键可以不给」—— 实测把 `fields.get(name)` 换成
    `fields[name]`（缺键就 KeyError 的实现）它照样绿。而 record 的契约写的是
    「缺的记 NULL」，那就得有一条真的**不传键**的用例来钉住它。"""
    cause = _unique("cause-omitted")
    fee_log.record(conn, cause=cause, status="rejected")
    row = _row_where(conn, "cause", cause)
    assert row is not None, "record 一行都没写进去"
    assert tuple(row) == (cause, None, None, None, None, None, "rejected")
    _forget(conn, "cause", cause)


def _split_top_level(body: str):
    """按**顶层逗号**切分定义块：括号里的逗号（DECIMAL(12,2)）不是分隔符。"""
    parts, current, depth = [], [], 0
    for ch in body:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
            continue
        current.append(ch)
    parts.append("".join(current))
    return parts


def _ddl_columns():
    """从 DDL 里刮出列名：先按括号配对取出定义块，再按顶层逗号切出每条定义。

    为什么不能按行首 token 数（上一版就是这么写的）：往已有列的**行尾**追加一个新列
    （`status ... , user_input TEXT NULL`）时，那一行的行首 token 仍是 status ——
    列名集里看不出多了东西，两条白名单同时被骗过（IF NOT EXISTS 又让活库那条也
    看不见）。按定义切则多一个定义就多一个名字。
    口径的代价：注释文本里若出现半角逗号，会被当成定义分隔符切出一个碎片，其首
    token 是噪声 —— 方向是多出名字而判红，属 fail-loud。
    """
    text = fee_log.DDL
    start = text.index("(")
    depth = 0
    body = ""
    for offset, ch in enumerate(text[start:], start):
        depth += 1 if ch == "(" else -1 if ch == ")" else 0
        if ch == ")" and depth == 0:
            body = text[start + 1:offset]
            break
    names = set()
    for definition in _split_top_level(body):
        tokens = definition.split()
        head = tokens[0].lower() if tokens else ""
        # INDEX/PRIMARY KEY 这类是索引声明，不是列
        if head in {"", "primary", "index", "key", "unique", "constraint"}:
            continue
        names.add(head)
    return names


def test_ddl_declares_exactly_the_seven_columns():
    """上面那条查的是**库里现有的表**，而 DDL 是 IF NOT EXISTS —— 表一旦建出来，
    之后谁把 DDL 改成多一列「question」，那张老表纹丝不动，查表结构的用例照绿，
    要等哪次重建库才会出问题（那时已经上线了）。故这里直接钉**将要建表的文本**：
    AC-19 要管的是「发出去的建表语句里根本没有能存问题原文的列」。"""
    assert _ddl_columns() == {"id", "ts", "cause", "snippet_id", "range_low",
                              "range_high", "model", "request_id", "status"}


def test_columns_are_exactly_the_documented_seven(conn):
    """AC-19 那句「表里没有这些列」若只按名字黑名单查（question/user_text/ip），
    换成 `user_question`、`user_input`、`client_ip` 就全都查不出来 —— 黑名单天然
    漏项。真正的工程化形式是**白名单**：列集合必须恰好等于被裁决过的这七列
    （id/ts 是主键与库侧时间戳），多一列都说明有人往里塞了新东西（含身份信息，
    也含被裁决不加的 unit），必须先过裁决。"""
    with conn.cursor() as cur:
        cur.execute("SHOW COLUMNS FROM fee_generation_log")
        columns = {row[0].lower() for row in cur.fetchall()}
    assert columns == {"id", "ts", "cause", "snippet_id", "range_low",
                       "range_high", "model", "request_id", "status"}
