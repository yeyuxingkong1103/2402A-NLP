# 入库链路的质量决定检索质量，而检索质量决定费用区间能不能指回片段。
# 切分规则的测试用真实语料的前几段做夹具（不造数据）。
import pathlib
import sys

import pytest

# 与 test_ask.py / test_smoke_retrieval.py 同写法：把 tools/ 挂进搜寻路径
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# 本模块的公开面（brief 冻结的三个名字 + 入库链路的其余零件 + 检索）
from ingest_fee import (  # noqa: E402
    FEE_COLLECTION, FEE_DIR, FEE_FIELDS, FEE_FILTER_EXPR, FEE_OUTPUT_FIELDS,
    FEE_TOP_K, SOURCES, STATUS_VALID, build_rows, encode_rows, extract_source_no,
    iter_docs, main, search, split_paragraphs,
)

# 检索参数必须与法条侧同源（app/db/milvus 的那几个常量），故这里从那边取来比
from app.db.milvus import DENSE_TOP_K, RRF_K, SPARSE_TOP_K  # noqa: E402

# 精排的输入限额同样从法条侧取（不在这里抄一个数）：有 scorer 时粗排要按它扩召回
from app.retrieval.rerank import RERANK_INPUT_TOPK  # noqa: E402

# 真实语料文件（排除 _sources/ 与下划线开头的条目，与 main 同口径）
CORPUS = sorted(p for p in FEE_DIR.glob("*.txt") if not p.name.startswith("_"))

# 语料现状：4 份、合计 39 段（2026-09-29 实测）。语料增删文件时这条会红——
# 那正是提醒把 SOURCES 与报告里的数字一并更新，而不是让数字悄悄过期
EXPECTED_UNITS = 39

# 各文件实测段数：分文件断言才能挡住「某一份整个被漏掉、总数恰好由别的文件补齐」。
# 2026-09-29 语料重建后重测（26→39 段）：prepare 脚本现在**只让单行的短单元**
# 并入邻居、并以顶级小节（「一、」）为硬边界，于是「二、计时收费」这类短而完整的
# 小节不再被黏进下一节 —— 上一版把两个小节塞进同一段，检索命中后「依据」指向
# 段首那个、数字却来自另一个（真链路冒烟实证）
UNITS_PER_DOC = {
    "司法部等三部门关于规范律师服务收费的意见（司发通〔2021〕87号）": 9,
    "北京智深律师事务所收费标准": 14,
    "北京昌民律师事务所收费标准": 10,
    "北京融理律师事务所收费标准": 6,
}


def test_split_keeps_article_like_lines_as_separate_units():
    text = "第一条 律师服务收费实行政府指导价。\n\n第二条 收费标准由省制定。"
    units = split_paragraphs(text)
    assert len(units) == 2
    assert units[0].startswith("第一条")


def test_split_drops_empty_units():
    assert split_paragraphs("a\n\n\n\nb") == ["a", "b"]


def test_build_rows_carries_required_fields(tmp_path):
    doc = tmp_path / "收费办法.txt"
    doc.write_text("第一条 律师服务收费实行政府指导价。", encoding="utf-8")
    rows = build_rows(doc, region="全国", effective_date="2024-01-01")
    assert rows[0]["source_doc"] == "收费办法"
    assert rows[0]["source_no"] == "第一条"
    assert rows[0]["region"] == "全国"
    assert rows[0]["effective_date"] == "2024-01-01"
    assert rows[0]["status"] == "现行有效"
    assert rows[0]["text"]


def test_collection_name_is_frozen():
    assert FEE_COLLECTION == "fee_corpus"


# === 以下为真实语料与编排层的用例 ===

def _rows_of(stem: str) -> list[dict]:
    """取真实语料某一份的行。地区与生效日走 SOURCES，与 main 同一入口。"""
    meta = SOURCES[stem]
    return build_rows(FEE_DIR / f"{stem}.txt", meta["region"], meta["effective_date"])


def _all_rows() -> list[dict]:
    """全部真实语料的行。每次现算：切分与编号提取都是纯函数，直接读文件即可。"""
    return [row for stem in UNITS_PER_DOC for row in _rows_of(stem)]


def test_every_doc_is_split_into_its_measured_units():
    # 分文件比对实测段数：只断言总数 26 的话，把两份文件的行对调也照样绿
    for stem, expected in UNITS_PER_DOC.items():
        assert len(_rows_of(stem)) == expected, stem
    assert len(_all_rows()) == EXPECTED_UNITS
    # 语料目录本身就是这 4 份（多了少了都说明 SOURCES 该同步）
    assert sorted(p.stem for p in CORPUS) == sorted(UNITS_PER_DOC)


def test_iter_docs_skips_underscore_entries(tmp_path):
    # _sources/ 放原始下载件与来源说明，一旦被当成语料灌库，
    # 检索就会指回一份「来源说明」而不是收费口径 —— 且不会报错
    (tmp_path / "收费办法.txt").write_text("第一条 …", encoding="utf-8")
    (tmp_path / "_probe_sh.html.txt").write_text("占位", encoding="utf-8")
    (tmp_path / "_sources").mkdir()
    (tmp_path / "_sources" / "原件.txt").write_text("占位", encoding="utf-8")
    assert [p.name for p in iter_docs(tmp_path)] == ["收费办法.txt"]


def test_source_no_reads_real_headings():
    # 每条断言的首行都是语料原文（标注了出处），不是编的
    # 昌民「二、 计时收费」段（编号后是空格的写法）
    assert extract_source_no("二、 计时收费\n1． 普通律师 800~1000 元/小时") == "二、"
    # 智深「2、民事案件收费标准」段
    assert extract_source_no("2、民事案件收费标准\n2.1 基础服务费收费标准") == "2、"
    # 智深「2.2 风险代理收费标准」段（小数点式编号）
    assert extract_source_no("2.2 风险代理收费标准\n按基础费用+标的额比例差额累进") == "2.2"
    # 87 号「（一）提升律师服务收费合理化水平。」段（编号后直接是整句话）
    assert extract_source_no("（一）提升律师服务收费合理化水平。律师服务收费项目…") == "（一）"


def test_source_no_does_not_mistake_money_for_a_number():
    # 「7.4 万元…」是智深价目表里真实存在的一行 —— 下面两条先钉住它确实在语料里，
    # 免得夹具沦为一段与真实语料无关的字面量（语料重建前这行还当过段首：价目表
    # 被误切，它落到了新段开头）。7.4 是**费率**不是序号，取成 source_no 会让
    # 「依据」指向一个不存在的编号
    zhishen = (FEE_DIR / "北京智深律师事务所收费标准.txt").read_text(encoding="utf-8")
    assert "7.4 万元+标的额 100 万元以上部分的4%" in zhishen
    assert "43.4 万元+标的额 1000 万元以上部分的 2%" in zhishen
    assert extract_source_no("7.4 万元+标的额 100 万元以上部分的4%\n1000 万元以上部分") == ""
    assert extract_source_no("43.4 万元+标的额 1000 万元以上部分的 2%") == ""
    # 标题类段落没有编号，退化为空串（渲染方按空值跳过，不会印出「依据：文件 」）
    assert extract_source_no("法律服务收费标准（试行）\n一、刑事案件收费标准") == ""
    assert extract_source_no("北京智深律师事务所\n律师服务收费标准（2025 年）") == ""


def test_downstream_text_fields_are_always_str():
    # tools/ask.py 对 source_doc / source_no 做 " ".join(...)：非 str 会抛 TypeError，
    # 而那时答案已算好 —— 用户什么都看不到。law_chunks 的 article_no 是 int，别照抄
    for row in _all_rows():
        for key in ("source_doc", "source_no", "region", "effective_date", "status"):
            assert isinstance(row[key], str), (key, row[key])
        assert row["text"].strip()
        assert isinstance(row["chunk_id"], str)


def test_chunk_ids_are_stable_unique_and_come_from_the_source_code():
    # 主键显式且稳定 = 重跑 upsert 覆盖（幂等）。若改成 auto_id 或带随机段，
    # 这条会红 —— 那种实现重跑会让集合从 26 变 52 且不报错
    ids = [row["chunk_id"] for row in _all_rows()]
    assert len(ids) == len(set(ids)), "主键重复会把两段挤成一行"
    assert ids == [row["chunk_id"] for row in _all_rows()]
    for stem, meta in SOURCES.items():
        assert _rows_of(stem)[0]["chunk_id"].startswith(meta["code"] + "-")


def test_declared_varchar_lengths_fit_the_real_corpus():
    # Milvus 的 VARCHAR 长度按字节算，且**定短了是静默截断**——截掉的可能正是费用数字。
    # 这里拿真实语料的最长值去撞声明的上限，改小上限就会红
    limits = {name: params["max_length"] for name, _, params in FEE_FIELDS
              if "max_length" in params}
    for key in ("text", "source_doc", "source_no", "region", "effective_date", "status"):
        longest = max(len(row[key].encode("utf-8")) for row in _all_rows())
        assert longest <= limits[key], f"{key} 最长 {longest} 字节 > 上限 {limits[key]}"


def test_schema_and_rows_agree_on_field_names():
    # 行里多一个字段，Milvus 会因 enable_dynamic_field=False 直接拒收；
    # 少一个字段则下游 .get() 拿到 None。两边必须同名单
    declared = {name for name, _, _ in FEE_FIELDS}
    assert {"text", "source_doc", "source_no"} <= declared, "下游硬契约字段"
    assert declared - {"dense", "sparse"} == set(_all_rows()[0]), "行与 schema 字段名不一致"
# === 编排层：编码拼装与 main 的入库流程（Milvus 与模型都是替身） ===

class FakeModel:
    """替身 encoder：固定形状的假向量。真模型由 Step 5 的实跑出场。"""

    def __init__(self):
        self.seen: list[str] = []

    def encode(self, texts, batch_size=None, return_dense=True,
               return_sparse=True, return_colbert_vecs=False):
        self.seen += list(texts)
        return {"dense_vecs": [[0.0] * 1023 + [1.0]] * len(texts),
                "lexical_weights": [{1: 0.5, 2: 0.25}] * len(texts)}


class FakeClient:
    """替身 Milvus：只实现 main 会碰到的那几个方法，并记录被写入的行。"""

    def __init__(self, entities: int | None = None):
        # entities=None 表示「服务端实体数 == 本次提交条数」，即理想情况
        self.entities = entities
        self.upserted: list[list[dict]] = []
        self.flushed: list[str] = []

    def has_collection(self, name):
        return True  # 存在即跳过建集合：schema 由 test_schema_* 单独钉

    def upsert(self, name, rows):
        self.upserted.append(list(rows))
        return {"upsert_count": len(rows)}

    def flush(self, name):
        self.flushed.append(name)

    def query(self, name, filter=None, output_fields=None):
        got = len(self.upserted[0]) if self.entities is None else self.entities
        return [{"count(*)": got}]


def test_encode_rows_pairs_every_row_with_both_vectors():
    rows = _all_rows()
    model = FakeModel()
    encoded = encode_rows(rows, model)
    # 条数与顺序必须逐行对齐：zip 截断是静默的，少一行就等于丢了整段语料
    assert len(encoded) == len(rows)
    assert model.seen == [row["text"] for row in rows]
    for row, out in zip(rows, encoded):
        assert row["text"] == out["text"] and row["chunk_id"] == out["chunk_id"]
        assert len(out["dense"]) == 1024 and out["sparse"]


def test_main_pushes_every_row_and_checks_entity_count():
    # 这条同时验证：目录遍历（26 段全在）、编码、写库、以及实体数校验通过
    client, model = FakeClient(), FakeModel()
    stats = main(client=client, model=model, doc_dir=FEE_DIR)
    assert stats == {"docs": 4, "rows": EXPECTED_UNITS,
                     "written": EXPECTED_UNITS, "entities": EXPECTED_UNITS}
    assert len(client.upserted[0]) == EXPECTED_UNITS
    assert client.flushed == [FEE_COLLECTION]
    assert stats["written"] != 0, "0 条也会让实体数校验通过（0 == 0），故单独钉住"


def test_main_raises_when_entity_count_does_not_match_rows():
    # 语料删过文件（旧行还在）或主键撞了，实体数就会对不上。
    # 静默继续 = 检索阶段才暴露，且检索不会报错，只会给出过期口径
    client = FakeClient(entities=EXPECTED_UNITS + 3)
    with pytest.raises(RuntimeError, match="实体数"):
        main(client=client, model=FakeModel(), doc_dir=FEE_DIR)


def test_main_refuses_an_empty_corpus_dir(tmp_path):
    # 空目录会产出 0 行，而 0 == 0 能让实体数校验通过 —— 「成功」地把库灌成空。
    # 故这一支必须自己拦住（否则删空目录就是一个静默清库按钮）
    with pytest.raises(RuntimeError, match="没有语料"):
        main(client=FakeClient(), model=FakeModel(), doc_dir=tmp_path)


def test_main_refuses_a_doc_without_registered_metadata(tmp_path):
    # region / effective_date 是 build_rows 的入参，SOURCES 里没有的新文件
    # 只能靠调用方现填 —— 静默给个空串会让「全国/北京」这层元数据失去意义
    (tmp_path / "新加的所收费标准.txt").write_text("一、计时收费", encoding="utf-8")
    with pytest.raises(KeyError):
        main(client=FakeClient(), model=FakeModel(), doc_dir=tmp_path)


# === 检索：双路召回 + 效力过滤 + 空查询护栏（Task 9 的公开面，此前零用例） ===
# Task 9 报告自己把这条列成存疑 9：`search()` 与空查询护栏的唯一证据是真跑。
# 两个已发生过的回归都落在这里（N8 空查询假故障、效力过滤漏挂），故用替身补网。

class FakeSearchClient:
    """替身 Milvus 的检索面：记下 hybrid_search 的入参，返回预置命中行。"""

    def __init__(self, rows=()):
        self.rows = list(rows)
        self.calls: list[dict] = []

    def hybrid_search(self, collection, reqs, ranker=None, limit=None,
                      output_fields=None):
        self.calls.append({"collection": collection, "reqs": list(reqs),
                           "ranker": ranker, "limit": limit,
                           "output_fields": output_fields})
        # 返回结构照 Milvus：list[list[hit]]，字段在 hit["entity"] 里
        return [[{"entity": dict(row)} for row in self.rows]]


def test_search_returns_empty_for_a_blank_query_without_touching_services():
    """N8：案由认不出时查询串是空串，放它进编码器会抛「sparse 转换后为空」，
    被 attach 记成「费用信息暂时不可用」——把输入不合法说成服务故障。
    旁路两条都要断言：既没编码、也没发检索请求（只断言返回 [] 时，
    「先编码、异常前返回空」的实现照样绿）。"""
    client, model = FakeSearchClient(), FakeModel()
    assert search(client, model, "   ") == []
    assert model.seen == [] and client.calls == []


def test_search_sends_both_routes_with_the_status_filter():
    """双路请求（dense + sparse）、效力过滤挂在**每一条**上、字段集合取回 status。
    只钉「发了一次请求」的话，退化成单路的实现照样绿；过滤只挂一路时另一路
    仍能召回过期口径 —— 这条同时是设计 §七「不采用非现行片段」的主防线证据。"""
    row = {"text": "不超过1万元的，每件交纳50元", "source_doc": "诉讼费用交纳办法",
           "source_no": "第十三条", "status": "现行有效"}
    client = FakeSearchClient([row])
    got = search(client, FakeModel(), "民间借贷")
    call = client.calls[0]
    assert call["collection"] == FEE_COLLECTION
    assert call["output_fields"] == FEE_OUTPUT_FIELDS
    assert call["limit"] == FEE_TOP_K
    # 字段集合逐字钉住：漏掉 status 时编排层的效力兜底就无从判定（那正是 Task 9
    # 的缺口），漏掉溯源字段则「依据」印不出来
    assert FEE_OUTPUT_FIELDS == ["text", "source_doc", "source_no", "status"]
    assert [req.anns_field for req in call["reqs"]] == ["dense", "sparse"]
    assert [req.limit for req in call["reqs"]] == [DENSE_TOP_K, SPARSE_TOP_K]
    assert [req.expr for req in call["reqs"]] == [FEE_FILTER_EXPR] * 2
    assert FEE_FILTER_EXPR == 'status == "现行有效"'
    # 写侧（入库行写的 status）与读侧（检索的过滤条件）必须同源。检索面搬进
    # app/recommend/fee_search.py 之后，两边各有一处「现行有效」的字面量，
    # 分叉时检索不会报错、只是零命中 —— 用户看到的是「暂无费用口径依据」，
    # 而库里明明有 39 段。故在这里把两个值钉在一起，分叉即红
    assert FEE_FILTER_EXPR == f'status == "{STATUS_VALID}"'
    # 索引参数与排序口径从 app/db/milvus 取（法条侧同款断言），不在这里抄第二份。
    # pymilvus 只在私有属性 `_k` 上暴露 RRF 的 k（实测本机版本），故按它比对
    assert RRF_K == 60 and call["ranker"]._k == RRF_K
    assert got == [{name: row[name] for name in FEE_OUTPUT_FIELDS}]


def test_search_normalizes_every_returned_field_to_str():
    """返回行恒为 str（渲染方对 source_doc/source_no 做 " ".join）：缺字段或值为
    None 时退化成空串，而不是留一个 None 到渲染期才炸。status 退化成空串这一点
    还与编排层相接：空串不是现行 → 判拒（fail-closed），不会当成缺省放行。"""
    client = FakeSearchClient([{"text": "每件交纳50元"}])
    got = search(client, FakeModel(), "收费")
    assert set(got[0]) == set(FEE_OUTPUT_FIELDS)
    assert got[0]["source_doc"] == "" and got[0]["status"] == ""
    assert all(isinstance(value, str) for value in got[0].values())


# === 精排钩子（用户 2026-09-29 裁决：技术方案 6.4 的「混合检索 + 精排」缺了后半截） ===
# 默认 None 必须退回纯 RRF：本函数的既有调用方与用例行为一字不变，而精排要加载
# 2.2GB 模型，不能默认开。下面两条一正一反地钉住「有钩子才变、没钩子照旧」。


def test_search_reranks_with_the_injected_scorer_and_widens_recall():
    """有 scorer 时：粗排按精排输入限额扩召回、重排后的顺序即返回顺序、四个契约
    字段原样带过（fees.estimate 只读 text/status/source_doc/source_no）。

    只断言「传了 scorer」拦不住「search 内部把它丢掉」这类坏法，故断言点放在
    **顺序**上：替身精排给 B 段高分，返回的第一条就必须是 B 段。
    """
    rows = [{"text": "A 段：每件 1 元至 5 元", "source_doc": "甲所",
             "source_no": "一、", "status": "现行有效"},
            {"text": "B 段：每件 1 元至 10 元", "source_doc": "乙所",
             "source_no": "二、", "status": "现行有效"}]
    client, seen = FakeSearchClient(rows), []

    def scorer(pairs):
        seen.extend(pairs)
        return [0.1 if "A 段" in text else 0.9 for _, text in pairs]

    got = search(client, FakeModel(), "每件收费", scorer=scorer)
    assert client.calls[0]["limit"] == RERANK_INPUT_TOPK, "有精排时粗排要扩召回"
    assert seen == [("每件收费", rows[0]["text"]), ("每件收费", rows[1]["text"])]
    assert [hit["text"] for hit in got] == [rows[1]["text"], rows[0]["text"]]
    assert [hit["rerank_score"] for hit in got] == [0.9, 0.1]
    assert got[0]["source_no"] == "二、" and got[0]["status"] == "现行有效"


def test_search_without_a_scorer_keeps_the_rrf_only_limit():
    """没给 scorer 时限额就是 top_k（本批之前的纯 RRF 行为）。只钉「有 scorer 时
    扩到 30」的话，把扩召开放进无条件分支（默认也召回 30 条）照样绿 —— 那样
    每次问答都多取 27 条候选，纯 RRF 路径的行为被静默改变。"""
    client = FakeSearchClient([{"text": "收费", "status": "现行有效"}])
    search(client, FakeModel(), "收费")
    assert client.calls[0]["limit"] == FEE_TOP_K and RERANK_INPUT_TOPK != FEE_TOP_K
