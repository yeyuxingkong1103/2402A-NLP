# schema 测试分两层：FIELDS 是纯声明，可离线断言；建集合与索引要 Milvus 在线。
import pathlib

import pytest
from pymilvus import DataType

from app.db.milvus import (
    COLLECTION, DENSE_DIM, DENSE_INDEX, FIELDS, MILVUS_URI, SCALAR_INDEX_FIELDS,
    SPARSE_INDEX, build_index_params, build_schema, create_collection, describe, ensure_collection,
    entity_count, get_client, insert_chunks,
)

# 在线测试一律落在自己的集合上，绝不碰生产集合 law_chunks：全量回归就是质量闸门，
# 若这里用 COLLECTION，跑一次 pytest 就把 Task 5 灌进去的向量清空，而且不报错。
# 隔离靠显式传 name，不靠 monkeypatch——本行的 import 已把 COLLECTION 绑进本模块的
# 命名空间，改 app.db.milvus.COLLECTION 改不到这个名字，"以为隔离了其实没有"。
TEST_COLLECTION = "law_chunks_schema_test"


def _field(name):
    return next(f for f in FIELDS if f[0] == name)


def _sample_row() -> dict:
    """一条字段齐全的最小行。dense 用最简正交向量，本文件的测试都不算相似度。"""
    return {
        "chunk_id": "0123456789abcdef", "dense": [0.0] * (DENSE_DIM - 1) + [1.0],
        "sparse": {1: 0.5}, "law_id": "minfadian", "law_version": "v1",
        "article_no": 1, "article_no_cn": "一", "paragraph_no": None, "item_no": None,
        "path": "第一编 总则 > 第一章 基本规定", "status": "现行有效",
        "effective_date": "2021-01-01", "parent_id": None, "chunk_type": "father",
        "source_hash": "0" * 32, "text": "第一条 为了保护民事主体的合法权益。",
    }


def test_dense_dim_matches_bge_m3():
    # bge-m3 的 dense 是 1024 维；这个数字错了要重灌整个集合
    assert DENSE_DIM == 1024
    assert _field("dense")[2]["dim"] == 1024


def test_primary_key_is_chunk_id_varchar():
    # 与设计文档第七节一致：用 chunk_id 而非 auto_id，才拿得到重跑幂等
    assert _field("chunk_id")[2]["is_primary"] is True
    assert _field("chunk_id")[1] == DataType.VARCHAR
    assert _field("chunk_id")[2]["max_length"] == 16


def test_law_id_is_partition_key():
    # 技术方案 4.4 定的 partition_key，切多法规时不用改 schema
    assert _field("law_id")[2]["is_partition_key"] is True


def test_lengths_have_margin_over_measured_max():
    # 实测：text 最长 400 字、path 最长 50 字。定短了 Milvus 会截断正文且不报错
    assert _field("text")[2]["max_length"] >= 2000
    assert _field("path")[2]["max_length"] >= 255


def test_nullable_fields_are_the_expected_three():
    # 只有款号、项号、父指针可为空——其余字段空了属于数据异常，不该被 schema 放过
    nullable = {f[0] for f in FIELDS if f[2].get("nullable")}
    assert nullable == {"paragraph_no", "item_no", "parent_id"}


def test_scalar_indexes_cover_filter_fields():
    # 5.2 的过滤条件用到 status 与 chunk_type，effective_date 供时效过滤，
    # article_no 供条款号通路——这四个不建索引会全表扫
    assert set(SCALAR_INDEX_FIELDS) == {"status", "effective_date", "chunk_type", "article_no"}


def test_dense_index_params_match_spec():
    assert DENSE_INDEX["index_type"] == "HNSW"
    assert DENSE_INDEX["metric_type"] == "COSINE"
    assert DENSE_INDEX["params"] == {"M": 16, "efConstruction": 200}


def test_sparse_index_params_match_spec():
    assert SPARSE_INDEX["index_type"] == "SPARSE_INVERTED_INDEX"
    assert SPARSE_INDEX["metric_type"] == "IP"


def _milvus_available() -> bool:
    try:
        get_client().list_collections()
        return True
    except Exception:
        return False


requires_milvus = pytest.mark.skipif(not _milvus_available(), reason="Milvus 未在线")


def _production_count() -> int | None:
    """生产集合此刻的真实实体数；集合不存在返回 None——"不存在"也是待守状态之一。"""
    client = get_client()
    if not client.has_collection(COLLECTION):
        return None
    return entity_count(client)


@pytest.fixture(autouse=True)
def guard_production_collection():
    """每条测试前后比对生产集合 law_chunks 的实体数，变了就判失败。

    本文件的隔离只靠"记得传 name=TEST_COLLECTION"这个约定，漏一处（比如将来新增
    调用点忘了写）就是静默砸在生产库上：重建要 16.7 分钟 + 2.3GB 模型，而本项目
    无版本控制，砸了不可恢复。判据是**行数变化**而不是"一律禁止访问"——集成测试
    本来就只读生产集合，读不该算触碰。
    """
    try:
        before = _production_count()
    except Exception:
        # Milvus 不在线就无从比对，直接放过：本文件的纯声明测试必须能离线跑
        yield
        return
    yield
    after = _production_count()
    assert after == before, (
        f"生产集合 {COLLECTION} 的实体数被测试改动：{before} → {after}。"
        "要操作集合的测试必须传 name=TEST_COLLECTION")


def test_test_collection_is_not_the_production_collection():
    # 护住的是整个文件：两个名字一旦相等，下面每个 finally 里的 drop_collection
    # 就砸在生产向量库上，全量回归会静默清空 Task 5 的数据
    assert TEST_COLLECTION != COLLECTION


@requires_milvus
def test_create_collection_creates_all_fields_and_indexes():
    client = get_client()
    create_collection(client, drop_existing=True, name=TEST_COLLECTION)
    try:
        got = {f["name"] for f in describe(client, name=TEST_COLLECTION)}
        assert got == {f[0] for f in FIELDS}
        # list_indexes 返回字符串列表——这里正是被它绊过一次的地方
        assert set(client.list_indexes(TEST_COLLECTION)) == {
            "dense", "sparse", *SCALAR_INDEX_FIELDS}
    finally:
        client.drop_collection(TEST_COLLECTION)


@requires_milvus
def test_ensure_collection_is_idempotent():
    # 重跑入库不该因为"集合已存在"而炸，也不能把已有数据重建掉——
    # 只断言 has_collection 的话，ensure 里写成 drop 再 create 照样能过
    client = get_client()
    create_collection(client, drop_existing=True, name=TEST_COLLECTION)
    try:
        assert insert_chunks(client, [_sample_row()], name=TEST_COLLECTION) == 1
        ensure_collection(client, name=TEST_COLLECTION)
        ensure_collection(client, name=TEST_COLLECTION)
        assert client.has_collection(TEST_COLLECTION)
        assert entity_count(client, name=TEST_COLLECTION) == 1
    finally:
        client.drop_collection(TEST_COLLECTION)


@requires_milvus
def test_insert_and_upsert_are_idempotent():
    # 同一 chunk_id 写两次，实体数必须还是 1——这正是选 VARCHAR 主键换来的性质
    client = get_client()
    create_collection(client, drop_existing=True, name=TEST_COLLECTION)
    try:
        row = _sample_row()
        assert insert_chunks(client, [row], name=TEST_COLLECTION) == 1
        client.flush(TEST_COLLECTION)
        assert entity_count(client, name=TEST_COLLECTION) == 1
        assert insert_chunks(client, [row], name=TEST_COLLECTION) == 1
        client.flush(TEST_COLLECTION)
        assert entity_count(client, name=TEST_COLLECTION) == 1
    finally:
        client.drop_collection(TEST_COLLECTION)


class _FakeClient:
    """只记录调用、不连 Milvus 的替身，用来走真服务端够不到的两条分支。

    实测 schema 不符 / 类型错 / 超长 / 同批重复主键，一律由 pymilvus 直接抛异常，
    服务端不会"少收几条"——所以写多少条就是多少条，upsert_count 偏小的路只能在
    替身上走。这也是为什么守卫测试不能拿"会被拒收的行"来写：那种行在 upsert 调用
    处就炸了，删掉守卫测试照样过，等于没测。
    """

    def __init__(self, accepted: int | None = None):
        self.accepted = accepted
        self.calls: list[tuple[str, str]] = []

    def upsert(self, name, rows):
        self.calls.append(("upsert", name))
        accepted = len(rows) if self.accepted is None else self.accepted
        return {"upsert_count": accepted,
                "ids": [row["chunk_id"] for row in rows[:accepted]]}

    def flush(self, name):
        self.calls.append(("flush", name))


def _sample_rows(n: int) -> list[dict]:
    """n 条主键互不相同的行。真 Milvus 同批重复主键会被拒收，替身测试不踩这个坑。"""
    rows = []
    for i in range(n):
        row = _sample_row()
        row["chunk_id"] = f"{i:016d}"
        rows.append(row)
    return rows


def test_insert_rejects_empty_rows():
    # 空列表时不该向 Milvus 发出任何请求。断言 flush 而非只断言 upsert：
    # 守卫删掉后 range(0, 0, ...) 本就不迭代，upsert 自然不会被调用，
    # 唯一会冒出来的调用就是 flush——那才是 `if not rows` 真正省下的动作。
    fake = _FakeClient()
    assert insert_chunks(fake, [], name=TEST_COLLECTION) == 0
    assert fake.calls == []


def test_insert_raises_when_milvus_accepts_fewer_than_submitted():
    # 返回 len(rows) 会让这道门槛退化成 input == input，少写也报满额成功；
    # 数目的唯一凭证是 upsert 回报的 upsert_count，对不上必须中断（Global Constraint）
    rows = _sample_rows(3)
    # 先证明同一个替身在计数相符时能正常返回——否则"抛异常"可能来自别的毛病
    assert insert_chunks(_FakeClient(), rows, name=TEST_COLLECTION) == 3
    with pytest.raises(RuntimeError, match="提交 3 条.*实际接受 2 条"):
        insert_chunks(_FakeClient(accepted=2), rows, name=TEST_COLLECTION)


class _FlakyClient:
    """前 fail_times 次 upsert 抛异常、之后正常返回的替身，模拟瞬时网络抖动。"""

    def __init__(self, fail_times: int):
        self.fail_times = fail_times
        self.upsert_calls = 0
        self.flushed = False

    def upsert(self, name, rows):
        self.upsert_calls += 1
        if self.upsert_calls <= self.fail_times:
            raise RuntimeError("模拟瞬时抖动")
        return {"upsert_count": len(rows)}

    def flush(self, name):
        self.flushed = True


def test_insert_retries_transient_failure(monkeypatch):
    # 设计文档第六节要求"写入失败重试 3 次"：一趟灌库 16.7 分钟，
    # 一次抖动不该让它整个白跑。只断言"抛不抛"的写法删掉重试照样过，
    # 所以这里数的是 upsert 的调用次数
    sleeps: list[float] = []
    monkeypatch.setattr("app.db.milvus.time.sleep", sleeps.append)
    flaky = _FlakyClient(fail_times=2)
    assert insert_chunks(flaky, _sample_rows(2), name=TEST_COLLECTION) == 2
    assert flaky.upsert_calls == 3
    assert flaky.flushed
    # 退避必须短：这是批处理里的等，不是给人看的
    assert sum(sleeps) <= 5


def test_insert_gives_up_after_three_failed_attempts(monkeypatch):
    # 重试要封顶：真故障（schema 不符、服务掉了）重试多少次都不会好，
    # 而"无限重试"会把已经炸了的灌库拖成看起来卡住
    monkeypatch.setattr("app.db.milvus.time.sleep", lambda _: None)
    flaky = _FlakyClient(fail_times=99)
    with pytest.raises(RuntimeError, match="模拟瞬时抖动"):
        insert_chunks(flaky, _sample_rows(2), name=TEST_COLLECTION)
    assert flaky.upsert_calls == 3
