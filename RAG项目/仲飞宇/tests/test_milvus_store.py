"""MilvusStore 的连接生命周期。

只测「连接是怎么建立的」，不碰真的 Milvus：Milvus Lite 一个进程只能开一个，
真正的连接行为在集成/手工验证里覆盖。
"""
from __future__ import annotations

import threading
import time

from app.core.config import Settings
from app.core.store import milvus_store
from app.core.store.milvus_store import role_filter, MilvusStore


class _FakeClient:
    """记录被构造了几次，并在构造里睡一会以放大竞态窗口。"""

    instances = 0
    collections_calls = 0

    def __init__(self, uri: str):
        type(self).instances += 1
        time.sleep(0.05)
        self.uri = uri

    def list_collections(self):
        type(self).collections_calls += 1
        return []


def test_concurrent_first_connect_creates_one_client(monkeypatch):
    """并发首次连接只能起一个客户端。

    这是启动时那条「Milvus 不可达 + DataDirLockedError」的来源：同步路由在线程池里
    跑，两个请求同时首次访问，各自去开 Milvus Lite 的嵌入式服务，抢数据目录的锁。
    实测启动 11 秒后的 /health 就撞上过：两条 DataDirLockedError，随后服务才起来。
    """
    _FakeClient.instances = 0
    monkeypatch.setattr(milvus_store, "MilvusClient", _FakeClient)
    store = MilvusStore(Settings(milvus_uri="./data/milvus.db"))

    threads = [threading.Thread(target=store.connect) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert _FakeClient.instances == 1, f"起了 {_FakeClient.instances} 个客户端，说明首次连接没串行化"


def test_ping_reports_unreachable_without_raising(monkeypatch, caplog):
    """连不上时 ping 返回 False 并留痕，不抛异常（健康检查要能降级返回）。"""
    def boom(*_a, **_kw):
        raise RuntimeError("Open local milvus failed")

    monkeypatch.setattr(milvus_store, "MilvusClient", boom)
    store = MilvusStore(Settings(milvus_uri="./data/milvus.db"))

    assert store.ping() is False
    assert "Milvus 不可达" in caplog.text
    assert "另一个进程" in caplog.text, "要把本地 Lite 的锁这层原因说出来，否则只有一句 Open local milvus failed"


class _CreateRaceClient:
    """替身：模拟「建集合」这段 check-then-act 的竞态。

    真实表现（Milvus Lite，2026-09-21 实测）：全新库上 2 线程并发首次上传，
    输的那个收 `MilvusException: File exists: collections/role_knowledge`，
    该请求 500 且 chunk 一条没落库（embedding 白花）。这里用「建完才算存在」
    + 建的时候睡一会来忠实复刻这个窗口。
    """

    def __init__(self, uri: str):
        self.created = False
        self.create_calls = 0

    def has_collection(self, name: str) -> bool:
        return self.created

    def create_schema(self, **_kw):
        class _Schema:
            def add_field(self, *_a, **_kw):
                pass

        return _Schema()

    def prepare_index_params(self):
        class _P:
            def add_index(self, *_a, **_kw):
                pass

        return _P()

    def create_collection(self, collection_name: str, schema, index_params) -> None:
        self.create_calls += 1
        time.sleep(0.02)  # 建集合有点耗时，留出"另一个线程也看到不存在"的窗口
        if self.create_calls > 1:
            raise RuntimeError(f"File exists: collections/{collection_name}")
        self.created = True


def test_concurrent_first_insert_creates_collection_once(monkeypatch):
    """并发首次入库只能建一次集合。

    `connect()` 早就有 `_connect_lock`，但「查存在性 -> 建集合」这段复合操作没人管，
    两个同步请求同时首次上传就双双看到 has_collection=False，一起去建。
    """
    monkeypatch.setattr(milvus_store, "MilvusClient", _CreateRaceClient)
    store = MilvusStore(Settings(milvus_uri="./data/milvus.db"))
    errors: list[str] = []
    barrier = threading.Barrier(2)

    def worker(role_id: str):
        try:
            barrier.wait()
            store.create_collection()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{role_id}: {exc}")

    threads = [threading.Thread(target=worker, args=(f"role{i}",)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"并发建集合炸了：{errors}"
    assert store.client.create_calls == 1, "建了不止一次，说明 check-then-act 没串行化"


class _DeleteStubClient:
    """替身：删除路径用到的那几个方法，附带模拟真实 Milvus 的 delete_count 行为。"""

    def __init__(self, uri: str, matched: int = 3, query_raises: bool = False):
        self.matched = matched
        self.query_raises = query_raises
        self.deleted = []

    def has_collection(self, name: str) -> bool:
        return True

    def load_collection(self, name: str) -> None:
        pass

    def query(self, **_kw):
        if self.query_raises:
            raise RuntimeError("count(*) 不被支持")
        return [{"count(*)": self.matched}]

    def delete(self, name: str, filter=None):  # noqa: A002 - 跟 pymilvus 的签名一致
        self.deleted.append(filter)
        return {"delete_count": 0}  # 真实表现：过滤删除时恒为 0


def test_delete_by_source_reports_real_count(monkeypatch):
    """返回值必须是真删了几条——不能用 Milvus 的 delete_count。

    实测（Milvus Lite，2026-09-21）：插入 3 条后 count() 38 → 删除 → 35，删除确实生效，
    而 delete_count 恒为 0。若照抄它，调用方的日志会永远打印「先删除旧数据 0 条」。
    """
    monkeypatch.setattr(milvus_store, "MilvusClient", _DeleteStubClient)
    store = MilvusStore(Settings(milvus_uri="./data/milvus.db"))

    assert store.delete_by_source("psychologist", "probe://x") == 3
    assert store.client.deleted == [
        'role_id == "psychologist" and source == "probe://x"'
    ], "删除条件必须同时限定角色与来源：多角色共享一张表，只按 source 删会误删别的角色"


def test_delete_by_source_still_deletes_when_count_unavailable(monkeypatch):
    """统计失败不能连删除一起跳过——退化成「删了但不知道几条」，返回 0。"""
    monkeypatch.setattr(
        milvus_store, "MilvusClient", lambda uri: _DeleteStubClient(uri, query_raises=True)
    )
    store = MilvusStore(Settings(milvus_uri="./data/milvus.db"))

    assert store.delete_by_source("psychologist", "probe://x") == 0
    assert len(store.client.deleted) == 1, "数不出来也必须照删"


def test_role_filter_rejects_expression_injection():
    """角色 id 会拼进过滤表达式，必须白名单。"""
    import pytest

    assert role_filter("psychologist") == 'role_id == "psychologist"'
    for bad in ('x" or role_id != "', "有中文", "", "a" * 65):
        with pytest.raises(ValueError):
            role_filter(bad)
