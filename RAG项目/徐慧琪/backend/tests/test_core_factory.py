# 工厂是「CLI 与 HTTP 共用装配」的落点（设计 §三 定调第 3 条），所以这里钉的全是
# **装配契约**：重资源各装一次且是同一对象、附加区块只对公众侧装、路径先验后加载。
# 真正的加载与连库不在这层测 —— 那由 CLI 真跑（tools/ask.py）覆盖，本文件的替身
# 只回答「工厂把谁递给了谁、递了几次」。
import logging
import os
import pathlib
import subprocess
import sys

import pytest

from app.core import config, factory
from app.db.milvus import MILVUS_URI
from app.db.mysql import DEFAULT_CONFIG
from app.generation.profiles import SIDE_INTERNAL, SIDE_PUBLIC


def _stub_heavy_resources(monkeypatch) -> tuple[list[tuple], dict]:
    """把工厂里每个**对外构造点**换成替身，返回 (调用流水, 替身对象表)。

    流水里记的是**名字与顺序**：装了几次、按什么次序装，都是这份契约的一部分。

    替身清单必须覆盖工厂里每一个建连接/建对象的名字：漏一个的表现是「真去连了
    本机 MySQL」——本机在跑时它照样绿（连上了），只在库不在时才红，而那时报的
    是驱动异常、看起来与用例要测的东西无关（任务 4 加 account_conn 时实测踩到）。
    """
    calls: list[tuple] = []
    objects = {name: object() for name in
               ("conn", "account_conn", "client", "encoder", "reranker", "extras_fn")}

    def make(target: str, name: str):
        def build(*args, **kwargs):
            calls.append((target, args, kwargs))
            return objects[name]
        return build

    for target, name in (("connect", "conn"),
                         ("connect_autocommit", "account_conn"),
                         ("get_client", "client"),
                         ("load_model", "encoder"), ("load_reranker", "reranker"),
                         ("build_extras_fn", "extras_fn")):
        monkeypatch.setattr(f"app.core.factory.{target}", make(target, name))
    return calls, objects


class _RecordingAnswerer:
    """替身 Answerer：只记下工厂递进来的 kwargs，不跑任何链路。"""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _cfg(tmp_path, *, embed_files=("sparse_linear.pt",),
         rerank_files=("model.safetensors",)) -> config.Config:
    """造一份路径有效的配置（临时目录里的占位文件），返回值。"""
    paths = []
    for name, files in (("embed", embed_files), ("rerank", rerank_files)):
        directory = tmp_path / name
        directory.mkdir()
        for filename in files:
            (directory / filename).write_text("placeholder", encoding="utf-8")
        paths.append(str(directory))
    return config.Config(milvus_uri=MILVUS_URI, mysql=dict(DEFAULT_CONFIG),
                         embed_model_path=paths[0], reranker_model_path=paths[1])


def test_every_heavy_resource_is_built_exactly_once(monkeypatch, tmp_path):
    """四件重资源 + 账号连接各构造一次，且顺序固定（连接 → 账号连接 → 客户端
    → 编码器 → 精排）。

    次数是这条的要点：加载两次 bge-m3 不会报错，只会让显存翻倍、让两侧各自
    持一份可能分叉的口径 —— 正是设计 §三 要消掉的那类浪费与漂移。
    """
    calls, objects = _stub_heavy_resources(monkeypatch)
    services = factory.build_services(config=_cfg(tmp_path))
    assert [name for name, _, _ in calls] == ["connect", "connect_autocommit",
                                              "get_client", "load_model",
                                              "load_reranker"]
    assert (services.conn, services.client) == (objects["conn"], objects["client"])
    assert (services.encoder, services.reranker) == (objects["encoder"],
                                                    objects["reranker"])
    assert services.account_conn is objects["account_conn"]


def test_both_connections_are_built_from_the_same_mysql_config(monkeypatch, tmp_path):
    """两条连接必须用**同一份** mysql 配置（都来自 cfg.mysql）。

    钉「工厂把配置递对了」这一层，**不钉 autocommit** —— 那个键是
    db/mysql.connect_autocommit 内部加的（`connect(autocommit=True, **overrides)`），
    工厂递下去的 kwargs 里根本没有它。本用例的旧版本正是断言
    `kwargs["autocommit"] is True`，恒红：它测不出任何真实改动，且只在真跑全量时
    才暴露。autocommit 本身改由两处真连库的用例承重 —— test_db_mysql 钉服务端
    状态位，test_api_deps 钉「停用即时生效」的端到端后果。

    这里钉的是真实故障形态「两条连接分叉」：账号连接若接到别的库或别的账号上，
    登录查的是另一份数据，表现是「刚建的账号登不上」—— 与「口令错」一模一样。
    """
    calls, _ = _stub_heavy_resources(monkeypatch)
    cfg = _cfg(tmp_path)
    factory.build_services(config=cfg)
    by_name = {name: kwargs for name, _, kwargs in calls}
    assert by_name["connect"] == cfg.mysql
    assert by_name["connect_autocommit"] == cfg.mysql


def test_the_answerer_receives_that_same_batch_of_objects(monkeypatch, tmp_path):
    """Answerer 必须拿到**同一批**对象，不是为它另建一份。

    另建一份（比如给它单开一个 encoder）在功能上跑得通，只是每次问答多一份
    内存与一次加载，且两条链路的口径从此可分叉 —— 故用身份断言钉住。
    账号连接**不进** Answerer：问答链路不该拿到它（它只服务鉴权与登录），
    多给一个依赖就多一条将来被误用的路。
    """
    _, objects = _stub_heavy_resources(monkeypatch)
    monkeypatch.setattr("app.core.factory.Answerer", _RecordingAnswerer)
    services = factory.build_services(config=_cfg(tmp_path))
    got = services.answerer.kwargs
    assert (got["conn"], got["client"]) == (objects["conn"], objects["client"])
    assert (got["encoder"], got["reranker"]) == (objects["encoder"],
                                                objects["reranker"])
    assert got["extras_fn"] is None
    assert "account_conn" not in got


def test_the_lawyer_side_never_builds_the_extras_dependencies(monkeypatch, tmp_path):
    """律师侧红线（数据不出域）：不构造附加区块真依赖，因而不碰 DeepSeek 客户端。

    判据放在 extras_fn 与工厂调用两处：只查「传给 Answerer 的是 None」时，工厂
    先建好再丢掉也照样绿 —— 而那时密钥缺失会把律师侧一起拖死。
    """
    calls, _ = _stub_heavy_resources(monkeypatch)
    monkeypatch.setattr("app.core.factory.Answerer", _RecordingAnswerer)
    services = factory.build_services(SIDE_INTERNAL, config=_cfg(tmp_path))
    assert services.answerer.kwargs["extras_fn"] is None
    assert [name for name, _, _ in calls if name == "build_extras_fn"] == []


def test_the_public_side_builds_extras_with_its_own_resources(monkeypatch, tmp_path):
    """公众侧要区块：真依赖用同一批重资源装（费用检索与法条侧共用编码器与精排）。"""
    calls, objects = _stub_heavy_resources(monkeypatch)
    monkeypatch.setattr("app.core.factory.Answerer", _RecordingAnswerer)
    services = factory.build_services(SIDE_PUBLIC, with_extras=True,
                                      config=_cfg(tmp_path))
    assert [name for name, _, _ in calls if name == "build_extras_fn"] == ["build_extras_fn"]
    _, args, _ = [c for c in calls if c[0] == "build_extras_fn"][0]
    assert args == (objects["conn"], objects["client"], objects["encoder"],
                    objects["reranker"])
    assert services.answerer.kwargs["extras_fn"] is objects["extras_fn"]


def test_asking_for_extras_on_the_lawyer_side_fails_before_building_anything(
        monkeypatch, tmp_path):
    """「律师侧 + 要区块」是调用方的编程错误，当场抛而不是静默忽略。

    静默忽略会让配错与配对在行为上完全一样，而律师侧本来就没有区块 —— 于是这个
    错永远暴露不了。同时钉住「拒绝了就别有副作用」：一个连接都不该建出去。
    """
    calls, _ = _stub_heavy_resources(monkeypatch)
    with pytest.raises(ValueError, match="公众侧"):
        factory.build_services(SIDE_INTERNAL, with_extras=True, config=_cfg(tmp_path))
    assert calls == []


def test_a_missing_model_file_stops_the_assembly_before_anything_is_built(
        monkeypatch, tmp_path):
    """路径自检在加载之前：模型目录失效时要立刻说，而不是先连库、连完再报。

    （bge-m3 的加载是 2.3GB 读盘，失败点越晚越像链路故障而不像环境问题。）
    """
    calls, _ = _stub_heavy_resources(monkeypatch)
    with pytest.raises(config.ConfigError):
        factory.build_services(config=_cfg(tmp_path, embed_files=()))
    assert calls == []


# 子进程探针。之所以要**真跑一次装出来的 extras_fn**而不是只 import：反向依赖
# 曾经住在函数体里（`from ingest_fee import search`），import 期不执行 —— 2026-09-29
# 复核实测，按那个真实回退形态改回去后，只做 import 的探针仍 exit=0。探针自带替身
# （形状同 backend/tests/test_extras.py，空命中 → 走到 no_corpus 即止），故运行时不
# 加载模型、不连库、不调 DeepSeek；`get_llm` 在函数体内 import，替身必须放模块属性。
_PROBE = """
import sys
import unittest.mock

import app.core.factory
import app.generation.llm_router as llm_router
import app.recommend.extras
import app.recommend.fee_search


class _Encoder:
    def encode(self, texts, **kwargs):
        return {"dense_vecs": [[0.0] * 1023 + [1.0]] * len(texts),
                "lexical_weights": [{1: 0.5}] * len(texts)}


class _Milvus:
    def hybrid_search(self, *args, **kwargs):
        return [[]]


llm_router.get_llm = lambda side: object()
extras_fn = app.recommend.extras.build_extras_fn(
    unittest.mock.MagicMock(), _Milvus(), _Encoder())
assert extras_fn("押金不退怎么办")["fee"]["status"] == "no_corpus"
assert "ingest_fee" not in sys.modules, "服务进程的装配链 import 了 tools/ 下的模块"
"""


def test_the_backend_imports_and_runs_the_extras_without_the_tools_directory():
    """服务进程的 sys.path 上没有 tools/（uvicorn 从 backend/ 起跑），所以装配链
    一旦（直接或间接）import tools 下的模块，HTTP 侧就是 ModuleNotFoundError。

    子进程 + 清空 PYTHONPATH 是为了让这条断言说的是「服务真的 import 得到」，
    而不是「本测试进程的 sys.path 恰好也有 tools/」——后者恒真，等于没有断言。
    探针跑替身不跑真装配：加载模型要几十秒，那不是这条要测的。两条断言分工：
    import `app.recommend.fee_search` 挡住「钱路检索面被删/搬回 tools/」，
    真跑一次 extras_fn 挡住「函数体内反向 import 回潮」（只 import 的版本对
    后一形态恒绿，见 _PROBE 的注释）。
    """
    backend = pathlib.Path(__file__).resolve().parents[1]
    proc = subprocess.run([sys.executable, "-c", _PROBE], cwd=str(backend),
                          capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": ""})
    assert proc.returncode == 0, proc.stderr


class _Closer:
    """会记账的句柄替身（回收用例共用）。name 就是它在断言里的身份。"""

    def __init__(self, name: str, closed: list, *, boom: bool = False) -> None:
        self.name = name
        self._closed = closed
        self._boom = boom

    def close(self) -> None:
        if self._boom:
            raise RuntimeError(f"{self.name} 已经断了")
        self._closed.append(self.name)


def _services(conn, client, account_conn=None) -> factory.Services:
    """只填可关闭句柄的 Services：模型字段是 None（健康检查与释放都不碰它们）。

    account_conn 默认 None：它是任务 4 加的账号路径连接，而本文件大部分用例
    （探针、回收、失败路径）都不碰账号 —— 缺省挡住的是「为了加一个字段去改
    五处无关的构造」，不是「可以忘记装配它」（真装配路径由 build_services 钉住）。
    """
    return factory.Services(conn=conn, client=client, encoder=None, reranker=None,
                            answerer=None, account_conn=account_conn)


def test_services_close_releases_the_connection_and_the_client():
    """Services.close() 要真的把句柄关掉（三个都关，任务 4 起）。

    lifespan 的用例用的是替身服务，它只证明「有人调了 close」—— 把本方法的 body
    换成空实现时那些用例照样绿（2026-09-29 变异实测），所以回收这件事必须在**真
    Services** 上再钉一遍，否则「加了 close()」只是形状。
    account_conn 一并放进来：新加的句柄最容易漏进释放循环（漏了的表现是进程退出
    时连接不还，而它在任何健康检查里都看不见）。
    """
    closed: list[str] = []
    _services(_Closer("conn", closed), _Closer("client", closed),
              _Closer("account_conn", closed)).close()
    assert closed == ["conn", "account_conn", "client"]


def test_one_failing_close_neither_stops_the_other_nor_raises(caplog):
    """一个句柄关不掉时：另一个照关、且**不抛**，但要留 warning。

    不抛是两处调用方的共同要求：装配失败路径上正在抛的是真因（模型加载失败），
    close 再抛就把真因盖成「close 出错」；lifespan 关闭路径上抛则会把一次正常退出
    变成崩溃。留 warning 挡住另一种偷懒 —— 静默吞掉会让「连接慢慢漏光」与「一切
    正常」在日志上完全一样（③b-1 的同口径）。
    """
    closed: list[str] = []
    services = _services(_Closer("conn", closed, boom=True), _Closer("client", closed))
    with caplog.at_level(logging.WARNING):
        services.close()
    assert closed == ["client"]
    assert any("conn" in record.getMessage() for record in caplog.records)
