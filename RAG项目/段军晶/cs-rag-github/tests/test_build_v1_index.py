# -*- coding: utf-8 -*-
"""build_v1_index 的单元测试：安全护栏 + 内容还原

背景：
    scripts/build_v1_index.py 会 drop 重建集合（构建分支的 drop_existing=True、
    以及 --drop 分支）。若名字指到不该碰的集合，后果是不可恢复的数据丢失：
    · 主集合 cs_kb_chunks → 知识库 127 条向量连同 V2 视觉描述一起丢失
    · 其它项目的集合（djj3_2 等）→ 第三方数据丢失，本脚本无法重建
    因此 main() 用**白名单**护栏：只放行 cs_kb_v1_* 前缀，空名与主集合一律拒绝，
    且不执行任何 drop / create。

    另一半测试针对 strip_vision —— 本任务的全部价值就是「还原精确」，
    若标记或拼接方式将来被改动，strip_vision 会静默地「找不到标记就原样返回」，
    脚本照常报成功，产出的索引只是悄悄带着 V2 内容。故必须钉死。

本测试**不连接 Milvus**：通过 monkeypatch 替换 settings.milvus_collection，
并替换 milvus_client 的 create_collection / drop_collection 为记录调用的桩函数，
因此可在默认真空环境下快速运行（无需 slow 标记）。
"""

import pytest

from backend.config import settings
from scripts import build_v1_index


@pytest.fixture
def 记录调用的桩(monkeypatch):
    """
    把会真正改动 Milvus 的两个函数替换成记录调用的桩。

    返回一个 list，任何对 create_collection / drop_collection 的调用都会
    以 ("create"|"drop", 集合名) 的形式追加进去。测试据此断言「护栏拦住了，
    一个破坏性调用都没发生」。
    """
    calls = []

    def fake_create_collection(name=None, **kwargs):
        calls.append(("create", name))

    def fake_drop_collection(name=None, **kwargs):
        calls.append(("drop", name))

    monkeypatch.setattr(build_v1_index.milvus_client, "create_collection", fake_create_collection)
    monkeypatch.setattr(build_v1_index.milvus_client, "drop_collection", fake_drop_collection)
    return calls


def test_构建分支拒绝主集合且不建集合(monkeypatch, 记录调用的桩):
    """传入主集合名 → 返回非 0，且 create_collection 一次都没被调用"""
    monkeypatch.setattr(settings, "milvus_collection", "cs_kb_main_for_test")

    code = build_v1_index.main(["--collection", "cs_kb_main_for_test"])

    assert code != 0, "操作主集合必须返回非 0 退出码"
    assert 记录调用的桩 == [], f"护栏未拦住破坏性调用：{记录调用的桩}"


def test_drop分支拒绝主集合且不删集合(monkeypatch, 记录调用的桩):
    """
    --drop 分支同样受护栏保护。

    这是更危险的一条路径：--drop 会直接删除集合。若护栏只挡构建分支，
    这里仍会把主集合删掉。
    """
    monkeypatch.setattr(settings, "milvus_collection", "cs_kb_main_for_test")

    code = build_v1_index.main(["--collection", "cs_kb_main_for_test", "--drop"])

    assert code != 0, "drop 主集合必须返回非 0 退出码"
    assert 记录调用的桩 == [], f"护栏未拦住删除操作：{记录调用的桩}"


def test_护栏在_collection_exists_之前生效(monkeypatch, 记录调用的桩):
    """
    护栏必须早于任何 Milvus 交互，连「查询集合是否存在」都不该发生。

    否则一旦 Milvus 不可达，脚本会先抛连接异常，用户看到的是误导性的
    网络错误，而非「你传了主集合名」这条真正的原因。
    """
    monkeypatch.setattr(settings, "milvus_collection", "cs_kb_main_for_test")
    touched = []

    def fake_collection_exists(name=None):
        touched.append(name)
        return True

    monkeypatch.setattr(
        build_v1_index.milvus_client, "collection_exists", fake_collection_exists
    )

    code = build_v1_index.main(["--collection", "cs_kb_main_for_test", "--drop"])

    assert code != 0
    assert touched == [], f"护栏生效前不应触碰 Milvus，实际查询了：{touched}"


def test_临时集合不被护栏误伤(monkeypatch, 记录调用的桩):
    """
    反向验证：护栏不能把正常用途也挡掉。

    默认集合名 cs_kb_v1_orig 与主集合不同，--drop 分支应正常走到
    drop_collection（此处被桩记录但不会真的删）。
    """
    monkeypatch.setattr(settings, "milvus_collection", "cs_kb_main_for_test")
    monkeypatch.setattr(
        build_v1_index.milvus_client, "collection_exists", lambda name=None: True
    )

    code = build_v1_index.main(["--collection", "cs_kb_v1_orig", "--drop"])

    assert code == 0, "临时集合应被允许操作"
    assert 记录调用的桩 == [("drop", "cs_kb_v1_orig")], (
        f"应恰好删除临时集合一次，实际={记录调用的桩}"
    )


# ---------------------------------------------------------------------------
# 空名绕过（Critical 1）
# ---------------------------------------------------------------------------

def test_空集合名被拒绝且不删集合(monkeypatch, 记录调用的桩):
    """
    ★ Critical 回归测试 ★

    空名**不等于**主集合名，所以只比较字面串的护栏会放行；但 milvus_client
    内部有 `name = name or settings.milvus_collection` 的回落逻辑，空名会被
    悄悄解释成主集合，于是 --drop 会删掉知识库主集合。

    `--collection "$COL"`（COL 未设置）正是这种情形，且本项目下一步就要用
    环境变量切换集合，写法很自然 —— 故这是一条真实的误用路径。
    """
    monkeypatch.setattr(settings, "milvus_collection", "cs_kb_main_for_test")

    code = build_v1_index.main(["--collection", "", "--drop"])

    assert code != 0, "空集合名必须返回非 0 退出码"
    assert 记录调用的桩 == [], f"空名绕过了护栏，发生了破坏性调用：{记录调用的桩}"


def test_空集合名被拒绝_构建分支(monkeypatch, 记录调用的桩):
    """构建分支同样要挡住空名：create_collection('', drop_existing=True)
    回落成主集合后，会先用 V1 的无稀疏 schema 把主集合删掉再重建。"""
    monkeypatch.setattr(settings, "milvus_collection", "cs_kb_main_for_test")

    code = build_v1_index.main(["--collection", ""])

    assert code != 0
    assert 记录调用的桩 == [], f"空名绕过了护栏：{记录调用的桩}"


# ---------------------------------------------------------------------------
# 白名单：保护其它项目的集合（Minor 5 升级为必改）
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "名字",
    ["djj3_2", "djj3_4", "djj4", "djj4_2", "djj4_5", "zzx3_2cxlx", "cs_kb_chunks"],
)
def test_非白名单前缀的集合一律被拒绝(名字, monkeypatch, 记录调用的桩):
    """
    该 Milvus 实例上共存着其它项目的 6 个集合（djj3_2 等，共 4 万余条向量）。
    `--collection djj3_2 --drop` 会删掉另一个项目的数据，且本脚本完全无法重建。
    故护栏采用白名单：只放行 cs_kb_v1_* 前缀。

    cs_kb_chunks 亦在参数中：它既命中「等于主集合」规则，也不以 cs_kb_v1_
    开头，两条规则都应拦住它。
    """
    monkeypatch.setattr(settings, "milvus_collection", "cs_kb_main_for_test")

    code = build_v1_index.main(["--collection", 名字, "--drop"])

    assert code != 0, f"{名字} 不应被允许操作"
    assert 记录调用的桩 == [], f"{名字} 绕过了护栏：{记录调用的桩}"


def test_护栏只放行_cs_kb_v1_前缀(monkeypatch, 记录调用的桩):
    """白名单是前缀匹配而非精确匹配：cs_kb_v1_ 下的其它名字也应放行
    （例如将来要并存多个版本的临时集合）。"""
    monkeypatch.setattr(settings, "milvus_collection", "cs_kb_main_for_test")
    monkeypatch.setattr(
        build_v1_index.milvus_client, "collection_exists", lambda name=None: True
    )

    code = build_v1_index.main(["--collection", "cs_kb_v1_other", "--drop"])

    assert code == 0, "cs_kb_v1_ 前缀的临时集合应被允许"
    assert 记录调用的桩 == [("drop", "cs_kb_v1_other")]


# ---------------------------------------------------------------------------
# strip_vision：本任务的核心还原逻辑
# ---------------------------------------------------------------------------

def test_剥离标记及其后内容():
    """图片块的新 content = 旧 content + 标记 + 描述，剥离后应精确还原旧 content"""
    assert build_v1_index.strip_vision(
        "图 2 软件质量模型\n[视觉解析] 这是一张质量模型示意图"
    ) == "图 2 软件质量模型"


def test_无标记时原样返回():
    """V2 未改动的 121 个块不含标记，必须逐字原样返回"""
    assert build_v1_index.strip_vision("普通正文内容") == "普通正文内容"


def test_空字符串不报错():
    """边界：空 content 不应抛异常"""
    assert build_v1_index.strip_vision("") == ""


def test_标记在开头时返回空串():
    """边界：标记位于最开头（旧 content 为空）时，应返回空串而非原串"""
    assert build_v1_index.strip_vision("\n[视觉解析] 仅有描述") == ""


def test_只保留第一个标记之前的内容():
    """
    边界：正文里若本身出现了「视觉解析」字样但没有前导的 "\\n[视觉解析] "
    完整标记，不应被误切。这里构造的是「正文含相似词 + 末尾才是真标记」。
    """
    text = "本文讨论视觉解析技术\n[视觉解析] 真正的描述"
    assert build_v1_index.strip_vision(text) == "本文讨论视觉解析技术"


def test_标记常量与_chunk_split_一致():
    """
    ★ 契约测试 ★

    build_v1_index.VISION_MARK 必须与 scripts/chunk_split._VISION_MARK 完全一致。

    这是**写入方与还原方之间的隐式契约**：chunk_split 用 _VISION_MARK 拼接，
    build_v1_index 用 VISION_MARK 剥离。两者一旦漂移，strip_vision 会找不到
    标记而静默原样返回 —— 脚本照常报「成功」，产出的索引却悄悄带着 V2 内容，
    且没有任何报错。用测试把这条隐式耦合钉成显式契约。
    """
    from scripts.chunk_split import _VISION_MARK

    assert build_v1_index.VISION_MARK == _VISION_MARK, (
        "拼接方与还原方的标记不一致，还原会静默失效"
    )
