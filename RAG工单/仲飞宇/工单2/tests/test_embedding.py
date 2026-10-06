"""EmbeddingClient 的分批与重试——这两个都是**可用性**行为，不是性能优化。

两个 bug 都是「换成数据集语料」之后才暴露的：手写样例一个角色只有几十个 chunk，
`build_chunks` 一次调用的文本量很小，整批一个请求、失败不重试都看不出问题；语料换成
公开数据集后单个文件就是几百上千个 chunk，链路立刻在半路整段中断（实测三次里断两次）。

所以这两条路径必须钉住：
  * 分批：一次请求的条数不能超过 embed_batch_size（实测 Ollama 一次 838 条直接 400）；
  * 重试：Ollama 会奇偶交替地 400（内部 tokenize 竞态），而 400 是 openai SDK 不重试的
    错误码，不自己重试就等于把偶发抖动放大成整轮失败。

用假客户端（不联网、不依赖 Ollama），只验客户端自己的行为：拆了几次、顺序对不对、
失败会不会重试、条数不符会不会抛。
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.core.config import Settings
from app.core.embedding import EmbeddingClient


class FakeEmbeddings:
    """假的 /embeddings：记录每次收到的文本，按「第几条」编码进向量，便于验顺序。"""

    def __init__(self, *, fail_times: int = 0, drop: int = 0):
        self.calls: list[list[str]] = []
        self.fail_times = fail_times  # 前 N 次调用抛错，之后成功
        self.drop = drop  # 少返回几条（模拟上游按上限截断）

    def create(self, *, model: str, input: list[str]):
        self.calls.append(list(input))
        if len(self.calls) <= self.fail_times:
            raise RuntimeError("Error code: 400 - tokenize: connection refused")
        data = [
            # 向量按**文本内容**编码（首字符码点），不是按批内下标：按下标的话每批都从 0 开始，
            # 「返回顺序与输入一致」就验不出来了。
            SimpleNamespace(index=i, embedding=[float(ord(input[i][0])), 1.0, 0.0])
            for i in range(len(input) - self.drop)
        ]
        return SimpleNamespace(data=data)


def make_client(batch_size: int, fake: FakeEmbeddings) -> EmbeddingClient:
    """构造一个 provider 非 dummy 的客户端，再把底层 OpenAI 换成假件。

    provider 必须是 ollama/openai_compat：dummy 分支会直接返回伪向量，走不到分批逻辑。
    """
    settings = Settings(
        llm_provider="dummy",
        embed_provider="ollama",
        embed_batch_size=batch_size,
        sql_url="sqlite://",  # 不连库；EmbeddingClient 也用不到
    )
    client = EmbeddingClient(settings)
    client._client = SimpleNamespace(embeddings=fake)
    return client


def test_splits_into_batches_and_keeps_order():
    """5 条、批大小 2 → 拆成 2+2+1 三次；返回顺序必须与输入一致。"""
    fake = FakeEmbeddings()
    client = make_client(2, fake)

    vecs = client.embed_texts(["a", "b", "c", "d", "e"])

    assert [len(c) for c in fake.calls] == [2, 2, 1], "没有按 embed_batch_size 拆批"
    assert [c for call in fake.calls for c in call] == ["a", "b", "c", "d", "e"], "批间顺序被打乱"
    assert len(vecs) == 5
    # 向量按文本首字符编码（a<b<c<d<e 码点递增），归一化后仍保持单调——顺序错位就会翻转
    assert vecs[0][0] < vecs[4][0], "返回顺序与输入不对应"


def test_retries_transient_failure():
    """第 1 次失败、第 2 次成功：结果正常返回，且确实重试过（调用数 > 批数）。"""
    fake = FakeEmbeddings(fail_times=1)
    client = make_client(10, fake)

    vecs = client.embed_texts(["a", "b"])

    assert len(vecs) == 2
    assert len(fake.calls) == 2, "失败后没有重试"


def test_raises_after_retries_exhausted():
    """一直失败时不能无限重试，要抛出来——入库侧靠异常走「先算后删」，不许静默丢块。"""
    fake = FakeEmbeddings(fail_times=99)
    client = make_client(10, fake)

    with pytest.raises(RuntimeError, match="连续 3 次失败"):
        client.embed_texts(["a", "b"])
    assert len(fake.calls) == 3, "重试次数不是 3"


def test_count_mismatch_raises():
    """上游少返回时抛错：否则 zip(texts, vectors) 会静默丢尾块，日志还显示「成功 N 块」。"""
    fake = FakeEmbeddings(drop=1)
    client = make_client(10, fake)

    with pytest.raises(RuntimeError, match="返回条数不符"):
        client.embed_texts(["a", "b"])


def test_empty_input_makes_no_request():
    """空批量不发请求：各家对 input=[] 的回应不统一，不该让空调用变成一次错误。"""
    fake = FakeEmbeddings()
    client = make_client(10, fake)

    assert client.embed_texts([]) == []
    assert fake.calls == []


def test_batch_size_is_at_least_one():
    """batch_size 配成 0 或负数时按 1 处理，否则 range 步长为 0 会死循环。"""
    fake = FakeEmbeddings()
    client = make_client(0, fake)

    assert len(client.embed_texts(["a", "b"])) == 2
    assert [len(c) for c in fake.calls] == [1, 1]
