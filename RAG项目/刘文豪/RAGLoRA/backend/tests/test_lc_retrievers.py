# -*- coding: utf-8 -*-
"""检索器适配层测试：重点是「没有重复加载模型」。

本模块在守护什么
================

`backend/app/services/lc_retrievers.py` 存在的**全部意义**就是：把 bge-m3 接进
LangChain 接口时**不重复加载模型**。`langchain_huggingface.HuggingFaceEmbeddings`
会新建一份 bge-m3 实例（fp32 约 2.3GB），本机 15.7GB 内存下两份即 4.6GB，不可接受；
因此 `BgeM3DenseEmbeddings` 委托给 `app.services.embed` 的全局单例。

初版测试只断言了「向量维度 1024 / 元素是 float / device 是 cpu」——**一个用
`HuggingFaceEmbeddings` 新建第二份 bge-m3 的错误实现同样能通过这三条断言**。
也就是说「防止模型重复加载」这条命脉当时没有任何测试守护；唯一的证据来自一个
已删除、不可复现的一次性探针脚本。

本文件把那次探针的三条证据固化为**常驻测试**（见 `test_adapter_*` / `test_single_*`）：

  A. `lc_retrievers.encode / encode_one is embed.encode / encode_one`
     —— 适配器持有的是**同一个函数对象**，结构上不可能构造第二份模型。
  B. `langchain_huggingface` 从未进入 `sys.modules`
     —— 证明没有走「会新建模型」的那条路。
  C. 进程内 `XLMRobertaModel` 实例恰为 1 个，**且该实例就是 `embed._model`**
     —— 直接数模型并核对身份（计数 + 身份缺一不可，原因见该测试 docstring）。

**不要删这几条测试**：删了就等于把「不重复加载」这条约定重新交还给运气。
"""
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.documents import Document  # noqa: E402

from app.services import lc_retrievers  # noqa: E402


def test_hit_to_doc_roundtrip():
    """hit -> Document -> hit 应保留关键字段。"""
    hit = {"id": 7, "score": 0.5, "text": "正文", "source": "a.pdf",
           "page": 3, "law_name": None, "article_no": None, "collection": "kb_medical"}
    doc = lc_retrievers.hit_to_doc(hit)
    assert isinstance(doc, Document)
    assert doc.page_content == "正文"
    assert doc.metadata["source"] == "a.pdf"

    back = lc_retrievers.doc_to_hit(doc)
    assert back["text"] == "正文"
    assert back["source"] == "a.pdf"
    assert back["page"] == 3
    assert back["id"] == 7


def test_embeddings_delegates_to_existing_singleton():
    """BgeM3DenseEmbeddings 必须复用 embed 模块的模型，而非新建实例。"""
    from app.services import embed

    emb = lc_retrievers.BgeM3DenseEmbeddings()
    vec = emb.embed_query("高血压")
    assert len(vec) == 1024
    assert isinstance(vec[0], float)
    # embed 模块的全局单例已被这次调用加载
    assert embed.current_device() == "cpu"


def test_adapter_holds_the_same_function_objects():
    """【回归护栏・结构】适配器必须直接引用 embed 模块的函数对象。

    这是三条护栏里最硬的一条：`lc_retrievers.encode is embed.encode` 为真，
    意味着适配器若要调用编码能力，**只可能**通过 `embed` 模块里那个持全局
    `_model` 单例的函数。它没有任何地方可以存放自己的一份模型。

    回归推演：若把 `BgeM3DenseEmbeddings` 改成
    `langchain_huggingface.HuggingFaceEmbeddings` 实现（或改从
    `langchain_huggingface` 导入编码函数、自己 new 一份模型），
    本测试立刻失败 —— 要么 `lc_retrievers.encode` 不存在，要么它不再是
    `embed.encode` 这个对象。

    本测试**不加载模型**（只做属性比较），毫秒级。
    """
    from app.services import embed

    assert hasattr(lc_retrievers, "encode"), "适配层不再持有 encode 引用"
    assert hasattr(lc_retrievers, "encode_one"), "适配层不再持有 encode_one 引用"
    assert lc_retrievers.encode is embed.encode, (
        "lc_retrievers.encode 不是 embed.encode —— 适配层可能自建了一份模型"
    )
    assert lc_retrievers.encode_one is embed.encode_one, (
        "lc_retrievers.encode_one 不是 embed.encode_one —— 适配层可能自建了一份模型"
    )


def test_langchain_huggingface_never_imported():
    """【回归护栏・路径】`langchain_huggingface` 从未被导入。

    `langchain_huggingface.HuggingFaceEmbeddings` 会新建一份 bge-m3，它所在的
    模块一旦被 import 就说明有人走上了那条路。本测试断言该模块**不在**
    `sys.modules` 里。

    注意：这要求整个测试进程都没人 import 过它。当前代码库只有
    `lc_retrievers.py` 的一个 docstring 提到这个名字（文本提及不触发导入），
    所以断言成立。若将来有别的模块合法地需要 HuggingFaceEmbeddings，
    应先和本测试的设计意图对齐，而不是直接删掉它。

    本测试**不加载模型**，毫秒级。
    """
    assert "langchain_huggingface" not in sys.modules, (
        "langchain_huggingface 被导入了 —— 很可能有人用 HuggingFaceEmbeddings "
        "新建了第二份 bge-m3"
    )


def test_single_bge_m3_instance_in_process():
    """【回归护栏・直接计数】干净进程内 `XLMRobertaModel` 实例恰为 1 个，且就是 embed 单例。

    前两条是间接证据（结构 / 路径）；这一条直接数模型对象。

    ⚠️ 为什么必须在**子进程**里数（2026-09-16 修正）
    ------------------------------------------------
    原实现直接在测试进程内 `gc` 计数，假设「一份 bge-m3 恰好对应一个
    `XLMRobertaModel` 实例」。**这个假设是错的**：

        `bge-reranker-v2-m3` 本身就是 XLM-RoBERTa 模型，它的
        `XLMRobertaForSequenceClassification` **内部含一个 `XLMRobertaModel`
        子模块**（参数量 567M，与 bge-m3 的 568M 极接近，极易混淆）。

    而本套件里 `test_lc_chain.py` 会调用精排（`rerank.rerank`），精排一加载，
    进程里就多出第二个 `XLMRobertaModel` —— **那不是 bge-m3 的副本，是精排的基座**。
    实测：单独跑本文件 = 1；与 `test_lc_chain.py` 同跑 = 2（误报失败）。

    因此本测试改为在**子进程**中度量：子进程只导入 `lc_retrievers` 做一次编码，
    精排从未被加载，计数才有明确含义。这与 `test_ragas_compat.py` 隔离子进程
    的做法一致。

    为什么要先 `embed_query`：只有编码一次才会真正触发模型加载，否则实例数为 0。

    ⚠️ 只断言 `len(instances) == 1` **不够**（重要，勿删这一行）：
    实测发现，「把适配器整体换成 HuggingFaceEmbeddings、`embed` 单例从此不再被
    任何代码加载」这种**纯回归**下，进程里同样只剩 1 个模型（那唯一一个就是
    HuggingFaceEmbeddings 新建的），`==1` 会误判为通过。因此额外断言
    **那唯一一个实例就是 `embed._model` 本身**——纯回归下 `embed._model` 为
    `None`，断言立即失败，回归被捕获。

    HuggingFaceEmbeddings 内部把基座模型包在 `SentenceTransformer` 里，且实测
    构造时**立即加载**（非惰性），其基座同样被 gc 记为 `XLMRobertaModel` 实例，
    所以不会因「模型没被计数」而漏检。
    """
    import subprocess
    import sys as _sys

    probe = textwrap.dedent(
        """
        import gc, sys
        sys.path.insert(0, %r)
        from app.services import lc_retrievers, embed

        # 触发一次真实编码，确保模型已加载。
        lc_retrievers.BgeM3DenseEmbeddings().embed_query("高血压")

        from transformers import XLMRobertaModel
        gc.collect()
        inst = [o for o in gc.get_objects() if type(o) is XLMRobertaModel]

        # 只把结论以固定前缀回传，避免把模型日志混进断言信息。
        print("COUNT=%%d" %% len(inst))
        print("IS_SINGLETON=%%s" %% (
            len(inst) == 1 and inst[0] is embed._model
        ))
        print("EMBED_LOADED=%%s" %% (embed._model is not None))
        """
    ) % str(Path(__file__).resolve().parents[1])

    proc = subprocess.run(
        [_sys.executable, "-c", probe],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=600, cwd=str(Path(__file__).resolve().parents[1]),
    )
    out = proc.stdout or ""
    assert proc.returncode == 0, (
        f"子进程退出码 {proc.returncode}\n--- stdout ---\n{out[-2000:]}\n"
        f"--- stderr ---\n{(proc.stderr or '')[-2000:]}"
    )

    def _field(key: str) -> str:
        for line in out.splitlines():
            if line.startswith(key + "="):
                return line.split("=", 1)[1].strip()
        raise AssertionError(f"子进程未回传 {key}；stdout=\n{out[-2000:]}")

    count = int(_field("COUNT"))
    assert count == 1, (
        f"干净子进程内发现 {count} 个 bge-m3 基座实例（期望 1）—— "
        f"适配层很可能自建了第二份模型"
    )
    assert _field("EMBED_LOADED") == "True", (
        "embed 单例未被加载 —— 适配层可能绕开了 embed 模块"
    )
    assert _field("IS_SINGLETON") == "True", (
        "子进程内唯一的 XLMRobertaModel 不是 embed 单例（embed._model）—— "
        "适配层绕开了 embed 模块、自建了一份模型"
    )


def test_hybrid_retriever_returns_documents(isolated_qdrant):
    """混合检索最终仍走手写实现，返回 LangChain Document。

    显式请求 `isolated_qdrant`（review Minor 1）：只有这条测试真正需要 Qdrant，
    所以只让它拿到快照目录，避免 `autouse` 对全体测试的进程级副作用。
    """
    r = lc_retrievers.HybridRetriever(collection="kb_medical", k=3)
    docs = r.invoke("高血压的诊断标准")
    assert isinstance(docs, list)
    for d in docs:
        assert isinstance(d, Document)
    assert len(docs) <= 3
