# -*- coding: utf-8 -*-
"""服务层测试：对话编排 / 知识库服务（LLM、Milvus、嵌入均注入假实现）。"""

from types import SimpleNamespace
import pytest
from app.core.prompts import DEFAULT_PROMPT_TEMPLATE
from app.services.chat_service import ChatService
import fitz
from app.services.knowledge_service import KnowledgeService

# ---------- 对话编排（原 tests/test_chat_service.py） ----------

class FakeMemory:
    """与 RedisMemoryStore 相同接口的内存实现。"""

    def __init__(self):
        self.sessions = {}  # (user_id, role_id) -> list[{role, content}]

    def get_history(self, user_id, role_id):
        return list(self.sessions.get((user_id, role_id), []))

    def append(self, user_id, role_id, sender, content):
        self.sessions.setdefault((user_id, role_id), []).append(
            {"role": sender, "content": content}
        )

    def clear(self, user_id, role_id):
        self.sessions.pop((user_id, role_id), None)


class FakeLLM:
    """记录收到的 messages，返回预设回复。"""

    def __init__(self, reply="知道了"):
        self.reply = reply
        self.last_messages = None

    async def chat(self, messages):
        self.last_messages = messages
        return self.reply

    async def chat_stream(self, messages):
        self.last_messages = messages
        for chunk in ["好", "的", "呢"]:
            yield chunk


ROLE = SimpleNamespace(
    name="小助手", persona="热心朋友", prompt_template=DEFAULT_PROMPT_TEMPLATE
)


@pytest.fixture()
def make_service():
    def _make(llm=None, memory=None):
        return ChatService(llm=llm or FakeLLM(), memory=memory or FakeMemory())
    return _make


async def test_chat_returns_reply_and_persists_both_turns(make_service):
    memory = FakeMemory()
    service = make_service(memory=memory)

    reply = await service.chat(ROLE, "你好", user_id=1, role_id=2)

    assert reply == "知道了"
    assert memory.get_history(1, 2) == [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "知道了"},
    ]


async def test_chat_injects_persona_and_history_into_llm_messages(make_service):
    memory = FakeMemory()
    memory.append(1, 2, "user", "上次问过的问题")
    memory.append(1, 2, "assistant", "上次的回答")
    llm = FakeLLM()
    service = make_service(llm=llm, memory=memory)

    await service.chat(ROLE, "新的问题", user_id=1, role_id=2)

    system = llm.last_messages[0]["content"]
    assert "热心朋友" in system          # 人设注入
    assert "用户：上次问过的问题" in system  # 历史注入
    assert "小助手：上次的回答" in system
    assert llm.last_messages[-1] == {"role": "user", "content": "新的问题"}


async def test_chat_stream_yields_chunks_and_saves_full_reply(make_service):
    memory = FakeMemory()
    service = make_service(memory=memory)

    chunks = [c async for c in service.chat_stream(ROLE, "你好", user_id=1, role_id=2)]

    assert "".join(chunks) == "好的呢"
    assert memory.get_history(1, 2)[-1] == {"role": "assistant", "content": "好的呢"}


async def test_chat_injects_knowledge_into_system_prompt(make_service):
    llm = FakeLLM()
    service = make_service(llm=llm)

    await service.chat(ROLE, "高血压怎么办", user_id=1, role_id=2, knowledge="低盐饮食。")

    system = llm.last_messages[0]["content"]
    assert "【知识库内容】" in system
    assert "低盐饮食。" in system


async def test_chat_without_knowledge_omits_section(make_service):
    llm = FakeLLM()
    service = make_service(llm=llm)

    await service.chat(ROLE, "你好", user_id=1, role_id=2, knowledge="")

    assert "【知识库内容】" not in llm.last_messages[0]["content"]


async def test_chat_injects_longterm_memories_into_prompt(make_service):
    llm = FakeLLM()
    service = make_service(llm=llm)

    await service.chat(ROLE, "我的猫怎么样", user_id=1, role_id=2,
                       memories=["用户：我养了一只猫\n小助手：真可爱！"])

    system = llm.last_messages[0]["content"]
    assert "【长期记忆】" in system
    assert "我养了一只猫" in system


async def test_chat_without_memories_omits_section(make_service):
    llm = FakeLLM()
    service = make_service(llm=llm)

    await service.chat(ROLE, "你好", user_id=1, role_id=2)

    assert "【长期记忆】" not in llm.last_messages[0]["content"]


async def test_chat_stream_failure_does_not_save_partial_reply(make_service):
    class BrokenLLM(FakeLLM):
        async def chat_stream(self, messages):
            yield "半截"
            raise RuntimeError("模型断流")

    memory = FakeMemory()
    service = make_service(llm=BrokenLLM(), memory=memory)

    with pytest.raises(RuntimeError):
        async for _ in service.chat_stream(ROLE, "你好", user_id=1, role_id=2):
            pass

    # 用户消息已写入，但角色回复不应写入半截内容
    assert memory.get_history(1, 2) == [{"role": "user", "content": "你好"}]

# ---------- 知识库服务（原 tests/test_knowledge_service.py） ----------

def make_pdf(text: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    # insert_text 不自动换行，长行会溢出页面被裁剪；逐行按词边界折行（约 60 字符/行，保留原换行）
    wrapped_lines = []
    for line in text.split("\n"):
        if not line:
            wrapped_lines.append("")
            continue
        current = ""
        for word in line.split(" "):
            if current and len(current) + len(word) + 1 > 60:
                wrapped_lines.append(current)
                current = word
            else:
                current = f"{current} {word}".strip()
        wrapped_lines.append(current)
    page.insert_text((72, 72), "\n".join(wrapped_lines), fontsize=12)
    return doc.tobytes()


class FakeEmbedder:
    """返回可辨识的伪向量：文档向量 = [len(text) % 7]，查询向量 = [1] * 1024。"""

    def __init__(self, dim=1024):
        self.dim = dim
        self.embedded_docs = []

    def embed_documents(self, texts):
        self.embedded_docs.extend(texts)
        return [[len(t) % 7] * self.dim for t in texts]

    def embed_query(self, text):
        return [1.0] * self.dim


class FakeReranker:
    """倒序重排：最后一个候选最相关。"""

    def __init__(self):
        self.calls = []

    def rerank(self, query, passages):
        self.calls.append((query, passages))
        return list(reversed(passages))


def role_id_from_name(name):
    # 测试假件：v3 collection 名 -> role_id（"role_kb_v3_2" -> 2）
    return int(name.rsplit("_", 1)[-1])


class FakeMilvus:
    @staticmethod
    def collection_name(role_id):
        return f"role_kb_v3_{role_id}"

    def __init__(self):
        self.collections = {}  # role_id -> list[dict(chunk)]
        self.parents = {}  # role_id -> list[dict(parent)]
        self.searches = []
        self.parent_searches = []

    def ensure_parent_collection(self, role_id):
        self.parents.setdefault(role_id, [])

    def insert_parents(self, role_id, records):
        self.ensure_parent_collection(role_id)
        self.parents[role_id].extend(records)

    def delete_parents_by_source(self, role_id, source):
        before = len(self.parents.get(role_id, []))
        self.parents[role_id] = [
            r for r in self.parents.get(role_id, []) if r["source"] != source
        ]
        return before - len(self.parents[role_id])

    def get_parents(self, role_id, parent_ids):
        return {
            r["parent_id"]: r["text"]
            for r in self.parents.get(role_id, [])
            if r["parent_id"] in parent_ids
        }

    def hybrid_search_with_parents(self, name, query_dense, query_sparse, limit):
        self.parent_searches.append((name, limit))
        return [
            (c["text"], c.get("parent_id", ""))
            for c in self.collections.get(role_id_from_name(name), [])
        ][:limit]

    def ensure_collection(self, role_id):
        self.collections.setdefault(role_id, [])

    def insert_chunks(self, role_id, chunks):
        self.ensure_collection(role_id)
        self.collections[role_id].extend(chunks)

    def hybrid_search(self, role_id, query_dense, query_sparse, limit):
        self.searches.append((role_id, query_dense, query_sparse, limit))
        texts = [c["text"] for c in self.collections.get(role_id, [])]
        return texts[:limit]

    def all_texts(self, role_id):
        return [c["text"] for c in self.collections.get(role_id, [])]

    def list_sources(self, role_id):
        sources = {}
        for c in self.collections.get(role_id, []):
            sources[c["source"]] = sources.get(c["source"], 0) + 1
        return [{"source": s, "chunks": n} for s, n in sources.items()]

    def delete_source(self, role_id, source):
        before = len(self.collections.get(role_id, []))
        self.collections[role_id] = [
            c for c in self.collections.get(role_id, []) if c["source"] != source
        ]
        return before - len(self.collections[role_id])


@pytest.fixture()
def svc():
    milvus = FakeMilvus()
    return KnowledgeService(
        embedder=FakeEmbedder(), reranker=FakeReranker(), milvus=milvus
    ), milvus


def test_ingest_pdf_parses_chunks_embeds_and_inserts(svc):
    service, milvus = svc
    # 换行分隔句子：PDF 提取时句间空格可能丢失，换行是可靠边界
    text = "salt diet recommended.\n" + "exercise helps.\n" * 50
    pdf = make_pdf(text)

    result = service.ingest_pdf(role_id=2, pdf_bytes=pdf, source="高血压指南.pdf")

    assert result["source"] == "高血压指南.pdf"
    assert result["chunks"] >= 1
    stored = milvus.collections[2]
    assert len(stored) == result["chunks"]
    assert all("text" in c and "dense" in c and "sparse" in c for c in stored)
    assert all("summary" in c and c["summary"] for c in stored)  # 摘要字段
    assert all(c["source"] == "高血压指南.pdf" for c in stored)
    # 所有文档文本都进入了库（不丢失）
    assert "".join(c["text"] for c in stored).count("exercise helps.") >= 50


def test_ingest_pdf_strips_watermark_when_enabled(svc):
    """上传带水印 PDF：入库分块应不含水印文字。"""
    import fitz

    service, milvus = svc
    doc = fitz.open()
    for _ in range(3):
        page = doc.new_page()
        page.insert_text((72, 72), "hypertension knowledge content.", fontsize=12)
        page.insert_text((200, 400), "CONFIDENTIAL", fontsize=24, color=(0.8, 0.8, 0.8),
                         morph=(fitz.Point(200, 400), fitz.Matrix(45)))
    pdf = doc.tobytes()

    result = service.ingest_pdf(role_id=2, pdf_bytes=pdf, source="wm.pdf")

    assert result["chunks"] >= 1
    joined = "".join(c["text"] for c in milvus.collections[2])
    assert "CONFIDENTIAL" not in joined
    assert "hypertension knowledge content" in joined


def test_ingest_pdf_without_text_raises(svc):
    service, _ = svc
    pdf = make_pdf("")  # 空页面，无文本
    with pytest.raises(ValueError):
        service.ingest_pdf(role_id=2, pdf_bytes=pdf, source="空.pdf")


async def test_retrieve_runs_hybrid_search_then_rerank_top_k(svc):
    service, milvus = svc
    service.ingest_pdf(role_id=2, pdf_bytes=make_pdf("knowledge A. " * 100), source="doc.pdf")

    results = await service.retrieve(role_id=2, query="query text", top_k=2, recall_k=10)

    # 混合检索被调用，且收到了稠密+稀疏查询向量
    assert milvus.searches, "应调用混合检索"
    _, q_dense, q_sparse, limit = milvus.searches[-1]
    assert len(q_dense) == 1024
    assert isinstance(q_sparse, dict)
    assert limit == 10
    # 重排后被截断到 top_k
    assert len(results) <= 2


async def test_retrieve_rerank_receives_candidates_in_order(svc):
    service, milvus = svc
    reranker = FakeReranker()
    service2 = KnowledgeService(embedder=FakeEmbedder(), reranker=reranker, milvus=milvus)
    service2.ingest_pdf(role_id=2, pdf_bytes=make_pdf("candidate doc. " * 100), source="d.pdf")

    results = await service2.retrieve(role_id=2, query="question", top_k=1, recall_k=5)

    query, passages = reranker.calls[-1]
    assert query == "question"
    assert len(passages) <= 5
    assert results == [passages[-1]]  # 倒序重排取第 1 个 = 原最后一个


async def test_retrieve_without_reranker_returns_hybrid_top_k(svc):
    service, milvus = svc
    service2 = KnowledgeService(embedder=FakeEmbedder(), reranker=None, milvus=milvus)
    service2.ingest_pdf(role_id=2, pdf_bytes=make_pdf("content. " * 100), source="d.pdf")

    results = await service2.retrieve(role_id=2, query="question", top_k=2, recall_k=10)

    assert 0 < len(results) <= 2


class FakeRewriter:
    """返回固定检索变体，可配置抛异常。"""

    def __init__(self, variants, fail=False):
        self.variants = variants
        self.fail = fail
        self.queries = []

    async def rewrite(self, query):
        self.queries.append(query)
        if self.fail:
            raise RuntimeError("rewrite failed")
        return self.variants


async def test_retrieve_with_rewriter_searches_all_variants_and_merges(svc):
    service, milvus = svc
    service.ingest_pdf(role_id=2, pdf_bytes=make_pdf("content alpha. " * 40), source="a.pdf")
    rewriter = FakeRewriter(["expanded query one", "expanded query two"])
    service2 = KnowledgeService(embedder=FakeEmbedder(), reranker=FakeReranker(), milvus=milvus)
    service2.query_rewriter = rewriter

    results = await service2.retrieve(role_id=2, query="original", top_k=4, recall_k=10)

    assert rewriter.queries == ["original"]
    # 原问题 + 2 个变体各检索一次
    assert len(milvus.searches) == 3
    assert 0 < len(results) <= 4


async def test_retrieve_rewriter_failure_falls_back_to_original_query(svc):
    service, milvus = svc
    service.ingest_pdf(role_id=2, pdf_bytes=make_pdf("content beta. " * 40), source="b.pdf")
    service.query_rewriter = FakeRewriter(["variant"], fail=True)

    results = await service.retrieve(role_id=2, query="original", top_k=4, recall_k=10)

    assert len(milvus.searches) == 1  # 只按原问题检索
    assert 0 < len(results) <= 4


async def test_retrieve_without_rewriter_unchanged(svc):
    service, milvus = svc
    service.ingest_pdf(role_id=2, pdf_bytes=make_pdf("content gamma. " * 40), source="c.pdf")

    results = await service.retrieve(role_id=2, query="original", top_k=4, recall_k=10)

    assert len(milvus.searches) == 1
    assert 0 < len(results) <= 4


async def test_ingest_heading_mode_prepends_heading_to_chunks(svc):
    """chunking_mode=heading：块文本携带标题上下文。"""
    _, milvus = svc
    service = KnowledgeService(
        embedder=FakeEmbedder(), reranker=FakeReranker(), milvus=milvus,
        chunking_mode="heading", chunk_size=200,
    )
    text = (
        "1. Diagnosis\n"
        + "hypertension is diagnosed at 140 over 90. " * 20
        + "\n2. Lifestyle\n"
        + "salt intake should be below 5 grams. " * 20
    )
    pdf = make_pdf(text)

    result = service.ingest_pdf(role_id=2, pdf_bytes=pdf, source="h.pdf")

    assert result["chunks"] >= 2
    joined = "\n".join(c["text"] for c in milvus.collections[2])
    assert "1. Diagnosis" in joined
    assert "2. Lifestyle" in joined


def test_ingest_semantic_mode_splits_by_similarity(svc):
    """chunking_mode=semantic：相似度低的句子边界切开。"""
    _, milvus = svc
    sentences = ["apple fruit.", "apple healthy.", "car fuel.", "car service."]

    class ControlledEmbedder:
        """按内容生成向量：apple 话题 → [1,0] 方向，car 话题 → [0,1] 方向。

        句子级（语义切分）与块级（dense 向量化）两种调用都能处理。
        """

        def _vec(self, text):
            v = [0.0, 0.0]
            if "apple" in text:
                v[0] += 1.0
            if "car" in text:
                v[1] += 1.0
            return v

        def embed_documents(self, texts):
            return [self._vec(t) for t in texts]

        def embed_query(self, text):
            return [1.0, 1.0]

    service = KnowledgeService(
        embedder=ControlledEmbedder(), reranker=FakeReranker(), milvus=milvus,
        chunking_mode="semantic", chunk_size=700,
    )
    pdf = make_pdf("\n".join(sentences))  # 句间空格提取时可能丢失，用换行分隔

    result = service.ingest_pdf(role_id=2, pdf_bytes=pdf, source="s.pdf")

    assert result["chunks"] == 2
    texts = [c["text"] for c in milvus.collections[2]]
    assert any("apple" in t and "car" not in t for t in texts)
    assert any("car" in t and "apple" not in t for t in texts)


async def test_ingest_parent_child_mode_stores_parents_and_links(svc):
    """parent_child 模式：父块入库 + 子块带 parent_id。"""
    _, milvus = svc
    service = KnowledgeService(
        embedder=FakeEmbedder(), reranker=FakeReranker(), milvus=milvus,
        chunking_mode="parent_child", parent_chunk_size=300, chunk_size=100,
    )
    pdf = make_pdf("parent one content. " * 40 + "parent two content. " * 40)

    result = service.ingest_pdf(role_id=2, pdf_bytes=pdf, source="pc.pdf")

    assert result["chunks"] >= 2
    stored = milvus.collections[2]
    assert all(c.get("parent_id") for c in stored)  # 每个子块有父链接
    parents = milvus.parents.get(2, [])
    assert len(parents) >= 2  # 父块已入库
    parent_ids = {c["parent_id"] for c in stored}
    assert parent_ids <= {p["parent_id"] for p in parents}  # 链接完整


async def test_retrieve_parent_child_mode_returns_full_parents(svc):
    """检索命中子块 → 返回完整父块文本（而非子块）。"""
    service, milvus = svc
    service2 = KnowledgeService(
        embedder=FakeEmbedder(), reranker=FakeReranker(), milvus=milvus,
        chunking_mode="parent_child", parent_chunk_size=300, chunk_size=100,
    )
    pdf = make_pdf("alpha topic content. " * 40 + "beta topic content. " * 40)
    service2.ingest_pdf(role_id=2, pdf_bytes=pdf, source="pc2.pdf")

    results = await service2.retrieve(role_id=2, query="alpha topic", top_k=2, recall_k=10)

    assert results, "应返回父块文本"
    assert any("alpha topic" in r for r in results)
    # 父块文本比子块长（返回的是父块而非子块）
    child_lens = {len(c["text"]) for c in milvus.collections[2]}
    assert all(len(r) > max(child_lens) - 20 or len(r) >= 200 for r in results)


async def test_ingest_image_uses_ocr_then_chunks(svc):
    """图片入库：OCR 文本 → 清洗 → 分块 → 向量化。"""
    import fitz as _fitz

    service, milvus = svc

    class FakeOCR:
        def __call__(self, image_bytes):
            return "ocr text line one.\nocr text line two."

    service.ocr_engine = FakeOCR()
    pix = _fitz.Pixmap(_fitz.csRGB, _fitz.IRect(0, 0, 10, 10))
    pix.set_rect(pix.irect, (255, 255, 255))
    png = pix.tobytes("png")

    result = service.ingest_image(role_id=2, image_bytes=png, source="img.png")

    assert result["chunks"] >= 1
    joined = "\n".join(c["text"] for c in milvus.collections[2])
    assert "ocr text line one" in joined
    assert "ocr text line two" in joined


def test_ingest_pdf_includes_extracted_tables(svc):
    """PDF 入库：正文与表格文本合并入库。"""
    import fitz as _fitz

    service, milvus = svc
    doc = _fitz.open()
    page = doc.new_page()
    for x in (100, 200, 300):
        page.draw_line((x, 100), (x, 180))
    for y in (100, 140, 180):
        page.draw_line((100, y), (300, y))
    page.insert_text((110, 130), "name", fontsize=10)
    page.insert_text((210, 130), "value", fontsize=10)
    page.insert_text((110, 170), "salt", fontsize=10)
    page.insert_text((210, 170), "5g", fontsize=10)
    pdf = doc.tobytes()

    result = service.ingest_pdf(role_id=2, pdf_bytes=pdf, source="table.pdf")

    assert result["chunks"] >= 1
    joined = "\n".join(c["text"] for c in milvus.collections[2])
    assert "name" in joined and "5g" in joined  # 表格内容已入库


async def test_bm25_stats_rebuilt_after_new_ingest(svc):
    """新增文档后再次检索，BM25 词表应包含新文档的词（缓存失效）。"""
    service, milvus = svc
    service.ingest_pdf(role_id=2, pdf_bytes=make_pdf("old vocab. " * 50), source="a.pdf")
    await service.retrieve(role_id=2, query="old vocab", top_k=1, recall_k=5)
    first_fit_vocab_size = len(service._bm25_cache[2].vocab)

    service.ingest_pdf(role_id=2, pdf_bytes=make_pdf("new vocab. " * 50), source="b.pdf")
    await service.retrieve(role_id=2, query="new vocab", top_k=1, recall_k=5)

    assert len(service._bm25_cache[2].vocab) > first_fit_vocab_size


def test_ingest_pdf_applies_text_cleaning(svc):
    """入库前清洗：控制字符/乱码行不进入分块。"""
    import fitz

    service, milvus = svc
    doc = fitz.open()
    page = doc.new_page()
    # ASCII 内容 + 乱码字符（base-14 字体下非 ASCII 会变点，清洗主要断言控制字符）
    page.insert_text((72, 72), "content line one.\n===\ncontent line two.", fontsize=12)
    pdf = doc.tobytes()

    service.ingest_pdf(role_id=2, pdf_bytes=pdf, source="c.pdf")

    joined = "\n".join(c["text"] for c in milvus.collections[2])
    assert "content line one" in joined
    assert "content line two" in joined
    assert "===" not in joined  # 清洗掉了纯符号行


def test_ingest_same_pdf_twice_dedupes_chunks(svc):
    """同一 PDF 重复上传：第二次不新增任何块（hash 去重）。"""
    service, milvus = svc
    pdf = make_pdf("content. " * 50)

    first = service.ingest_pdf(role_id=2, pdf_bytes=pdf, source="d.pdf")
    second = service.ingest_pdf(role_id=2, pdf_bytes=pdf, source="d.pdf")

    assert first["chunks"] >= 1
    assert second["chunks"] == 0
    assert len(milvus.collections[2]) == first["chunks"]


def test_ingest_new_pdf_adds_only_new_chunks(svc):
    """部分重叠的 PDF：只入库新增块，重复块跳过。"""
    service, milvus = svc
    base = make_pdf("shared content. " * 50)
    extended = make_pdf("shared content. " * 50 + "new unique content. " * 50)

    first = service.ingest_pdf(role_id=2, pdf_bytes=base, source="e.pdf")
    second = service.ingest_pdf(role_id=2, pdf_bytes=extended, source="e2.pdf")

    assert first["chunks"] >= 1
    assert 0 < second["chunks"] < first["chunks"] + 100
    # 总块数 = 两轮去重后的有效块数
    assert len(milvus.collections[2]) == first["chunks"] + second["chunks"]
    joined = "".join(c["text"] for c in milvus.collections[2])
    assert "shared content" in joined
    assert "new unique content" in joined


def test_reingest_same_source_replaces_document(svc):
    """同 source 重传：旧块删除、新块入库（文档级替换）。"""
    service, milvus = svc
    pdf1 = make_pdf("version one content. " * 50)
    pdf2 = make_pdf("version two content. " * 50)

    service.ingest_pdf(role_id=2, pdf_bytes=pdf1, source="f.pdf")
    service.ingest_pdf(role_id=2, pdf_bytes=pdf2, source="f.pdf")

    joined = "".join(c["text"] for c in milvus.collections[2])
    assert "version one" not in joined
    assert "version two" in joined


def test_delete_source_removes_all_chunks(svc):
    service, milvus = svc
    service.ingest_pdf(role_id=2, pdf_bytes=make_pdf("content. " * 50), source="a.pdf")
    service.ingest_pdf(role_id=2, pdf_bytes=make_pdf("content. " * 50), source="b.pdf")

    removed = service.delete_source(role_id=2, source="a.pdf")

    assert removed >= 1
    assert all(c["source"] != "a.pdf" for c in milvus.collections[2])


# ---------- 长期记忆服务（原 app/services/memory_service.py） ----------

from app.services.memory_service import MemoryService


class NamedFakeMilvus:
    """按 collection 名存储的假 Milvus（记忆服务用）。"""

    def __init__(self):
        self.collections = {}  # name -> list[dict(chunk)]
        self.searches = []

    def ensure_named_collection(self, name):
        self.collections.setdefault(name, [])

    def insert_into(self, name, records):
        self.ensure_named_collection(name)
        self.collections[name].extend(records)

    def hybrid_search_in(self, name, query_dense, query_sparse, limit):
        self.searches.append((name, limit))
        return [c["text"] for c in self.collections.get(name, [])][:limit]

    def all_texts_in(self, name):
        return [c["text"] for c in self.collections.get(name, [])]


class MemoryEmbedder:
    """按内容生成向量：猫话题 → [1,0]，运动话题 → [0,1]。"""

    def embed_documents(self, texts):
        return [self._v(t) for t in texts]

    def embed_query(self, text):
        return self._v(text)

    def _v(self, text):
        v = [0.0, 0.0]
        if "猫" in text:
            v[0] += 1.0
        if "运动" in text:
            v[1] += 1.0
        return v


def test_memory_service_remembers_turn_as_single_entry():
    milvus = NamedFakeMilvus()
    svc = MemoryService(embedder=MemoryEmbedder(), reranker=None, milvus=milvus)

    svc.remember(user_id=1, role_id=2, role_name="小阳",
                 user_input="我养了一只猫", reply="真可爱！")

    entries = milvus.collections.get(svc._memory_collection(1, 2), [])
    assert len(entries) == 1
    text = entries[0]["text"]
    assert "用户：我养了一只猫" in text
    assert "小阳：真可爱！" in text
    assert entries[0]["summary"]
    assert "dense" in entries[0] and "sparse" in entries[0]


def test_memory_service_retrieves_related_memories_with_threshold():
    milvus = NamedFakeMilvus()
    svc = MemoryService(embedder=MemoryEmbedder(), reranker=None, milvus=milvus)
    svc.remember(1, 2, "小阳", "我养了一只猫", "真可爱！")
    svc.remember(1, 2, "小阳", "我每天去运动", "注意休息")

    related = svc.retrieve(user_id=1, role_id=2, query="我的猫最近怎么样", top_k=3)

    assert any("猫" in m for m in related)
    assert all("运动" not in m for m in related)  # 不相关记忆被相似度阈值过滤


def test_memory_service_returns_empty_without_memories():
    milvus = NamedFakeMilvus()
    svc = MemoryService(embedder=MemoryEmbedder(), reranker=None, milvus=milvus)
    assert svc.retrieve(user_id=1, role_id=2, query="任何问题") == []


def test_memory_service_isolation_between_users():
    milvus = NamedFakeMilvus()
    svc = MemoryService(embedder=MemoryEmbedder(), reranker=None, milvus=milvus)
    svc.remember(1, 2, "小阳", "我养了一只猫", "真可爱！")

    other = svc.retrieve(user_id=99, role_id=2, query="我的猫", top_k=3)

    assert other == []
