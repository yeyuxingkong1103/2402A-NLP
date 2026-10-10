# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rag04.config import Settings, get_settings, PROJECT_ROOT
from rag04.pipeline import RAGPipeline, IngestStats, ask
from rag04.retrieve.bm25 import BM25Index
from rag04.schema import Chunk, Hit

GY1 = PROJECT_ROOT / "招股说明书1.pdf"
GY2 = PROJECT_ROOT / "招股说明书2.pdf"
needs_corpus = pytest.mark.skipif(not GY2.exists(), reason="语料缺失")


def test_ingest_stats_fields():
    st = IngestStats(doc_id="d", n_text=1, n_table=2, n_figure=3, n_chunk=6, seconds=1.0)
    assert st.n_chunk == 6
    assert st.warnings == []


def test_pipeline_health_reports_components():
    s = get_settings()
    p = RAGPipeline(s)
    h = p.health()
    assert set(["qdrant", "bm25", "figures", "clip", "llm"]).issubset(h.keys())


@needs_corpus
@pytest.mark.integration
def test_full_ingest_small_slice(tmp_path):
    """只跑前 3 页，验证编排连通性（真正全量见 scripts/build_index.py）。"""
    s = get_settings()
    with patch("rag04.pipeline._page_limit", 3):
        st = p_ingest_one(s, GY2, tmp_path)
    assert st.n_chunk > 0
    assert st.seconds > 0


def p_ingest_one(s, pdf, tmp_path):
    from rag04.pipeline import ingest_document
    from rag04.ingest.store import VectorStore
    store = VectorStore(s, path=tmp_path / "q")
    store.ensure_collections()
    out = ingest_document(pdf, s, store=store, page_limit=3)
    store.close()
    return out


def test_ask_returns_answer_object_with_mocked_retrieval(monkeypatch):
    s = get_settings()
    hits = [Hit(chunk_id="c1", doc_id="d", page=1, block_type="text",
                source_id="s", text="武汉力源本次发行2000万股", score=1.0)]

    monkeypatch.setattr(
        "rag04.pipeline._retrieve",
        lambda q, s, st, bm, ef, cf, boilerplate_ids=None: hits)

    fake_llm = MagicMock()
    fake_llm.generate.return_value = MagicMock(
        answer="2000万股", lang="zh", citations=[], hits=hits,
        latency_ms=1.0, llm_backend="deepseek", refused=False,
    )
    ans = ask("本次发行股数是多少", s, store=MagicMock(), bm25=MagicMock(), llm=fake_llm)
    assert ans.answer == "2000万股"


# --- RC1：样板判据的载入/缓存与 baseline 控制组隔离 ---

class _Rec:
    def __init__(self, rid, payload):
        self.id = rid
        self.payload = payload


class _FakeBPStore:
    """样板判据只用到 counts() 与 client.scroll。"""

    def __init__(self, pages=25):
        self._pages = pages

    def counts(self):
        return {"text_chunks": self._pages}

    @property
    def client(self):
        outer = self

        class _C:
            def scroll(self, coll, limit=100_000, with_payload=True):
                if coll != "text_chunks":
                    return [], None
                return ([_Rec(i, {"chunk_id": f"h{i}", "doc_id": "d",
                                  "page": p, "block_type": "text",
                                  "text": "武汉兴图新科电子股份有限公司 招股意向书"})
                         for i, p in enumerate(range(1, outer._pages + 1))], None)

        return _C()


def test_pipeline_builds_caches_and_passes_boilerplate(tmp_path, monkeypatch):
    s = Settings(pipeline_mode="full_04", data_dir=tmp_path)
    p = RAGPipeline(s)
    p.store = _FakeBPStore()
    p.bm25 = BM25Index()
    p._ensure_loaded()

    assert p.boilerplate is not None, "full_04 应载入样板判据"
    assert p.boilerplate.ids == {f"h{i}" for i in range(25)}
    cache = tmp_path / "boilerplate.json"
    assert cache.exists(), "判据必须落盘缓存，避免每次查询重扫索引"

    captured = {}

    def fake_ask(question, s_, store=None, bm25=None, llm=None,
                 boilerplate_ids=None):
        captured["ids"] = boilerplate_ids
        from rag04.schema import Answer
        return Answer(question=question, answer="", lang="zh")

    monkeypatch.setattr("rag04.pipeline.ask", fake_ask)
    p.ask("法定代表人是谁")
    assert captured["ids"] == p.boilerplate.ids, "问答必须把样板集合传给检索"


def test_baseline_pipeline_skips_boilerplate(tmp_path):
    """baseline_03 是控制组：不引入本阶段检索侧修复。"""
    s = Settings(pipeline_mode="baseline_03", data_dir=tmp_path)
    p = RAGPipeline(s)
    p.store = _FakeBPStore()
    p.bm25 = BM25Index()
    p._ensure_loaded()
    assert p.boilerplate is None
    assert not (tmp_path / "boilerplate.json").exists()


def test_pipeline_survives_boilerplate_build_failure(tmp_path, monkeypatch):
    """样板判据失败只降级不过滤，绝不阻断问答。"""
    s = Settings(pipeline_mode="full_04", data_dir=tmp_path)
    p = RAGPipeline(s)

    class _Boom(_FakeBPStore):
        def counts(self):
            raise RuntimeError("counts-boom")

        @property
        def client(self):
            raise RuntimeError("scroll-boom")

    p.store = _Boom()
    p.bm25 = BM25Index()
    p._ensure_loaded()
    assert p.boilerplate is None


# --- Fix 6：重建默认清库；build(names=...) 限定语料 ---

def _iso_settings(tmp_path, **kw):
    """把 Settings 的所有落盘路径指到 tmp_path，测试绝不碰真实 data/。"""
    return Settings(
        project_root=tmp_path, data_dir=tmp_path / "data",
        qdrant_path=tmp_path / "qdrant", models_dir=tmp_path / "models",
        fig_cache_dir=tmp_path / "figs", vlm_cache_dir=tmp_path / "vlm",
        log_dir=tmp_path / "logs", reports_dir=tmp_path / "reports", **kw)


class _FakeBuildStore:
    """build_all 只用到这几个方法。"""

    instances: list["_FakeBuildStore"] = []

    def __init__(self, settings):
        self.cleared = 0
        self.ensured = 0
        self.closed = False
        _FakeBuildStore.instances.append(self)

    def ensure_collections(self):
        self.ensured += 1

    def clear_collections(self):
        self.cleared += 1
        return {"text_chunks": 7}

    @property
    def client(self):
        class _C:
            def scroll(self, coll, limit=100_000, with_payload=True):
                return [], None

        return _C()

    def close(self):
        self.closed = True


class _FakeBM25:
    def __init__(self):
        self.chunks = []

    def build(self, chunks):
        self.chunks = list(chunks)

    def save(self, path):
        self.saved = path


def _stub_build_deps(monkeypatch, ingested):
    import rag04.ingest.store as store_mod
    import rag04.pipeline as p
    import rag04.retrieve.bm25 as bm25_mod

    _FakeBuildStore.instances = []
    monkeypatch.setattr(store_mod, "VectorStore", _FakeBuildStore)
    monkeypatch.setattr(bm25_mod, "BM25Index", _FakeBM25)
    monkeypatch.setattr(
        p, "ingest_document",
        lambda pdf, s, store=None: (ingested.append(Path(pdf).name)
                                    or IngestStats(doc_id=Path(pdf).stem, n_chunk=1)))


def test_build_all_names_restricts_corpus(tmp_path, monkeypatch):
    s = _iso_settings(tmp_path, corpus=("a.pdf", "b.pdf"))
    (tmp_path / "a.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "b.pdf").write_bytes(b"%PDF-1.4")
    ingested: list[str] = []
    _stub_build_deps(monkeypatch, ingested)

    from rag04.pipeline import build_all
    stats = build_all(s, names=["b.pdf"])
    assert ingested == ["b.pdf"], f"names 必须限定重建范围，实测 {ingested}"
    assert [st.doc_id for st in stats] == ["b"]

    ingested.clear()
    build_all(s)
    assert ingested == ["a.pdf", "b.pdf"], "缺省仍是配置语料全量"


def test_build_all_reset_clears_collections_when_asked(tmp_path, monkeypatch):
    s = _iso_settings(tmp_path, corpus=("a.pdf",))
    (tmp_path / "a.pdf").write_bytes(b"%PDF-1.4")
    _stub_build_deps(monkeypatch, [])

    from rag04.pipeline import build_all
    build_all(s, reset=False)
    assert _FakeBuildStore.instances[-1].cleared == 0
    build_all(s, reset=True)
    assert _FakeBuildStore.instances[-1].cleared == 1, "reset=True 必须先清空三库"


def test_pipeline_build_defaults_to_reset_and_threads_names(tmp_path, monkeypatch):
    """RAGPipeline.build(names=None, reset=True)：接口签名与线程行为（API/UI 依赖）。"""
    import rag04.pipeline as p

    s = _iso_settings(tmp_path, corpus=("a.pdf",))
    captured = {}

    def fake_build_all(s_, reset=False, names=None):
        captured["reset"] = reset
        captured["names"] = names
        return [IngestStats(doc_id="d", n_chunk=1)]

    monkeypatch.setattr(p, "build_all", fake_build_all)
    monkeypatch.setattr(RAGPipeline, "warmup", lambda self: captured.setdefault("warmup", True))
    pipe = RAGPipeline(s)

    assert pipe.build() is not None
    assert captured["reset"] is True, "重建默认必须清库（新旧 chunk_id 共存会污染检索）"
    assert captured["names"] is None and captured["warmup"] is True

    pipe.build(names=["招股说明书1.pdf"], reset=False)
    assert captured["names"] == ["招股说明书1.pdf"] and captured["reset"] is False


def test_preload_survives_sklearn_import_failure(monkeypatch):
    """回归：sklearn 预热失败必须静默吞掉，不得在导入期抛异常。

    旧实现把 logger 定义在 _preload_fragile_deps() 调用点之后，预热失败时
    反而在 except 里抛 NameError，与「失败不阻断导入」的承诺相悖。
    必须让模块体在全新命名空间执行（delitem 后重新 import）才能复现：
    reload 复用旧命名空间，logger 仍有旧绑定，测不出该顺序问题。
    """
    import builtins
    import importlib
    import sys

    import rag04.pipeline  # noqa: F401  确保原模块在 sys.modules，供 delitem 恢复

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sklearn":
            raise ImportError("simulated sklearn failure")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, "rag04.pipeline")
    importlib.import_module("rag04.pipeline")   # win32 下导入期即触发预热


class _FakeStore:
    """按 block_type 校验维度（模拟 Qdrant 的维度约束），并留档写入内容。"""

    DIMS = {"text": 1024, "table": 1024, "image": 512}

    def __init__(self):
        self.chunks: list[Chunk] = []
        self.vectors: list[list[float]] = []

    def upsert_chunks(self, chunks, vectors):
        if len(chunks) != len(vectors):
            raise ValueError(f"数量不匹配：{len(chunks)} vs {len(vectors)}")
        for c, v in zip(chunks, vectors):
            if len(v) != self.DIMS[c.block_type]:
                raise ValueError(
                    f"维度不匹配：{c.chunk_id} 是 {c.block_type} 块，"
                    f"收到 {len(v)} 维（应为 {self.DIMS[c.block_type]}）")
        self.chunks = list(chunks)
        self.vectors = [list(v) for v in vectors]
        return len(chunks)


class _FakeDoc:
    page_count = 1

    def close(self):
        pass


def _text_chunk() -> Chunk:
    return Chunk(chunk_id="txt1", doc_id="d", page=1, block_type="text",
                 source_id="text#1#0", text="正文", extra={})


def _image_chunk_no_clip() -> Chunk:
    return Chunk(chunk_id="img1", doc_id="d", page=1, block_type="image",
                 source_id="fig#1", text="图：组织结构图", extra={})


def _prep_ingest(monkeypatch, chunks):
    """把 ingest_document 的解析依赖全部换成桩，只让「向量化+入库」分支跑真代码。"""
    import rag04.ingest.chunker as chunker
    import rag04.ingest.loader as loader
    import rag04.ingest.tables as tables
    import rag04.pipeline as p
    import rag04.retrieve.embed as embed

    monkeypatch.setattr(loader, "open_pdf", lambda pdf: _FakeDoc())
    monkeypatch.setattr(loader, "iter_text_blocks", lambda doc, doc_id: iter(()))
    monkeypatch.setattr(tables, "iter_all_tables", lambda pdf, doc_id: iter(()))
    monkeypatch.setattr(p, "_extract_figures", lambda doc, doc_id, s, limit: [])
    monkeypatch.setattr(chunker, "build_chunks", lambda t, tb, f, s: list(chunks))
    monkeypatch.setattr(embed, "embed_texts",
                        lambda texts, s: [[0.0] * 1024 for _ in texts])
    monkeypatch.setattr(embed, "embed_one", lambda text, s: [0.0] * 1024)


def _prep_text_only(monkeypatch, blocks):
    """只替换解析依赖，让 build_chunks 与入库分支跑真代码（RC1-in/RC5 验证用）。"""
    import rag04.ingest.loader as loader
    import rag04.ingest.tables as tables
    import rag04.pipeline as p
    import rag04.retrieve.embed as embed

    monkeypatch.setattr(loader, "open_pdf", lambda pdf: _FakeDoc())
    monkeypatch.setattr(loader, "iter_text_blocks", lambda doc, doc_id: iter(blocks))
    monkeypatch.setattr(tables, "iter_all_tables", lambda pdf, doc_id: iter(()))
    monkeypatch.setattr(p, "_extract_figures", lambda doc, doc_id, s, limit: [])
    monkeypatch.setattr(embed, "embed_texts",
                        lambda texts, s: [[0.0] * 1024 for _ in texts])


def _header_body_blocks(n=25):
    from rag04.schema import TextBlock

    def tb(page, text):
        return TextBlock(doc_id="招股说明书2", page=page, bbox=(0, 0, 100, 10),
                         text=text)

    blocks = [tb(p, "武汉力源信息技术股份有限公司     招股意向书")
              for p in range(1, n + 1)]
    blocks += [tb(p, f"第{p}页的正文段落，内容各不相同。") for p in range(1, n + 1)]
    return blocks


def test_ingest_drops_boilerplate_blocks_in_full_mode(monkeypatch):
    """RC1-in：页眉块不得进入索引（重建后它们不应占用 BM25 词频与向量库）。"""
    from rag04.pipeline import ingest_document

    _prep_text_only(monkeypatch, _header_body_blocks())
    store = _FakeStore()
    st = ingest_document(GY2, get_settings(), store=store)

    assert st.n_text == 25 and st.n_text_boilerplate == 25
    assert store.chunks and all("招股意向书" not in c.text for c in store.chunks)
    assert len(store.chunks) == 25


def test_ingest_keeps_boilerplate_blocks_in_baseline_mode(monkeypatch):
    """baseline_03 是对照组：不得引入入库侧过滤。"""
    from rag04.pipeline import ingest_document

    _prep_text_only(monkeypatch, _header_body_blocks())
    store = _FakeStore()
    st = ingest_document(GY2, get_settings("baseline_03"), store=store)

    assert st.n_text_boilerplate == 0 and st.n_text == 50
    assert any("招股意向书" in c.text for c in store.chunks)


def test_image_chunk_without_clip_vector_uses_clip_text_tower(monkeypatch):
    """无 CLIP 图像向量时用 CLIP 文本塔（512 维同空间）兜底，绝不写 1024 维进图像库。"""
    import rag04.ingest.vlparser as vlp
    from rag04.pipeline import ingest_document

    _prep_ingest(monkeypatch, [_text_chunk(), _image_chunk_no_clip()])
    monkeypatch.setattr(vlp, "clip_encode_text", lambda text, s: [0.5] * 512)

    store = _FakeStore()
    st = ingest_document(GY2, get_settings(), store=store)

    assert st.n_chunk == 2
    img_idx = [i for i, c in enumerate(store.chunks) if c.block_type == "image"]
    assert len(img_idx) == 1                      # 图像块保留且被真正写入图像库
    assert len(store.vectors[img_idx[0]]) == 512


def test_image_chunk_without_clip_vector_dropped_when_clip_unavailable(
        monkeypatch, caplog):
    """CLIP 文本塔也失败时丢弃该块并告警，构建不得中断。"""
    import rag04.ingest.vlparser as vlp
    from rag04.pipeline import ingest_document

    _prep_ingest(monkeypatch, [_text_chunk(), _image_chunk_no_clip()])

    def boom(text, s):
        raise RuntimeError("CLIP 模型缺失")

    monkeypatch.setattr(vlp, "clip_encode_text", boom)

    store = _FakeStore()
    with caplog.at_level(logging.WARNING, logger="rag04.pipeline"):
        st = ingest_document(GY2, get_settings(), store=store)

    assert st.n_chunk == 2
    assert [c.block_type for c in store.chunks] == ["text"]   # 图像块被丢弃
    assert len(store.vectors) == 1 and len(store.vectors[0]) == 1024
    assert "img1" in caplog.text                              # 告警点名 chunk_id


def test_image_chunk_wrong_dim_text_tower_vector_is_dropped(monkeypatch, caplog):
    """文本塔返回非 512 维（异常后端）时同样丢弃，绝不写错维度进图像库。"""
    import rag04.ingest.vlparser as vlp
    from rag04.pipeline import ingest_document

    _prep_ingest(monkeypatch, [_text_chunk(), _image_chunk_no_clip()])
    monkeypatch.setattr(vlp, "clip_encode_text", lambda text, s: [0.0] * 1024)

    store = _FakeStore()
    with caplog.at_level(logging.WARNING, logger="rag04.pipeline"):
        ingest_document(GY2, get_settings(), store=store)

    assert [c.block_type for c in store.chunks] == ["text"]
    assert "img1" in caplog.text
