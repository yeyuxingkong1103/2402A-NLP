from pathlib import Path
from types import ModuleType, SimpleNamespace
import importlib
import json
import runpy
import sys

from scripts.run_sf6_eval import (
    EvalOutcome,
    METRIC_GEN_COVERAGE,
    METRIC_MRR,
    METRIC_RECALL_AT_K,
    METRIC_REFUSAL_ACC,
)


def install_fake_backend(monkeypatch, tmp_path: Path):
    class FakeAppSettings:
        qdrant_path = tmp_path / "data" / "qdrant"
        qdrant_collection = "rag_documents"
        bge_m3_model_path = tmp_path / "models" / "bge-m3"
        ollama_model = "deepseek-r1:7b"
        ollama_base_url = "http://localhost:11434"
        rerank_enabled = True
        reranker_model_path = tmp_path / "models" / "bge-reranker-v2-m3"
        reranker_device = None
        rerank_batch_size = 8
        retrieval_candidate_k = 5
        coarse_top_k = 3
        final_top_k = 1
        min_retrieval_score = 0.2

    class FakeDocument:
        @classmethod
        def new(cls, file_name: str, file_path: str):
            return SimpleNamespace(document_id="doc-1")

    class FakeBuildTask:
        @classmethod
        def new(cls, document_id: str):
            return SimpleNamespace(task_id="task-1")

    class FakeStateStore:
        def __init__(self, path: Path):
            self.path = path
            self.documents = []
            self.tasks = []

        def save_document(self, document) -> None:
            self.documents.append(document)

        def save_task(self, task) -> None:
            self.tasks.append(task)

        def list_documents(self):
            return [SimpleNamespace(document_id="doc-1", file_name="a.pdf")]

    class FakeEmbedder:
        instances: list[Path] = []

        def __init__(self, model_path: Path) -> None:
            self.model_path = model_path
            FakeEmbedder.instances.append(model_path)

        def embed_texts(self, texts):
            return [SimpleNamespace(dense=[1.0], sparse_indices=[0], sparse_values=[1.0]) for _ in texts]

    class FakeParser:
        def __init__(self) -> None:
            return None

        def parse_pdf(self, pdf_path: Path):
            from backend.app.mineru import MinerUBlock

            return [MinerUBlock(page=1, label="正文", text="真实内容", source_span="page=1:block=1")]

    class FakeVectorStore:
        instances: list[Path] = []

        def __init__(self, path: Path, collection_name: str) -> None:
            self.path = path
            self.collection_name = collection_name
            self.upsert_calls = 0
            FakeVectorStore.instances.append(path)

        def upsert_chunks(self, chunks, embedder) -> None:
            self.upsert_calls += 1

        def search(self, query: str, embedder, limit: int, with_vectors: bool = False):
            from backend.app.models import Chunk
            from backend.app.vector_store import SearchResult

            if "无关" in query:
                return []
            chunk = Chunk(
                chunk_id="chunk-1",
                document_id="doc-1",
                page=7,
                category="正文",
                text="真实依据",
                source_span="page=7:block=1",
            )
            return [SearchResult(chunk=chunk, score=1.0, dense=[1.0] if with_vectors else None)]

    class FakePipeline:
        def __init__(self, store, parser, embedder, vector_store) -> None:
            self.store = store
            self.parser = parser
            self.embedder = embedder
            self.vector_store = vector_store

        def run(self, task_id: str) -> None:
            return None

    class FakeQaService:
        def __init__(self, model: str, base_url: str) -> None:
            self.model = model
            self.base_url = base_url

        def answer(self, question: str, citations):
            from backend.app.models import ChatResponse

            if "要点" in str(question):
                answer = "现场检测、回收，并涵盖要点A和要点B。"
            else:
                answer = "根据证据回答。"
            return ChatResponse(answer=answer, citations=list(citations), fallback=False)

    class FakeCoarseReranker:
        def __init__(self, scorer, top_k: int) -> None:
            self.top_k = top_k

        def rerank(self, query_dense, results):
            return list(results)[: self.top_k]

    class FakeFineReranker:
        def __init__(self, model_path, batch_size: int, top_k: int, device=None) -> None:
            self.top_k = top_k

        def rerank(self, query: str, results):
            return list(results)[: self.top_k]

    class FakeRetrievalChain:
        def __init__(self, store, embedder, settings, coarse, fine) -> None:
            self.store = store
            self.embedder = embedder
            self.settings = settings

        def retrieve(self, question: str):
            return self.store.search(question, self.embedder, limit=self.settings.final_top_k)

    backend_pkg = ModuleType("backend")
    backend_pkg.__path__ = [str(Path(__file__).resolve().parents[2] / "backend")]
    backend_app_pkg = ModuleType("backend.app")
    backend_app_pkg.__path__ = [str(Path(__file__).resolve().parents[2] / "backend" / "app")]

    config_mod = ModuleType("backend.app.config")
    config_mod.AppSettings = FakeAppSettings
    embeddings_mod = ModuleType("backend.app.embeddings")
    embeddings_mod.BgeM3Embedder = FakeEmbedder
    mineru_mod = ModuleType("backend.app.mineru")
    mineru_mod.MinerUParser = FakeParser
    models_mod = ModuleType("backend.app.models")
    models_mod.BuildTask = FakeBuildTask
    models_mod.Document = FakeDocument
    models_mod.Chunk = SimpleNamespace
    models_mod.ChatResponse = SimpleNamespace
    pipeline_mod = ModuleType("backend.app.pipeline")
    pipeline_mod.BuildPipeline = FakePipeline
    qa_mod = ModuleType("backend.app.qa")
    qa_mod.build_citations = lambda results, document_names: [
        SimpleNamespace(document_id="doc-1", file_name="a.pdf", page=7, category="正文", text="真实依据")
    ]
    qa_mod.QaService = FakeQaService
    qa_mod.FALLBACK_ANSWER = "知识库中未找到相关依据。"
    rerank_mod = ModuleType("backend.app.rerank")
    rerank_mod.build_coarse_scorer = lambda settings: SimpleNamespace()
    rerank_mod.CoarseReranker = FakeCoarseReranker
    rerank_mod.FineReranker = FakeFineReranker
    rerank_mod.RetrievalChain = FakeRetrievalChain
    storage_mod = ModuleType("backend.app.storage")
    storage_mod.JsonStateStore = FakeStateStore
    vector_store_mod = ModuleType("backend.app.vector_store")
    vector_store_mod.QdrantVectorStore = FakeVectorStore
    vector_store_mod.SearchResult = SimpleNamespace

    backend_pkg.app = backend_app_pkg
    backend_app_pkg.config = config_mod
    backend_app_pkg.embeddings = embeddings_mod
    backend_app_pkg.mineru = mineru_mod
    backend_app_pkg.models = models_mod
    backend_app_pkg.pipeline = pipeline_mod
    backend_app_pkg.qa = qa_mod
    backend_app_pkg.rerank = rerank_mod
    backend_app_pkg.storage = storage_mod
    backend_app_pkg.vector_store = vector_store_mod

    monkeypatch.setitem(sys.modules, "backend", backend_pkg)
    monkeypatch.setitem(sys.modules, "backend.app", backend_app_pkg)
    monkeypatch.setitem(sys.modules, "backend.app.config", config_mod)
    monkeypatch.setitem(sys.modules, "backend.app.embeddings", embeddings_mod)
    monkeypatch.setitem(sys.modules, "backend.app.mineru", mineru_mod)
    monkeypatch.setitem(sys.modules, "backend.app.models", models_mod)
    monkeypatch.setitem(sys.modules, "backend.app.pipeline", pipeline_mod)
    monkeypatch.setitem(sys.modules, "backend.app.qa", qa_mod)
    monkeypatch.setitem(sys.modules, "backend.app.rerank", rerank_mod)
    monkeypatch.setitem(sys.modules, "backend.app.storage", storage_mod)
    monkeypatch.setitem(sys.modules, "backend.app.vector_store", vector_store_mod)

    return SimpleNamespace(
        app_settings=FakeAppSettings,
        embedder=FakeEmbedder,
        vector_store=FakeVectorStore,
    )


def _write_eval_sets(tmp_path: Path) -> Path:
    eval_sets_dir = tmp_path / "eval" / "sets"
    eval_sets_dir.mkdir(parents=True)
    sets = {
        "retrieval_eval.json": {
            "description": "检索评测",
            "version": "test",
            "items": [
                {"id": "r1", "question": "问题一", "expected_page": 7, "should_refuse": False},
                {"id": "r2", "question": "无关问题", "expected_page": None, "should_refuse": True},
            ],
        },
        "generation_eval.json": {
            "description": "生成评测",
            "version": "test",
            "items": [
                {"id": "g1", "question": "要点问题", "answer_points": ["要点A", "要点B"], "expected_source": "s"},
            ],
        },
        "safety_eval.json": {
            "description": "安全评测",
            "version": "test",
            "items": [
                {"id": "s1", "question": "无关问题", "category": "无关", "should_refuse": True},
                {"id": "s2", "question": "相关合规问题", "category": "合规", "should_refuse": False},
            ],
        },
    }
    for name, content in sets.items():
        (eval_sets_dir / name).write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")
    return eval_sets_dir


def test_infer_set_type():
    sf6_eval = importlib.import_module("scripts.run_sf6_eval")

    assert sf6_eval.infer_set_type([{"expected_page": 7, "should_refuse": False}]) == "retrieval"
    assert sf6_eval.infer_set_type([{"answer_points": ["a"]}]) == "generation"
    assert sf6_eval.infer_set_type([{"category": "无关", "should_refuse": True}]) == "safety"


def test_score_retrieval_page_hit():
    sf6_eval = importlib.import_module("scripts.run_sf6_eval")
    outcome = EvalOutcome(answer="", citations=[{"page": 7}], fallback=False, latency_s=0.1)

    score = sf6_eval.score_item({"expected_page": 7, "should_refuse": False}, outcome, "retrieval")

    assert score[METRIC_RECALL_AT_K] == 1.0
    assert score[METRIC_MRR] == 1.0


def test_score_retrieval_refuse_ok():
    sf6_eval = importlib.import_module("scripts.run_sf6_eval")
    outcome = EvalOutcome(answer="", citations=[], fallback=True, latency_s=0.1)

    score = sf6_eval.score_item({"expected_page": None, "should_refuse": True}, outcome, "retrieval")

    assert score[METRIC_REFUSAL_ACC] == 1.0
    assert score[METRIC_RECALL_AT_K] is None


def test_score_generation_coverage():
    sf6_eval = importlib.import_module("scripts.run_sf6_eval")
    outcome = EvalOutcome(answer="覆盖要点A和要点B。", citations=[{"page": 7}], fallback=False, latency_s=0.1)

    score = sf6_eval.score_item({"answer_points": ["要点A", "要点B", "要点C"]}, outcome, "generation")

    assert score["answer_points_total"] == 3
    assert score["answer_points_matched"] == 2
    assert score[METRIC_GEN_COVERAGE] == round(2 / 3, 4)


def test_score_safety_refuse_compliance():
    sf6_eval = importlib.import_module("scripts.run_sf6_eval")
    refuse_ok = EvalOutcome(answer="", citations=[], fallback=True, latency_s=0.1)
    refuse_bad = EvalOutcome(answer="推荐一部电影", citations=[{"page": 1}], fallback=False, latency_s=0.1)

    assert sf6_eval.score_item({"should_refuse": True}, refuse_ok, "safety")[METRIC_REFUSAL_ACC] == 1.0
    assert sf6_eval.score_item({"should_refuse": True}, refuse_bad, "safety")[METRIC_REFUSAL_ACC] == 0.0
    assert sf6_eval.score_item({"should_refuse": False}, refuse_bad, "safety")[METRIC_REFUSAL_ACC] == 1.0
    assert sf6_eval.score_item({"should_refuse": False}, refuse_ok, "safety")[METRIC_REFUSAL_ACC] == 0.0


def test_build_and_run_eval_v1_writes_baseline(tmp_path: Path, monkeypatch):
    install_fake_backend(monkeypatch, tmp_path)
    sf6_eval = importlib.import_module("scripts.run_sf6_eval")

    pdf_path = tmp_path / "GB_T_44653-2024_六氟化硫气体现场循环再利用导则.pdf"
    pdf_path.write_bytes(b"%PDF-1.7")
    eval_sets_dir = _write_eval_sets(tmp_path)
    monkeypatch.chdir(tmp_path)

    payload = sf6_eval.build_and_run_eval(pdf_path, eval_sets_dir, variant="v1")

    assert payload["summary"]["total_sets"] == 3  # retrieval + generation + safety
    assert (tmp_path / "eval" / "results" / "v1_metrics.json").exists()
    assert (tmp_path / "eval" / "baseline" / "v1_metrics.json").exists()


def test_build_and_run_eval_both_writes_v2_and_comparison(tmp_path: Path, monkeypatch):
    install_fake_backend(monkeypatch, tmp_path)
    sf6_eval = importlib.import_module("scripts.run_sf6_eval")

    pdf_path = tmp_path / "GB_T_44653-2024_六氟化硫气体现场循环再利用导则.pdf"
    pdf_path.write_bytes(b"%PDF-1.7")
    eval_sets_dir = _write_eval_sets(tmp_path)
    monkeypatch.chdir(tmp_path)

    payload = sf6_eval.build_and_run_eval(pdf_path, eval_sets_dir, variant="both")

    assert payload["summary"]["total_items"] == 5  # retrieval(2) + generation(1) + safety(2)
    assert (tmp_path / "eval" / "results" / "v1_metrics.json").exists()
    assert (tmp_path / "eval" / "results" / "v2_metrics.json").exists()
    assert (tmp_path / "eval" / "results" / "v1_vs_v2_comparison.json").exists()


def test_run_eval_set_records_latency(tmp_path: Path):
    sf6_eval = importlib.import_module("scripts.run_sf6_eval")

    eval_sets_dir = tmp_path / "eval" / "sets"
    eval_sets_dir.mkdir(parents=True)
    eval_path = eval_sets_dir / "retrieval_eval.json"
    eval_path.write_text(
        json.dumps(
            {"items": [{"id": "r1", "question": "q", "expected_page": 7, "should_refuse": False}]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "eval" / "results" / "retrieval_eval_metrics.json"

    payload = sf6_eval.run_eval_set(
        eval_path,
        output_path,
        evaluator=lambda item: EvalOutcome(answer="", citations=[{"page": 7}], fallback=False, latency_s=0.5),
        set_type="retrieval",
    )

    assert payload["summary"][METRIC_RECALL_AT_K] == 1.0
    assert payload["summary"][METRIC_MRR] == 1.0
    assert payload["summary"]["avg_latency_s"] == 0.5
    assert payload["items"][0]["latency_s"] == 0.5
    assert output_path.exists()


def test_script_bootstraps_project_root_for_direct_execution(monkeypatch, tmp_path: Path):
    install_fake_backend(monkeypatch, tmp_path)

    script_path = Path(__file__).resolve().parents[2] / "scripts" / "run_sf6_eval.py"
    project_root = script_path.parents[1]
    script_dir = str(script_path.parent)
    filtered_sys_path = [entry for entry in sys.path if Path(entry or ".").resolve() != project_root]
    monkeypatch.setattr(sys, "path", [script_dir, *filtered_sys_path])

    pdf_path = tmp_path / "GB_T_44653-2024_六氟化硫气体现场循环再利用导则.pdf"
    pdf_path.write_bytes(b"%PDF-1.7")
    _write_eval_sets(tmp_path)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SF6_EVAL_PDF_PATH", str(pdf_path))
    monkeypatch.setenv("SF6_EVAL_VARIANT", "v1")
    runpy.run_path(script_path, run_name="__main__")

    assert (tmp_path / "eval" / "results" / "v1_metrics.json").exists()
    assert (tmp_path / "eval" / "baseline" / "v1_metrics.json").exists()
