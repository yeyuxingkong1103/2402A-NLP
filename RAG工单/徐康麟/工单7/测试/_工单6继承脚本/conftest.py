# -*- coding: utf-8 -*-
"""
工单：人工智能NLP-RAG-图像内容解析及检索优化
文件：测试/conftest.py（T6：测试三级共用夹具）

提供：golden 判分库 / 16 题记录 / 诊断层证据 / 服务客户端（离线可用则跳过）/ 冻结产物读取 / PDF 只读。
不产生任何成绩：所有数字来自 测试/测试数据/ 下的实跑产物或本次在线实测。
"""
from __future__ import annotations

import importlib.util
import json
import urllib.error
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GOLDEN_LIB = ROOT / "测试" / "common" / "golden.py"
DATA = ROOT / "测试" / "测试数据"
SHOTS = ROOT / "测试" / "测试截图"
BASE_URL = "http://127.0.0.1:8123"
GOLDEN_SHA256_ANCHOR = "6bc8f0077c32084f345738687c01bea4a2ff02f2fe0448d7e2e6be9e38128254"
INDEX_CHUNK_TOTAL = 2676          # 20:42:06 重建后的现役索引
INDEX_IMAGE_CHUNKS = 77


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def golden():
    """判分库（normalize / judge_answer / check_retrieval_hit_layers / page_carries）。"""
    return _load(GOLDEN_LIB, "golden_lib_acceptance")


@pytest.fixture(scope="session")
def records(golden) -> list[dict]:
    return golden.load_golden()


@pytest.fixture(scope="session")
def by_id(records) -> dict:
    return {r["id"]: r for r in records}


@pytest.fixture(scope="session")
def diagnostic(golden) -> dict:
    return golden.load_diagnostic_evidence()


@pytest.fixture(scope="session")
def service():
    """在线服务客户端；/health 不可用时 skip（离线层不依赖服务）。"""
    class _Svc:
        def __init__(self, base: str):
            self.base = base

        def health(self, timeout=60):
            with urllib.request.urlopen(self.base + "/health", timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))

        def ask(self, question: str, lang: str = "zh", use_cache: bool = False,
                top_k: int = 5, timeout: float = 180):
            body = json.dumps({"question": question, "lang": lang, "use_cache": use_cache,
                               "top_k": top_k}, ensure_ascii=False).encode("utf-8")
            req = urllib.request.Request(self.base + "/api/ask", data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))

    svc = _Svc(BASE_URL)
    try:
        svc.health(timeout=10)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        pytest.skip(f"在线服务不可用（{BASE_URL}/health: {type(exc).__name__}）——在线层跳过")
    return svc


@pytest.fixture(scope="session")
def artifacts() -> dict:
    """读取冻结的评测产物（测试/测试数据/）。缺失即失败——产物应由 run_16q_eval.py 生成。"""
    out = {}
    for name in ("qa_results.jsonl", "accuracy_report.json", "negatives.jsonl",
                 "bilingual.jsonl", "run_meta.json"):
        p = DATA / name
        assert p.exists(), f"缺评测产物 {p}（先运行 测试/在线/run_16q_eval.py）"
        if name.endswith(".jsonl"):
            out[name] = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            out[name] = json.loads(p.read_text(encoding="utf-8"))
    return out


@pytest.fixture(scope="session")
def index_meta() -> dict:
    p = ROOT / "研发" / "data" / "index" / "index_manifest.json"
    assert p.exists(), f"缺索引 manifest {p}"
    return json.loads(p.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def pdfs():
    fitz = pytest.importorskip("fitz")
    docs = {}
    for name in ("招股说明书1.pdf", "招股说明书2.pdf"):
        docs[name] = fitz.open(ROOT / "研发" / "data" / "raw" / name)
    yield docs
    for d in docs.values():
        d.close()
