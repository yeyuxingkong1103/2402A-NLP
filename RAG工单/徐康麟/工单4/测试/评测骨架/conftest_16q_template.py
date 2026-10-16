# -*- coding: utf-8 -*-
"""
工单：人工智能NLP-RAG-图像内容解析及检索优化
文件：测试/评测骨架/conftest_16q_template.py（**T6 骨架**：在线评测的 pytest 夹具模板）

用法（T6）：
  1. 复制为 `测试/在线/conftest.py`（或 `测试/在线/test_e2e_16q.py` 同目录下）；
  2. 按现场改 `BASE_URL` / `ENDPOINT` / 预热与独占自检命令；
  3. 数字只来自真实调用——**夹具本身不产生任何成绩**。

⚠️ 约束：不得在夹具里内联关键词/答案；判分与命中一律调用 `测试/common/golden.py`。
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GOLDEN_LIB = ROOT / "测试" / "common" / "golden.py"

#: 现场参数（T6 按部署实际填写；本骨架不预设可用端点）
BASE_URL = "http://127.0.0.1:8123"
ENDPOINT = "/api/ask"
REQUEST_TIMEOUT = 30.0


def _load_golden_lib():
    spec = importlib.util.spec_from_file_location("golden_lib", GOLDEN_LIB)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def golden():
    """golden 判分库（normalize / judge_answer / check_retrieval_hit / load_golden_index）。"""
    return _load_golden_lib()


@pytest.fixture(scope="session")
def golden_index(golden) -> dict:
    """16 题 golden 索引：key = 原始 id（5, 6, 1, 2, 3, 4, 260, 95, 33, 34, 957, 793, 795, 543, 531, 207）。"""
    return golden.load_golden_index()


@pytest.fixture(scope="session")
def ordered_ids(golden) -> list[int]:
    """按用户提示词顺序的 16 题 id（报告逐题表必须按此顺序）。"""
    return list(golden.EXPECTED_ORDER)


@pytest.fixture(scope="session")
def service_client():
    """问答服务客户端（骨架）。返回 dict；**失败时如实抛错/记录 error，不得返回伪造答案**。"""
    import requests

    class _Client:
        def __init__(self, base_url: str, endpoint: str, timeout: float):
            self.base_url, self.endpoint, self.timeout = base_url, endpoint, timeout

        def ask(self, question: str) -> dict:
            resp = requests.post(self.base_url.rstrip("/") + self.endpoint,
                                 json={"question": question}, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()

    return _Client(BASE_URL, ENDPOINT, REQUEST_TIMEOUT)


@pytest.fixture(scope="session")
def run_meta() -> dict:
    """运行元信息骨架：T6 必须把实际值填进来（`None` 表示尚未确认，禁止猜）。"""
    return {
        "base_url": BASE_URL,
        "endpoint": ENDPOINT,
        "model": None,                    # 待 T6 实跑填充
        "warmup_done": None,              # 待 T6 实跑填充
        "exclusive_ollama_verified": None,  # 待 T6 实跑填充（附 ollama ps / ESTABLISHED 证据）
        "generation_top_k": None,         # 待 T6 实跑填充（用户要求 K=3）
        "hit_judgement_top_k": 5,         # golden 固定
    }


@pytest.fixture(scope="session")
def results_dir(tmp_path_factory) -> Path:
    """本次评测的产物目录（qa_results.csv/jsonl、accuracy_report.json 写在这里）。"""
    return tmp_path_factory.mktemp("t6_eval")


@pytest.fixture(scope="session")
def judge(golden):
    """判分便捷函数：judge(record, answer) → 结构化判分结果。"""
    return lambda record, answer: golden.judge_answer(record, answer)


@pytest.fixture(scope="session")
def retrieval_hit(golden):
    """检索命中便捷函数：hit(record, chunks) → 命中结构化结果（前 K 条内命中即算）。"""
    return lambda record, chunks: golden.check_retrieval_hit(record, chunks)


def dump_results(results: list[dict], path: Path) -> None:
    """把逐题结果写成 JSONL（供报告与 accuracy_report.json 使用）；不填任何默认数字。"""
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in results) + "\n",
                    encoding="utf-8")
