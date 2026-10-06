# -*- coding: utf-8 -*-
# 工单编号：人工智能 NLP-RAG 项目-LightRAG 优化任务
# 模块：evaluation/run_ragas_lightrag —— RAG / LightRAG 两套方案的 RAGAS 标准评估（工单 12）
# 用法（独立 venv）：
#   set RAGAS_TAG=rag            && <venv_ragas>\python.exe evaluation\run_ragas_lightrag.py
#   set RAGAS_TAG=lightrag_mix   && <venv_ragas>\python.exe evaluation\run_ragas_lightrag.py
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.stdout.reconfigure(encoding="utf-8")

from ragas import evaluate  # noqa: E402
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.metrics import (Faithfulness, AnswerRelevancy, LLMContextRecall,  # noqa: E402
                           LLMContextPrecisionWithReference, AnswerCorrectness)
from ragas.run_config import RunConfig  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402
from langchain_core.embeddings import Embeddings  # noqa: E402
import json as _json  # noqa: E402
import urllib.request as _url  # noqa: E402


class OllamaNativeEmbeddings(Embeddings):
    """直接走 Ollama 原生 /api/embed（不用 /v1/embeddings）。
    实测：RAGAS 通过 OpenAI 兼容端点批量嵌入时会把 Ollama 服务拖死（请求挂起、GPU 空闲），
    换成原生端点后稳定（与项目其他模块同一个嵌入通道）。"""

    def __init__(self, base="http://localhost:11434", model="bge-m3", batch=8, timeout=180):
        self.base, self.model, self.batch, self.timeout = base, model, batch, timeout

    def _embed(self, texts):
        texts = [t if isinstance(t, str) else str(t) for t in texts]
        out = []
        for i in range(0, len(texts), self.batch):
            payload = {"model": self.model, "input": texts[i:i + self.batch], "keep_alive": "30m"}
            req = _url.Request(self.base + "/api/embed",
                               data=_json.dumps(payload).encode("utf-8"),
                               headers={"Content-Type": "application/json"})
            with _url.urlopen(req, timeout=self.timeout) as r:
                out.extend(_json.loads(r.read().decode("utf-8"))["embeddings"])
        return out

    def embed_documents(self, texts):
        return self._embed(list(texts))

    def embed_query(self, text):
        return self._embed([text])[0]

COLS = ["faithfulness", "answer_relevancy", "llm_context_precision_with_reference",
        "context_recall", "answer_correctness"]


def metric_of(key):
    return {"faithfulness": Faithfulness,
            "answer_relevancy": AnswerRelevancy,
            "context_precision": LLMContextPrecisionWithReference,
            "context_recall": LLMContextRecall,
            "answer_correctness": AnswerCorrectness,
            "llm_context_precision_with_reference": LLMContextPrecisionWithReference}[key]()


def main():
    tag = os.environ.get("RAGAS_TAG", "rag")
    base = os.environ.get("OLLAMA_OPENAI_BASE", "http://localhost:11434/v1")
    inp = os.path.join(HERE, "lightrag_ragas_in_%s.json" % tag)
    out_path = os.path.join(HERE, "ragas_lightrag_%s.json" % tag)
    data = json.load(open(inp, encoding="utf-8"))["detail"]
    print("[ragas] tag=%s in=%s n=%d" % (tag, os.path.basename(inp), len(data)))
    samples = [SingleTurnSample(user_input=r["question"], response=r["answer"],
                                retrieved_contexts=r["contexts"], reference=r["gold_answer"])
               for r in data]
    ds = EvaluationDataset(samples=samples)
    llm = LangchainLLMWrapper(ChatOpenAI(base_url=base, api_key="***", model="qwen2:7b",
                                         temperature=0.0, timeout=300, max_retries=1))
    emb = LangchainEmbeddingsWrapper(OllamaNativeEmbeddings())
    metrics = [metric_of(k.strip()) for k in
               os.environ.get("RAGAS_METRICS", "faithfulness,answer_relevancy,context_precision,context_recall,answer_correctness").split(",")]
    out = {"tag": tag, "n": len(data), "scores": {}}
    if os.path.exists(out_path):                       # 合并已有结果（支持只补跑个别指标）
        try:
            old = json.load(open(out_path, encoding="utf-8"))
            if old.get("n") == len(data):
                out["scores"].update(old.get("scores") or {})
                print("[ragas] 读入已有结果：", out["scores"], flush=True)
        except Exception:                              # noqa: BLE001
            pass
    if os.path.exists(out_path):                      # 已跑过的指标保留，便于单指标补跑
        try:
            prev = json.load(open(out_path, encoding="utf-8"))
            if prev.get("tag") == tag:
                out["scores"].update(prev.get("scores") or {})
                print("[ragas] 已合并历史结果：", list(out["scores"]), flush=True)
        except Exception:                             # noqa: BLE001
            pass
    if os.path.exists(out_path):                       # 合并已有结果（支持只补跑个别指标）
        try:
            old = json.load(open(out_path, encoding="utf-8"))
            if old.get("tag") == tag:
                out["scores"].update({k: v for k, v in (old.get("scores") or {}).items()})
        except Exception:                              # noqa: BLE001
            pass
    if os.path.exists(out_path):          # 断点重跑：保留已有指标，只补/覆盖本次跑的
        try:
            old = json.load(open(out_path, encoding="utf-8"))
            if old.get("tag") == tag:
                out["scores"].update(old.get("scores") or {})
        except Exception:  # noqa: BLE001
            pass
    if os.path.exists(out_path):          # 支持「补跑单个指标」：合并已有分数而不是覆盖
        try:
            old = json.load(open(out_path, encoding="utf-8"))
            if old.get("scores"):
                out["scores"].update({k: v for k, v in old["scores"].items() if v is not None})
                print("[ragas] 已合并历史分数：", list(out["scores"]), flush=True)
        except Exception as e:  # noqa: BLE001
            print("[ragas] 历史结果读取失败：", e, flush=True)
    if os.path.exists(out_path):                      # 增量补跑：保留已有指标
        try:
            old = json.load(open(out_path, encoding="utf-8"))
            if old.get("tag") == tag:
                out["scores"].update({k: v for k, v in (old.get("scores") or {}).items()})
        except Exception:  # noqa: BLE001
            pass
    if os.path.exists(out_path):                      # 续跑：合并已有分数
        try:
            out["scores"].update(json.load(open(out_path, encoding="utf-8")).get("scores") or {})
        except Exception:                             # noqa: BLE001
            pass
    print("\n==== RAGAS (%s, n=%d) ====" % (tag, len(data)), flush=True)
    for m in metrics:
        # 每个指标单独跑 + 增量落盘：某个指标卡住时不丢已完成的结果
        res = evaluate(ds, metrics=[m], llm=llm, embeddings=emb,
                       run_config=RunConfig(max_workers=1, timeout=180), raise_exceptions=False)
        df = res.to_pandas()
        col = next((c for c in df.columns
                    if c not in ("user_input", "response", "retrieved_contexts", "reference")), None)
        vals = [float(x) for x in df[col].tolist() if x == x] if col else []
        name = (getattr(m, "name", None) or col or "metric")
        out["scores"][name] = round(sum(vals) / len(vals), 4) if vals else None
        print("  %-42s %s   (有效样本 %d/%d)" % (name, out["scores"][name], len(vals), len(data)), flush=True)
        json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("saved ->", os.path.basename(out_path), flush=True)


if __name__ == "__main__":
    main()
