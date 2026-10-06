# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 【RAGAS评估模块 · evaluator.py】用真实 RAGAS 四指标评估 RAG 答案，评判 LLM 复用 DeepSeek
# 编写日期：2026-10-04
import sys
import re
import json
import types
from pathlib import Path
from typing import List, Dict
import numpy as np

# 引用标注模式：中文"（依据：第X页）"、英文"(Source: p.X)"
_CITE_PATTERNS = [
    re.compile(r"[（(]\s*依据[:：][^）)]*[）)]"),
    re.compile(r"[（(]\s*Source[:：][^）)]*[）)]", re.IGNORECASE),
]


def _strip_citation(text: str) -> str:
    """
    剥离答案末尾的页码引用标注
    RAGAS faithfulness 会把答案拆成原子声明逐一核验，页码引用属于参考元信息
    而非实质答案内容，若不剥离会被误判为"无依据声明"从而拉低忠实度分数
    """
    out = text or ""
    for pat in _CITE_PATTERNS:
        out = pat.sub("", out)
    return out.strip()

# ---- 兼容 shim：ragas 0.4.3 硬编码导入旧路径 langchain_community.chat_models.vertexai，
# 而新版 langchain_community 已移除该模块。系统不用 VertexAI，注入占位模块避免导入失败 ----
try:
    from langchain_community.chat_models.vertexai import ChatVertexAI  # noqa: F401
except Exception:
    _shim = types.ModuleType("langchain_community.chat_models.vertexai")
    _shim.ChatVertexAI = None
    sys.modules["langchain_community.chat_models.vertexai"] = _shim

from datasets import Dataset
from langchain_openai import ChatOpenAI
from langchain_core.embeddings import Embeddings
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

import config
from embedder import Embedder


class _LocalEmbeddings(Embeddings):
    """把本地 bge-m3 Embedder 适配为 LangChain Embeddings 接口，供 RAGAS 使用"""

    def __init__(self):
        self._emb = Embedder()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量文档向量化"""
        return self._emb.encode(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        """单查询向量化"""
        return self._emb.encode_one(text).tolist()


def _judge_llm() -> ChatOpenAI:
    """构建 RAGAS 评判 LLM（DeepSeek，OpenAI 兼容协议；n=1 因 DeepSeek 不支持多候选）"""
    return ChatOpenAI(
        model=config.LLM_MODEL,
        api_key=config.LLM_API_KEY,
        base_url=config.LLM_BASE_URL,
        temperature=0.0,
        timeout=120,
        n=1,
    )


def run_ragas(records: List[Dict]) -> Dict:
    """
    执行真实 RAGAS 评估
    records: 每条含 {question, answer, contexts(list[str]), ground_truth}
    返回 {summary: {四指标均值}, per_question: [...]}
    """
    # ragas 要求字段齐全，统一补齐
    norm = []
    for r in records:
        if not r.get("contexts"):
            r["contexts"] = [""]
        norm.append({
            "question": r["question"],
            "answer": _strip_citation(r.get("answer", "")),  # 评估实质内容，剥离页码引用
            "contexts": r["contexts"],
            "ground_truth": r.get("ground_truth", ""),
        })

    dataset = Dataset.from_list(norm)
    # DeepSeek 仅支持单次生成(n=1)，将 answer_relevancy 逆向问题生成数设为 1，
    # 避免 ragas 请求多候选触发 BadRequest 导致该题指标被记 0
    answer_relevancy.strictness = 1
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    print(f"[INFO] RAGAS 评估开始，共 {len(norm)} 条，评判模型={config.LLM_MODEL}")
    # 复杂多块上下文的核验可能超过默认 180s，放宽到 400s 避免瞬时 TimeoutError
    from ragas.run_config import RunConfig
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=_judge_llm(),
        embeddings=_LocalEmbeddings(),
        raise_exceptions=False,   # 单条失败不拖垮整体
        run_config=RunConfig(timeout=400),
    )

    # 汇总均值：ragas 0.4.3 用 result[指标名] 返回每问分数列表
    metric_names = ["faithfulness", "answer_relevancy",
                    "context_precision", "context_recall"]
    summary = {}
    for name in metric_names:
        try:
            summary[name] = round(float(np.nanmean(result[name])), 4)
        except Exception:
            summary[name] = 0.0
    summary["n"] = len(norm)

    # 每问明细：直接用 result.scores（每问分数字典）与原始数据配对，
    # 不依赖 to_pandas 的列名（不同 ragas 版本列结构有差异）
    scores_list = result.scores
    per_q = []
    for i, item in enumerate(norm):
        sc = scores_list[i] if i < len(scores_list) else {}

        def _v(n_name):
            """从分数字典安全取值，NaN/缺失记 0"""
            try:
                v = float(sc.get(n_name))
                return 0.0 if np.isnan(v) else round(v, 4)
            except Exception:
                return 0.0

        per_q.append({
            "question": item["question"],
            "answer": item["answer"],
            "ground_truth": item.get("ground_truth", "") or "",
            **{name: _v(name) for name in metric_names},
        })
    print(f"[OK] RAGAS 完成: {summary}")
    return {"summary": summary, "per_question": per_q}


def load_ground_truth() -> Dict[str, Dict]:
    """加载工单 10 题 ground truth（question → item）"""
    path = config.BASE_DIR / "03-测试" / "ground_truth.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return {it["question"]: it for it in data["items"]}


def evaluate_workorder(ask_fn=None) -> Dict:
    """
    对工单 10 题做完整评估：RAG 问答（含引用）→ 配 ground truth → RAGAS
    ask_fn: 可注入问答函数（同步 ask 或异步桥接），默认 rag_engine.ask
    """
    import rag_engine
    ask_fn = ask_fn or rag_engine.ask
    gt_map = load_ground_truth()

    records: List[Dict] = []
    for q in config.WORKORDER_QUESTIONS:
        text = q["question"]
        resp = ask_fn(text)
        gt = gt_map.get(text, {})
        records.append({
            "id": q["id"],
            "question": text,
            "answer": resp.get("answer", ""),
            "contexts": [r["text"] for r in resp.get("refs", [])],
            "ground_truth": gt.get("ground_truth", ""),
            "latency_ms": resp.get("latency_ms"),
        })
    out = run_ragas(records)
    # 把延迟信息合并回明细
    lat_map = {r["question"]: r["latency_ms"] for r in records}
    id_map = {r["question"]: r["id"] for r in records}
    for p in out["per_question"]:
        p["id"] = id_map.get(p["question"])
        p["latency_ms"] = lat_map.get(p["question"])
    return out


if __name__ == "__main__":
    # 离线执行：python evaluator.py
    import sys
    res = evaluate_workorder()
    print(json.dumps(res["summary"], ensure_ascii=False, indent=2))
    Path("ragas_result.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8"
    )

# ====================================================================
# 技术备注：
# 1. RAG：RAGAS（RAG Assessment）是 RAG 专用无参考/有参考评估框架：
#    - faithfulness：答案是否忠于检索上下文（事实一致性，反幻觉）
#    - answer_relevancy：答案与问题的相关程度（问题→答案逆向生成校验）
#    - context_precision：检索上下文是否精准命中（ground truth 导向）
#    - context_recall：ground truth 信息是否被上下文覆盖
# 2. Transformer：评判过程由 LLM（Transformer）基于分解-核验完成。
# 3. Fine-tuning：评估结果可量化验证 Embedding/LLM 微调收益。
# ====================================================================
