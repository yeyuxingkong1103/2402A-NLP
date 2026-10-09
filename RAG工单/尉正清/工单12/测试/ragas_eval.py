# 工单编号：人工智能NLP-RAG项目-LightRAG优化
"""用 RAGAS 评估 RAG 与 LightRAG 两套系统（产出物 3 的后半）

四个指标，全都要，因为工单产出物写的是「RAGAS 评估指标对比」而不是某一个：

    faithfulness       回答是否忠于检索到的上下文（不编造）
    answer_relevancy   回答是否切题
    context_precision  检索到的上下文里，相关内容的排序质量（需要标准答案）
    context_recall     标准答案里的信息，检索上下文覆盖了多少（需要标准答案）

后两个需要 `reference`（标准答案），见 研发/testset.py 的 gold_answer。

⚠️ 跑法（需要 rag_gd1 环境，ragas/bge-m3 都在那儿）：

    D:/Anaconda/envs/rag_gd1/python.exe ragas_eval.py

用法：
    python ragas_eval.py [--kb rag|lightrag|both] [--max-contexts 8]
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DEV = HERE.parent / "研发"
sys.path.insert(0, str(DEV))

# 上下文条数上限：RAG 固定返回 TOP_K=8 段，LightRAG 返回的是一整块拼接文本
# （可能十几段）。不拉齐的话 context_precision 的算法在两边面对的候选数不同，
# 比出来的差异里混着「谁给的上下文多」。两边都截到前 N 段。
MAX_CONTEXTS = 8
CTX_CHARS = 900      # 与 RAG 的 CHUNK_SIZE=800 对齐，保证两边证据体量可比

METRIC_NAMES = ["faithfulness", "answer_relevancy",
                "context_precision", "context_recall"]


def load_records(kb):
    p = RESULTS / f"{kb}.json"
    if not p.exists():
        raise SystemExit(f"[缺少结果] {p} 不存在，先跑：run_qa.py --kb {kb}")
    return json.loads(p.read_text(encoding="utf-8"))["records"]


class BGEM3Embeddings:
    """给 RAGAS 用的 bge-m3 嵌入适配器。

    为什么不用 ragas 自带的 `HuggingFaceEmbeddings`：它在 0.4.3 里没有实现
    `embed_query`，而 `BaseRagasEmbeddings` 把这四个方法都标成了抽象方法，
    于是 `AnswerRelevancy` 一跑就 `AttributeError`。

    实测报错：
        AttributeError('HuggingFaceEmbeddings' object has no attribute 'embed_query')
    """
    def __init__(self, model_path, device="cuda"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(str(model_path), device=device)

    def _enc(self, texts):
        import numpy as np
        v = self.model.encode(list(texts), batch_size=16,
                              normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(v, dtype=np.float32)

    def embed_query(self, text):
        return self._enc([text])[0]

    def embed_documents(self, texts):
        return list(self._enc(texts))

    async def aembed_query(self, text):
        return self.embed_query(text)

    async def aembed_documents(self, texts):
        return self.embed_documents(texts)


def build_dataset(records, max_contexts, ctx_chars=0):
    """转成 RAGAS 的 EvaluationDataset。

    `ctx_chars` > 0 时把每段上下文截到该长度。两边都截，是为了让**证据体量可比**：
    RAG 的块由 CHUNK_SIZE=800 切出来，LightRAG 的段是整页（两千字以上），
    不截的话 context_precision 面对的信息量差一倍多。
    """
    from ragas import EvaluationDataset, SingleTurnSample

    samples = []
    for r in records:
        if not r["answer"] or r.get("error"):
            continue
        if not r.get("gold_answer"):
            continue                      # 没标准答案的题，context_* 算不了
        samples.append(SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=[(c[:ctx_chars] if ctx_chars else c)
                                for c in r["contexts"][:max_contexts] if c.strip()],
            reference=r["gold_answer"],
        ))
    return EvaluationDataset(samples=samples)


async def run(kb, max_contexts, ctx_chars):
    from ragas import aevaluate
    from ragas.llms import llm_factory
    from ragas.metrics import (AnswerRelevancy, ContextPrecision, ContextRecall,
                               Faithfulness)
    import openai

    from config import BGE_M3_PATH, LLM_API_BASE, LLM_API_KEY, LLM_MODEL

    records = load_records(kb)
    ds = build_dataset(records, max_contexts, ctx_chars)
    print(f"[{kb}] 可评估 {len(ds.samples)} 题"
          f"（共 {len(records)} 题，跳过无答案/无标准答案的）")
    if not ds.samples:
        return None

    client = openai.OpenAI(api_key=LLM_API_KEY, base_url=LLM_API_BASE)
    # ⚠️ 判分模型也必须关掉 reasoning，这是本轮最要紧的一处。
    # deepseek-flash 的思维链和正文共用 max_tokens 额度：不关的话
    #   ① 每次调用 33 秒而不是 1.9 秒 —— 全量评估从 40 分钟变成十几个小时；
    #   ② 思维链吃掉输出额度，结构化输出被截断，抛 IncompleteOutputException，
    #      对应指标的格子直接是空的（首轮实测 faithfulness 就整列缺失）。
    # max_tokens 一并给足，双保险。
    llm = llm_factory(LLM_MODEL, client=client, max_tokens=8192,
                      extra_body={"reasoning_effort": "none"})
    emb = BGEM3Embeddings(BGE_M3_PATH)

    result = await aevaluate(
        ds,
        metrics=[Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()],
        llm=llm, embeddings=emb, raise_exceptions=False,
    )
    scores = {k: (None if v != v else float(v))       # NaN -> None
              for k, v in result._repr_dict.items() if k in METRIC_NAMES}
    return scores, result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", choices=["rag", "lightrag", "both"], default="both")
    ap.add_argument("--max-contexts", type=int, default=MAX_CONTEXTS)
    ap.add_argument("--ctx-chars", type=int, default=CTX_CHARS,
                    help="每段上下文截断长度，0 表示不截")
    args = ap.parse_args()

    kbs = ["rag", "lightrag"] if args.kb == "both" else [args.kb]
    out = {}
    for kb in kbs:
        print(f"\n{'=' * 60}\n  RAGAS 评估：{kb}\n{'=' * 60}")
        try:
            res = asyncio.run(run(kb, args.max_contexts, args.ctx_chars))
        except Exception as exc:                       # noqa: BLE001
            print(f"  ✗ 失败：{type(exc).__name__}: {str(exc)[:200]}")
            continue
        if res is None:
            continue
        scores, _ = res
        out[kb] = scores
        for k, v in scores.items():
            print(f"    {k:20} {'—' if v is None else f'{v:.4f}'}")

    if out:
        (RESULTS / "ragas.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已写入 {RESULTS / 'ragas.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
