# ============================================================
# 工单编号：人工智能NLP-RAG-Graph RAG优化任务
# 项目名称：Graph RAG 优化
# 文件：evaluate_09.py
# 说明：测量 Graph RAG 的 context_precision 和 context_recall
# ============================================================
import torch
from typing import Any, Optional, List
from ragas import evaluate, RunConfig
from ragas.metrics import context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset
from langchain_core.language_models.llms import LLM
from graph_rag_engine import GraphRAGEngine
from ground_truth_ccf import GROUND_TRUTH_CCF


engine = GraphRAGEngine(
    model_path="/root/autodl-tmp/models/models/Qwen--Qwen2.5-7B-Instruct/snapshots/master",
    db_path="/root/autodl-tmp/projects/RAG/ccf_vector_db",
    embed_path="/root/autodl-tmp/models/models/AI-ModelScope--bge-small-zh-v1.5/snapshots/master",
    reranker_path="/root/autodl-tmp/models/bge-reranker-base",
    kg_path="/root/autodl-tmp/projects/RAG/kg_graph.json",
)


class QwenForEval(LLM):
    model: Any
    tokenizer: Any

    @property
    def _llm_type(self) -> str:
        return "qwen"

    def _call(self, prompt, stop=None, run_manager=None, **kwargs):
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=256, do_sample=False,
            )
        return self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        ).strip()


eval_llm = QwenForEval(model=engine.model, tokenizer=engine.tokenizer)
eval_llm_wrapper = LangchainLLMWrapper(eval_llm)
eval_emb_wrapper = LangchainEmbeddingsWrapper(engine.embeddings)

questions, answers, contexts = [], [], []
for i, q in enumerate(GROUND_TRUTH_CCF.keys(), 1):
    print(f"[{i}/10] {q[:40]}...")
    ans, docs, triples = engine.ask(q)
    questions.append(q)
    answers.append(ans)
    # 上下文 = 向量检索块 + 图谱三元组
    ctx = [d.page_content for d in docs] + [f"图谱：{t}" for t in triples]
    contexts.append(ctx)

dataset = Dataset.from_dict({
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": [GROUND_TRUTH_CCF[q] for q in questions],
})

print("\nRAGAS 评估中（约 5-8 分钟）...")
result = evaluate(
    dataset,
    metrics=[context_precision, context_recall],
    llm=eval_llm_wrapper,
    embeddings=eval_emb_wrapper,
    run_config=RunConfig(timeout=300, max_retries=3, max_wait=60, max_workers=4),
)

print("=" * 60)
print("Graph RAG 上下文指标评估结果：")
print(result)
print("=" * 60)