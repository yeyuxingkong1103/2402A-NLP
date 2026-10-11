# ============================================================
# 工单编号：人工智能NLP-RAG-Embedding模型微调任务
# 文件：evaluate_11.py
# 说明：对比微调前后嵌入模型的 context_precision 和 context_recall
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
            outputs = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)
        return self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        ).strip()


def run_eval(db_path, embed_path, tag):
    print(f"\n{'='*60}")
    print(f"评估 [{tag}]")
    print(f"{'='*60}")

    engine = GraphRAGEngine(
        model_path="/root/autodl-tmp/models/models/Qwen--Qwen2.5-7B-Instruct/snapshots/master",
        db_path=db_path,
        embed_path=embed_path,
        reranker_path="/root/autodl-tmp/models/models/BAAI--bge-reranker-base/snapshots/master",
        kg_path="/root/autodl-tmp/projects/RAG/kg_graph.json",
    )

    eval_llm = QwenForEval(model=engine.model, tokenizer=engine.tokenizer)
    eval_llm_wrapper = LangchainLLMWrapper(eval_llm)
    eval_emb_wrapper = LangchainEmbeddingsWrapper(engine.embeddings)

    questions, answers, contexts = [], [], []
    for i, q in enumerate(GROUND_TRUTH_CCF.keys(), 1):
        print(f"  [{i}/10] {q[:40]}...")
        ans, docs, triples = engine.ask(q)
        questions.append(q)
        answers.append(ans)
        ctx = [d.page_content for d in docs] + [f"图谱：{t}" for t in triples]
        contexts.append(ctx)

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": [GROUND_TRUTH_CCF[q] for q in questions],
    })

    print(f"\n[{tag}] RAGAS 评估中（约 5 分钟）...")
    result = evaluate(
        dataset,
        metrics=[context_precision, context_recall],
        llm=eval_llm_wrapper,
        embeddings=eval_emb_wrapper,
        run_config=RunConfig(timeout=300, max_retries=3, max_wait=60, max_workers=4),
    )
    print(f"\n[{tag}] 评估结果：{result}")
    return result


# ============ 微调前 ============
base_result = run_eval(
    db_path="/root/autodl-tmp/projects/RAG/ccf_vector_db",
    embed_path="/root/autodl-tmp/models/models/AI-ModelScope--bge-small-zh-v1.5/snapshots/master",
    tag="微调前",
)

# ============ 微调后 ============
ft_result = run_eval(
    db_path="/root/autodl-tmp/projects/RAG/ccf_vector_db_finetuned",
    embed_path="/root/autodl-tmp/projects/RAG/finetuned_bge",
    tag="微调后",
)

# ============ 汇总对比 ============
print("\n" + "=" * 60)
print("微调前后对比汇总")
print("=" * 60)
print(f"微调前：{base_result}")
print(f"微调后：{ft_result}")
print("=" * 60)
