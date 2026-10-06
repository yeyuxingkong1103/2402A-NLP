# ============================================================
# 工单编号：人工智能NLP-RAG-功能测试及评估
# 项目名称：PDF文档的功能测试及评估
# 文件：evaluate_ccf.py
# 说明：对 CCF 年报的 10 题进行 RAGAS 评估
# ============================================================
import torch
from typing import Any, Optional, List
from ragas import evaluate, RunConfig
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset
from langchain_core.language_models.llms import LLM
from rag_engine import RAGEngine
from ground_truth_ccf import GROUND_TRUTH_CCF


# 加载 CCF 向量库
engine = RAGEngine(
    model_path="/root/autodl-tmp/models/models/Qwen--Qwen2.5-7B-Instruct/snapshots/master",
    db_path="/root/autodl-tmp/projects/RAG/ccf_vector_db",
    embed_path="/root/autodl-tmp/models/models/AI-ModelScope--bge-small-zh-v1.5/snapshots/master",
    reranker_path="/root/autodl-tmp/models/bge-reranker-base",
)


# 包装 Qwen 为 LangChain LLM（供 RAGAS 调用）
class QwenForEval(LLM):
    model: Any
    tokenizer: Any

    @property
    def _llm_type(self) -> str:
        return "qwen"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs,
    ) -> str:
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


# 收集 10 题的问答 + 检索上下文
questions, answers, contexts = [], [], []
for i, q in enumerate(GROUND_TRUTH_CCF.keys(), 1):
    print(f"[{i}/10] 处理：{q[:40]}...")
    ans, docs = engine.ask(q, retrieval_mode="hybrid", rerank_method="bge")
    questions.append(q)
    answers.append(ans)
    contexts.append([d.page_content for d in docs])

dataset = Dataset.from_dict({
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": [GROUND_TRUTH_CCF[q] for q in questions],
})

print("\n开始 RAGAS 评估（约 5-10 分钟）...")
run_config = RunConfig(timeout=300, max_retries=3, max_wait=60, max_workers=4)

result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy],
    llm=eval_llm_wrapper,
    embeddings=eval_emb_wrapper,
    run_config=run_config,
)

print("=" * 60)
print("CCF 年报 RAGAS 评估结果：")
print(result)
print("=" * 60)