# 工单编号：人工智能 NLP-RAG-基于 PDF 文档的问答系统
import os
os.environ["RAGAS_DO_NOT_TRACK"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["JOBLIB_MULTIPROCESSING"] = "0"
import json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings

# 用 DeepSeek 兼容 OpenAI 接口
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key="你的DeepSeekKey",
    base_url="https://api.deepseek.com/v1",
    temperature=0
)

# RAGAS 需要 embeddings，用本地 bge_m3
embeddings = HuggingFaceEmbeddings(
    model_name=r"D:\models（文本嵌入模型）\bge_m3"
)

with open("eval_results.json", "r", encoding="utf-8") as f:
    results = json.load(f)

data = {
    "question": [r["question"] for r in results],
    "answer": [r["rag_answer"] for r in results],
    "contexts": [["第%s页" % p for p in r["rag_context_pages"]] for r in results],
    "ground_truth": ["" for _ in results]
}
dataset = Dataset.from_dict(data)

result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision],
    llm=llm,
    embeddings=embeddings
)
print(result)
