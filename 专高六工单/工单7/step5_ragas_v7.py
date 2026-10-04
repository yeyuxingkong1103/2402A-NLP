# 工单编号：人工智能 NLP-RAG-功能测试及评估
import json
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from config_v7 import DEEPSEEK_API_KEY, EMBED_MODEL_PATH

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com/v1",
    temperature=0
)
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_PATH)

with open("eval_results_v7.json", "r", encoding="utf-8") as f:
    results = json.load(f)

data = {
    "question": [r["question"] for r in results],
    "answer": [r["answer"] for r in results],
    "contexts": [[c["text"] for c in r["contexts"]] for r in results],
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