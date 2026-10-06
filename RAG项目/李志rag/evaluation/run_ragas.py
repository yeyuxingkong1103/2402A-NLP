import json
import os
from pathlib import Path

# RAGAS 会通过 GitPython读取版本信息；项目评测本身不依赖 Git。
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

import requests
from datasets import Dataset
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, faithfulness

load_dotenv()
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

BASE_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8000/api/v1")
USERNAME = os.getenv("RAG_TEST_USERNAME", os.getenv("ADMIN_USERNAME", "admin"))
PASSWORD = os.getenv("RAG_TEST_PASSWORD", os.getenv("ADMIN_PASSWORD", "ChangeMe123!"))


def main() -> None:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("请先设置 DEEPSEEK_API_KEY")
    login = requests.post(
        f"{BASE_URL}/auth/login",
        json={"username": USERNAME, "password": PASSWORD},
        timeout=30,
    )
    login.raise_for_status()
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
    roles = requests.get(f"{BASE_URL}/roles", headers=headers, timeout=30).json()
    role_id = roles[0]["id"]
    samples = json.loads(Path("evaluation/dataset.json").read_text(encoding="utf-8"))
    rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for index, sample in enumerate(samples):
        response = requests.post(
            f"{BASE_URL}/chat",
            headers=headers,
            json={"message": sample["question"], "role_id": role_id, "session_id": f"eval-{index}"},
            timeout=180,
        )
        response.raise_for_status()
        data = response.json()
        rows["question"].append(sample["question"])
        rows["answer"].append(data["answer"])
        rows["contexts"].append([item["text"] for item in data["sources"]])
        rows["ground_truth"].append(sample["reference"])
    llm = ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        api_key=key,
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )
    embeddings = HuggingFaceEmbeddings(
        model_name=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
        cache_folder=os.getenv("HUGGINGFACE_CACHE_DIR"),
        model_kwargs={"device": os.getenv("MODEL_DEVICE", "cpu"), "local_files_only": True},
        encode_kwargs={"normalize_embeddings": True},
    )
    result = evaluate(
        Dataset.from_dict(rows),
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=llm,
        embeddings=embeddings,
    )
    print(result)


if __name__ == "__main__":
    main()
