import argparse
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from FlagEmbedding import BGEM3FlagModel, FlagReranker
from openai import OpenAI
from pymilvus import MilvusClient

from answer_test import rerank_hits
from hybrid_search import embed_query, hybrid_search


load_dotenv()

DEFAULT_COLLECTION = "hypertension_rag"
DEFAULT_URI = "http://127.0.0.1:19530"
DEFAULT_EMBEDDING_MODEL_PATH = r"D:\桌面缓存\bge-m3"
DEFAULT_RERANKER_MODEL_PATH = r"D:\桌面缓存\bge-reranker-v2-m3\bge-reranker-v2-m3"
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_TIMEOUT = 60
DEFAULT_MIN_RERANK_SCORE = 0.3
DEFAULT_OUTPUT = Path("data/processed/rag_answer_v2.json")
DEFAULT_QUESTIONS = [
    "氨氯地平的禁忌证是什么？",
    "高血压患者运动时要注意什么？",
    "心肌梗死患者应该选什么降压药？",
]


class DeepSeekError(RuntimeError):
    pass


def preview(text, limit=100):
    return " ".join((text or "").split())[:limit]


def build_retrieved_context(chunks):
    blocks = []
    for chunk in chunks:
        blocks.append(f"[来源: {chunk.get('section_path', '')}]\n{chunk.get('text', '')}")
    return "\n\n".join(blocks)


def build_prompt(question, retrieved_chunks, conversation_context="", long_term_memory=""):
    conversation_context = conversation_context or "无"
    long_term_memory = long_term_memory or "无"
    return f"""你是基层高血压管理助手。请仅基于以下上下文回答问题。

要求：

如果上下文没有相关信息，回答"根据现有指南无法回答该问题"

必须在回答的最后一行，用 [章节路径] 格式列出所有引用的来源。这是强制要求，不能不写。

涉及药物剂量、禁忌证时，必须原文引用

用简洁、易懂的中文回答

最近对话和用户已确认的长期记忆只用于理解上下文，不是医学证据。长期记忆和最近对话不能替代指南上下文，也不能作为引用来源。

【最近对话】
{conversation_context}

【用户已确认的长期记忆】
{long_term_memory}

【上下文】
{retrieved_chunks}

【问题】
{question}

text
"""


def build_fallback_prompt(question, conversation_context="", long_term_memory=""):
    conversation_context = conversation_context or "无"
    long_term_memory = long_term_memory or "无"
    return f"""你是基层高血压管理助手。当前未检索到本地知识库或数据库中的可用内容。

请基于你的通用医学知识，用中文给出谨慎、简洁的回答。

要求：

必须明确说明：本回答由大模型根据通用知识生成，不是基于数据库中的指南或本地知识库。

不要编造章节路径、数据库来源或指南引用。

不能假装拥有实时数据；对于天气、新闻、交通等实时信息，如果没有对应工具或数据源，必须明确说明无法获取实时信息，不要编造具体结果。

涉及诊断、用药、剂量、禁忌证和治疗调整时，提醒用户以医生判断和权威指南为准。

最近对话和用户已确认的长期记忆只用于理解上下文，不是医学证据。

【最近对话】
{conversation_context}

【用户已确认的长期记忆】
{long_term_memory}

【问题】
{question}

text
"""


def call_deepseek_api(prompt, model=DEFAULT_MODEL, timeout=DEFAULT_TIMEOUT):
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise DeepSeekError("DEEPSEEK_API_KEY is not set")

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            stream=False,
            timeout=timeout,
        )
    except Exception as exc:
        raise DeepSeekError(f"DeepSeek API request failed: {exc}") from exc

    return str(response.choices[0].message.content or "")


def has_citation(answer):
    return bool(re.search(r"\[[^\[\]]+\]", answer or ""))


def strip_r1_reasoning(answer):
    value = str(answer or "").strip()
    markers = ["最终答案：", "最终回答：", "答案："]
    positions = [(value.find(marker), marker) for marker in markers if value.find(marker) >= 0]
    if positions:
        position, marker = min(positions, key=lambda item: item[0])
        return value[position + len(marker) :].strip()

    lines = value.splitlines()
    reasoning_prefixes = ("思考", "让我", "首先")
    while lines and lines[0].strip().startswith(reasoning_prefixes):
        lines.pop(0)
    return "\n".join(lines).strip()


def ensure_citation(answer, top1_section_path):
    value = str(answer or "").strip()
    if "[" in value and "]" in value:
        return value, True
    return f"{value}\n\n（参考来源：{top1_section_path}）", "auto_appended"


def process_answer(answer, top1_section_path):
    original_length = len(answer or "")
    cleaned = strip_r1_reasoning(answer)
    cleaned_length = len(cleaned)
    print(f"R1 思考过滤：处理前长度={original_length}，处理后长度={cleaned_length}")
    final_answer, citation_status = ensure_citation(cleaned, top1_section_path)
    return final_answer, citation_status


def is_unanswerable(answer):
    return "根据现有指南无法回答" in (answer or "")


def format_retrieved(chunks):
    return [
        {
            "section_path": chunk.get("section_path", ""),
            "chunk_type": chunk.get("chunk_type", ""),
            "rerank_score": chunk.get("rerank_score"),
            "text_preview": preview(chunk.get("text", ""), 100),
        }
        for chunk in chunks
    ]


def retrieve_top5(client, collection, embedding_model, reranker, question, dense_top_k=20, bm25_top_k=20, rrf_k=60, fused_top_k=10, rerank_top_k=5):
    query_vector = embed_query(embedding_model, question)
    _, _, fused_hits = hybrid_search(
        client=client,
        collection=collection,
        question=question,
        query_vector=query_vector,
        dense_top_k=dense_top_k,
        bm25_top_k=bm25_top_k,
        rrf_k=rrf_k,
    )
    reranked_hits = rerank_hits(reranker, question, fused_hits[:fused_top_k])
    return reranked_hits[:rerank_top_k]


def answer_question(
    client,
    collection,
    embedding_model,
    reranker,
    question,
    model,
    timeout,
    conversation_context="",
    long_term_memory="",
):
    retrieval_start = time.perf_counter()
    top5 = retrieve_top5(client, collection, embedding_model, reranker, question)
    retrieval_ms = round((time.perf_counter() - retrieval_start) * 1000)

    if not top5 or float(top5[0].get("rerank_score", 0.0) or 0.0) < DEFAULT_MIN_RERANK_SCORE:
        prompt = build_fallback_prompt(
            question,
            conversation_context=conversation_context,
            long_term_memory=long_term_memory,
        )
        generation_start = time.perf_counter()
        answer = call_deepseek_api(prompt, model=model, timeout=timeout)
        generation_ms = round((time.perf_counter() - generation_start) * 1000)
        cleaned = strip_r1_reasoning(answer)
        notice = "未检索到本地知识库内容，本回答由大模型生成，不是基于数据库中的指南。"
        if "不是基于数据库" not in cleaned:
            cleaned = f"{notice}\n\n{cleaned}"
        return {
            "question": question,
            "retrieved": [],
            "answer": cleaned,
            "has_citation": False,
            "is_unanswerable": False,
            "answer_mode": "llm_fallback",
            "notice": notice,
            "latency": {
                "retrieval_ms": retrieval_ms,
                "generation_ms": generation_ms,
                "total_ms": retrieval_ms + generation_ms,
            },
        }

    prompt = build_prompt(
        question,
        build_retrieved_context(top5),
        conversation_context=conversation_context,
        long_term_memory=long_term_memory,
    )
    generation_start = time.perf_counter()
    answer = call_deepseek_api(prompt, model=model, timeout=timeout)
    generation_ms = round((time.perf_counter() - generation_start) * 1000)
    top1_section_path = top5[0].get("section_path", "") if top5 else ""
    answer, citation_status = process_answer(answer, top1_section_path)

    return {
        "question": question,
        "retrieved": format_retrieved(top5),
        "answer": answer,
        "has_citation": citation_status,
        "is_unanswerable": is_unanswerable(answer),
        "answer_mode": "rag",
        "notice": "本回答基于本地知识库检索结果生成。",
        "latency": {
            "retrieval_ms": retrieval_ms,
            "generation_ms": generation_ms,
            "total_ms": retrieval_ms + generation_ms,
        },
    }


def print_result(result):
    print(f"问题：{result['question']}")
    print("检索 top-5：")
    for index, item in enumerate(result["retrieved"], start=1):
        score = item.get("rerank_score")
        score_text = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
        print(f"  {index}. {item.get('section_path')} | {item.get('chunk_type')} | rerank={score_text}")
    print("最终回答：")
    print(result["answer"])
    print(f"总耗时：{result['latency']['total_ms']} ms")
    print()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="RAG answer generation with hybrid retrieval, BGE rerank, and DeepSeek API.")
    parser.add_argument("--uri", default=DEFAULT_URI)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL_PATH)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL_PATH)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--questions", nargs="*", default=DEFAULT_QUESTIONS)
    return parser.parse_args(argv)


def main():
    args = parse_args()

    embedding_model = BGEM3FlagModel(args.embedding_model, use_fp16=False)
    reranker = FlagReranker(args.reranker_model, use_fp16=False)
    client = MilvusClient(uri=args.uri)
    client.load_collection(collection_name=args.collection)

    results = []
    for question in args.questions:
        result = answer_question(
            client=client,
            collection=args.collection,
            embedding_model=embedding_model,
            reranker=reranker,
            question=question,
            model=args.model,
            timeout=args.timeout,
        )
        results.append(result)
        print_result(result)

    output = {"model": args.model, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"result_json: {args.output.as_posix()}")


if __name__ == "__main__":
    main()
