"""本地向量检索 + DeepSeek 答案生成。"""

from __future__ import annotations

import html
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import numpy as np
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
INDEX_PATH = DATA_DIR / "index.faiss"
CHUNKS_PATH = DATA_DIR / "chunks.json"
MODEL_PATH = Path(
    r"C:\Users\lenovo\.cache\huggingface\hub\models--BAAI--bge-small-zh-v1.5"
    r"\snapshots\7999e1d3359715c523056ef9478215996d62a620"
)


def save_faiss_index(index, path: Path) -> None:
    """让 Python 负责中文路径，避免 Windows FAISS 直接打开文件失败。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(faiss.serialize_index(index).tobytes())
    temporary.replace(path)


def load_faiss_index(path: Path):
    serialized = np.frombuffer(path.read_bytes(), dtype="uint8").copy()
    return faiss.deserialize_index(serialized)


@dataclass
class Chunk:
    chunk_id: str
    text: str
    title: str
    page: int
    content_type: str = "text"


@dataclass
class Hit:
    chunk: Chunk
    score: float


def normalize_question(question: str) -> str:
    text = html.unescape(question or "")
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        raise ValueError("问题不能为空")
    return text


def understand_query(question: str) -> dict:
    """用小型规则完成意图识别和检索词扩展，避免额外 LLM 请求。"""
    text = normalize_question(question)
    intent = "general"
    if re.search(r"多少|金额|收入|资本|资金", text):
        intent = "amount"
    elif re.search(r"比重|占比|比例|%|％", text):
        intent = "ratio"
    elif re.search(r"谁|法定代表人|董事长", text):
        intent = "person"
    elif re.search(r"哪个标准|技术标准|哪个工程", text):
        intent = "name"
    elif re.search(r"上游|下游|行业|领域", text):
        intent = "industry"

    queries = [text]
    short = re.sub(r"根据.*?(招股意向书|招股说明书)[,，]?​?", "", text)
    short = short.replace("武汉兴图新科电子股份有限公司", "兴图新科")
    if short != text:
        queries.append(short)
    if "分别" in text or "报告期" in text:
        keywords = re.sub(r"[？?]​?$", "", short)
        keywords = re.sub(r"是多少|为多少|有哪些", "", keywords)
        queries.append(keywords)

    return {
        "question": text,
        "intent": intent,
        "queries": list(dict.fromkeys(q for q in queries if q))[:3],
    }


class LocalIndex:
    def __init__(self, index, chunks: list[Chunk], encoder):
        self.index = index
        self.chunks = chunks
        self.encoder = encoder

    @classmethod
    def load(cls) -> "LocalIndex":
        if not INDEX_PATH.exists() or not CHUNKS_PATH.exists():
            raise FileNotFoundError("尚未建立索引，请先运行 prepare.py")
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"本地 BGE 模型不存在：{MODEL_PATH}")
        index = load_faiss_index(INDEX_PATH)
        raw = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
        chunks = [Chunk(**item) for item in raw]
        if index.ntotal != len(chunks):
            raise ValueError("向量索引与文本元数据不匹配，请重新运行 prepare.py")
        encoder = SentenceTransformer(str(MODEL_PATH))
        return cls(index, chunks, encoder)

    def search(self, queries: list[str], top_k: int = 8) -> list[Hit]:
        vectors = self.encoder.encode(
            queries, normalize_embeddings=True, convert_to_numpy=True
        ).astype("float32")
        scores, ids = self.index.search(vectors, min(top_k, len(self.chunks)))
        best: dict[str, Hit] = {}
        for row_scores, row_ids in zip(scores, ids):
            for score, idx in zip(row_scores, row_ids):
                if idx < 0:
                    continue
                chunk = self.chunks[int(idx)]
                hit = Hit(chunk, float(score))
                if chunk.chunk_id not in best or score > best[chunk.chunk_id].score:
                    best[chunk.chunk_id] = hit
        return sorted(best.values(), key=lambda item: item.score, reverse=True)[:top_k]


class DeepSeek:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("请先在 .env 中填写 DEEPSEEK_API_KEY")
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
            timeout=15.0,
            max_retries=1,
        )

    async def _chat(self, messages: list[dict]) -> str:
        try:
            response = await self.client.chat.completions.create(
                model="deepseek-flash",
                messages=messages,
                temperature=0.1,
                max_tokens=350,
                extra_body={"thinking": {"type": "disabled"}},
            )
            content = response.choices[0].message.content
            if not content:
                raise RuntimeError("DeepSeek 未返回答案")
            return content.strip()
        except Exception as exc:
            raise RuntimeError(f"DeepSeek 请求失败：{type(exc).__name__}") from None

    async def grounded_answer(self, question: str, hits: list[Hit]) -> str:
        evidence = "\n\n".join(
            f"[第 {hit.chunk.page} 页] {hit.chunk.title}\n{hit.chunk.text}"
            for hit in hits
        )
        return await self._chat(
            [
                {
                    "role": "system",
                    "content": (
                        "你是招股说明书问答助手。只能根据用户提供的文档片段回答，"
                        "不得使用外部知识补全。优先给结论和准确数值，回答简洁。"
                        "如果证据不足，只回答：文档中未找到足够依据。"
                    ),
                },
                {
                    "role": "user",
                    "content": f"问题：{question}\n\n文档片段：\n{evidence}",
                },
            ]
        )

    async def pure_answer(self, question: str) -> str:
        return await self._chat(
            [
                {"role": "system", "content": "请直接、简洁地回答用户问题。"},
                {"role": "user", "content": question},
            ]
        )


class RAGSystem:
    def __init__(self, api_key: str):
        self.index = LocalIndex.load()
        self.llm = DeepSeek(api_key)

    async def ask(self, question: str) -> dict:
        started = time.perf_counter()
        plan = understand_query(question)

        retrieval_start = time.perf_counter()
        hits = self.index.search(plan["queries"], top_k=8)
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
        useful = [hit for hit in hits if hit.score >= 0.35][:6]
        if not useful:
            total_ms = (time.perf_counter() - started) * 1000
            return {
                "answer": "文档中未找到足够依据。",
                "intent": plan["intent"],
                "citations": [],
                "retrieval_ms": round(retrieval_ms, 1),
                "llm_ms": 0.0,
                "total_ms": round(total_ms, 1),
            }

        llm_start = time.perf_counter()
        answer = await self.llm.grounded_answer(plan["question"], useful)
        llm_ms = (time.perf_counter() - llm_start) * 1000
        citations = []
        seen = set()
        for hit in useful:
            key = (hit.chunk.page, hit.chunk.text)
            if key in seen:
                continue
            seen.add(key)
            citations.append(
                {
                    "page": hit.chunk.page,
                    "title": hit.chunk.title,
                    "text": hit.chunk.text,
                    "score": round(hit.score, 4),
                }
            )
        return {
            "answer": answer,
            "intent": plan["intent"],
            "citations": citations[:4],
            "retrieval_ms": round(retrieval_ms, 1),
            "llm_ms": round(llm_ms, 1),
            "total_ms": round((time.perf_counter() - started) * 1000, 1),
        }


def save_chunks(chunks: list[Chunk]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CHUNKS_PATH.write_text(
        json.dumps([asdict(item) for item in chunks], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
