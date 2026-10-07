"""本地向量检索 + DeepSeek 答案生成。"""

from __future__ import annotations

import html
import json
import pickle
import re
import asyncio
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
LEXICAL_PATH = DATA_DIR / "lexical.pkl"
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
    document: str = "prospectus_1"
    company: str = "武汉兴图新科电子股份有限公司"


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
    if re.search(r"比重|占比|比例|%|％", text):
        intent = "ratio"
    elif re.search(r"多少|金额|收入|资本|资金", text):
        intent = "amount"
    elif re.search(r"谁|法定代表人|董事长", text):
        intent = "person"
    elif re.search(r"哪个标准|技术标准|哪个工程", text):
        intent = "name"
    elif re.search(r"上游|下游|行业|领域", text):
        intent = "industry"

    document = ""
    company = ""
    if re.search(r"力源信息|武汉力源", text):
        document = "prospectus_2"
        company = "武汉力源信息技术股份有限公司"
    elif re.search(r"兴图新科|武汉兴图", text):
        document = "prospectus_1"
        company = "武汉兴图新科电子股份有限公司"

    queries = [text]
    short = re.sub(r"根据.*?(招股意向书|招股说明书)[,，]?​?", "", text)
    short = short.replace("武汉兴图新科电子股份有限公司", "兴图新科")
    short = short.replace("武汉力源信息技术股份有限公司", "力源信息")
    if short != text:
        queries.append(short)
    if "分别" in text or "报告期" in text:
        keywords = re.sub(r"[？?]​?$", "", short)
        keywords = re.sub(r"是多少|为多少|有哪些", "", keywords)
        queries.append(keywords)

    # 针对招股说明书常见问法增加同义检索词。
    aliases = {
        "军用领域": "军品 军用 收入 主营业务",
        "比重": "军用领域 收入 主营业务收入 占比 比重",
        "技术标准": "国家标准 行业标准 标准制定",
        "上游": "产业链上游 原材料 供应商 企业",
        "下游": "产业链下游 客户 应用行业",
        "重要供应商": "核心供应商 重要供应商 领域",
        "科技进步一等奖": "国家科学技术进步奖 一等奖 工程",
        "注册资本": "注册资本 股本 万元",
        "法定代表人": "法定代表人 董事长 发行人基本情况",
        "补充流动资金": "募集资金 募投项目 补充流动资金 金额",
        "发行股数": "本次发行 发行股数 发行后总股本 比例",
        "募集资金拟投资": "募集资金用途 投资项目 计划总投资",
        "存在控制关系": "关联方 持股比例 与本公司关系 控股股东",
        "不存在控制关系": "不存在控制关系的关联方 企业名称 与本公司关系",
        "组织结构图": "发行人组织结构图 销售部 大客户销售部 销售处",
        "市场应用结构与增长": "2008年中国IC市场应用结构与增长 增长率 行业",
    }
    for phrase, expansion in aliases.items():
        if phrase in text:
            queries.append(expansion)

    return {
        "question": text,
        "intent": intent,
        "document": document,
        "company": company,
        "queries": list(dict.fromkeys(q for q in queries if q))[:4],
    }


class LocalIndex:
    def __init__(self, index, chunks: list[Chunk], encoder, lexical=None):
        self.index = index
        self.chunks = chunks
        self.encoder = encoder
        self.lexical = lexical

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
        lexical = None
        if LEXICAL_PATH.exists():
            with LEXICAL_PATH.open("rb") as stream:
                lexical = pickle.load(stream)
        return cls(index, chunks, encoder, lexical)

    def search(self, queries: list[str], top_k: int = 8, document: str = "") -> list[Hit]:
        vectors = self.encoder.encode(
            queries, normalize_embeddings=True, convert_to_numpy=True
        ).astype("float32")
        candidate_k = len(self.chunks) if document else min(max(top_k * 4, 24), len(self.chunks))
        scores, ids = self.index.search(vectors, candidate_k)
        vector_scores = np.zeros(len(self.chunks), dtype="float32")
        for row_scores, row_ids in zip(scores, ids):
            for score, idx in zip(row_scores, row_ids):
                if idx >= 0:
                    vector_scores[int(idx)] = max(vector_scores[int(idx)], float(score))

        lexical_scores = np.zeros(len(self.chunks), dtype="float32")
        if self.lexical:
            vectorizer, matrix = self.lexical
            query_matrix = vectorizer.transform(queries)
            values = matrix @ query_matrix.T
            lexical_scores = np.asarray(values.max(axis=1).toarray()).ravel().astype("float32")

        combined = 0.72 * np.clip(vector_scores, 0.0, 1.0) + 0.28 * lexical_scores
        if document:
            allowed = np.asarray(
                [chunk.document == document for chunk in self.chunks], dtype=bool
            )
            combined[~allowed] = -1.0
        # 完整短语命中可以稳定抬高表格和基本信息的排名。
        phrases = [
            phrase
            for query in queries
            for phrase in re.findall(r"[\u4e00-\u9fffA-Za-z0-9%\.]{3,}", query)
            if phrase not in {"武汉兴图新科电子股份有限公司", "根据", "报告期内"}
        ]
        for idx, chunk in enumerate(self.chunks):
            if document and chunk.document != document:
                continue
            haystack = f"{chunk.title} {chunk.text}"
            matches = sum(1 for phrase in phrases if phrase in haystack)
            combined[idx] += min(matches * 0.025, 0.10)
            if chunk.content_type == "figure":
                query_text = " ".join(queries)
                if (
                    "组织结构图" in query_text and "组织结构图" in chunk.title
                ) or (
                    "市场应用结构与增长" in query_text
                    and "市场应用结构与增长" in chunk.title
                ):
                    combined[idx] += 0.85

        ranked = np.argsort(-combined)
        selected: list[int] = [int(idx) for idx in ranked[:top_k] if combined[idx] > 0]
        # 补入高分文本块的相邻内容，防止多年数据或表格说明被切断。
        expanded = list(selected)
        for idx in selected[:4]:
            for neighbor in (idx - 1, idx + 1):
                if 0 <= neighbor < len(self.chunks) and neighbor not in expanded:
                    if (
                        self.chunks[neighbor].document == self.chunks[idx].document
                        and abs(self.chunks[neighbor].page - self.chunks[idx].page) <= 1
                    ):
                        expanded.append(neighbor)
        return [
            Hit(self.chunks[idx], float(combined[idx] if idx in selected else combined[selected[0]] * 0.8))
            for idx in expanded[: top_k + 4]
        ]


class DeepSeek:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("请先在 .env 中填写 DEEPSEEK_API_KEY")
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
            timeout=2.5,
            max_retries=0,
        )
        self._semaphore = asyncio.Semaphore(6)

    async def _chat(self, messages: list[dict]) -> str:
        try:
            async with self._semaphore:
                response = await self.client.chat.completions.create(
                    model="deepseek-flash",
                    messages=messages,
                    temperature=0.0,
                    max_tokens=220,
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
            f"[{hit.chunk.company}·第 {hit.chunk.page} 页] {hit.chunk.title}\n{hit.chunk.text[:700]}"
            for hit in hits
        )
        return await self._chat(
            [
                {
                    "role": "system",
                    "content": (
                        "你是招股说明书问答助手。只能根据用户提供的文档片段回答，"
                        "不得使用外部知识补全。优先给结论和准确数值，回答简洁。"
                        "如果问题要求‘分别’或‘哪些’，必须完整列出所有期间或项目，不得只回答一项。"
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
        self._cache: dict[str, dict] = {}

    @staticmethod
    def _fallback(hits: list[Hit]) -> str:
        if not hits:
            return "文档中未找到足够依据。"
        extracts = []
        for hit in hits[:2]:
            text = re.sub(r"\s+", " ", hit.chunk.text).strip()
            extracts.append(f"{hit.chunk.company}《{hit.chunk.document}》第 {hit.chunk.page} 页：{text[:260]}")
        return "根据检索到的原文：\n" + "\n".join(extracts)

    async def ask(self, question: str) -> dict:
        started = time.perf_counter()
        plan = understand_query(question)
        cache_key = plan["question"]
        if cache_key in self._cache:
            cached = dict(self._cache[cache_key])
            cached["cached"] = True
            cached["total_ms"] = round((time.perf_counter() - started) * 1000, 1)
            return cached

        retrieval_start = time.perf_counter()
        hits = self.index.search(
            plan["queries"], top_k=8, document=plan["document"]
        )
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
        useful = [hit for hit in hits if hit.score >= 0.22][:8]
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
        remaining = max(0.2, 2.75 - (llm_start - started))
        try:
            answer = await asyncio.wait_for(
                self.llm.grounded_answer(plan["question"], useful), timeout=remaining
            )
        except (asyncio.TimeoutError, RuntimeError):
            answer = self._fallback(useful)
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
                    "document": hit.chunk.document,
                    "company": hit.chunk.company,
                    "content_type": hit.chunk.content_type,
                    "text": hit.chunk.text,
                    "score": round(hit.score, 4),
                }
            )
        result = {
            "answer": answer,
            "intent": plan["intent"],
            "document": plan["document"],
            "company": plan["company"],
            "citations": citations[:4],
            "retrieval_ms": round(retrieval_ms, 1),
            "llm_ms": round(llm_ms, 1),
            "total_ms": round((time.perf_counter() - started) * 1000, 1),
            "cached": False,
        }
        if len(self._cache) >= 100:
            self._cache.pop(next(iter(self._cache)))
        self._cache[cache_key] = dict(result)
        return result


def save_chunks(chunks: list[Chunk]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CHUNKS_PATH.write_text(
        json.dumps([asdict(item) for item in chunks], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
