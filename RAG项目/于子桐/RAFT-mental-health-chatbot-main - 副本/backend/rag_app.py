# -*- coding: utf-8 -*-
"""
本地可跑的 RAG 问答服务（Milvus + bge-m3 + DeepSeek）

原项目 upstream/inference/RAG.ipynb 是 Colab 专用的：T4 GPU 上跑 LLaMA-2-7B、Pinecone 存向量、
ngrok 开公网。本机没有 CUDA，那份 notebook 跑不起来，所以这里另起一个本地版——
检索侧接 ingest 阶段建好的 Milvus 集合（同一个 bge-m3），生成侧换成 DeepSeek API。
原 notebook 保留不动，仍可作为 Colab 参考。

跑之前：
    1. docker start rag1-milvus-etcd rag1-milvus-minio rag1-milvus-standalone
    2. 复制 .env.example 为 .env，填上 LLM_API_KEY
    3. 集合还没建的话先跑 python -u ingest/ingest_to_milvus.py

启动：
    python -u backend/rag_app.py                  # 默认 127.0.0.1:8000
    python -u backend/rag_app.py --check          # 只体检：加载模型+检索一次，不调 LLM
    python -u backend/rag_app.py --host 0.0.0.0 --port 8000

接口（与 frontend/streamlit_app.py 的契约保持不变）：
    POST /generate  {"question": str, "conversation_history": [{"role","content"}, ...]}
                 →  {"response": str, "sources": [...]}
    GET  /health →  {"ok": bool, "detail": str}

配置全部走环境变量，没有任何硬编码的 key。缺 LLM_API_KEY 直接启动失败，不会静默兜底。
"""

import os
import sys
import argparse
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent.parent

# 和 ingest 脚本一样强制 UTF-8 输出：不然重定向到文件时按系统代码页写，
# 中文提示会变成乱码或者直接 UnicodeEncodeError（bat 里有 chcp 65001 配合）。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# .env 放在项目根；python-dotenv 没装也不影响，直接读环境变量即可
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass


# ======================== 配置 ========================

# ---- 向量库。变量名跟 ingest/ingest_to_milvus.py 保持一致 —— 两边必须指向同一个库，
#      否则查的是一份和入库结果无关的集合，而且不报错，只是答得驴唇不对马嘴。
MILVUS_HOST = os.environ.get("MH_MILVUS_HOST") or os.environ.get("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.environ.get("MH_MILVUS_PORT") or os.environ.get("MILVUS_PORT", "19530"))
COLLECTION = os.environ.get("MH_MILVUS_COLLECTION", "mental_health_docs")

# ---- 向量模型。必须和入库时用的完全一致：换个模型向量空间就对不上，
#      结果会很差但不会报错，很难发现。bge-m3 是 1024 维。
LOCAL_BGE_M3 = Path(r"D:\专高三资料\bge-m3")
EMBEDDING_MODEL = os.environ.get(
    "MH_EMBEDDING_MODEL",
    str(LOCAL_BGE_M3) if LOCAL_BGE_M3.is_dir() else "BAAI/bge-m3",
)
EMBEDDING_DIM = 1024

# ---- 生成侧
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-chat")
LLM_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "60"))
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "1024"))

TOP_K = int(os.environ.get("RAG_TOP_K", "4"))          # 沿用 notebook 的 k=4
MAX_HISTORY = int(os.environ.get("RAG_MAX_HISTORY", "10"))


# 检索提示词。取自 notebook cell 29 的"共情陪伴者"模板 —— 强调要用检索到的
# 具体建议作答，并且只回一段、不要自问自答（那是给 LLaMA-2 聊天模型加的约束，
# 换成 API 后留着无害，能防止模型演起双人对话）。
SYSTEM_PROMPT = """You are a compassionate emotional support companion with specific knowledge about
social skills and anxiety management, emotional intelligence and coping with stress.

The following contains relevant expert advice for this situation:
{context}

Using this specific guidance, and adapting it to the user's needs, provide a detailed, EMPATHETIC response
that incorporates these particular strategies and techniques.

Provide ONE single response. Do not generate any follow-up conversation or user replies. End your response after giving advice.
"""

# DeepSeek 挂掉/超时时的兜底。宁可给一句诚实的道歉 + 求助指引，也不要甩 500 让前端崩掉。
FALLBACK = (
    "抱歉，我这边的生成服务暂时不可用，没法给出完整的回应，请稍后再试一次。\n\n"
    "如果你现在状态很不好，请联系当地的心理援助热线，或者找一个你信任的人聊聊。"
)


def _die(title: str, lines: List[str]):
    """启动失败时打一段人话，而不是抛栈。"""
    print("\n" + "=" * 64)
    print(f"[启动失败] {title}")
    for line in lines:
        print(f"  {line}")
    print("=" * 64 + "\n")
    raise SystemExit(1)


# ======================== 运行时状态 ========================


class RAGState:
    """
    向量模型和 Milvus 连接在启动时加载一次，请求里只做检索和生成。

    原 notebook 每次启动都 from_documents 重新嵌入并上传整个语料 —— 那是 Colab
    的一次性环境才受得了。这里只 load()，不往库里写任何东西。
    """

    def __init__(self):
        self.embedder = None
        self.coll = None
        self.llm = None
        self.ready = False

    def load(self, need_llm: bool = True):
        self._check_model()
        if need_llm:
            self._check_llm()
        self._load_embedder()
        self._connect_milvus()
        if need_llm:
            self._init_llm()
        self.ready = True

    # ---- 下面几个 _check_* 只负责"提前报错报得清楚"，真正的加载在后面 ----

    def _check_model(self):
        path = Path(EMBEDDING_MODEL)
        looks_local = path.is_absolute() or "/" in EMBEDDING_MODEL or "\\" in EMBEDDING_MODEL
        if looks_local and not path.is_dir():
            _die(
                "向量模型目录不存在",
                [
                    f"配置的路径: {EMBEDDING_MODEL}",
                    "",
                    "要么把 bge-m3 放回这个目录，要么用环境变量指向别处：",
                    '    set MH_EMBEDDING_MODEL=D:\\你的路径\\bge-m3',
                    "",
                    "注意必须和入库时用的是同一个模型，否则检索结果会莫名其妙地差。",
                ],
            )

    def _check_llm(self):
        if not LLM_API_KEY:
            _die(
                "没有配置 LLM_API_KEY",
                [
                    "本服务不内置任何 key。请这样配置：",
                    "",
                    f"    1. 在 {ROOT} 下复制 .env.example 为 .env",
                    "    2. 填上 LLM_API_KEY=你的key",
                    "",
                    "或者临时设环境变量：",
                    "    set LLM_API_KEY=sk-xxxx        (Windows cmd)",
                    "    export LLM_API_KEY=sk-xxxx     (bash)",
                    "",
                    f"当前生成服务: {LLM_BASE_URL} / {LLM_MODEL}",
                    "换成别家 OpenAI 兼容服务的话，连 LLM_BASE_URL 和 LLM_MODEL 一起改。",
                ],
            )

    def _load_embedder(self):
        from sentence_transformers import SentenceTransformer

        print(f"加载向量模型 {EMBEDDING_MODEL} ...")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        # sentence-transformers 6.0 把方法改了名，老版本还是原来的叫法
        get_dim = (getattr(self.embedder, "get_embedding_dimension", None)
                   or self.embedder.get_sentence_embedding_dimension)
        dim = get_dim()
        if dim != EMBEDDING_DIM:
            _die(
                "向量维度对不上",
                [
                    f"集合是按 {EMBEDDING_DIM} 维建的，当前模型是 {dim} 维: {EMBEDDING_MODEL}",
                    "多半是 MH_EMBEDDING_MODEL 指到了别的模型。",
                ],
            )
        print(f"   模型就绪（{dim} 维）")

    def _connect_milvus(self):
        from pymilvus import Collection, connections, utility

        print(f"连接 Milvus {MILVUS_HOST}:{MILVUS_PORT} ...")
        try:
            connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
        except Exception as e:
            _die(
                "连不上 Milvus",
                [
                    f"{MILVUS_HOST}:{MILVUS_PORT} -> {e}",
                    "",
                    "容器没起的话：",
                    "    docker start rag1-milvus-etcd rag1-milvus-minio rag1-milvus-standalone",
                    "",
                    "起完等十几秒再试；还不行就 docker ps 看看三个容器是不是都 Up。",
                ],
            )

        if not utility.has_collection(COLLECTION):
            _die(
                f"集合 {COLLECTION} 不存在",
                [
                    "向量库是空的，先建索引：",
                    "",
                    "    python -u ingest/ingest_to_milvus.py",
                    "",
                    "（约 13 分钟，7827 块；期间别并发跑第二个）",
                ],
            )

        self.coll = Collection(COLLECTION)
        self.coll.load()
        print(f"   集合 {COLLECTION} 已加载")

    def _init_llm(self):
        from openai import OpenAI

        self.llm = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL, timeout=LLM_TIMEOUT)
        print(f"   生成模型 {LLM_MODEL} @ {LLM_BASE_URL}")


state = RAGState()


# ======================== 检索 ========================


def retrieve(question: str) -> List[dict]:
    """中文问题 → 跨语言召回英文段落。返回按相似度排序、已去重的命中。"""
    vec = state.embedder.encode([question], normalize_embeddings=True)[0].tolist()
    res = state.coll.search(
        [vec],
        "vector",
        {"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=TOP_K,
        output_fields=["content", "book", "chapter_title", "section", "page_start"],
    )

    hits, seen = [], set()
    for h in res[0]:
        e = h.entity
        text = (e.get("content") or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        hits.append(
            {
                "content": text,
                "book": e.get("book") or "",
                "chapter_title": e.get("chapter_title") or "",
                "section": e.get("section") or "",
                "page_start": e.get("page_start") or 0,
                "score": round(float(h.distance), 4),
            }
        )
    return hits


def build_context(hits: List[dict]) -> str:
    """把命中拼成给模型看的上下文，带出处，方便它引用具体建议。"""
    blocks = []
    for i, h in enumerate(hits, 1):
        where = f"《{h['book']}》 {h['chapter_title']} (p{h['page_start']})"
        blocks.append(f"[{i}] {where}\n{h['content']}")
    return "\n\n".join(blocks) if blocks else "(没有检索到相关资料)"


# ======================== HTTP 服务 ========================

from fastapi import FastAPI  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402


class Message(BaseModel):
    role: str
    content: str


class GenerateRequest(BaseModel):
    question: str
    conversation_history: List[Message] = Field(default_factory=list)


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.load()
    yield


app = FastAPI(title="Mental Health RAG (local)", lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "ok": state.ready,
        "detail": f"{COLLECTION} @ {MILVUS_HOST}:{MILVUS_PORT} / embeddings={EMBEDDING_MODEL} / llm={LLM_MODEL}",
    }


# 注意是 def 不是 async def：embedding 和 Milvus 检索都是阻塞调用，
# 写成 async def 会把事件循环卡死，并发请求全部排队。FastAPI 对普通 def
# 会自动丢线程池，这才是对的。
@app.post("/generate")
def generate(req: GenerateRequest):
    question = (req.question or "").strip()
    if not question:
        return {"response": "请先说说你想聊什么。", "sources": []}

    # 前端把本轮问题也塞进 history 一起发过来了，这里去掉避免重复问两遍
    history = list(req.conversation_history)
    if history and history[-1].role == "user" and history[-1].content.strip() == question:
        history = history[:-1]
    history = history[-MAX_HISTORY:]  # 只留最近几轮，别让 prompt 无限膨胀

    try:
        hits = retrieve(question)
    except Exception as e:
        print(f"[检索失败] {type(e).__name__}: {e}")
        return {"response": FALLBACK, "sources": []}

    if not hits:
        return {"response": FALLBACK, "sources": []}

    messages = [{"role": "system", "content": SYSTEM_PROMPT.format(context=build_context(hits))}]
    messages += [{"role": m.role, "content": m.content} for m in history if m.role in ("user", "assistant")]
    messages.append({"role": "user", "content": question})

    try:
        resp = state.llm.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            max_tokens=LLM_MAX_TOKENS,   # 原 notebook 把这行注释掉了，于是无界生成直到 OOM
            temperature=0.7,
        )
        answer = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"[生成失败] {type(e).__name__}: {e}")
        return {"response": FALLBACK, "sources": []}

    if not answer:
        answer = FALLBACK

    return {
        "response": answer,
        "sources": [
            {
                "book": h["book"],
                "chapter_title": h["chapter_title"],
                "section": h["section"],
                "page_start": h["page_start"],
                "score": h["score"],
            }
            for h in hits
        ],
    }


# ======================== 启动 ========================


def main():
    ap = argparse.ArgumentParser(description="本地 RAG 问答服务（Milvus + bge-m3 + DeepSeek）")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--check", action="store_true",
                    help="只体检：加载模型、连库、检索一次并打印命中，不连 LLM 不起服务")
    ap.add_argument("--q", default="我社交焦虑怎么办", help="配合 --check 用的查询")
    args = ap.parse_args()

    if args.check:
        # 不检查 LLM 配置，这样没填 key 也能单独验证检索侧
        state.load(need_llm=False)
        print("\n" + "=" * 64)
        print(f"检索测试: {args.q}")
        print("=" * 64)
        hits = retrieve(args.q)
        if not hits:
            print("  没召回到任何内容 —— 集合是空的？")
            return 1
        for i, h in enumerate(hits, 1):
            print(f"\n[{i}] 相似度 {h['score']:.4f}  《{h['book']}》")
            print(f"    {h['chapter_title']}  (p{h['page_start']})  {h['section']}")
            print("    " + h["content"][:260].replace("\n", " ") + " ...")
        print("\n检索侧正常。加 LLM_API_KEY 后可以直接跑 python -u backend/rag_app.py 起服务。")
        return 0

    if not LLM_API_KEY:
        _die(
            "没有配置 LLM_API_KEY",
            [
                f"在 {ROOT} 下复制 .env.example 为 .env 并填上 key；",
                "只想验证检索侧的话可以先跑: python -u backend/rag_app.py --check",
            ],
        )

    import uvicorn

    print("=" * 64)
    print("本地 RAG 服务")
    print(f"  接口      : http://{args.host}:{args.port}/generate")
    print(f"  Milvus    : {MILVUS_HOST}:{MILVUS_PORT} / {COLLECTION}")
    print(f"  向量模型  : {EMBEDDING_MODEL}")
    print(f"  生成模型  : {LLM_MODEL} @ {LLM_BASE_URL}")
    print(f"  检索条数  : top_k={TOP_K}")
    print("=" * 64)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
