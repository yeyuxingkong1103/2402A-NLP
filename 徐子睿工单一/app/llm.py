# -*- coding: utf-8 -*-
# 工单编号：人工智能 NLP-RAG-基于 PDF 文档的问答系统优化
# 模块：llm —— 本地模型客户端（Ollama：嵌入 + 生成）
# 说明：嵌入用 bge-m3（1024 维，多语），生成用 qwen2:7b。RAG 与“纯 LLM”用同一模型，
#       保证对比实验公平。带超时与重试；流式生成用于前端打字机效果。

import json
import time
import urllib.request

from config import OLLAMA_BASE, EMBED_MODEL, GEN_MODEL


def _post(path, payload, timeout=180):
    url = OLLAMA_BASE + path
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def embed(texts, model=None, retries=2):
    """批量嵌入。texts: list[str] -> list[list[float]]"""
    if isinstance(texts, str):
        texts = [texts]
    model = model or EMBED_MODEL
    last = None
    for _ in range(retries + 1):
        try:
            out = _post("/api/embed", {"model": model, "input": texts}, timeout=300)
            embs = out.get("embeddings") or []
            if len(embs) == len(texts):
                return embs
            last = out
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.0)
    raise RuntimeError("embed failed: %r" % (last,))


def chat(messages, model=None, temperature=0.1, stream=False, timeout=300, **kw):
    """对话生成。流式时返回生成器（逐段 yield 文本）。"""
    model = model or GEN_MODEL
    payload = {"model": model, "messages": messages, "stream": stream,
               "options": {"temperature": temperature, **kw}}
    if not stream:
        out = _post("/api/chat", payload, timeout=timeout)
        return (out.get("message") or {}).get("content", "")
    # 流式
    url = OLLAMA_BASE + "/api/chat"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

    def gen():
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw.decode("utf-8"))
                except Exception:  # noqa: BLE001
                    continue
                piece = (obj.get("message") or {}).get("content", "")
                if piece:
                    yield piece
                if obj.get("done"):
                    break
    return gen()


def health():
    try:
        req = urllib.request.Request(OLLAMA_BASE + "/api/tags")
        with urllib.request.urlopen(req, timeout=5) as r:
            tags = json.loads(r.read().decode("utf-8"))
        names = [m["name"] for m in tags.get("models", [])]
        return {"ok": True, "models": names,
                "has_embed": any(EMBED_MODEL in n for n in names),
                "has_gen": any(GEN_MODEL in n for n in names)}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}
