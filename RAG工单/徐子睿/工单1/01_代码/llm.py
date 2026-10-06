# -*- coding: utf-8 -*-
# 工单编号：人工智能 NLP-RAG 项目-RAG 性能瓶颈识别与优化
# 关联工单：人工智能NLP-RAG-基于PDF文档的问答系统优化 | 人工智能NLP-RAG-混合检索任务
# 模块：llm —— 本地模型客户端（Ollama：嵌入 + 生成）
# 说明：嵌入用 bge-m3（1024 维，多语），生成用 qwen2:7b。RAG 与“纯 LLM”用同一模型，
#       保证对比实验公平。带超时与重试；流式生成用于前端打字机效果。
#       工单 13 性能优化：
#        ① 嵌入结果 LRU 缓存（同一查询/多路展开/多轮对话重复字符串零成本命中）；
#        ② keep_alive 长驻，避免模型被卸载后重新加载；
#        ③ warmup()：服务启动时预热嵌入与生成模型，消除“首问 3s 冷启动”。

import json
import time
import urllib.request
from collections import OrderedDict

from config import OLLAMA_BASE, EMBED_MODEL, GEN_MODEL

_KEEP_ALIVE = "30m"
_EMB_CACHE = OrderedDict()
_EMB_CACHE_MAX = 1024
_CACHE_STAT = {"hit": 0, "miss": 0}


def _post(path, payload, timeout=180):
    url = OLLAMA_BASE + path
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def embed(texts, model=None, retries=2, use_cache=True):
    """批量嵌入。texts: list[str] -> list[list[float]]
    工单 13：先查 LRU 缓存，未命中的才发 HTTP（批量发一次）。"""
    if isinstance(texts, str):
        texts = [texts]
    model = model or EMBED_MODEL
    out = [None] * len(texts)
    todo, pos = [], []
    for i, t in enumerate(texts):
        hit = _EMB_CACHE.get((model, t)) if use_cache else None
        if hit is not None:
            _EMB_CACHE.move_to_end((model, t))
            _CACHE_STAT["hit"] += 1
            out[i] = hit
        else:
            _CACHE_STAT["miss"] += 1
            todo.append(t)
            pos.append(i)
    if not todo:
        return out
    last = None
    for _ in range(retries + 1):
        try:
            resp = _post("/api/embed", {"model": model, "input": todo, "keep_alive": _KEEP_ALIVE},
                         timeout=300)
            embs = resp.get("embeddings") or []
            if len(embs) == len(todo):
                for p, t, e in zip(pos, todo, embs):
                    out[p] = e
                    if use_cache:
                        _EMB_CACHE[(model, t)] = e
                        if len(_EMB_CACHE) > _EMB_CACHE_MAX:
                            _EMB_CACHE.popitem(last=False)
                return out
            last = resp
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.0)
    raise RuntimeError("embed failed: %r" % (last,))


def cache_stat():
    return dict(_CACHE_STAT, size=len(_EMB_CACHE))


def warmup(gen=True):
    """服务启动预热：把嵌入（必要时含生成）模型提前载入显存/内存，消除首问冷启动。"""
    t0 = time.perf_counter()
    info = {"embed_ms": None, "gen_ms": None}
    try:
        embed(["预热"], use_cache=False)
        info["embed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    except Exception as e:  # noqa: BLE001
        info["embed_err"] = repr(e)
    if gen:
        t1 = time.perf_counter()
        try:
            chat([{"role": "user", "content": "hi"}], stream=False, timeout=180, num_predict=1)
            info["gen_ms"] = round((time.perf_counter() - t1) * 1000, 1)
        except Exception as e:  # noqa: BLE001
            info["gen_err"] = repr(e)
    return info


def chat(messages, model=None, temperature=0.1, stream=False, timeout=300, **kw):
    """对话生成。流式时返回生成器（逐段 yield 文本）。工单 13：keep_alive 长驻。"""
    model = model or GEN_MODEL
    payload = {"model": model, "messages": messages, "stream": stream,
               "keep_alive": _KEEP_ALIVE,
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
