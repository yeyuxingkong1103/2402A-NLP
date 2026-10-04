# -*- coding: utf-8 -*-
"""
Ollama 客户端封装
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
说明：统一封装 LLM 对话生成 / 文本向量化，带重试与超时控制，
     并强制绕过系统代理（本机代理会劫持 127.0.0.1 请求导致连接失败）。
"""
import os
import time
import json
import requests

# 关键：清除代理环境变量，否则 127.0.0.1 请求会被 http_proxy 劫持
for _k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
    os.environ.pop(_k, None)

from config import OLLAMA_BASE, LLM_MODEL, EMBED_MODEL, EMBED_BATCH, VL_MODEL


class OllamaClient:
    """Ollama HTTP API 客户端（含重试机制）"""

    def __init__(self, base_url=None, llm_model=None, embed_model=None, vl_model=None):
        self.base = base_url or OLLAMA_BASE
        self.llm_model = llm_model or LLM_MODEL
        self.embed_model = embed_model or EMBED_MODEL
        self.vl_model = vl_model or VL_MODEL
        self.session = requests.Session()
        self.session.trust_env = False  # 不读取系统代理

    # ── 健康检查 ───────────────────────────────────────────
    def is_alive(self):
        try:
            r = self.session.get(f"{self.base}/api/tags", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    # ── LLM 生成 ───────────────────────────────────────────
    def chat(self, messages, temperature=0.1, num_predict=1024, retries=3, timeout=120):
        """调用 /api/chat 生成回答。messages: [{"role","content"}, ...]"""
        payload = {
            "model": self.llm_model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": num_predict},
        }
        last_err = None
        for i in range(retries):
            try:
                r = self.session.post(f"{self.base}/api/chat", json=payload, timeout=timeout)
                r.raise_for_status()
                return r.json()["message"]["content"].strip()
            except Exception as e:
                last_err = e
                time.sleep(2 * (i + 1))  # 退避重试
        raise RuntimeError(f"LLM 调用失败(重试{retries}次): {last_err}")

    def generate(self, prompt, **kw):
        """便捷方法：单条 prompt 生成"""
        return self.chat([{"role": "user", "content": prompt}], **kw)

    # ── 多模态视觉理解（工单04） ───────────────────────────
    def vl_describe(self, image_path, prompt, retries=2, timeout=600):
        """用多模态模型理解图片内容，返回描述文本"""
        import base64
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        payload = {
            "model": self.vl_model,
            "messages": [{"role": "user", "content": prompt, "images": [b64]}],
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 500},
        }
        last_err = None
        for i in range(retries):
            try:
                r = self.session.post(f"{self.base}/api/chat", json=payload, timeout=timeout)
                r.raise_for_status()
                return r.json()["message"]["content"].strip()
            except Exception as e:
                last_err = e
                time.sleep(3 * (i + 1))
        raise RuntimeError(f"视觉模型调用失败: {last_err}")

    # ── 向量化 ─────────────────────────────────────────────
    def embed(self, text, retries=3):
        """单条文本向量化，返回 list[float]"""
        last_err = None
        for i in range(retries):
            try:
                r = self.session.post(
                    f"{self.base}/api/embeddings",
                    json={"model": self.embed_model, "prompt": text[:4000]},
                    timeout=60,
                )
                r.raise_for_status()
                return r.json()["embedding"]
            except Exception as e:
                last_err = e
                time.sleep(1.5 * (i + 1))
        raise RuntimeError(f"向量化失败: {last_err}")

    def embed_batch(self, texts, batch_size=None, progress=True):
        """批量向量化（顺序执行，带进度显示）"""
        batch_size = batch_size or EMBED_BATCH
        vecs = []
        for i, t in enumerate(texts):
            vecs.append(self.embed(t))
            if progress and (i + 1) % batch_size == 0:
                print(f"  向量化进度: {i+1}/{len(texts)}")
        return vecs


# 模块级单例，供各工单直接导入使用
client = OllamaClient()
