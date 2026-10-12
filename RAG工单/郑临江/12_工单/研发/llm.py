# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG项目-LightRAG优化
LLM 调用封装：配置 API Key 时走 OpenAI/DashScope 兼容接口，
否则返回 None（离线模式由上层回退为“仅检索”答案）。
"""
import json
import urllib.request

import config


def call_llm(prompt, system=""):
    if not config.LLM_API_KEY:
        return None
    try:
        req = urllib.request.Request(
            config.LLM_BASE_URL.rstrip("/") + "/chat/completions",
            data=json.dumps({
                "model": config.LLM_MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            }).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {config.LLM_API_KEY}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
    except Exception as e:
        print("[llm] 调用失败，回退离线模式:", e)
        return None
