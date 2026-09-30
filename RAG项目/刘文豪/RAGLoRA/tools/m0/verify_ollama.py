# -*- coding: utf-8 -*-
"""M0：Ollama OpenAI 兼容接口 —— 流式生成 + 中文可用性验证。"""
import time
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

print("[1] 非流式补全")
t0 = time.time()
r = client.chat.completions.create(
    model="qwen2.5:7b",
    messages=[{"role":"user","content":"用一句话说明高血压患者为什么要少吃盐。"}],
    temperature=0.3, max_tokens=120,
)
print(f"    耗时 {time.time()-t0:.2f}s | tokens={r.usage.completion_tokens if r.usage else '?'}")
print(f"    回答: {r.choices[0].message.content.strip()}")
print()

print("[2] 流式补全（SSE 打字机效果）")
t0 = time.time()
tft = None
buf = []
stream = client.chat.completions.create(
    model="qwen2.5:7b",
    messages=[{"role":"user","content":"民法典规定的违约责任有哪几种？只列要点。"}],
    temperature=0.3, max_tokens=150, stream=True,
)
for chunk in stream:
    d = chunk.choices[0].delta.content
    if d:
        if tft is None:
            tft = time.time() - t0
        buf.append(d)
full = "".join(buf)
print(f"    首 token 延迟 {tft:.2f}s | 总耗时 {time.time()-t0:.2f}s | 长度 {len(full)}")
print(f"    回答: {full.strip()[:150]}")
