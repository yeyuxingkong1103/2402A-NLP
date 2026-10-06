# -*- coding: utf-8 -*-
"""retrieval/rerank.py —— LLM 精排。

在链路中的位置：
    retrieve_with_trace 的第四步：让本地模型逐条读候选并打相关性分，重排。

为什么要有这一步：
    RRF 只用了"排名"信息，没有真正读内容判断相关性。让模型读一遍并打 0-5 分，
    排序稳定性明显更好（见《项目总览.md》V3 迭代）。

三层降级（fail-open，任何一层失败都不让检索整体崩掉）：
    1. 没有候选 -> 直接返回空
    2. 调模型失败（Ollama 没起/超时）-> 回退 RRF 顺序
    3. 模型输出解析不出分数 -> 同样回退 RRF 顺序
"""
from __future__ import annotations

import re
from typing import Any

import requests

from .config import CHAT_URL, RERANK_MODEL

def _chat(prompt: str) -> str:
    """调 Ollama 的 chat 接口，让模型给候选片段打分。

    参数：
        prompt: 已拼好的"问题 + 候选片段"提示词
    返回：
        模型输出的纯文本（形如 "1:5\\n2:3"）。
    异常：
        网络/HTTP 错误直接抛出，由 rerank 捕获并回退到 RRF 顺序。

    参数选择说明：
        temperature=0 让打分可复现（同一批候选每次得到同样的分数，便于对比调试）；
        num_predict=160 是因为打分输出很短，限制长度既加速又避免模型啰嗦自解释。
    """
    response = requests.post(
        CHAT_URL,
        json={
            "model": RERANK_MODEL,
            "messages": [
                {"role": "system", "content": "给每个候选片段打 0 到 5 分，只输出编号:分数，例如 1:5。"},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": 0, "num_predict": 160},
        },
        timeout=180,  # 本地 7B 模型对 15 个候选打分需要时间，给足 3 分钟
    )
    response.raise_for_status()
    return response.json()["message"]["content"].strip()

def rerank(candidates: list[dict[str, Any]], query: str, limit: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """用大模型对融合后的候选做相关性精排。

    参数：
        candidates: fuse_hits 的输出（默认 15 条）
        query: 改写后的检索串
        limit: 最终保留几条（默认 5）
    返回：
        (排序后的候选列表, 元信息)
        元信息含 applied(是否真的用了精排结果) / parsed(解析出几个分数) / raw(模型原始输出) / note(说明)

    为什么要有这一步：
        RRF 只用了"排名"信息，没有真正读内容判断相关性。让模型逐条读一遍并打 0-5 分，
        排序稳定性明显更好（见《项目总览.md》V3 迭代）。

    三层降级，任何一层失败都不让检索整体崩掉：
        1. 没有候选 → 直接返回空
        2. 调模型失败（Ollama 没起/超时）→ 回退 RRF 顺序，rerank_source="fallback"
        3. 模型答非所问、输出解析不出分数 → 同样回退 RRF 顺序
        "让一次精排失败拖垮整个检索流程"是绝不能接受的，所以这里全部 fail-open。
    """
    if not candidates:
        return [], {"applied": False, "parsed": 0, "raw": "", "note": "无候选片段"}
    lines = []
    for number, item in enumerate(candidates, 1):
        # 每条截断到 260 字：15 条候选全给全文会超出模型上下文，且打分只需判断相关性、不需读全
        text = item["text"].replace("\n", " ")[:260]
        lines.append(f"[{number}] (第{item['page']}页 {item['section']}) {text}")
    try:
        raw = _chat(f"【问题】{query}\n\n【候选片段】\n" + "\n".join(lines))
    except Exception as exc:
        fallback = [item | {"rerank_score": item["rrf_score"], "rerank_source": "fallback"} for item in candidates[:limit]]
        return fallback, {"applied": False, "parsed": 0, "raw": "", "note": f"精排失败，回退 RRF：{exc}"}

    # 宽松解析 "1:5" / "1：5" 这类输出；越界编号（模型幻觉出不存在的编号）直接丢弃
    scores = {
        int(match.group(1)): float(match.group(2))
        for match in re.finditer(r"(\d+)\s*[:：]\s*(\d(?:\.\d+)?)", raw)
        if 1 <= int(match.group(1)) <= len(candidates)
    }
    if not scores:
        fallback = [item | {"rerank_score": item["rrf_score"], "rerank_source": "fallback"} for item in candidates[:limit]]
        return fallback, {"applied": False, "parsed": 0, "raw": raw, "note": "精排输出无法解析，回退 RRF"}

    # 模型没打分的候选记 0 分（不会排前面，但仍留在列表里可被人工检查）
    ranked = [item | {"rerank_score": scores.get(number, 0), "rerank_source": "ollama"} for number, item in enumerate(candidates, 1)]
    # 三级排序键：精排分 → RRF 分 → 页码。后两级保证同分时顺序稳定、结果可复现
    ranked.sort(key=lambda item: (-item["rerank_score"], -item["rrf_score"], item["page"]))
    return ranked[:limit], {"applied": True, "parsed": len(scores), "raw": raw, "note": f"已解析 {len(scores)}/{len(candidates)} 个候选"}
