# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-功能测试及评估
# （上游来源：工单06 混合检索；本工单复用于 ccf_competition 语料的功能测试与评估）
"""
工单编号：人工智能NLP-RAG-混合检索任务
（工单 PDF 原文：人工智能NLP-RAG-混合检索任务）

模块：图像块增强（复用上游工单04 的多模态解析成果）

背景
----
16 道验收题里的 id 5（组织结构图）与 id 6（2008 年中国 IC 市场应用结构与增长图）
答案**既不在正文也不在表格里**，只存在于 PDF 插图中。上游工单04 已用多模态大模型
对插图做过结构化描述（162 条，落盘 data/vlm_cache/），并渲染了原图（data/figures/）。

本模块的职责：把「图 → 描述缓存」的映射建立起来，并把上游分块结果里
占位的图像块（text="[图片]"）**就地替换为 VLM 结构化描述**，使图像内容
进入向量检索与全文检索（两路都能命中"汽车 14.0%""IC卡 -2.0%"这类图内事实）。

映射规则（由上游产物实测确定）
----------------------------
  · 图文件名：`{文档名}_p{页码1基}_{md5前12位}.png`；
  · 页码换算：`page_idx = 页码1基 - 1`（0 基，与 chunk 对齐；
    实测：IC 市场图 content_list page_idx=71 → 图 p72；page_idx=309 → 图 p310）；
  · 描述缓存键：`md5(图片字节)[:16] + "_" + md5(图注)[:8] + ".json"`（上游 vlparser 口径）。

约束：只读上游产物（本工程内已复制），零网络请求；映射不全时如实统计并打印，
不静默丢块。
"""
from __future__ import annotations

from src import bootstrap  # noqa: F401  —— 必须最先导入

import glob
import hashlib
import json
import os
import re
import threading

from src import config

_FIG_RE = re.compile(r"^(?P<stem>.+?)_p(?P<page>\d+)_(?P<hash>[0-9a-f]+)\.png$", re.I)
_TEXT_MAX = 1500          # 描述入库长度上限（bge-m3 单块预算内）

_cache: dict[tuple[str, int], list[dict]] | None = None
_lock = threading.Lock()


def _figure_dir() -> str:
    return os.path.join(config.DATA_DIR, "figures")


def _vlm_dir() -> str:
    return os.path.join(config.DATA_DIR, "vlm_cache")


def build_mapping(force: bool = False) -> dict[tuple[str, int], list[dict]]:
    """构建 (source, page_idx) → [{image_id, image_path, description(一句话), full}] 映射。"""
    global _cache
    with _lock:
        if _cache is not None and not force:
            return _cache

        by_prefix: dict[str, str] = {}
        for p in glob.glob(os.path.join(_vlm_dir(), "*.json")):
            by_prefix[os.path.basename(p)[:16]] = p

        mapping: dict[tuple[str, int], list[dict]] = {}
        n_fig = n_matched = 0
        for f in sorted(glob.glob(os.path.join(_figure_dir(), "*.png"))):
            name = os.path.basename(f)
            m = _FIG_RE.match(name)
            if not m:
                continue
            n_fig += 1
            digest = hashlib.md5(_read_bytes(f)).hexdigest()[:16]
            cache_file = by_prefix.get(digest)
            if not cache_file:
                continue
            try:
                with open(cache_file, encoding="utf-8") as fh:
                    payload = json.load(fh)
            except Exception:
                continue
            desc = (payload.get("description") or "").strip()
            if not desc:
                continue
            n_matched += 1
            key = (f"{m.group('stem')}.pdf", int(m.group('page')) - 1)
            mapping.setdefault(key, []).append({
                "image_id": name,
                "image_path": os.path.relpath(f, config.ROOT).replace("\\", "/"),
                "full": desc,
                "one_line": _first_sentence(desc),
            })
        print(f"[image_blocks] 图 {n_fig} 张，命中 VLM 描述 {n_matched} 条，"
              f"覆盖 {len(mapping)} 个(文档,页)", flush=True)
        _cache = mapping
        return mapping


def _read_bytes(path: str) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()


def _first_sentence(desc: str) -> str:
    """取描述的第一句"一句话概括"（上游 VLM 输出的结构化首行）。"""
    for line in desc.splitlines():
        line = line.strip().lstrip("1.").strip()
        if len(line) >= 8:
            return line[:160]
    return desc[:160]


def enrich_image_chunks(chunks: list[dict],
                        mapping: dict[tuple[str, int], list[dict]] | None = None
                        ) -> tuple[list[dict], dict]:
    """把图像占位块替换为 VLM 描述块；非图像块原样返回。

    返回 (新块列表, 统计)。统计里如实给出 matched / unmatched / n_images，
    未命中的图像块保留原占位文本（可检索性差但不丢失，便于人工排查）。
    """
    mapping = mapping if mapping is not None else build_mapping()
    out: list[dict] = []
    stat = {"n_images": 0, "matched": 0, "unmatched": 0, "n_new_blocks": 0}

    for c in chunks:
        if c.get("block_type") != "image":
            out.append(c)
            continue
        stat["n_images"] += 1
        key = (c.get("source", ""), int(c.get("page_idx", -1)))
        infos = mapping.get(key) or []
        if not infos:
            stat["unmatched"] += 1
            out.append(c)
            continue
        stat["matched"] += 1
        # 一页可能有多张图：各图单独成块（引用时按 image_id 区分）
        for j, info in enumerate(infos):
            desc = info["full"][:_TEXT_MAX]
            block = dict(c)
            block["chunk_id"] = f"{c.get('chunk_id', 'c')}-img{j}" if len(infos) > 1 \
                else c.get("chunk_id")
            block["text"] = f"[图像] {info['one_line']}\n{desc}"
            block["block_type"] = "image"
            block["image_id"] = info["image_id"]
            block["image_path"] = info["image_path"]
            block["image_description"] = desc
            block["parent_text"] = block["text"]
            out.append(block)
            stat["n_new_blocks"] += 1

    return out, stat


def info() -> dict:
    """映射健康度（供界面/文档展示）。"""
    m = build_mapping()
    return {
        "figures": len(glob.glob(os.path.join(_figure_dir(), "*.png"))),
        "descriptions": len(glob.glob(os.path.join(_vlm_dir(), "*.json"))),
        "covered_pages": len(m),
        "figure_dir": _figure_dir(),
        "vlm_cache_dir": _vlm_dir(),
    }
